import argparse
import serial
import time
import re
from array import array
from pathlib import Path

DEFAULT_COM_PORT = "COM10"
DEFAULT_BAUDRATE = 921600
GOLDEN_HEX = Path("golden_patterns/full_video.hex")
GOLDEN_TRACE = Path("golden_patterns/golden_trace.txt")

MB_SIZE = 16
SEARCH_RANGE = 32
WINDOW_SIZE = MB_SIZE + 2 * SEARCH_RANGE  # 80

CMD_LOAD_WINDOW = ord("W")
CMD_LOAD_CUR = ord("C")

TRACE_RE = re.compile(
    r"Frame=(\d+)\s+MB_Row=(\d+)\s+MB_Col=(\d+)\s+\|\s+MV_X=([-\d]+)\s+MV_Y=([-\d]+)\s+SAD=(\d+)"
)


class GoldenVideo:
    def __init__(self, hex_path: Path):
        if not hex_path.exists():
            raise FileNotFoundError(f"找不到 HEX 檔案: {hex_path}")
        self.hex_path = hex_path
        self.width, self.height = self._parse_header()
        self.frame_size = self.width * self.height
        self.pixels = self._load_pixels()
        if len(self.pixels) % self.frame_size != 0:
            raise ValueError("full_video.hex 大小與影像規格不相符")
        self.total_frames = len(self.pixels) // self.frame_size

    def _parse_header(self):
        width = height = None
        with self.hex_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.startswith("//"):
                    break
                line = line.strip()
                if "Width" in line and "Height" in line:
                    parts = line.replace(",", "").split()
                    for p in parts:
                        if p.startswith("Width="):
                            width = int(p.split("=")[1])
                        elif p.startswith("Height="):
                            height = int(p.split("=")[1])
                if width and height:
                    break
        if width is None or height is None:
            raise ValueError("full_video.hex 檔頭缺少寬高資訊")
        return width, height

    def _load_pixels(self):
        pixels = array("B")
        with self.hex_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("//"):
                    continue
                pixels.append(int(line, 16))
        return pixels

    def get_frame_pixels(self, frame_idx: int):
        if frame_idx < 0 or frame_idx >= self.total_frames:
            raise ValueError(f"Frame {frame_idx} 超出範圍 (0~{self.total_frames-1})")
        start = frame_idx * self.frame_size
        end = start + self.frame_size
        return self.pixels[start:end]

    def get_macroblock(self, frame_idx: int, mb_row: int, mb_col: int):
        frame_pixels = self.get_frame_pixels(frame_idx)
        block = []
        for r in range(MB_SIZE):
            row_offset = (mb_row + r) * self.width + mb_col
            block.extend(frame_pixels[row_offset : row_offset + MB_SIZE])
        return block


def load_golden_trace(trace_path: Path):
    if not trace_path.exists():
        raise FileNotFoundError(f"找不到黃金答案檔: {trace_path}")
    entries = []
    with trace_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("Frame="):
                continue
            m = TRACE_RE.match(line)
            if not m:
                continue
            frame, mb_row, mb_col, mvx, mvy, sad = map(int, m.groups())
            entries.append(
                {
                    "frame": frame,
                    "mb_row": mb_row,
                    "mb_col": mb_col,
                    "mv_x": mvx,
                    "mv_y": mvy,
                    "sad": sad,
                }
            )
    if not entries:
        raise ValueError("黃金答案檔案沒有任何有效資料")
    return entries


def select_cases(args, trace_entries):
    if args.frame is not None and args.mb_row is not None and args.mb_col is not None:
        for entry in trace_entries:
            if (
                entry["frame"] == args.frame
                and entry["mb_row"] == args.mb_row
                and entry["mb_col"] == args.mb_col
            ):
                return [entry]
        raise ValueError("在黃金答案中找不到指定的 Frame/MB 組合")
    count = max(1, args.case_count)
    idx = max(0, min(args.case_index, len(trace_entries) - 1))
    end_idx = min(len(trace_entries), idx + count)
    return trace_entries[idx:end_idx]


def compute_sad(block_a, block_b):
    return sum(abs(a - b) for a, b in zip(block_a, block_b))


def send_payload_with_echo(ser, data, label, progress_step=1024):
    total = len(data)
    print(f"傳送 {label} ({total} bytes)...")
    for idx, value in enumerate(data, start=1):
        byte = bytes([value])
        ser.write(byte)
        echo = ser.read(1)
        if echo != byte:
            raise RuntimeError(f"{label} 第 {idx-1} byte echo mismatch (sent {value:#04x}, got {echo})")
        if progress_step and idx % progress_step == 0:
            print(f"  -> {label} 進度 {idx}/{total}")
    print(f"✓ {label} 傳送完成")


def send_command(ser, cmd_byte, desc):
    ser.write(bytes([cmd_byte]))
    echo = ser.read(1)
    if echo != bytes([cmd_byte]):
        raise RuntimeError(f"{desc} 命令 echo mismatch (sent {cmd_byte:#04x}, got {echo})")


def compute_window_base(mb_row, mb_col, width, height):
    max_x = max(0, width - WINDOW_SIZE)
    max_y = max(0, height - WINDOW_SIZE)
    base_x = max(0, min(mb_col - SEARCH_RANGE, max_x))
    base_y = max(0, min(mb_row - SEARCH_RANGE, max_y))
    return base_x, base_y


def extract_window(frame_pixels, width, base_x, base_y):
    window = []
    for r in range(WINDOW_SIZE):
        row_offset = (base_y + r) * width + base_x
        window.extend(frame_pixels[row_offset : row_offset + WINDOW_SIZE])
    return window


def extract_block(frame_pixels, width, mb_row, mb_col):
    block = []
    for r in range(MB_SIZE):
        row_offset = (mb_row + r) * width + mb_col
        block.extend(frame_pixels[row_offset : row_offset + MB_SIZE])
    return block


def send_window(ser, base_x, base_y, window_bytes):
    header = [
        base_x & 0xFF,
        (base_x >> 8) & 0xFF,
        base_y & 0xFF,
        (base_y >> 8) & 0xFF,
    ]
    send_command(ser, CMD_LOAD_WINDOW, "LOAD_WINDOW")
    send_payload_with_echo(ser, header, "Window Header", progress_step=0)
    send_payload_with_echo(ser, window_bytes, "Reference Window")


def send_current_block(ser, mb_col, mb_row, block_bytes):
    header = [
        mb_col & 0xFF,
        (mb_col >> 8) & 0xFF,
        mb_row & 0xFF,
        (mb_row >> 8) & 0xFF,
    ]
    send_command(ser, CMD_LOAD_CUR, "LOAD_CURRENT_BLOCK")
    send_payload_with_echo(ser, header, "Current Block Header", progress_step=0)
    send_payload_with_echo(ser, block_bytes, "Current Block")


def receive_fpga_result(ser):
    result_bytes = []
    for _ in range(4):
        b = ser.read(1)
        if len(b) != 1:
            raise RuntimeError("等待 FPGA 結果逾時")
        result_bytes.append(b[0])
        ser.write(b"K")
    raw_mv_x, raw_mv_y, sad_h, sad_l = result_bytes
    parse_signed = lambda v: (v - 64) if (v & 0x20) else v
    fpga_mv_x = parse_signed(raw_mv_x & 0x3F)
    fpga_mv_y = parse_signed(raw_mv_y & 0x3F)
    fpga_sad = (sad_h << 8) | sad_l
    return fpga_mv_x, fpga_mv_y, fpga_sad


def parse_args():
    parser = argparse.ArgumentParser(description="HEXBS FPGA 快速資料驗證")
    parser.add_argument("--port", default=DEFAULT_COM_PORT, help="Serial COM port")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUDRATE, help="Baud rate")
    parser.add_argument("--hex-path", type=Path, default=GOLDEN_HEX, help="full_video.hex 路徑")
    parser.add_argument(
        "--trace-path", type=Path, default=GOLDEN_TRACE, help="golden_trace.txt 路徑"
    )
    parser.add_argument("--case-index", type=int, default=0, help="從哪個案例開始測試")
    parser.add_argument("--case-count", type=int, default=1, help="一次測試幾個案例")
    parser.add_argument("--frame", type=int, help="指定 Frame")
    parser.add_argument("--mb-row", type=int, help="指定 MB Row (像素座標)")
    parser.add_argument("--mb-col", type=int, help="指定 MB Col (像素座標)")
    return parser.parse_args()


def run_verification():
    args = parse_args()
    try:
        video = GoldenVideo(args.hex_path)
        trace_entries = load_golden_trace(args.trace_path)
        cases = select_cases(args, trace_entries)
    except Exception as exc:
        print(f"[設定錯誤] {exc}")
        return

    valid_cases = [c for c in cases if c["frame"] > 0]
    if not valid_cases:
        print("沒有 frame >= 1 的案例可測試。")
        return

    print("=== HEXBS FPGA 快速驗證模式 ===")
    print(f"影像規格: {video.width}x{video.height}, Frame Size={video.frame_size} bytes")
    print(f"本次準備測試 {len(valid_cases)} 筆案例")

    try:
        ser = serial.Serial(args.port, args.baud, timeout=5, xonxoff=False)
    except Exception as exc:
        print(f"[連線失敗] {exc}")
        return

    time.sleep(2)
    print(f"已連線 {args.port} (baud={args.baud})，開始傳輸區域資料...")

    last_window_key = None

    try:
        for entry in valid_cases:
            frame_idx = entry["frame"]
            mb_row = entry["mb_row"]
            mb_col = entry["mb_col"]
            exp_mv_x = entry["mv_x"]
            exp_mv_y = entry["mv_y"]
            exp_sad = entry["sad"]

            print(f"\n[CASE] Frame={frame_idx} MB(Row={mb_row}, Col={mb_col})")

            ref_frame = video.get_frame_pixels(frame_idx - 1)
            cur_frame = video.get_frame_pixels(frame_idx)

            base_x, base_y = compute_window_base(mb_row, mb_col, video.width, video.height)
            window_key = (frame_idx, base_x, base_y)

            if window_key != last_window_key:
                window_bytes = extract_window(ref_frame, video.width, base_x, base_y)
                send_window(ser, base_x, base_y, window_bytes)
                last_window_key = window_key

            cur_block = extract_block(cur_frame, video.width, mb_row, mb_col)
            send_current_block(ser, mb_col, mb_row, cur_block)

            ref_row = max(0, min(mb_row + exp_mv_y, video.height - MB_SIZE))
            ref_col = max(0, min(mb_col + exp_mv_x, video.width - MB_SIZE))
            try:
                ref_block = video.get_macroblock(frame_idx - 1, ref_row, ref_col)
                confirm_sad = compute_sad(cur_block, ref_block)
            except Exception:
                confirm_sad = None

            print(
                f"  黃金值 MV=({exp_mv_x}, {exp_mv_y}), SAD={exp_sad}"
                + (f" (重新計算={confirm_sad})" if confirm_sad is not None else "")
            )

            print("  等待 FPGA 回傳結果...")
            try:
                fpga_mv_x, fpga_mv_y, fpga_sad = receive_fpga_result(ser)
            except RuntimeError as exc:
                print(f"  × {exc}")
                return

            print(f"  [FPGA] MV=({fpga_mv_x}, {fpga_mv_y}), SAD={fpga_sad}")
            mv_match = (fpga_mv_x == exp_mv_x) and (fpga_mv_y == exp_mv_y)
            sad_match = (fpga_sad == exp_sad)

            if mv_match and sad_match:
                print("  ✓ 與黃金資料完全一致！")
            else:
                print("  × 與黃金資料不一致")
                if not mv_match:
                    print(f"    - MV mismatch: HW({fpga_mv_x},{fpga_mv_y}) vs EXP({exp_mv_x},{exp_mv_y})")
                if not sad_match:
                    diff = fpga_sad - exp_sad
                    print(f"    - SAD mismatch: HW={fpga_sad}, EXP={exp_sad} (差值 {diff})")
                print("    請確認視窗資料或搜尋範圍是否正確。")

    finally:
        ser.close()


if __name__ == "__main__":
    run_verification()
