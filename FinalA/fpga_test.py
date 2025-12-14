import argparse
import serial
import time
import re
from array import array
from pathlib import Path

DEFAULT_COM_PORT = "COM10"
DEFAULT_BAUDRATE = 115200
GOLDEN_HEX = Path("golden_patterns/full_video.hex")
GOLDEN_TRACE = Path("golden_patterns/golden_trace.txt")

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

    def get_macroblock(self, frame_idx: int, mb_row: int, mb_col: int):
        if frame_idx < 0 or frame_idx >= self.total_frames:
            raise ValueError(f"Frame {frame_idx} 超出範圍 (0~{self.total_frames-1})")
        if mb_row < 0 or mb_col < 0 or mb_row + 16 > self.height or mb_col + 16 > self.width:
            raise ValueError("Macroblock 座標超出畫面範圍")
        frame_offset = frame_idx * self.frame_size
        block = []
        for r in range(16):
            row_offset = frame_offset + (mb_row + r) * self.width + mb_col
            block.extend(self.pixels[row_offset : row_offset + 16])
        return block

    def get_frame(self, frame_idx: int):
        if frame_idx < 0 or frame_idx >= self.total_frames:
            raise ValueError(f"Frame {frame_idx} 超出範圍 (0~{self.total_frames-1})")
        start = frame_idx * self.frame_size
        end = start + self.frame_size
        return self.pixels[start:end].tolist()

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

def select_case(args, trace_entries):
    if args.frame is not None and args.mb_row is not None and args.mb_col is not None:
        for entry in trace_entries:
            if (
                entry["frame"] == args.frame
                and entry["mb_row"] == args.mb_row
                and entry["mb_col"] == args.mb_col
            ):
                return entry
        raise ValueError("在黃金答案中找不到指定的 Frame/MB 組合")
    idx = max(0, min(args.case_index, len(trace_entries) - 1))
    return trace_entries[idx]

def compute_sad(block_a, block_b):
    return sum(abs(a - b) for a, b in zip(block_a, block_b))

def send_payload_with_echo(ser, data, label, progress_step=8192):
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

def parse_args():
    parser = argparse.ArgumentParser(description="HEXBS FPGA Real-Data Verifier")
    parser.add_argument("--port", default=DEFAULT_COM_PORT, help="Serial COM port")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUDRATE, help="Baud rate")
    parser.add_argument(
        "--hex-path", type=Path, default=GOLDEN_HEX, help="full_video.hex 路徑"
    )
    parser.add_argument(
        "--trace-path", type=Path, default=GOLDEN_TRACE, help="golden_trace.txt 路徑"
    )
    parser.add_argument(
        "--case-index",
        type=int,
        default=0,
        help="要測試的黃金答案索引 (與 golden_trace.txt 位置一致)",
    )
    parser.add_argument("--frame", type=int, help="指定 Frame 編號 (覆寫 case-index)")
    parser.add_argument("--mb-row", type=int, help="指定 MB Row (像素座標，需為 16 的倍數)")
    parser.add_argument("--mb-col", type=int, help="指定 MB Col (像素座標，需為 16 的倍數)")
    return parser.parse_args()

def run_verification():
    args = parse_args()
    try:
        video = GoldenVideo(args.hex_path)
        trace_entries = load_golden_trace(args.trace_path)
        target = select_case(args, trace_entries)
    except Exception as exc:
        print(f"[設定錯誤] {exc}")
        return

    frame_idx = target["frame"]
    if frame_idx == 0:
        print("Frame 0 沒有上一張參考圖，請挑選 frame >= 1 的資料")
        return

    print("=== HEXBS FPGA 真實資料驗證 ===")
    print(f"影像規格: {video.width}x{video.height}, Frame Size={video.frame_size} bytes")
    print(
        f"測試樣本: Frame {frame_idx} | MB(Row={target['mb_row']}, Col={target['mb_col']})"
    )

    try:
        cur_block = video.get_macroblock(frame_idx, target["mb_row"], target["mb_col"])
    except Exception as exc:
        print(f"[讀取資料失敗] {exc}")
        return

    exp_mv_x = target["mv_x"]
    exp_mv_y = target["mv_y"]
    exp_sad = target["sad"]

    try:
        ref_block = video.get_macroblock(
            frame_idx - 1, target["mb_row"] + exp_mv_y, target["mb_col"] + exp_mv_x
        )
        confirm_sad = compute_sad(cur_block, ref_block)
    except Exception:
        confirm_sad = None

    print(
        f"黃金答案: MV=({exp_mv_x}, {exp_mv_y}), SAD={exp_sad}"
        + (f" (重新計算={confirm_sad})" if confirm_sad is not None else "")
    )

    try:
        ser = serial.Serial(args.port, args.baud, timeout=5, xonxoff=False)
    except Exception as exc:
        print(f"[連線失敗] {exc}")
        return

    time.sleep(2)
    print(f"已連線到 {args.port}，開始傳送影像資料...")

    try:
        ref_frame = video.get_frame(frame_idx - 1)
        cur_frame = video.get_frame(frame_idx)
        send_payload_with_echo(ser, ref_frame, "Reference Frame")
        send_payload_with_echo(ser, cur_frame, "Current Frame")
        mb_cfg = [
            target["mb_col"] & 0xFF,
            (target["mb_col"] >> 8) & 0xFF,
            target["mb_row"] & 0xFF,
            (target["mb_row"] >> 8) & 0xFF,
        ]
        send_payload_with_echo(ser, mb_cfg, "MB Config", progress_step=0)
        print("✓ 影像資料與配置下載完成")

        print("等待 FPGA 回傳結果...")
        result_bytes = []
        for _ in range(4):
            b = ser.read(1)
            if len(b) != 1:
                print("× 等待 FPGA 結果逾時")
                ser.close()
                return
            result_bytes.append(b[0])
            ser.write(b"K")

        raw_mv_x, raw_mv_y, sad_h, sad_l = result_bytes
        parse_signed = lambda v: (v - 64) if (v & 0x20) else v
        fpga_mv_x = parse_signed(raw_mv_x & 0x3F)
        fpga_mv_y = parse_signed(raw_mv_y & 0x3F)
        fpga_sad = (sad_h << 8) | sad_l

        print(f"[FPGA] MV=({fpga_mv_x}, {fpga_mv_y}), SAD={fpga_sad}")
        mv_match = (fpga_mv_x == exp_mv_x) and (fpga_mv_y == exp_mv_y)
        sad_match = (fpga_sad == exp_sad)

        if mv_match and sad_match:
            print("✓ 與黃金資料完全一致！")
        else:
            print("× 與黃金資料不一致")
            if not mv_match:
                print(
                    f"  - MV mismatch: HW({fpga_mv_x},{fpga_mv_y}) vs EXP({exp_mv_x},{exp_mv_y})"
                )
            if not sad_match:
                diff = fpga_sad - exp_sad
                print(f"  - SAD mismatch: HW={fpga_sad}, EXP={exp_sad} (差值 {diff})")
            print("  請確認 Smart_Top 的記憶體資料來源與 HEXBS 版本是否同步。")

    finally:
        ser.close()

if __name__ == "__main__":
    run_verification()
