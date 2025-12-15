import argparse
import serial
import time
import re
from array import array
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

DEFAULT_COM_PORT = "COM10"
DEFAULT_BAUDRATE = 921600
GOLDEN_HEX = Path("golden_patterns/full_video.hex")
GOLDEN_TRACE = Path("golden_patterns/golden_trace.txt")

MB_SIZE = 16

CMD_LOAD_REF_FRAME = ord("R")
CMD_LOAD_CUR_FRAME = ord("F")
CMD_RUN_CASE = ord("C")
ACK_BYTE = ord("K")

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
    if getattr(args, "all", False):
        return trace_entries
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


def extract_block(frame_pixels, width, mb_row, mb_col):
    block = []
    for r in range(MB_SIZE):
        row_offset = (mb_row + r) * width + mb_col
        block.extend(frame_pixels[row_offset : row_offset + MB_SIZE])
    return block


class TimingTracker:
    def __init__(self):
        self.stats = defaultdict(list)
        self.labels = {}

    @contextmanager
    def measure(self, key, detail=None, summary_label=None):
        if summary_label and key not in self.labels:
            self.labels[key] = summary_label
        elif key not in self.labels:
            self.labels[key] = summary_label or detail or key
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.stats[key].append(duration)
            label = detail or self.labels.get(key, key)
            print(f"[時間] {label}: {duration:.3f}s")

    def summary(self):
        if not self.stats:
            return
        print("\n=== 時間統計摘要 ===")
        for key, values in self.stats.items():
            label = self.labels.get(key, key)
            total = sum(values)
            avg = total / len(values)
            print(
                f"{label}: 次數={len(values)}, 平均={avg:.3f}s, "
                f"最長={max(values):.3f}s, 總計={total:.3f}s"
            )


def wait_for_ack(ser, desc):
    ack = ser.read(1)
    if ack != bytes([ACK_BYTE]):
        raise RuntimeError(f"{desc} 未收到 ACK (got {ack})")


def write_all(ser, data, label, progress_step=8192):
    if isinstance(data, memoryview):
        view = data
        own_view = False
    else:
        view = memoryview(data)
        own_view = True
    total = len(view)
    sent = 0
    next_progress = progress_step if progress_step else total + 1
    print(f"傳送 {label} ({total} bytes)...")
    while sent < total:
        written = ser.write(view[sent:])
        if written is None:
            written = 0
        if written == 0:
            raise RuntimeError(f"{label} 傳輸停滯在 {sent}/{total} bytes")
        sent += written
        if progress_step and sent >= next_progress:
            print(f"  -> {label} 進度 {sent}/{total}")
            next_progress += progress_step
    print(f"✓ {label} 傳輸完成 ({total} bytes)")
    if own_view:
        view.release()


def send_frame(ser, cmd_byte, frame_idx, frame_bytes, label, progress_step):
    ser.write(bytes([cmd_byte]))
    header = bytes(
        [
            frame_idx & 0xFF,
            (frame_idx >> 8) & 0xFF,
        ]
    )
    ser.write(header)
    write_all(ser, frame_bytes, label, progress_step=progress_step)
    wait_for_ack(ser, f"{label} 完成")


def send_case_command(ser, mb_col, mb_row):
    payload = bytes(
        [
            mb_col & 0xFF,
            (mb_col >> 8) & 0xFF,
            mb_row & 0xFF,
            (mb_row >> 8) & 0xFF,
        ]
    )
    ser.write(bytes([CMD_RUN_CASE]))
    ser.write(payload)


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
    parser.add_argument(
        "--all",
        action="store_true",
        help="忽略 case-index/count，直接跑完整份黃金 trace",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="遇到與黃金資料不符時立即停止測試",
    )
    parser.add_argument(
        "--fail-limit",
        type=int,
        default=0,
        help="指定發現多少筆錯誤後停止；0 代表不限",
    )
    return parser.parse_args()


def run_verification():
    args = parse_args()
    timings = TimingTracker()
    ser = None
    try:
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

        ser.reset_input_buffer()
        ser.reset_output_buffer()
        time.sleep(2)
        print(f"已連線 {args.port} (baud={args.baud})，Frame 將以整批載入。")

        progress_step = max(1024, video.frame_size // 10)
        fail_count = 0
        fail_details = []
        stop_reason = None
        loaded_ref_frame = None
        loaded_cur_frame = None
        cached_ref_frame = None
        cached_cur_frame = None

        for entry in valid_cases:
            frame_idx = entry["frame"]
            mb_row = entry["mb_row"]
            mb_col = entry["mb_col"]
            exp_mv_x = entry["mv_x"]
            exp_mv_y = entry["mv_y"]
            exp_sad = entry["sad"]
            ref_frame_idx = frame_idx - 1
            cur_frame_idx = frame_idx

            print(f"\n[CASE] Frame={frame_idx} MB(Row={mb_row}, Col={mb_col})")

            if ref_frame_idx != loaded_ref_frame:
                cached_ref_frame = video.get_frame_pixels(ref_frame_idx)
                with timings.measure(
                    "ref_frame_transfer",
                    f"傳輸參考 Frame {ref_frame_idx}",
                    summary_label="參考 Frame 傳輸",
                ):
                    send_frame(
                        ser,
                        CMD_LOAD_REF_FRAME,
                        ref_frame_idx,
                        cached_ref_frame,
                        f"參考 Frame {ref_frame_idx}",
                        progress_step,
                    )
                loaded_ref_frame = ref_frame_idx

            if cur_frame_idx != loaded_cur_frame:
                cached_cur_frame = video.get_frame_pixels(cur_frame_idx)
                with timings.measure(
                    "cur_frame_transfer",
                    f"傳輸當前 Frame {cur_frame_idx}",
                    summary_label="當前 Frame 傳輸",
                ):
                    send_frame(
                        ser,
                        CMD_LOAD_CUR_FRAME,
                        cur_frame_idx,
                        cached_cur_frame,
                        f"當前 Frame {cur_frame_idx}",
                        progress_step,
                    )
                loaded_cur_frame = cur_frame_idx
            cur_frame = cached_cur_frame

            cur_block = extract_block(cur_frame, video.width, mb_row, mb_col)
            with timings.measure("case_command", "送出案例", summary_label="案例指令傳送"):
                send_case_command(ser, mb_col, mb_row)

            ref_row = max(0, min(mb_row + exp_mv_y, video.height - MB_SIZE))
            ref_col = max(0, min(mb_col + exp_mv_x, video.width - MB_SIZE))
            confirm_sad = None
            try:
                with timings.measure("sad_recompute", "重新計算 SAD", summary_label="CPU SAD 驗證"):
                    ref_block = video.get_macroblock(ref_frame_idx, ref_row, ref_col)
                    confirm_sad = compute_sad(cur_block, ref_block)
            except Exception:
                confirm_sad = None

            print(
                f"  黃金值 MV=({exp_mv_x}, {exp_mv_y}), SAD={exp_sad}"
                + (f" (重新計算={confirm_sad})" if confirm_sad is not None else "")
            )

            print("  等待 FPGA 回傳結果...")
            try:
                with timings.measure("fpga_latency", "等待 FPGA 結果", summary_label="FPGA 結果等待"):
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
                print("    請確認 frame 載入與搜尋範圍是否正確。")
                fail_count += 1
                fail_details.append(
                    {
                        "frame": frame_idx,
                        "mb_row": mb_row,
                        "mb_col": mb_col,
                        "exp_mv": (exp_mv_x, exp_mv_y),
                        "exp_sad": exp_sad,
                        "hw_mv": (fpga_mv_x, fpga_mv_y),
                        "hw_sad": fpga_sad,
                        "sad_diff": fpga_sad - exp_sad,
                    }
                )
                if args.fail_fast:
                    print("    已啟用 fail-fast，測試在第一個失敗點停止。")
                    stop_reason = "fail-fast"
                    break
                if args.fail_limit > 0 and fail_count >= args.fail_limit:
                    print(f"    已累積 {fail_count} 筆錯誤，達到 --fail-limit 設定，上述錯誤為最後一筆。")
                    stop_reason = "fail-limit"
                    break

        if fail_details:
            print("\n=== 測試錯誤摘要 ===")
            for idx, info in enumerate(fail_details, start=1):
                exp_mv_x, exp_mv_y = info["exp_mv"]
                hw_mv_x, hw_mv_y = info["hw_mv"]
                print(
                    f"[{idx}] Frame={info['frame']} MB(Row={info['mb_row']}, Col={info['mb_col']})"
                    f" | EXP MV=({exp_mv_x},{exp_mv_y}) SAD={info['exp_sad']}"
                    f" | HW MV=({hw_mv_x},{hw_mv_y}) SAD={info['hw_sad']} (差值 {info['sad_diff']})"
                )
            if stop_reason == "fail-fast":
                print("※ 已於第一筆錯誤後停止 (fail-fast)")
            elif stop_reason == "fail-limit":
                print(f"※ 因達到 --fail-limit 限制 ({fail_count} 筆) 而結束測試")
    finally:
        if ser is not None:
            ser.close()
        timings.summary()


if __name__ == "__main__":
    run_verification()
