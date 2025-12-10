import numpy as np
from typing import IO, Iterator, Tuple, List, Dict, Optional, Callable, Any
import itertools
import os
import sys
import math
import time
from datetime import datetime
import concurrent.futures

# --- 1. 匯入 Numba JIT ---
try:
    from numba import jit, int32, int64
    from numba.experimental import jitclass
except ImportError:
    print("警告: 'numba' 未安裝。程式將以純 Python 慢速模式執行。")
    def jit(nopython=True, nogil=True):
        def decorator(func): return func
        return decorator
    def jitclass(cls_or_spec=None, spec=None):
        def decorator(cls): return cls
        return decorator
    int32 = int
    int64 = int

# --- 2. 匯入輔助套件 ---
try:
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils import get_column_letter
except ImportError:
    openpyxl = None

try:
    from PIL import Image
except ImportError:
    print("警告: 未安裝 Pillow，無法輸出驗證圖片。")
    Image = None

# ==========================================
# 核心類別
# ==========================================
class Frame:
    def __init__(self, y: np.ndarray, cb: np.ndarray, cr: np.ndarray):
        self.y, self.cb, self.cr = y, cb, cr
        self.height, self.width = y.shape
        
    @staticmethod
    def create_blank(width, height):
        return Frame(np.full((height, width), 128, dtype=np.uint8),
                     np.full((height//2, width//2), 128, dtype=np.uint8),
                     np.full((height//2, width//2), 128, dtype=np.uint8))
                     
    def put_y_macroblock(self, r, c, block):
        self.y[r:r+16, c:c+16] = block

class Y4MReader:
    def __init__(self, filepath: str):
        self.file = open(filepath, 'rb')
        self._parse_header()

    def _parse_header(self):
        line = self.file.readline().decode('ascii').strip()
        for p in line.split(' '):
            if p.startswith('W'): self.width = int(p[1:])
            elif p.startswith('H'): self.height = int(p[1:])
        self.y_size = self.width * self.height
        self.c_size = (self.width // 2) * (self.height // 2)

    def __iter__(self): return self

    def __next__(self) -> Frame:
        if not self.file.readline(): raise StopIteration
        y = np.frombuffer(self.file.read(self.y_size), dtype=np.uint8).reshape((self.height, self.width))
        c_sz = self.c_size
        cb = np.frombuffer(self.file.read(c_sz), dtype=np.uint8).reshape((self.height//2, self.width//2))
        cr = np.frombuffer(self.file.read(c_sz), dtype=np.uint8).reshape((self.height//2, self.width//2))
        return Frame(y, cb, cr)

    def close(self): self.file.close()

def pad_frame(frame: Frame) -> Frame:
    h, w = frame.y.shape
    ph, pw = (16 - h % 16) % 16, (16 - w % 16) % 16
    if ph == 0 and pw == 0: return frame
    return Frame(np.pad(frame.y, ((0, ph), (0, pw)), 'edge'),
                 np.pad(frame.cb, ((0, ph//2), (0, pw//2)), 'edge'),
                 np.pad(frame.cr, ((0, ph//2), (0, pw//2)), 'edge'))

# ==========================================
# JIT 函式
# ==========================================
spec = [('mv_r', int32), ('mv_c', int32), ('sad', int64), ('check_points', int32)]
@jitclass(spec)
class MotionVectorResult:
    def __init__(self, mv_r, mv_c, sad, check_points):
        self.mv_r, self.mv_c, self.sad, self.check_points = mv_r, mv_c, sad, check_points

@jit(nopython=True, nogil=True) 
def calculate_sad(b1: np.ndarray, b2: np.ndarray) -> int:
    return np.sum(np.abs(b1.astype(np.int16) - b2.astype(np.int16)))

@jit(nopython=True, nogil=True) 
def get_ref_block(ref: np.ndarray, r: int, c: int, dr: int, dc: int) -> np.ndarray:
    h, w = ref.shape
    rr = min(max(r + dr, 0), h - 16)
    cc = min(max(c + dc, 0), w - 16)
    return ref[rr:rr+16, cc:cc+16]

def calc_psnr(orig: Frame, recon: Frame) -> float:
    mse = np.mean((orig.y.astype(np.float32) - recon.y.astype(np.float32)) ** 2)
    return 20 * math.log10(255.0) - 10 * math.log10(mse) if mse > 0 else float('inf')

# ==========================================
# 演算法 (Full, TSS, Diamond, HEXBS)
# ==========================================
@jit(nopython=True, nogil=True)
def algo_full_search(cur, ref, r, c, rng):
    best_sad, best_dr, best_dc, checks = 999999, 0, 0, 0
    for dr in range(-rng, rng + 1):
        for dc in range(-rng, rng + 1):
            sad = calculate_sad(cur, get_ref_block(ref, r, c, dr, dc))
            checks += 1
            if sad < best_sad: best_sad, best_dr, best_dc = sad, dr, dc
    return MotionVectorResult(best_dr, best_dc, best_sad, checks)

@jit(nopython=True, nogil=True)
def algo_tss(cur, ref, r, c, rng):
    best_sad, best_dr, best_dc, checks, step = 999999, 0, 0, 0, 4
    while step >= 1:
        offsets = [(0, 0)] if step == 4 else []
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i or j: offsets.append((i, j))
        ls, lr, lc = best_sad, best_dr, best_dc
        for i, j in offsets:
            dr, dc = best_dr + i * step, best_dc + j * step
            if not (-rng <= dr <= rng and -rng <= dc <= rng): continue
            sad = calculate_sad(cur, get_ref_block(ref, r, c, dr, dc))
            checks += 1
            if sad < ls: ls, lr, lc = sad, dr, dc
        best_sad, best_dr, best_dc, step = ls, lr, lc, step // 2
    return MotionVectorResult(best_dr, best_dc, best_sad, checks)

@jit(nopython=True, nogil=True)
def algo_diamond(cur, ref, r, c, rng):
    LDSP = [(0,0), (0,2), (0,-2), (2,0), (-2,0), (1,1), (1,-1), (-1,1), (-1,-1)]
    SDSP = [(0,0), (0,1), (0,-1), (1,0), (-1,0)]
    cache = np.zeros((2*rng+1, 2*rng+1), dtype=np.int8)
    checks, best_dr, best_dc = 0, 0, 0
    best_sad = calculate_sad(cur, get_ref_block(ref, r, c, 0, 0))
    cache[rng, rng] = 1; checks += 1
    
    while True:
        ctr_r, ctr_c, found = best_dr, best_dc, False
        for i, j in LDSP:
            dr, dc = ctr_r + i, ctr_c + j
            if not (-rng<=dr<=rng and -rng<=dc<=rng) or cache[dr+rng, dc+rng]: continue
            sad = calculate_sad(cur, get_ref_block(ref, r, c, dr, dc))
            checks += 1; cache[dr+rng, dc+rng] = 1
            if sad < best_sad: best_sad, best_dr, best_dc, found = sad, dr, dc, True
        if not found: break
    for i, j in SDSP:
        dr, dc = best_dr + i, best_dc + j
        if not (-rng<=dr<=rng and -rng<=dc<=rng) or cache[dr+rng, dc+rng]: continue
        sad = calculate_sad(cur, get_ref_block(ref, r, c, dr, dc))
        checks += 1; cache[dr+rng, dc+rng] = 1
        if sad < best_sad: best_sad, best_dr, best_dc = sad, dr, dc
    return MotionVectorResult(best_dr, best_dc, best_sad, checks)

@jit(nopython=True, nogil=True)
def algo_hexbs(cur, ref, r, c, rng):
    LHP = [(2,0), (1,2), (-1,2), (-2,0), (-1,-2), (1,-2)]
    SHP = [(1,0), (0,1), (-1,0), (0,-1)]
    cache = np.zeros((2*rng+1, 2*rng+1), dtype=np.int8)
    checks, best_dr, best_dc = 0, 0, 0
    best_sad = calculate_sad(cur, get_ref_block(ref, r, c, 0, 0))
    cache[rng, rng] = 1; checks += 1
    
    while True:
        ctr_r, ctr_c, center_best = best_dr, best_dc, True
        for i, j in LHP:
            dr, dc = ctr_r + i, ctr_c + j
            if not (-rng<=dr<=rng and -rng<=dc<=rng) or cache[dr+rng, dc+rng]: continue
            sad = calculate_sad(cur, get_ref_block(ref, r, c, dr, dc))
            checks += 1; cache[dr+rng, dc+rng] = 1
            if sad < best_sad: best_sad, best_dr, best_dc, center_best = sad, dr, dc, False
        if center_best: break
    for i, j in SHP:
        dr, dc = best_dr + i, best_dc + j
        if not (-rng<=dr<=rng and -rng<=dc<=rng) or cache[dr+rng, dc+rng]: continue
        sad = calculate_sad(cur, get_ref_block(ref, r, c, dr, dc))
        checks += 1; cache[dr+rng, dc+rng] = 1
        if sad < best_sad: best_sad, best_dr, best_dc = sad, dr, dc
    return MotionVectorResult(best_dr, best_dc, best_sad, checks)

# ==========================================
# 分析流程
# ==========================================
def process_row_task(row_idx, width, cur_y, ref_y, algo_func, rng):
    results = []
    for col_idx in range(0, width, 16):
        cur_mb = cur_y[row_idx : row_idx+16, col_idx : col_idx+16].copy()
        res = algo_func(cur_mb, ref_y, row_idx, col_idx, rng)
        results.append(res)
    return row_idx, results

# [新增] 視覺化驗證函式
def save_verification_images(vname, algo_name, cur_frame, recon_frame, out_dir):
    if Image is None: return
    
    # 1. 儲存重建圖 (Reconstructed)
    recon_img = Image.fromarray(recon_frame.y)
    recon_img.save(f"{out_dir}/{vname}_{algo_name}_Recon.png")
    
    # 2. 儲存誤差圖 (Residual / Difference)
    # 取絕對值差，並放大 5 倍讓誤差更明顯 (方便肉眼檢查)
    diff = np.abs(cur_frame.y.astype(int) - recon_frame.y.astype(int)).astype(np.uint8)
    diff_enhanced = diff * 5 
    diff_img = Image.fromarray(diff_enhanced)
    diff_img.save(f"{out_dir}/{vname}_{algo_name}_Error.png")

def analyze_video(name, path, algos, rng, executor, save_images=False, out_dir=""):
    # print(f"    -> Analyzing {name}...", end='', flush=True)
    if not os.path.exists(path): return {}
    
    stats = {k: {'psnr':[], 'time':[], 'checks':[]} for k in algos}
    reader = Y4MReader(path)
    
    try:
        ref = pad_frame(next(reader))
    except StopIteration: return {}
        
    try:
        frame_idx = 0
        while True:
            try: cur = pad_frame(next(reader))
            except StopIteration: break
            
            for algo_name, algo_func in algos.items():
                t_start = time.perf_counter()
                futures = []
                for r in range(0, cur.height, 16):
                    futures.append(executor.submit(process_row_task, r, cur.width, cur.y, ref.y, algo_func, rng))
                
                total_checks = 0
                recon = Frame.create_blank(cur.width, cur.height)
                for f in concurrent.futures.as_completed(futures):
                    r_idx, row_res = f.result()
                    for i, mb_res in enumerate(row_res):
                        c_idx = i * 16
                        total_checks += mb_res.check_points
                        ref_blk = get_ref_block(ref.y, r_idx, c_idx, mb_res.mv_r, mb_res.mv_c)
                        recon.put_y_macroblock(r_idx, c_idx, ref_blk)
                
                t_end = time.perf_counter()
                mb_count = (cur.height // 16) * (cur.width // 16)
                stats[algo_name]['time'].append(t_end - t_start)
                stats[algo_name]['checks'].append(total_checks / mb_count)
                stats[algo_name]['psnr'].append(calc_psnr(cur, recon))
                
                # [關鍵] 只在第 1 幀 (Frame 0) 輸出驗證圖片
                if save_images and frame_idx == 0:
                    save_verification_images(name, algo_name, cur, recon, out_dir)
            
            ref = cur
            frame_idx += 1
    finally:
        reader.close()
    return stats

# ==========================================
# 繪圖與報表
# ==========================================
def save_excel_report(all_results, output_path):
    if not openpyxl: return
    wb = openpyxl.Workbook()
    if "Sheet" in wb.sheetnames: del wb["Sheet"]
    
    ws_sum = wb.create_sheet("Summary (Avg 10 Runs)")
    ws_sum.append(["Video", "Algorithm", "Avg PSNR (dB)", "Avg Checks", "Avg Time (s)", "PSNR Diff", "Checks Red.%", "Speedup"])
    
    for cell in ws_sum[1]: cell.font = Font(bold=True, color="FFFFFF"); cell.fill = PatternFill(start_color="4F81BD", fill_type="solid")

    for vname, algos_data in all_results.items():
        fs_psnr, fs_checks, fs_time = None, None, None
        for algo, d in algos_data.items():
            if "Full" in algo:
                fs_psnr, fs_checks, fs_time = np.mean(d['psnr']), np.mean(d['checks']), np.mean(d['time'])
                break
        
        for algo, d in algos_data.items():
            avg_p, avg_c, avg_t = np.mean(d['psnr']), np.mean(d['checks']), np.mean(d['time'])
            p_diff, c_red, spd = "-", "-", "-"
            if fs_psnr:
                p_diff = f"{avg_p - fs_psnr:+.3f}"
                c_red = f"{(fs_checks - avg_c)/fs_checks*100:.1f}%"
                spd = f"{fs_time/avg_t:.1f}x" if avg_t > 0 else "-"
            ws_sum.append([vname, algo, round(avg_p,2), round(avg_c,1), round(avg_t,4), p_diff, c_red, spd])
            
    wb.save(output_path)
    print(f"\n✅ Excel 報告已儲存: {output_path}")

def plot_charts(video_name, stats, out_dir):
    import matplotlib.pyplot as plt
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 16))
    ax1.set_title(f'{video_name} - PSNR'); ax1.set_ylabel('dB')
    ax2.set_title(f'{video_name} - Checks'); ax2.set_ylabel('Count')
    ax3.set_title(f'{video_name} - Time'); ax3.set_ylabel('Sec')
    
    fs_txt = "Full Search (Benchmark):\n"
    has_fs = False
    for name, data in stats.items():
        x = range(len(data['time']))
        if 'Full' in name:
            has_fs = True
            fs_txt += f" Checks:{np.mean(data['checks']):.1f}, Time:{np.mean(data['time']):.4f}s"
            ax1.plot(x, data['psnr'], 'k--', alpha=0.5, label=name)
        else:
            ax1.plot(x, data['psnr'], label=name); ax2.plot(x, data['checks'], label=name); ax3.plot(x, data['time'], label=name)
            
    for ax in [ax1, ax2, ax3]: ax.legend(); ax.grid(True)
    if has_fs: fig.text(0.5, 0.02, fs_txt, ha='center', bbox=dict(fc='white', alpha=0.9))
    fig.savefig(f"{out_dir}/{video_name}_Chart.png")
    plt.close(fig)

if __name__ == '__main__':
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"Final_Result_{ts}"
    os.makedirs(out_dir, exist_ok=True)
    
    VIDEOS = {
        "Garden": r"video/garden_sif.y4m", 
        "Football": r"video/football_cif.y4m", 
        "Tennis": r"video/tennis_sif.y4m"
    }
    ALGOS = {"FullSearch": algo_full_search, "TSS": algo_tss, "Diamond": algo_diamond, "HEXBS": algo_hexbs}
    
    workers = os.cpu_count() or 4
    NUM_RUNS = 10
    final_avg = {}
    
    print(f"=== 啟動驗證與效能測試 (Workers: {workers}) ===")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for run_i in range(NUM_RUNS):
            print(f"\n📢 Run {run_i + 1}/{NUM_RUNS} ...")
            # 只有第一次執行 (Run 0) 會輸出驗證圖片
            save_imgs = (run_i == 0)
            
            for vname, vpath in VIDEOS.items():
                print(f"   -> {vname}...", end='', flush=True)
                cur_res = analyze_video(vname, vpath, ALGOS, 15, executor, save_images=save_imgs, out_dir=out_dir)
                print(" Done")
                
                if vname not in final_avg:
                    final_avg[vname] = {algo: {'psnr': d['psnr'], 'checks': d['checks'], 'time': d['time']} for algo, d in cur_res.items()}
                else:
                    for algo in ALGOS:
                        final_avg[vname][algo]['time'] = [sum(x) for x in zip(final_avg[vname][algo]['time'], cur_res[algo]['time'])]

        print("\n📊 計算平均值並產出報告...")
        for vname in final_avg:
            for algo in final_avg[vname]:
                final_avg[vname][algo]['time'] = [t/NUM_RUNS for t in final_avg[vname][algo]['time']]
            plot_charts(vname, final_avg[vname], out_dir)
            
        save_excel_report(final_avg, f"{out_dir}/Final_Report.xlsx")
        print("\n=== 全部完成，請檢查資料夾中的 _Error.png 圖片是否正確 ===")