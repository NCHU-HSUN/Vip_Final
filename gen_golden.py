import os
import sys

import numpy as np

# 引用原本的程式碼，確保算法邏輯 100% 一致
from Motion_EstimationG import DEFAULT_SEARCH_RANGE, Y4MReader, pad_frame, algo_hexbs, Frame

# ================= 使用者設定區 (User Settings) =================
# 影片路徑
VIDEO_PATH = r"video/garden_sif.y4m" 

# 限制處理幀數 (None 代表處理全部，可用環境變數 HEXBS_MAX_FRAMES 覆寫)
ENV_MAX_FRAMES = os.environ.get("HEXBS_MAX_FRAMES")
MAX_FRAMES_TO_PROCESS = None if not ENV_MAX_FRAMES else int(ENV_MAX_FRAMES)

# 搜尋範圍 (必須與 Verilog 與 Python 主程式一致，可用 HEXBS_SEARCH_RANGE 覆寫)
SEARCH_RANGE = int(os.environ.get("HEXBS_SEARCH_RANGE", DEFAULT_SEARCH_RANGE))

# 輸出檔名
OUTPUT_DIR = "golden_patterns"
OUTPUT_HEX_FILE = os.path.join(OUTPUT_DIR, "full_video.hex")       # 給 Verilog 讀的記憶體檔
OUTPUT_TRACE_FILE = os.path.join(OUTPUT_DIR, "golden_trace.txt")   # 給 Verilog 比對的答案卷
# ==============================================================

def write_frame_to_hex(f_handle, frame_data):
    """將一張 Frame 的像素轉成 Hex 寫入檔案"""
    # frame_data 是 2D numpy array (Height x Width)
    # 我們按 Raster Scan 順序 (由左到右，由上到下) 寫入
    flat_pixels = frame_data.flatten()
    for p in flat_pixels:
        f_handle.write(f"{p:02X}\n")

def run_full_generation():
    print(f"🚀 開始執行全影片數據生成...")
    print(f"📂 讀取影片: {VIDEO_PATH}")
    print(f"🔧 搜尋範圍設定: ±{SEARCH_RANGE}")
    if MAX_FRAMES_TO_PROCESS:
        print(f"🧪 只處理前 {MAX_FRAMES_TO_PROCESS} 幀以加速驗證")
    
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ 錯誤: 找不到影片檔案 {VIDEO_PATH}")
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        reader = Y4MReader(VIDEO_PATH)
    except Exception as e:
        print(f"❌ 讀取失敗: {e}")
        return

    # 準備輸出檔案
    f_hex = open(OUTPUT_HEX_FILE, "w")
    f_trace = open(OUTPUT_TRACE_FILE, "w")

    # 寫入 Hex 檔頭資訊 (註解)，方便你查看寬高
    # 注意：第一幀讀進來後才知道 Padding 後的大小
    try:
        raw_ref = next(reader)
        ref = pad_frame(raw_ref) # Padding 後的 Reference Frame
    except StopIteration:
        print("❌ 影片太短，無法讀取第一幀")
        return

    h, w = ref.y.shape
    print(f"ℹ️  影像規格 (含 Padding): 寬={w}, 高={h}")
    print(f"ℹ️  每個 Frame 佔用的記憶體大小: {w*h} bytes")
    print(f"ℹ️  Verilog 參數建議: FRAME_WIDTH={w}, FRAME_HEIGHT={h}")
    
    f_hex.write(f"// Image Info: Width={w}, Height={h}\n")
    f_hex.write(f"// Frame Size: {w*h} bytes per frame\n")
    f_hex.write(f"// Structure: [Frame 0 Data] [Frame 1 Data] ...\n")

    # 1. 寫入第一幀 (Frame 0 / Reference) 到 Hex
    print(f"💾 正在寫入 Frame 0 (Reference)...")
    write_frame_to_hex(f_hex, ref.y)

    frame_count = 1
    total_mbs = (h // 16) * (w // 16)

    while True:
        # 檢查是否達到設定的上限
        if MAX_FRAMES_TO_PROCESS and frame_count >= MAX_FRAMES_TO_PROCESS:
            print(f"✋ 已達到設定的幀數上限 ({MAX_FRAMES_TO_PROCESS})，停止處理。")
            break

        try:
            raw_cur = next(reader)
            cur = pad_frame(raw_cur) # Padding 後的 Current Frame
        except StopIteration:
            break # 影片讀完了

        print(f"🔄 正在處理 Frame {frame_count} (Current)...")
        
        # 2. 寫入當前幀 (Current) 到 Hex
        # Verilog 記憶體會接著上一幀的屁股繼續寫
        f_hex.write(f"// --- Start of Frame {frame_count} ---\n")
        write_frame_to_hex(f_hex, cur.y)

        # 3. 執行 HEXBS 算法產生黃金答案
        # 遍歷每一個 Macroblock
        f_trace.write(f"--- Frame {frame_count} Analysis ---\n")
        
        mb_idx = 0
        for r in range(0, h, 16):
            for c in range(0, w, 16):
                # 擷取 Current Block
                cur_block = cur.y[r:r+16, c:c+16]
                
                # 呼叫你的 HEXBS 算法
                # 注意：這裡傳入的是整張 ref.y，算法內部會自己處理邊界與搜尋
                # 這正是你想要的「在 Verilog 內部切割」的模擬
                result = algo_hexbs(cur_block, ref.y, r, c, SEARCH_RANGE)
                
                # 寫入 Trace 檔
                # 格式: MB_X(Col), MB_Y(Row), MV_X, MV_Y, SAD
                # 為了方便 Verilog 比對，我們用比較好 parse 的格式
                f_trace.write(f"Frame={frame_count} MB_Row={r:<4} MB_Col={c:<4} | MV_X={result.mv_c:<3} MV_Y={result.mv_r:<3} SAD={result.sad}\n")
                mb_idx += 1
        
        # 更新 Reference Frame，準備下一輪 (Frame N 變成 Frame N-1)
        ref = cur
        frame_count += 1

    f_hex.close()
    f_trace.close()
    reader.close()
    
    print(f"\n✅ 完成！")
    print(f"📄 1. 記憶體檔案: {OUTPUT_HEX_FILE} (請在 Verilog 使用 $readmemh 讀取)")
    print(f"📄 2. 黃金答案卷: {OUTPUT_TRACE_FILE} (用於檢查正確性)")
    print(f"💡 提示: 在 Verilog 中，Frame 0 從位址 0 開始，Frame 1 從位址 {w*h} 開始。")

if __name__ == "__main__":
    run_full_generation()
