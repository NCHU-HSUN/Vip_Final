import numpy as np
import os

# ==========================================
# 參數設定 (跟你的 Motion_Estimation.py 類似)
# ==========================================
SEARCH_RANGE = 32       # 與論文/硬體一致：±32 pixels
BLOCK_SIZE = 16         # H.264 標準 Macroblock 大小
FILENAME = "video/garden_sif.y4m" # 請確保路徑正確，或改成你有的影片

# ==========================================
# 簡易 Y4M 讀取器 (簡化版)
# ==========================================
class SimpleY4MReader:
    def __init__(self, filepath):
        self.file = open(filepath, 'rb')
        header = self.file.readline().decode('ascii')
        self.width = int(header.split()[1][1:])
        self.height = int(header.split()[2][1:])
        self.frame_len = self.width * self.height + (self.width * self.height) // 2
        self.file.readline() # 跳過 FRAME header

    def read_y_frame(self):
        # 讀取 FRAME header (除了第一幀外)
        if self.file.tell() > 100: 
            line = self.file.readline()
            if not line: return None
            
        y_data = self.file.read(self.width * self.height)
        # 跳過 U, V
        self.file.read((self.width * self.height) // 2)
        
        if not y_data: return None
        return np.frombuffer(y_data, dtype=np.uint8).reshape((self.height, self.width))

# ==========================================
# 核心功能：存成 Hex 檔
# ==========================================
def save_hex(data, filename, comment=""):
    """
    將 numpy array 存成 hex 格式，Verilog 的 $readmemh 可以讀取
    格式：每個 pixel 一行，例如：
    1A
    FF
    00
    ...
    """
    with open(filename, 'w') as f:
        # 扁平化 array 並轉成 hex
        flat_data = data.flatten()
        print(f"正在寫入 {filename}... 大小: {flat_data.shape}")
        for pixel in flat_data:
            f.write(f"{pixel:02X}\n") 
    print(f"✅ {filename} 儲存完成。")

# ==========================================
# 執行流程
# ==========================================
def main():
    if not os.path.exists(FILENAME):
        print(f"❌ 找不到影片: {FILENAME}，請修改程式碼中的路徑！")
        print("程式結束。")
        return        
    else:
        reader = SimpleY4MReader(FILENAME)
        ref_frame = reader.read_y_frame() # Frame 0 (Reference)
        cur_frame = reader.read_y_frame() # Frame 1 (Current)

    # -------------------------------------------------
    # 我們只針對 (0,0) 這個 Macroblock 來做硬體驗證
    # -------------------------------------------------
    mb_r, mb_c = 64, 64 # 隨便選一個中間的位置，避免邊界問題
    
    # 1. 抓取 Current Block (16x16)
    cur_block = cur_frame[mb_r:mb_r+BLOCK_SIZE, mb_c:mb_c+BLOCK_SIZE]
    
    # 2. 抓取 Search Window (Reference Frame 的一塊區域)
    # Search Window 大小通常是 (2*Range + Block_Size)
    sw_r_start = mb_r - SEARCH_RANGE
    sw_c_start = mb_c - SEARCH_RANGE
    sw_height = BLOCK_SIZE + 2 * SEARCH_RANGE
    sw_width = BLOCK_SIZE + 2 * SEARCH_RANGE
    
    search_window = ref_frame[sw_r_start : sw_r_start + sw_height, 
                              sw_c_start : sw_c_start + sw_width]

    # 3. 儲存檔案
    save_hex(cur_block, "cur_block.hex")       # 256 個 bytes
    save_hex(search_window, "search_window.hex") # (16+14)^2 = 900 個 bytes (假設 Range=7)

    print("\n=== 硬體驗證數據準備完成 ===")
    print(f"Current Block Hex: cur_block.hex (16x16)")
    print(f"Search Window Hex: search_window.hex (大小取決於 Search Range)")
    print(f"Search Range: {SEARCH_RANGE}")
    print(f"Macroblock 位置: ({mb_r}, {mb_c})")
    
    # 4. (選用) 用 Python 算一次標準答案給你看，讓你知道硬體要算出什麼
    # 這裡只做一個簡單的全搜尋示意，你的 Code 已經有 Hexagon 了
    print("\n--- Python 預算答案 (Golden Answer) ---")
    min_sad = float('inf')
    best_mv = (0, 0)
    
    # 暴力搜一次確認答案
    for dy in range(-SEARCH_RANGE, SEARCH_RANGE+1):
        for dx in range(-SEARCH_RANGE, SEARCH_RANGE+1):
            ref_blk = search_window[SEARCH_RANGE+dy : SEARCH_RANGE+dy+16,
                                    SEARCH_RANGE+dx : SEARCH_RANGE+dx+16]
            diff = cur_block.astype(int) - ref_blk.astype(int)
            sad = np.sum(np.abs(diff))
            if sad < min_sad:
                min_sad = sad
                best_mv = (dy, dx)
    
    print(f"預期硬體輸出的 MV: {best_mv}")
    print(f"預期硬體輸出的 SAD: {min_sad}")

if __name__ == "__main__":
    main()
