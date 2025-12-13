import serial
import time
import re
from tqdm import tqdm

# ==========================================
# 1. 設定 & UART 指令集
# ==========================================
COM_PORT = 'COM10'  # 請根據您的環境修改
BAUD_RATE = 115200

# --- 與 Smart_Top.v 中的指令集對應 ---
CMD_WRITE_MEM       = 0xA0
CMD_RUN_TEST        = 0xA1
CMD_SET_FRAME_ADDRS = 0xA2
CMD_ECHO            = 0xA3

# --- 影像/記憶體參數 ---
VIDEO_WIDTH  = 352
VIDEO_HEIGHT = 240
FRAME_SIZE   = VIDEO_WIDTH * VIDEO_HEIGHT

# ==========================================
# 2. UART 通訊輔助函式
# ==========================================
class FpgaDriver:
    def __init__(self, port, baud):
        self.ser = serial.Serial(port, baud, timeout=2)
        time.sleep(2) # 等待序列埠穩定

    def _send_cmd(self, cmd):
        self.ser.write(bytes([cmd]))

    def _send_bytes(self, data):
        self.ser.write(data)

    def _read_bytes(self, num):
        return self.ser.read(num)

    def check_connection(self):
        print("Checking connection...")
        self._send_cmd(CMD_ECHO)
        response = self._read_bytes(1)
        if response == bytes([0xEE]):
            print("Connection successful (ECHO OK).")
            return True
        print(f"Connection failed. Expected 0xEE, got {response.hex()}.")
        return False

    def set_frame_addrs(self, cur_addr, ref_addr):
        self._send_cmd(CMD_SET_FRAME_ADDRS)
        self._send_bytes(cur_addr.to_bytes(4, 'big'))
        self._send_bytes(ref_addr.to_bytes(4, 'big'))

    def write_memory(self, addr, data):
        chunk_size = 128 # 每次傳輸的塊大小
        for i in tqdm(range(0, len(data), chunk_size), desc=f"Writing to 0x{addr:06X}"):
            chunk = data[i:i+chunk_size]
            current_addr = addr + i
            
            self._send_cmd(CMD_WRITE_MEM)
            # 傳送 3B Address + 1B count
            self._send_bytes(current_addr.to_bytes(3, 'big'))
            self._send_bytes(len(chunk).to_bytes(1, 'big'))
            # 傳送數據
            self._send_bytes(chunk)
            time.sleep(0.001) # 給予 FPGA 一點處理時間

    def run_test(self, mb_x, mb_y):
        self._send_cmd(CMD_RUN_TEST)
        self._send_bytes(mb_x.to_bytes(2, 'big'))
        self._send_bytes(mb_y.to_bytes(2, 'big'))
        
        result_bytes = []
        for _ in range(4):
            b = self._read_bytes(1)
            if not b:
                print("Error: Timeout waiting for result byte from FPGA.")
                return None
            result_bytes.append(b[0])
            self._send_bytes(b'K') # Send ACK

        # 解析結果
        raw_mv_x, raw_mv_y, sad_h, sad_l = result_bytes
        def parse_signed(val): 
            val = val & 0x3F # 確保只取 6 位
            return (val - 64) if (val & 0x20) else val

        fpga_mv_x = parse_signed(raw_mv_x)
        fpga_mv_y = parse_signed(raw_mv_y)
        fpga_sad  = (sad_h << 8) | sad_l
        return (fpga_mv_x, fpga_mv_y, fpga_sad)

    def close(self):
        self.ser.close()

# ==========================================
# 3. 資料解析函式
# ==========================================
def parse_golden_trace(filepath):
    print(f"Parsing golden trace: {filepath}")
    tests = []
    with open(filepath, 'r') as f:
        for line in f:
            if not line.strip() or line.startswith('---'):
                continue
            
            # 使用正則表達式解析
            match = re.search(r'Frame=(\d+)\s+MB_Row=(\d+)\s+MB_Col=(\d+)\s+\|\s+MV_X=(-?\d+)\s+MV_Y=(-?\d+)\s+SAD=(\d+)', line)
            if match:
                test = {
                    'frame': int(match.group(1)),
                    'mb_y': int(match.group(2)),
                    'mb_x': int(match.group(3)),
                    'exp_mv_x': int(match.group(4)),
                    'exp_mv_y': int(match.group(5)),
                    'exp_sad': int(match.group(6))
                }
                tests.append(test)
    print(f"Found {len(tests)} test cases.")
    return tests

def load_video_hex(filepath):
    print(f"Loading video hex: {filepath}")
    with open(filepath, 'r') as f:
        hex_string = f.read().replace('\n', '').replace(' ', '')
    return bytes.fromhex(hex_string)

# ==========================================
# 4. 主測試流程
# ==========================================
def main():
    print("--- FPGA Golden Verification System (方案B) ---")
    
    # --- 步驟 1: 載入與解析資料 ---
    try:
        golden_tests = parse_golden_trace('../golden_patterns/golden_trace.txt')
        video_data = load_video_hex('../golden_patterns/full_video.hex')
    except FileNotFoundError as e:
        print(f"Error: Required file not found - {e}. Make sure you are running from the 'FinalA' directory.")
        return

    # --- 步驟 2: 連線與初始化 FPGA ---
    try:
        driver = FpgaDriver(COM_PORT, BAUD_RATE)
        print(f"\nConnected to {COM_PORT} at {BAUD_RATE} baud.")
        
        print("\nIMPORTANT: Please press the 'btnC' (Reset) on the FPGA now.")
        input("Press Enter to continue after resetting the board...")

        if not driver.check_connection():
            return
            
        # --- 步驟 3: 上傳影像資料 ---
        print("\nUploading Frame 0 to FPGA RAM...")
        driver.write_memory(0, video_data[0:FRAME_SIZE])
        
        print("\nUploading Frame 1 to FPGA RAM...")
        driver.write_memory(FRAME_SIZE, video_data[FRAME_SIZE : FRAME_SIZE*2])

        # 設定初始幀位址 (測試 Frame 1, 參考 Frame 0)
        current_frame_in_fpga = 1
        driver.set_frame_addrs(FRAME_SIZE, 0) # Cur=Frame1, Ref=Frame0
        print(f"\nInitial frame addresses set: CUR=0x{_c:06X}, REF=0x{_r:06X}".format(_c=FRAME_SIZE, _r=0))

    except Exception as e:
        print(f"An error occurred during setup: {e}")
        return

    # --- 步驟 4: 執行測試迴圈 ---
    total_tests = 0
    passed_tests = 0
    
    print("\n--- Starting Test Execution ---")
    
    # 篩選要執行的測試，可以從這邊修改要跑的範圍
    tests_to_run = [t for t in golden_tests if t['frame'] > 0]

    for test in tests_to_run:
        total_tests += 1
        frame_idx = test['frame']
        
        print(f"\nRunning Test #{total_tests}: Frame={frame_idx}, MB=({test['mb_x']}, {test['mb_y']})")
        
        # --- 檢查是否需要載入新幀 ---
        if frame_idx != current_frame_in_fpga:
            print(f"Frame change detected. Loading Frame {frame_idx}...")
            # 我們輪流使用兩個 RAM buffer
            if frame_idx % 2 == 1: # 奇數幀 (1, 3, 5...)
                # 載入到 Buffer 0, 參考 Buffer 1
                ref_addr = FRAME_SIZE
                cur_addr = 0
                driver.write_memory(cur_addr, video_data[frame_idx*FRAME_SIZE : (frame_idx+1)*FRAME_SIZE])
            else: # 偶數幀 (2, 4, 6...)
                # 載入到 Buffer 1, 參考 Buffer 0
                ref_addr = 0
                cur_addr = FRAME_SIZE
                driver.write_memory(cur_addr, video_data[frame_idx*FRAME_SIZE : (frame_idx+1)*FRAME_SIZE])

            driver.set_frame_addrs(cur_addr, ref_addr)
            current_frame_in_fpga = frame_idx
            print(f"New addresses set: CUR=0x{cur_addr:06X}, REF=0x{ref_addr:06X}")

        # --- 執行單次測試 ---
        hw_result = driver.run_test(test['mb_x'], test['mb_y'])

        if hw_result is None:
            print("[FAIL] Did not receive a valid result from FPGA.")
            continue
            
        hw_mv_x, hw_mv_y, hw_sad = hw_result
        exp_mv_x, exp_mv_y, exp_sad = test['exp_mv_x'], test['exp_mv_y'], test['exp_sad']

        # --- 比對結果 ---
        mv_ok = (hw_mv_x == exp_mv_x) and (hw_mv_y == exp_mv_y)
        sad_ok = (hw_sad == exp_sad)

        if mv_ok and sad_ok:
            passed_tests += 1
            print(f"[PASS] | Exp MV:({exp_mv_x},{exp_mv_y}) | HW MV:({hw_mv_x},{hw_mv_y}) | SAD Exp:{exp_sad} HW:{hw_sad}")
        else:
            print(f"[FAIL] | Exp MV:({exp_mv_x},{exp_mv_y}) | HW MV:({hw_mv_x},{hw_mv_y}) | SAD Exp:{exp_sad} HW:{hw_sad}")
            # 可以在這邊設中斷點或直接結束
            # break 

    # --- 步驟 5: 產出總結報告 ---
    print("\n--- Test Summary ---")
    print(f"Total Tests Run: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    
    driver.close()
    print("\nVerification complete. Serial port closed.")

if __name__ == "__main__":
    main()
