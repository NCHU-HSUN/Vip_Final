import serial
import time
import random

# ==========================================
# 設定區
# ==========================================
COM_PORT = 'COM10'  # 請確認 Port
BAUD_RATE = 115200
WIDTH = 32          
HEIGHT = 240
REF_START_ADDR = 1000
MB_X_POS = 0
MB_Y_POS = 0

class VirtualFPGAEnv:
    def __init__(self, width, height, ref_start):
        self.width = width
        self.height = height
        self.ref_start = ref_start

    def get_ref_pixel(self, x, y):
        pixel_addr = self.ref_start + (y * self.width) + x
        return pixel_addr & 0xFF 

    def get_ref_block(self, top_left_x, top_left_y):
        max_x = self.width - 16
        max_y = self.height - 16
        clamp_x = max(0, min(top_left_x, max_x))
        clamp_y = max(0, min(top_left_y, max_y))
        
        block = []
        for r in range(16):
            for c in range(16):
                px = clamp_x + c
                py = clamp_y + r
                block.append(self.get_ref_pixel(px, py))
        return block

def calculate_sad(block_a, block_b):
    sad = 0
    for i in range(256):
        sad += abs(block_a[i] - block_b[i])
    return sad

def run_verification():
    print(f"--- 啟動 FPGA 終極驗證 V3 (WIDTH={WIDTH}) ---")
    
    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=5, xonxoff=False)
        time.sleep(2) 
        print("連線成功。")

        # 1. 產生測試資料
        random.seed()
        test_data = [random.randint(0, 255) for _ in range(256)]
        
        # 計算特徵值
        data_sum = sum(test_data)
        print(f"★ 本次亂數資料總和 (CheckSum): {data_sum}")
        print("(如果不按 Reset，FPGA 可能會忽視這組新資料)")

        print("1. 發送並驗證輸入資料...")
        data_error = False
        for i in range(256):
            sent_byte = bytes([test_data[i]])
            ser.write(sent_byte)
            echo = ser.read(1)
            if echo != sent_byte:
                data_error = True
        
        if data_error:
            print("❌ 傳輸錯誤 (Echo Mismatch)")
            return
        else:
            print("✅ 傳輸成功 (Echo Match)")

        print("2. 接收 FPGA 運算結果...")
        result_bytes = []
        for i in range(4):
            b = ser.read(1)
            if len(b) == 1:
                result_bytes.append(b[0])
                ser.write(b'K')
            else:
                print("Timeout waiting for result."); return

        # 解析結果
        raw_mv_x, raw_mv_y, sad_h, sad_l = result_bytes
        def parse_signed(val): return (val - 64) if (val & 0x20) else val
        fpga_mv_x = parse_signed(raw_mv_x & 0x3F)
        fpga_mv_y = parse_signed(raw_mv_y & 0x3F)
        fpga_sad  = (sad_h << 8) | sad_l
        
        print(f"\n[FPGA 回傳] MV=({fpga_mv_x}, {fpga_mv_y}), SAD={fpga_sad}")
        
        # Python 驗算
        env = VirtualFPGAEnv(WIDTH, HEIGHT, REF_START_ADDR)
        check_block = env.get_ref_block(MB_X_POS + fpga_mv_x, MB_Y_POS + fpga_mv_y)
        check_sad = calculate_sad(test_data, check_block)
        
        print(f"[Python 驗算] SAD={check_sad}", end="")
        
        if check_sad == fpga_sad:
            print(" ✅ MATCH!")
            print("\n🎉 恭喜！本次驗證完全正確！")
        else:
            print(" ❌ MISMATCH")
            print(f"差異: {abs(check_sad - fpga_sad)}")
            print("\n💡 提示：如果上次成功，這次失敗，請記得在執行前按一下 FPGA 上的 btnC (Reset)！")

        ser.close()

    except Exception as e:
        print(f"錯誤: {e}")

if __name__ == "__main__":
    run_verification()