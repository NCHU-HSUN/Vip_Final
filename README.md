# 用於 VLSI 設計的運動估計算法分析

本專案實作並分析了多種用於影片壓縮的運動估計算法，主要專注於為硬體實現而優化的六邊形-菱形搜尋（Hexagon-Diamond Search, HEXBS）演算法。專案提供了基於 Python 的工具，用於演算法模擬、性能比較以及為 Verilog 驗證生成黃金測試向量（Golden Vectors）。

## 主要功能

- **演算法比較：** 實作並比較以下運動估計算法：
  - 全域搜尋（Full Search, FS）
  - 三步搜尋（Three-Step Search, TSS）
  - 菱形搜尋（Diamond Search, DS）
  - 六邊形-菱形搜尋（Hexagon-Diamond Search, HEXBS）
- **性能分析：** 生成詳細的 Excel（`.xlsx`）報告和圖表，基於以下指標比較演算法性能：
  - 峰值信噪比（PSNR）
  - 平均搜尋點數（Checks）
  - 執行時間
- **黃金向量生成：** 從指定的影片檔案中，為 Verilog 模擬創建參考數據（`golden_trace.txt`）和記憶體初始化檔案（`full_video.hex`）。
- **硬體實現：** 包含一個 HEXBS 運動估計模組的 Verilog 實現。
- **合成支援：** 包含用於合成 Verilog 模組的腳本和約束檔案。

## 專案結構

```
/workspaces/Vip_Final/
├─── Motion_EstimationG.py   # 用於演算法分析和比較的主要 Python 腳本
├─── gen_golden.py           # 用於生成 Verilog 黃金向量的 Python 腳本
├─── requirements.txt        # Python 相依套件
├─── README.md               # 本檔案
├─── video/                  # 用於測試的來源影片檔案
├─── golden_patterns/        # 生成的黃金向量輸出目錄
├─── ME_Ultimate_*/          # 分析結果（報告、圖表）的輸出目錄
├─── HEXBS/                  # HEXBS 硬體模組的 Verilog 原始碼
│    ├─── me_hexbs_topB.v
│    └─── Makefile           # 用於 Verilog 模擬的 Makefile
└─── syn/                    # 合成腳本與報告
```

## 環境建置與安裝

1.  **複製儲存庫：**
    ```bash
    git clone <repository-url>
    cd Vip_Final
    ```

2.  **安裝 Python 相依套件：**
    請確保您已安裝 Python 3。然後，使用 `requirements.txt` 檔案安裝所需的套件。
    ```bash
    pip install -r requirements.txt
    ```

## 使用方法

### 1. 演算法分析

若要對所有提供的影片執行完整的演算法比較分析，請執行主要腳本。結果（包含一份 Excel 報告和比較圖表）將被保存在一個名為 `ME_Ultimate_<timestamp>` 的新目錄中。

```bash
python Motion_EstimationG.py
```

### 2. 黃金向量生成

若要為 Verilog 測試平台生成基於 `garden_sif.y4m` 影片的黃金向量，請執行 `gen_golden.py` 腳本。輸出檔案將被放置在 `golden_patterns/` 目錄中。

```bash
python gen_golden.py
```

您可以修改腳本中的 `VIDEO_PATH` 變數來指定其他影片。

## 硬體（Verilog）組件

`HEXBS/` 目錄包含了運動估計模組的 Verilog 實現。您可以使用提供的 `Makefile` 搭配相容的 Verilog 模擬器（例如 VCS）來執行模擬。模擬將使用 `gen_golden.py` 腳本生成的黃金向量來驗證其正確性。

### HEXBS 用法

1.  **進入 HEXBS 目錄：**
    ```bash
    cd HEXBS/
    ```

2.  **執行模擬：**
    使用 `make` 指令來編譯並執行 Verilog 模擬。預設使用 VCS 模擬器。
    ```bash
    make
    ```
    您也可以透過 `SIM` 變數指定其他支援的模擬器（例如 `iverilog` 或 `ncverilog`）：
    ```bash
    make SIM=iverilog
    ```

3.  **清除模擬檔案：**
    若要清除所有模擬產生的檔案（例如 `simv`, `sim.log`, `*.vcd`），請執行以下指令：
    ```bash
    make clean
    ```