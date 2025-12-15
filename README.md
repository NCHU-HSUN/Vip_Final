# HEXBS 運動估計軟硬體整合專案

本專案聚焦於 Hexagon-Based Search (HEXBS) 運動估計演算法，提供「Python 端演算法分析 → 黃金向量生成 → Verilog RTL 驗證 → 合成 (DC)」的一條龍流程，方便在 VLSI/FPGA 專案中快速驗證與交付硬體 IP。

## 功能亮點

- **多演算法 Benchmark：** `Motion_EstimationG.py` 同時實作 Full Search、Three-Step Search、Diamond Search 與 HEXBS，支援多執行緒＋Numba JIT 加速。
- **完整報表與視覺化：** 自動輸出 PSNR、平均搜尋點 (Checks)、執行時間的 Excel 報表、曲線圖 (`ME_Ultimate_*`) 以及可視化驗證影像。
- **Golden Pattern 產生器：** `gen_golden.py` 從 Y4M 影片生成 `full_video.hex`（供 `$readmemh`）與 `golden_trace.txt`（參考答案），可搭配環境變數快速調整搜尋範圍/幀數。
- **HEXBS 硬體 IP：** `HEXBS/me_hexbs_topB.v` 為完整 RTL，含 testbench (`HEXBS/golden.v`)、架構文件與 IP 規格，支援自動化 Makefile 模擬流程。
- **(未驗證)合成支援：** `syn/` 內建 Design Compiler 腳本、SDC 以及報告輸出路徑，方便銜接 Stage 4/5。

## 目錄總覽

```
Vip_Final/
├── Motion_EstimationG.py      # 演算法分析主程式
├── gen_golden.py              # Golden Pattern 產生器
├── golden_patterns/           # full_video.hex、golden_trace.txt 會輸出到此處
├── video/                     # Garden/Football/Tennis 範例影片 (Y4M)
├── ME_Ultimate_*/             # 每次分析的報告、圖表、驗證影像、HW trace
├── HEXBS/
│   ├── me_hexbs_topB.v        # RTL
│   ├── golden.v               # system-level testbench
│   ├── HEXBS_Architecture.md  # 架構圖 (Mermaid)
│   └── IP Specification.md    # I/O 定義與操作流程
├── FinalA/                    # Vivado 2025.1 專案 (Smart_Top + UART host script)
├── syn/                       # DC filelist / SDC / 腳本 / report 目錄
├── Cell-Based_Design_Flow.md  # ASIC Flow 筆記
├── VLSI_Architecture_Design_of_Motion_Estimation_Block...pdf
└── README.md
```

## 系統需求

- Python 3.10+（建議使用虛擬環境）  
- `pip install -r requirements.txt`（Numba、NumPy、Matplotlib、OpenPyXL、Pillow…）  
- Verilog 模擬器：VCS、NC-Verilog、或 Icarus Verilog (Makefile 參數 `SIM=`)  
- 合成工具（選用）：Synopsys Design Compiler + Standard Cell `.db`

## 建置與安裝

```bash
git clone <repository-url>
cd Vip_Final
python -m venv .venv && source .venv/bin/activate   # 選用
pip install -r requirements.txt
```

## Python 演算法分析流程

執行 `Motion_EstimationG.py` 會自動掃描 `video/` 底下的 Garden / Football / Tennis 影片並跑 3 輪 (`NUM_RUNS=3`) 比較。

```bash
python Motion_EstimationG.py
```

### 輸出內容

- `ME_Ultimate_<timestamp>/ME_Ultimate_Result_<timestamp>.xlsx`：Summary、Per Loop、Frame-by-Frame 詳表。
- `ME_Ultimate_<timestamp>/plots/*.png`：各演算法 PSNR / Checks / Time 曲線。
- `ME_Ultimate_<timestamp>/verification_images/`：第一輪的重建影像與差異圖。
- `ME_Ultimate_<timestamp>/*_hw_mapping_trace.json`：HEXBS FSM/搜尋紀錄，可對照 RTL 狀態機。

### 常用參數

- `HEXBS_SEARCH_RANGE`：覆寫預設 ±32 搜尋範圍（Python 與 RTL 要一致）。
- `video/` 內可換成自備的 Y4M 影片；若放在其他路徑，請直接修改 `VIDEОS` 或使用軟連結。

## 黃金向量 (Golden Patterns) 產生

以 Python 端的 HEXBS 演算法輸出硬體驗證資料：

```bash
python gen_golden.py
```

輸出檔位於 `golden_patterns/`：

- `full_video.hex`：一幀接一幀排列的像素資料，供 testbench `$readmemh`.
- `golden_trace.txt`：逐 Macroblock 的 MV/SAD 參考答案，testbench 會逐行對比。

環境變數：

- `HEXBS_MAX_FRAMES`：例如 `export HEXBS_MAX_FRAMES=3` 只產生前 3 幀，加速驗證。
- `HEXBS_SEARCH_RANGE`：需與 RTL/主程式一致，預設 32。
- `VIDEO_PATH`（直接編輯腳本）或利用軟連結指向其他影片。

## HEXBS Verilog 模組與模擬

`HEXBS/me_hexbs_topB.v` 為 RTL，`HEXBS/golden.v` 為 system-level testbench，會自動載入 `golden_patterns/` 的資料並跑完整 Frame/MB sweep，內建邊界條件測試與 `waveform_boundary.vcd` dump。

### 模擬步驟

1. 產生最新的 `full_video.hex` / `golden_trace.txt`。  
2. 進入 `HEXBS/`：`cd HEXBS`.  
3. 執行 `make`（預設 VCS）；或指定 `make SIM=iverilog`。  
4. 檢查 `sim.log` 與終端輸出是否出現 `[FAIL]`，無錯誤時會輸出 `PERFECT MATCH!`。  
5. `make clean` 可移除 `simv`、`*.vcd`、暫存檔。

重要檔案：

- `HEXBS_Architecture.md`：以 Mermaid 描述 Control FSM、Address/SAD/Decision datapath。
- `IP Specification.md`：I/O、搜尋流程、效能估算，方便串接 SoC。
- `simulation/`：可放模擬波形或附加腳本。

## 合成與後續流程

`syn/` 目錄提供 Design Compiler 範例流程：

1. 編輯 `syn/filelist.f` 與 `syn/constraints/hexbs_constraints.sdc`（時脈預設 10ns）。  
2. 設定 `TARGET_LIBS` / `LINK_LIBS` 環境變數。  
3. 執行：`dc_shell -f syn/scripts/dc_compile_hexbs.tcl | tee syn/logs/dc.log`.  
4. 產出 `syn/hexbs_top_syn.v`、`syn/hexbs_top_syn.sdf`、`syn/reports/*.rpt`。  
5. 進行 Gate-level Simulation 時，可沿用 `HEXBS/golden.v` 測試平台。

更多 ASIC Flow 筆記請參考 `Cell-Based_Design_Flow.md` 與隨附 PDF。

## Vivado FPGA 原型 (Nexys A7)

`FinalA/` 目錄是針對 Digilent Nexys A7 (XC7A35T-1CPG236C) 建立的 Vivado 2025.1 專案，主體由 `Smart_Top` 結合 `hexbs_top`、雙埠 BRAM 影像緩衝與自訂 UART 協定 (921600 bps)。LED[15:8] 會顯示最近一次運動向量的 X 分量，LED[7:0] 顯示 Y 分量，方便肉眼確認。

### 建構 Bitstream

1. 至 `FinalA/` 打開 `FinalA.xpr`（Vivado 2025.1+）。  
2. 確認 `sources_1/new` 中的 `Smart_Top.v`、`UART_Modules.v`、`hexbs_top.v` 都在工程內，目標還原點為 `xc7a35tcpg236-1`。  
3. `Flow Navigator → Generate Bitstream`，完成後於 Hardware Manager 將 bitstream 燒錄至 Nexys A7（100 MHz `clk`、`btnC` 做 reset、`RsRx`/`RsTx` 對應 USB-UART）。  

### UART 驗證腳本

`FinalA/fpga_test.py` 會利用 `full_video.hex` / `golden_trace.txt` 透過 UART 將 Frame 與待測 Macroblock 座標串流到 FPGA：

```bash
pip install pyserial
python FinalA/fpga_test.py --port COM10 --case-count 20
```

- 會自動以 `CMD_LOAD_REF (R)` / `CMD_LOAD_CUR (F)` 傳輸整幀，再用 `CMD_RUN_CASE (C)` 觸發 HEXBS，並對照回傳的 MV/SAD。  
- `--all` 可跑完整個 `golden_trace`，或用 `--frame/--mb-row/--mb-col` 指定單一案例。  
- `--fail-fast`、`--fail-limit` 可在硬體結果不符時及早停止。  

執行腳本前請先用 `gen_golden.py` 產生最新的 golden patterns，並確認 FPGA 已燒錄最新版 bitstream。腳本輸出會同步列出 CPU 重新計算的 SAD（作為雙重驗證），也會列出任何不一致案例以便追蹤。

## 相關參考

- `VLSI_Architecture_Design_of_Motion_Estimation_Block_with_Hexagon-Diamond_Search_Pattern_for_Real-Time_Video_Processing.pdf`：理論背景。
- `verilog/`：簡化版的 counter 示範，可作為基礎模板。
- `FinalA/`：Vivado 專案試驗檔案（僅供參考）。

如需擴充，建議以此 README 為導覽，依序從 Python 演算法 → Golden Pattern → RTL 模擬 → 合成與實體化，確保軟硬體保持一致。
