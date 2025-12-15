# Stage 4 – Logic Synthesis (HEXBS)

本資料夾提供將 `hexbs_top` 交給邏輯合成工具 (以 Synopsys Design Compiler 為例) 的步驟。

## 1. 目錄/檔案說明

| 路徑 | 說明 |
| --- | --- |
| `syn/filelist.f` | 列出待合成的 RTL 檔案 (目前只需 `HEXBS/me_hexbs_topB.v`) |
| `syn/constraints/hexbs_constraints.sdc` | 預設 SDC (10ns clock、簡易 I/O 需求，可依 ASIC target 調整) |
| `syn/scripts/dc_compile_hexbs.tcl` | Design Compiler 自動化腳本 |
| `syn/reports/` | 合成報告輸出位置 |
| `syn/work/` | DC WORK library |

## 2. 執行條件

1. 事先取得目標 standard cell library (`*.db`) 與對應 link lib。
2. 在 shell 設定以下環境變數：

```bash
export PROJ_DIR=/workspaces/Vip_Final
export TARGET_LIBS="/path/to/typical.db"
export LINK_LIBS="* $TARGET_LIBS"
```

若需要多個 corner，可在 `TARGET_LIBS`/`LINK_LIBS` 中加入多個 .db。

## 3. 合成步驟

```bash
cd $PROJ_DIR
dc_shell -f syn/scripts/dc_compile_hexbs.tcl | tee syn/logs/dc.log
```

腳本會自動：

1. `analyze`/`elaborate` `hexbs_top`
2. 套用 `syn/constraints/hexbs_constraints.sdc`
3. `compile_ultra`
4. 產生報告 (`syn/reports/*.rpt`) 與 netlist/SDF (`syn/hexbs_top_syn.v`, `syn/hexbs_top_syn.sdf`)

## 4. 合成後驗證 (Stage 5 前的建議)

1. 將 `syn/hexbs_top_syn.v` 與技術庫 cell netlist 一起交給模擬器。
2. 使用 `HEXBS/golden.v` testbench (或 gate-level 專用 testbench) 驗證功能是否仍與 Stage‑3 一致。
3. 若 Timing report 顯示違反，從 `hexbs_timing.rpt` 中找出 critical path，回到 RTL 調整 (例如 SAD 累加器、FSM) 後重跑腳本。

## 5. 可依需求調整

- 修改 `syn/filelist.f` 以加入額外子模組。
- 在 `syn/constraints/hexbs_constraints.sdc` 中補上 interface-specific 的 `set_input_delay`/`set_output_delay` 或多時脈設定。
- 若採用其他合成工具 (如 Yosys/OpenROAD flow)，可沿用同一份 filelist / SDC，只需替換執行腳本。
