# ==============================================================================
#  Filename    : synthesis.tcl
#  Author      : HSUN
#  Description : Standard Logic Synthesis Script for HEXBS
#  Version     : 1.0 (Foolproof Edition)
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. 環境設定 (Setting the Kitchen)
# 告訴大廚，你的食材在哪，廚具在哪
# ------------------------------------------------------------------------------
set TOP_MODULE "HEXBS_Top"          ; # 設定你的頂層模組名稱 (記得改成你真正的 Top Name)
set SRC_DIR    "../src"             ; # 原始碼路徑

# 設定製程庫 (這裡假設只有一個 slow corner 的庫，實際可能會有更多)
# search_path 是告訴 DC 去哪裡找檔案
set search_path "$search_path . $SRC_DIR ../lib" 
set target_library "your_process_slow.db" ; # 這是晶圓廠給的目標庫 (Target)
set link_library   "* $target_library"    ; # 連結庫，* 代表記憶體中已有的設計

# ------------------------------------------------------------------------------
# 2. 讀入設計 (Reading & Translating)
# 翻譯：把 Verilog 讀進來，並理解它的結構
# ------------------------------------------------------------------------------
sh rm -rf work                  ; # 清除舊的暫存
define_design_lib WORK -path ./work

# Analyze: 檢查語法錯誤 (像是在檢查食材有沒有壞掉)
analyze -format verilog [list $SRC_DIR/hexbs.v $SRC_DIR/sub_module.v] 

# Elaborate: 建立電路架構 (把食材切好，準備開火)
# 這一步會把參數 (Parameter) 展開，建立泛型的邏輯圖
elaborate $TOP_MODULE

# 檢查是否有浮接 (Floating) 或多重驅動 (Multi-driven) 的問題
current_design $TOP_MODULE
link
check_design

# ------------------------------------------------------------------------------
# 3. 設定約束 (Setting Constraints) - 這是最關鍵的一步！
# 翻譯：給大廚下死命令，這道菜必須多快做好
# ------------------------------------------------------------------------------
# 創造一個時脈，週期設為 10ns (即 100MHz)
# [get_ports clk] 這裡的 clk 必須跟你 Verilog input 的名稱一模一樣
create_clock -name "clk" -period 10 -waveform {0 5} [get_ports clk]

# 告訴 DC 不要優化 clk 線路本身 (這是佈局佈線 APR 的事)
set_dont_touch_network [get_clocks clk]

# 設定 I/O 延遲 (這是在模擬晶片外部環境的延遲)
set_input_delay  1.0 -clock clk [all_inputs]
set_output_delay 1.0 -clock clk [all_outputs]

# ------------------------------------------------------------------------------
# 4. 開始合成 (Mapping & Optimization)
# 翻譯：大火快炒！將泛型邏輯閘換成台積電的邏輯閘，並用力擠壓面積和時間
# ------------------------------------------------------------------------------
# compile_ultra 是強力合成模式，會自動做很多優化
compile_ultra

# ------------------------------------------------------------------------------
# 5. 輸出結果與報告 (Reporting)
# 翻譯：上菜，並附上營養成分表
# ------------------------------------------------------------------------------
# 寫出 Netlist (這就是變身後的 Verilog，裡面都是 AND, OR, DFF...)
write -format verilog -hierarchy -output "../netlist/${TOP_MODULE}_syn.v"

# 寫出 SDF (Standard Delay Format) - 這是給後續模擬用的「時間延遲資訊」
write_sdf -version 2.1 "../netlist/${TOP_MODULE}_syn.sdf"

# 寫出 DDC (這是 DC 的資料庫格式，以後要看波形或 debug 可以讀這個)
write -format ddc -hierarchy -output "../netlist/${TOP_MODULE}_syn.ddc"

# 產生報告
report_area > ../report/area.rpt  ; # 面積報告 (看用了多少 gate count)
report_timing > ../report/timing.rpt ; # 時間報告 (看有沒有 setup time violation)
report_power > ../report/power.rpt  ; # 功耗報告

puts "=================================================================="
puts "  Synthesis Finished! Check reports in ../report/  "
puts "=================================================================="