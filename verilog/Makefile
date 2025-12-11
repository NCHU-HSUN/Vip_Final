# ==========================================
#  Verilog 通用自動化腳本 (Makefile)
#  適用：Icarus Verilog + GTKWave / WaveTrace
# ==========================================

# 1. 變數設定
# 找出當前資料夾下所有的 .v 檔案
SRCS = $(wildcard *.v)
# 設定輸出執行檔的名稱
PROJ = simulation
# 設定波形檔名稱 (必須跟你的 testbench $dumpfile 裡面的名字一樣)
WAVE = waveform.vcd

# 2. 主要指令 (輸入 make 會執行的動作)
# 邏輯：先編譯(compile) -> 再執行(run)
all: clean compile run

# 編譯：把所有 .v 檔變成一個執行檔
compile:
	@echo "🔨 Compiling..."
	iverilog -o $(PROJ) $(SRCS)

# 執行：跑模擬，並產生波形
run:
	@echo "🚀 Running Simulation..."
	vvp $(PROJ)
	@echo "✅ Done! Waveform generated: $(WAVE)"

# 3. 清除指令 (輸入 make clean 會執行的動作)
clean:
	@echo "🧹 Cleaning up..."
	rm -f $(PROJ) $(WAVE)

# 偽目標設定 (避免資料夾剛好有跟指令同名的檔案)
.PHONY: all compile run clean