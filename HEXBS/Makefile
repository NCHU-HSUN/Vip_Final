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
# 預設不輸出波形，必要時可改成 1 或用 +dump_wave 覆寫
WAVE_DUMP ?= 0
VVP_ARGS ?=
# Icarus 編譯旗標 (可在這裡統一設定)
IVERILOG_FLAGS = -g2012 -DWAVEFILE=\"$(WAVE)\" -DENABLE_WAVE_DUMP_DEFAULT=$(WAVE_DUMP)

# 2. 主要指令 (輸入 make 會執行的動作)
# 邏輯：先編譯(compile) -> 再執行(run)
all: clean compile run

# 編譯：把所有 .v 檔變成一個執行檔
compile:
	@echo "🔨 Compiling..."
	iverilog -o $(PROJ) $(IVERILOG_FLAGS) $(SRCS)

# 執行：跑模擬，可透過 WAVE_DUMP/VVP_ARGS 控制是否輸出波形
run:
	@echo "🚀 Running Simulation..."
	vvp $(PROJ) $(VVP_ARGS)
	@if [ "$(WAVE_DUMP)" = "1" ] || printf '%s' "$(VVP_ARGS)" | grep -q "dump_wave"; then \
		echo "✅ Waveform generated: $(WAVE)"; \
	else \
		echo "ℹ️  Waveform dump disabled (use make wave or +dump_wave)."; \
	fi

# 額外目標：針對所有情境 (波形一定輸出)
wave: clean
	@echo "🌊 Running with waveform dump enabled (+dump_wave)..."
	$(MAKE) compile WAVE_DUMP=1
	$(MAKE) run WAVE_DUMP=1 VVP_ARGS=+dump_wave

# 3. 清除指令 (輸入 make clean 會執行的動作)
clean:
	@echo "🧹 Cleaning up..."
	rm -f $(PROJ) $(WAVE)

# 偽目標設定 (避免資料夾剛好有跟指令同名的檔案)
.PHONY: all compile run wave clean
