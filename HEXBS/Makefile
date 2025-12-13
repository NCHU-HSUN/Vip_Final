# Makefile for HEXBS Verification

# Default Simulator: vcs (Synopsys), iverilog (Icarus), or ncverilog (Cadence)
SIM ?= vcs

# Source Files
TB_FILE   = golden.v
# Find all .v files in current directory as DUT, excluding the testbench
DUT_FILES = $(filter-out $(TB_FILE), $(wildcard *.v))

# Output
EXEC      = simv
LOG       = sim.log

.PHONY: all run clean

all: run

run:
	@echo "Starting Simulation with $(SIM)..."
ifeq ($(SIM), vcs)
	# VCS Compilation and Run
	vcs -full64 -sverilog -debug_access+all \
		$(TB_FILE) $(DUT_FILES) \
		-o $(EXEC) \
		-l $(LOG)
	./$(EXEC) +dump_wave
else ifeq ($(SIM), iverilog)
	# Icarus Verilog Compilation and Run
	iverilog -g2012 -o $(EXEC) $(TB_FILE) $(DUT_FILES)
	vvp $(EXEC) +dump_wave
else ifeq ($(SIM), ncverilog)
	# NC-Verilog Run
	ncverilog +access+rwc +v2k \
		$(TB_FILE) $(DUT_FILES) \
		+dump_wave
else
	@echo "Error: SIM=$(SIM) not supported. Use vcs, iverilog, or ncverilog."
endif

clean:
	rm -rf $(EXEC) $(EXEC).daidir csrc *.log *.vcd ucli.key