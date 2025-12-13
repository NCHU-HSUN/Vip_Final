# Makefile for HEXBS Verification using iverilog

# Source Files
TB_FILE   = golden.v
# Find all .v files in current directory as DUT, excluding the testbench
DUT_FILES = $(filter-out $(TB_FILE), $(wildcard *.v))

# Output
EXEC      = simulation
LOG       = sim.log

.PHONY: all run clean

all: run

run:
	@echo "Starting Simulation with iverilog..."
	# Icarus Verilog Compilation and Run
	iverilog -g2012 -o $(EXEC) $(TB_FILE) $(DUT_FILES)
	vvp $(EXEC)

clean:
	rm -rf $(EXEC) *.log *.vcd
