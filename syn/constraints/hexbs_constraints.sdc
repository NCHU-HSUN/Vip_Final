# HEXBS synthesis constraints (SDC)

# Clock definition: 100 MHz default (10 ns period)
create_clock -name clk -period 10 [get_ports clk]

# Define input transition / drive strength (placeholder)
set_drive 0.02 [all_inputs]

# Define output load (placeholder capacitance)
set_load 0.05 [all_outputs]

# Input/output delays relative to clk (assume 20% cycle)
set_input_delay 2 -clock clk [remove_from_collection [all_inputs] [list clk rst_n]]
set_output_delay 2 -clock clk [all_outputs]

# Reset as asynchronous, exclude from timing
set_false_path -from [get_ports rst_n]

# Multi-cycle consideration for done flag if handshake allows (example: 2 cycles)
# set_multicycle_path 2 -to [get_ports done]
