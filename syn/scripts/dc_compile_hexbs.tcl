###########################################################################
# HEXBS Logic Synthesis Script (Design Compiler)
#
# Usage:
#   export PROJ_DIR=/workspaces/Vip_Final
#   export TARGET_LIBS="/path/to/lib/typical.db"
#   export LINK_LIBS="* $TARGET_LIBS"
#   dc_shell -f syn/scripts/dc_compile_hexbs.tcl | tee syn/logs/dc.log
###########################################################################

if { ![info exists ::env(PROJ_DIR)] } {
    echo "ERROR: Please export PROJ_DIR to repository root."
    quit -f
}

set proj_dir $::env(PROJ_DIR)
set rtl_filelist [file join $proj_dir syn/filelist.f]
set sdc_file     [file join $proj_dir syn/constraints/hexbs_constraints.sdc]
set report_dir   [file join $proj_dir syn/reports]
set work_dir     [file join $proj_dir syn/work]

file mkdir $report_dir
file mkdir $work_dir

# Library setup (expects env vars to be set outside)
if { ![info exists ::env(TARGET_LIBS)] } {
    echo "ERROR: TARGET_LIBS env variable not set."
    quit -f
}
set_app_var search_path [list $proj_dir $proj_dir/HEXBS $proj_dir/syn ]
set target_library $::env(TARGET_LIBS)
set link_library   $::env(LINK_LIBS)

set_app_var uniquify_naming_style "%s_%d"
set_app_var hdlin_auto_save_templates true

define_design_lib WORK -path $work_dir

# Read RTL
analyze -format sverilog -f $rtl_filelist
elaborate hexbs_top
current_design hexbs_top
link

check_design

# Apply constraints
if {[file exists $sdc_file]} {
    source -echo $sdc_file
} else {
    echo "WARNING: SDC file $sdc_file not found, proceeding without explicit constraints."
}

# Compile
compile_ultra -no_autoungroup

# Reports
report_timing  -max_paths 10 -delay_type max >  $report_dir/hexbs_timing.rpt
report_area                     >  $report_dir/hexbs_area.rpt
report_power -analysis_effort medium > $report_dir/hexbs_power.rpt
check_design                    >  $report_dir/hexbs_check.rpt

# Outputs
write -hierarchy -format verilog -output $proj_dir/syn/hexbs_top_syn.v
write_sdf $proj_dir/syn/hexbs_top_syn.sdf

echo "Synthesis complete. Netlist: syn/hexbs_top_syn.v"
quit
