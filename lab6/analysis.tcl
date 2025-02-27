set target_library NangateOpenCellLibrary_typical.db ; # specify library
set link_library NangateOpenCellLibrary_typical.db

read_verilog netlist_full.v ; # read netlist
read_saif -input power_full.saif -instance_name testbench_mul/mul ; # read SAIF
report_power -analysis_effort high > report.txt ; # create power report
quit