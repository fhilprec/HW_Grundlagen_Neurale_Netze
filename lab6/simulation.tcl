vlib work
vlog testbench_mul.v
vlog netlist_mul.v
vlog NangateOpenCellLibrary.v
vsim -sdftype /testbench_mul/dut=sdf_full.sdf work.testbench_mul;# annotate multiplier with sdf file
power add /testbench_mul/dut/*
run 10 us
power report -all -bsaif power_full.saif; # create power report
quit -f
