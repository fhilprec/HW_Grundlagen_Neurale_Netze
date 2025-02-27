vlib work ; # create library work

vlog testbench_mul.v ; # add file to library
vlog mul.v
vsim work.testbench_mul ; # simulate module testbench_add in lib work

run 10 us ; # run simulation for 10 us
quit -f