vlib work ; # create library work

vlog testbench_neuron.v ; # add file to library
vlog neuron.v
vsim work.testbench_neuron ; # simulate module testbench_add in lib work

run 10 us ; # run simulation for 10 us
quit -f