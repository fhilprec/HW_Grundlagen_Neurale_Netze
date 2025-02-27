set target_library NangateOpenCellLibrary_typical.db ; # tell synopsis which library to use
set link_library NangateOpenCellLibrary_typical.db
read_file neuron.v ; # read specified input file
link ; # link design with library
compile_ultra ; # compile design with extra effort
write -f verilog -hierarchy -output netlist_neuron.v ; # save result as verilog netlist
quit