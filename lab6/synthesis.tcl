set target_library NangateOpenCellLibrary_typical.db ; # tell synopsis which library to use
set link_library NangateOpenCellLibrary_typical.db

read_file mul.v ; # read specified input file
link ; # link design with library
compile_ultra ; # compile design with extra effort
write -f verilog -hierarchy -output netlist_mul.v ; # save result as verilog netlist
write_sdf sdf_full.sdf ; # create SDF file
report_area > area-report.txt ; # create area report
quit