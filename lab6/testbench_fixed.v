`timescale 1ns/1ps
module testbench();
reg [7:0] input0;
reg clk;
wire [15:0] output0;

MUL_FIXED dut(.a(input0), .y(output0));

initial forever begin
	clk = 0; #5;
	clk = 1; #5;
end

integer data_file;
initial begin
	input0 = 'd0;
	#10;

	data_file = $fopen("data.dat", "r");
	while (!$feof(data_file)) begin
		$fscanf(data_file, "%d\n", input0);
		#10;
	end
	$finish();
end
endmodule
