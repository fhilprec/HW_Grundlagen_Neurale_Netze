`timescale 1ns/1ps
module testbench();
reg [7:0] input0;
reg [7:0] input1;
reg clk;
wire [15:0] output0;

MUL dut(.clk(clk), .a(input0), .b(input1), .y(output0));

initial forever begin
	clk = 0; #5;
	clk = 1; #5;
end

integer data_file;
initial begin
	input0 = 'd0;
	input1 = 'd0;
	#10;

	data_file = $fopen("data.dat", "r");
	while (!$feof(data_file)) begin
		$fscanf(data_file, "%d\n", input0);
		input1 = $random%128;
		#10;
	end
	$finish();
end
endmodule