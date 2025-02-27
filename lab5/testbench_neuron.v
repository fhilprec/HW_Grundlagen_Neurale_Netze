`timescale 1ns / 1ps
module testbench_neuron();
	reg[4*8+16-1:0] vectors[0:1000]; // 4 inputs (8 bits each) + expected output (16 bits)
	reg signed [7:0] inp0;
	reg signed [7:0] inp1;
	reg signed [7:0] inp2;
	reg signed [7:0] inp3;
	wire signed [15:0] out;
	reg signed [15:0] expected_out;
	reg clk, reset;
	reg [10:0] number;
	
	NEURON dut(.inp0(inp0), .inp1(inp1), .inp2(inp2), .inp3(inp3), .out(out));
	
	always begin
		clk=1; #5; clk=0; #5;
	end
	
	// file reading
	integer file, stat, count;
	initial begin
		$readmemb("neuron.tv", vectors);
		number = 0;
		reset = 1; #27; reset = 0;
	end
	
	always @(posedge clk) begin
		{inp0, inp1, inp2, inp3, expected_out} = vectors[number];
	end
	
	always @(negedge clk) begin
		if(~reset) begin
			if (expected_out !== out) begin
				$display("Error at vector %d", number);
				$display("inp0: %d, inp1: %d, inp2: %d, inp3: %d", inp0, inp1, inp2, inp3);
				$display("Expected: %d, Got: %d", expected_out, out);
				$finish;
			end
			number = number + 1;
			if (vectors[number] === {4*8+16{1'bx}}) begin
				$display("SUCCESS!");
				$finish;
			end
		end
	end
endmodule