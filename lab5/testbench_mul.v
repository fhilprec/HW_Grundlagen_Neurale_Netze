`timescale 1ns / 1ps

module testbench_mul();
	reg[1*16+2*8-1:0] vectors[0:1000];
	reg[7:0] a;
	reg[7:0] b;
	wire[15:0] c;
	reg[15:0] y;

	reg clk, reset;
	reg [10:0] number;


	MUL dut(.a(a), .b(b), .y(c));
	always begin
		clk=1; #5; clk=0; #5;
	end

	// file reading
	integer file, stat, count;
	initial begin
		$readmemb("mul.tv", vectors);
		number = 0;
		reset = 1; #27; reset = 0;
	end

	always @(posedge clk) begin
		{a, b, y} = vectors[number];
	end

	always @(negedge clk) begin
		if(~reset) begin
			if (y !== c) begin
				$display("Error at vector %d", number);
				$display("a: %d, b: %d, expected %d, got %d", a, b, y, c);
				$finish;
			end

			number = number + 1;
			if (vectors[number] === 48'bx) begin
				$display("SUCCESS!");
				$finish;
			end
		end
	end


endmodule
