module MUL (
  input wire signed [7:0] a,
  output wire signed [16:0] y
);
  parameter WEIGHT = 0;  // parameters require a default value in verilog
  wire signed [7:0] b = WEIGHT;
  
  assign y = a * b;
endmodule