module NEURON (
  input wire signed [7:0] inp0,
  input wire signed [7:0] inp1,
  input wire signed [7:0] inp2,
  input wire signed [7:0] inp3,
  output wire signed [15:0] out
);
  parameter [7:0] w0 = -8'sd5;  // s: signed, d: decimal
  parameter [7:0] w1 = 8'sd10;
  parameter [7:0] w2 = 8'sd27;
  parameter [7:0] w3 = -8'sd13;
  parameter [7:0] b = 8'sd7;
  
  wire signed [15:0] mul0_out, mul1_out, mul2_out, mul3_out;
  wire signed [15:0] add0_out, add1_out, add2_out, add3_out;
  
  // Multiply inputs with weights
  MUL mul0 (.a(inp0), .b(w0), .y(mul0_out));
  MUL mul1 (.a(inp1), .b(w1), .y(mul1_out));
  MUL mul2 (.a(inp2), .b(w2), .y(mul2_out));
  MUL mul3 (.a(inp3), .b(w3), .y(mul3_out));
  
  // Convert bias to 16 bits
  wire signed [15:0] bias = {{8{b[7]}}, b};
  
  // Add all products and bias
  ADD add0 (.a(mul0), .b(mul1), .y(add0_out));
  ADD add1 (.a(add0_out), .b(mul2), .y(add1_out));
  ADD add2 (.a(add1_out), .b(mul3), .y(add2_out));
  ADD add3 (.a(add2_out), .b(bias), .y(add3_out));
  
  // ReLU function: if negative, output 0
  assign out = (add3_out[15] == 1'b1) ? 16'sd0 : add3_out;
  
endmodule