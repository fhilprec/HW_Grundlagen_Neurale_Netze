module MUL (
  input wire signed [7:0] a,
  input wire signed [7:0] b,
  output wire signed [15:0] y
);
  assign y = a * b;
endmodule