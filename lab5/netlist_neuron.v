/////////////////////////////////////////////////////////////
// Created by: Synopsys DC Ultra(TM) in wire load mode
// Version   : N-2017.09-SP3
// Date      : Thu Feb 27 13:32:16 2025
/////////////////////////////////////////////////////////////


module NEURON ( inp0, inp1, inp2, inp3, out );
  input [7:0] inp0;
  input [7:0] inp1;
  input [7:0] inp2;
  input [7:0] inp3;
  output [15:0] out;
  wire   n2;
  tri   \*Logic1* ;
  tri   [7:0] inp0;
  tri   [7:0] inp1;
  tri   [7:0] inp2;
  tri   [7:0] inp3;
  tri   [15:0] mul0_out;
  tri   [15:0] mul1_out;
  tri   [15:0] mul2_out;
  tri   [15:0] mul3_out;
  tri   [15:0] add0_out;
  tri   [15:0] add1_out;
  tri   [15:0] add2_out;
  tri   [15:0] add3_out;
  tri   n3;
  assign out[15] = 1'b0;

  MUL mul0_inst ( .a(inp0), .b({1'b1, 1'b1, 1'b1, 1'b1, 1'b1, 1'b0, 1'b1, 1'b1}), .y(mul0_out) );
  MUL mul1_inst ( .a(inp1), .b({1'b0, 1'b0, 1'b0, 1'b0, 1'b1, 1'b0, 1'b1, 1'b0}), .y(mul1_out) );
  MUL mul2_inst ( .a(inp2), .b({1'b0, 1'b0, 1'b0, 1'b1, 1'b1, 1'b0, 1'b1, 1'b1}), .y(mul2_out) );
  MUL mul3_inst ( .a(inp3), .b({1'b1, 1'b1, 1'b1, 1'b1, 1'b0, 1'b0, 1'b1, 1'b1}), .y(mul3_out) );
  ADD add0_inst ( .a(mul0_out), .b(mul1_out), .y(add0_out) );
  ADD add1_inst ( .a(add0_out), .b(mul2_out), .y(add1_out) );
  ADD add2_inst ( .a(add1_out), .b(mul3_out), .y(add2_out) );
  ADD add3_inst ( .a(add2_out), .b({1'b0, 1'b0, 1'b0, 1'b0, 1'b0, 1'b0, 1'b0, 
        1'b0, 1'b0, 1'b0, 1'b0, 1'b0, 1'b0, 1'b1, 1'b1, 1'b1}), .y(add3_out)
         );
  AND2_X1 U5 ( .A1(add3_out[14]), .A2(n2), .ZN(out[14]) );
  AND2_X1 U6 ( .A1(add3_out[13]), .A2(n2), .ZN(out[13]) );
  AND2_X1 U7 ( .A1(add3_out[12]), .A2(n2), .ZN(out[12]) );
  AND2_X1 U8 ( .A1(add3_out[11]), .A2(n2), .ZN(out[11]) );
  AND2_X1 U9 ( .A1(add3_out[10]), .A2(n2), .ZN(out[10]) );
  AND2_X1 U10 ( .A1(add3_out[9]), .A2(n2), .ZN(out[9]) );
  AND2_X1 U11 ( .A1(add3_out[8]), .A2(n2), .ZN(out[8]) );
  AND2_X1 U12 ( .A1(add3_out[7]), .A2(n2), .ZN(out[7]) );
  AND2_X1 U13 ( .A1(add3_out[6]), .A2(n2), .ZN(out[6]) );
  AND2_X1 U14 ( .A1(add3_out[5]), .A2(n2), .ZN(out[5]) );
  AND2_X1 U15 ( .A1(add3_out[4]), .A2(n2), .ZN(out[4]) );
  AND2_X1 U16 ( .A1(add3_out[3]), .A2(n2), .ZN(out[3]) );
  AND2_X1 U17 ( .A1(add3_out[2]), .A2(n2), .ZN(out[2]) );
  AND2_X1 U18 ( .A1(add3_out[1]), .A2(n2), .ZN(out[1]) );
  AND2_X1 U19 ( .A1(add3_out[0]), .A2(n2), .ZN(out[0]) );
  INV_X1 U4 ( .A(add3_out[15]), .ZN(n2) );
endmodule

