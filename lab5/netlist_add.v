/////////////////////////////////////////////////////////////
// Created by: Synopsys DC Ultra(TM) in wire load mode
// Version   : N-2017.09-SP3
// Date      : Thu Feb 27 11:06:47 2025
/////////////////////////////////////////////////////////////


module ADD ( a, b, y );
  input [15:0] a;
  input [15:0] b;
  output [15:0] y;
  wire   n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14, n15, n16,
         n17;

  AND2_X1 U2 ( .A1(a[0]), .A2(b[0]), .ZN(n12) );
  FA_X1 U3 ( .A(b[11]), .B(a[11]), .CI(n2), .CO(n13), .S(y[11]) );
  FA_X1 U4 ( .A(b[10]), .B(a[10]), .CI(n3), .CO(n2), .S(y[10]) );
  FA_X1 U5 ( .A(b[9]), .B(a[9]), .CI(n4), .CO(n3), .S(y[9]) );
  FA_X1 U6 ( .A(b[8]), .B(a[8]), .CI(n5), .CO(n4), .S(y[8]) );
  FA_X1 U7 ( .A(b[7]), .B(a[7]), .CI(n6), .CO(n5), .S(y[7]) );
  FA_X1 U8 ( .A(b[6]), .B(a[6]), .CI(n7), .CO(n6), .S(y[6]) );
  FA_X1 U9 ( .A(b[5]), .B(a[5]), .CI(n8), .CO(n7), .S(y[5]) );
  FA_X1 U10 ( .A(b[4]), .B(a[4]), .CI(n9), .CO(n8), .S(y[4]) );
  FA_X1 U11 ( .A(b[3]), .B(a[3]), .CI(n10), .CO(n9), .S(y[3]) );
  FA_X1 U12 ( .A(b[2]), .B(a[2]), .CI(n11), .CO(n10), .S(y[2]) );
  FA_X1 U13 ( .A(b[1]), .B(a[1]), .CI(n12), .CO(n11), .S(y[1]) );
  XOR2_X1 U14 ( .A(a[0]), .B(b[0]), .Z(y[0]) );
  FA_X1 U15 ( .A(b[12]), .B(a[12]), .CI(n13), .CO(n14), .S(y[12]) );
  FA_X1 U16 ( .A(b[13]), .B(a[13]), .CI(n14), .CO(n15), .S(y[13]) );
  FA_X1 U17 ( .A(b[14]), .B(a[14]), .CI(n15), .CO(n16), .S(y[14]) );
  XNOR2_X1 U18 ( .A(n16), .B(b[15]), .ZN(n17) );
  XNOR2_X1 U19 ( .A(n17), .B(a[15]), .ZN(y[15]) );
endmodule

