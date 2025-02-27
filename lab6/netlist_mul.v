/////////////////////////////////////////////////////////////
// Created by: Synopsys DC Ultra(TM) in wire load mode
// Version   : N-2017.09-SP3
// Date      : Thu Feb 27 13:50:00 2025
/////////////////////////////////////////////////////////////


module MUL ( a, b, y );
  input [7:0] a;
  input [7:0] b;
  output [15:0] y;
  wire   n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14, n15, n16,
         n17, n18, n19, n20, n21, n22, n23, n24, n25, n26, n27, n28, n29, n30,
         n31, n32, n33, n34, n35, n36, n37, n38, n39, n40, n41, n42, n43, n44,
         n45, n46, n47, n48, n49, n50, n51, n52, n53, n54, n55, n56, n57, n58,
         n59, n60, n61, n62, n63, n64, n65, n66, n67, n68, n69, n70, n71, n72,
         n73, n74, n75, n76, n77, n78, n79, n80, n81, n82, n83, n84, n85, n86,
         n87, n88, n89, n90, n91, n92, n93, n94, n95, n96, n97, n98, n99, n100,
         n101, n102, n103, n104, n105, n106, n107, n108, n109, n110, n111,
         n112, n113, n114, n115, n116, n117, n118, n119, n120, n121, n122,
         n123, n124, n125, n126, n127, n128, n129, n130, n131, n132, n133,
         n134, n135, n136, n137, n138, n139, n140, n141, n142, n143, n144,
         n145, n146, n147, n148, n149, n150, n151, n152, n153, n154, n155,
         n156, n157, n158, n159, n160, n161, n162, n163, n164, n165, n166,
         n167, n168, n169, n170, n171, n172, n173, n174, n175, n176;

  AOI221_X2 U1 ( .B1(a[4]), .B2(a[5]), .C1(n4), .C2(n80), .A(n102), .ZN(n79)
         );
  INV_X1 U2 ( .A(a[0]), .ZN(n47) );
  INV_X1 U3 ( .A(b[0]), .ZN(n105) );
  NOR2_X1 U4 ( .A1(n47), .A2(n105), .ZN(y[0]) );
  NOR2_X1 U5 ( .A1(a[1]), .A2(n47), .ZN(n117) );
  INV_X1 U6 ( .A(a[1]), .ZN(n10) );
  NOR2_X1 U7 ( .A1(n47), .A2(n10), .ZN(n49) );
  INV_X1 U8 ( .A(n49), .ZN(n114) );
  NOR2_X1 U9 ( .A1(b[1]), .A2(n114), .ZN(n1) );
  AOI21_X1 U10 ( .B1(n117), .B2(b[1]), .A(n1), .ZN(n2) );
  AOI211_X1 U11 ( .C1(a[0]), .C2(b[1]), .A(b[0]), .B(n10), .ZN(n118) );
  AOI221_X1 U12 ( .B1(y[0]), .B2(n2), .C1(n10), .C2(n2), .A(n118), .ZN(y[1])
         );
  INV_X1 U13 ( .A(a[5]), .ZN(n80) );
  INV_X1 U14 ( .A(a[6]), .ZN(n3) );
  OAI22_X1 U15 ( .A1(n80), .A2(n3), .B1(a[6]), .B2(a[5]), .ZN(n33) );
  INV_X1 U16 ( .A(n33), .ZN(n169) );
  INV_X1 U17 ( .A(a[7]), .ZN(n157) );
  INV_X1 U18 ( .A(b[6]), .ZN(n12) );
  OAI22_X1 U19 ( .A1(n157), .A2(b[6]), .B1(n12), .B2(a[7]), .ZN(n159) );
  OAI221_X1 U20 ( .B1(n3), .B2(n157), .C1(a[6]), .C2(a[7]), .A(n33), .ZN(n31)
         );
  INV_X1 U21 ( .A(n31), .ZN(n168) );
  INV_X1 U22 ( .A(b[5]), .ZN(n34) );
  AOI22_X1 U23 ( .A1(a[7]), .A2(b[5]), .B1(n34), .B2(n157), .ZN(n8) );
  AOI22_X1 U24 ( .A1(n169), .A2(n159), .B1(n168), .B2(n8), .ZN(n162) );
  INV_X1 U25 ( .A(a[3]), .ZN(n111) );
  INV_X1 U26 ( .A(a[4]), .ZN(n4) );
  OAI22_X1 U27 ( .A1(n111), .A2(n4), .B1(a[4]), .B2(a[3]), .ZN(n83) );
  INV_X1 U28 ( .A(n83), .ZN(n102) );
  INV_X1 U29 ( .A(b[7]), .ZN(n158) );
  OAI22_X1 U30 ( .A1(n80), .A2(b[7]), .B1(n158), .B2(a[5]), .ZN(n5) );
  AOI22_X1 U31 ( .A1(a[5]), .A2(b[6]), .B1(n12), .B2(n80), .ZN(n9) );
  AOI22_X1 U32 ( .A1(n102), .A2(n5), .B1(n79), .B2(n9), .ZN(n161) );
  OAI21_X1 U33 ( .B1(n102), .B2(n79), .A(n5), .ZN(n6) );
  INV_X1 U34 ( .A(n6), .ZN(n160) );
  INV_X1 U35 ( .A(n7), .ZN(n166) );
  INV_X1 U36 ( .A(n161), .ZN(n126) );
  INV_X1 U37 ( .A(b[4]), .ZN(n50) );
  AOI22_X1 U38 ( .A1(a[7]), .A2(b[4]), .B1(n50), .B2(n157), .ZN(n20) );
  AOI22_X1 U39 ( .A1(n169), .A2(n8), .B1(n168), .B2(n20), .ZN(n125) );
  AOI22_X1 U40 ( .A1(a[5]), .A2(b[5]), .B1(n34), .B2(n80), .ZN(n18) );
  AOI22_X1 U41 ( .A1(n102), .A2(n9), .B1(n79), .B2(n18), .ZN(n17) );
  INV_X1 U42 ( .A(a[2]), .ZN(n11) );
  XNOR2_X1 U43 ( .A(n10), .B(n11), .ZN(n112) );
  AOI22_X1 U44 ( .A1(a[3]), .A2(n158), .B1(b[7]), .B2(n111), .ZN(n13) );
  OAI221_X1 U45 ( .B1(n11), .B2(n111), .C1(a[2]), .C2(a[3]), .A(n112), .ZN(
        n113) );
  AOI22_X1 U46 ( .A1(a[3]), .A2(n12), .B1(b[6]), .B2(n111), .ZN(n36) );
  OAI22_X1 U47 ( .A1(n112), .A2(n13), .B1(n113), .B2(n36), .ZN(n39) );
  INV_X1 U48 ( .A(n39), .ZN(n16) );
  AOI21_X1 U49 ( .B1(n112), .B2(n113), .A(n13), .ZN(n15) );
  INV_X1 U50 ( .A(n14), .ZN(n165) );
  FA_X1 U51 ( .A(n17), .B(n16), .CI(n15), .CO(n124), .S(n24) );
  INV_X1 U52 ( .A(b[3]), .ZN(n64) );
  AOI22_X1 U53 ( .A1(b[3]), .A2(a[7]), .B1(n157), .B2(n64), .ZN(n19) );
  INV_X1 U54 ( .A(b[2]), .ZN(n91) );
  OAI22_X1 U55 ( .A1(n91), .A2(a[7]), .B1(n157), .B2(b[2]), .ZN(n29) );
  AOI22_X1 U56 ( .A1(n169), .A2(n19), .B1(n168), .B2(n29), .ZN(n38) );
  AOI22_X1 U57 ( .A1(a[5]), .A2(b[4]), .B1(n50), .B2(n80), .ZN(n26) );
  AOI22_X1 U58 ( .A1(n102), .A2(n18), .B1(n79), .B2(n26), .ZN(n37) );
  AOI22_X1 U59 ( .A1(n169), .A2(n20), .B1(n168), .B2(n19), .ZN(n22) );
  INV_X1 U60 ( .A(n21), .ZN(n130) );
  FA_X1 U61 ( .A(n24), .B(n23), .CI(n22), .CO(n21), .S(n25) );
  INV_X1 U62 ( .A(n25), .ZN(n133) );
  AOI22_X1 U63 ( .A1(b[3]), .A2(a[5]), .B1(n80), .B2(n64), .ZN(n46) );
  AOI22_X1 U64 ( .A1(n102), .A2(n26), .B1(n79), .B2(n46), .ZN(n45) );
  OAI22_X1 U65 ( .A1(n158), .A2(n117), .B1(a[1]), .B2(b[7]), .ZN(n27) );
  INV_X1 U66 ( .A(n27), .ZN(n44) );
  AOI221_X1 U67 ( .B1(b[0]), .B2(n169), .C1(a[6]), .C2(n33), .A(n157), .ZN(n62) );
  INV_X1 U68 ( .A(b[1]), .ZN(n100) );
  AOI22_X1 U69 ( .A1(b[1]), .A2(n157), .B1(a[7]), .B2(n100), .ZN(n30) );
  OAI221_X1 U70 ( .B1(b[0]), .B2(a[7]), .C1(n105), .C2(n157), .A(n168), .ZN(
        n28) );
  OAI21_X1 U71 ( .B1(n33), .B2(n30), .A(n28), .ZN(n61) );
  NAND2_X1 U72 ( .A1(n62), .A2(n61), .ZN(n60) );
  INV_X1 U73 ( .A(n29), .ZN(n32) );
  OAI22_X1 U74 ( .A1(n33), .A2(n32), .B1(n31), .B2(n30), .ZN(n54) );
  OAI22_X1 U75 ( .A1(n111), .A2(b[5]), .B1(n34), .B2(a[3]), .ZN(n51) );
  INV_X1 U76 ( .A(n51), .ZN(n35) );
  OAI22_X1 U77 ( .A1(n112), .A2(n36), .B1(n113), .B2(n35), .ZN(n53) );
  NOR2_X1 U78 ( .A1(n54), .A2(n53), .ZN(n52) );
  FA_X1 U79 ( .A(n39), .B(n38), .CI(n37), .CO(n23), .S(n41) );
  INV_X1 U80 ( .A(n40), .ZN(n132) );
  FA_X1 U81 ( .A(n42), .B(n52), .CI(n41), .CO(n40), .S(n43) );
  INV_X1 U82 ( .A(n43), .ZN(n136) );
  FA_X1 U83 ( .A(n45), .B(n44), .CI(n60), .CO(n42), .S(n58) );
  AOI22_X1 U84 ( .A1(b[2]), .A2(a[5]), .B1(n80), .B2(n91), .ZN(n74) );
  AOI22_X1 U85 ( .A1(n102), .A2(n46), .B1(n79), .B2(n74), .ZN(n68) );
  NAND2_X1 U86 ( .A1(a[1]), .A2(n47), .ZN(n115) );
  NOR2_X1 U87 ( .A1(b[6]), .A2(n115), .ZN(n48) );
  AOI221_X1 U88 ( .B1(n49), .B2(n158), .C1(n117), .C2(b[7]), .A(n48), .ZN(n67)
         );
  INV_X1 U89 ( .A(n112), .ZN(n119) );
  INV_X1 U90 ( .A(n113), .ZN(n107) );
  AOI22_X1 U91 ( .A1(a[3]), .A2(b[4]), .B1(n50), .B2(n111), .ZN(n65) );
  AOI22_X1 U92 ( .A1(n119), .A2(n51), .B1(n107), .B2(n65), .ZN(n66) );
  AOI21_X1 U93 ( .B1(n54), .B2(n53), .A(n52), .ZN(n56) );
  INV_X1 U94 ( .A(n55), .ZN(n135) );
  FA_X1 U95 ( .A(n58), .B(n57), .CI(n56), .CO(n55), .S(n59) );
  INV_X1 U96 ( .A(n59), .ZN(n139) );
  OAI21_X1 U97 ( .B1(n62), .B2(n61), .A(n60), .ZN(n72) );
  OAI22_X1 U98 ( .A1(b[5]), .A2(n115), .B1(b[6]), .B2(n114), .ZN(n63) );
  AOI21_X1 U99 ( .B1(n117), .B2(b[6]), .A(n63), .ZN(n77) );
  AOI22_X1 U100 ( .A1(a[3]), .A2(b[3]), .B1(n64), .B2(n111), .ZN(n92) );
  AOI22_X1 U101 ( .A1(n119), .A2(n65), .B1(n107), .B2(n92), .ZN(n76) );
  NAND2_X1 U102 ( .A1(b[0]), .A2(n169), .ZN(n75) );
  FA_X1 U103 ( .A(n68), .B(n67), .CI(n66), .CO(n57), .S(n70) );
  INV_X1 U104 ( .A(n69), .ZN(n138) );
  FA_X1 U105 ( .A(n72), .B(n71), .CI(n70), .CO(n69), .S(n73) );
  INV_X1 U106 ( .A(n73), .ZN(n142) );
  OAI22_X1 U107 ( .A1(n100), .A2(a[5]), .B1(n80), .B2(b[1]), .ZN(n78) );
  AOI22_X1 U108 ( .A1(n102), .A2(n74), .B1(n79), .B2(n78), .ZN(n86) );
  FA_X1 U109 ( .A(n77), .B(n76), .CI(n75), .CO(n71), .S(n85) );
  AOI221_X1 U110 ( .B1(b[0]), .B2(n102), .C1(a[4]), .C2(n83), .A(n80), .ZN(n90) );
  INV_X1 U111 ( .A(n78), .ZN(n82) );
  OAI221_X1 U112 ( .B1(b[0]), .B2(a[5]), .C1(n105), .C2(n80), .A(n79), .ZN(n81) );
  OAI21_X1 U113 ( .B1(n83), .B2(n82), .A(n81), .ZN(n89) );
  NAND2_X1 U114 ( .A1(n90), .A2(n89), .ZN(n88) );
  INV_X1 U115 ( .A(n84), .ZN(n141) );
  FA_X1 U116 ( .A(n86), .B(n85), .CI(n88), .CO(n84), .S(n87) );
  INV_X1 U117 ( .A(n87), .ZN(n145) );
  OAI21_X1 U118 ( .B1(n90), .B2(n89), .A(n88), .ZN(n97) );
  AOI22_X1 U119 ( .A1(b[2]), .A2(a[3]), .B1(n111), .B2(n91), .ZN(n101) );
  AOI22_X1 U120 ( .A1(n119), .A2(n92), .B1(n107), .B2(n101), .ZN(n96) );
  OAI22_X1 U121 ( .A1(b[5]), .A2(n114), .B1(b[4]), .B2(n115), .ZN(n93) );
  AOI21_X1 U122 ( .B1(n117), .B2(b[5]), .A(n93), .ZN(n95) );
  INV_X1 U123 ( .A(n94), .ZN(n144) );
  FA_X1 U124 ( .A(n97), .B(n96), .CI(n95), .CO(n94), .S(n98) );
  INV_X1 U125 ( .A(n98), .ZN(n148) );
  OAI22_X1 U126 ( .A1(b[3]), .A2(n115), .B1(b[4]), .B2(n114), .ZN(n99) );
  AOI21_X1 U127 ( .B1(n117), .B2(b[4]), .A(n99), .ZN(n122) );
  AOI22_X1 U128 ( .A1(b[1]), .A2(a[3]), .B1(n111), .B2(n100), .ZN(n108) );
  AOI22_X1 U129 ( .A1(n119), .A2(n101), .B1(n107), .B2(n108), .ZN(n121) );
  NAND2_X1 U130 ( .A1(b[0]), .A2(n102), .ZN(n120) );
  INV_X1 U131 ( .A(n103), .ZN(n147) );
  OAI22_X1 U132 ( .A1(b[2]), .A2(n115), .B1(b[3]), .B2(n114), .ZN(n104) );
  AOI21_X1 U133 ( .B1(n117), .B2(b[3]), .A(n104), .ZN(n110) );
  AOI22_X1 U134 ( .A1(b[0]), .A2(a[3]), .B1(n111), .B2(n105), .ZN(n106) );
  AOI22_X1 U135 ( .A1(n119), .A2(n108), .B1(n107), .B2(n106), .ZN(n109) );
  NOR2_X1 U136 ( .A1(n109), .A2(n110), .ZN(n150) );
  AOI21_X1 U137 ( .B1(n110), .B2(n109), .A(n150), .ZN(n153) );
  AOI221_X1 U138 ( .B1(b[0]), .B2(n113), .C1(n112), .C2(n113), .A(n111), .ZN(
        n152) );
  OAI22_X1 U139 ( .A1(b[1]), .A2(n115), .B1(b[2]), .B2(n114), .ZN(n116) );
  AOI21_X1 U140 ( .B1(n117), .B2(b[2]), .A(n116), .ZN(n155) );
  AOI21_X1 U141 ( .B1(b[0]), .B2(n119), .A(n118), .ZN(n156) );
  NOR2_X1 U142 ( .A1(n155), .A2(n156), .ZN(n154) );
  FA_X1 U143 ( .A(n122), .B(n121), .CI(n120), .CO(n103), .S(n123) );
  INV_X1 U144 ( .A(n123), .ZN(n149) );
  FA_X1 U145 ( .A(n126), .B(n125), .CI(n124), .CO(n14), .S(n127) );
  INV_X1 U146 ( .A(n127), .ZN(n128) );
  FA_X1 U147 ( .A(n130), .B(n129), .CI(n128), .CO(n164), .S(y[11]) );
  FA_X1 U148 ( .A(n133), .B(n132), .CI(n131), .CO(n129), .S(y[10]) );
  FA_X1 U149 ( .A(n136), .B(n135), .CI(n134), .CO(n131), .S(y[9]) );
  FA_X1 U150 ( .A(n139), .B(n138), .CI(n137), .CO(n134), .S(y[8]) );
  FA_X1 U151 ( .A(n142), .B(n141), .CI(n140), .CO(n137), .S(y[7]) );
  FA_X1 U152 ( .A(n145), .B(n144), .CI(n143), .CO(n140), .S(y[6]) );
  FA_X1 U153 ( .A(n148), .B(n147), .CI(n146), .CO(n143), .S(y[5]) );
  FA_X1 U154 ( .A(n151), .B(n150), .CI(n149), .CO(n146), .S(y[4]) );
  FA_X1 U155 ( .A(n153), .B(n152), .CI(n154), .CO(n151), .S(y[3]) );
  AOI21_X1 U156 ( .B1(n156), .B2(n155), .A(n154), .ZN(y[2]) );
  AOI22_X1 U157 ( .A1(a[7]), .A2(b[7]), .B1(n158), .B2(n157), .ZN(n167) );
  AOI22_X1 U158 ( .A1(n169), .A2(n167), .B1(n168), .B2(n159), .ZN(n172) );
  FA_X1 U159 ( .A(n162), .B(n161), .CI(n160), .CO(n163), .S(n7) );
  INV_X1 U160 ( .A(n163), .ZN(n171) );
  FA_X1 U161 ( .A(n166), .B(n165), .CI(n164), .CO(n170), .S(y[12]) );
  OAI21_X1 U162 ( .B1(n169), .B2(n168), .A(n167), .ZN(n175) );
  INV_X1 U163 ( .A(n172), .ZN(n174) );
  FA_X1 U164 ( .A(n172), .B(n171), .CI(n170), .CO(n173), .S(y[13]) );
  FA_X1 U165 ( .A(n175), .B(n174), .CI(n173), .CO(n176), .S(y[14]) );
  INV_X1 U166 ( .A(n176), .ZN(y[15]) );
endmodule

