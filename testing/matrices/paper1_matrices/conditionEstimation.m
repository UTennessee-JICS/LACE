clc; clear all; close all;

matrices = string('ani5_crop.mtx'); 
matrices(2) = 'apache2_rcm.mtx'
matrices(3) = 'L2D_1024_5pt.mtx'
matrices(4) = 'L3D_64_27pt.mtx'
matrices(5) = 'ecology2_rcm.mtx'
matrices(6) = 'G3_circuit_rcm.mtx'
matrices(7) = 'offshore_rcm.mtx'
matrices(8) = 'parabolic_fem_rcm.mtx'
matrices(9) = 'thermal2_rcm.mtx'

for i=1:9
   display(char(matrices(i)))
   tmpfile = load(char(matrices(i)));
   tmpcsr = spconvert(tmpfile);
   cest(i) = condest(tmpcsr)
end

[matrices', cest']