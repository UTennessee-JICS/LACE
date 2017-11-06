clc; clear all; close all;

tic
[Aani5, rows, cols, entries] = mmread('testing/matrices/paper1_matrices/ani5_crop.mtx');
toc(tic)
tic
condest(Aani5)
toc(tic)
tic
[fani5, smani5, lmani5, srani5, lrani5, lsani5, siani5] = plot_eigs(Aani5,1,'ani5\_crop');
toc(tic)
tic
invAani5 = inv(Aani5);
toc(tic)
tic
mmwrite('ani5_crop_inv.mtx',invAani5)
toc(tic)

tic
[A30p30n, rows, cols, entries] = mmread('testing/matrices/30p30n.mtx');
toc(tic)
tic
condest(A30p30n)
toc(tic)
tic
[f30p30n, sm30p30n, lm30p30n, sr30p30n, lr30p30n, ls30p30n, si30p30n] = plot_eigs(A30p30n,2,'30p30n');
toc(tic)
tic
invA30p30n = inv(A30p30n);
toc(tic)
tic
mmwrite('30p30n_inv.mtx',invA30p30n)
toc(tic)


% Elapsed time is 0.000209 seconds.
% 
% ans =
% 
%    1.4713e+04
% 
% Elapsed time is 0.000012 seconds.
% 
% f = 
% 
%   Figure (1) with properties:
% 
%       Number: 1
%         Name: ''
%        Color: [0.9400 0.9400 0.9400]
%     Position: [440 378 560 420]
%        Units: 'pixels'
% 
%   Show all properties
% 
% 
% ans = 
% 
%   Figure (1) with properties:
% 
%       Number: 1
%         Name: ''
%        Color: [0.9400 0.9400 0.9400]
%     Position: [440 378 560 420]
%        Units: 'pixels'
% 
%   Show all properties
% 
% Elapsed time is 0.000030 seconds.
% Elapsed time is 0.000020 seconds.
% Elapsed time is 0.000635 seconds.
% Elapsed time is 0.000021 seconds.
% 
% ans =
% 
%    3.6963e+15
% 
% Elapsed time is 0.000214 seconds.
% 
% f = 
% 
%   Figure (2) with properties:
% 
%       Number: 2
%         Name: ''
%        Color: [0.9400 0.9400 0.9400]
%     Position: [440 378 560 420]
%        Units: 'pixels'
% 
%   Show all properties
% 
% 
% ans = 
% 
%   Figure (2) with properties:
% 
%       Number: 2
%         Name: ''
%        Color: [0.9400 0.9400 0.9400]
%     Position: [440 378 560 420]
%        Units: 'pixels'
% 
%   Show all properties
% 
% Elapsed time is 0.000010 seconds.
% 
% Error using inv
% Requested 211685x211685 (25.3GB) array exceeds maximum array
% size preference. Creation of arrays greater than this limit
% may take a long time and cause MATLAB to become
% unresponsive. See array size limit or preference panel for
% more information.
% 
% Error in cond_eigs (line 29)
% invA30p30n = inv(A30p30n);
 
