clc; clear all; close all;

%[A, rows, cols, entries] = mmread('testing/matrices/Trefethen_20.mtx');
[A, rows, cols, entries] = mmread('testing/matrices/paper1_matrices/ani5_crop.mtx');
%[A, rows, cols, entries] = mmread('testing/matrices/paper1_matrices/apache2_rcm.mtx');


atv = @(vec) A*vec;

b = ones(rows,1);
x0 = zeros(rows,1);
params = [ 1.0e-3, 2000, 0 ];

[x, reserror, ortherr_mgs, total_iters] = gmres_orthog_est(x0, b, atv, params);

rows
total_iters

r0 = b-A*x0;
r0n = norm(r0)

r = b-A*x;
rn = norm(r)

relres = rn/r0n
% x
% 
% setup.type = 'nofill';
% [L U] = ilu(A, setup);
% 
% v1 = 2.294157e-01 * ones(rows,1);
% 
% y = L\v1;
% z = U\y;
% 
% Ainv = inv(A);
% z1 = Ainv*r;
% 
% z-z1

%ortherr_mgs(total_iters)

figure(1)
semilogy(reserror)
xlabel('Search Directions')
ylabel('L2 Residual Norm')
title('GMRES Convergence history')

figure(2)
plot(ortherr_mgs)
xlabel('Search Directions')
ylabel('Infinity Norm of Orthogonality Error in Krylov Search Space')
title('History of Krylov Search Space Orthogonality Error')

condest_A = condest(A)