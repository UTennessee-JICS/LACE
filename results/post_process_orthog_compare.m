clc; clear all; close all;

%log_test_solve_FGMRES_orthog_ani5_crop
log_test_solve_FGMRES_orthog_30p30n
log_test_solve_FGMRES_orthog_30p30n_lt

figure(1)
semilogy(FGMRES_mkltrsv_search)
hold all
semilogy(FGMRES_lt_mkltrsv_search)
xlabel('Search Directions')
ylabel('L2 Residual Norm')
title('GMRES Convergence history')
legend('FGMRES+PariLU(1.e-15)+MKLCSRTRSV','FGMRES+PariLU(1.5e+5)+MKLCSRTRSV')

figure(2)
semilogy(FGMRES_mkltrsv_ortherr)
hold all
semilogy(FGMRES_lt_mkltrsv_ortherr)
xlabel('Search Directions')
ylabel('Infinity Norm of Orthogonality Error in Krylov Search Space')
title('History of Krylov Search Space Orthogonality Error')
legend('FGMRES+PariLU(1.e-15)+MKLCSRTRSV','FGMRES+PariLU(1.5e+5)+MKLCSRTRSV',...
    'Location','NorthWest')
