clc; clear all; close all;

log_FGMRES_30p30n
log_FGMRES_Householder_opt1_30p30n

figure(1)
semilogy(FGMRES_mkltrsv_search)
hold all
semilogy(FGMRES_Householders_mkltrsv_search)
xlabel('Search Directions')
ylabel('L2 Residual Norm')
title('FGMRES Convergence history for one psuedo-time step of the 30p30n benchmark')
legend('FGMRES(MGS)+PariLU(1.5e+5)+MKLCSRTRSV','FGMRES(HouseHolder)+PariLU(1.5e+5)+MKLCSRTRSV')

figure(2)
semilogy(FGMRES_mkltrsv_ortherr)
hold all
semilogy(FGMRES_Householders_mkltrsv_ortherr)
xlabel('Search Directions')
ylabel('Infinity Norm of Orthogonality Error in Krylov Search Space')
title({'History of Krylov Search Space Orthogonality Error in'; 'FGMRES for one psuedo-time step of the 30p30n benchmark'})
legend('FGMRES(MGS)+PariLU(1.5e+5)+MKLCSRTRSV','FGMRES(HouseHolder)+PariLU(1.5e+5)+MKLCSRTRSV',...    
    'Location','NorthWest')
