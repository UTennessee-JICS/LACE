clc; clear all; close all;

%log_test_solve_FGMRES_ani5_crop_omp8
%log_test_solve_FGMRES_30p30n_ones_omp8
log_test_solve_FGMRES_30p30n_rhs_omp8

figure(1)
semilogy(FGMRES_search)
xlabel('Iterations')
ylabel('L2 Residual Norm')
str = sprintf('Convergence history for\n %s\n FMGRES(tol=%.3e) with PariLU(tol=%.3e) and ParCSRTRSV(tol=1.0)\n with %d OpenMP threads, %d GMRES search directions', matrix, gmres_param_rtol, user_precond_reduction, PariLU_v0_3_omp_num_threads, gmres_search_directions );
title(str,'Interpreter','none')
str = sprintf('%s_convergence_hist', matrix);
print(str,'-dpng')

figure(2)
plot(ParCSRTRSV_L)
hold all
plot(ParCSRTRSV_U)
xlabel('Iterations')
ylabel('ParCSRTRSV Sweeps')
str = sprintf('ParCSRTRSV history for\n %s\n FMGRES(tol=%.3e) with PariLU(tol=%.3e) and ParCSRTRSV(tol=1.0)\n with %d OpenMP threads, %d GMRES search directions', matrix, gmres_param_rtol, user_precond_reduction, PariLU_v0_3_omp_num_threads, gmres_search_directions );
title(str,'Interpreter','none')
legend('L','U','Location','southoutside','Orientation','horizontal')
str = sprintf('%s_parcsrtrsv_hist', matrix);
print(str,'-dpng')