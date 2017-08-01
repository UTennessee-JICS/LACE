clc; clear all; close all;

%log_test_solve_FGMRES_ani5_crop_omp8
%log_test_solve_FGMRES_30p30n_ones_omp8
log_test_solve_FGMRES_30p30n_rhs_omp8
%log_test_solve_GMRES_precond_30p30n_rhs_omp8
log_test_solve_GMRES_precond_MKL_30p30n_rhs_omp8
% ./test_solve_pariLU0_MKL_FGMRES testing/matrices/30p30n.mtx testing/matrices/30p30n-b.mtx out_test NONE 0 3.55064e-05 2000 2000 1 > results/log_test_solve_MKL_FGMRES_MKL_30p30n_rhs_omp8_2.m
log_test_solve_MKL_FGMRES_MKL_30p30n_rhs_omp8_2

figure(1)
%semilogy(GMRES_search)
semilogy(MKL_FGMRES_search)
hold all
semilogy(PGMRES_search)
semilogy(FGMRES_search)
xlabel('Search Directions')
ylabel('L2 Residual Norm')
%str = sprintf('Convergence history for\n %s\n (P/F)GMRES(tol=%.3e) with PariLU(tol=%.3e) and ParCSRTRSV(tol=1.0)\n with %d OpenMP threads, %d GMRES search directions', matrix, gmres_param_rtol, user_precond_reduction, PariLU_v0_3_omp_num_threads, gmres_search_directions );
str = sprintf('Convergence history comparison for\n %s solved with GMRES variants (rtol=%.3e)', matrix, gmres_param_rtol );
title(str,'Interpreter','none')
%legend('MKL\_csriLU0+MKL\_FGMRES\_search','MKL\_csriLU0+PGMRES','PariLU+ParCSRTRSV+FGMRES','Location','southoutside','Orientation','horizontal')
legend('MKL\_csriLU0+MKL\_CSRTRSV+MKL\_FGMRES\_search','MKL\_csriLU0+MKL\_CSRTRSV+PGMRES','PariLU+ParCSRTRSV+FGMRES','Location','northeast')
x = size(MKL_FGMRES_search,2);
y = MKL_FGMRES_search(x);
label1 = sprintf('%d, %e \\rightarrow', x, y);
text(x, y, label1, 'HorizontalAlignment','right');
x = size(FGMRES_search,2);
y = FGMRES_search(x);
label1 = sprintf('%d, %e \\rightarrow', x, y);
text(x, y, label1, 'HorizontalAlignment','right');
str = sprintf('%s_MKL_PGMRES_FGMRES_convergence_hist', matrix);
print(str,'-dpng')

% figure(2)
% plot(ParCSRTRSV_L)
% hold all
% plot(ParCSRTRSV_U)
% xlabel('Search Directions')
% ylabel('ParCSRTRSV Sweeps')
% str = sprintf('ParCSRTRSV history for\n %s\n FGMRES(tol=%.3e) with PariLU(tol=%.3e) and ParCSRTRSV(tol=1.0)\n with %d OpenMP threads, %d GMRES search directions', matrix, gmres_param_rtol, user_precond_reduction, PariLU_v0_3_omp_num_threads, gmres_search_directions );
% title(str,'Interpreter','none')
% legend('L','U','Location','southoutside','Orientation','horizontal')
% str = sprintf('%s_precond_parcsrtrsv_hist', matrix);
% print(str,'-dpng')