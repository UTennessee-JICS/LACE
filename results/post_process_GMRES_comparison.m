clc; clear all; close all;

log_test_solve_PariLU_MKL_FGMRES_Parcsrtrsv_30p30n_30p30nb_omp_
log_test_solve_MKLiLU_MKL_FGMRES_Parcsrtrsv_30p30n_30p30nb_omp_
log_test_solve_PariLU_MKL_FGMRES_MKLcsrtrsv_30p30n_30p30nb_omp_
log_test_solve_MKLiLU_MKL_FGMRES_MKLcsrtrsv_30p30n_30p30nb_omp_
log_test_solve_FGMRES_PariLU_MKLtrsv_30p30n_30p30nb_omp_
log_test_solve_FGMRES_PariLU_Parcsrtrsv_30p30n_30p30nb_omp_

figure(1)
semilogy(MKL_FGMRES_mkltrsv_search)
hold all
semilogy(MKL_FGMRES_partrsv_search)
semilogy(MKL_FGMRES_parilu_mkltrsv_search)
semilogy(MKL_FGMRES_parilu_partrsv_search)
semilogy(FGMRES_mkltrsv_search)
semilogy(FGMRES_search)
xlabel('Search Directions')
ylabel('L2 Residual Norm')
%str = sprintf('Convergence history for\n %s\n (P/F)GMRES(tol=%.3e) with PariLU(tol=%.3e) and ParCSRTRSV(tol=1.0)\n with %d OpenMP threads, %d GMRES search directions', matrix, gmres_param_rtol, user_precond_reduction, PariLU_v0_3_omp_num_threads, gmres_search_directions );
str = sprintf('Convergence history comparison for\n %s solved with GMRES variants (rtol=%.3e)', matrix, gmres_param_rtol );
title(str,'Interpreter','none')
%legend('MKL\_csriLU0+MKL\_FGMRES\_search','MKL\_csriLU0+PGMRES','PariLU+ParCSRTRSV+FGMRES','Location','southoutside','Orientation','horizontal')
legend('MKL\_csriLU0+MKL\_CSRTRSV+MKL\_FGMRES',...
    'MKL\_csriLU0+ParCSRTRSV+MKL\_FGMRES',...
    'PariLU+MKL\_CSRTRSV+MKL\_FGMRES',...
    'PariLU+ParCSRTRSV+MKL\_FGMRES',...
    'PariLU+MKL\_CSRTRSV+FGMRES',...
    'PariLU+ParCSRTRSV+FGMRES',...
    'Location','northeast')
x = size(MKL_FGMRES_mkltrsv_search,2);
y = MKL_FGMRES_mkltrsv_search(x);
label1 = sprintf('%d, %e \\rightarrow', x, y);
text(x, y, label1, 'HorizontalAlignment','right');
x = size(FGMRES_search,2);
y = FGMRES_search(x);
label1 = sprintf('%d, %e \\rightarrow', x, y);
text(x, y, label1, 'HorizontalAlignment','right');
str = sprintf('%s_compare_GMRES_convergence_hist', matrix);
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