% File testing/matrices/Trefethen_20.mtx basename Trefethen_20.mtx name Trefethen_20 
matrix = 'Trefethen_20'
% rhs vector name is ONES 
% Output directory is out_test
% Output file base name is out_test/Trefethen_20_solution.mtx
% Reading sparse matrix from file (testing/matrices/Trefethen_20.mtx): done. Converting to CSR:
% Detected symmetric case. done.
% creating a vector of 19 ones for the rhs.
gmres_param_tol_type = 1
gmres_param_rtol = 1.000000e-03
gmres_param_search_max = 2000
% data_gmres_basic begin
rnorm2 = 4.358899e+00; tol = 1.000000e-03; rtol = 4.358899e-03;
GMRES_basic_ortherr(1) = 9.4736842105263164e-01;
GMRES_basic_search(1) = 2.0373314267235942e+00;
GMRES_basic_ortherr(2) = 1.0468583068452428e+00;
GMRES_basic_search(2) = 1.1908642794890152e+00;
GMRES_basic_ortherr(3) = 1.2645689097110804e+00;
GMRES_basic_search(3) = 7.5895118288829899e-01;
GMRES_basic_ortherr(4) = 1.4523314479665650e+00;
GMRES_basic_search(4) = 5.3050378374967566e-01;
GMRES_basic_ortherr(5) = 1.5675061738593234e+00;
GMRES_basic_search(5) = 3.6900988444086835e-01;
GMRES_basic_ortherr(6) = 1.5830580305889534e+00;
GMRES_basic_search(6) = 2.6612585067378669e-01;
GMRES_basic_ortherr(7) = 1.6496313916856495e+00;
GMRES_basic_search(7) = 2.0367342817988895e-01;
GMRES_basic_ortherr(8) = 1.6651533362653324e+00;
GMRES_basic_search(8) = 1.5180773299993269e-01;
GMRES_basic_ortherr(9) = 1.6036274858762460e+00;
GMRES_basic_search(9) = 1.1332389088318004e-01;
GMRES_basic_ortherr(10) = 1.5370329846009021e+00;
GMRES_basic_search(10) = 7.3853972281428537e-02;
GMRES_basic_ortherr(11) = 1.6565809600576624e+00;
GMRES_basic_search(11) = 4.7342679084860693e-02;
GMRES_basic_ortherr(12) = 1.6725818728041575e+00;
GMRES_basic_search(12) = 3.2772539780009463e-02;
GMRES_basic_ortherr(13) = 1.6835326473216685e+00;
GMRES_basic_search(13) = 2.2878447520346238e-02;
GMRES_basic_ortherr(14) = 1.7028495430236530e+00;
GMRES_basic_search(14) = 1.3199661017844703e-02;
GMRES_basic_ortherr(15) = 1.7031078480095663e+00;
GMRES_basic_search(15) = 5.8818027034970152e-03;
GMRES_basic_ortherr(16) = 1.7345093008316352e+00;
GMRES_basic_search(16) = 2.4113001762682638e-03;
% Writing dense matrix to file (out_test/Trefethen_20_solution.mtx): done
% external check of rnorm2 = 2.4113001762685183e-03;

gmres_search_directions = 16;
gmres_solve_time = 1.917410e-02;
gmres_initial_residual = 4.358899e+00;
gmres_final_residual = 2.411300e-03;


% ################################################################################
% Matrix: testing/matrices/Trefethen_20.mtx
% 	19 -by- 19 with 147 non-zeros
% Solver: GMRES
% 	search directions: 16
% 	solve time [s]: 1.917410e-02
% 	initial residual: 4.358899e+00
% 	final residual: 2.411300e-03
% ################################################################################


% Done.
