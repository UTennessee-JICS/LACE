/*
 * make -f makefile_beacon test_iLU -B
 */

#include <stdlib.h>
#include <stdio.h>
#include <libgen.h>
#include <string>
#include <vector>
#include <omp.h>

#include "gpu_tests.h"

#ifndef DATA_TYPE_H
  # define DATA_TYPE_H
  typedef double dataType;
#endif

#include "mmio.h"
#include "sparse_types.h"

int
main(int argc, char * argv[])
{
  // begin with a square matrix A
  // char * sparse_filename = NULL;
  char * sparse_basename = NULL;
  char * output_dir      = NULL;
  char sparse_name[256];
  char output_basename[256];
  char output_x[256];

  char default_solver[] = "FGMRES";
  char * solver_name    = NULL;
  char default_matrix[] = "matrices/Trefethen_20.mtx";
  char * matrix_name    = NULL;

  char default_L[]   = "matrices/Trefethen_20_L.mtx";
  char * L_name      = NULL;
  char default_U[]   = "matrices/Trefethen_20_U.mtx";
  char * U_name      = NULL;
  char default_rhs[] = "ONES";
  char * rhs_name    = NULL;

  char default_output_dir[] = "out_";

  data_d_gmres_param gmres_param;
  data_d_gmres_log gmres_log;

  // set default GMRES prameters
  gmres_param.tol_type = 0;
  gmres_param.rtol     = 1.0e-6;
  gmres_param.user_csrtrsv_choice = 1;
  gmres_param.monitorOrthog       = 0;

  if (argc < 7) {
    printf("Usage %s --solver <name> --matrix <name> --L <name> --U <name> --RHS <name> --outDir <name> ", argv[0]);
    printf("--tolType <0/1> --tol <#> --searchMax <#> --csrtrsvType <0/1> --monitorOrthog <0/1>\n");
    return 1;
  }

  if (argc > 1) {
    int count = 1;
    while (count < argc) {
      if ( (strcmp(argv[count], "--solver") == 0) &&
        count + 1 < argc)
      {
        matrix_name = argv[count + 1];
        count       = count + 2;
      } else if ( (strcmp(argv[count], "--matrix") == 0) &&
        count + 1 < argc)
      {
        matrix_name = argv[count + 1];
        count       = count + 2;
      } else if ( (strcmp(argv[count], "--L") == 0) &&
        count + 1 < argc)
      {
        L_name = argv[count + 1];
        count  = count + 2;
      } else if ( (strcmp(argv[count], "--U") == 0) &&
        count + 1 < argc)
      {
        U_name = argv[count + 1];
        count  = count + 2;
      } else if ( (strcmp(argv[count], "--RHS") == 0) &&
        count + 1 < argc)
      {
        rhs_name = argv[count + 1];
        count    = count + 2;
      } else if ( (strcmp(argv[count], "--outDir") == 0) &&
        count + 1 < argc)
      {
        output_dir = argv[count + 1];
        count      = count + 2;
      } else if ( (strcmp(argv[count], "--tolType") == 0) &&
        count + 1 < argc)
      {
        gmres_param.tol_type = atoi(argv[count + 1]);
        count = count + 2;
      } else if ( (strcmp(argv[count], "--tol") == 0) &&
        count + 1 < argc)
      {
        gmres_param.rtol = atof(argv[count + 1]);
        count = count + 2;
      } else if ( (strcmp(argv[count], "--searchMax") == 0) &&
        count + 1 < argc)
      {
        gmres_param.search_max = atoi(argv[count + 1]);
        count = count + 2;
      } else if ( (strcmp(argv[count], "--csrtrsvType") == 0) &&
        count + 1 < argc)
      {
        gmres_param.user_csrtrsv_choice = atoi(argv[count + 1]);
        count = count + 2;
      } else if ( (strcmp(argv[count], "--monitorOrthog") == 0) &&
        count + 1 < argc)
      {
        gmres_param.monitorOrthog = atoi(argv[count + 1]);
        count = count + 2;
      } else {
        count++;
      }
    }
  }

  if (solver_name == NULL) {
    solver_name = default_solver;
  }
  if (matrix_name == NULL) {
    matrix_name = default_matrix;
  }
  if (L_name == NULL) {
    L_name = default_L;
  }
  if (U_name == NULL) {
    U_name = default_U;
  }
  if (rhs_name == NULL) {
    rhs_name = default_rhs;
  }
  if (output_dir == NULL) {
    output_dir = default_output_dir;
  }


  dataType zero   = 0.0;
  dataType one    = 1.0;
  dataType negone = -1.0;
  dataType rnorm2 = 0.0;

  data_d_matrix Asparse = { Magma_CSR };
  CHECK(data_z_csr_mtx(&Asparse, matrix_name) );

  data_d_matrix L = { Magma_CSR };
  CHECK(data_z_csr_mtx(&L, L_name) );
  
  data_d_matrix U = { Magma_CSR };
  CHECK(data_z_csr_mtx(&U, U_name) );

  //data_zprint_csr( L );
  //data_zprint_csr( U );

  data_d_matrix rhs_vector = { Magma_DENSE };
  rhs_vector.major = MagmaRowMajor;
  if (strcmp(rhs_name, "ONES") == 0) {
    printf("%% creating a vector of %d ones for the rhs.\n", Asparse.num_rows);
    CHECK(data_zvinit(&rhs_vector, Asparse.num_rows, 1, one) );
  } else {
    CHECK(data_z_dense_mtx(&rhs_vector, rhs_vector.major, rhs_name) );
  }

  data_d_matrix x = { Magma_DENSE };
  CHECK(data_zvinit(&x, Asparse.num_rows, 1, zero) );

  if (strcmp(solver_name, "FGMRES") == 0) {
    data_fgmres(&Asparse, &rhs_vector, &x, &L, &U, &gmres_param, &gmres_log);
  } else if (strcmp(solver_name, "FGMRESH") == 0) {
    data_fgmres_householder(&Asparse, &rhs_vector, &x, &L, &U, &gmres_param, &gmres_log);
  }


  data_d_matrix r = { Magma_DENSE };
  data_zvinit(&r, Asparse.num_rows, 1, zero);
  data_z_spmv(negone, &Asparse, &x, zero, &r);
  data_zaxpy(Asparse.num_rows, one, rhs_vector.val, 1, r.val, 1);
  for (int i = 0; i < Asparse.num_rows; i++) {
    GMRESDBG("r.val[%d] = %.16e\n", i, r.val[i]);
  }
  rnorm2 = data_dnrm2(Asparse.num_rows, r.val, 1);
  printf("%% external check of rnorm2 = %.16e;\n\n", rnorm2);

  // printf solver summary
  printf("gmres_search_directions = %d;\n", gmres_log.search_directions);
  printf("gmres_solve_time = %e;\n", gmres_log.solve_time);
  printf("gmres_initial_residual = %e;\n", gmres_log.initial_residual);
  printf("gmres_final_residual = %e;\n", gmres_log.final_residual);

  printf("\n\n");
  printf("%% ################################################################################\n");
  printf("%% Matrix: %s\n%% \t%d -by- %d with %d non-zeros\n",
    matrix_name, Asparse.num_rows, Asparse.num_cols, Asparse.nnz);
  if (strcmp(solver_name, "FGMRES") == 0) {
    printf("%% Solver: FGMRES\n");
  } else if (strcmp(solver_name, "FGMRESH") == 0) {
    printf("%% Solver: FGMRESH\n");
  }
  printf("%% \tsearch directions: %d\n", gmres_log.search_directions);
  printf("%% \tsolve time [s]: %e\n", gmres_log.solve_time);
  printf("%% \tinitial residual: %e\n", gmres_log.initial_residual);
  printf("%% \tfinal residual: %e\n", gmres_log.final_residual);
  printf("%% ################################################################################\n");
  printf("\n\n");
  printf("%% Done.\n");
  fflush(stdout);

  std::string s1(matrix_name);
  sparse_basename = basename(matrix_name);
  char * ext;
  ext = strrchr(sparse_basename, '.');
  strncpy(sparse_name, sparse_basename, int(ext - sparse_basename) );

  strcpy(output_basename, output_dir);
  strcat(output_basename, "/");
  strcat(output_basename, sparse_name);

  strcpy(output_x, output_basename);
  char suffixBuffer[256];
  sprintf(suffixBuffer, "_%s.mtx", solver_name);
  strcat(output_x, suffixBuffer);
  data_zwrite_csr_mtx(x, x.major, output_x);

  data_zmfree(&Asparse);
  data_zmfree(&x);
  data_zmfree(&rhs_vector);
  data_zmfree(&r);

  return 0;
} // main

// main
