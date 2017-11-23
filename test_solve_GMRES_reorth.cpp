#include "include/sparse.h"
#include <stdio.h>
#include <stdlib.h>
#include <libgen.h>
#include <string.h>
#include <vector>
#include <omp.h>
#include <float.h>
#include "math.h"
#include "mkl_blas.h"
#include "mkl_spblas.h"
#include "mkl_rci.h"

int
main(int argc, char * argv[])
{
  char * sparse_filename;
  char * rhs_filename;
  char * sparse_basename;
  char sparse_name[256];
  char * output_dir;
  char output_basename[256];

  dataType zero   = 0.0;
  dataType one    = 1.0;
  dataType negone = -1.0;

  dataType rnorm2 = 0.0;

  if (argc < 4) {
    printf("Usage %s <matrix> <rhs vector> <output directory> "
      "[GMRES_tolerance_type] [GMRES_tolerance] "
      "[max_search_dir] [reorth]\n", argv[0]);
    return 1;
  } else   {
    sparse_filename = argv[1];
    rhs_filename    = argv[2];
    output_dir      = argv[3];
    sparse_basename = basename(sparse_filename);
    char * ext;
    ext = strrchr(sparse_basename, '.');
    strncpy(sparse_name, sparse_basename, int(ext - sparse_basename) );
    printf("%% File %s basename %s name %s \n",
      sparse_filename, sparse_basename, sparse_name);
    printf("matrix = '%s'\n", sparse_name);
    printf("%% rhs vector name is %s \n", rhs_filename);
    printf("%% Output directory is %s\n", output_dir);
    strcpy(output_basename, output_dir);
    strcat(output_basename, "/");
    strcat(output_basename, sparse_name);
    strcat(output_basename, "_solution.mtx");
    printf("%% Output file base name is %s\n", output_basename);
  }
  data_d_matrix Asparse = { Magma_CSR };
  CHECK(data_z_csr_mtx(&Asparse, sparse_filename) );
  data_d_matrix rhs_vector = { Magma_DENSE };
  rhs_vector.major = MagmaRowMajor;

  // data_d_matrix A_org = {Magma_CSR};
  // CHECK( data_zmconvert( Asparse, &A_org, Magma_CSR, Magma_CSR ) );

  // Setup rhs
  if (strcmp(rhs_filename, "ONES") == 0) {
    printf("%% creating a vector of %d ones for the rhs.\n", Asparse.num_rows);
    CHECK(data_zvinit(&rhs_vector, Asparse.num_rows, 1, one) );
  } else   {
    CHECK(data_z_dense_mtx(&rhs_vector, rhs_vector.major, rhs_filename) );
  }
  // data_d_matrix rhs_org = {Magma_DENSE};
  // rhs_org.major = MagmaRowMajor;
  // CHECK( data_zmconvert( rhs_vector, &rhs_org, Magma_DENSE, Magma_DENSE ) );

  data_d_matrix x = { Magma_DENSE };
  CHECK(data_zvinit(&x, Asparse.num_rows, 1, zero) );

  data_d_gmres_param gmres_param;
  data_d_gmres_log gmres_log;

  // Set type of tolerance for stopping citeria for GMRES
  if (argc >= 5) {
    gmres_param.tol_type = atoi(argv[4]);
    printf("gmres_param_tol_type = %d\n", gmres_param.tol_type);
  }

  // Set tolerance for stopping citeria for GMRES
  if (argc >= 6) {
    gmres_param.rtol = atof(argv[5]);
    printf("gmres_param_rtol = %e\n", gmres_param.rtol);
  }

  // Set search directions
  if (argc >= 7) {
    gmres_param.search_max = atoi(argv[6]);
    printf("gmres_param_search_max = %d\n", gmres_param.search_max);
  }

  // Set reorthoganilization
  if (argc >= 8) {
    gmres_param.reorth = atoi(argv[7]);
    printf("gmres_param_reorth = %d\n", gmres_param.reorth);
  }

  data_gmres_reorth(&Asparse, &rhs_vector, &x, &gmres_param, &gmres_log);

  for (int i = 0; i < Asparse.num_rows; i++) {
    GMRESDBG("x.val[%d] = %.16e\n", i, x.val[i]);
  }

  data_zwrite_dense(x, output_basename);

  data_d_matrix r = { Magma_DENSE };
  data_zvinit(&r, Asparse.num_rows, 1, zero);
  data_z_spmv(negone, &Asparse, &x, zero, &r);
  data_zaxpy(Asparse.num_rows, one, rhs_vector.val, 1, r.val, 1);
  for (int i = 0; i < Asparse.num_rows; i++) {
    GMRESDBG("r.val[%d] = %.16e\n", i, r.val[i]);
  }
  rnorm2 = data_dnrm2(Asparse.num_rows, r.val, 1);
  printf("%% external check of rnorm2 = %.16e;\n\n", rnorm2);

  printf("gmres_search_directions = %d;\n", gmres_log.search_directions);
  printf("gmres_solve_time = %e;\n", gmres_log.solve_time);
  printf("gmres_initial_residual = %e;\n", gmres_log.initial_residual);
  printf("gmres_final_residual = %e;\n", gmres_log.final_residual);

  printf("\n\n");
  printf("%% ################################################################################\n");
  printf("%% Matrix: %s\n%% \t%d -by- %d with %d non-zeros\n",
    sparse_filename, Asparse.num_rows, Asparse.num_cols, Asparse.nnz);
  printf("%% Solver: GMRES\n");
  printf("%% \tsearch directions: %d\n", gmres_log.search_directions);
  printf("%% \tsolve time [s]: %e\n", gmres_log.solve_time);
  printf("%% \tinitial residual: %e\n", gmres_log.initial_residual);
  printf("%% \tfinal residual: %e\n", gmres_log.final_residual);
  printf("%% ################################################################################\n");
  printf("\n\n");
  printf("%% Done.\n");
  fflush(stdout);

  data_zmfree(&Asparse);
  data_zmfree(&x);
  data_zmfree(&rhs_vector);
  data_zmfree(&r);
  return 0;
} // main
