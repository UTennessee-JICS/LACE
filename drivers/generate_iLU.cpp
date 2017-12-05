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
  // typedef float dataType;
  typedef double dataType;
#endif

#include "mmio.h"
#include "sparse_types.h"

int
main(int argc, char * argv[])
{
  // begin with a square matrix A
  char * sparse_filename = NULL;
  char * sparse_basename = NULL;
  char * sparse_name = NULL;
  char * output_dir = NULL;
  char * output_basename = NULL;
  char * output_L = NULL;
  char * output_U = NULL;

  char default_matrix[] = "matrices/Trefethen_20.mtx";
  char * matrix_name    = NULL;

  char default_output_basename[] = "out_";
  char * rhs_name = NULL;

  data_d_preconditioner_log p03_log;
  p03_log.maxSweeps = 1;

  if (argc < 7) {
    printf("Usage %s --matrix <name> --sweeps <#> --outDir <name>\n", argv[0]);
    return 1;
  } 

  if (argc > 1) {
    int count = 1;
    while (count < argc) {
      if ( (strcmp(argv[count], "--matrix") == 0) &&
        count + 1 < argc)
      {
        matrix_name = argv[count + 1];
        count       = count + 2;
      } else if ( (strcmp(argv[count], "--sweeps") == 0) &&
        count + 1 < argc)
      {
        p03_log.maxSweeps = atoi(argv[count + 1]);
        count = count + 2;
      } else if ( (strcmp(argv[count], "--outDir") == 0) &&
        count + 1 < argc)
      {
        output_basename = argv[count + 1];
        count = count + 2;
      } else {
        count++;
      }
    }
  }

  // load A matrix
  if (matrix_name == NULL) {
    matrix_name = default_matrix;
  }
  if (rhs_name == NULL) {
    output_basename = default_output_basename;
  }

  data_d_matrix Asparse = { Magma_CSR };
  CHECK(data_z_csr_mtx(&Asparse, matrix_name) );


  // =========================================================================
  // PariLU v0.3
  // =========================================================================
  printf("%% PariLU v0.3 prescribed number of sweeps\n");
  // Separate the strictly lower and upper elements
  // into L, and U respectively.
  data_d_matrix L    = { Magma_CSRL };
  data_d_matrix U    = { Magma_CSCU };
  dataType reduction = 1.0e-10;
  data_PariLU_v0_3_prescribedSweeps(&Asparse, &L, &U, reduction, &p03_log);
  // Check ||A-LU||_Frobenius
  dataType Ares       = 0.0;
  dataType Anonlinres = 0.0;
  data_d_matrix LU    = { Magma_CSR };
  data_zmconvert(Asparse, &LU, Magma_CSR, Magma_CSR);
  data_zilures(Asparse, L, U, &LU, &Ares, &Anonlinres);
  printf("PariLU_v0_3_maxSweeps = %d\n", p03_log.maxSweeps);
  printf("PariLU_v0_3_sweeps = %d\n", p03_log.sweeps);
  printf("PariLU_v0_3_tol = %e\n", p03_log.tol);
  printf("PariLU_v0_3_finalStep = %e\n", p03_log.finalStep);
  printf("PariLU_v0_3_A_Frobenius = %e\n", p03_log.A_Frobenius);
  printf("PariLU_v0_3_generation_time = %e\n", p03_log.precond_generation_time);
  printf("PariLU_v0_3_initial_residual = %e\n", p03_log.initial_residual);
  printf("PariLU_v0_3_initial_nonlinear_residual = %e\n", p03_log.initial_nonlinear_residual);
  printf("PariLU_v0_3_csrilu0_res = %e\n", Ares);
  printf("PariLU_v0_3_csrilu0_nonlinres = %e\n", Anonlinres);
  printf("PariLU_v0_3_omp_num_threads = %d\n", p03_log.omp_num_threads);

  fflush(stdout);

  data_zmfree(&LU);
  data_zmfree(&Asparse);

  return 0;
} // main
