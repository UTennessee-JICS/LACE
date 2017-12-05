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
  //char * sparse_filename = NULL;
  char * sparse_basename = NULL;
  char * output_dir      = NULL;
  char sparse_name[256];
  char output_basename[256];
  char output_L[256];
  char output_U[256];

  char default_matrix[] = "matrices/Trefethen_20.mtx";
  char * matrix_name    = NULL;

  char default_output_dir[] = "out_";

  data_d_preconditioner_log p03_log;

  p03_log.maxSweeps = 1;

  if (argc < 9) {
    printf("Usage %s --matrix <name> --sweeps <#> --threads <#> --outDir <name>\n", argv[0]);
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
      } else if ( (strcmp(argv[count], "--threads") == 0) &&
        count + 1 < argc)
      {
        p03_log.omp_num_threads = atoi(argv[count + 1]);
        count = count + 2;
      } else if ( (strcmp(argv[count], "--outDir") == 0) &&
        count + 1 < argc)
      {
        output_dir = argv[count + 1];
        count      = count + 2;
      } else {
        count++;
      }
    }
  }

  // load A matrix
  if (matrix_name == NULL) {
    matrix_name = default_matrix;
  }
  if (output_dir == NULL) {
    output_dir = default_output_dir;
  }

  data_d_matrix Asparse = { Magma_CSR };
  CHECK(data_z_csr_mtx(&Asparse, matrix_name) );


  // =========================================================================
  // PariLU v0.3 GPU
  // =========================================================================
  printf("%% PariLU v0.3 GPU prescribed number of sweeps\n");
  // Separate the strictly lower and upper elements
  // into L, and U respectively.
  data_d_matrix L    = { Magma_CSRL };
  data_d_matrix U    = { Magma_CSCU };
  dataType reduction = 1.0e-10;
  data_PariLU_v0_3_gpu_prescribedSweeps(&Asparse, &L, &U, reduction, &p03_log);
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


  std::string s1(matrix_name);
  std::cout << s1.substr(0, s1.find_last_of("\\/")) << std::endl;
  sparse_basename = basename(matrix_name);
  std::cout << sparse_basename << std::endl;
  char * ext;
  ext = strrchr(sparse_basename, '.');
  strncpy(sparse_name, sparse_basename, int(ext - sparse_basename) );
  printf("File %s basename %s name %s \n",
    matrix_name, sparse_basename, sparse_name);

  printf("Output directory is %s\n", output_dir);
  strcpy(output_basename, output_dir);
  strcat(output_basename, "/");
  strcat(output_basename, sparse_name);
  printf("Output file base name is %s\n", output_basename);

  strcpy(output_L, output_basename);
  char suffixBuffer[256];
  sprintf(suffixBuffer, "_LpariLUv03_%02dsweeps_%04dthreads.mtx", p03_log.sweeps, p03_log.omp_num_threads);
  strcat(output_L, suffixBuffer);
  strcpy(output_U, output_basename);
  sprintf(suffixBuffer, "_UpariLUv03_%02dsweeps_%04dthreads.mtx", p03_log.sweeps, p03_log.omp_num_threads);
  strcat(output_U, suffixBuffer);
  data_zwrite_csr_mtx(L, L.major, output_L);
  data_zwrite_csr_mtx(U, U.major, output_U);

  data_zmfree(&LU);
  data_zmfree(&Asparse);

  return 0;
} // main

// main
