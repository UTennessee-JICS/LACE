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
  char * sparse_filename;
  char * sparse_basename;
  char sparse_name[256];
  char * output_dir;
  char output_basename[256];
  char output_L[256];
  char output_U[256];
  int tile = 100;

  if (argc < 4) {
    printf("Usage %s <matrix> <tile size> <output directory>\n", argv[0]);
    return 1;
  } else   {
    sparse_filename = argv[1];
    tile            = atoi(argv[2]);
    output_dir      = argv[3];
    sparse_basename = basename(sparse_filename);
    char * ext;
    ext = strrchr(sparse_basename, '.');
    strncpy(sparse_name, sparse_basename, int(ext - sparse_basename) );
    printf("File %s basename %s name %s \n",
      sparse_filename, sparse_basename, sparse_name);
    printf("tile size is %d \n", tile);
    printf("Output directory is %s\n", output_dir);
    strcpy(output_basename, output_dir);
    strcat(output_basename, sparse_name);
    printf("Output file base name is %s\n", output_basename);
  }
  data_d_matrix Asparse = { Magma_CSR };
  CHECK(data_z_csr_mtx(&Asparse, sparse_filename) );

  // Scale matrix to have a unit diagonal
  if (argc >= 5) {
    if (strcmp(argv[4], "UNITDIAG") == 0) {
      printf("rescaling UNITDIAG\n");
      data_zmscale(&Asparse, Magma_UNITDIAG);
    }
  }

  bool comparison = true;
  data_d_matrix A = { Magma_CSR };
  data_zmconvert(Asparse, &A, Magma_CSR, Magma_CSR);
  data_d_matrix B = { Magma_CSR };
  data_zmconvert(Asparse, &B, Magma_CSR, Magma_CSR);

  // =========================================================================
  // MKL csrilu0  (Benchmark)
  // =========================================================================
  printf("%% MKL csrilu0 (Benchmark)\n");
  data_d_matrix Amkl = { Magma_CSR };
  data_zmconvert(Asparse, &Amkl, Magma_CSR, Magma_CSR);

  dataType wstart = omp_get_wtime();
  CHECK(data_dcsrilu0_mkl(&Amkl) );
  dataType wend = omp_get_wtime();
  printf("%% MKL csrilu0 required %f wall clock seconds as measured by omp_get_wtime()\n", wend - wstart);


  data_d_matrix Lmkl = { Magma_CSRL };
  Lmkl.diagorder_type = Magma_UNITY;
  data_zmconvert(Amkl, &Lmkl, Magma_CSR, Magma_CSRL);
  printf("test if Lmkl is lower: ");
  data_zcheckupperlower(&Lmkl);
  printf(" done.\n");
  data_d_matrix Umkl = { Magma_CSRU };
  Umkl.diagorder_type = Magma_VALUE;
  data_zmconvert(Amkl, &Umkl, Magma_CSR, Magma_CSRU);
  printf("test if Umkl is upper: ");
  data_zcheckupperlower(&Umkl);
  printf(" done.\n");
  data_d_matrix LUmkl = { Magma_CSR };
  data_zmconvert(Amkl, &LUmkl, Magma_CSR, Magma_CSR);

  dataType Amklres       = 0.0;
  dataType Amklnonlinres = 0.0;
  data_zilures(A, Lmkl, Umkl, &LUmkl, &Amklres, &Amklnonlinres);

  printf("MKL_csrilu0_res = %e\n", Amklres);
  printf("MKL_csrilu0_nonlinres = %e\n", Amklnonlinres);
  strcpy(output_L, output_basename);
  strcat(output_L, "_Lmkl.mtx");
  strcpy(output_U, output_basename);
  strcat(output_U, "_Umkl.mtx");
  data_zwrite_csr_mtx(Lmkl, Lmkl.major, output_L);
  data_zwrite_csr_mtx(Umkl, Umkl.major, output_U);
  data_zmfree(&Amkl);
  fflush(stdout);
  // =========================================================================

  // =========================================================================
  // PariLU v0.0
  // =========================================================================
  printf("%% PariLU v0.0 to 5 sweeps\n");
  // Separate the strictly lower and upper elements
  // into L, and U respectively.
  data_d_matrix L5 = { Magma_CSRL };
  data_d_matrix U5 = { Magma_CSCU };
  data_PariLU_v0_0(&A, &L5, &U5);

  printf("test if L is lower: ");
  data_zcheckupperlower(&L5);
  printf(" done.\n");
  printf("test if U is lower: ");
  data_zcheckupperlower(&U5);
  printf(" done.\n");
  // Check ||A-LU||_Frobenius
  dataType Ares       = 0.0;
  dataType Anonlinres = 0.0;
  data_d_matrix LU    = { Magma_CSR };
  data_zmconvert(A, &LU, Magma_CSR, Magma_CSR);
  data_zilures(A, L5, U5, &LU, &Ares, &Anonlinres);
  printf("PariLUv0_0-5_csrilu0_res = %e\n", Ares);
  printf("PariLUv0_0-5_csrilu0_nonlinres = %e\n", Anonlinres);
  fflush(stdout);

  EXPECT_LE(Ares, (Amklres * 10.0), comparison);
  EXPECT_LE(Anonlinres, (Amklnonlinres * 10.0), comparison);

  data_zmfree(&L5);
  data_zmfree(&U5);
  data_zmfree(&LU);

  // =========================================================================
  // PariLU v0.3
  // =========================================================================
  printf("%% PariLU v0.3\n");
  // Separate the strictly lower and upper elements
  // into L, and U respectively.
  data_d_matrix L = { Magma_CSRL };
  data_d_matrix U = { Magma_CSCU };
  dataType reduction = 1.0e-10;
  data_d_preconditioner_log p03_log;
  data_PariLU_v0_3(&A, &L, &U, reduction, &p03_log);
  // Check ||A-LU||_Frobenius
  Ares       = 0.0;
  Anonlinres = 0.0;
  LU         = { Magma_CSR };
  data_zmconvert(A, &LU, Magma_CSR, Magma_CSR);
  data_zilures(A, L, U, &LU, &Ares, &Anonlinres);
  printf("PariLUv0_3_csrilu0_res = %e\n", Ares);
  printf("PariLUv0_3_csrilu0_nonlinres = %e\n", Anonlinres);
  fflush(stdout);
  EXPECT_LE(Ares, (Amklres * 10.0), comparison);
  EXPECT_LE(Anonlinres, (Amklnonlinres * 10.0), comparison);

  data_zmfree(&LU);

  // =========================================================================
  // PariLU v0.3 gpu
  // =========================================================================
  printf("%% PariLU v0.3 GPU\n");
  L         = { Magma_CSRL };
  U         = { Magma_CSCU };
  reduction = 1.0e-15;
  data_d_preconditioner_log parilu_log;
  data_PariLU_v0_3_gpu(&A, &L, &U, reduction, &parilu_log, tile);
  Ares       = 0.0;
  Anonlinres = 0.0;
  LU         = { Magma_CSR };
  data_zmconvert(A, &LU, Magma_CSR, Magma_CSR);
  data_zilures(A, L, U, &LU, &Ares, &Anonlinres);
  printf("PariLU_v0_3_omp_num_threads = %d\n", parilu_log.omp_num_threads);
  printf("PariLU_v0_3_sweeps = %d\n", parilu_log.sweeps);
  printf("PariLU_v0_3_tol = %e\n", parilu_log.tol);
  printf("PariLU_v0_3_A_Frobenius = %e\n", parilu_log.A_Frobenius);
  printf("PariLU_v0_3_generation_time = %e\n", parilu_log.precond_generation_time);
  printf("PariLU_v0_3_initial_residual = %e\n", parilu_log.initial_residual);
  printf("PariLU_v0_3_initial_nonlinear_residual = %e\n", parilu_log.initial_nonlinear_residual);
  fflush(stdout);

  EXPECT_LE(Ares, (Amklres * 10.0), comparison);
  EXPECT_LE(Anonlinres, (Amklnonlinres * 10.0), comparison);

  data_zmfree(&LU);
  data_zmfree(&Asparse);
  data_zmfree(&A);
  data_zmfree(&B);
  data_zmfree(&Amkl);
  data_zmfree(&Lmkl);
  data_zmfree(&Umkl);

  return 0;
}
