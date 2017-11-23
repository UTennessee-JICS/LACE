/*
 * make -f makefile_beacon test_iLU -B
 */

#include <stdlib.h>
#include <stdio.h>
#include <libgen.h>
#include <string>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <vector>
#include <omp.h>

#ifndef DATA_TYPE_H
  # define DATA_TYPE_H
  // typedef float dataType;
  typedef double dataType;
#endif

#include "include/mmio.h"
#include "include/sparse_types.h"

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
  // char sparse_filename[] = "testing/matrices/dRdQ_sm.mtx";
  // char sparse_filename[] = "testing/matrices/30p30n.mtx";
  // char sparse_filename[] = "testing/matrices/paper1_matrices/ani5_crop.mtx";
  data_d_matrix Asparse = { Magma_CSR };
  CHECK(data_z_csr_mtx(&Asparse, sparse_filename) );

  // Scale matrix to have a unit diagonal
  if (argc >= 5) {
    if (strcmp(argv[4], "UNITDIAG") == 0) {
      printf("rescaling UNITDIAG\n");
      data_zmscale(&Asparse, Magma_UNITDIAG);
      // data_zwrite_csr( &Asparse );
    }
  }

  data_d_matrix A = { Magma_CSR };
  data_zmconvert(Asparse, &A, Magma_CSR, Magma_CSR);
  data_d_matrix B = { Magma_CSR };
  data_zmconvert(Asparse, &B, Magma_CSR, Magma_CSR);
  // data_zdisplay_dense( &A );
  // data_zmfree( &Asparse );

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
  // data_zmfree( &Lmkl );
  // data_zmfree( &Umkl );
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
  // data_zmfree( &L );
  // data_zmfree( &U );
  // data_zmfree( &LU );
  fflush(stdout);

  data_d_matrix Ldiff = { Magma_CSRL };
  dataType Lres       = 0.0;
  dataType Lnonlinres = 0.0;
  data_zdiff_csr(&Lmkl, &L5, &Ldiff, &Lres, &Lnonlinres);
  // data_zwrite_csr( &Ldiff );
  printf("L_res = %e\n", Lres);
  printf("L_nonlinres = %e\n", Lnonlinres);
  fflush(stdout);

  data_d_matrix Udiff = { Magma_CSRU };
  dataType Ures       = 0.0;
  dataType Unonlinres = 0.0;
  data_zdiff_csr(&Umkl, &U5, &Udiff, &Ures, &Unonlinres);
  // data_zwrite_csr( &Udiff );
  printf("U_res = %e\n", Ures);
  printf("U_nonlinres = %e\n", Unonlinres);
  fflush(stdout);
  dataType vmaxA = 0.0;
  int imaxA      = 0;
  int jmaxA      = 0;
  data_maxfabs_csr(Ldiff, &imaxA, &jmaxA, &vmaxA);
  printf("max(fabs(Ldiff)) = (%d,%d) %e\n", imaxA, jmaxA, vmaxA);
  data_maxfabs_csr(Udiff, &imaxA, &jmaxA, &vmaxA);
  printf("max(fabs(Udiff)) = (%d,%d) %e\n", imaxA, jmaxA, vmaxA);

  printf("test if Ldiff is lower: ");
  data_zcheckupperlower(&Ldiff);
  printf(" done.\n");
  printf("test if Udiff is lower: ");
  data_zcheckupperlower(&Udiff);
  printf(" done.\n");

  strcpy(output_L, output_basename);
  strcat(output_L, "_L5pariLUv0_0.mtx");
  strcpy(output_U, output_basename);
  strcat(output_U, "_U5pariLUv0_0.mtx");
  data_zwrite_csr_mtx(L5, L5.major, output_L);
  data_zwrite_csr_mtx(U5, U5.major, output_U);
  strcpy(output_L, output_basename);
  strcat(output_L, "_L5pariLUv0_0_diff.mtx");
  strcpy(output_U, output_basename);
  strcat(output_U, "_U5pariLUv0_0_diff.mtx");
  data_zwrite_csr_mtx(Ldiff, Ldiff.major, output_L);
  data_zwrite_csr_mtx(Udiff, Udiff.major, output_U);
  data_zmfree(&Ldiff);
  data_zmfree(&Udiff);

  // data_zmfree( &L );
  // data_zmfree( &U );
  data_zmfree(&LU);

  // =========================================================================

  // =========================================================================
  // PariLU v0.1
  // =========================================================================
  printf("%% PariLU v0.1 to interative convergence\n");
  // Separate the strictly lower and upper elements
  // into L, and U respectively.
  data_d_matrix L = { Magma_CSRL };
  // L = {Magma_CSRL};
  // L.diagorder_type = Magma_UNITY;
  // data_zmconvert(A, &L, Magma_CSR, Magma_CSRL);
  data_d_matrix U = { Magma_CSCU };
  // U = {Magma_CSCU};
  // U.diagorder_type = Magma_VALUE;
  // data_zmconvert(A, &U, Magma_CSR, Magma_CSRU);
  data_PariLU_v0_1(&A, &L, &U);
  // Check ||A-LU||_Frobenius
  Ares       = 0.0;
  Anonlinres = 0.0;
  LU         = { Magma_CSR };
  data_zmconvert(A, &LU, Magma_CSR, Magma_CSR);
  data_zilures(A, L, U, &LU, &Ares, &Anonlinres);
  printf("PariLUv0_1_csrilu0_res = %e\n", Ares);
  printf("PariLUv0_1_csrilu0_nonlinres = %e\n", Anonlinres);
  // data_zmfree( &L );
  // data_zmfree( &U );
  // data_zmfree( &LU );
  fflush(stdout);

  Ldiff      = { Magma_CSRL };
  Lres       = 0.0;
  Lnonlinres = 0.0;
  data_zdiff_csr(&Lmkl, &L, &Ldiff, &Lres, &Lnonlinres);
  // data_zwrite_csr( &Ldiff );
  printf("L_res = %e\n", Lres);
  printf("L_nonlinres = %e\n", Lnonlinres);
  fflush(stdout);
  Udiff      = { Magma_CSRU };
  Ures       = 0.0;
  Unonlinres = 0.0;
  data_zdiff_csr(&Umkl, &U, &Udiff, &Ures, &Unonlinres);
  // data_zwrite_csr( &Udiff );
  printf("U_res = %e\n", Ures);
  printf("U_nonlinres = %e\n", Unonlinres);
  fflush(stdout);
  data_maxfabs_csr(Ldiff, &imaxA, &jmaxA, &vmaxA);
  printf("max(fabs(Ldiff)) = (%d,%d) %e\n", imaxA, jmaxA, vmaxA);
  data_maxfabs_csr(Udiff, &imaxA, &jmaxA, &vmaxA);
  printf("max(fabs(Udiff)) = (%d,%d) %e\n", imaxA, jmaxA, vmaxA);

  strcpy(output_L, output_basename);
  strcat(output_L, "_LpariLUv0_1.mtx");
  strcpy(output_U, output_basename);
  strcat(output_U, "_UpariLUv0_1.mtx");
  data_zwrite_csr_mtx(L, L.major, output_L);
  data_zwrite_csr_mtx(U, U.major, output_U);
  strcpy(output_L, output_basename);
  strcat(output_L, "_LpariLUv0_1_diff.mtx");
  strcpy(output_U, output_basename);
  strcat(output_U, "_UpariLUv0_1_diff.mtx");
  data_zwrite_csr_mtx(Ldiff, Ldiff.major, output_L);
  data_zwrite_csr_mtx(Udiff, Udiff.major, output_U);
  data_zmfree(&Ldiff);
  data_zmfree(&Udiff);

  // data_zmfree( &L );
  // data_zmfree( &U );
  data_zmfree(&LU);

  // =========================================================================

  // =========================================================================
  // PariLU v0.2
  // =========================================================================
  printf("%% PariLU v0.2\n");
  // Separate the strictly lower and upper elements
  // into L, and U respectively.
  // data_d_matrix L = {Magma_CSRL};
  L = { Magma_CSRL };
  // L.diagorder_type = Magma_UNITY;
  // data_zmconvert(A, &L, Magma_CSR, Magma_CSRL);
  // data_d_matrix U = {Magma_CSCU};
  U = { Magma_CSCU };
  // U.diagorder_type = Magma_VALUE;
  // data_zmconvert(A, &U, Magma_CSR, Magma_CSRU);
  dataType reduction = 1.0e-10;
  data_PariLU_v0_2(&A, &L, &U, reduction);
  // Check ||A-LU||_Frobenius
  Ares       = 0.0;
  Anonlinres = 0.0;
  LU         = { Magma_CSR };
  data_zmconvert(A, &LU, Magma_CSR, Magma_CSR);
  data_zilures(A, L, U, &LU, &Ares, &Anonlinres);
  printf("PariLUv0_2_csrilu0_res = %e\n", Ares);
  printf("PariLUv0_2_csrilu0_nonlinres = %e\n", Anonlinres);
  // data_zmfree( &L );
  // data_zmfree( &U );
  // data_zmfree( &LU );
  fflush(stdout);

  Ldiff      = { Magma_CSRL };
  Lres       = 0.0;
  Lnonlinres = 0.0;
  data_zdiff_csr(&Lmkl, &L, &Ldiff, &Lres, &Lnonlinres);
  // data_zwrite_csr( &Ldiff );
  printf("L_res = %e\n", Lres);
  printf("L_nonlinres = %e\n", Lnonlinres);
  fflush(stdout);
  Udiff      = { Magma_CSRU };
  Ures       = 0.0;
  Unonlinres = 0.0;
  data_zdiff_csr(&Umkl, &U, &Udiff, &Ures, &Unonlinres);
  // data_zwrite_csr( &Udiff );
  printf("U_res = %e\n", Ures);
  printf("U_nonlinres = %e\n", Unonlinres);
  fflush(stdout);
  data_maxfabs_csr(Ldiff, &imaxA, &jmaxA, &vmaxA);
  printf("max(fabs(Ldiff)) = (%d,%d) %e\n", imaxA, jmaxA, vmaxA);
  data_maxfabs_csr(Udiff, &imaxA, &jmaxA, &vmaxA);
  printf("max(fabs(Udiff)) = (%d,%d) %e\n", imaxA, jmaxA, vmaxA);
  strcpy(output_L, output_basename);
  strcat(output_L, "_LpariLUv0_2.mtx");
  strcpy(output_U, output_basename);
  strcat(output_U, "_UpariLUv0_2.mtx");
  data_zwrite_csr_mtx(L, L.major, output_L);
  data_zwrite_csr_mtx(U, U.major, output_U);
  strcpy(output_L, output_basename);
  strcat(output_L, "_LpariLUv0_2_diff.mtx");
  strcpy(output_U, output_basename);
  strcat(output_U, "_UpariLUv0_2_diff.mtx");
  data_zwrite_csr_mtx(Ldiff, Ldiff.major, output_L);
  data_zwrite_csr_mtx(Udiff, Udiff.major, output_U);
  data_zmfree(&Ldiff);
  data_zmfree(&Udiff);

  // data_zmfree( &L );
  // data_zmfree( &U );
  data_zmfree(&LU);


  // =========================================================================
  // PariLU v0.3
  // =========================================================================
  printf("%% PariLU v0.3\n");
  // Separate the strictly lower and upper elements
  // into L, and U respectively.
  // data_d_matrix L = {Magma_CSRL};
  L = { Magma_CSRL };
  // L.diagorder_type = Magma_UNITY;
  // data_zmconvert(A, &L, Magma_CSR, Magma_CSRL);
  // data_d_matrix U = {Magma_CSCU};
  U = { Magma_CSCU };
  // U.diagorder_type = Magma_VALUE;
  // data_zmconvert(A, &U, Magma_CSR, Magma_CSRU);
  reduction = 1.0e-10;
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
  // data_zmfree( &L );
  // data_zmfree( &U );
  // data_zmfree( &LU );
  fflush(stdout);

  data_zmfree(&LU);


  // =========================================================================
  // PariLU v0.4
  // =========================================================================
  printf("%% PariLU v0.4\n");
  // Separate the strictly lower and upper elements
  // into L, and U respectively.
  // data_d_matrix L = {Magma_CSRL};
  L = { Magma_CSRL };
  // L.diagorder_type = Magma_UNITY;
  // data_zmconvert(A, &L, Magma_CSR, Magma_CSRL);
  // data_d_matrix U = {Magma_CSCU};
  U = { Magma_CSCU };
  // U.diagorder_type = Magma_VALUE;
  // data_zmconvert(A, &U, Magma_CSR, Magma_CSRU);
  reduction = 1.0e-10;
  data_PariLU_v0_4(&A, &L, &U, reduction);
  // Check ||A-LU||_Frobenius
  Ares       = 0.0;
  Anonlinres = 0.0;
  LU         = { Magma_CSR };
  data_zmconvert(A, &LU, Magma_CSR, Magma_CSR);
  data_zilures(A, L, U, &LU, &Ares, &Anonlinres);
  printf("PariLUv0_4_csrilu0_res = %e\n", Ares);
  printf("PariLUv0_4_csrilu0_nonlinres = %e\n", Anonlinres);
  // data_zmfree( &L );
  // data_zmfree( &U );
  // data_zmfree( &LU );
  fflush(stdout);


  // ===============================
  printf("\n===============================\n");
  printf("Compare block factorization\n");
  data_d_matrix LU_coffe      = { Magma_CSR };
  char const * coffe_filename = "30p30n-iLU0.mtx";
  int read = data_z_csr_mtx(&LU_coffe, coffe_filename);

  if (read) {
    data_d_matrix L_coffe = { Magma_CSRL };
    L_coffe.diagorder_type = Magma_UNITY;
    printf("before extracting L\n");
    data_zmconvert(LU_coffe, &L_coffe, Magma_CSR, Magma_CSRL);
    data_d_matrix U_coffe = { Magma_CSRU };
    U_coffe.diagorder_type = Magma_VALUE;
    printf("before extracting U\n");
    data_zmconvert(LU_coffe, &U_coffe, Magma_CSR, Magma_CSRU);
    LU = { Magma_CSR };
    data_zmconvert(A, &LU, Magma_CSR, Magma_CSR);
    data_zilures(A, L_coffe, U_coffe, &LU, &Ares, &Anonlinres);
    printf("Coffe_res = %e\n", Ares);
    printf("Coffe_nonlinres = %e\n", Anonlinres);
    Lres       = 0.0;
    Lnonlinres = 0.0;
    printf("before L diff\n");
    data_zdiff_csr(&Lmkl, &L_coffe, &Ldiff, &Lres, &Lnonlinres);
    printf("L_res = %e\n", Lres);
    printf("L_nonlinres = %e\n", Lnonlinres);
    Ures       = 0.0;
    Unonlinres = 0.0;
    printf("before U diff\n");
    data_zdiff_csr(&Umkl, &U_coffe, &Udiff, &Ures, &Unonlinres);
    printf("U_res = %e\n", Ures);
    printf("U_nonlinres = %e\n", Unonlinres);
    fflush(stdout);

    data_zwrite_csr_mtx(L_coffe, L_coffe.major, "L_coffe_30p30n.mtx");
    data_zwrite_csr_mtx(U_coffe, U_coffe.major, "U_coffe_30p30n.mtx");
    data_zwrite_csr_mtx(Ldiff, Ldiff.major, "Ldiff_30p30n_mkl_coffe.mtx");
    data_zwrite_csr_mtx(Udiff, Udiff.major, "Udiff_30p30n_mkl_coffe.mtx");

    data_maxfabs_csr(Ldiff, &imaxA, &jmaxA, &vmaxA);
    printf("max(fabs(Ldiff)) = (%d,%d) %e\n", imaxA, jmaxA, vmaxA);
    data_maxfabs_csr(Udiff, &imaxA, &jmaxA, &vmaxA);
    printf("max(fabs(Udiff)) = (%d,%d) %e\n", imaxA, jmaxA, vmaxA);

    data_zmfree(&Ldiff);
    data_zmfree(&Udiff);

    Lres       = 0.0;
    Lnonlinres = 0.0;
    printf("before L5 diff\n");
    data_zdiff_csr(&L5, &L_coffe, &Ldiff, &Lres, &Lnonlinres);
    printf("L_res = %e\n", Lres);
    printf("L_nonlinres = %e\n", Lnonlinres);
    Ures       = 0.0;
    Unonlinres = 0.0;
    printf("before U5 diff\n");
    data_zdiff_csr(&U5, &U_coffe, &Udiff, &Ures, &Unonlinres);
    printf("U_res = %e\n", Ures);
    printf("U_nonlinres = %e\n", Unonlinres);
    fflush(stdout);

    data_zwrite_csr_mtx(Ldiff, Ldiff.major, "L5diff_30p30n_pariLU_coffe.mtx");
    data_zwrite_csr_mtx(Udiff, Udiff.major, "U5diff_30p30n_pariLU_coffe.mtx");
    data_maxfabs_csr(Ldiff, &imaxA, &jmaxA, &vmaxA);
    printf("max(fabs(Ldiff)) = (%d,%d) %e\n", imaxA, jmaxA, vmaxA);
    data_maxfabs_csr(Udiff, &imaxA, &jmaxA, &vmaxA);
    printf("max(fabs(Udiff)) = (%d,%d) %e\n", imaxA, jmaxA, vmaxA);
    data_zmfree(&Ldiff);
    data_zmfree(&Udiff);

    Lres       = 0.0;
    Lnonlinres = 0.0;
    printf("before L diff\n");
    data_zdiff_csr(&L, &L_coffe, &Ldiff, &Lres, &Lnonlinres);
    printf("L_res = %e\n", Lres);
    printf("L_nonlinres = %e\n", Lnonlinres);
    Ures       = 0.0;
    Unonlinres = 0.0;
    printf("before U diff\n");
    data_zdiff_csr(&U, &U_coffe, &Udiff, &Ures, &Unonlinres);
    printf("U_res = %e\n", Ures);
    printf("U_nonlinres = %e\n", Unonlinres);
    fflush(stdout);

    data_zwrite_csr_mtx(Ldiff, Ldiff.major, "Ldiff_30p30n_pariLU_coffe.mtx");
    data_zwrite_csr_mtx(Udiff, Udiff.major, "Udiff_30p30n_pariLU_coffe.mtx");
    data_maxfabs_csr(Ldiff, &imaxA, &jmaxA, &vmaxA);
    printf("max(fabs(Ldiff)) = (%d,%d) %e\n", imaxA, jmaxA, vmaxA);
    data_maxfabs_csr(Udiff, &imaxA, &jmaxA, &vmaxA);
    printf("max(fabs(Udiff)) = (%d,%d) %e\n", imaxA, jmaxA, vmaxA);
    data_zmfree(&Ldiff);
    data_zmfree(&Udiff);

    data_zmfree(&L);
    data_zmfree(&U);
    // data_zmfree( &LU );
    data_zmfree(&L_coffe);
    data_zmfree(&U_coffe);
    data_zmfree(&LU_coffe);
  }

  // ===============================
  printf("\n===============================\n");

  DEV_CHECKPT;
  // =========================================================================

  /*
   * // =========================================================================
   * // PariLU v4.0
   * // =========================================================================
   * printf("%% PariLU v4.0\n");
   * // Separate the strictly lower and upper elements
   * // into L, and U respectively.
   * //data_d_matrix L = {Magma_CSRL};
   * L = {Magma_CSRL};
   * //L.diagorder_type = Magma_UNITY;
   * //data_zmconvert(A, &L, Magma_CSR, Magma_CSRL);
   * //data_d_matrix U = {Magma_CSCU};
   * U = {Magma_CSRU};
   * //U.diagorder_type = Magma_VALUE;
   * //data_zmconvert(A, &U, Magma_CSR, Magma_CSRU);
   * data_PariLU_v4_0( &A, &L, &U);
   * // Check ||A-LU||_Frobenius
   * Ares = 0.0;
   * Anonlinres = 0.0;
   * LU = {Magma_CSR};
   * data_zmconvert(A, &LU, Magma_CSR, Magma_CSR);
   * data_zilures(A, L, U, &LU, &Ares, &Anonlinres);
   * printf("PariLUv4_0_csrilu0_res = %e\n", Ares);
   * printf("PariLUv4_0_csrilu0_nonlinres = %e\n", Anonlinres);
   * //data_zmfree( &L );
   * //data_zmfree( &U );
   * //data_zmfree( &LU );
   * fflush(stdout);
   *
   * Ldiff = {Magma_CSR};
   * Lres = 0.0;
   * Lnonlinres = 0.0;
   * data_zdiff_csr(&Lmkl, &L, &Ldiff, &Lres, &Lnonlinres );
   * //data_zwrite_csr( &Ldiff );
   * printf("L_res = %e\n", Lres);
   * printf("L_nonlinres = %e\n", Lnonlinres);
   * fflush(stdout);
   * Udiff = {Magma_CSR};
   * Ures = 0.0;
   * Unonlinres = 0.0;
   * data_zdiff_csr(&Umkl, &U, &Udiff, &Ures, &Unonlinres );
   * //data_zwrite_csr( &Udiff );
   * printf("U_res = %e\n", Ures);
   * printf("U_nonlinres = %e\n", Unonlinres);
   * fflush(stdout);
   * strcpy( output_L, output_basename );
   * strcat( output_L, "_LpariLUv4_0.mtx" );
   * strcpy( output_U, output_basename );
   * strcat( output_U, "_UpariLUv4_0.mtx" );
   * data_zwrite_csr_mtx( L, L.major, output_L );
   * data_zwrite_csr_mtx( U, U.major, output_U );
   * strcpy( output_L, output_basename );
   * strcat( output_L, "_LpariLUv4_0_diff.mtx" );
   * strcpy( output_U, output_basename );
   * strcat( output_U, "_UpariLUv4_0_diff.mtx" );
   * data_zwrite_csr_mtx( Ldiff, Ldiff.major, output_L );
   * data_zwrite_csr_mtx( Udiff, Udiff.major, output_U );
   * data_zmfree( &Ldiff );
   * data_zmfree( &Udiff );
   *
   * data_zmfree( &L );
   * data_zmfree( &U );
   * data_zmfree( &LU );
   *
   * // =========================================================================
   */

  /*
   * // =========================================================================
   * // PariLU v0.1
   * // =========================================================================
   * //
   * // Separate the strictly lower and upper, elements
   * // into L, U respectively.
   * // Convert U to column major storage.
   * printf("%% PariLU v0.1\n");
   * L = {Magma_DENSEL};
   * U = {Magma_DENSEU};
   * data_PariLU_v0_1( &A, &L, &U);
   * // Check ||A-LU||_Frobenius
   * data_zfrobenius_LUresidual(A, L, U, &Adiff);
   * printf("PariLUv0_1_res = %e\n", Adiff);
   * data_zmfree( &L );
   * data_zmfree( &U );
   * fflush(stdout);
   * // =========================================================================
   *
   * // =========================================================================
   * // PariLU v1.0
   * // =========================================================================
   * //
   * // Separate the strictly lower, upper elements
   * // into L and U respectively.
   * printf("%% PariLU v1.0\n");
   * L = {Magma_DENSEL};
   * U = {Magma_DENSEU};
   * // PariLU with dot products replacing summations
   * data_PariLU_v1_0( &A, &L, &U);
   * // Check ||A-LU||_Frobenius
   * data_zfrobenius_LUresidual(A, L, U, &Adiff);
   * printf("PariLUv1_0_res = %e\n", Adiff);
   * data_zmfree( &L );
   * data_zmfree( &U );
   * fflush(stdout);
   * // =========================================================================
   *
   * // =========================================================================
   * // PariLU v1.1
   * // =========================================================================
   * //
   * // Separate the strictly lower, upper elements
   * // into L and U respectively.
   * // Convert U to column major storage.
   * printf("%% PariLU v1.1\n");
   * L = {Magma_DENSEL};
   * U = {Magma_DENSEU};
   * // PariLU with dot products replacing summations
   * data_PariLU_v1_1( &A, &L, &U);
   * // Check ||A-LU||_Frobenius
   * data_zfrobenius_LUresidual(A, L, U, &Adiff);
   * printf("PariLUv1_1_res = %e\n", Adiff);
   * data_zmfree( &L );
   * data_zmfree( &U );
   * fflush(stdout);
   * // =========================================================================
   *
   * // =========================================================================
   * // PariLU v1.2
   * // =========================================================================
   * //
   * // Separate the strictly lower, strictly upper, and diagonal elements
   * // into L, U, and D respectively.
   * // PariLU with dot products and a tiled access pattern
   * printf("%% PariLU v1.2\n");
   * L = {Magma_DENSEL};
   * U = {Magma_DENSEU};
   * int tile = 8;
   * // PariLU with dot products replacing summations
   * data_PariLU_v1_2( &A, &L, &U, tile);
   * // Check ||A-LU||_Frobenius
   * data_zfrobenius_LUresidual(A, L, U, &Adiff);
   * printf("PariLUv1_2_res = %e\n", Adiff);
   * data_zmfree( &L );
   * data_zmfree( &U );
   * fflush(stdout);
   * // =========================================================================
   * data_zmconvert( B, &A, Magma_DENSE, Magma_DENSE );  // reset A
   * // =========================================================================
   * // PariLU v1.3
   * // =========================================================================
   * //
   * // Separate the strictly lower, strictly upper, and diagonal elements
   * // into L, U, and D respectively.
   * // Convert U to column major storage.
   * printf("%% PariLU v1.3\n");
   * L = {Magma_DENSEL};
   * U = {Magma_DENSEU};
   * tile = 8;
   * // PariLU with dot products replacing summations
   * data_PariLU_v1_3( &A, &L, &U, tile);
   * // Check ||A-LU||_Frobenius
   * data_zfrobenius_LUresidual(A, L, U, &Adiff);
   * printf("PariLUv1_3_res = %e\n", Adiff);
   * data_zmfree( &L );
   * data_zmfree( &U );
   * fflush(stdout);
   * // =========================================================================
   * data_zmconvert( B, &A, Magma_DENSE, Magma_DENSE );  // reset A
   * // =========================================================================
   * // PariLU v2.0
   * // =========================================================================
   * //
   * // Separate the strictly lower, strictly upper, and diagonal elements
   * // into L, U, and D respectively.
   * // PariLU with matrix-vector products and a tiled access pattern
   * printf("%% PariLU v2.0\n");
   * L = {Magma_DENSEL};
   * U = {Magma_DENSEU};
   * tile = 8;
   * // PariLU with dot products replacing summations
   * data_PariLU_v2_0( &A, &L, &U, tile);
   * // Check ||A-LU||_Frobenius
   * data_zfrobenius_LUresidual(A, L, U, &Adiff);
   * printf("PariLUv2_0_res = %e\n", Adiff);
   * data_zmfree( &L );
   * data_zmfree( &U );
   * fflush(stdout);
   * // =========================================================================
   *
   * // =========================================================================
   * // PariLU v2.1
   * // =========================================================================
   * //
   * // Separate the strictly lower, strictly upper, and diagonal elements
   * // into L, U, and D respectively.
   * // Convert U to column major storage.
   *
   * // PariLU with matrix-vector products and a tiled access pattern
   *
   * // Check A-LU
   * // Check ||A-LU||_Frobenius
   * // =========================================================================
   *
   */

  /*
   * data_zmfree( &A );
   * data_zmconvert( B, &A, Magma_CSR, Magma_CSR );  // reset A
   * // =========================================================================
   * // PariLU v3.0
   * // =========================================================================
   * //
   * // Separate the strictly lower, strictly upper, and diagonal elements
   * // into L, U, and D respectively.
   * // PariLU with matrix-matrix products and a tiled access pattern
   * printf("%% PariLU v3.0\n");
   * fflush(stdout);
   * //data_d_matrix L = {Magma_CSRL};
   * L = {Magma_CSRL};
   * //data_d_matrix U = {Magma_CSRU};
   * U = {Magma_CSRU};
   * //int tile = 5;
   * // PariLU with dgemm operations replacing summations
   * data_PariLU_v3_0( &A, &L, &U, tile);
   * // Check ||A-LU||_Frobenius
   * //data_d_matrix LU = {Magma_CSR};
   * //dataType Ares = 0.0;
   * //dataType Anonlinres = 0.0;
   * Ares = 0.0;
   * Anonlinres = 0.0;
   * data_zilures(A, L, U, &LU, &Ares, &Anonlinres);
   * printf("PariLUv3_0_csrilu0_res = %e\n", Ares);
   * printf("PariLUv3_0_csrilu0_nonlinres = %e\n", Anonlinres);
   * //data_zmfree( &L );
   * //data_zmfree( &U );
   * fflush(stdout);
   *
   * // diff mkl and PariLU
   * //data_d_matrix Ldiff = {Magma_CSR};
   * //dataType Lres = 0.0;
   * //dataType Lnonlinres = 0.0;
   * Lres = 0.0;
   * Lnonlinres = 0.0;
   * data_zdiff_csr(&Lmkl, &L, &Ldiff, &Lres, &Lnonlinres );
   * //data_zwrite_csr( &Ldiff );
   * printf("L_res = %e\n", Lres);
   * printf("L_nonlinres = %e\n", Lnonlinres);
   * fflush(stdout);
   * //data_d_matrix Udiff = {Magma_CSR};
   * //dataType Ures = 0.0;
   * //dataType Unonlinres = 0.0;
   * Ures = 0.0;
   * Unonlinres = 0.0;
   * data_zdiff_csr(&Umkl, &U, &Udiff, &Ures, &Unonlinres );
   * //data_zwrite_csr( &Udiff );
   * printf("U_res = %e\n", Ures);
   * printf("U_nonlinres = %e\n", Unonlinres);
   * fflush(stdout);
   *
   * data_zmfree( &Ldiff );
   * data_zmfree( &Udiff );
   *
   * data_zmfree( &L );
   * data_zmfree( &U );
   * data_zmfree( &LU );
   *
   * // =========================================================================
   *
   * data_zmfree( &A );
   * data_zmconvert( B, &A, Magma_CSR, Magma_CSR );  // reset A
   * // =========================================================================
   * // PariLU v3.1
   * // =========================================================================
   * //
   * // Separate the strictly lower, strictly upper, and diagonal elements
   * // into L, U, and D respectively.
   * // PariLU with matrix-matrix products and a tiled access pattern limited to active tiles
   * printf("%% PariLU v3.1\n");
   * //data_d_matrix L = {Magma_CSRL};
   * L = {Magma_CSRL};
   * //data_d_matrix U = {Magma_CSRU};
   * U = {Magma_CSRU};
   * //int tile = 5;
   * // PariLU with dgemm operations replacing summations
   * data_PariLU_v3_1( &A, &L, &U, tile);
   * // Check ||A-LU||_Frobenius
   * //data_d_matrix LU = {Magma_CSR};
   * //dataType Ares = 0.0;
   * //dataType Anonlinres = 0.0;
   * Ares = 0.0;
   * Anonlinres = 0.0;
   * data_zilures(A, L, U, &LU, &Ares, &Anonlinres);
   * printf("PariLUv3_1_csrilu0_res = %e\n", Ares);
   * printf("PariLUv3_1_csrilu0_nonlinres = %e\n", Anonlinres);
   * //data_zmfree( &L );
   * //data_zmfree( &U );
   * fflush(stdout);
   *
   * // diff mkl and PariLU
   * //data_d_matrix Ldiff = {Magma_CSR};
   * //dataType Lres = 0.0;
   * //dataType Lnonlinres = 0.0;
   * Lres = 0.0;
   * Lnonlinres = 0.0;
   * data_zdiff_csr(&Lmkl, &L, &Ldiff, &Lres, &Lnonlinres );
   * //data_zwrite_csr( &Ldiff );
   * printf("L_res = %e\n", Lres);
   * printf("L_nonlinres = %e\n", Lnonlinres);
   * fflush(stdout);
   * //data_d_matrix Udiff = {Magma_CSR};
   * //dataType Ures = 0.0;
   * //dataType Unonlinres = 0.0;
   * Ures = 0.0;
   * Unonlinres = 0.0;
   * data_zdiff_csr(&Umkl, &U, &Udiff, &Ures, &Unonlinres );
   * //data_zwrite_csr( &Udiff );
   * printf("U_res = %e\n", Ures);
   * printf("U_nonlinres = %e\n", Unonlinres);
   * fflush(stdout);
   *
   * data_zmfree( &Ldiff );
   * data_zmfree( &Udiff );
   *
   * data_zmfree( &L );
   * data_zmfree( &U );
   * data_zmfree( &LU );
   *
   * // =========================================================================
   */

  /*
   * data_zmfree( &A );
   * data_zmconvert( B, &A, Magma_CSR, Magma_CSR );  // reset A
   * // =========================================================================
   * // PariLU v3.2
   * // =========================================================================
   * //
   * // Separate the strictly lower, strictly upper, and diagonal elements
   * // into L, U, and D respectively.
   * // PariLU with sparse matrix-matrix products and a tiled access pattern limited to active tiles
   * printf("%% PariLU v3.2\n");
   * fflush(stdout);
   * //data_d_matrix L = {Magma_CSRL};
   * L = {Magma_CSRL};
   * //data_d_matrix U = {Magma_CSRU};
   * U = {Magma_CSRU};
   * //int tile = 5;
   * // PariLU with dgemm operations replacing summations
   * data_PariLU_v3_2( &A, &L, &U, tile);
   * // Check ||A-LU||_Frobenius
   * //data_d_matrix LU = {Magma_CSR};
   * //dataType Ares = 0.0;
   * //dataType Anonlinres = 0.0;
   * Ares = 0.0;
   * Anonlinres = 0.0;
   * data_zilures(A, L, U, &LU, &Ares, &Anonlinres);
   * printf("PariLUv3_2_csrilu0_res = %e\n", Ares);
   * printf("PariLUv3_2_csrilu0_nonlinres = %e\n", Anonlinres);
   * //data_zmfree( &L );
   * //data_zmfree( &U );
   * fflush(stdout);
   *
   * // diff mkl and PariLU
   * //data_d_matrix Ldiff = {Magma_CSR};
   * //dataType Lres = 0.0;
   * //dataType Lnonlinres = 0.0;
   * Lres = 0.0;
   * Lnonlinres = 0.0;
   * data_zdiff_csr(&Lmkl, &L, &Ldiff, &Lres, &Lnonlinres );
   * //data_zwrite_csr( &Ldiff );
   * printf("L_res = %e\n", Lres);
   * printf("L_nonlinres = %e\n", Lnonlinres);
   * fflush(stdout);
   * //data_d_matrix Udiff = {Magma_CSR};
   * //dataType Ures = 0.0;
   * //dataType Unonlinres = 0.0;
   * Ures = 0.0;
   * Unonlinres = 0.0;
   * data_zdiff_csr(&Umkl, &U, &Udiff, &Ures, &Unonlinres );
   * //data_zwrite_csr( &Udiff );
   * printf("U_res = %e\n", Ures);
   * printf("U_nonlinres = %e\n", Unonlinres);
   * fflush(stdout);
   *
   * data_zmfree( &Ldiff );
   * data_zmfree( &Udiff );
   *
   * data_zmfree( &L );
   * data_zmfree( &U );
   * data_zmfree( &LU );
   *
   * // =========================================================================
   */

  /*
   * data_zmfree( &A );
   * data_zmconvert( B, &A, Magma_CSR, Magma_CSR );  // reset A
   * // =========================================================================
   * // PariLU v3.5
   * // =========================================================================
   * //
   * // Separate the strictly lower, strictly upper, and diagonal elements
   * // into L, U, and D respectively.
   * // PariLU with sparse matrix-matrix products and a tiled access pattern limited to active tiles
   * printf("%% PariLU v3.5\n");
   * fflush(stdout);
   * //data_d_matrix L = {Magma_CSRL};
   * L = {Magma_CSRL};
   * //data_d_matrix U = {Magma_CSRU};
   * U = {Magma_CSRU};
   * //int tile = 256;
   * // PariLU with dgemm operations replacing summations
   * data_PariLU_v3_5( &A, &L, &U, tile);
   * // Check ||A-LU||_Frobenius
   * //data_d_matrix LU = {Magma_CSR};
   * //dataType Ares = 0.0;
   * //dataType Anonlinres = 0.0;
   * Ares = 0.0;
   * Anonlinres = 0.0;
   * data_zilures(A, L, U, &LU, &Ares, &Anonlinres);
   * printf("PariLUv3_5_csrilu0_res = %e\n", Ares);
   * printf("PariLUv3_5_csrilu0_nonlinres = %e\n", Anonlinres);
   * //data_zmfree( &L );
   * //data_zmfree( &U );
   * fflush(stdout);
   *
   * // diff mkl and PariLU
   * //data_d_matrix Ldiff = {Magma_CSR};
   * //dataType Lres = 0.0;
   * //dataType Lnonlinres = 0.0;
   * Lres = 0.0;
   * Lnonlinres = 0.0;
   * data_zdiff_csr(&Lmkl, &L, &Ldiff, &Lres, &Lnonlinres );
   * //data_zwrite_csr( &Ldiff );
   * printf("L_res = %e\n", Lres);
   * printf("L_nonlinres = %e\n", Lnonlinres);
   * fflush(stdout);
   * //data_d_matrix Udiff = {Magma_CSR};
   * //dataType Ures = 0.0;
   * //dataType Unonlinres = 0.0;
   * Ures = 0.0;
   * Unonlinres = 0.0;
   * data_zdiff_csr(&Umkl, &U, &Udiff, &Ures, &Unonlinres );
   * //data_zwrite_csr( &Udiff );
   * printf("U_res = %e\n", Ures);
   * printf("U_nonlinres = %e\n", Unonlinres);
   * fflush(stdout);
   * strcpy( output_L, output_basename );
   * strcat( output_L, "_LpariLUv3_5.mtx" );
   * strcpy( output_U, output_basename );
   * strcat( output_U, "_UpariLUv3_5.mtx" );
   * data_zwrite_csr_mtx( L, L.major, output_L );
   * data_zwrite_csr_mtx( U, U.major, output_U );
   * strcpy( output_L, output_basename );
   * strcat( output_L, "_LpariLUv3_5_diff.mtx" );
   * strcpy( output_U, output_basename );
   * strcat( output_U, "_UpariLUv3_5_diff.mtx" );
   * data_zwrite_csr_mtx( Ldiff, Ldiff.major, output_L );
   * data_zwrite_csr_mtx( Udiff, Udiff.major, output_U );
   *
   * data_zmfree( &Ldiff );
   * data_zmfree( &Udiff );
   *
   * data_zmfree( &L );
   * data_zmfree( &U );
   * data_zmfree( &LU );
   */

  /*
   * // =========================================================================
   * data_zmfree( &A );
   * data_zmconvert( B, &A, Magma_CSR, Magma_CSR );  // reset A
   * // =========================================================================
   * // PariLU v3.6
   * // =========================================================================
   * //
   * // Separate the strictly lower, strictly upper, and diagonal elements
   * // into L, U, and D respectively.
   * // PariLU with sparse matrix-matrix products and a tiled access pattern limited to active tiles
   * printf("%% PariLU v3.6\n");
   * fflush(stdout);
   * //data_d_matrix L = {Magma_CSRL};
   * L = {Magma_CSRL};
   * //data_d_matrix U = {Magma_CSRU};
   * U = {Magma_CSRU};
   * //tile = 256;
   * // PariLU with dgemm operations replacing summations
   * data_PariLU_v3_6( &A, &L, &U, tile);
   * // Check ||A-LU||_Frobenius
   * //data_d_matrix LU = {Magma_CSR};
   * //dataType Ares = 0.0;
   * //dataType Anonlinres = 0.0;
   * Ares = 0.0;
   * Anonlinres = 0.0;
   * data_zilures(A, L, U, &LU, &Ares, &Anonlinres);
   * printf("PariLUv3_6_csrilu0_res = %e\n", Ares);
   * printf("PariLUv3_6_csrilu0_nonlinres = %e\n", Anonlinres);
   * //data_zmfree( &L );
   * //data_zmfree( &U );
   * fflush(stdout);
   *
   * // diff mkl and PariLU
   * //data_d_matrix Ldiff = {Magma_CSR};
   * //dataType Lres = 0.0;
   * //dataType Lnonlinres = 0.0;
   * Lres = 0.0;
   * Lnonlinres = 0.0;
   * data_zdiff_csr(&Lmkl, &L, &Ldiff, &Lres, &Lnonlinres );
   * //data_zwrite_csr( &Ldiff );
   * printf("L_res = %e\n", Lres);
   * printf("L_nonlinres = %e\n", Lnonlinres);
   * fflush(stdout);
   * //data_d_matrix Udiff = {Magma_CSR};
   * //dataType Ures = 0.0;
   * //dataType Unonlinres = 0.0;
   * Ures = 0.0;
   * Unonlinres = 0.0;
   * data_zdiff_csr(&Umkl, &U, &Udiff, &Ures, &Unonlinres );
   * //data_zwrite_csr( &Udiff );
   * printf("U_res = %e\n", Ures);
   * printf("U_nonlinres = %e\n", Unonlinres);
   * fflush(stdout);
   * strcpy( output_L, output_basename );
   * strcat( output_L, "_LpariLUv3_6.mtx" );
   * strcpy( output_U, output_basename );
   * strcat( output_U, "_UpariLUv3_6.mtx" );
   * data_zwrite_csr_mtx( L, L.major, output_L );
   * data_zwrite_csr_mtx( U, U.major, output_U );
   * strcpy( output_L, output_basename );
   * strcat( output_L, "_LpariLUv3_6_diff.mtx" );
   * strcpy( output_U, output_basename );
   * strcat( output_U, "_UpariLUv3_6_diff.mtx" );
   * data_zwrite_csr_mtx( Ldiff, Ldiff.major, output_L );
   * data_zwrite_csr_mtx( Udiff, Udiff.major, output_U );
   *
   * data_zmfree( &Ldiff );
   * data_zmfree( &Udiff );
   *
   * data_zmfree( &L );
   * data_zmfree( &U );
   * data_zmfree( &LU );
   * // =========================================================================
   *
   * data_zmfree( &A );
   * data_zmconvert( B, &A, Magma_CSR, Magma_CSR );  // reset A
   * // =========================================================================
   * // PariLU v3.7
   * // =========================================================================
   * //
   * // Separate the strictly lower, strictly upper, and diagonal elements
   * // into L, U, and D respectively.
   * // PariLU with sparse matrix-matrix products and a tiled access pattern limited to active tiles
   * printf("%% PariLU v3.7\n");
   * fflush(stdout);
   * //data_d_matrix L = {Magma_CSRL};
   * L = {Magma_CSRL};
   * //data_d_matrix U = {Magma_CSRU};
   * U = {Magma_CSRU};
   * //tile = 256;
   * // PariLU with dgemm operations replacing summations
   * data_PariLU_v3_7( &A, &L, &U, tile);
   * // Check ||A-LU||_Frobenius
   * //data_d_matrix LU = {Magma_CSR};
   * //dataType Ares = 0.0;
   * //dataType Anonlinres = 0.0;
   * Ares = 0.0;
   * Anonlinres = 0.0;
   * data_zilures(A, L, U, &LU, &Ares, &Anonlinres);
   * printf("PariLUv3_7_csrilu0_res = %e\n", Ares);
   * printf("PariLUv3_7_csrilu0_nonlinres = %e\n", Anonlinres);
   * //data_zmfree( &L );
   * //data_zmfree( &U );
   * fflush(stdout);
   *
   * // diff mkl and PariLU
   * //data_d_matrix Ldiff = {Magma_CSR};
   * //dataType Lres = 0.0;
   * //dataType Lnonlinres = 0.0;
   * Lres = 0.0;
   * Lnonlinres = 0.0;
   * data_zdiff_csr(&Lmkl, &L, &Ldiff, &Lres, &Lnonlinres );
   * //data_zwrite_csr( &Ldiff );
   * printf("L_res = %e\n", Lres);
   * printf("L_nonlinres = %e\n", Lnonlinres);
   * fflush(stdout);
   * //data_d_matrix Udiff = {Magma_CSR};
   * //dataType Ures = 0.0;
   * //dataType Unonlinres = 0.0;
   * Ures = 0.0;
   * Unonlinres = 0.0;
   * data_zdiff_csr(&Umkl, &U, &Udiff, &Ures, &Unonlinres );
   * //data_zwrite_csr( &Udiff );
   * printf("U_res = %e\n", Ures);
   * printf("U_nonlinres = %e\n", Unonlinres);
   * fflush(stdout);
   * strcpy( output_L, output_basename );
   * strcat( output_L, "_LpariLUv3_7.mtx" );
   * strcpy( output_U, output_basename );
   * strcat( output_U, "_UpariLUv3_7.mtx" );
   * data_zwrite_csr_mtx( L, L.major, output_L );
   * data_zwrite_csr_mtx( U, U.major, output_U );
   * strcpy( output_L, output_basename );
   * strcat( output_L, "_LpariLUv3_7_diff.mtx" );
   * strcpy( output_U, output_basename );
   * strcat( output_U, "_UpariLUv3_7_diff.mtx" );
   * data_zwrite_csr_mtx( Ldiff, Ldiff.major, output_L );
   * data_zwrite_csr_mtx( Udiff, Udiff.major, output_U );
   *
   * data_zmfree( &Ldiff );
   * data_zmfree( &Udiff );
   *
   * data_zmfree( &L );
   * data_zmfree( &U );
   * data_zmfree( &LU );
   * // =========================================================================
   */

  /*
   * // =========================================================================
   * data_zmfree( &A );
   * data_zmconvert( B, &A, Magma_CSR, Magma_CSR );  // reset A
   * // =========================================================================
   * // PariLU v3.8
   * // =========================================================================
   * //
   * // Separate the strictly lower, strictly upper, and diagonal elements
   * // into L, U, and D respectively.
   * // PariLU with sparse matrix-matrix products and a tiled access pattern limited to active tiles
   * printf("%% PariLU v3.8\n");
   * fflush(stdout);
   * //data_d_matrix L = {Magma_CSRL};
   * L = {Magma_CSRL};
   * //data_d_matrix U = {Magma_CSRU};
   * U = {Magma_CSRU};
   * //tile = 256;
   * // PariLU with dgemm operations replacing summations
   * data_PariLU_v3_8( &A, &L, &U, tile);
   * // Check ||A-LU||_Frobenius
   * //data_d_matrix LU = {Magma_CSR};
   * //dataType Ares = 0.0;
   * //dataType Anonlinres = 0.0;
   * Ares = 0.0;
   * Anonlinres = 0.0;
   * data_zilures(A, L, U, &LU, &Ares, &Anonlinres);
   * printf("PariLUv3_8_csrilu0_res = %e\n", Ares);
   * printf("PariLUv3_8_csrilu0_nonlinres = %e\n", Anonlinres);
   * //data_zmfree( &L );
   * //data_zmfree( &U );
   * fflush(stdout);
   *
   * // diff mkl and PariLU
   * //data_d_matrix Ldiff = {Magma_CSR};
   * //dataType Lres = 0.0;
   * //dataType Lnonlinres = 0.0;
   * Lres = 0.0;
   * Lnonlinres = 0.0;
   * data_zdiff_csr(&Lmkl, &L, &Ldiff, &Lres, &Lnonlinres );
   * //data_zwrite_csr( &Ldiff );
   * printf("L_res = %e\n", Lres);
   * printf("L_nonlinres = %e\n", Lnonlinres);
   * fflush(stdout);
   * //data_d_matrix Udiff = {Magma_CSR};
   * //dataType Ures = 0.0;
   * //dataType Unonlinres = 0.0;
   * Ures = 0.0;
   * Unonlinres = 0.0;
   * data_zdiff_csr(&Umkl, &U, &Udiff, &Ures, &Unonlinres );
   * //data_zwrite_csr( &Udiff );
   * printf("U_res = %e\n", Ures);
   * printf("U_nonlinres = %e\n", Unonlinres);
   * fflush(stdout);
   * strcpy( output_L, output_basename );
   * strcat( output_L, "_LpariLUv3_8.mtx" );
   * strcpy( output_U, output_basename );
   * strcat( output_U, "_UpariLUv3_8.mtx" );
   * data_zwrite_csr_mtx( L, L.major, output_L );
   * data_zwrite_csr_mtx( U, U.major, output_U );
   * strcpy( output_L, output_basename );
   * strcat( output_L, "_LpariLUv3_8_diff.mtx" );
   * strcpy( output_U, output_basename );
   * strcat( output_U, "_UpariLUv3_8_diff.mtx" );
   * data_zwrite_csr_mtx( Ldiff, Ldiff.major, output_L );
   * data_zwrite_csr_mtx( Udiff, Udiff.major, output_U );
   *
   * data_zmfree( &Ldiff );
   * data_zmfree( &Udiff );
   *
   * data_zmfree( &L );
   * data_zmfree( &U );
   * data_zmfree( &LU );
   * // =========================================================================
   */

  /*
   * getchar();
   * // =========================================================================
   * data_zmfree( &A );
   * data_zmconvert( B, &A, Magma_CSR, Magma_CSR );  // reset A
   * // =========================================================================
   * // PariLU v3.9
   * // =========================================================================
   * //
   * // Separate the strictly lower, strictly upper, and diagonal elements
   * // into L, U, and D respectively.
   * // PariLU with sparse matrix-matrix products and a tiled access pattern limited to active tiles
   * printf("%% PariLU v3.9\n");
   * fflush(stdout);
   * //data_d_matrix L = {Magma_CSRL};
   * L = {Magma_CSRL};
   * //data_d_matrix U = {Magma_CSRU};
   * U = {Magma_CSRU};
   * //tile = 256;
   * // PariLU with dgemm operations replacing summations
   * data_PariLU_v3_9( &A, &L, &U, tile);
   * // Check ||A-LU||_Frobenius
   * //data_d_matrix LU = {Magma_CSR};
   * //dataType Ares = 0.0;
   * //dataType Anonlinres = 0.0;
   * Ares = 0.0;
   * Anonlinres = 0.0;
   * data_zilures(A, L, U, &LU, &Ares, &Anonlinres);
   * printf("PariLUv3_9_csrilu0_res = %e\n", Ares);
   * printf("PariLUv3_9_csrilu0_nonlinres = %e\n", Anonlinres);
   * fflush(stdout);
   *
   * // diff mkl and PariLU
   * //data_d_matrix Ldiff = {Magma_CSR};
   * //dataType Lres = 0.0;
   * //dataType Lnonlinres = 0.0;
   * Lres = 0.0;
   * Lnonlinres = 0.0;
   * data_zdiff_csr(&Lmkl, &L, &Ldiff, &Lres, &Lnonlinres );
   * //data_zwrite_csr( &Ldiff );
   * printf("L_res = %e\n", Lres);
   * printf("L_nonlinres = %e\n", Lnonlinres);
   * fflush(stdout);
   * //data_d_matrix Udiff = {Magma_CSR};
   * //dataType Ures = 0.0;
   * //dataType Unonlinres = 0.0;
   * Ures = 0.0;
   * Unonlinres = 0.0;
   * data_zdiff_csr(&Umkl, &U, &Udiff, &Ures, &Unonlinres );
   * //data_zwrite_csr( &Udiff );
   * printf("U_res = %e\n", Ures);
   * printf("U_nonlinres = %e\n", Unonlinres);
   * fflush(stdout);
   * strcpy( output_L, output_basename );
   * strcat( output_L, "_LpariLUv3_9.mtx" );
   * strcpy( output_U, output_basename );
   * strcat( output_U, "_UpariLUv3_9.mtx" );
   * data_zwrite_csr_mtx( L, L.major, output_L );
   * data_zwrite_csr_mtx( U, U.major, output_U );
   * strcpy( output_L, output_basename );
   * strcat( output_L, "_LpariLUv3_9_diff.mtx" );
   * strcpy( output_U, output_basename );
   * strcat( output_U, "_UpariLUv3_9_diff.mtx" );
   * data_zwrite_csr_mtx( Ldiff, Ldiff.major, output_L );
   * data_zwrite_csr_mtx( Udiff, Udiff.major, output_U );
   *
   * data_zmfree( &Ldiff );
   * data_zmfree( &Udiff );
   *
   * data_zmfree( &L );
   * data_zmfree( &U );
   * data_zmfree( &LU );
   * // =========================================================================
   *
   */

  data_zmfree(&Asparse);
  data_zmfree(&A);
  data_zmfree(&B);
  data_zmfree(&Amkl);
  data_zmfree(&Lmkl);
  data_zmfree(&Umkl);
  // testing::InitGoogleTest(&argc, argv);
  // return RUN_ALL_TESTS();
  return 0;
} // main
