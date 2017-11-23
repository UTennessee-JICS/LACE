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
  data_z_csr_mtx(&Asparse, sparse_filename);
  data_d_matrix A = { Magma_CSR };
  data_zmconvert(Asparse, &A, Magma_CSR, Magma_CSR);
  data_d_matrix B = { Magma_DENSE };
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
  // data_LUnp_mkl( &Amkl );
  data_dcsrilu0_mkl(&Amkl);
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
  // data_zfrobenius_inplaceLUresidual(A, Amkl, &Amkldiff);
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
