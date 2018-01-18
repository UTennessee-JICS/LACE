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

  if (argc < 5) {
    printf("Usage %s --matrix <name> --outDir <name>\n", argv[0]);
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
  // MKL csrilu0  (Benchmark)
  // =========================================================================
  printf("%% MKL csrilu0  (Benchmark)\n");


  data_d_matrix A = { Magma_CSR };
  data_zmconvert(Asparse, &A, Magma_CSR, Magma_CSR);
  dataType wstart = omp_get_wtime();
  CHECK(data_dcsrilu0_mkl(&A) );
  dataType wend = omp_get_wtime();
  printf("%% MKL csrilu0 required %f wall clock seconds as measured by omp_get_wtime()\n", wend - wstart
  );

  data_d_matrix Lmkl = { Magma_CSRL };
  Lmkl.storage_type   = Magma_CSRL;
  Lmkl.diagorder_type = Magma_UNITY;
  data_zmconvert(A, &Lmkl, Magma_CSR, Magma_CSRL);
  printf("test if Lmkl is lower: ");
  data_zcheckupperlower(&Lmkl);
  printf(" done.\n");
  data_d_matrix Umkl = { Magma_CSRU };
  Umkl.storage_type   = Magma_CSRU;
  Umkl.diagorder_type = Magma_VALUE;
  data_zmconvert(A, &Umkl, Magma_CSR, Magma_CSRU);
  printf("test if Umkl is upper: ");
  data_zcheckupperlower(&Umkl);
  printf(" done.\n");
  data_d_matrix LUmkl = { Magma_CSR };
  data_zmconvert(A, &LUmkl, Magma_CSR, Magma_CSR);

  dataType Amklres       = 0.0;
  dataType Amklnonlinres = 0.0;
  // Check ||A-LU||_Frobenius for the whole and restricted to A's sparsity pattern
  data_zilures(Asparse, Lmkl, Umkl, &LUmkl, &Amklres, &Amklnonlinres);
  printf("MKL_csrilu0_res = %e\n", Amklres);
  printf("MKL_csrilu0_nonlinres = %e\n", Amklnonlinres);


  std::string s1(matrix_name);
  sparse_basename = basename(matrix_name);
  char * ext;
  ext = strrchr(sparse_basename, '.');
  strncpy(sparse_name, sparse_basename, size_t(ext - sparse_basename) );
  sparse_name[ size_t(ext - sparse_basename) ] = '\0';

  strcpy(output_basename, output_dir);
  strcat(output_basename, "/");
  strcat(output_basename, sparse_name);

  strcpy(output_L, output_basename);
  char suffixBuffer[256];
  sprintf(suffixBuffer, "_LGE.mtx");
  strcat(output_L, suffixBuffer);
  strcpy(output_U, output_basename);
  sprintf(suffixBuffer, "_UGE.mtx");
  strcat(output_U, suffixBuffer);
  data_zwrite_csr_mtx(Lmkl, Lmkl.major, output_L);
  // data_d_matrix Ucsr = { Magma_CSRU };
  // CHECK(data_zmconvert(Umkl, &Ucsr, Magma_CSC, Magma_CSR) );
  // Ucsr.storage_type = Magma_CSRU;
  // Ucsr.fill_mode    = MagmaUpper;
  // data_zwrite_csr_mtx(Ucsr, Ucsr.major, output_U);
  data_zwrite_csr_mtx(Umkl, Umkl.major, output_U);

  data_zmfree(&Lmkl);
  data_zmfree(&Umkl);
  // data_zmfree(&Ucsr);
  data_zmfree(&LUmkl);
  data_zmfree(&A);
  data_zmfree(&Asparse);

  return 0;
} // main

// main
