#include <stdio.h>

#include "mmio.h"
// #include "sparse.h"
// #include "container_tests.h"

int
main(int argc, char * argv[])
{
  printf("C program using LACE.\n\n");
  printf("argc = %d\n", argc);
  for (int i = 0; i < argc; ++i) {
    printf("agv[%d] = %s\n", i, argv[i]);
  }
  fflush(stdout);

  char default_matrix[] = "matrices/Trefethen_20.mtx";
  char * matrix_name    = NULL;
  int * tile_size       = (int *) malloc(sizeof(int));
  (*tile_size) = 8;

  // parse command line arguments
  if (argc > 1) {
    int count = 1;
    while (count < argc) {
      if ( (strcmp(argv[count], "--matrix") == 0) &&
        count + 1 < argc)
      {
        matrix_name = argv[count + 1];
        count       = count + 2;
      } else if ( (strcmp(argv[count], "--tile") == 0) &&
        count + 1 < argc)
      {
        (*tile_size) = atoi(argv[count + 1]);
        count        = count + 2;
      } else {
        count++;
      }
    }
  }

  // load A matrix
  if (matrix_name == NULL) {
    matrix_name = default_matrix;
  }
  printf("A will be read from %s\n", matrix_name);
  data_d_matrix A = { Magma_CSR };
  CHECK(data_z_csr_mtx((&A), matrix_name) );

  // ===================================================================
  // MKL csrilu0  (Benchmark)
  // ===================================================================
  printf("%% MKL csrilu0  (Benchmark)\n");
  data_d_matrix Amkl = { Magma_CSR };
  data_zmconvert(A, &Amkl, Magma_CSR, Magma_CSR);

  dataType wstart = omp_get_wtime();
  CHECK(data_dcsrilu0_mkl(&Amkl) );
  dataType wend = omp_get_wtime();
  printf("%% MKL csrilu0 required %f wall clock seconds as measured by omp_get_wtime()\n", wend - wstart);

  data_d_matrix Lmkl = { Magma_CSRL };
  Lmkl.storage_type   = Magma_CSRL;
  Lmkl.diagorder_type = Magma_UNITY;
  data_zmconvert(Amkl, (&Lmkl), Magma_CSR, Magma_CSRL);
  printf("test if Lmkl is lower: ");
  data_zcheckupperlower(&Lmkl);
  printf(" done.\n");

  data_d_matrix Umkl = { Magma_CSRU };
  Umkl.storage_type   = Magma_CSRU;
  Umkl.diagorder_type = Magma_VALUE;
  data_zmconvert(Amkl, (&Umkl), Magma_CSR, Magma_CSRU);
  printf("test if Umkl is upper: ");
  data_zcheckupperlower(&Umkl);
  printf(" done.\n");
  data_d_matrix LUmkl = { Magma_CSR };
  data_zmconvert(Amkl, &LUmkl, Magma_CSR, Magma_CSR);

  dataType Amklres       = 0.0;
  dataType Amklnonlinres = 0.0;
  // Check ||A-LU||_Frobenius for the whole and restricted to A's sparsity pattern
  data_zilures(A, Lmkl, Umkl, (&LUmkl), (&Amklres), (&Amklnonlinres));
  printf("MKL_csrilu0_res = %e\n", Amklres);
  printf("MKL_csrilu0_nonlinres = %e\n", Amklnonlinres);
  data_zmfree(&Amkl);
  data_zmfree(&LUmkl);
  fflush(stdout);

  // =========================================================================
  // PariLU v0.3
  // ====================================================================+======
  printf("%% PariLU v0.3\n");

  data_d_matrix L    = { Magma_CSRL };
  data_d_matrix U    = { Magma_CSRU };
  dataType reduction = 1.0e-15;
  data_d_preconditioner_log parilu_log;
  // Separate the strictly lower and upper elements
  // into L, and U respectively.
  data_PariLU_v0_3(&A, &L, &U, reduction, &parilu_log);
  // Check ||A-LU||_Frobenius
  dataType Ares       = 0.0;
  dataType Anonlinres = 0.0;
  data_d_matrix LU    = { Magma_CSR };
  data_zmconvert(A, &LU, Magma_CSR, Magma_CSR);
  data_zilures(A, L, U, &LU, &Ares, &Anonlinres);
  printf("PariLU_v0_3_omp_num_threads = %d\n", parilu_log.omp_num_threads)
  ;
  printf("PariLU_v0_3_sweeps = %d\n", parilu_log.sweeps);
  printf("PariLU_v0_3_tol = %e\n", parilu_log.tol);
  printf("PariLU_v0_3_A_Frobenius = %e\n", parilu_log.A_Frobenius);
  printf("PariLU_v0_3_generation_time = %e\n", parilu_log.precond_generation_time);
  printf("PariLU_v0_3_initial_residual = %e\n", parilu_log.initial_residual);
  printf("PariLU_v0_3_initial_nonlinear_residual = %e\n", parilu_log.initial_nonlinear_residual);
  printf("PariLU_v0_3_csrilu0_res = %e\n", Ares);
  printf("PariLU_v0_3_csrilu0_nonlinres = %e\n", Anonlinres);

  fflush(stdout);

  data_zmfree(&L);
  data_zmfree(&U);
  data_zmfree(&LU);


  data_zmfree(&A);
} /* main */
