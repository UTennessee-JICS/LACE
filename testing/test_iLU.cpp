
#include <stdlib.h>
#include <stdio.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <vector>
#include <omp.h>

#include "mmio.h"
#include "sparse.h"
#include "container_tests.h"

#include "test_cmd_line.h"

class iLUTest: public
  ::testing::Test
{
protected:
  iLUTest() {}

  // per-test-case set-up
  static void SetUpTestCase() {
    printf("seting up\n");
    fflush(stdout);

    printf("my_argc = %d\n", my_argc);
    for (int i=0; i< my_argc; ++i) {
      printf("my_agv[%d] = %s\n", i, my_argv[i]);
    }
    fflush(stdout);

    char default_matrix[] = "matrices/Trefethen_20.mtx";
    char* matrix_name = NULL;
    tile_size = new int();
    (*tile_size) = 8;

    // parse command line arguments
    if (my_argc>1) {
      int count = 1;
      while (count < my_argc) {
        if ( (strcmp(my_argv[count], "--matrix") == 0)
            && count+1 < my_argc ) {
          matrix_name = my_argv[count+1];
          count = count + 2;
        }
        else if ( (strcmp(my_argv[count], "--tile") == 0)
            && count+1 < my_argc ) {
          (*tile_size) = atoi(my_argv[count+1]);
          count = count + 2;
        }
        else {
          count++;
        }
      }
    }

    // load A matrix
    if (matrix_name == NULL ) {
      matrix_name = default_matrix;
    }
    printf("A will be read from %s\n", matrix_name);
    A = new data_d_matrix();
    A->storage_type = Magma_CSR;
    CHECK( data_z_csr_mtx( A, matrix_name ) );

    // =========================================================================
    // MKL csrilu0  (Benchmark)
    // =========================================================================
    printf("%% MKL csrilu0  (Benchmark)\n");
    data_d_matrix Amkl = {Magma_CSR};
    data_zmconvert((*A), &Amkl, Magma_CSR, Magma_CSR);

    dataType wstart = omp_get_wtime();
    CHECK( data_dcsrilu0_mkl( &Amkl ) );
    dataType wend = omp_get_wtime();
    printf("%% MKL csrilu0 required %f wall clock seconds as measured by omp_get_wtime()\n", wend-wstart );

    //data_d_matrix Lmkl = {Magma_CSRL};
    Lmkl = new data_d_matrix();
    Lmkl->storage_type = Magma_CSRL;
    Lmkl->diagorder_type = Magma_UNITY;
    data_zmconvert(Amkl, Lmkl, Magma_CSR, Magma_CSRL);
    printf("test if Lmkl is lower: ");
    data_zcheckupperlower( Lmkl );
    printf(" done.\n");
    //data_d_matrix Umkl = {Magma_CSRU};
    Umkl = new data_d_matrix();
    Umkl->storage_type = Magma_CSRU;
    Umkl->diagorder_type = Magma_VALUE;
    data_zmconvert(Amkl, Umkl, Magma_CSR, Magma_CSRU);
    printf("test if Umkl is upper: ");
    data_zcheckupperlower( Umkl );
    printf(" done.\n");
    data_d_matrix LUmkl = {Magma_CSR};
    data_zmconvert(Amkl, &LUmkl, Magma_CSR, Magma_CSR);

    Amklres = new dataType();
    (*Amklres) = 0.0;
    Amklnonlinres = new dataType();
    (*Amklnonlinres) = 0.0;
    // Check ||A-LU||_Frobenius for the whole and restricted to A's sparsity pattern
    data_zilures( (*A), (*Lmkl), (*Umkl), &LUmkl, Amklres, Amklnonlinres);
    printf("MKL_csrilu0_res = %e\n", (*Amklres));
    printf("MKL_csrilu0_nonlinres = %e\n", (*Amklnonlinres));
    data_zmfree( &Amkl );
    data_zmfree( &LUmkl );
    fflush(stdout);
  }

  // per-test-case tear-down
  static void TearDownTestCase() {
    data_zmfree( A );
    data_zmfree( Lmkl );
    data_zmfree( Umkl );
    delete A;
    delete Lmkl;
    delete Umkl;
    delete Amklres;
    delete Amklnonlinres;
    if (tile_size != NULL ) {
      delete tile_size;
    }
  }

  // per-test set-up and tear-down
  virtual void SetUp() {}
  virtual void TearDown() {}

  // shared by all tests
  static data_d_matrix* A;
  static data_d_matrix* Lmkl;
  static data_d_matrix* Umkl;
  static dataType* Amklres;
  static dataType* Amklnonlinres;
  static int* tile_size;
};

data_d_matrix* iLUTest::A = NULL;
data_d_matrix* iLUTest::Lmkl = NULL;
data_d_matrix* iLUTest::Umkl = NULL;
dataType* iLUTest::Amklres = NULL;
dataType* iLUTest::Amklnonlinres = NULL;
int* iLUTest::tile_size = NULL;

TEST_F(iLUTest, PariLUv0_0) {
  // =========================================================================
  // PariLU v0.0
  // =========================================================================
  printf("%% PariLU v0.0\n");

  data_d_matrix L = {Magma_CSRL};
  data_d_matrix U = {Magma_CSRU};
  // Separate the strictly lower and upper elements
  // into L, and U respectively.
  data_PariLU_v0_0( A, &L, &U);
  // Check ||A-LU||_Frobenius
  dataType Ares = 0.0;
  dataType Anonlinres = 0.0;
  data_d_matrix LU = {Magma_CSR};
  data_zmconvert((*A), &LU, Magma_CSR, Magma_CSR);
  data_zilures((*A), L, U, &LU, &Ares, &Anonlinres);
  printf("PariLUv0_0-5_csrilu0_res = %e\n", Ares);
  printf("PariLUv0_0-5_csrilu0_nonlinres = %e\n", Anonlinres);

  fflush(stdout);

  EXPECT_LE( Ares, (*Amklres)*10.0 );
  EXPECT_LE( Anonlinres, (*Amklnonlinres)*10.0 );

  data_zmfree( &L );
  data_zmfree( &U );
  data_zmfree( &LU );
  // =========================================================================
}

TEST_F(iLUTest, PariLUv0_3) {
  // =========================================================================
  // PariLU v0.0
  // =========================================================================
  printf("%% PariLU v0.3\n");

  data_d_matrix L = {Magma_CSRL};
  data_d_matrix U = {Magma_CSRU};
  dataType reduction = 1.0e-15;
  data_d_preconditioner_log parilu_log;
  // Separate the strictly lower and upper elements
  // into L, and U respectively.
  data_PariLU_v0_3( A, &L, &U, reduction, &parilu_log );
  // Check ||A-LU||_Frobenius
  dataType Ares = 0.0;
  dataType Anonlinres = 0.0;
  data_d_matrix LU = {Magma_CSR};
  data_zmconvert((*A), &LU, Magma_CSR, Magma_CSR);
  data_zilures((*A), L, U, &LU, &Ares, &Anonlinres);
  printf("PariLU_v0_3_omp_num_threads = %d\n", parilu_log.omp_num_threads );
  printf("PariLU_v0_3_sweeps = %d\n", parilu_log.sweeps );
  printf("PariLU_v0_3_tol = %e\n", parilu_log.tol );
  printf("PariLU_v0_3_A_Frobenius = %e\n", parilu_log.A_Frobenius );
  printf("PariLU_v0_3_generation_time = %e\n", parilu_log.precond_generation_time );
  printf("PariLU_v0_3_initial_residual = %e\n", parilu_log.initial_residual );
  printf("PariLU_v0_3_initial_nonlinear_residual = %e\n", parilu_log.initial_nonlinear_residual );
  printf("PariLUv0_3_csrilu0_res = %e\n", Ares);
  printf("PariLUv0_3_csrilu0_nonlinres = %e\n", Anonlinres);

  fflush(stdout);

  EXPECT_LE( Ares, (*Amklres)*10.0 );
  EXPECT_LE( Anonlinres, (*Amklnonlinres)*10.0 );

  data_zmfree( &L );
  data_zmfree( &U );
  data_zmfree( &LU );
  // =========================================================================
}
