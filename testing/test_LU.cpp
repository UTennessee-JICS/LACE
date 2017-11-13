
#include <stdlib.h>
#include <stdio.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <vector>
#include <omp.h>

#include "mmio.h"
#include "sparse_types.h"
#include "container_tests.h"

#include "test_cmd_line.h"

class LUTest: public
  ::testing::Test
{
protected:
  LUTest() {}

  // per-test-case set-up
  static void SetUpTestCase() {
    printf("seting up\n");
    fflush(stdout);

    printf("my_argc = %d\n", my_argc);
    for (int i=0; i< my_argc; ++i) {
      printf("my_agv[%d] = %s\n", i, my_argv[i]);
    }
    fflush(stdout);

    int dim = 1000;
    char default_matrix[] = "LARNV";
    char* matrix_name = NULL ;
    // parse command line arguments
    if (my_argc>1) {
      int count = 1;
      while (count < my_argc) {
        if ( (strcmp(my_argv[count], "--matrix") == 0)
            && count+1 < my_argc ) {
          matrix_name = my_argv[count+1];
          count = count + 2;
        }
        else if ( (strcmp(my_argv[count], "--dim") == 0)
            && count+1 < my_argc ) {
          dim = atoi(my_argv[count+1]);
          count = count + 2;
        }
        else {
          count++;
        }
      }
    }

    // generate or load A matrix
    if (matrix_name == NULL ) {
      matrix_name = default_matrix;
    }
    if ( (strcmp(matrix_name, "LARNV") == 0) ) {
      printf("A will be a generated %d x %d LARNV matrix\n", dim, dim);
      fflush(stdout);
      A = new data_d_matrix;
      CHECK( data_zvinit( A, dim, dim, 0.0, MagmaRowMajor ) );

      int ione = 1;
      int ISEED[4] = {0,0,0,1};
      CHECK( LAPACKE_dlarnv( ione, ISEED, A->nnz, A->val ) );
      for ( int i = 0; i<A->num_rows; i++ ) {
        A->val[ i*A->ld + i ] += 1.0e3;
      }
    }
    else {
      printf("A will be read from %s and converted to a dense matrix\n", matrix_name);
      data_d_matrix Asparse = {Magma_CSR};
      CHECK( data_z_csr_mtx( &Asparse, matrix_name ) );
      A = new data_d_matrix;
      data_zmconvert( Asparse, A, Asparse.storage_type, Magma_DENSE );
      data_zmfree( &Asparse );
    }

    //data_zdisplay_dense( &A );
    //data_d_matrix B = {Magma_DENSE};
    //data_zmconvert( A, &B, Magma_DENSE, Magma_DENSE );

    // =========================================================================
    // MKL LU with no pivoting (Benchmark)
    // =========================================================================
    printf("%% MKL LU with no pivoting (Benchmark)\n");
    data_d_matrix Amkl = {Magma_DENSE};
    data_zmconvert((*A), &Amkl, Magma_DENSE, Magma_DENSE);

    dataType wstart = omp_get_wtime();
    data_LUnp_mkl( &Amkl );
    dataType wend = omp_get_wtime();
    printf("%% MKL LU with no pivoting required %f wall clock seconds as measured by omp_get_wtime()\n", wend-wstart );

    Amkldiff = new dataType;
    (*Amkldiff) = 0.0;
    // Check ||A-LU||_Frobenius
    data_zfrobenius_inplaceLUresidual((*A), Amkl, Amkldiff);
    printf("MKL_LUnp_res = %e\n", (*Amkldiff));
    data_zmfree( &Amkl );
    fflush(stdout);
  }

  // per-test-case tear-down
  static void TearDownTestCase() {
    data_zmfree( A );
    free( Amkldiff );
  }

  // per-test set-up and tear-down
  virtual void SetUp() {}
  virtual void TearDown() {}

  // shared by all tests
  static data_d_matrix* A;// = {Magma_DENSE};
  static dataType* Amkldiff;
};

data_d_matrix* LUTest::A = NULL;
dataType* LUTest::Amkldiff = NULL;

TEST_F(LUTest, ParLUv0_0) {
  // =========================================================================
  // ParLU v0.0
  // =========================================================================
  printf("%% ParLU v0.0\n");
  data_d_matrix L = {Magma_DENSEL};
  data_d_matrix U = {Magma_DENSEU};
  dataType Adiff = 0.0;
  // Separate the strictly lower and upper elements
  // into L, and U respectively.
  L = {Magma_DENSEL};
  U = {Magma_DENSEU};
  data_ParLU_v0_0( A, &L, &U);
  // Check ||A-LU||_Frobenius
  data_zfrobenius_LUresidual((*A), L, U, &Adiff);
  printf("ParLUv0_0_res = %e\n", Adiff);
  fflush(stdout);

  EXPECT_LE( Adiff, (*Amkldiff)*10.0 );

  data_zmfree( &L );
  data_zmfree( &U );
  // =========================================================================
}

TEST_F(LUTest, ParLUv0_1) {
  // =========================================================================
  // ParLU v0.1
  // =========================================================================
  printf("%% ParLU v0.1\n");
  data_d_matrix L = {Magma_DENSEL};
  data_d_matrix U = {Magma_DENSEU};
  dataType Adiff = 0.0;
  // Separate the strictly lower and upper, elements
  // into L, U respectively.
  // Convert U to column major storage.
  printf("%% ParLU v0.1\n");
  L = {Magma_DENSEL};
  U = {Magma_DENSEU};
  data_ParLU_v0_1( A, &L, &U);
  // Check ||A-LU||_Frobenius
  data_zfrobenius_LUresidual((*A), L, U, &Adiff);
  printf("ParLUv0_1_res = %e\n", Adiff);
  fflush(stdout);

  EXPECT_LE( Adiff, (*Amkldiff)*10.0 );

  data_zmfree( &L );
  data_zmfree( &U );
  // =========================================================================
}

TEST_F(LUTest, ParLUv1_0) {
  // =========================================================================
  // ParLU v1.0
  // =========================================================================
  printf("%% ParLU v1.0\n");
  data_d_matrix L = {Magma_DENSEL};
  data_d_matrix U = {Magma_DENSEU};
  dataType Adiff = 0.0;
  // Separate the strictly lower, upper elements
  // into L and U respectively.
  printf("%% ParLU v1.0\n");
  L = {Magma_DENSEL};
  U = {Magma_DENSEU};
  // ParLU with dot products replacing summations
  data_ParLU_v1_0( A, &L, &U);
  // Check ||A-LU||_Frobenius
  data_zfrobenius_LUresidual((*A), L, U, &Adiff);
  printf("ParLUv1_0_res = %e\n", Adiff);
  fflush(stdout);

  EXPECT_LE( Adiff, (*Amkldiff)*10.0 );

  data_zmfree( &L );
  data_zmfree( &U );
  // =========================================================================
}
