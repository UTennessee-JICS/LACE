
#include <stdlib.h>
#include <stdio.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <vector>
#include <omp.h>

#include "mmio.h"
#include "sparse_types.h"
#include "container_tests.h"

TEST(perform_LU, larnv10) {

  int dim = 10;
  data_d_matrix A = {Magma_DENSE};
  A.num_rows = dim;
  A.num_cols = dim;
  A.nnz = dim*dim;
  A.true_nnz = A.nnz;
  A.ld = dim;
  A.major = MagmaRowMajor;
  int ione = 1;
  int ISEED[4] = {0,0,0,1};
  A.val = (dataType*) calloc( A.nnz, sizeof(dataType) );
  CHECK( LAPACKE_dlarnv( ione, ISEED, A.nnz, A.val ) );
  for ( int i = 0; i<A.num_rows; i++ ) {
    for ( int j = 0; j<A.num_cols; j++ ) {
      if (i == j) {
        A.val[ i*A.ld + j ] += 1.0e3;
      }
    }
  }

  //data_zdisplay_dense( &A );
  //data_d_matrix B = {Magma_DENSE};
  //data_zmconvert( A, &B, Magma_DENSE, Magma_DENSE );

  // =========================================================================
  // MKL LU with no pivoting (Benchmark)
  // =========================================================================
  printf("%% MKL LU with no pivoting (Benchmark)\n");
  data_d_matrix Amkl = {Magma_DENSE};
  data_zmconvert(A, &Amkl, Magma_DENSE, Magma_DENSE);

  dataType wstart = omp_get_wtime();
  data_LUnp_mkl( &Amkl );
  dataType wend = omp_get_wtime();
  printf("%% MKL LU with no pivoting required %f wall clock seconds as measured by omp_get_wtime()\n", wend-wstart );

  dataType Amkldiff = 0.0;
  // Check ||A-LU||_Frobenius
  data_zfrobenius_inplaceLUresidual(A, Amkl, &Amkldiff);
  printf("MKL_LUnp_res = %e\n", Amkldiff);
  data_zmfree( &Amkl );
  fflush(stdout);


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
  data_ParLU_v0_0( &A, &L, &U);
  // Check ||A-LU||_Frobenius
  data_zfrobenius_LUresidual(A, L, U, &Adiff);
  printf("ParLUv0_0_res = %e\n", Adiff);
  fflush(stdout);

  EXPECT_LE( Adiff, Amkldiff*10.0 );

  data_zmfree( &A );
  data_zmfree( &L );
  data_zmfree( &U );
}
