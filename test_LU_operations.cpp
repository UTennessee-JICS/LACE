/*
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <vector>
#include <omp.h>

#include "include/mmio.h"
#include "include/sparse_types.h"

int
main(int argc, char * argv[])
{
  char dense_filename[] = "testing/matrices/io_dense_test.mtx";

  data_d_matrix A = { Magma_DENSE };

  if (A.major != MagmaRowMajor && A.major != MagmaColMajor)
    printf("A.major = %d : it has not been set\n", A.major);
  else
    printf("A.major = %d\n", A.major);
  A.major = MagmaRowMajor;
  data_z_dense_mtx(&A, MagmaRowMajor, dense_filename);
  // data_zprint_dense( A );

  data_d_matrix B = { Magma_DENSE };
  data_z_dense_mtx(&B, MagmaRowMajor, dense_filename);
  // data_zprint_dense( B );

  dataType dotanswer = 0;
  dotanswer = data_zdot(3, A.val, 1, B.val, 1);
  for (int i = 0; i < 3; i++) {
    printf("%e %e\n", A.val[i], B.val[i]);
  }
  printf("dotanswer = %e\n", dotanswer);

  dataType alpha = 1.0;
  dataType beta  = 0.0;
  dataType y[7];
  printf("A.ld=%d\n", A.ld);
  data_dgemv_mkl(A.major, MagmaNoTrans, 7, 5, alpha, A.val, A.ld,
    B.val, 1, beta, y, 1);
  for (int i = 0; i < 7; i++) {
    printf("%e\n", y[i]);
  }

  // TODO: write a create matrix method that initializes structure
  data_d_matrix C = { Magma_DENSE };
  C.num_rows = 7;
  C.num_cols = 7;
  C.ld       = 7;
  C.nnz      = 49;
  C.val      = (dataType *) malloc(C.num_rows * C.num_cols * sizeof(dataType) );


  printf("\nfull matrices\nA.ld=%d, B.ld=%d, C.ld=%d\n", A.ld, B.ld, C.ld);
  data_dgemm_mkl(A.major, MagmaNoTrans, MagmaTrans, 7, 7, 5,
    alpha, A.val, A.ld,
    B.val, B.ld, beta, C.val, C.ld);
  for (int i = 0; i < 5; i++) {
    printf("%e\n", C.val[i]);
  }

  data_d_matrix D = { Magma_DENSE };
  D.num_rows = 3;
  D.num_cols = 3;
  D.ld       = 3;
  D.val      = (dataType *) malloc(D.num_rows * D.num_cols * sizeof(dataType) );


  printf("\nsub matrices 1\nA.ld=%d, B.ld=%d, C.ld=%d\n", A.ld, B.ld, D.ld);
  data_dgemm_mkl(A.major, MagmaNoTrans, MagmaTrans, 3, 3, 3,
    alpha, A.val, A.ld,
    B.val, B.ld, beta, D.val, D.ld);
  for (int i = 0; i < 5; i++) {
    printf("%e\n", D.val[i]);
  }

  printf("\nsub matrices 2\nA.ld=%d, B.ld=%d, C.ld=%d\n", A.ld, B.ld, D.ld);
  data_dgemm_mkl(A.major, MagmaNoTrans, MagmaTrans, 3, 3, 3,
    alpha, &A.val[2], A.ld,
    &B.val[2], B.ld, beta, D.val, D.ld);
  for (int i = 0; i < 5; i++) {
    printf("%e\n", D.val[i]);
  }

  printf("\nsub matrices 3\nA.ld=%d, B.ld=%d, C.ld=%d\n", A.ld, B.ld, D.ld);
  data_dgemm_mkl(A.major, MagmaNoTrans, MagmaTrans, 3, 3, 3,
    alpha, &A.val[2 * A.ld], A.ld,
    &B.val[2 * B.ld], B.ld, beta, D.val, D.ld);
  for (int i = 0; i < 9; i++) {
    printf("%e\n", D.val[i]);
  }

  printf("\nsub matrices 4\nA.ld=%d, B.ld=%d, C.ld=%d\n", A.ld, B.ld, D.ld);
  data_dgemm_mkl(A.major, MagmaNoTrans, MagmaTrans, 3, 3, 3,
    alpha, &A.val[2 * A.ld + 2], A.ld,
    &B.val[2 * B.ld + 2], B.ld, beta, D.val, D.ld);
  for (int i = 0; i < 9; i++) {
    printf("%e\n", D.val[i]);
  }


  data_zmfree(&B);
  data_zmconvert(A, &B, Magma_DENSE, Magma_DENSE);
  // data_zdisplay_dense( &B );
  data_zmfree(&B);

  B.diagorder_type = Magma_NODIAG;
  data_zmconvert(A, &B, Magma_DENSE, Magma_DENSEL);
  // data_zdisplay_dense( &B );
  data_zmfree(&B);

  B.diagorder_type = Magma_VALUE;
  data_zmconvert(A, &B, Magma_DENSE, Magma_DENSEU);
  // data_zdisplay_dense( &B );

  printf("\ntranpose U\n");
  data_zmfree(&C);
  data_zmtranspose(B, &C);
  // data_zdisplay_dense( &C );

  printf("\nextract diagonal\n");
  data_d_matrix E = { Magma_DENSE };
  data_zmconvert(A, &E, Magma_DENSE, Magma_DENSED);
  // data_zprint_dense( E );

  data_zmfree(&A);
  data_zmfree(&B);
  data_zmfree(&C);
  data_zmfree(&D);
  data_zmfree(&E);

  // testing::InitGoogleTest(&argc, argv);
  // return RUN_ALL_TESTS();
  return 0;
} // main
