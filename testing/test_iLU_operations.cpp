/*

*/

#include <stdlib.h>
#include <stdio.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <vector>
#include <omp.h>

#include "mmio.h"
#include "sparse_types.h"

#define EXPECT_ARRAY_INT_EQ( length, ref, target) \
{ \
  int i = 0; \
  for(i=0; i<length; i++) { \
    EXPECT_EQ(ref[i], target[i]) \
    << "Arrays " #ref  " and " #target \
           "differ at index " << i; \
  } \
}

#define EXPECT_ARRAY_DOUBLE_EQ( length, ref, target) \
{ \
  int i = 0; \
  for(i=0; i<length; i++) { \
    EXPECT_DOUBLE_EQ(ref[i], target[i]) \
    << "Arrays " #ref  " and " #target \
           "differ at index " << i; \
  } \
}

TEST(Operations_sparse, densesubmatrices) {
  printf("\n\nBEGIN Operations_sparse densesubmatrices\n\n");
  char sparse_filename2[] = "matrices/Trefethen_20.mtx";
  data_d_matrix Asparse = {Magma_CSR};
  data_z_csr_mtx( &Asparse, sparse_filename2 );
  data_d_matrix A2 = {Magma_CSR};
  data_zmconvert( Asparse, &A2, Magma_CSR, Magma_CSR );
  data_d_matrix B2 = {Magma_CSR};
  data_zmconvert( Asparse, &B2, Magma_CSR, Magma_CSR );

  dataType sub1_valcheck[25] = {
    3.000000e+00, 1.000000e+00, 1.000000e+00, 0.000000e+00, 1.000000e+00,
    1.000000e+00, 5.000000e+00, 1.000000e+00, 1.000000e+00, 0.000000e+00,
    1.000000e+00, 1.000000e+00, 7.000000e+00, 1.000000e+00, 1.000000e+00,
    0.000000e+00, 1.000000e+00, 1.000000e+00, 1.100000e+01, 1.000000e+00,
    1.000000e+00, 0.000000e+00, 1.000000e+00, 1.000000e+00, 1.300000e+01 };
  dataType sub1_L_valcheck[25] = {
    1,     0,     0,     0,     0,
    1,     1,     0,     0,     0,
    1,     1,     1,     0,     0,
    0,     1,     1,     1,     0,
    1,     0,     1,     1,     1 };
  dataType sub1_U_valcheck[25] = {
    3,     1,     1,     0,     1,
    0,     5,     1,     1,     0,
    0,     0,     7,     1,     1,
    0,     0,     0,    11,     1,
    0,     0,     0,     0,    13 };
  dataType sub1_T_valcheck[25] = {
    3,     1,     1,     0,     1,
    3,     6,     2,     1,     1,
    3,     6,     9,     2,     2,
    0,     5,     8,    13,     2,
    3,     1,     8,    12,    16 };

  dataType alpha = 1.0;
  dataType beta = 0.0;

  int sub_m = 5;
  int sub_n = 5;
  int sub_mbegin = 0;
  int sub_nbegin = 0;
  dataType *As;
  As = (dataType*) calloc( sub_m*sub_n, sizeof(dataType) );
  data_sparse_subdense(sub_m, sub_n, sub_mbegin, sub_nbegin, &A2, As);
  EXPECT_ARRAY_DOUBLE_EQ( sub_m*sub_n, As, sub1_valcheck);

  data_d_matrix L = {Magma_CSRL};
  L.diagorder_type = Magma_UNITY;
  data_zmconvert( Asparse, &L, Magma_CSR, Magma_CSRL );
  dataType *Ls;
  Ls = (dataType*) calloc( sub_m*sub_n, sizeof(dataType) );
  data_sparse_subdense(sub_m, sub_n, sub_mbegin, sub_nbegin, &L, Ls);
  EXPECT_ARRAY_DOUBLE_EQ( sub_m*sub_n, Ls, sub1_L_valcheck);

  data_d_matrix U = {Magma_CSRU};
  U.diagorder_type = Magma_VALUE;
  data_zmconvert( Asparse, &U, Magma_CSR, Magma_CSRU );
  dataType *Us;
  Us = (dataType*) calloc( sub_m*sub_n, sizeof(dataType) );
  data_sparse_subdense(sub_m, sub_n, sub_mbegin, sub_nbegin, &U, Us);
  EXPECT_ARRAY_DOUBLE_EQ( sub_m*sub_n, Us, sub1_U_valcheck);

  data_d_matrix C = {Magma_DENSE};
  C.num_rows = sub_m;
  C.num_cols = sub_n;
  C.ld = sub_n;
  C.nnz = 25;
  C.major = MagmaRowMajor;
  C.val = (dataType*) malloc( C.num_rows*C.num_cols*sizeof(dataType) );

  data_dgemm_mkl( Asparse.major, MagmaNoTrans, MagmaNoTrans, sub_m, sub_n, sub_n,
    alpha, Ls, sub_n,
    Us, sub_n, beta, C.val, C.ld);
  EXPECT_ARRAY_DOUBLE_EQ(sub_m*sub_n, C.val, sub1_T_valcheck);
//
  dataType sub2_L_valcheck[50] = {
    0,     0,     1,     0,     0,     0,     1,     0,     1,     1,
    0,     0,     0,     1,     0,     0,     0,     1,     0,     1,
    0,     0,     0,     0,     1,     0,     0,     0,     1,     0,
    0,     0,     0,     0,     0,     1,     0,     0,     0,     1,
    0,     0,     0,     0,     0,     0,     1,     0,     0,     0 };
  dataType sub2_U_valcheck[50] = {
    0,     0,     0,     1,     0,
    1,     0,     0,     0,     1,
    0,     1,     0,     0,     0,
    1,     0,     1,     0,     0,
    1,     1,     0,     1,     0,
   17,     1,     1,     0,     1,
    0,    19,     1,     1,     0,
    0,     0,    23,     1,     1,
    0,     0,     0,    29,     1,
    0,     0,     0,     0,    31 };
  dataType sub2_T_valcheck[25] = {
    0,    20,     1,    30,    32,
    1,     0,    24,     1,    32,
    1,     1,     0,    30,     1,
   17,     1,     1,     0,    32,
    0,    19,     1,     1,     0 };

  printf("START Second TEST\n");

  sub_mbegin = 10;
  sub_nbegin = 5;
  int span = MIN( (sub_mbegin+sub_m), (sub_nbegin+sub_n) );
  printf("span = %d\n", span);
  dataType *Ls2;
  Ls2 = (dataType*) calloc( sub_m*span, sizeof(dataType) );
  data_sparse_subdense(sub_m, span, sub_mbegin, 0, &L, Ls2);
  EXPECT_ARRAY_DOUBLE_EQ( sub_m*span, Ls2, sub2_L_valcheck);

  dataType *Us2;
  Us2 = (dataType*) calloc( span*sub_n, sizeof(dataType) );
  data_sparse_subdense(span, sub_n, 0, sub_nbegin, &U, Us2);
  EXPECT_ARRAY_DOUBLE_EQ( span*sub_n, Us2, sub2_U_valcheck);

  data_dgemm_mkl( Asparse.major,
    MagmaNoTrans, MagmaNoTrans,
    sub_m, sub_n, span,
    alpha, Ls2, span,
    Us2, sub_n,
    beta, C.val, C.ld);
  EXPECT_ARRAY_DOUBLE_EQ(sub_m*sub_n, C.val, sub2_T_valcheck);

  data_zmfree( &Asparse );
  data_zmfree( &A2 );
  data_zmfree( &B2 );

  data_zmfree( &L );
  data_zmfree( &U );

  free( As );
  free( Ls );
  free( Us );

}

TEST(Operations_sparse, sparsesubmatrices) {

  int Asub_rowcheck[6] = { 0, 4, 8, 13, 17, 21 };
  int Asub_colcheck[21] = { 0, 1, 2, 4,
    0, 1, 2, 3,
    0, 1, 2, 3, 4,
    1, 2, 3, 4,
    0, 2, 3, 4 };
  dataType Asub_valcheck[21] = { 3.000000e+00, 1.000000e+00, 1.000000e+00,
    1.000000e+00, 1.000000e+00, 5.000000e+00, 1.000000e+00, 1.000000e+00,
    1.000000e+00, 1.000000e+00, 7.000000e+00, 1.000000e+00, 1.000000e+00,
    1.000000e+00, 1.000000e+00, 1.100000e+01, 1.000000e+00, 1.000000e+00,
    1.000000e+00, 1.000000e+00, 1.300000e+01 };

  int C_rowcheck[6] = { 0, 5, 10, 15, 20, 25 };
  int C_colcheck[25] = { 0, 1, 2, 3, 4,
    0, 1, 2, 3, 4,
    0, 1, 2, 3, 4,
    0, 1, 2, 3, 4,
    0, 1, 2, 3, 4 };
  dataType C_valcheck[25] = { 1.200000e+01, 9.000000e+00, 1.200000e+01,
    3.000000e+00, 1.700000e+01, 9.000000e+00, 2.800000e+01, 1.400000e+01,
    1.700000e+01, 3.000000e+00, 1.200000e+01, 1.400000e+01, 5.300000e+01,
    2.000000e+01, 2.200000e+01, 3.000000e+00, 1.700000e+01, 2.000000e+01,
    1.240000e+02, 2.500000e+01, 1.700000e+01, 3.000000e+00, 2.200000e+01,
    2.500000e+01, 1.720000e+02 };

  char sparse_filename2[] = "matrices/Trefethen_20.mtx";
  data_d_matrix Asparse = {Magma_CSR};
  data_z_csr_mtx( &Asparse, sparse_filename2 );

  int sub_m = 5;
  int sub_n = 5;
  int sub_mbegin = 0;
  int sub_nbegin = 0;

  data_d_matrix Asub = {Magma_CSR};
  data_sparse_subsparse_cs( sub_m, sub_n, sub_mbegin, sub_nbegin, &Asparse, &Asub );
  data_zwrite_csr( &Asub );

  EXPECT_EQ(5, Asub.num_rows);
  EXPECT_EQ(5, Asub.num_cols);
  EXPECT_EQ(21, Asub.nnz);
  EXPECT_ARRAY_INT_EQ( (Asub.num_rows+1), Asub.row, Asub_rowcheck);
  EXPECT_ARRAY_INT_EQ( Asub.nnz, Asub.col, Asub_colcheck);
  EXPECT_ARRAY_DOUBLE_EQ( Asub.nnz, Asub.val, Asub_valcheck);

  data_d_matrix Asub2 = {Magma_CSR};
  data_sparse_subsparse_cs( sub_m, sub_n, sub_mbegin, sub_nbegin, &Asparse, &Asub2 );
  data_zwrite_csr( &Asub2 );

  dataType cone = 1.0;
  data_d_matrix C = {Magma_CSR};
  data_z_spmm( cone, Asub, Asub2, &C );

  EXPECT_EQ(5, C.num_rows);
  EXPECT_EQ(5, C.num_cols);
  EXPECT_EQ(25, C.nnz);
  EXPECT_ARRAY_INT_EQ( (C.num_rows+1), C.row, C_rowcheck);
  EXPECT_ARRAY_INT_EQ( C.nnz, C.col, C_colcheck);
  EXPECT_ARRAY_DOUBLE_EQ( C.nnz, C.val, C_valcheck);

  data_zwrite_csr( &C );
  data_d_matrix D = {Magma_DENSE};
  data_zmconvert( C, &D, Magma_CSR, Magma_DENSE );
  data_zdisplay_dense( &D );


  data_zmfree( &Asparse );
  data_zmfree( &Asub );
  data_zmfree( &Asub2 );

  data_zmfree( &C );
  data_zmfree( &D );

}

TEST(Operations_sparse, sparsesubmatrices_lowerUpper) {

  int Lsub_rowcheck[6] = { 0, 0, 1, 3, 5, 8 };
  int Lsub_colcheck[8] = { 0, 0, 1, 1, 2, 0, 2, 3 };
  dataType Lsub_valcheck[8] = { 1.000000e+00, 1.000000e+00, 1.000000e+00,
	  1.000000e+00,	1.000000e+00,	1.000000e+00,	1.000000e+00,	1.000000e+00 };

  int Usub_rowcheck[6] = { 0, 3, 5, 7, 8, 8 };
  int Usub_colcheck[8] = { 1, 2, 4, 2, 3, 3, 4, 4 };
  dataType Usub_valcheck[8] = { 1.000000e+00, 1.000000e+00, 1.000000e+00,
    1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00 };

  int C_rowcheck[6] = { 0, 0, 3, 7, 10, 14 };
  int C_colcheck[14] = { 1, 2, 4, 1, 2, 3, 4, 2, 3, 4, 1, 2, 3, 4 };
  dataType C_valcheck[14] = { 1.000000e+00,	1.000000e+00,	1.000000e+00,
    1.000000e+00,	2.000000e+00,	1.000000e+00,	1.000000e+00,	1.000000e+00,
    2.000000e+00,	1.000000e+00,	1.000000e+00,	1.000000e+00,	1.000000e+00,
    3.000000e+00 };

  char sparse_filename2[] = "matrices/Trefethen_20.mtx";
  data_d_matrix Asparse = {Magma_CSR};
  data_z_csr_mtx( &Asparse, sparse_filename2 );

  data_d_matrix L = {Magma_CSRL};
  L.diagorder_type = Magma_UNITY;
  L.major = MagmaRowMajor;
  data_zmconvert(Asparse, &L, Magma_CSR, Magma_CSRL);

  data_d_matrix U = {Magma_CSRU};
  U.diagorder_type = Magma_VALUE;
  U.major = MagmaRowMajor;
  data_zmconvert(Asparse, &U, Magma_CSR, Magma_CSRU);

  int sub_m = 5;
  int sub_n = 5;
  int sub_mbegin = 0;
  int sub_nbegin = 0;

  data_d_matrix Lsub = {Magma_CSR};
  data_sparse_subsparse_cs_lowerupper( sub_m, sub_n, sub_mbegin, sub_nbegin, &L, &Lsub );
  data_zwrite_csr( &Lsub );

  EXPECT_EQ(5, Lsub.num_rows);
  EXPECT_EQ(5, Lsub.num_cols);
  EXPECT_EQ(8, Lsub.nnz);
  EXPECT_ARRAY_INT_EQ( (Lsub.num_rows+1), Lsub.row, Lsub_rowcheck);
  EXPECT_ARRAY_INT_EQ( Lsub.nnz, Lsub.col, Lsub_colcheck);
  EXPECT_ARRAY_DOUBLE_EQ( Lsub.nnz, Lsub.val, Lsub_valcheck);

  data_d_matrix Usub = {Magma_CSR};
  data_sparse_subsparse_cs_lowerupper( sub_m, sub_n, sub_mbegin, sub_nbegin, &U, &Usub );
  data_zwrite_csr( &Usub );

  EXPECT_EQ(5, Usub.num_rows);
  EXPECT_EQ(5, Usub.num_cols);
  EXPECT_EQ(8, Usub.nnz);
  EXPECT_ARRAY_INT_EQ( (Usub.num_rows+1), Usub.row, Usub_rowcheck);
  EXPECT_ARRAY_INT_EQ( Usub.nnz, Usub.col, Usub_colcheck);
  EXPECT_ARRAY_DOUBLE_EQ( Usub.nnz, Usub.val, Usub_valcheck);

  dataType cone = 1.0;
  data_d_matrix C = {Magma_CSR};
  data_z_spmm( cone, Lsub, Usub, &C );

  EXPECT_EQ(5, C.num_rows);
  EXPECT_EQ(5, C.num_cols);
  EXPECT_EQ(14, C.nnz);
  EXPECT_ARRAY_INT_EQ( (C.num_rows+1), C.row, C_rowcheck);
  EXPECT_ARRAY_INT_EQ( C.nnz, C.col, C_colcheck);
  EXPECT_ARRAY_DOUBLE_EQ( C.nnz, C.val, C_valcheck);

  data_zwrite_csr( &C );
  data_d_matrix D = {Magma_DENSE};
  data_zmconvert( C, &D, Magma_CSR, Magma_DENSE );
  data_zdisplay_dense( &D );


  data_zmfree( &Asparse );
  data_zmfree( &L );
  data_zmfree( &U );
  data_zmfree( &Lsub );
  data_zmfree( &Usub );

  data_zmfree( &C );
  data_zmfree( &D );

}

TEST(Operations_sparse, sparsesubmatrices_lowerUpper2) {

  int Lsub_rowcheck[6] = { 0, 4, 7, 9, 11, 12 };
  int Lsub_colcheck[12] = { 2, 6, 8, 9, 3, 7, 9, 4, 8, 5, 9, 6 };
  dataType Lsub_valcheck[12] = { 1.000000e+00,	1.000000e+00,	1.000000e+00,
	  1.000000e+00,	1.000000e+00,	1.000000e+00,	1.000000e+00,	1.000000e+00,
	  1.000000e+00,	1.000000e+00,	1.000000e+00,	1.000000e+00 };

  int Usub_rowcheck[11] = { 0, 1, 3, 4, 6, 9, 12, 14, 16, 17, 17 };
  int Usub_colcheck[17] = { 3, 0, 4, 1, 0, 2, 0, 1, 3, 1, 2, 4, 2, 3, 3, 4, 4 };
  dataType Usub_valcheck[17] = { 1.000000e+00,	1.000000e+00,	1.000000e+00,
	  1.000000e+00,	1.000000e+00,	1.000000e+00,	1.000000e+00,	1.000000e+00,
	  1.000000e+00,	1.000000e+00,	1.000000e+00,	1.000000e+00,	1.000000e+00,
	  1.000000e+00,	1.000000e+00,	1.000000e+00,	1.000000e+00 };

  int C_rowcheck[6] = { 0, 4, 8, 12, 15, 17 };
  int C_colcheck[17] = { 1, 2, 3, 4, 0, 2, 3, 4, 0, 1, 3, 4, 1, 2, 4, 2, 3 };
  dataType C_valcheck[17] = { 1.000000e+00,	1.000000e+00,	1.000000e+00,
    1.000000e+00,	1.000000e+00,	1.000000e+00,	1.000000e+00,	1.000000e+00,
    1.000000e+00,	1.000000e+00,	1.000000e+00,	1.000000e+00,	1.000000e+00,
    1.000000e+00,	1.000000e+00,	1.000000e+00,	1.000000e+00 };

  char sparse_filename2[] = "matrices/Trefethen_20.mtx";
  data_d_matrix Asparse = {Magma_CSR};
  data_z_csr_mtx( &Asparse, sparse_filename2 );

  data_d_matrix L = {Magma_CSRL};
  L.diagorder_type = Magma_UNITY;
  L.major = MagmaRowMajor;
  data_zmconvert(Asparse, &L, Magma_CSR, Magma_CSRL);
  data_d_matrix D = {Magma_DENSE};
  data_zmconvert( L, &D, Magma_CSR, Magma_DENSE );
  printf("\nL :\n");
  data_zdisplay_dense( &D );
  data_zmfree( &D );

  data_d_matrix U = {Magma_CSRU};
  U.diagorder_type = Magma_VALUE;
  U.major = MagmaRowMajor;
  data_zmconvert(Asparse, &U, Magma_CSR, Magma_CSRU);
  data_zmconvert( U, &D, Magma_CSR, Magma_DENSE );
  printf("\nU : \n");
  data_zdisplay_dense( &D );
  data_zmfree( &D );

  int sub_m = 5;
  int sub_n = 5;
  int sub_mbegin = 10;
  int sub_nbegin = 5;
  int span = MIN( (sub_mbegin+sub_m - 0), (sub_nbegin+sub_n - 0) );

  data_d_matrix Lsub = {Magma_CSR};
  data_sparse_subsparse_cs_lowerupper( sub_m, span, sub_mbegin, 0, &L, &Lsub );
  data_zwrite_csr( &Lsub );
  data_zmconvert( Lsub, &D, Magma_CSR, Magma_DENSE );
  printf("\nLsub\n");
  data_zdisplay_dense( &D );
  data_zmfree( &D );

  EXPECT_EQ(5, Lsub.num_rows);
  EXPECT_EQ(10, Lsub.num_cols);
  EXPECT_EQ(12, Lsub.nnz);
  EXPECT_ARRAY_INT_EQ( (Lsub.num_rows+1), Lsub.row, Lsub_rowcheck);
  EXPECT_ARRAY_INT_EQ( Lsub.nnz, Lsub.col, Lsub_colcheck);
  EXPECT_ARRAY_DOUBLE_EQ( Lsub.nnz, Lsub.val, Lsub_valcheck);

  data_d_matrix Usub = {Magma_CSR};
  data_sparse_subsparse_cs_lowerupper( span, sub_n, 0, sub_nbegin, &U, &Usub );
  //data_sparse_subsparse_cs_lowerupper( 1, sub_n, 0, sub_nbegin, &U, &Usub );
  //data_sparse_subsparse_cs( 1, sub_n, 0, sub_nbegin, &U, &Usub );
  data_zwrite_csr( &Usub );
  data_zmconvert( Usub, &D, Magma_CSR, Magma_DENSE );
  printf("\nUsub\n");
  data_zdisplay_dense( &D );
  data_zmfree( &D );

  EXPECT_EQ(10, Usub.num_rows);
  EXPECT_EQ(5, Usub.num_cols);
  EXPECT_EQ(17, Usub.nnz);
  EXPECT_ARRAY_INT_EQ( (Usub.num_rows+1), Usub.row, Usub_rowcheck);
  EXPECT_ARRAY_INT_EQ( Usub.nnz, Usub.col, Usub_colcheck);
  EXPECT_ARRAY_DOUBLE_EQ( Usub.nnz, Usub.val, Usub_valcheck);

  dataType cone = 1.0;
  data_d_matrix C = {Magma_CSR};
  data_z_spmm( cone, Lsub, Usub, &C );

  EXPECT_EQ(5, C.num_rows);
  EXPECT_EQ(5, C.num_cols);
  EXPECT_EQ(17, C.nnz);
  EXPECT_ARRAY_INT_EQ( (C.num_rows+1), C.row, C_rowcheck);
  EXPECT_ARRAY_INT_EQ( C.nnz, C.col, C_colcheck);
  EXPECT_ARRAY_DOUBLE_EQ( C.nnz, C.val, C_valcheck);

  data_zwrite_csr( &C );
  //data_d_matrix D = {Magma_DENSE};
  data_zmconvert( C, &D, Magma_CSR, Magma_DENSE );
  printf("\nC\n");
  data_zdisplay_dense( &D );


  data_zmfree( &Asparse );
  data_zmfree( &L );
  data_zmfree( &U );
  data_zmfree( &Lsub );
  data_zmfree( &Usub );

  data_zmfree( &C );
  data_zmfree( &D );

}

TEST(Operations_sparse, sparsesubmatrices_lowerUpper3) {

  int Lsub_rowcheck[6] = { 0, 3, 6, 9, 13, 17 };
  int Lsub_colcheck[17] = { 1, 3, 4, 2, 4, 5, 3, 5, 6, 0, 4, 6, 7, 1, 5, 7, 8 };
  dataType Lsub_valcheck[17] = { 1.000000e+00,	1.000000e+00,	1.000000e+00,
	  1.000000e+00,	1.000000e+00,	1.000000e+00,	1.000000e+00,	1.000000e+00,
	  1.000000e+00,	1.000000e+00,	1.000000e+00,	1.000000e+00,	1.000000e+00,
	  1.000000e+00,	1.000000e+00,	1.000000e+00,	1.000000e+00 };

  int Usub_rowcheck[11] = { 0, 0, 0, 1, 2, 3, 4, 6, 7, 9, 12 };
  int Usub_colcheck[12] = { 0, 1, 2, 3, 0, 4, 1, 0, 2, 0, 1, 3 };
  dataType Usub_valcheck[12] = { 1.000000e+00,	1.000000e+00,	1.000000e+00,
	  1.000000e+00,	1.000000e+00,	1.000000e+00,	1.000000e+00,	1.000000e+00,
	  1.000000e+00,	1.000000e+00,	1.000000e+00,	1.000000e+00 };

  int C_rowcheck[6] = { 0, 2, 5, 9, 13, 17 };
  int C_colcheck[17] = { 1, 2, 0, 2, 3, 1, 0, 3, 4, 0, 2, 1, 4, 1, 0, 2, 3 };
  dataType C_valcheck[17] = { 1.000000e+00,	1.000000e+00,	1.000000e+00,
    1.000000e+00,	1.000000e+00,	1.000000e+00,	1.000000e+00,	1.000000e+00,
    1.000000e+00,	1.000000e+00,	1.000000e+00,	1.000000e+00,	1.000000e+00,
    1.000000e+00,	1.000000e+00,	1.000000e+00,	1.000000e+00 };

  char sparse_filename2[] = "matrices/Trefethen_20.mtx";
  data_d_matrix Asparse = {Magma_CSR};
  data_z_csr_mtx( &Asparse, sparse_filename2 );

  data_d_matrix L = {Magma_CSRL};
  L.diagorder_type = Magma_UNITY;
  L.major = MagmaRowMajor;
  data_zmconvert(Asparse, &L, Magma_CSR, Magma_CSRL);
  data_d_matrix D = {Magma_DENSE};
  data_zmconvert( L, &D, Magma_CSR, Magma_DENSE );
  printf("\nL :\n");
  data_zdisplay_dense( &D );
  data_zmfree( &D );

  data_d_matrix U = {Magma_CSRU};
  U.diagorder_type = Magma_VALUE;
  U.major = MagmaRowMajor;
  data_zmconvert(Asparse, &U, Magma_CSR, Magma_CSRU);
  data_zmconvert( U, &D, Magma_CSR, Magma_DENSE );
  printf("\nU : \n");
  data_zdisplay_dense( &D );
  data_zmfree( &D );

  int sub_m = 5;
  int sub_n = 5;
  int sub_mbegin = 5;
  int sub_nbegin = 10;
  int span = MIN( (sub_mbegin+sub_m - 0), (sub_nbegin+sub_n - 0) );

  data_d_matrix Lsub = {Magma_CSR};
  data_sparse_subsparse_cs_lowerupper( sub_m, span, sub_mbegin, 0, &L, &Lsub );
  data_zwrite_csr( &Lsub );
  data_zmconvert( Lsub, &D, Magma_CSR, Magma_DENSE );
  printf("\nLsub\n");
  data_zdisplay_dense( &D );
  data_zmfree( &D );

  EXPECT_EQ(5, Lsub.num_rows);
  EXPECT_EQ(10, Lsub.num_cols);
  EXPECT_EQ(17, Lsub.nnz);
  EXPECT_ARRAY_INT_EQ( (Lsub.num_rows+1), Lsub.row, Lsub_rowcheck);
  EXPECT_ARRAY_INT_EQ( Lsub.nnz, Lsub.col, Lsub_colcheck);
  EXPECT_ARRAY_DOUBLE_EQ( Lsub.nnz, Lsub.val, Lsub_valcheck);

  data_d_matrix Usub = {Magma_CSR};
  data_sparse_subsparse_cs_lowerupper( span, sub_n, 0, sub_nbegin, &U, &Usub );
  //data_sparse_subsparse_cs_lowerupper( 1, sub_n, 0, sub_nbegin, &U, &Usub );
  //data_sparse_subsparse_cs( 1, sub_n, 0, sub_nbegin, &U, &Usub );
  data_zwrite_csr( &Usub );
  data_zmconvert( Usub, &D, Magma_CSR, Magma_DENSE );
  printf("\nUsub\n");
  data_zdisplay_dense( &D );
  data_zmfree( &D );

  EXPECT_EQ(10, Usub.num_rows);
  EXPECT_EQ(5, Usub.num_cols);
  EXPECT_EQ(12, Usub.nnz);
  EXPECT_ARRAY_INT_EQ( (Usub.num_rows+1), Usub.row, Usub_rowcheck);
  EXPECT_ARRAY_INT_EQ( Usub.nnz, Usub.col, Usub_colcheck);
  EXPECT_ARRAY_DOUBLE_EQ( Usub.nnz, Usub.val, Usub_valcheck);

  dataType cone = 1.0;
  data_d_matrix C = {Magma_CSR};
  data_z_spmm( cone, Lsub, Usub, &C );

  EXPECT_EQ(5, C.num_rows);
  EXPECT_EQ(5, C.num_cols);
  EXPECT_EQ(17, C.nnz);
  EXPECT_ARRAY_INT_EQ( (C.num_rows+1), C.row, C_rowcheck);
  EXPECT_ARRAY_INT_EQ( C.nnz, C.col, C_colcheck);
  EXPECT_ARRAY_DOUBLE_EQ( C.nnz, C.val, C_valcheck);

  data_zwrite_csr( &C );
  //data_d_matrix D = {Magma_DENSE};
  data_zmconvert( C, &D, Magma_CSR, Magma_DENSE );
  printf("\nC\n");
  data_zdisplay_dense( &D );


  data_zmfree( &Asparse );
  data_zmfree( &L );
  data_zmfree( &U );
  data_zmfree( &Lsub );
  data_zmfree( &Usub );

  data_zmfree( &C );
  data_zmfree( &D );

}
