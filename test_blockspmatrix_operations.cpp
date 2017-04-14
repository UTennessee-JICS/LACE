/*

*/

#include <stdlib.h>
#include <stdio.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <vector>

#include "include/mmio.h"
#include "include/sparse_types.h"

//+++++++
#define EXPECT_ITERABLE_DOUBLE_EQ( TYPE, ref, target) \
{ \
const TYPE& _ref(ref); \
const TYPE& _target(target); \
TYPE::const_iterator tarIter   = _target.begin(); \
TYPE::const_iterator refIter = _ref.begin(); \
unsigned int i = 0; \
while(refIter != _ref.end()) { \
    if ( tarIter == _target.end() ) { \
        ADD_FAILURE() << #target \
            " has a smaller length than " #ref ; \
        break; \
    } \
    EXPECT_DOUBLE_EQ(* refIter, * tarIter) \
        << "Vectors " #ref  " (refIter) " \
           "and " #target " (tarIter) " \
           "differ at index " << i; \
    ++refIter; ++tarIter; ++i; \
} \
EXPECT_TRUE( tarIter == _target.end() ) \
    << #ref " has a smaller length than " \
       #target ; \
}
//+++++++

//+++++++
#define EXPECT_ARRAY_DOUBLE_EQ( length, ref, target) \
{ \
  unsigned int i = 0; \
  for(i=0; i<length; i++) { \
    EXPECT_DOUBLE_EQ(ref[i], target[i]) \
    << "Arrays " #ref  " and " #target \
           "differ at index " << i; \
  } \
}
//+++++++ 

//+++++++
#define EXPECT_ARRAY_INT_EQ( length, ref, target) \
{ \
  unsigned int i = 0; \
  for(i=0; i<length; i++) { \
    EXPECT_EQ(ref[i], target[i]) \
    << "Arrays " #ref  " and " #target \
           "differ at index " << i; \
  } \
}
//+++++++ 

TEST(Conversion, csr_to_bsr) {

  dataType absr_check[8] = { 1.000000e+00, 2.000000e+00, 3.000000e+00, 
  4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00 };
  int bsrcol_check[2] = { 0, 1 };
  int bsrrow_check[3] = { 0, 1, 2 };
  int nnzblocks_check = 2;
  
  int m = 4;
  int mblk = 2;
  int ldblock = mblk*mblk;
  
  double acsr[8] = { 1., 2., 3., 4., 5., 6., 7., 8. };
  int ja[8] = { 1, 2, 1, 2, 3, 4, 3, 4 };
  int ia[5] = { 1, 3, 5, 7, 9 };
  int job[6] = { 0, 1, 0, 0, 0, -1 };
  int nnzblocks = -1;
  int info;
  
  mkl_dcsrbsr(job, &m, &mblk, &ldblock, acsr, ja, ia, NULL, NULL, &nnzblocks, &info);
  EXPECT_EQ(nnzblocks, nnzblocks_check);
  
  int bsr_num_rows = (m + mblk - 1)/mblk;
  double*  absr;
  int* bsrcol;
  int* bsrrow;
  absr = (dataType*) calloc(nnzblocks*ldblock, sizeof(dataType) );
  bsrrow = (int*) calloc(bsr_num_rows+1, sizeof(int) );
  bsrcol = (int*) calloc(nnzblocks, sizeof(int) );
  
  int job2[6] = { 0, 1, 0, 0, 0, 1 };
  mkl_dcsrbsr(job2, &m, &mblk, &ldblock, acsr, ja, ia, absr, bsrcol, bsrrow, &info);
  
  
  EXPECT_ARRAY_DOUBLE_EQ(2, bsrcol, bsrcol_check);
  EXPECT_ARRAY_DOUBLE_EQ(3, bsrrow, bsrrow_check);
  EXPECT_ARRAY_DOUBLE_EQ(8, absr, absr_check);
  
}

TEST(Conversion, csr_to_bsr_lace) {
  
  int info;
  dataType absr_check[8] = { 1.000000e+00, 2.000000e+00, 3.000000e+00, 
  4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00 };
  int bsrcol_check[2] = { 0, 1 };
  int bsrrow_check[3] = { 0, 1, 2 };
  int nnzblocks_check = 2;
  
  data_d_matrix A = {Magma_CSR};
  data_d_matrix B = {Magma_BCSR};
  
  A.num_rows = 4;
  A.num_cols = 4;
  A.nnz = 8;
  LACE_CALLOC(A.val, A.nnz);
  A.val[0] = 1.;
  A.val[1] = 2.;
  A.val[2] = 3.;
  A.val[3] = 4.;
  A.val[4] = 5.;
  A.val[5] = 6.;
  A.val[6] = 7.;
  A.val[7] = 8.;
  LACE_CALLOC(A.row, (A.num_rows+1));
  A.row[0] = 1;
  A.row[1] = 3;
  A.row[2] = 5;
  A.row[3] = 7;
  A.row[4] = 9;
  LACE_CALLOC(A.col, A.nnz);
  A.col[0] = 1;
  A.col[1] = 2;
  A.col[2] = 1;
  A.col[3] = 2;
  A.col[4] = 3;
  A.col[5] = 4;
  A.col[6] = 3;
  A.col[7] = 4;
  
  B.blocksize = 2;
  B.ldblock = B.blocksize*B.blocksize;
  
  int job[6] = { 0, 1, 0, 0, 0, -1 };
  mkl_dcsrbsr(job, &A.num_rows, &B.blocksize, &B.ldblock, A.val, A.col, A.row, NULL, NULL, &B.numblocks, &info);
  EXPECT_EQ(B.numblocks, nnzblocks_check);
  
  B.num_rows = (A.num_rows + B.blocksize - 1)/B.blocksize;
  LACE_CALLOC(B.val, B.numblocks*B.ldblock);
  LACE_CALLOC(B.row, (B.num_rows+1));
  LACE_CALLOC(B.col, B.numblocks);
  
  job[5] = 1;
  mkl_dcsrbsr(job, &A.num_rows, &B.blocksize, &B.ldblock, A.val, A.col, A.row, B.val, B.col, B.row, &info);

  EXPECT_ARRAY_DOUBLE_EQ(2, B.col, bsrcol_check);
  EXPECT_ARRAY_DOUBLE_EQ(3, B.row, bsrrow_check);
  EXPECT_ARRAY_DOUBLE_EQ(8, B.val, absr_check);
  
}


int main(int argc, char* argv[])
{
  char filename[] = "testing/matrices/sparisty_test.mtx";
  
  //data_d_matrix A = {Magma_CSR};
  //data_z_csr_mtx( &A, filename ); 
  //data_zprint_csr( A );
  //
  //data_d_matrix B = {Magma_CSR};
  //data_z_csr_mtx( &B, filename ); 
  //data_zprint_csr( B );
  //
  //data_d_matrix F = {Magma_DENSE};
  //data_zmconvert( A, &F, Magma_CSR, Magma_DENSE );
  //data_d_matrix G = {Magma_BCSR};
  
  
  int m = 4;
  int mblk = 2;
  int ldblock = mblk*mblk;
  
  double acsr[8] = { 1., 2., 3., 4., 5., 6., 7., 8. };
  int ja[8] = { 1, 2, 1, 2, 3, 4, 3, 4 };
  int ia[5] = { 1, 3, 5, 7, 9 };
  int job[6] = { 0, 1, 0, 0, 0, -1 };
  int nnzblocks = -1;
  int info;
  
  mkl_dcsrbsr(job, &m, &mblk, &ldblock, acsr, ja, ia, NULL, NULL, &nnzblocks, &info);
  printf("info = %d, nnzblocks = %d \n", info, nnzblocks);
  
  int bsr_num_rows = (m + mblk - 1)/mblk;
  double*  absr;
  int* bsrcol;
  int* bsrrow;
  absr = (dataType*) calloc(nnzblocks*ldblock, sizeof(dataType) );
  bsrrow = (int*) calloc(bsr_num_rows+1, sizeof(int) );
  bsrcol = (int*) calloc(nnzblocks, sizeof(int) );
  
  int job2[6] = { 0, 1, 0, 0, 0, 1 };
  mkl_dcsrbsr(job2, &m, &mblk, &ldblock, acsr, ja, ia, absr, bsrcol, bsrrow, &info);
  
  for (int i=0; i<bsr_num_rows; i++ ) {
    printf("row %d:\n", i);
    for (int j=bsrrow[i]; j<bsrrow[i+1]; j++) {
      printf("block %d bcol %d\n", j, bsrcol[j]);
      for (int k=0; k<ldblock; k++ ) {
        printf("%e ", absr[j*ldblock+k]);
      }
    }
    printf("\n");
  }
  printf("bsr_num_rows = %d\n", bsr_num_rows);
  printf("bsrrows:\n");
  for (int i=0; i<bsr_num_rows+1; i++ ) {
    printf("%d, ", bsrrow[i]);
  }
  printf("\nbsrcols:\n");
  for (int i=0; i<nnzblocks; i++ ) {
    printf("%d, ", bsrcol[i]);
  }
  printf("\nabsr:\n");
  for (int i=0; i<nnzblocks*ldblock; i++ ) {
    printf("%e, ", absr[i]);
  }
  printf("\n");
  
  data_d_matrix A = {Magma_CSR};
  data_d_matrix B = {Magma_BCSR};
  
  A.num_rows = 4;
  A.num_cols = 4;
  A.nnz = 8;
  LACE_CALLOC(A.val, A.nnz);
  A.val[0] = 1.;
  A.val[1] = 2.;
  A.val[2] = 3.;
  A.val[3] = 4.;
  A.val[4] = 5.;
  A.val[5] = 6.;
  A.val[6] = 7.;
  A.val[7] = 8.;
  LACE_CALLOC(A.row, (A.num_rows+1));
  A.row[0] = 1;
  A.row[1] = 3;
  A.row[2] = 5;
  A.row[3] = 7;
  A.row[4] = 9;
  LACE_CALLOC(A.col, A.nnz);
  A.col[0] = 1;
  A.col[1] = 2;
  A.col[2] = 1;
  A.col[3] = 2;
  A.col[4] = 3;
  A.col[5] = 4;
  A.col[6] = 3;
  A.col[7] = 4;
  
  B.blocksize = 2;
  B.ldblock = B.blocksize*B.blocksize;
  
  mkl_dcsrbsr(job, &A.num_rows, &B.blocksize, &B.ldblock, A.val, A.col, A.row, NULL, NULL, &B.numblocks, &info);
  printf("info = %d, B.numblocks = %d \n", info, B.numblocks);
  
  B.num_rows = (A.num_rows + B.blocksize - 1)/B.blocksize;
  LACE_CALLOC(B.val, B.numblocks*B.ldblock);
  LACE_CALLOC(B.row, (B.num_rows+1));
  LACE_CALLOC(B.col, B.numblocks);
  
  mkl_dcsrbsr(job2, &A.num_rows, &B.blocksize, &B.ldblock, A.val, A.col, A.row, B.val, B.col, B.row, &info);
  for (int i=0; i<B.num_rows; i++ ) {
    printf("row %d:\n", i);
    for (int j=B.row[i]; j<B.row[i+1]; j++) {
      printf("block %d bcol %d\n", j, B.col[j]);
      for (int k=0; k<B.ldblock; k++ ) {
        printf("%e ", B.val[j*ldblock+k]);
      }
    }
    printf("\n");
  }
  printf("bsr_num_rows = %d\n", B.num_rows);
  printf("bsrrows:\n");
  for (int i=0; i<B.num_rows+1; i++ ) {
    printf("%d, ", B.row[i]);
  }
  printf("\nbsrcols:\n");
  for (int i=0; i<nnzblocks; i++ ) {
    printf("%d, ", bsrcol[i]);
  }
  printf("\nabsr:\n");
  for (int i=0; i<B.numblocks*B.ldblock; i++ ) {
    printf("%e, ", absr[i]);
  }
  printf("\n");
  
  //data_zmconvert( B, &G, Magma_CSR, Magma_BCSR );
  DEV_CHECKPT
  //data_zprint_csr( B );
  

  
  data_zmfree( &A );
  data_zmfree( &B );
  
  //data_zmfree( &F );
  //data_zmfree( &G );
  
  
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
  
  printf("done\n");
  fflush(stdout); 
  return 0;
  
}