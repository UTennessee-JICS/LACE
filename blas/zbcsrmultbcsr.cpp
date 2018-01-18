/*
 *  -- LACE (version 0.0) --
 *     Univ. of Tennessee, Knoxville
 *
 *     @author Stephen Wood
 *
 */

#include "../include/sparse.h"
#include <mkl.h>
#include <stdlib.h>
#include <stdio.h>
#include <algorithm>
#include <ctime>


extern "C"
int
data_diagbcsr_mult_bcsr(
  data_d_matrix * diagA,
  data_d_matrix * A)
{
  int info      = 0;
  dataType one  = 1.0;
  dataType zero = 0.0;

  DEV_CHECKPT

  data_d_matrix bhandle = { Magma_DENSE };

  bhandle.num_rows  = A->blocksize;
  bhandle.num_cols  = A->blocksize;
  bhandle.blocksize = A->blocksize;
  bhandle.nnz       = bhandle.num_rows * bhandle.num_cols;
  bhandle.true_nnz  = bhandle.nnz;
  bhandle.ld        = bhandle.num_cols;
  bhandle.major     = MagmaRowMajor;
  // LACE_CALLOC(bhandle.val, bhandle.nnz);


  data_d_matrix binvhandle = { Magma_DENSE };
  binvhandle.num_rows  = A->blocksize;
  binvhandle.num_cols  = A->blocksize;
  binvhandle.blocksize = A->blocksize;
  binvhandle.nnz       = binvhandle.num_rows * binvhandle.num_cols;
  binvhandle.true_nnz  = binvhandle.nnz;
  binvhandle.ld        = binvhandle.num_cols;
  binvhandle.major     = MagmaRowMajor;
  // LACE_CALLOC(binvhandle.val, binvhandle.nnz);

  data_d_matrix result = { Magma_DENSE };
  result.num_rows  = A->blocksize;
  result.num_cols  = A->blocksize;
  result.blocksize = A->blocksize;
  result.nnz       = result.num_rows * result.num_cols;
  result.true_nnz  = result.nnz;
  result.ld        = result.num_cols;
  result.major     = MagmaRowMajor;
  LACE_CALLOC(result.val, result.nnz);

  DEV_CHECKPT

  for (int i = 0; i < A->num_rows; i++) {
    // printf("row %d:\n", i);
    for (int j = A->row[i]; j < A->row[i + 1]; j++) {
      // printf("block %d bcol %d\n", j, A->col[j]);

      bhandle.val    = &A->val[j * A->ldblock];
      binvhandle.val = &diagA->val[A->col[j] * diagA->ldblock];

      // printf("A:\n");
      // data_zdisplay_dense( &bhandle );
      //
      // printf("diagAinv:\n");
      // data_zdisplay_dense( &binvhandle );

      // data_zprint_dense( bhandle );
      // DEV_CHECKPT
      // data_zprint_dense( binvhandle );

      // data_inverse( &bhandle, &binvhandle );

      data_dgemm_mkl(MagmaRowMajor, MagmaNoTrans, MagmaNoTrans,
        bhandle.num_rows, bhandle.num_cols, binvhandle.num_cols,
        one, bhandle.val, bhandle.ld, binvhandle.val, binvhandle.ld,
        zero, result.val, result.ld);

      // printf("bhandle*binvhandle block %d bcol %d : \n", j, A->col[j]);
      // data_zdisplay_dense( &result );

      for (int ii = 0; ii < A->ldblock; ii++) {
        A->val[j * A->ldblock + ii] = result.val[ii];
      }

      // data_zprint_dense( bhandle );
      // DEV_CHECKPT
      // data_zprint_dense( binvhandle );
    }
  }

  data_zmfree(&result);

  return info;
} // data_diagbcsr_mult_bcsr
