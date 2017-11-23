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

extern "C"
int
data_dcsrilu0_mkl(data_d_matrix * dA)
{
  int info = 0;
  int n    = dA->num_rows;
  int ipar[128];
  dataType dpar[128];

  #pragma omp parallel
  #pragma omp for nowait
  for (int i = 0; i < 128; i++) {
    ipar[i] = 0;
    dpar[i] = 0.0;
  }
  ipar[0]  = n;                 // problem size (number of rows)
  ipar[1]  = 6;                 // error log control 6: to screan
  ipar[2]  = 1;                 // RCI stage
  ipar[3]  = 0;                 // iteration count
  ipar[4]  = (int) MIN(150, n); // maximum number of iterations
  ipar[5]  = 1;                 // error output switch
  ipar[6]  = 1;                 // warning output switch
  ipar[7]  = 0;                 // dfgmres iteration stopping test switch
  ipar[8]  = 1;                 // dfgmres residual stopping test switch
  ipar[9]  = 1;                 // user defined stopping test switch
  ipar[10] = 0;                 // preconditioned dfgmres switch
  ipar[11] = 0;                 // zero norm check switch
  ipar[15] = ipar[4];           // default non-restart GMRES
  ipar[30] = 0;                 // 0: stop if diagonal entry < dpar[30], 1: replace with dpar[31]

  dpar[30] = 1.0e-16; // small value
  dpar[31] = 1.0e-10; // value that replaces diagonal elements < dpar[30]

  // increment to create one-based indexing of the array parameters
  #pragma omp parallel
  #pragma omp for nowait
  for (int i = 0; i < dA->nnz; i++) {
    dA->col[i] += 1;
  }
  #pragma omp parallel
  #pragma omp for nowait
  for (int i = 0; i < dA->num_rows + 1; i++) {
    dA->row[i] += 1;
  }
  // start = magma_sync_wtime( queue );
  dcsrilu0( (int *) &n, dA->val, dA->row, dA->col, dA->val,
    ipar, dpar, (int *) &info);
  // end = magma_sync_wtime( queue );
  //    t_cusparse = end-start;

  // decrement to create zero-based indexing of the array parameters
  #pragma omp parallel
  #pragma omp for nowait
  for (int i = 0; i < dA->nnz; i++) {
    dA->col[i] -= 1;
  }
  #pragma omp parallel
  #pragma omp for nowait
  for (int i = 0; i < dA->num_rows + 1; i++) {
    dA->row[i] -= 1;
  }
  return info;
} // data_dcsrilu0_mkl
