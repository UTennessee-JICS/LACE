/*
 *  -- LACE (version 0.0) --
 *     Univ. of Tennessee, Knoxville
 *
 *     @author Stephen Wood
 *
 */
#include "../include/sparse.h"
#include <mkl.h>
#include <float.h>
#include <stdlib.h>
#include <stdio.h>

/**
 *  Purpose
 *  -------
 *
 *  Assess the orthogonality of the upper-left search x search portion of a matrix.
 *
 *
 *  Arguments
 *  ---------
 *
 *  @param[in]
 *  krylov      data_d_matrix*
 *              descriptor for matrix krylov
 *
 *  @param[in,out]
 *  ortherr     dataType *
 *              Infinity norm of orthogonality error
 *
 *  @param[in,out]
 *  imax        int*
 *              row where maximum row sum is found
 *
 *  @param[in]
 *  search      int
 *              extent of the matrix to be assesed
 *
 *  @ingroup datasparse_orthogonality
 ********************************************************************/


extern "C"
void
data_orthogonality_error(data_d_matrix * krylov,
  dataType *                             ortherr,
  int *                                  imax,
  int                                    search)
{
  dataType zero   = 0.0;
  dataType one    = 1.0;
  dataType negone = -1.0;

  data_d_matrix eye = { Magma_DENSE };

  data_zvinit(&eye, search, search, zero);
  #pragma omp parallel
  //#pragma omp for simd schedule(monotonic:static) nowait
  #pragma omp for schedule(static) nowait
  #pragma simd
  #pragma vector aligned
  for (int e = 0; e < eye.ld; e++) {
    eye.val[idx(e, e, eye.ld)] = 1.0;
  }


  #pragma omp parallel
  //#pragma omp for collapse(2) schedule(monotonic:static) nowait
  #pragma omp for collapse(2) schedule(static) nowait
  #pragma vector aligned
  for (int i = 0; i < eye.num_rows; i++) {
    for (int j = 0; j < eye.num_cols; j++) {
      dataType sum = 0.0;
      sum = data_zdot_mkl(krylov->num_rows,
          &(krylov->val[idx(0, i, krylov->ld)]), 1,
          &(krylov->val[idx(0, j, krylov->ld)]), 1);
      eye.val[idx(i, j, eye.ld)] -= sum;
    }
  }

  data_infinity_norm(&eye, imax, ortherr);

  data_zmfree(&eye);
} // data_orthogonality_error

extern "C"
void
data_orthogonality_error_incremental(data_d_matrix * krylov,
  dataType *                                         ortherr,
  int *                                              imax,
  int                                                search)
{
  dataType zero   = 0.0;
  dataType one    = 1.0;
  dataType negone = -1.0;

  data_d_matrix eye = { Magma_DENSE };

  data_zvinit(&eye, search, 1, zero);
  eye.val[(search - 1)] = 1.0;
  dataType inorm = 0.0;

  #pragma omp parallel
  //#pragma omp for schedule(monotonic:static) reduction(+:inorm) nowait
  #pragma omp for schedule(static) reduction(+:inorm) nowait
  #pragma vector aligned
  for (int i = 0; i < eye.num_rows; ++i) {
    dataType sum = data_zdot_mkl(krylov->num_rows,
        &(krylov->val[idx(0, i, krylov->ld)]), 1,
        &(krylov->val[idx(0, (search - 1), krylov->ld)]), 1);
    eye.val[i] -= sum;
    inorm      += fabs(eye.val[i]);
  }

  (*ortherr) = inorm;

  data_zmfree(&eye);
}
