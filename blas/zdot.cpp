/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s
       @author Stephen Wood

*/
#include "../include/sparse.h"
#include <mkl.h>

/*******************************************************************************
    Purpose
    -------

    This routine computes a dot product of a vector, dx, with
    another vector, dy, on the CPU.

    Arguments
    ---------
    @param[in]
    n           int
                the number of elements in vectors x and y.

    @param[in]
    dx          dataType*
                input vector dx.

    @param[in]
    incx        int
                the increment for the elements of dx.

    @param[in,out]
    dy          dataType*
                input/output vector dy.

    @param[in]
    incy        int
                the increment for the elements of dy.


    @ingroup magmasparse_zblas
*******************************************************************************/

extern "C"
dataType
data_zdot(
    int n,
    dataType* dx, int incx,
    dataType* dy, int incy )
{
  dataType result = 0.0;

  #pragma omp parallel
  #pragma omp for reduction(+:result) nowait
  for ( int i=0; i<n; i++ ) {
    result += dx[ i*incx ] * dy[ i*incy ];
  }

  return result;
}

extern "C"
dataType
data_zdot_mkl(
    int n,
    dataType* dx, int incx,
    dataType* dy, int incy )
{
  dataType result = 0.0;

  result = cblas_ddot(n, dx, incx, dy, incy);

  return result;
}
