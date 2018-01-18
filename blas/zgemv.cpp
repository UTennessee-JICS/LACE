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

    ZGEMV  performs one of the matrix-vector operations

    y := alpha*A*x + beta*y,   or   y := alpha*A^T*x + beta*y,   or

    y := alpha*A^H*x + beta*y,

    where alpha and beta are scalars, x and y are vectors and A is an
    m by n matrix.

    Arguments
    ---------
    @param[in]
    transA      transA is CHARACTER*1
                On entry, TRANS specifies the operation to be performed as
                follows:

                transA = 'N' or 'n'   y := alpha*A*x + beta*y.

                transA = 'T' or 't'   y := alpha*A^T*x + beta*y.

                transA = 'C' or 'c'   y := alpha*A^H*x + beta*y.

    @param[in]
    m           int
                number of rows of the matrix A.

    @param[in]
    n           int
                number of columns of the matrix A.

    @param[in]
    alpha       magmaDoubleComplex
                scalar.

    @param[in]
    dA          magmaDoubleComplex_ptr
                input matrix dA.

    @param[in]
    lda        int
                the increment for the elements of dx.

    @param[in]
    dx          magmaDoubleComplex_ptr
                input vector dx.

    @param[in]
    incx        int
                the increment for the elements of dx.

    @param[in]
    beta        magmaDoubleComplex
                scalar.

    @param[in,out]
    dy          magmaDoubleComplex_ptr
                input vector dy.

    @param[in]
    incy        int
                the increment for the elements of dy.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zblas
*******************************************************************************/

extern "C"
void
data_dgemv_mkl(
    data_order_t layoutA, data_trans_t transA, int m, int n,
    dataDouble alpha,
    dataDouble_const_ptr dA, int lda,
    dataDouble_const_ptr dx, int incx, dataDouble beta,
    dataDouble_ptr dy, int incy )
{
    cblas_dgemv( cblas_order_const(layoutA), cblas_trans_const(transA), m, n, alpha, dA, lda, dx,
      incx, beta, dy, incy);
}
