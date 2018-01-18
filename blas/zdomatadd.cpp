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

                transA = 'N' or 'n'   y := alpha*A*B + beta*C.

                transA = 'T' or 't'   y := alpha*A^T*B + beta*C.

                transA = 'C' or 'c'   y := alpha*A^H*B + beta*C.

    @param[in]
    transB      transB is CHARACTER*1
                On entry, TRANS specifies the operation to be performed as
                follows:

                transB = 'N' or 'n'   y := alpha*A*B + beta*C.

                transB = 'T' or 't'   y := alpha*A*B^T + beta*C.

                transB = 'C' or 'c'   y := alpha*A*B^H + beta*C.

    @param[in]
    m           int
                number of rows of the matrix A and rows of the matrix C.

    @param[in]
    n           int
                number of columns of the matrix B and collumns of the matrix C.

    @param[in]
    k           int
                number of columns of the matrix A and rows of matrix B.

    @param[in]
    alpha       magmaDoubleComplex
                scalar.

    @param[in]
    dA          magmaDoubleComplex_ptr
                input matrix dA.

    @param[in]
    lda         int
                the increment for the elements of dA.

    @param[in]
    dB          magmaDoubleComplex_ptr
                input matrix dB.

    @param[in]
    ldb         int
                the increment for the elements of dB.

    @param[in]
    beta        magmaDoubleComplex
                scalar.

    @param[in,out]
    dC          magmaDoubleComplex_ptr
                input vector dy.

    @@param[in]
    ldc         int
                the increment for the elements of dC.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zblas
*******************************************************************************/

//void mkl_domatadd (char ordering, char transa, char transb,
//size_t m, size_t n, const double alpha, const double * A,
//size_t lda, const double beta, const double * B, size_t ldb,
//double * C, size_t ldc);
extern "C"
void
data_domatadd_mkl(
    data_order_t layoutA, data_trans_t transA, data_trans_t transB,
    int m, int n,
    dataDouble alpha, dataDouble_const_ptr dA, int lda,
    dataDouble beta, dataDouble_const_ptr dB, int ldb,
    dataDouble_ptr dC, int ldc )
{
    mkl_domatadd(cblas_order_const(layoutA), cblas_trans_const(transA), cblas_trans_const(transB),
                m, n, alpha, dA, lda, beta, dB, ldb, dC, ldc);
}


extern "C"
void
data_domatadd(
    dataDouble alpha, data_d_matrix* A, data_trans_t transA,
    dataDouble beta, data_d_matrix* B, data_trans_t transB,
    data_d_matrix* C )
{
    mkl_domatadd(cblas_order_const(A->major), cblas_trans_const(transA), cblas_trans_const(transB),
                A->num_rows, A->num_cols, alpha, A->val, A->ld, beta, B->val, B->ld, C->val, C->ld);
}
