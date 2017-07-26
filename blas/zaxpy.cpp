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
    
    This routine computes dy = alpha *  dx + dy on the CPU.
    
    Arguments
    ---------
    @param[in]
    n           data_int_t
                the number of elements in vectors dx and dy.

    @param[in]
    alpha       dataType
                scalar multiplier.
                
    @param[in]
    dx          dataType_const_ptr
                input vector dx.
                
    @param[in]
    incx        data_int_t
                the increment for the elements of dx.
                
    @param[in,out]
    dy          dataType_ptr
                input/output vector dy.
                
    @param[in]
    incy        data_int_t
                the increment for the elements of dy.

    @ingroup datasparse_zblas
*******************************************************************************/

extern "C" 
void
data_zaxpy(
    data_int_t n,
    dataType alpha,
    dataType_const_ptr dx, data_int_t incx,
    dataType_ptr       dy, data_int_t incy )
{     
    cblas_daxpy(n, alpha, dx, incx, dy, incy); 
}
