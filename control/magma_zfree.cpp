/*
    -- DEV (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
       @author Hartwig Anzt
       @author Stephen Wood
*/
#include "../include/sparse.h"


/**
    Purpose
    -------

    Free the memory of a data_z_matrix.


    Arguments
    ---------

    @param[in,out]
    A           data_z_matrix*
                matrix to free
    @param[in]
    queue       data_queue_t
                Queue to execute in.

    @ingroup datasparse_zaux
    ********************************************************************/

extern "C" 
int
data_zmfree(
    data_d_matrix *A )
{
    if ( A->storage_type == Magma_ELL || A->storage_type == Magma_ELLPACKT ){
        free( A->val );
        free( A->col );
        A->num_rows = 0;
        A->num_cols = 0;
        A->nnz = 0; 
        A->true_nnz = 0;
    }
    else if (A->storage_type == Magma_ELLD ) {
        free( A->val );
        free( A->col );
        A->num_rows = 0;
        A->num_cols = 0;
        A->nnz = 0; 
        A->true_nnz = 0;
    }
    else if ( A->storage_type == Magma_ELLRT ) {
        free( A->val );
        free( A->row );
        free( A->col );
        A->num_rows = 0;
        A->num_cols = 0;
        A->nnz = 0;
        A->true_nnz = 0;
    }
    else if ( A->storage_type == Magma_SELLP ) {
        free( A->val );
        free( A->row );
        free( A->col );
        A->num_rows = 0;
        A->num_cols = 0;
        A->nnz = 0; 
        A->true_nnz = 0;
    }
    else if ( A->storage_type == Magma_CSRLIST ) {
        free( A->val );
        free( A->row );
        free( A->col );
        free( A->list );
        A->num_rows = 0;
        A->num_cols = 0;
        A->nnz = 0; 
        A->true_nnz = 0;
    }
    else if ( A->storage_type == Magma_CSR || A->storage_type == Magma_CSRD
                                    || A->storage_type == Magma_CSRL
                                    || A->storage_type == Magma_CSRU
                                    || A->storage_type == Magma_CSC
                                    || A->storage_type == Magma_CSCD
                                    || A->storage_type == Magma_CSCL
                                    || A->storage_type == Magma_CSCU ) {
        free( A->val );
        free( A->col );
        free( A->row );
        if ( A->rowidx != NULL ) {
          free( A->rowidx );
        }
        A->num_rows = 0;
        A->num_cols = 0;
        A->pad_rows = 0;
        A->pad_cols = 0;
        A->nnz = 0; 
        A->true_nnz = 0;
        A->major = (data_order_t) 0;
        A->diameter = 0;
    }
    else if (  A->storage_type == Magma_CSRCOO || A->storage_type == Magma_CSCCOO ) {
        free( A->val );
        free( A->col );
        free( A->row );
        free( A->rowidx );
        A->num_rows = 0;
        A->num_cols = 0;
        A->pad_rows = 0;
        A->pad_cols = 0;
        A->nnz = 0; 
        A->true_nnz = 0;
        A->major = (data_order_t) 0;
        A->diameter = 0;
    }
    else if ( A->storage_type == Magma_BCSR ) {
        free( A->val );
        free( A->col );
        free( A->row );
        free( A->blockinfo );
        A->num_rows = 0;
        A->num_cols = 0;
        A->pad_rows = 0;
        A->pad_cols = 0;
        A->nnz = 0; 
        A->true_nnz = 0;
        A->blockinfo = 0;
        A->major = (data_order_t) 0;
        A->diameter = 0;
    }
    else if ( A->storage_type == Magma_DENSE  || A->storage_type == Magma_DENSEL 
                                         || A->storage_type == Magma_DENSEU 
                                         || A->storage_type == Magma_DENSED ) {
        free( A->val );
        A->num_rows = 0;
        A->num_cols = 0;
        A->pad_rows = 0;
        A->pad_cols = 0;
        A->nnz = 0; 
        A->true_nnz = 0;
        A->major = (data_order_t) 0;
    }
    A->val = NULL;
    A->col = NULL;
    A->row = NULL;
    A->rowidx = NULL;
    A->blockinfo = NULL;
    A->diag = NULL;
    A->list = NULL;
    
    return DEV_SUCCESS;
}
