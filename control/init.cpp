
#include <stdio.h>
#include "../include/sparse.h"

extern "C" data_int_t
data_zvinit(
    data_d_matrix *x,
    data_int_t num_rows,
    data_int_t num_cols,
    dataType values )
{
    data_int_t info = 0;
    
    x->val = NULL;
    x->diag = NULL;
    x->row = NULL;
    x->rowidx = NULL;
    x->col = NULL;
    x->blockinfo = NULL;
    x->storage_type = Magma_DENSE;
    x->sym = Magma_GENERAL;
    x->diagorder_type = Magma_VALUE;
    x->fill_mode = MagmaFull;
    x->num_rows = num_rows;
    x->num_cols = num_cols;
    x->nnz = num_rows*num_cols;
    x->max_nnz_row = num_cols;
    x->diameter = 0;
    x->blocksize = 1;
    x->numblocks = 1;
    x->alignment = 1;
    x->major = MagmaColMajor;
    x->ld = num_rows;
    
    LACE_CALLOC( x->val, x->nnz );
    
    for( data_int_t i=0; i<x->nnz; i++) {
         x->val[i] = values;
    }
    
    return info; 
}
