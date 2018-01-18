#include <stdio.h>
#include "../include/sparse.h"

extern "C" data_int_t
data_zvinit(
  data_d_matrix * x,
  data_int_t      num_rows,
  data_int_t      num_cols,
  dataType        value,
  data_order_t    major)
{
  data_int_t info = 0;

  data_zmfree(x);

  x->val            = NULL;
  x->diag           = NULL;
  x->row            = NULL;
  x->rowidx         = NULL;
  x->col            = NULL;
  x->blockinfo      = NULL;
  x->storage_type   = Magma_DENSE;
  x->sym            = Magma_GENERAL;
  x->diagorder_type = Magma_VALUE;
  x->fill_mode      = MagmaFull;
  x->num_rows       = num_rows;
  x->num_cols       = num_cols;
  x->pad_rows       = 0;
  x->pad_cols       = 0;
  x->nnz            = num_rows * num_cols;
  x->max_nnz_row    = num_cols;
  x->diameter       = 0;
  x->blocksize      = 1;
  x->numblocks      = 1;
  x->alignment      = 1;
  x->major          = major; // data_order_t major = MagmaColMajor is default see include/sparse.h
  x->ld = num_rows;

  LACE_CALLOC(x->val, x->nnz);

  for (data_int_t i = 0; i < x->nnz; ++i) {
    x->val[i] = value;
  }

  return info;
} // data_zvinit
