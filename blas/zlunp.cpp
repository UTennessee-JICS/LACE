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
void
data_LUnp_mkl(data_d_matrix * A)
{
  LAPACKE_mkl_dgetrfnpi(A->major, A->num_rows, A->num_cols,
    A->num_rows, A->val, A->ld);
}
