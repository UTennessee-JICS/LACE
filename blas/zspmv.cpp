#include <stdio.h>
#include "../include/sparse.h"
#include <mkl_spblas.h>

#define CALL_AND_CHECK_STATUS(function, error_message) \
  do { \
    if (function != SPARSE_STATUS_SUCCESS)                     \
    {                                                         \
      printf(error_message); fflush(0);                         \
      info = 1;                                                 \
      goto cleanup;                                             \
    }                                                         \
  } while (0)

/*******************************************************************************
*   Purpose
*   -------
*
*   For a given input sparse matrix A and vectors x, y and scalars alpha, beta
*   the wrapper determines the suitable SpMV computing
*             y = alpha * A * x + beta * y.
*   Arguments
*   ---------
*
*   @param[in]
*   alpha       dataType
*               scalar alpha
*
*   @param[in]
*   A           data_d_matrix
*               sparse matrix A
*
*   @param[in]
*   x           data_d_matrix
*               input vector x
*
*   @param[in]
*   beta        dataType
*               scalar beta
*   @param[out]
*   y           data_d_matrix
*               output vector y
*
*   @ingroup datasparse_zblas
*******************************************************************************/

extern "C"
int
data_z_spmv(
  dataType        alpha,
  data_d_matrix * A,
  data_d_matrix * x,
  dataType        beta,
  data_d_matrix * y)
{
  data_int_t info = 0;

  if (A->num_cols == x->num_rows && x->num_cols == 1) {
    if (A->storage_type == Magma_CSR || A->storage_type == Magma_CUCSR ||
      A->storage_type == Magma_CSRL ||
      A->storage_type == Magma_CSRU)
    {
      mkl_dcsrmv("N", &A->num_rows, &A->num_cols,
        &alpha, "GFNC", A->val,
        A->col, A->row, A->row + 1,
        x->val, &beta, y->val);
    } else   {
      printf("error: format not supported.\n");
      info = DEV_ERR_NOT_SUPPORTED;
    }
  } else   {
    printf("error: dimensions not matched A->num_cols=%d x->num_rows=%d.\n",
      A->num_cols, x->num_rows);
    info = DEV_ERR_NOT_SUPPORTED;
  }

  return info;
}
