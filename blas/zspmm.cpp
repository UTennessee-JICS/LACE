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
*   For a given input sparse matrices A and B, and scalar alpha
*   the wrapper determines the suitable SpMM computing
*             C = alpha * A * B.
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
*   B           data_d_matrix
*               sparse matrix B
*
*   @param[out]
*   C           data_d_matrix
*               sparse matrix C
*
*
*   @ingroup datasparse_zblas
*******************************************************************************/

extern "C"
int
data_z_spmm(
  dataType        alpha,
  data_d_matrix   A,
  data_d_matrix   B,
  data_d_matrix * C)
{
  int info = 0;
  sparse_matrix_t csrA = NULL, csrB = NULL, csrC = NULL;
  dataType * values_C = NULL;
  int * pointerB_C = NULL, * pointerE_C = NULL, * columns_C = NULL;
  int rows_C, cols_C, nnz_C;
  sparse_index_base_t indexing = SPARSE_INDEX_BASE_ZERO;
  int ii = 0;
  int coltmp;
  dataType valtmp;

  // create MKL sparse matrix handles
  CALL_AND_CHECK_STATUS(mkl_sparse_d_create_csr(&csrA, indexing,
    A.num_rows, A.num_cols, A.row,
    A.row + 1, A.col, A.val),
    "Error after MKL_SPARSE_D_CREATE_CSR, csrA\n");

  CALL_AND_CHECK_STATUS(mkl_sparse_d_create_csr(&csrB, indexing,
    B.num_rows, B.num_cols, B.row,
    B.row + 1, B.col, B.val),
    "Error after MKL_SPARSE_D_CREATE_CSR, csrB\n");

  if (A.num_cols == B.num_rows) {
    if (A.storage_type == Magma_CSR ||
      A.storage_type == Magma_CSRL ||
      A.storage_type == Magma_CSRU ||
      A.storage_type == Magma_CSRCOO)
    {
      // CALL_AND_CHECK_STATUS(
      //  mkl_sparse_spmm( SPARSE_OPERATION_NON_TRANSPOSE,
      //    csrA, csrB, &csrC ),
      //    "Error after MKL_SPARSE_SPMM\n");
      // info = DEV_SUCCESS;
      info = mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE,
          csrA, csrB, &csrC);
    } else   {
      printf("error: format not supported.\n");
      info = DEV_ERR_NOT_SUPPORTED;
    }
  }

  // CALL_AND_CHECK_STATUS( mkl_sparse_d_export_csr( csrC, &indexing,
  //  &rows_C, &cols_C,
  //  &pointerB_C, &pointerE_C, &columns_C, &values_C ),
  //  "Error after MKL_SPARSE_D_EXPORT_CSR\n");
  info = mkl_sparse_d_export_csr(csrC, &indexing,
      &rows_C, &cols_C,
      &pointerB_C, &pointerE_C, &columns_C, &values_C);

  // ensure column indices are in ascending order in every row
  // printf( "\n RESULTANT MATRIX C:\nrow# : (value, column) (value, column)\n" );
  ii = 0;
  for (int i = 0; i < rows_C; i++) {
    // printf("row#%d:", i); fflush(0);
    for (int j = pointerB_C[i]; j < pointerE_C[i]; j++) {
      // printf(" (%e, %6d)", values_C[ii], columns_C[ii] ); fflush(0);
      if (j + 1 < pointerE_C[i] && columns_C[ii] > columns_C[ii + 1]) {
        // printf("\nSWAP!!!\n");
        valtmp            = values_C[ii];
        values_C[ii]      = values_C[ii + 1];
        values_C[ii + 1]  = valtmp;
        coltmp            = columns_C[ii];
        columns_C[ii]     = columns_C[ii + 1];
        columns_C[ii + 1] = coltmp;
      }
      ii++;
    }
    // printf( "\n" );
  }
  // printf( "_____________________________________________________________________  \n" );

  nnz_C = pointerE_C[ rows_C - 1 ];

  // fill in information for C
  C->storage_type = A.storage_type;
  // C->sym = A.sym;
  C->diagorder_type = Magma_VALUE;
  C->fill_mode      = MagmaFull;
  C->num_rows       = rows_C;
  C->num_cols       = cols_C;
  C->nnz      = nnz_C;
  C->true_nnz = nnz_C;
  // memory allocation
  // CHECK( magma_zmalloc( &C->dval, nnz_C ));
  // C->val = (dataType*) malloc( nnz_C*sizeof(dataType) );
  LACE_CALLOC(C->val, nnz_C);
  for (int i = 0; i < nnz_C; i++) {
    C->val[i] = values_C[i] * alpha;
  }
  // CHECK( magma_index_malloc( &C->drow, rows_C + 1 ));
  // C->row = (int*) malloc ( (rows_C+1)*sizeof(int));
  LACE_CALLOC(C->row, (rows_C + 1) );
  for (int i = 0; i < rows_C; i++) {
    C->row[i] = pointerB_C[i];
  }
  C->row[rows_C] = pointerE_C[rows_C - 1];
  // CHECK( magma_index_malloc( &C->dcol, nnz_C ));
  // C->col = (int*) malloc ( (nnz_C)*sizeof(int));
  LACE_CALLOC(C->col, nnz_C);
  for (int i = 0; i < nnz_C; i++) {
    C->col[i] = columns_C[i];
  }

  if (mkl_sparse_destroy(csrA) != SPARSE_STATUS_SUCCESS) {
    printf(" Error after MKL_SPARSE_DESTROY, csrA \n");
    fflush(0);
    info = 1;
  }
  if (mkl_sparse_destroy(csrB) != SPARSE_STATUS_SUCCESS) {
    printf(" Error after MKL_SPARSE_DESTROY, csrB \n");
    fflush(0);
    info = 1;
  }
  if (mkl_sparse_destroy(csrC) != SPARSE_STATUS_SUCCESS) {
    printf(" Error after MKL_SPARSE_DESTROY, csrC \n");
    fflush(0);
    info = 1;
  }

cleanup:
  return info;
} // data_z_spmm

extern "C"
int
data_z_spmm_handle(
  dataType          alpha,
  sparse_matrix_t * csrA,
  sparse_matrix_t * csrB,
  data_d_matrix *   C)
{
  int info = 0;
  sparse_matrix_t csrC = NULL;
  dataType * values_C = NULL;
  int * pointerB_C = NULL, * pointerE_C = NULL, * columns_C = NULL;
  int rows_C, cols_C, nnz_C;
  sparse_index_base_t indexing = SPARSE_INDEX_BASE_ZERO;
  int ii = 0;
  int coltmp;
  dataType valtmp;

  //// create MKL sparse matrix handles
  info = mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE,
      *csrA, *csrB, &csrC);

  info = mkl_sparse_d_export_csr(csrC, &indexing,
      &rows_C, &cols_C,
      &pointerB_C, &pointerE_C, &columns_C, &values_C);

  // ensure column indices are in ascending order in every row
  // printf( "\n RESULTANT MATRIX C:\nrow# : (value, column) (value, column)\n" );
  ii = 0;
  for (int i = 0; i < rows_C; i++) {
    // printf("row#%d:", i); fflush(0);
    for (int j = pointerB_C[i]; j < pointerE_C[i]; j++) {
      // printf(" (%e, %6d)", values_C[ii], columns_C[ii] ); fflush(0);
      if (j + 1 < pointerE_C[i] && columns_C[ii] > columns_C[ii + 1]) {
        // printf("\nSWAP!!!\n");
        valtmp            = values_C[ii];
        values_C[ii]      = values_C[ii + 1];
        values_C[ii + 1]  = valtmp;
        coltmp            = columns_C[ii];
        columns_C[ii]     = columns_C[ii + 1];
        columns_C[ii + 1] = coltmp;
      }
      ii++;
    }
    // printf( "\n" );
  }
  // printf( "_____________________________________________________________________  \n" );

  nnz_C = pointerE_C[ rows_C - 1 ];

  // fill in information for C
  C->storage_type   = Magma_CSR;
  C->diagorder_type = Magma_VALUE;
  C->fill_mode      = MagmaFull;
  C->num_rows       = rows_C;
  C->num_cols       = cols_C;
  C->nnz      = nnz_C;
  C->true_nnz = nnz_C;
  // memory allocation
  // C->val = (dataType*) malloc( nnz_C*sizeof(dataType) );
  LACE_CALLOC(C->val, nnz_C);
  for (int i = 0; i < nnz_C; i++) {
    C->val[i] = values_C[i] * alpha;
  }
  // C->row = (int*) malloc ( (rows_C+1)*sizeof(int));
  LACE_CALLOC(C->row, (rows_C + 1) );
  for (int i = 0; i < rows_C; i++) {
    C->row[i] = pointerB_C[i];
  }
  C->row[rows_C] = pointerE_C[rows_C - 1];
  // C->col = (int*) malloc ( (nnz_C)*sizeof(int));
  LACE_CALLOC(C->col, nnz_C);
  for (int i = 0; i < nnz_C; i++) {
    C->col[i] = columns_C[i];
  }

  return info;
} // data_z_spmm_handle

extern "C"
int
data_z_spmm_batch(
  dataType          alpha,
  sparse_matrix_t * csrA,
  sparse_matrix_t * csrB,
  sparse_matrix_t * csrC)
{
  int info = 0;

  info = mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE,
      *csrA, *csrB, csrC);
  return info;
}
