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

#define CALL_AND_CHECK_STATUS(function, error_message) \
  do { \
    if (function != SPARSE_STATUS_SUCCESS)                     \
    {                                                         \
      printf(error_message); fflush(0);                         \
      info = 1;                                                 \
    }                                                         \
  } while (0)

extern "C"
int
data_PariLU_v4_0(data_d_matrix * A, data_d_matrix * L, data_d_matrix * U)
{
  data_d_matrix Atmp = { Magma_CSRCOO };

  data_zmconvert(*A, &Atmp, Magma_CSR, Magma_CSRCOO);

  // Separate the lower and upper elements into L and U respectively.
  L->diagorder_type = Magma_UNITY;
  data_zmconvert(*A, L, Magma_CSR, Magma_CSRL);
  printf("\nL:\n");
  data_zwrite_csr(L);

  U->diagorder_type = Magma_VALUE;
  data_zmconvert(*A, U, Magma_CSR, Magma_CSRU);
  printf("\nU:\n");
  data_zwrite_csr(U);

  // Separate the strictly lower and upper elements into L and U respectively
  // for spmm
  data_d_matrix Ls = { Magma_CSRL };
  Ls.diagorder_type = Magma_NODIAG;
  data_zmconvert(*A, &Ls, Magma_CSR, Magma_CSRL);
  printf("\nLs:\n");
  data_zwrite_csr(&Ls);

  data_d_matrix Us = { Magma_CSRU };
  Us.diagorder_type = Magma_NODIAG;
  data_zmconvert(*A, &Us, Magma_CSR, Magma_CSRU);
  printf("\nUs:\n");
  data_zwrite_csr(&Us);

  data_d_matrix Ud = { Magma_CSRU };
  Ud.diagorder_type = Magma_VALUE;
  data_zmconvert(*A, &Ud, Magma_CSR, Magma_CSRU);
  printf("\nUd:\n");
  data_zwrite_csr(&Ud);

  // PariLU element wise update
  int iter = 0;
  // dataType tmp = 0.0;
  dataType step  = 1.0;
  dataType tol   = 1.0e-15;
  dataType Anorm = 0.0;

  int num_threads = 0;

  // int i, j;
  int ii = 0;
  int coltmp;
  dataType valtmp;

  int info = 0;
  // int status = 0;
  int spmm_info   = 0;
  int export_info = 0;

  sparse_matrix_t csrL = NULL, csrU = NULL, csrC = NULL;
  dataType * values_C = NULL;
  int * pointerB_C = NULL, * pointerE_C = NULL, * columns_C = NULL;
  int rows_C, cols_C; // , nnz_C;
  sparse_index_base_t indexing = SPARSE_INDEX_BASE_ZERO;

  //// create MKL sparse matrix handles
  // CALL_AND_CHECK_STATUS( mkl_sparse_d_create_csr( &csrL, indexing,
  //  L->num_rows, L->num_cols, L->row,
  //  L->row+1, L->col, L->val ),
  //  "Error after MKL_SPARSE_D_CREATE_CSR, csrL\n");
  //
  // CALL_AND_CHECK_STATUS( mkl_sparse_d_create_csr( &csrU, indexing,
  //  U->num_rows, U->num_cols, U->row,
  //  U->row+1, U->col, U->val ),
  //  "Error after MKL_SPARSE_D_CREATE_CSR, csrU\n");

  // create MKL sparse matrix handles
  CALL_AND_CHECK_STATUS(mkl_sparse_d_create_csr(&csrL, indexing,
    Ls.num_rows, Ls.num_cols, Ls.row,
    Ls.row + 1, Ls.col, Ls.val),
    "Error after MKL_SPARSE_D_CREATE_CSR, csrL\n");

  CALL_AND_CHECK_STATUS(mkl_sparse_d_create_csr(&csrU, indexing,
    Us.num_rows, Us.num_cols, Us.row,
    Us.row + 1, Us.col, Us.val),
    "Error after MKL_SPARSE_D_CREATE_CSR, csrU\n");


  data_d_matrix C  = { Magma_CSR };
  data_d_matrix Lc = { Magma_CSRL };
  data_d_matrix Uc = { Magma_CSRU };

  data_zfrobenius(*A, &Anorm);
  printf("%% Anorm = %e\n", Anorm);

  dataType wstart = omp_get_wtime();
  while (step > tol) {
    // while ( iter < 10 ) {
    step = 0.0;

    // caLculate update
    spmm_info = mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE,
        csrL, csrU, &csrC);

    export_info = mkl_sparse_d_export_csr(csrC, &indexing,
        &rows_C, &cols_C,
        &pointerB_C, &pointerE_C, &columns_C, &values_C);

    if (spmm_info != SPARSE_STATUS_SUCCESS ||
      export_info != SPARSE_STATUS_SUCCESS)
    {
      printf("%% iter = %d spmm_info = %d, export_info = %d\n",
        iter, spmm_info, export_info);
      info = -1;
      return info;
    }

    // printf("\nC:\n");
    // for (int ip=0; ip<pointerE_C[ rows_C-1 ]; ip++) {
    //  printf("[?, %d] %e\t", columns_C[ip], values_C[ip]);
    // }
    // printf("\n");

    // printf("C rows = %d, C cols = %d, C nnz = %d\n", rows_C, cols_C, pointerE_C[ rows_C-1 ] );

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


    // fill in information for C
    C.storage_type = Magma_CSR;
    // C.sym = A.sym;
    C.diagorder_type = Magma_VALUE;
    C.fill_mode      = MagmaFull;
    C.num_rows       = rows_C;
    C.num_cols       = cols_C;
    C.nnz      = pointerE_C[ rows_C - 1 ];
    C.true_nnz = C.nnz;
    // memory allocation
    // CHECK( magma_zmalloc( &C.dval, nnz_C ));
    C.val = (dataType *) malloc(C.nnz * sizeof(dataType) );
    for (int i = 0; i < C.nnz; i++) {
      C.val[i] = values_C[i];
    }
    // CHECK( magma_index_malloc( &C.drow, rows_C + 1 ));
    C.row = (int *) malloc( (rows_C + 1) * sizeof(int));
    for (int i = 0; i < rows_C; i++) {
      C.row[i] = pointerB_C[i];
    }
    C.row[rows_C] = pointerE_C[rows_C - 1];
    // CHECK( magma_index_malloc( &C.dcol, nnz_C ));
    C.col = (int *) malloc(C.nnz * sizeof(int));
    for (int i = 0; i < C.nnz; i++) {
      C.col[i] = columns_C[i];
    }
    printf("\nC:\n");
    data_zwrite_csr(&C);

    Lc.diagorder_type = Magma_NODIAG;
    data_zmconvert(C, &Lc, Magma_CSR, Magma_CSRL);
    printf("\nLc:\n");
    data_zwrite_csr(&Lc);

    Uc.diagorder_type = Magma_VALUE;
    data_zmconvert(C, &Uc, Magma_CSR, Magma_CSRU);
    printf("\nUc:\n");
    data_zwrite_csr(&Uc);


    // Us[i,j] = A[i,j] - Uc[i,j]
    // data_zsubtract_guided_csr( &Us, U, &Uc );
    // printf("Us - Uc complete\n");
    data_zsubtract_guided_csr(&Ud, U, &Uc, &step);
    printf("Ud - Uc complete\n");
    data_zset_csr(&Us, &Ud);

    // Ls[i,j] = A[i,j] - Lc[i,j]
    data_zsubtract_guided_csr(&Ls, L, &Lc, &step);
    printf("Ls - Lc complete\n");

    // Ls[i,j] /= U[j,j]
    data_zdiagdivide_csr(&Ls, &Ud);
    printf("Ls /= Ud complete\n");


    step /= Anorm;
    iter++;
    printf("%% iteration = %d step = %e\n", iter, step);
  }
  dataType wend     = omp_get_wtime();
  dataType ompwtime = (dataType) (wend - wstart) / ((dataType) iter);

  data_zset_csr(L, &Ls);
  data_zset_csr(U, &Ud);

  printf("\nL:\n");
  data_zwrite_csr(L);

  printf(
    "%% ParLU v4.0 used %d OpenMP threads and required %d iterations, %f wall clock seconds, and an average of %f wall clock seconds per iteration as measured by omp_get_wtime()\n",
    num_threads, iter, wend - wstart, ompwtime);
  data_zmfree(&Atmp);


  // cleanup:
  return info;
} // data_PariLU_v4_0
