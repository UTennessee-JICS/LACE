/*
 *  -- MAGMA (version 2.0) --
 *     Univ. of Tennessee, Knoxville
 *     Univ. of California, Berkeley
 *     Univ. of Colorado, Denver
 *     @date
 *
 *     @precisions normal z -> s d c
 *     @author Hartwig Anzt
 *     @author Stephen Wood
 *
 */
#include "sparse.h"

/**
 *  Purpose
 *  -------
 *
 *  Takes an strictly lower triangular matrix L and an upper triangular matrix U
 *  and merges them into a matrix A containing the upper and lower triangular
 *  parts.
 *
 *  Arguments
 *  ---------
 *
 *  @param[in]
 *  L           data_d_matrix
 *              input strictly lower triangular matrix L
 *
 *  @param[in]
 *  U           data_d_matrix
 *              input upper triangular matrix U
 *
 *  @param[out]
 *  A           data_d_matrix*
 *              output matrix
 *
 *  @ingroup magmasparse_zaux
 ********************************************************************/

extern "C"
int
data_zmlumerge(
  data_d_matrix   L,
  data_d_matrix   U,
  data_d_matrix * A)
{
  data_int_t info = 0;

  if (L.storage_type == Magma_CSR && U.storage_type == Magma_CSR) {
    CHECK(data_zmconvert(L, A, Magma_CSR, Magma_CSR));
    free(A->col);
    free(A->val);
    // make sure only values from strictly lower triangular elements of L are used
    data_int_t z = 0;
    for (data_int_t i = 0; i < A->num_rows; i++) {
      for (data_int_t j = L.row[i]; j < L.row[i + 1]; j++) {
        if (L.col[j] < i) { // skip diagonal elements
          z++;
        }
      }
      for (data_int_t j = U.row[i]; j < U.row[i + 1]; j++) {
        z++;
      }
    }
    A->nnz = z;
    // fill A with the new structure
    LACE_CALLOC(A->col, A->nnz);
    LACE_CALLOC(A->val, A->nnz);
    z = 0;
    for (data_int_t i = 0; i < A->num_rows; i++) {
      A->row[i] = z;
      for (data_int_t j = L.row[i]; j < L.row[i + 1]; j++) {
        if (L.col[j] < i) { // skip diagonal elements
          A->col[z] = L.col[j];
          A->val[z] = L.val[j];
          z++;
        }
      }
      for (data_int_t j = U.row[i]; j < U.row[i + 1]; j++) {
        A->col[z] = U.col[j];
        A->val[z] = U.val[j];
        z++;
      }
    }
    A->row[A->num_rows] = z;
    A->nnz = z;
  } else {
    DEV_PRINTF("%% warning: %s , within %s ; matrix in wrong formats L = %d, U = %d.\n",
      __FILE__, __FUNCTION__, L.storage_type, U.storage_type);
    data_d_matrix LL = { Magma_CSR };
    data_d_matrix UU = { Magma_CSR };

    CHECK(data_zmconvert(L, &LL, L.storage_type, Magma_CSR) );
    CHECK(data_zmconvert(U, &UU, U.storage_type, Magma_CSR) );
    info = data_zmlumerge(LL, UU, A);

    data_zmfree(&LL);
    data_zmfree(&UU);
  }
  // cleanup:
  if (info != 0) {
    data_zmfree(A);
  }
  return info;
} // data_zmlumerge
