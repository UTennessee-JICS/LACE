/*
 *  -- LACE (version 0.0) --
 *     Univ. of Tennessee, Knoxville
 *
 *     @author Stephen Wood
 *
 */
#include "../include/sparse.h"
#include <mkl.h>
#include <float.h>
#include <stdlib.h>
#include <stdio.h>

/**
 *  Purpose
 *  -------
 *
 *  Householder triangularization.  [U,R] = hqrd(X);
 *  Generators of Householder reflections stored in U->
 *  H_k = I - U(:,k)*U(:,k)'.
 *  prod(H_m ... H_1)X = [R; 0]
 *  where m = min(size(X))
 *  G. W. Stewart, "Matrix Algorithms, Volume 1", SIAM, 1998.
 *
 *  Arguments
 *  ---------
 *
 *  @param[in]
 *  n           int
 *              number of rows elements in X
 *
 *  @param[in,out]
 *  X           dataType *
 *              vector to be reflected
 *
 *  @param[in,out]
 *  u           dataType *
 *              reflected vector
 *
 *  @param[in]
 *  nu          dataType *
 *              norm of X
 *
 *  @ingroup datasparse_orthogonality
 ********************************************************************/


extern "C"
void
data_modified_gram_schmidt(int search,
  data_d_matrix *              krylov,
  data_d_matrix *              h,
  data_d_matrix *              u)
{
  GMRESDBG("data_modified_gram_schmidt begin\n");

  data_int_t n = krylov->num_rows;

  // Modified Gram-Schmidt
  for (int j = 0; j <= search; j++) {
    h->val[idx(j, search, h->ld)] = 0.0;
    for (int i = 0; i < n; i++) {
      h->val[idx(j, search, h->ld)] = h->val[idx(j, search, h->ld)]
        + krylov->val[idx(i, j, krylov->ld)] * u->val[i];
    }
    for (int i = 0; i < n; i++) {
      u->val[i] = u->val[i]
        - h->val[idx(j, search, h->ld)] * krylov->val[idx(i, j, krylov->ld)];
      GMRESDBG("\tu->val[%d] = %e\n", i, u->val[i]);
    }
  }
  h->val[idx((search + 1), search, h->ld)] = data_dnrm2(n, u->val, 1);

  GMRESDBG("h->val[idx(search,search,h->ld)] =%e\n", h->val[idx(search, search, h->ld)]);
  GMRESDBG("h->val[idx((search+1),search,h->ld)] =%e\n", h->val[idx((search + 1), search, h->ld)]);


  // Watch out for happy breakdown
  if (fabs(h->val[idx((search + 1), search, h->ld)]) > 0) {
    for (int i = 0; i < n; i++) {
      krylov->val[idx(i, (search + 1), krylov->ld)] =
        u->val[i] / h->val[idx((search + 1), search, h->ld)];
      GMRESDBG("--\tu->val[%d] = %e\n", i, u->val[i]);
      GMRESDBG("--\tkrylov->val[idx(%d,%d,%d)] = %e\n", i, (search + 1), krylov->ld,
        krylov->val[idx(i, (search + 1), krylov->ld)]);
    }
  } else {
    printf("%%\t******* happy breakdown **********\n");
  }
} // data_modified_gram_schmidt
