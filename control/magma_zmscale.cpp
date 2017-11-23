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
#include <stdio.h>
#include "../include/sparse.h"

/**
 *  Purpose
 *  -------
 *
 *  Scales a matrix.
 *
 *  Arguments
 *  ---------
 *
 *  @param[in,out]
 *  A           data_d_matrix*
 *              input/output matrix
 *
 *  @param[in]
 *  scaling     data_scale_t
 *              scaling type (unit rownorm / unit diagonal)
 *
 *  @ingroup datasparse_zaux
 ********************************************************************/

extern "C"
int
data_zmscale(
  data_d_matrix * A,
  data_scale_t    scaling)
{
  int info = 0;

  dataType * tmp = NULL;

  data_d_matrix CSRA = { Magma_CSR };

  if (A->num_rows != A->num_cols && scaling != Magma_NOSCALE) {
    printf("%% warning: non-square matrix.\n");
    printf("%% Fallback: no scaling.\n");
    scaling = Magma_NOSCALE;
  }


  if (A->storage_type == Magma_CSRCOO) {
    if (scaling == Magma_NOSCALE) {
      // no scale
    } else if (A->num_rows == A->num_cols)   {
      if (scaling == Magma_UNITROW) {
        // scale to unit rownorm
        // tmp = (dataType*) calloc( A->num_rows, sizeof(dataType) );
        LACE_CALLOC(tmp, A->num_rows);
        for (int z = 0; z < A->num_rows; z++) {
          dataType s = 0.0;
          for (int f = A->row[z]; f < A->row[z + 1]; f++)
            s += A->val[f] * A->val[f];
          tmp[z] = 1.0 / sqrt(s);
        }
        for (int z = 0; z < A->nnz; z++) {
          A->val[z] = A->val[z] * tmp[A->col[z]] * tmp[A->rowidx[z]];
        }
      } else if (scaling == Magma_UNITDIAG)   {
        // scale to unit diagonal
        // tmp = (dataType*) calloc( A->num_rows, sizeof(dataType) );
        LACE_CALLOC(tmp, A->num_rows);
        for (int z = 0; z < A->num_rows; z++) {
          dataType s = 0.0;
          for (int f = A->row[z]; f < A->row[z + 1]; f++) {
            if (A->col[f] == z) {
              // add some identity matrix
              // A->val[f] = A->val[f] +  MAGMA_Z_MAKE( 100000.0, 0.0 );
              s = A->val[f];
            }
          }
          if (s == 0.0) {
            printf("%%error: zero diagonal element.\n");
            info = DEV_ERR;
          }
          tmp[z] = 1.0 / sqrt(s);
        }
        for (int z = 0; z < A->nnz; z++) {
          A->val[z] = A->val[z] * tmp[A->col[z]] * tmp[A->rowidx[z]];
        }
      } else   {
        printf("%%error: scaling not supported.\n");
        info = DEV_ERR_NOT_SUPPORTED;
      }
    } else   {
      printf("%%error: scaling not supported.\n");
      info = DEV_ERR_NOT_SUPPORTED;
    }
  } else   {
    data_storage_t A_storage = A->storage_type;
    data_zmconvert(*A, &CSRA, A->storage_type, Magma_CSRCOO);

    data_zmscale(&CSRA, scaling);

    data_zmfree(A);
    data_zmconvert(CSRA, A, Magma_CSRCOO, A_storage);
  }

  // cleanup:
  free(tmp);
  data_zmfree(&CSRA);
  return info;
} // data_zmscale

/**
 *  Purpose
 *  -------
 *
 *  Scales a matrix and a right hand side vector of a Ax = b system.
 *
 *  Arguments
 *  ---------
 *
 *  @param[in,out]
 *  A           data_d_matrix*
 *              input/output matrix
 *
 *  @param[in,out]
 *  b           data_d_matrix*
 *              input/output right hand side vector
 *
 *  @param[out]
 *  scaling_factors
 *              data_d_matrix*
 *              output scaling factors vector
 *
 *  @param[in]
 *  scaling     data_scale_t
 *              scaling type (unit rownorm / unit diagonal)
 *
 *  @ingroup datasparse_zaux
 ********************************************************************/

extern "C"
int
data_zmscale_matrix_rhs(
  data_d_matrix * A,
  data_d_matrix * b,
  data_d_matrix * scaling_factors,
  data_scale_t    scaling)
{
  int info = 0;

  // just use scaling_factors !!! Thanks Taylor
  dataType * tmp = NULL;

  if (A->num_rows != A->num_cols && scaling != Magma_NOSCALE) {
    printf("%% warning: non-square matrix.\n");
    printf("%% Fallback: no scaling.\n");
    scaling = Magma_NOSCALE;
  }

  if (A->storage_type == Magma_CSRCOO) { } else   {
    data_rowindex(A, &A->rowidx);
  }

  if (scaling == Magma_NOSCALE) {
    // no scale
  } else if (A->num_rows == A->num_cols)   {
    if (scaling == Magma_NOSCALE) { } else   {
      if (scaling == Magma_UNITROW) {
        // scale to unit rownorm by rows
        // tmp = (dataType*) calloc( A->num_rows, sizeof(dataType) );
        LACE_CALLOC(tmp, A->num_rows);
        for (int z = 0; z < A->num_rows; z++) {
          dataType s = 0.0;
          for (int f = A->row[z]; f < A->row[z + 1]; f++)
            s += A->val[f] * A->val[f];
          tmp[z] = 1.0 / sqrt(s);
        }
        #pragma omp parallel
        {
          #pragma omp for nowait
          for (int z = 0; z < A->nnz; z++) {
            A->val[z] = A->val[z] * tmp[A->rowidx[z]];
          }
        }
      } else if (scaling == Magma_UNITDIAG)   {
        // scale to unit diagonal by rows
        // tmp = (dataType*) calloc( A->num_rows, sizeof(dataType) );
        LACE_CALLOC(tmp, A->num_rows);
        for (int z = 0; z < A->num_rows; z++) {
          dataType s = 0.0;
          for (int f = A->row[z]; f < A->row[z + 1]; f++) {
            if (A->col[f] == z) {
              s = A->val[f];
            }
          }
          if (s == 0.0) {
            printf("%%error: zero diagonal element.\n");
            info = DEV_ERR;
          }
          tmp[z] = 1.0 / s;
        }
        #pragma omp parallel
        {
          #pragma omp for nowait
          for (int z = 0; z < A->nnz; z++) {
            A->val[z] = A->val[z] * tmp[A->rowidx[z]];
          }
        }
        // scaling_factors->num_rows = A->num_rows;
        // scaling_factors->num_cols = 1;
        // scaling_factors->ld = 1;
        // scaling_factors->nnz = A->num_rows;
        // scaling_factors->val = NULL;
        // scaling_factors->val = (dataType*) calloc( A->num_rows, sizeof(dataType) );
        // for ( int i=0; i<A->num_rows; i++ ) {
        //  scaling_factors->val[i] = tmp[i];
        //  b->val[i] = b->val[i] * tmp[i];
        // }
      } else if (scaling == Magma_UNITROWCOL)   {
        // scale to unit rownorm by rows and columns
        // tmp = (dataType*) calloc( A->num_rows, sizeof(dataType) );
        LACE_CALLOC(tmp, A->num_rows);
        for (int z = 0; z < A->num_rows; z++) {
          dataType s = 0.0;
          for (int f = A->row[z]; f < A->row[z + 1]; f++)
            s += A->val[f] * A->val[f];
          tmp[z] = 1.0 / sqrt(s);
        }
        #pragma omp parallel
        {
          #pragma omp for nowait
          for (int z = 0; z < A->nnz; z++) {
            A->val[z] = A->val[z] * tmp[A->col[z]] * tmp[A->rowidx[z]];
          }
        }
        // scaling_factors->num_rows = A->num_rows;
        // scaling_factors->num_cols = 1;
        // scaling_factors->ld = 1;
        // scaling_factors->nnz = A->num_rows;
        // scaling_factors->val = NULL;
        // scaling_factors->val = (dataType*) calloc( A->num_rows, sizeof(dataType) );
        // for ( int i=0; i<A->num_rows; i++ ) {
        //  scaling_factors->val[i] = tmp[i];
        //  b->val[i] = b->val[i] * tmp[i];
        // }
      } else if (scaling == Magma_UNITDIAGCOL)   {
        // scale to unit diagonal by rows and columns
        // tmp = (dataType*) calloc( A->num_rows, sizeof(dataType) );
        LACE_CALLOC(tmp, A->num_rows);
        for (int z = 0; z < A->num_rows; z++) {
          dataType s = 0.0;
          for (int f = A->row[z]; f < A->row[z + 1]; f++) {
            if (A->col[f] == z) {
              s = A->val[f];
            }
          }
          if (s == 0.0) {
            printf("%%error: zero diagonal element.\n");
            info = DEV_ERR;
          }
          tmp[z] = 1.0 / sqrt(s);
        }
        #pragma omp parallel
        {
          #pragma omp for nowait
          for (int z = 0; z < A->nnz; z++) {
            A->val[z] = A->val[z] * tmp[A->col[z]] * tmp[A->rowidx[z]];
          }
        }
        // scaling_factors->num_rows = A->num_rows;
        // scaling_factors->num_cols = 1;
        // scaling_factors->ld = 1;
        // scaling_factors->nnz = A->num_rows;
        // scaling_factors->val = NULL;
        // scaling_factors->val = (dataType*) calloc( A->num_rows, sizeof(dataType) );
        // for ( int i=0; i<A->num_rows; i++ ) {
        //  scaling_factors->val[i] = tmp[i];
        //  b->val[i] = b->val[i] * tmp[i];
        // }
      } else   {
        printf("%%error: scaling not supported.\n");
        info = DEV_ERR_NOT_SUPPORTED;
      }

      if (info != DEV_ERR_NOT_SUPPORTED) {
        scaling_factors->num_rows = A->num_rows;
        scaling_factors->num_cols = 1;
        scaling_factors->ld       = 1;
        scaling_factors->nnz      = A->num_rows;
        scaling_factors->val      = NULL;
        // scaling_factors->val = (dataType*) calloc( A->num_rows, sizeof(dataType) );
        LACE_CALLOC(tmp, A->num_rows);
        #pragma omp parallel
        {
          #pragma omp for nowait
          for (int i = 0; i < A->num_rows; i++) {
            scaling_factors->val[i] = tmp[i];
            b->val[i] = b->val[i] * tmp[i];
          }
        }
      }

      // need for scaling by columns???

      // return scaling factors always???

      // return only scaling factors and leave application to the
      // right hand side and solution vector to separate operations???
    }
  } else   {
    printf("%%error: scaling not supported.\n");
    info = DEV_ERR_NOT_SUPPORTED;
  }

  free(tmp);
  return info;
} // data_zmscale_matrix_rhs

/**
 *  Purpose
 *  -------
 *
 *  Adds a multiple of the Identity matrix to a matrix: A = A+add * I
 *
 *  Arguments
 *  ---------
 *
 *  @param[in,out]
 *  A           data_d_matrix*
 *              input/output matrix
 *
 *  @param[in]
 *  add         dataType
 *              scaling for the identity matrix
 *
 *  @ingroup datasparse_zaux
 ********************************************************************/

extern "C"
int
data_zmdiagadd(
  data_d_matrix * A,
  dataType        add)
{
  int info = 0;

  data_d_matrix CSRA = { Magma_CSR };

  if (A->storage_type == Magma_CSRCOO) {
    for (int z = 0; z < A->nnz; z++) {
      if (A->col[z] == A->rowidx[z]) {
        // add some identity matrix
        A->val[z] = A->val[z] + add;
      }
    }
  } else   {
    data_storage_t A_storage = A->storage_type;
    data_zmconvert(*A, &CSRA, A->storage_type, Magma_CSRCOO);

    data_zmdiagadd(&CSRA, add);

    data_zmfree(A);
    data_zmconvert(CSRA, A, Magma_CSRCOO, A_storage);
  }

  // cleanup:
  data_zmfree(&CSRA);
  return info;
}

/**
 *  Purpose
 *  -------
 *
 *  Generates n vectors of scaling factors from the A matrix
 *  and stores them in the factors matrix as column vectors in
 *  column major ordering.
 *
 *  Arguments
 *  ---------
 *
 *  @param[in]
 *  n           int
 *              number of diagonal scaling matrices
 *
 *  @param[in]
 *  scaling     data_scale_t*
 *              array of scaling specifiers
 *
 *  @param[in]
 *  side        data_side_t*
 *              array of side specifiers
 *
 *  @param[in]
 *  A           data_d_matrix*
 *              input matrix
 *
 *  @param[out]
 *  scaling_factors  data_d_matrix*
 *              array of diagonal matrices
 *
 *  @ingroup datasparse_zaux
 ********************************************************************/

extern "C"
int
data_zmscale_generate(
  int             n,
  data_scale_t *  scaling,
  data_side_t *   side,
  data_d_matrix * A,
  data_d_matrix * scaling_factors)
{
  int info = 0;

  data_d_matrix hA = { Magma_CSR }, CSRA = { Magma_CSR };

  printf("n = %d scale = %d side = %d\n", n, scaling[0], side[0]);

  if (A->num_rows != A->num_cols && scaling[0] != Magma_NOSCALE) {
    printf("%% warning: non-square matrix.\n");
    printf("%% Fallback: no scaling.\n");
    scaling[0] = Magma_NOSCALE;
  }


  // if ( A->storage_type == Magma_CSRCOO ) {
  for (int j = 0; j < n; j++) {
    printf("%% scaling[%d] = %d n=%d\n", j, scaling[j], n);
    if (scaling[j] == Magma_NOSCALE) {
      // no scale
    } else if (A->num_rows == A->num_cols)   {
      if (scaling[j] == Magma_UNITROW && side[j] != MagmaBothSides) {
        // scale to unit rownorm
        for (int z = 0; z < A->num_rows; z++) {
          dataType s = 0.0;
          for (int f = A->row[z]; f < A->row[z + 1]; f++)
            s += A->val[f] * A->val[f];
          scaling_factors[j].val[z] = 1.0 / sqrt(s);
        }
      } else if (scaling[j] == Magma_UNITDIAG && side[j] != MagmaBothSides)   {
        // scale to unit diagonal
        for (int z = 0; z < A->num_rows; z++) {
          dataType s = 0.0;
          for (int f = A->row[z]; f < A->row[z + 1]; f++) {
            if (A->col[f] == z) {
              s = A->val[f];
            }
          }
          if (s == 0.0) {
            printf("%%error: zero diagonal element.\n");
            info = DEV_ERR;
          }
          scaling_factors[j].val[z] = 1.0 / s;
        }
      } else if (scaling[j] == Magma_UNITCOL && side[j] != MagmaBothSides)   {
        // scale to unit column norm
        CHECK(data_zmtranspose(*A, &CSRA) );
        data_scale_t tscale = Magma_UNITROW;
        data_zmscale_generate(1, &tscale, &side[j], &CSRA,
          &scaling_factors[j]);
      } else if (scaling[j] == Magma_UNITROW && side[j] == MagmaBothSides)   {
        // scale to unit rownorm by rows and columns
        for (int z = 0; z < A->num_rows; z++) {
          dataType s = 0.0;
          for (int f = A->row[z]; f < A->row[z + 1]; f++)
            s += A->val[f] * A->val[f];
          scaling_factors[j].val[z] = 1.0 / s;
        }
      } else if (scaling[j] == Magma_UNITDIAG && side[j] == MagmaBothSides)   {
        // scale to unit diagonal by rows and columns
        printf("scale to unit diagonal by rows and columns\n");
        for (int z = 0; z < A->num_rows; z++) {
          dataType s = 0.0;
          for (int f = A->row[z]; f < A->row[z + 1]; f++) {
            if (A->col[f] == z) {
              s = A->val[f];
            }
          }
          if (s == 0.0) {
            printf("%%error: zero diagonal element.\n");
            info = DEV_ERR;
          }
          scaling_factors[j].val[z] = 1.0 / sqrt(s);
        }
      } else if (scaling[j] == Magma_UNITCOL && side[j] == MagmaBothSides)   {
        // scale to unit column norm
        CHECK(data_zmtranspose(*A, &CSRA) );
        data_scale_t tscale = Magma_UNITROW;
        data_zmscale_generate(1, &tscale, &side[j], &CSRA,
          &scaling_factors[j]);
      } else   {
        printf("%%error: scaling %d not supported line = %d.\n",
          scaling[j], __LINE__);
        info = DEV_ERR_NOT_SUPPORTED;
      }
    } else   {
      printf("%%error: scaling of non-square matrices %d not supported line = %d.\n",
        scaling[0], __LINE__);
      info = DEV_ERR_NOT_SUPPORTED;
    }
  }
  // }
  // else {
  //    magma_storage_t A_storage = A->storage_type;
  //    magma_location_t A_location = A->memory_location;
  //    CHECK( magma_zmtransfer( *A, &hA, A->memory_location, Magma_CPU, queue ));
  //    CHECK( magma_zmconvert( hA, &CSRA, hA.storage_type, Magma_CSRCOO, queue ));
  //
  //    CHECK( magma_zmscale_generate( n, scaling, side, &CSRA, scaling_factors, queue ));
  //
  //    magma_zmfree( &hA, queue );
  //    magma_zmfree( A, queue );
  //    CHECK( magma_zmconvert( CSRA, &hA, Magma_CSRCOO, A_storage, queue ));
  //    CHECK( magma_zmtransfer( hA, A, Magma_CPU, A_location, queue ));
  // }


  // cleanup:
  data_zmfree(&hA);
  data_zmfree(&CSRA);
  return info;
} // data_zmscale_generate

/**
 *  Purpose
 *  -------
 *
 *  Applies n diagonal scaling matrices to a matrix A;
 *  n=[1,2], factor[i] is applied to side[i] of the matrix.
 *
 *  Arguments
 *  ---------
 *
 *  @param[in]
 *  n           int
 *              number of diagonal scaling matrices
 *
 *  @param[in]
 *  side        data_side_t*
 *              array of side specifiers
 *
 *  @param[in]
 *  scaling_factors  data_d_matrix*
 *              array of diagonal matrices
 *
 *  @param[in,out]
 *  A           data_d_matrix*
 *              input/output matrix
 *
 *  @ingroup datasparse_zaux
 ********************************************************************/

extern "C"
int
data_zmscale_apply(
  int             n,
  data_side_t *   side,
  data_d_matrix * scaling_factors,
  data_d_matrix * A)
{
  int info = 0;

  // if ( A->storage_type == Magma_CSRCOO ) {
  if (A->rowidx == NULL) {
    printf("creating rowidx\n");
    fflush(stdout);
    CHECK(data_rowindex(A, &(A->rowidx) ));
  }

  for (int j = 0; j < n; j++) {
    if (A->num_rows == A->num_cols) {
      if (side[j] == MagmaLeft) {
        // scale by rows
        for (int z = 0; z < A->nnz; z++) {
          A->val[z] = A->val[z] * scaling_factors[j].val[A->rowidx[z]];
        }
      } else if (side[j] == MagmaBothSides)   {
        printf("scale by rows and columns \n");
        fflush(stdout);
        // scale by rows and columns
        for (int z = 0; z < A->nnz; z++) {
          A->val[z] = A->val[z]
            * scaling_factors[j].val[A->col[z]]
            * scaling_factors[j].val[A->rowidx[z]];
        }
      } else if (side[j] == MagmaRight)   {
        // scale by columns
        for (int z = 0; z < A->nnz; z++) {
          A->val[z] = A->val[z] * scaling_factors[j].val[A->rowidx[z]];
        }
      }
    }
  }

  // }
  // else {
  //    DEV_CHECKPT
  //    CHECK( data_rowindex( A, &A->rowidx ));
  //    data_storage_t Astore = A->storage_type;
  //    A->storage_type = Magma_CSRCOO;
  //    DEV_CHECKPT
  //    CHECK( data_zmscale_apply( n, side, scaling_factors, A ));
  //    free( A->rowidx );
  //    A->storage_type = Astore;
  //    DEV_CHECKPT
  // }


  // cleanup:

  return info;
} // data_zmscale_apply

/**
 *  Purpose
 *  -------
 *
 *  Multiplies a diagonal matrix (vecA) and a vector (vecB).
 *
 *  Arguments
 *  ---------
 *
 *  @param[in]
 *  vecA        data_d_matrix*
 *              input matrix
 *
 *  @param[in/out]
 *  vecB        data_d_matrix*
 *              input/output matrix
 *
 *
 *  @ingroup datasparse_zaux
 ********************************************************************/

extern "C"
int
data_zdimv(
  data_d_matrix * vecA,
  data_d_matrix * vecB)
{
  int info = 0;

  info = data_zlascl2(
    vecB->fill_mode, vecB->num_rows, vecB->num_cols,
    vecA->val,
    vecB->val, vecB->ld);

  return info;
}

extern "C"
int
data_zlascl2(
  data_type_t type, int m, int n,
  dataType * dD,
  dataType * dA,
  int ldda)
{
  int info = 0;

  if (type != MagmaLower && type != MagmaUpper && type != MagmaFull)
    info = -1;
  else if (m < 0)
    info = -2;
  else if (n < 0)
    info = -3;
  else if (ldda < MAX(1, m) )
    info = -5;

  if (info != 0) {
    return info;
  }

  for (int j = 0; j < n; j++) {
    for (int i = 0; i < m; i++) {
      double mul = dD[i];
      dA[i + j * ldda] *= mul;
    }
  }

  return info;
}
