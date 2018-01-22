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
#include <algorithm>
#include <ctime>
#include <math.h>
#include <cmath>
#include <cfloat>

extern "C"
int
data_forward_solve(data_d_matrix * L, data_d_matrix * x, data_d_matrix * rhs,
  const dataType tol, int * iter)
{
  int info = 0;

  if (L->storage_type == Magma_CSRL &&
    L->fill_mode == MagmaLower &&
    L->diagorder_type != Magma_NODIAG)
  {
    int j = 0;
    *iter = 0;
    dataType step       = 1.e8;
    dataType tmp        = 0.0;
    const dataType zero = dataType(0.0);

    while (step > tol) {
      step = zero;
      #pragma omp parallel
      {
        // #pragma omp for private(j, tmp) reduction(+:step) nowait
        #pragma omp for private(tmp) reduction(+:step) nowait
        for (int i = 0; i < L->num_rows; i++) {
          tmp = zero;
          for (int k = L->row[i]; k < L->row[i + 1] - 1; k++) {
            // j = L->col[k];
            // tmp += L->val[k]*x->val[j];
            tmp += L->val[k] * x->val[ L->col[k] ];
          }
          tmp       = (rhs->val[i] - tmp) / L->val[L->row[i + 1] - 1];
          step     += pow((x->val[i] - tmp), 2);
          x->val[i] = tmp;
        }
      }
      *iter = *iter + 1;
      printf("%% iteration = %d step = %e\n", *iter, step);
    }
  } else {
    info = -1;
    printf("L matrix storage %d and fill mode %d must be CSRL (%d) and lower (%d) for a forward solve.\n",
      L->storage_type, L->fill_mode, Magma_CSRL, MagmaLower);
  }

  return info;
} // data_forward_solve

extern "C"
int
data_backward_solve(data_d_matrix * U, data_d_matrix * x, data_d_matrix * rhs,
  const dataType tol, int * iter)
{
  int info = 0;

  if (U->storage_type == Magma_CSRU &&
    U->fill_mode == MagmaUpper &&
    U->diagorder_type != Magma_NODIAG)
  {
    int j = 0;
    *iter = 0;
    dataType step       = 1.e8;
    dataType tmp        = 0.0;
    const dataType zero = dataType(0.0);

    while (step > tol) {
      step = zero;
      #pragma omp parallel
      {
        // #pragma omp for private(j, tmp) reduction(+:step) nowait
        #pragma omp for private(tmp) reduction(+:step) nowait
        for (int i = U->num_rows - 1; i >= 0; i--) {
          tmp = zero;
          for (int k = U->row[i] + 1; k < U->row[i + 1]; k++) {
            // j = U->col[k];
            // tmp += U->val[k]*x->val[j];
            tmp += U->val[k] * x->val[ U->col[k] ];
          }
          tmp       = (rhs->val[i] - tmp) / U->val[U->row[i]];
          step     += pow((x->val[i] - tmp), 2);
          x->val[i] = tmp;
        }
      }
      *iter = *iter + 1;
      printf("%% iteration = %d step = %e\n", *iter, step);
    }
  } else {
    info = -1;
    printf("U matrix storage %d and fill mode %d must be CSRU (%d) and lower (%d) for a backward solve.\n",
      U->storage_type, U->fill_mode, Magma_CSRU, MagmaUpper);
  }

  return info;
} // data_backward_solve

extern "C"
int
data_forward_solve_permute(data_d_matrix * L, data_d_matrix * x, data_d_matrix * rhs,
  const dataType tol, int * iter)
{
  int info = 0;

  if (L->storage_type == Magma_CSRL &&
    L->fill_mode == MagmaLower &&
    L->diagorder_type != Magma_NODIAG)
  {
    int j = 0;
    *iter = 0;
    dataType step       = 1.e8;
    dataType tmp        = 0.0;
    const dataType zero = dataType(0.0);

    int * c;
    LACE_CALLOC(c, L->num_rows);
    #pragma omp parallel
    {
      #pragma omp for nowait
      for (int i = 0; i < L->num_rows; i++) {
        c[i] = i;
      }
    }
    std::srand(unsigned (std::time(0) ) );
    int i = 0;

    while (step > tol) {
      step = zero;
      #pragma omp parallel
      {
        #pragma omp for private(j, tmp) reduction(+:step) nowait
        for (int ii = 0; ii < L->num_rows; ii++) {
          i   = c[ii];
          tmp = zero;
          for (int k = L->row[i]; k < L->row[i + 1] - 1; k++) {
            j    = L->col[k];
            tmp += L->val[k] * x->val[j];
          }
          tmp       = (rhs->val[i] - tmp) / L->val[L->row[i + 1] - 1];
          step     += pow((x->val[i] - tmp), 2);
          x->val[i] = tmp;
        }
      }
      *iter = *iter + 1;
      printf("%% iteration = %d step = %e\n", *iter, step);

      std::random_shuffle(c, c + L->num_rows);
    }

    free(c);
  } else {
    info = -1;
    printf("L matrix storage %d and fill mode %d must be CSRL (%d) and lower (%d) for a forward solve.\n",
      L->storage_type, L->fill_mode, Magma_CSRL, MagmaLower);
  }

  return info;
} // data_forward_solve_permute

extern "C"
int
data_backward_solve_permute(data_d_matrix * U, data_d_matrix * x, data_d_matrix * rhs,
  const dataType tol, int * iter)
{
  int info = 0;

  if (U->storage_type == Magma_CSRU &&
    U->fill_mode == MagmaUpper &&
    U->diagorder_type != Magma_NODIAG)
  {
    int j = 0;
    *iter = 0;
    dataType step       = 1.e8;
    dataType tmp        = 0.0;
    const dataType zero = dataType(0.0);

    int * c;
    LACE_CALLOC(c, U->num_rows);
    #pragma omp parallel
    {
      #pragma omp for nowait
      for (int i = U->num_rows; i > 0; i--) {
        c[i] = i;
      }
    }
    std::srand(unsigned (std::time(0) ) );
    int i = 0;

    while (step > tol) {
      step = zero;
      #pragma omp parallel
      {
        #pragma omp for private(j, tmp) reduction(+:step) nowait
        for (int ii = U->num_rows - 1; ii >= 0; ii--) {
          i   = c[ii];
          tmp = zero;
          for (int k = U->row[i] + 1; k < U->row[i + 1]; k++) {
            j    = U->col[k];
            tmp += U->val[k] * x->val[j];
          }
          tmp       = (rhs->val[i] - tmp) / U->val[U->row[i]];
          step     += pow((x->val[i] - tmp), 2);
          x->val[i] = tmp;
        }
      }
      *iter = *iter + 1;
      printf("%% iteration = %d step = %e\n", *iter, step);

      std::random_shuffle(c, c + U->num_rows);
    }
  } else {
    info = -1;
    printf("U matrix storage %d and fill mode %d must be CSRU (%d) and lower (%d) for a backward solve.\n",
      U->storage_type, U->fill_mode, Magma_CSRU, MagmaUpper);
  }

  return info;
} // data_backward_solve_permute

// MKL/LAPACK like interface
extern "C"
int
data_parcsrtrsv(const data_uplo_t uplo, const data_storage_t storage,
  const data_diagorder_t diag,
  const int num_rows, const dataType * Aval, const int * row, const int * col,
  const dataType * rhsval, dataType * yval,
  const dataType tol, int * iter)
{
  int info = 0;

  if (storage == Magma_CSRL &&
    uplo == MagmaLower &&
    diag != Magma_NODIAG)
  {
    // int j = 0;
    *iter = 0;
    dataType step       = 1.e8;
    dataType tmp        = 0.0;
    const dataType zero = dataType(0.0);

    while (step > tol && std::isfinite(step) ) {
      step = zero;
      #pragma omp parallel
      {
        // #pragma omp for private(j, tmp) reduction(+:step) nowait
        #pragma omp for private(tmp) reduction(+:step) nowait
        for (int i = 0; i < num_rows; i++) {
          tmp = zero;
          for (int k = row[i]; k < row[i + 1] - 1; k++) {
            // j = col[k];
            // tmp += Aval[k]*yval[j];
            tmp += Aval[k] * yval[ col[k] ];
          }
          tmp     = (rhsval[i] - tmp) / Aval[row[i + 1] - 1];
          step   += pow((yval[i] - tmp), 2);
          yval[i] = tmp;
        }
      }
      *iter = *iter + 1;
      PARTRSVDBG("%% L iteration = %d step = %e\n", *iter, step);
    }
    // printf("ParCSRTRSV_L(%d) = %e\n", *iter, step);
  } else if (storage == Magma_CSRL ||
    uplo == MagmaLower)
  {
    info = -1;
    printf("L matrix storage %d and fill mode %d must be CSRL (%d) and lower (%d) for a forward solve.\n",
      storage, uplo, Magma_CSRL, MagmaLower);
  } else if (storage == Magma_CSRU &&
    uplo == MagmaUpper &&
    diag != Magma_NODIAG)
  {
    // int j = 0;
    *iter = 0;

    dataType step       = 1.e8;
    dataType tmp        = 0.0;
    const dataType zero = dataType(0.0);

    while (step > tol && std::isfinite(step) ) {
      step = zero;
      #pragma omp parallel
      {
        // #pragma omp for private(j, tmp) reduction(+:step) nowait
        #pragma omp for private(tmp) reduction(+:step) nowait
        for (int i = num_rows - 1; i >= 0; i--) {
          tmp = zero;
          for (int k = row[i] + 1; k < row[i + 1]; k++) {
            // j = col[k];
            // tmp += Aval[k]*yval[j];
            tmp += Aval[k] * yval[ col[k] ];
          }
          tmp     = (rhsval[i] - tmp) / Aval[row[i]];
          step   += pow((yval[i] - tmp), 2);
          yval[i] = tmp;
        }
      }
      *iter = *iter + 1;
      PARTRSVDBG("%% U iteration = %d step = %e\n", *iter, step);
    }
    // printf("ParCSRTRSV_U(%d) = %e\n", *iter, step);
  } else if (storage == Magma_CSRU ||
    uplo == MagmaUpper)
  {
    info = -1;
    printf("U matrix storage %d and fill mode %d must be CSRU (%d) and lower (%d) for a backward solve.\n",
      storage, uplo, Magma_CSRU, MagmaUpper);
  }


  return info;
} // data_parcsrtrsv

extern "C"
int
data_partrsv(
  const data_order_t     major,
  const data_uplo_t      uplo,
  const data_storage_t   storage,
  const data_diagorder_t diag,
  const int              num_rows,
  const dataType *       Aval,
  const int              lda,
  const dataType *       rhsval,
  const int              incr,
  dataType *             yval,
  const int              incx,
  const dataType         tol,
  int *                  iter)
{
  int info = 0;

  if (major == MagmaRowMajor) {
    if (storage == Magma_DENSEL &&
      uplo == MagmaLower &&
      diag != Magma_NODIAG)
    {
      *iter = 0;
      dataType step       = 1.e8;
      dataType tmp        = 0.0;
      const dataType zero = dataType(0.0);

      while (step > tol) {
        step = zero;
        #pragma omp parallel
        {
          #pragma omp for private(tmp) reduction(+:step) nowait
          for (int i = 0; i < num_rows; i++) {
            tmp = zero;
            for (int j = 0; j < i; j++) {
              tmp += Aval[j + i * lda] * yval[j];
            }
            tmp     = (rhsval[i] - tmp) / Aval[i + i * lda];
            step   += pow((yval[i] - tmp), 2);
            yval[i] = tmp;
          }
        }
        *iter = *iter + 1;
        // printf("%% iteration = %d step = %e\n", *iter, step);
      }
    } else if (storage == Magma_DENSEL ||
      uplo == MagmaLower)
    {
      info = -1;
      printf("L matrix storage %d and fill mode %d must be DENSEL (%d) and lower (%d) for a forward solve.\n",
        storage, uplo, Magma_DENSEL, MagmaLower);
    } else if (storage == Magma_DENSEU &&
      uplo == MagmaUpper &&
      diag != Magma_NODIAG)
    {
      *iter = 0;

      dataType step       = 1.e8;
      dataType tmp        = 0.0;
      const dataType zero = dataType(0.0);

      while (step > tol) {
        step = zero;
        #pragma omp parallel
        {
          #pragma omp for private(tmp) reduction(+:step) nowait
          for (int i = num_rows - 1; i >= 0; i--) {
            tmp = zero;
            for (int j = i + 1; j < num_rows; j++) {
              tmp += Aval[j + i * lda] * yval[j];
            }
            tmp     = (rhsval[i] - tmp) / Aval[i + i * lda];
            step   += pow((yval[i] - tmp), 2);
            yval[i] = tmp;
          }
        }
        *iter = *iter + 1;
        // printf("%% iteration = %d step = %e\n", *iter, step);
      }
    } else if (storage == Magma_DENSEU ||
      uplo == MagmaUpper)
    {
      info = -1;
      printf("U matrix storage %d and fill mode %d must be DENSEU (%d) and lower (%d) for a backward solve.\n",
        storage, uplo, Magma_DENSEU, MagmaUpper);
    }
  }

  return info;
} // data_partrsv

extern "C"
int
data_partrsv_dot(
  const data_order_t     major,
  const data_uplo_t      uplo,
  const data_storage_t   storage,
  const data_diagorder_t diag,
  const int              num_rows,
  dataType *             Aval,
  const int              lda,
  const dataType *       rhsval,
  const int              incr,
  dataType *             yval,
  const int              incx,
  const dataType         tol,
  int *                  iter)
{
  int info = 0;

  if (major == MagmaRowMajor) {
    if (storage == Magma_DENSEL &&
      uplo == MagmaLower &&
      diag != Magma_NODIAG)
    {
      *iter = 0;
      dataType step = 1.e8;
      dataType tmp  = 0.0;
      // dataType tmp2 = 0.0;
      // dataType relax = -1.0e-5; //5.0e-1;
      const dataType zero = dataType(0.0);

      while (step > tol) {
        step = zero;
        #pragma omp parallel
        {
          #pragma omp for private(tmp) reduction(+:step) nowait
          for (int i = 0; i < num_rows; i++) {
            tmp = zero;
            // for ( int j=0; j <i; j++) {
            //  tmp += Aval[j+i*lda]*yval[j];
            // }
            tmp = data_zdot_mkl(i, &Aval[i * lda], 1, yval, incx);
            // tmp2 = tmp;
            tmp   = (rhsval[i * incr] - tmp) / Aval[i + i * lda];
            step += pow((yval[i * incx] - tmp), 2);
            yval[i * incx] = tmp;
            // correct with error
            // tmp = rhsval[i] - (tmp2 + Aval[i+i*lda]*tmp);
            // if ( fabs( tmp ) > 0.0 ) {
            //  printf("%d %.16e %.16e\n", i, yval[i], tmp );
            //  yval[i] += relax*tmp*yval[i];
            // }
          }
        }
        *iter = *iter + 1;
        // printf("%% iteration = %d step = %e\n", *iter, step);
      }
    } else if (storage == Magma_DENSEL ||
      uplo == MagmaLower)
    {
      info = -1;
      printf("L matrix storage %d and fill mode %d must be DENSEL (%d) and lower (%d) for a forward solve.\n",
        storage, uplo, Magma_DENSEL, MagmaLower);
    } else if (storage == Magma_DENSEU &&
      uplo == MagmaUpper &&
      diag != Magma_NODIAG)
    {
      *iter = 0;

      dataType step       = 1.e8;
      dataType tmp        = 0.0;
      const dataType zero = dataType(0.0);

      while (step > tol) {
        step = zero;
        #pragma omp parallel
        {
          #pragma omp for private(tmp) reduction(+:step) nowait
          for (int i = num_rows - 1; i >= 0; i--) {
            tmp = zero;
            // for ( int j=i+1; j<num_rows; j++) {
            //  tmp += Aval[j+i*lda]*yval[j];
            // }
            tmp   = data_zdot_mkl( (num_rows - (i + 1)), &Aval[i + 1 + i * lda], 1, &yval[i + 1], incx);
            tmp   = (rhsval[i * incr] - tmp) / Aval[i + i * lda];
            step += pow((yval[i * incx] - tmp), 2);
            yval[i * incx] = tmp;
            // printf("yval[%d] = %e\n", i, tmp);
          }
        }
        *iter = *iter + 1;
        // printf("%% iteration = %d step = %e\n", *iter, step);
      }
    } else if (storage == Magma_DENSEU ||
      uplo == MagmaUpper)
    {
      info = -1;
      printf("U matrix storage %d and fill mode %d must be DENSEU (%d) and lower (%d) for a backward solve.\n",
        storage, uplo, Magma_DENSEU, MagmaUpper);
    }
  }

  return info;
} // data_partrsv_dot
