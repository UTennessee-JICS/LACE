/*
    -- LACE (version 0.0) --
       Univ. of Tennessee, Knoxville
       
       @author Stephen Wood

*/
#include "../include/sparse.h"
#include <mkl.h>
#include <stdlib.h>
#include <stdio.h>
#include <algorithm>
#include <ctime> 

extern "C" 
int
data_forward_solve( data_d_matrix* L, data_d_matrix* x, data_d_matrix* rhs,
  const dataType tol, int *iter ) 
{
  int info = 0;
  
  if ( L->storage_type == Magma_CSRL 
    && L->fill_mode == MagmaLower 
    && L->diagorder_type != Magma_NODIAG ) { 
  
    int j = 0;
    *iter = 0;
    dataType step = 1.e8;
    dataType tmp = 0.0;
    const dataType zero = dataType(0.0);
    
    while (step > tol) {
        
      step = zero;
      #pragma omp parallel
      {
        #pragma omp for private(j, tmp) reduction(+:step) nowait
        for ( int i=0; i < L->num_rows; i++ ) {
          tmp = zero;
          for ( int k=L->row[i]; k < L->row[i+1]-1; k++) {
            j = L->col[k];
            tmp += L->val[k]*x->val[j];  
          }
          tmp = (rhs->val[i] - tmp)/L->val[L->row[i+1] - 1];
          step += pow((x->val[i] - tmp), 2);
          x->val[i] = tmp;
        }
      }
      *iter = *iter + 1;
      printf("%% iteration = %d step = %e\n", *iter, step);
    }
  
  }
  else {
    info = -1;
    printf("L matrix storage %d and fill mode %d must be CSRL (%d) and lower (%d) for a forward solve.\n",
      L->storage_type, L->fill_mode, Magma_CSRL, MagmaLower );
    
  }
  
  return info;
}


extern "C" 
int
data_backward_solve( data_d_matrix* U, data_d_matrix* x, data_d_matrix* rhs,
  const dataType tol, int *iter ) 
{
  int info = 0;
  
  if ( U->storage_type == Magma_CSRU 
    && U->fill_mode == MagmaUpper 
    && U->diagorder_type != Magma_NODIAG ) { 
  
    int j = 0;
    *iter = 0;
    dataType step = 1.e8;
    dataType tmp = 0.0;
    const dataType zero = dataType(0.0);
    
    while (step > tol) {
        
      step = zero;
      #pragma omp parallel
      {
        #pragma omp for private(j, tmp) reduction(+:step) nowait
        for ( int i=U->num_rows-1; i>=0; i-- ) {
          tmp = zero;
          for ( int k=U->row[i]+1; k < U->row[i+1]; k++) {
            j = U->col[k];
            tmp += U->val[k]*x->val[j];  
          }
          tmp = (rhs->val[i] - tmp)/U->val[U->row[i]];
          step += pow((x->val[i] - tmp), 2);
          x->val[i] = tmp;
        }
      }
      *iter = *iter + 1;
      printf("%% iteration = %d step = %e\n", *iter, step);
    }
  
  }
  else {
    info = -1;
    printf("U matrix storage %d and fill mode %d must be CSRU (%d) and lower (%d) for a backward solve.\n",
      U->storage_type, U->fill_mode, Magma_CSRU, MagmaUpper );
    
  }
  
  return info;
}

extern "C" 
int
data_forward_solve_permute( data_d_matrix* L, data_d_matrix* x, data_d_matrix* rhs,
  const dataType tol, int *iter ) 
{
  int info = 0;
  
  if ( L->storage_type == Magma_CSRL 
    && L->fill_mode == MagmaLower 
    && L->diagorder_type != Magma_NODIAG ) { 
  
    int j = 0;
    *iter = 0;
    dataType step = 1.e8;
    dataType tmp = 0.0;
    const dataType zero = dataType(0.0);
    
    int* c;
    LACE_CALLOC(c, L->num_rows);
    #pragma omp parallel
    {
      #pragma omp for nowait
      for ( int i=0; i < L->num_rows; i++ ) {
        c[i] = i; 
      }
    }
    std::srand( unsigned ( std::time(0) ) );
    int i = 0;
    
    while (step > tol) {
      
      step = zero;
      #pragma omp parallel
      {
        #pragma omp for private(j, tmp) reduction(+:step) nowait
        for ( int ii=0; ii < L->num_rows; ii++ ) {
          i = c[ii];
          tmp = zero;
          for ( int k=L->row[i]; k < L->row[i+1]-1; k++) {
            j = L->col[k];
            tmp += L->val[k]*x->val[j];  
          }
          tmp = (rhs->val[i] - tmp)/L->val[L->row[i+1] - 1];
          step += pow((x->val[i] - tmp), 2);
          x->val[i] = tmp;
        }
      }
      *iter = *iter + 1;
      printf("%% iteration = %d step = %e\n", *iter, step);
      
      std::random_shuffle(c,c+L->num_rows);
      
    }
  
    free( c );
    
  }
  else {
    info = -1;
    printf("L matrix storage %d and fill mode %d must be CSRL (%d) and lower (%d) for a forward solve.\n",
      L->storage_type, L->fill_mode, Magma_CSRL, MagmaLower );
    
  }
  
  return info;
}

extern "C" 
int
data_backward_solve_permute( data_d_matrix* U, data_d_matrix* x, data_d_matrix* rhs,
  const dataType tol, int *iter ) 
{
  int info = 0;
  
  if ( U->storage_type == Magma_CSRU 
    && U->fill_mode == MagmaUpper 
    && U->diagorder_type != Magma_NODIAG ) { 
  
    int j = 0;
    *iter = 0;
    dataType step = 1.e8;
    dataType tmp = 0.0;
    const dataType zero = dataType(0.0);
    
    int* c;
    LACE_CALLOC(c, U->num_rows);
    #pragma omp parallel
    {
      #pragma omp for nowait
      for ( int i=U->num_rows; i > 0; i-- ) {
        c[i] = i; 
      }
    }
    std::srand( unsigned ( std::time(0) ) );
    int i = 0;
    
    while (step > tol) {
        
      step = zero;
      #pragma omp parallel
      {
        #pragma omp for private(j, tmp) reduction(+:step) nowait
        for ( int ii=U->num_rows-1; ii>=0; ii-- ) {
          i = c[ii];
          tmp = zero;
          for ( int k=U->row[i]+1; k < U->row[i+1]; k++) {
            j = U->col[k];
            tmp += U->val[k]*x->val[j];  
          }
          tmp = (rhs->val[i] - tmp)/U->val[U->row[i]];
          step += pow((x->val[i] - tmp), 2);
          x->val[i] = tmp;
        }
      }
      *iter = *iter + 1;
      printf("%% iteration = %d step = %e\n", *iter, step);
      
      std::random_shuffle(c,c+U->num_rows);
      
    }
  
  }
  else {
    info = -1;
    printf("U matrix storage %d and fill mode %d must be CSRU (%d) and lower (%d) for a backward solve.\n",
      U->storage_type, U->fill_mode, Magma_CSRU, MagmaUpper );
    
  }
  
  return info;
}

// MKL/LAPACK like interface
extern "C" 
int
data_parcsrtrsv( const data_uplo_t uplo, const data_storage_t storage, 
  const data_diagorder_t diag, 
  const int num_rows, const dataType *Aval, const int *row, const int *col, 
  const dataType *rhsval, dataType *yval,
  const dataType tol, int *iter  ) 
{
  int info = 0;
  
  if ( storage == Magma_CSRL 
    && uplo == MagmaLower 
    && diag != Magma_NODIAG ) { 
  
    int j = 0;
    *iter = 0;
    dataType step = 1.e8;
    dataType tmp = 0.0;
    const dataType zero = dataType(0.0);
    
    while (step > tol) {
        
      step = zero;
      #pragma omp parallel
      {
        #pragma omp for private(j, tmp) reduction(+:step) nowait
        for ( int i=0; i < num_rows; i++ ) {
          tmp = zero;
          for ( int k=row[i]; k < row[i+1]-1; k++) {
            j = col[k];
            tmp += Aval[k]*yval[j];  
          }
          tmp = (rhsval[i] - tmp)/Aval[row[i+1] - 1];
          step += pow((yval[i] - tmp), 2);
          yval[i] = tmp;
        }
      }
      *iter = *iter + 1;
      printf("%% iteration = %d step = %e\n", *iter, step);
    }
  
  }
  else if ( storage == Magma_CSRL 
    || uplo == MagmaLower ) {
    info = -1;
    printf("L matrix storage %d and fill mode %d must be CSRL (%d) and lower (%d) for a forward solve.\n",
      storage, uplo, Magma_CSRL, MagmaLower );  
  }
  
  else if ( storage == Magma_CSRU 
    && uplo == MagmaUpper 
    && diag != Magma_NODIAG ) { 
  
    int j = 0;
    *iter = 0;
    
    dataType step = 1.e8;
    dataType tmp = 0.0;
    const dataType zero = dataType(0.0);
    
    while (step > tol) {
        
      step = zero;
      #pragma omp parallel
      {
        #pragma omp for private(j, tmp) reduction(+:step) nowait
        for ( int i=num_rows-1; i>=0; i-- ) {
          tmp = zero;
          for ( int k=row[i]+1; k < row[i+1]; k++) {
            j = col[k];
            tmp += Aval[k]*yval[j];  
          }
          tmp = (rhsval[i] - tmp)/Aval[row[i]];
          step += pow((yval[i] - tmp), 2);
          yval[i] = tmp;
        }
      }
      *iter = *iter + 1;
      printf("%% iteration = %d step = %e\n", *iter, step);
    }
  
  }
  else if ( storage == Magma_CSRU 
    || uplo == MagmaUpper ) {
    info = -1;
    printf("U matrix storage %d and fill mode %d must be CSRU (%d) and lower (%d) for a backward solve.\n",
      storage, uplo, Magma_CSRU, MagmaUpper );  
  }
  
  
  return info;
}


extern "C" 
int
//void cblas_dtrsv (const CBLAS_LAYOUT Layout, const CBLAS_UPLO uplo, 
//  const CBLAS_TRANSPOSE trans, const CBLAS_DIAG diag, const MKL_INT n, 
//  const double *a, const MKL_INT lda, double *x, const MKL_INT incx);

data_partrsv( 
  const data_order_t major, 
  const data_uplo_t uplo, 
  const data_storage_t storage, 
  const data_diagorder_t diag, 
  const int num_rows, 
  const dataType *Aval, 
  const int lda, 
  const dataType *rhsval, 
  const int incr,
  dataType *yval,
  const int incx,
  const dataType tol, 
  int *iter ) 
{
  int info = 0;
  
  if ( major == MagmaRowMajor ) {
    if ( storage == Magma_DENSEL 
      && uplo == MagmaLower 
      && diag != Magma_NODIAG ) { 
    
      *iter = 0;
      dataType step = 1.e8;
      dataType tmp = 0.0;
      const dataType zero = dataType(0.0);
      
      while (step > tol) {
          
        step = zero;
        #pragma omp parallel
        {
          #pragma omp for private(tmp) reduction(+:step) nowait
          for ( int i=0; i < num_rows; i++ ) {
            tmp = zero;
            for ( int j=0; j <i; j++) {
              tmp += Aval[j+i*lda]*yval[j];  
            }
            tmp = (rhsval[i] - tmp)/Aval[i+i*lda];
            step += pow((yval[i] - tmp), 2);
            yval[i] = tmp;
          }
        }
        *iter = *iter + 1;
        printf("%% iteration = %d step = %e\n", *iter, step);
      }
    
    }
    else if ( storage == Magma_DENSEL 
      || uplo == MagmaLower ) {
      info = -1;
      printf("L matrix storage %d and fill mode %d must be DENSEL (%d) and lower (%d) for a forward solve.\n",
        storage, uplo, Magma_DENSEL, MagmaLower );  
    }
    
    else if ( storage == Magma_DENSEU 
      && uplo == MagmaUpper 
      && diag != Magma_NODIAG ) { 
    
      *iter = 0;
      
      dataType step = 1.e8;
      dataType tmp = 0.0;
      const dataType zero = dataType(0.0);
      
      while (step > tol) {
          
        step = zero;
        #pragma omp parallel
        {
          #pragma omp for private(tmp) reduction(+:step) nowait
          for ( int i=num_rows-1; i>=0; i-- ) {
            tmp = zero;
            for ( int j=i+1; j <num_rows; j++) {
              tmp += Aval[j+i*lda]*yval[j];  
            }
            tmp = (rhsval[i] - tmp)/Aval[i+i*lda];
            step += pow((yval[i] - tmp), 2);
            yval[i] = tmp;
          }
        }
        *iter = *iter + 1;
        printf("%% iteration = %d step = %e\n", *iter, step);
      }
    
    }
    else if ( storage == Magma_DENSEU 
      || uplo == MagmaUpper ) {
      info = -1;
      printf("U matrix storage %d and fill mode %d must be DENSEU (%d) and lower (%d) for a backward solve.\n",
        storage, uplo, Magma_DENSEU, MagmaUpper );  
    }
  
  }
  
  return info;
}
