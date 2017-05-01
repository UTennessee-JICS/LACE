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
data_forward_solve( data_d_matrix* L, data_d_matrix* x, data_d_matrix* rhs ) 
{
  int info = 0;
  
  if ( L->storage_type == Magma_CSRL 
    && L->fill_mode == MagmaLower 
    && L->diagorder_type != Magma_NODIAG ) { 
  
    int j = 0;
    int iter = 0;
    dataType tol = 1.e-15;
    dataType step = 1.e8;
    dataType tmp = 0.0;
    dataType tmp_step = 0.0;
    dataType diag_recip = 1.0;
    int diag_check = 0;
    
    while (step > tol) {
        
      step = 0;
      #pragma omp parallel
      {
        #pragma omp for private(j, diag_recip) reduction(+:step) nowait
        for ( int i=0; i < L->num_rows; i++ ) {
          tmp = dataType(0.0);
          for ( int k=L->row[i]; k < L->row[i+1]-1; k++) {
            j = L->col[k];
            tmp += L->val[k]*x->val[j];  
          }
          tmp = (rhs->val[i] - tmp)/L->val[L->row[i+1] - 1];
          step += pow((x->val[i] - tmp), 2);
          x->val[i] = tmp;
        }
      }
      iter = iter + 1;
      printf("%% iteration = %d step = %e\n", iter, step);
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
data_backward_solve( data_d_matrix* U, data_d_matrix* x, data_d_matrix* rhs ) 
{
  int info = 0;
  
  if ( U->storage_type == Magma_CSRU 
    && U->fill_mode == MagmaUpper 
    && U->diagorder_type != Magma_NODIAG ) { 
  
    int j = 0;
    int iter = 0;
    dataType tol = 1.e-15;
    dataType step = 1.e8;
    dataType tmp = 0.0;
    dataType tmp_step = 0.0;
    dataType diag_recip = 1.0;
    int diag_check = 0;
    
    while (step > tol) {
        
      step = 0;
      #pragma omp parallel
      {
        #pragma omp for private(j, diag_recip) reduction(+:step) nowait
        for ( int i=U->num_rows-1; i>=0; i-- ) {
          tmp = dataType(0.0);
          for ( int k=U->row[i]+1; k < U->row[i+1]; k++) {
            j = U->col[k];
            tmp += U->val[k]*x->val[j];  
          }
          tmp = (rhs->val[i] - tmp)/U->val[U->row[i]];
          step += pow((x->val[i] - tmp), 2);
          x->val[i] = tmp;
        }
      }
      iter = iter + 1;
      printf("%% iteration = %d step = %e\n", iter, step);
    }
  
  }
  else {
    info = -1;
    printf("U matrix storage %d and fill mode %d must be CSRL (%d) and lower (%d) for a forward solve.\n",
      U->storage_type, U->fill_mode, Magma_CSRU, MagmaUpper );
    
  }
  
  return info;
}

extern "C" 
int
data_forward_solve_permute( data_d_matrix* L, data_d_matrix* x, data_d_matrix* rhs ) 
{
  int info = 0;
  
  if ( L->storage_type == Magma_CSRL 
    && L->fill_mode == MagmaLower 
    && L->diagorder_type != Magma_NODIAG ) { 
  
    int j = 0;
    int iter = 0;
    dataType tol = 1.e-15;
    dataType step = 1.e8;
    dataType tmp = 0.0;
    dataType tmp_step = 0.0;
    dataType diag_recip = 1.0;
    int diag_check = 0;
    
    int* c;
    LACE_CALLOC(c, L->num_rows);
    #pragma omp parallel
    {
      #pragma omp for nowait
      for ( int i=0; i < L->num_rows; i++ ) {
        c[i] = i; 
      }
    }
    //std::srand(0);
    std::srand( unsigned ( std::time(0) ) );
    int i = 0;
    
    while (step > tol) {
      
      step = 0;
      #pragma omp parallel
      {
        #pragma omp for private(j, diag_recip) reduction(+:step) nowait
        for ( int ii=0; ii < L->num_rows; ii++ ) {
          i = c[ii];
          tmp = dataType(0.0);
          for ( int k=L->row[i]; k < L->row[i+1]-1; k++) {
            j = L->col[k];
            tmp += L->val[k]*x->val[j];  
          }
          tmp = (rhs->val[i] - tmp)/L->val[L->row[i+1] - 1];
          step += pow((x->val[i] - tmp), 2);
          x->val[i] = tmp;
        }
      }
      iter = iter + 1;
      printf("%% iteration = %d step = %e\n", iter, step);
      
      //for ( int i=0; i < L->num_rows; i++ ) {
      //  printf("c[%d]=%d\n", i, c[i]); 
      //}
      //std::next_permutation(c,c+L->num_rows);
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
data_backward_solve_permute( data_d_matrix* U, data_d_matrix* x, data_d_matrix* rhs ) 
{
  int info = 0;
  
  if ( U->storage_type == Magma_CSRU 
    && U->fill_mode == MagmaUpper 
    && U->diagorder_type != Magma_NODIAG ) { 
  
    int j = 0;
    int iter = 0;
    dataType tol = 1.e-15;
    dataType step = 1.e8;
    dataType tmp = 0.0;
    dataType tmp_step = 0.0;
    dataType diag_recip = 1.0;
    int diag_check = 0;
    
    int* c;
    LACE_CALLOC(c, U->num_rows);
    #pragma omp parallel
    {
      #pragma omp for nowait
      for ( int i=U->num_rows; i > 0; i-- ) {
        c[i] = i; 
      }
    }
    //std::srand(0);
    std::srand( unsigned ( std::time(0) ) );
    int i = 0;
    
    while (step > tol) {
        
      step = 0;
      #pragma omp parallel
      {
        #pragma omp for private(j, diag_recip) reduction(+:step) nowait
        for ( int i=U->num_rows-1; i>=0; i-- ) {
          tmp = dataType(0.0);
          for ( int k=U->row[i]+1; k < U->row[i+1]; k++) {
            j = U->col[k];
            tmp += U->val[k]*x->val[j];  
          }
          tmp = (rhs->val[i] - tmp)/U->val[U->row[i]];
          step += pow((x->val[i] - tmp), 2);
          x->val[i] = tmp;
        }
      }
      iter = iter + 1;
      printf("%% iteration = %d step = %e\n", iter, step);
      
      //for ( int i=0; i < U->num_rows; i++ ) {
      //  printf("c[%d]=%d\n", i, c[i]); 
      //}
      //std::prev_permutation(c,c+U->num_rows);
      std::random_shuffle(c,c+U->num_rows);
      
    }
  
  }
  else {
    info = -1;
    printf("U matrix storage %d and fill mode %d must be CSRL (%d) and lower (%d) for a forward solve.\n",
      U->storage_type, U->fill_mode, Magma_CSRU, MagmaUpper );
    
  }
  
  return info;
}