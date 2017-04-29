/*
    -- LACE (version 0.0) --
       Univ. of Tennessee, Knoxville
       
       @author Stephen Wood

*/
#include "../include/sparse.h"
#include <mkl.h>
#include <stdlib.h>
#include <stdio.h>

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
    
    dataType wstart = omp_get_wtime();
    x->val[0] = rhs->val[0]/L->val[0];
    while (step > tol) {
    //while (iter < 10) {
        
      step = 0;
      //#pragma omp parallel
      {
        //#pragma omp for private(j, diag_recip) reduction(+:step) nowait
        for ( int i=1; i < L->num_rows; i++ ) {
        //for ( int i=1; i < 10; i++ ) {
          tmp = 0.0;
          
          //printf("all of row %d : ", i);
          //for ( int k=L->row[i]; k < L->row[i+1]; k++) {
          //  j = L->col[k];
          //  printf("%d %e ", j, L->val[k]);
          //}
          //printf("\n");
          
          //printf("off diagonal row %d : ", i);
          for ( int k=L->row[i]; k < L->row[i+1]-1; k++) {
            j = L->col[k];
            //printf("%d %e ", j, L->val[k]);
            tmp += L->val[k]*x->val[j];  
          }
          tmp = (rhs->val[i] - tmp)/L->val[L->row[i+1] - 1];
          step += pow((x->val[i] - tmp), 2);
          x->val[i] = tmp;
          //printf(" => x = %e\n", x->val[i]);
        }
      }
      iter = iter + 1;
      printf("%% iteration = %d step = %e\n", iter, step);
    }
    dataType wend = omp_get_wtime();
  
  }
  else {
    info = -1;
    printf("L matrix storage %d and fill mode %d must be CSRL (%d) and lower (%d) for a forward solve.\n",
      L->storage_type, L->fill_mode, Magma_CSRL, MagmaLower );
    
  }
  
  return info;
}