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
void
data_ParLU_v1_2c( data_d_matrix* A, data_d_matrix* L, data_d_matrix* U, int tile ) 
{
  
  data_z_pad_dense(A, tile);
  
  // Separate the strictly lower, strictly upper, and diagonal elements 
  // into L, U, and D respectively.
  L->diagorder_type = Magma_NODIAG;
  data_zmconvert(*A, L, Magma_DENSE, Magma_DENSEL);
  
  U->diagorder_type = Magma_NODIAG;
  data_zmconvert(*A, U, Magma_DENSE, Magma_DENSEU);
  
  data_d_matrix D = {Magma_DENSED};
  data_zmconvert(*A, &D, Magma_DENSE, Magma_DENSED);
  
  // Set diagonal elements to the recipricol
  #pragma omp parallel  
  #pragma omp for nowait
  for (int i=0; i<D.nnz; i++) {
    D.val[ i ] = 1.0/D.val[ i ];
  }
  
  //printf("\nA\n");
  //data_zdisplay_dense( A );
  //printf("\nL\n");
  //data_zdisplay_dense( L );
  //printf("\nU\n");
  //data_zdisplay_dense( U );
  //printf("\nD\n");
  //data_zprint_dense( D );
  
  
  //data_z_pad_dense(A, tile);
  //data_z_pad_dense(L, tile);
  //data_z_pad_dense(U, tile);
  //data_z_pad_dense(&D, tile); 
  
  int row_limit = A->num_rows;
  int col_limit = A->num_cols;
  if (A->pad_rows > 0 && A->pad_cols > 0) {
    row_limit = A->pad_rows;
    col_limit = A->pad_cols;
  }
  
  // ParLU element wise
  dataType sumL = 0.0;
  dataType sumU = 0.0;
  int iter = 0;
  dataType tmp = 0.0;
  dataType step = 1.0;
  dataType tol = 1.0e-15;
  dataType Anorm = 0.0;
 
  int num_threads = 0;
  //int ti = 0;
  //int tj = 0;
  //int i = 0;
  //int j = 0;
  
  data_zfrobenius(*A, &Anorm);
  printf("%% Anorm = %e\n", Anorm);
  
  dataType wstart = omp_get_wtime();
  while ( step > tol ) {
  //while ( iter < 10 ) {  
    step = 0.0;
    //#pragma omp parallel private(sumL, sumU, tmp, ti, tj, i, j)
    #pragma omp parallel private(sumL, sumU, tmp)
    {
      num_threads = omp_get_num_threads();
      #pragma omp for reduction(+:step) collapse(2) nowait
      //#pragma omp for private(ti, tj, i, j) reduction(+:step) collapse(2) nowait
      //#pragma omp for schedule(static,1) reduction(+:step) nowait
      for (int ti=0; ti<row_limit; ti += tile) {
         for (int tj=0; tj<col_limit; tj += tile) {
           for (int i=ti; i<ti+tile; i++) {
             for (int j=tj; j<tj+tile; j++) {
               if (i>j) { // L
                 //sumL = 0.0;
                 //for (int k=0; k<j; k++) {
                 //  sumL += L->val[ i*A->ld + k ]*U->val[ k*A->ld + j ];
                 //}
                 //sumL = data_zdot( j, &L->val[ i*L->ld ], 1, &U->val[ j ], U->ld );
                 sumL = data_zdot_mkl( j, &L->val[ i*L->ld ], 1, &U->val[ j ], U->ld );
                 tmp = (A->val[ i*A->ld + j ] - sumL)*D.val[ j ];
                 step += pow( L->val[ i*A->ld + j ] - tmp, 2 );
                 L->val[ i*A->ld + j ] = tmp;
               }
               else if (i==j) {
                 //sumU = 0.0;
                 //for (int k=0; k<i; k++) {
                 //  sumU += L->val[ i*A->ld + k ]*U->val[ k*A->ld + i ];
                 //}
                 //sumU = data_zdot( i, &L->val[ i*L->ld ], 1, &U->val[ i ], U->ld );
                 sumU = data_zdot_mkl( i, &L->val[ i*L->ld ], 1, &U->val[ i ], U->ld );
                 tmp = 1.0/(A->val[ i*A->ld + i ] - sumU);
                 step += pow(D.val[ i ] - tmp, 2);
                 D.val[ i ] = tmp;
               }
               else { // U
                 //sumU = 0.0;
                 //for (int k=0; k<i; k++) {
                 //  sumU += L->val[ i*A->ld + k ]*U->val[ k*A->ld + j ];
                 //}
                 //sumU = data_zdot( i, &L->val[ i*L->ld ], 1, &U->val[ j ], U->ld );
                 sumU = data_zdot_mkl( i, &L->val[ i*L->ld ], 1, &U->val[ j ], U->ld );
                 tmp = (A->val[ i*A->ld + j ] - sumU);
                 step += pow(U->val[ i*A->ld + j ] - tmp, 2);
                 U->val[ i*A->ld + j ] = tmp;
               }
             }
           }
         }
      }
    }
    step /= Anorm;
    iter++;
    printf("%% iteration = %d step = %e\n", iter, step);
  }
  dataType wend = omp_get_wtime();
  dataType ompwtime = (dataType) (wend-wstart)/((dataType) iter);
  
  // Fill diagonal elements
  #pragma omp parallel  
  {
    #pragma omp for nowait
    for (int i=0; i<row_limit; i++) {
      L->val[ i*L->ld + i ] = 1.0;
      U->val[ i*U->ld + i ] = 1.0/D.val[ i ];
    }
  }
  L->diagorder_type = Magma_UNITY;
  U->diagorder_type = Magma_VALUE;
  
  printf("%% ParLU v1.2c used %d OpenMP threads and required %d iterations, %f wall clock seconds, and an average of %f wall clock seconds per iteration as measured by omp_get_wtime()\n", 
    num_threads, iter, wend-wstart, ompwtime );
  
  data_zmfree( &D );
  
}
