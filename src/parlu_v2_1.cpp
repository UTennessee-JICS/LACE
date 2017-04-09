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
data_ParLU_v2_1( data_d_matrix* A, data_d_matrix* L, data_d_matrix* U, int tile ) 
{
  
  data_z_pad_dense(A, tile);
  
  // Separate the strictly lower, strictly upper, and diagonal elements 
  // into L, U, and D respectively.
  L->diagorder_type = Magma_NODIAG;
  data_zmconvert(*A, L, Magma_DENSE, Magma_DENSEL);
  
  U->diagorder_type = Magma_NODIAG;
  // store U in column major
  //U->major = MagmaColMajor;
  U->major = MagmaRowMajor;
  data_zmconvert(*A, U, Magma_DENSE, Magma_DENSEU);
  
  data_d_matrix D = {Magma_DENSED};
  data_zmconvert(*A, &D, Magma_DENSE, Magma_DENSED);
  
  // Set diagonal elements to the recipricol
  #pragma omp parallel  
  #pragma omp for nowait
  for (int i=0; i<D.nnz; i++) {
    D.val[ i ] = 1.0/D.val[ i ];
  }
  
  int row_limit = A->num_rows;
  int col_limit = A->num_cols;
  if (A->pad_rows > 0 && A->pad_cols > 0) {
    row_limit = A->pad_rows;
    col_limit = A->pad_cols;
  }
  
  // ParLU element wise
  int iter = 0;
  dataType tmp = 0.0;
  dataType step = 1.0;
  dataType tol = 1.0e-15;
  dataType Anorm = 0.0;
 
  int num_threads = 0;
  
  dataType alpha = 1.0;
  dataType beta = 0.0;
  
  // setup a vector workspace for all threads
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  dataType *vworkspace;
  vworkspace = (dataType*) calloc( (tile*num_threads), sizeof(dataType) );
  
  data_zfrobenius(*A, &Anorm);
  printf("%% Anorm = %e\n", Anorm);
  
  dataType wstart = omp_get_wtime();
  while ( step > tol && iter < 100 ) {
  //while ( iter < 10 ) {  
    step = 0.0;
    #pragma omp parallel private(tmp)
    {
      //#pragma omp for schedule(static,1) collapse(2) reduction(+:step) nowait
      #pragma omp for schedule(static,1) reduction(+:step) nowait
      for (int ti=0; ti<row_limit; ti += tile) {
         for (int tj=0; tj<col_limit; tj += tile) {
           
           int thread_num = omp_get_thread_num(); 
           //printf("ti=%d tj=%d thread_num=%d\n", ti, tj, thread_num);
           dataType *vtmp = &(vworkspace[tile*thread_num]);
           
           if (ti>tj) { // strictly L tile
             //dataType vtmp[tile];
             for (int j=tj; j<tj+tile; j++) {
               data_dgemv_mkl( L->major, MagmaNoTrans, tile, tj+tile, 
                 alpha, &L->val[ti*L->ld], L->ld, 
                 &U->val[j], U->ld, beta, vtmp, 1 );
               for (int i=ti; i<ti+tile; i++) {
                 tmp = (A->val[ i*A->ld + j ] - vtmp[i-ti])*D.val[ j ];
                 step += pow( L->val[ i*A->ld + j ] - tmp, 2 );
                 L->val[ i*A->ld + j ] = tmp;
               }
             }
           }
           else if (ti==tj) { // diagonal tile with L and U elements
             //dataType vtmp[tile];
             //for (int i=ti; i<ti+tile; i++) {
             for (int j=tj; j<tj+tile; j++) {
                 //for (int j=tj; j<tj+tile; j++) {
               data_dgemv_mkl( L->major, MagmaNoTrans, tile, tj+tile, 
                 alpha, &L->val[ti*L->ld], L->ld, 
                 &U->val[j], U->ld, beta, vtmp, 1 );
               for (int i=ti; i<ti+tile; i++) {
                 if (i>j) {
                   tmp = (A->val[ i*A->ld + j ] - vtmp[i-ti])*D.val[ j ];
                   step += pow( L->val[ i*A->ld + j ] - tmp, 2 );
                   L->val[ i*A->ld + j ] = tmp;
                 }
                 else if (i==j) {
                   tmp = 1.0/(A->val[ i*A->ld + i ] - vtmp[i-ti]);
                   step += pow(D.val[ i ] - tmp, 2);
                   D.val[ i ] = tmp;
                 }
                 else {
                   tmp = (A->val[ i*A->ld + j ] - vtmp[i-ti]);
                   step += pow(U->val[ i*A->ld + j ] - tmp, 2);
                   U->val[ i*A->ld + j ] = tmp;
                 }
               }
             }
             
           }
           else { // strictly U tile
             //dataType vtmp[tile];
             for (int i=ti; i<ti+tile; i++) {
               data_dgemv_mkl( U->major, MagmaTrans, ti+tile, tile,  
                 alpha, &U->val[tj], U->ld, 
                 &L->val[i*L->ld], 1, beta, vtmp, 1 );
               for (int j=tj; j<tj+tile; j++) {
                 tmp = (A->val[ i*A->ld + j ] - vtmp[j-tj]);
                 step += pow( U->val[ i*A->ld + j ] - tmp, 2 );
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
  #pragma omp for nowait
  for (int i=0; i<row_limit; i++) {
    L->val[ i*L->ld + i ] = 1.0;
    U->val[ i*U->ld + i ] = 1.0/D.val[ i ];
  }
  
  printf("%% ParLU v2.1 used %d OpenMP threads and required %d iterations, %f wall clock seconds, and an average of %f wall clock seconds per iteration as measured by omp_get_wtime()\n", 
    num_threads, iter, wend-wstart, ompwtime );
  fflush(stdout); 
  
  data_zmfree( &D );
  free( vworkspace );
  
}
