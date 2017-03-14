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
data_ParLU_v1_0( data_d_matrix* A, data_d_matrix* L, data_d_matrix* U ) 
{
  // Separate the strictly lower and upper elements into L, and U respectively.
  L->diagorder_type = Magma_NODIAG;
  data_zmconvert(*A, L, Magma_DENSE, Magma_DENSEL);
  
  U->diagorder_type = Magma_VALUE;
  data_zmconvert(*A, U, Magma_DENSE, Magma_DENSEU);
  
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
  
  data_zfrobenius(*A, &Anorm);
  printf("%% Anorm = %e\n", Anorm);
  
  dataType wstart = omp_get_wtime();
  while ( step > tol ) {
    step = 0.0;
    #pragma omp parallel private(sumL, sumU, tmp)
    {
      num_threads = omp_get_num_threads();
      #pragma omp for schedule(static,1) reduction(+:step) nowait
      for (int i=0; i<row_limit; i++) {
        for (int j=0; j<i; j++) { // L
          //sumL = 0.0;
          //for (int k=0; k<j; k++) {
          //  sumL += L->val[ i*A->ld + k ]*U->val[ k*A->ld + j ];
          //}
          //sumL = data_zdot( j, &L->val[ i*L->ld ], 1, &U->val[ j ], U->ld );
          sumL = data_zdot_mkl( j, &L->val[ i*L->ld ], 1, &U->val[ j ], U->ld );
          tmp = (A->val[ i*A->ld + j ] - sumL)/U->val[ j*A->ld + j ];
          step += pow( L->val[ i*A->ld + j ] - tmp, 2 );
          L->val[ i*A->ld + j ] = tmp;
        }
        for (int j=i; j<col_limit; j++) { // U
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
    step /= Anorm;
    iter++;
    printf("%% iteration = %d step = %e\n", iter, step);
  }
  dataType wend = omp_get_wtime();
  dataType ompwtime = (dataType) (wend-wstart)/((dataType) iter);
  printf("%% ParLU v1.0 used %d OpenMP threads and required %d iterations, %f wall clock seconds, and an average of %f wall clock seconds per iteration as measured by omp_get_wtime()\n", 
    num_threads, iter, wend-wstart, ompwtime );
}
