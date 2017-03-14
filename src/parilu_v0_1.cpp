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
data_PariLU_v0_1( data_d_matrix* A, data_d_matrix* L, data_d_matrix* U ) 
{
  data_d_matrix Atmp = {Magma_CSRCOO};
  data_zmconvert(*A, &Atmp, Magma_CSR, Magma_CSRCOO);
  
  // Separate the lower and upper elements into L and U respectively.
  L->diagorder_type = Magma_UNITY;
  data_zmconvert(*A, L, Magma_CSR, Magma_CSRL);
  
  U->diagorder_type = Magma_VALUE;
  data_zmconvert(*A, U, Magma_CSR, Magma_CSCU);
  

  // ParLU element wise
  int i, j;
  int il, iu, jl, ju;
  
  int iter = 0;
  dataType tol = 1.0e-15;
  int num_threads = 0;
  
  dataType s = 0.0;
  dataType sp = 0.0;
  dataType tmp = 0.0;
  dataType step = 1.0;
  dataType Anorm = 0.0;
  dataType recipAnorm = 0.0;
  
  data_zfrobenius(*A, &Anorm);
  printf("%% Anorm = %e\n", Anorm);
  recipAnorm = 1.0/Anorm;
  
  std::vector<dataType> residuals;
  
  dataType wstart = omp_get_wtime();
  while ( step > tol ) {
    step = 0.0;
    
    sp = 0.0;
    
    #pragma omp parallel 
    {
      //num_threads = omp_get_num_threads();
      #pragma omp for private(i, j, il, iu, jl, ju, s, sp, tmp) reduction(+:step) nowait
      for (int k=0; k<A->nnz; k++ ) {
          i = Atmp.rowidx[k];
          j = A->col[k];
          s = A->val[k];
      
          il = L->row[i];
          iu = U->row[j];
          //printf("%d\n", k);
          while (il < L->row[i+1] && iu < U->row[j+1])
          {
              sp = 0.0;
              jl = L->col[il];
              ju = U->col[iu];
      
              // avoid branching
              sp = ( jl == ju ) ? L->val[il] * U->val[iu] : sp;
              s = ( jl == ju ) ? s-sp : s;
              il = ( jl <= ju ) ? il+1 : il;
              iu = ( jl >= ju ) ? iu+1 : iu;
          }
          // undo the last operation (it must be the last)
          s += sp;
          
          if ( i > j ) {     // modify l entry
              tmp = s / U->val[U->row[j+1]-1];
              step += pow( L->val[il-1] - tmp, 2 );
              L->val[il-1] = tmp; 
          }    
          else {            // modify u entry
              tmp = s;
              step += pow( U->val[iu-1] - tmp, 2 );
              U->val[iu-1] = tmp;
          }
      }
    }
    step *= recipAnorm;
    residuals.push_back( step );
  }
  iter = int(residuals.size());
  dataType wend = omp_get_wtime();
  dataType ompwtime = (dataType) (wend-wstart)/((dataType) iter);
  
  #pragma omp parallel 
  {
    num_threads = omp_get_num_threads();
  }
  
  for (int i=0; i<iter; i++) {
    printf("%% iteration = %d step = %e\n", i, residuals[i]); 
  }
  
  printf("%% PariLU v0.1 used %d OpenMP threads and required %d iterations, %f wall clock seconds, and an average of %f wall clock seconds per iteration as measured by omp_get_wtime()\n", 
    num_threads, iter, wend-wstart, ompwtime );
  printf("PariLUv0_1_OpenMP = %d \nPariLUv0_1_iter = %d \nPariLUv0_1_wall = %e \nPariLUv0_1_avgWall = %e \n", 
    num_threads, iter, wend-wstart, ompwtime );
  data_zmfree( &Atmp );
  
}
