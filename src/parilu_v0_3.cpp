/*
    -- LACE (version 0.0) --
       Univ. of Tennessee, Knoxville

       @author Stephen Wood

*/
#include "../include/sparse.h"
#include <mkl.h>
#include <float.h>
#include <stdlib.h>
#include <stdio.h>

extern "C"
void
data_PariLU_v0_3( data_d_matrix* A,
  data_d_matrix* L,
  data_d_matrix* U,
  dataType reduction,
  data_d_preconditioner_log* log )
{
  // Separate the lower and upper elements into L and U respectively.
  L->diagorder_type = Magma_UNITY;
  data_zmconvert(*A, L, Magma_CSR, Magma_CSRL);

  U->diagorder_type = Magma_VALUE;
  data_zmconvert(*A, U, Magma_CSR, Magma_CSCU);

  dataType Ares = 0.0;
  dataType Anonlinres = 0.0;
  data_d_matrix LU = {Magma_CSR};
  data_zmconvert(*A, &LU, Magma_CSR, Magma_CSR);
  data_zilures(*A, *L, *U, &LU, &Ares, &Anonlinres);
  PARILUDBG("PariLUv0_3_csrilu0_init_res = %e\n", Ares);
  PARILUDBG("PariLUv0_3_csrilu0_init_nonlinres = %e\n", Anonlinres);
  printf("PariLUv0_3_csrilu0_init_res = %e\n", Ares);
  printf("PariLUv0_3_csrilu0_init_nonlinres = %e\n", Anonlinres);

  // ParLU element wise
  int i, j;
  int il, iu, jl, ju;

  int iter = 0;
  dataType tol = 1.0e-40;
  tol = reduction*Ares;
  int itermax=20;
  PARILUDBG("PariLU_v0_3_tol = %e\n", tol);
  printf("PariLU_v0_3_tol = %e\n", tol);
  int num_threads = 0;

  dataType s = 0.0;
  dataType sp = 0.0;
  dataType tmp = 0.0;
  dataType step = FLT_MAX;
  dataType Anorm = 0.0;
  dataType recipAnorm = 0.0;

  data_zfrobenius(*A, &Anorm);
  PARILUDBG("PariLUv0_3_Anorm = %e\n", Anorm);
  printf("PariLUv0_3_Anorm = %e\n", Anorm);
  recipAnorm = 1.0/Anorm;

  dataType wstart = omp_get_wtime();
  data_rowindex(A, &(A->rowidx) );
  while ( (step > tol) && (iter < itermax) ) {
    step = 0.0;
    sp = 0.0;

    #pragma omp parallel
    {
    #pragma omp for private(i, j, il, iu, jl, ju, s, sp, tmp) reduction(+:step) nowait
      for (int k=0; k<A->nnz; k++ ) {
        //get row index i and column index j for element in A
        i = A->rowidx[k];
        j = A->col[k];
        //store element in s
        s = A->val[k];

        il = L->row[i];//get number of corresponding row in L
        iu = U->row[j];//get number of corresponding column in U (CSC form)
	
        //while still traversing current row i and column j
        while (il < L->row[i+1] && iu < U->row[j+1])
        {
	  sp = 0.0;
	  
	  jl = L->col[il];//get number of corresponding column in L
	  ju = U->col[iu];//get number of corresponding row in U (CSC form)
	  
	  // avoid branching
	  //if on diagonal block 
	  sp = ( jl == ju ) ? L->val[il] * U->val[iu] : sp;
	  
	  s = ( jl == ju ) ? s-sp : s;
	  //if not on diagonal, increment row index in L
	  il = ( jl <= ju ) ? il+1 : il;
	  //if not on diagonal, increment col index in U
	  iu = ( jl >= ju ) ? iu+1 : iu;
        }

        // undo the last operation (it must be the last)
        s += sp;

        //row greater than column
        if ( i > j ) {     // modify l entry
	  tmp = s / U->val[U->row[j+1]-1];
	  step += pow( L->val[il-1] - tmp, 2 );
	  L->val[il-1] = tmp;
        }
        //column greater than or equal to row
        else {            // modify u entry
	  tmp = s;
	  step += pow( U->val[iu-1] - tmp, 2 );
	  U->val[iu-1] = tmp;
        }
      }
    }
    step *= recipAnorm;
    iter++;
    //PARILUDBG("%% PariLUv0_3_iteration = %d step = %e\n", iter, step);
    printf("%% PariLUv0_3_iteration = %d step = %e\n", iter, step);
  }
  dataType wend = omp_get_wtime();
  dataType ompwtime = (dataType) (wend-wstart)/((dataType) iter);

  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }

  PARILUDBG("%% PariLU v0.3 used %d OpenMP threads and required %d iterations, %f wall clock seconds, and an average of %f wall clock seconds per iteration as measured by omp_get_wtime()\n",num_threads, iter, wend-wstart, ompwtime );
  printf("%% PariLU v0.3 used %d OpenMP threads and required %d iterations, %f wall clock seconds, and an average of %f wall clock seconds per iteration as measured by omp_get_wtime()\n",num_threads, iter, wend-wstart, ompwtime );

  PARILUDBG("PariLUv0_3_OpenMP = %d \nPariLUv0_3_iter = %d \nPariLUv0_3_wall = %e \nPariLUv0_3_avgWall = %e \n", num_threads, iter, wend-wstart, ompwtime );
  printf("PariLUv0_3_OpenMP = %d \nPariLUv0_3_iter = %d \nPariLUv0_3_wall = %e \nPariLUv0_3_avgWall = %e \n", num_threads, iter, wend-wstart, ompwtime );

  data_zmfree( &LU );

  log->sweeps = iter;
  log->tol = tol;
  log->A_Frobenius = Anorm;
  log->precond_generation_time = wend-wstart;
  log->initial_residual = Ares;
  log->initial_nonlinear_residual = Anonlinres;
  log->omp_num_threads = num_threads;

}
