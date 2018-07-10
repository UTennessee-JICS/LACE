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
data_PariLU_v0_3_bcsr( data_d_matrix* A,
  data_d_matrix* L,
  data_d_matrix* U,
  dataType reduction,
  data_d_preconditioner_log* log )
{
  // Separate the lower and upper elements into L and U respectively.
  L->diagorder_type = Magma_UNITY;
  data_zmconvert(*A, L, Magma_BCSR, Magma_BCSRL);
  U->diagorder_type = Magma_VALUE;
  data_zmconvert(*A, U, Magma_BCSR, Magma_BCSCU); 

  dataType Ares = 0.0;
  dataType Anonlinres = 0.0;

  data_d_matrix LU = {Magma_BCSR};
  //data_zvinit_gen(&LU);//ceb initialize new matrix
  data_zmconvert(*A, &LU, Magma_BCSR, Magma_BCSR);
  data_zilures_bcsr(*A, *L, *U, &LU, &Ares, &Anonlinres);

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
  PARILUDBG("PariLU_v0_3_bcsr_tol = %e\n", tol);
  printf("PariLU_v0_3_bcsr_tol = %e\n", tol);
  int num_threads = 0;
  dataType wstart, wend;

  dataType* s;
  dataType* sp;
  dataType* tmp;
  //printf("A->ldblock=%d\n",A->ldblock);
  LACE_CALLOC(s, A->ldblock);
  LACE_CALLOC(sp, A->ldblock);
  LACE_CALLOC(tmp, A->ldblock);

  dataType step = FLT_MAX;
  dataType Anorm = 0.0;
  dataType recipAnorm = 0.0;

  data_zfrobenius(*A, &Anorm);
  PARILUDBG("PariLUv0_3_bcsr_Anorm = %e\n", Anorm);
  printf("PariLUv0_3_bcsr_Anorm = %e\n", Anorm);
  recipAnorm = 1.0/Anorm;

  wstart = omp_get_wtime();
  data_rowindex(A, &(A->rowidx) );

  while ( (step > tol) && (iter < itermax) ) {
    step = 0.0;

    for(int kk=0; kk< A->ldblock; ++kk){sp[kk]=0.0;} //sp = 0.0;  
    //#pragma omp parallel
    {
      //#pragma omp for private(i, j, il, iu, jl, ju, s, sp, tmp) reduction(+:step) nowait
      for (int k=0; k<A->nnz; k++ ) {
        i = A->rowidx[k];
        j = A->col[k];
        for(int kk=0; kk< A->ldblock; ++kk){s[kk] = A->val[k*A->ldblock+kk];} //s = A->val[k]
        il = L->row[i];
        iu = U->row[j];
        while (il < L->row[i+1] && iu < U->row[j+1])
        {
	    for(int kk=0; kk< A->ldblock; ++kk){sp[kk]=0.0;} //sp = 0
            
            jl = L->col[il];
            ju = U->col[iu];
#if 0
            //if(A->ldblock==1){
               // avoid branching
               //sp[0] = ( jl == ju ) ? L->val[il] * U->val[iu] : sp[0];
               //s[0] = ( jl == ju ) ? s[0]-sp[0] : s[0];
	       //}
	       //else{
#else
            if(jl == ju) //diagonal blocks
	    { 
              //sp=L[il]*U[iu] (small matrix multiply)
               for(int ii=0; ii< A->blocksize; ++ii){
                  for(int jj=0; jj< A->blocksize; ++jj){
                     for(int kk=0; kk< A->blocksize; ++kk)
		       {sp[ii*A->blocksize+jj] += 
	                  L->val[il*A->ldblock+ii*A->blocksize+kk] *
                          U->val[iu*A->ldblock+kk*A->blocksize+jj];
		       }
                  }
               }
 //data_dgemm_mkl(layout,transA,transB,A->blocksize,A->blocksize,A->blocksize,alpha,A,lda,B,ldb,beta,C,ldc);
	       //DEV_CHECKPT
               for(int kk=0; kk< A->ldblock; ++kk){s[kk] -= sp[kk];} //s=s-sp
            }
#endif
	    //}

            il = ( jl <= ju ) ? il+1 : il;
            iu = ( jl >= ju ) ? iu+1 : iu;
        }
        // undo the last operation (it must be the last)
        for(int kk=0; kk< A->ldblock; ++kk){s[kk]+=sp[kk];} //s += sp;

        if ( i > j ) {     // modify l entry
            //for(int k=0; k<A->ldblock; ++k) tmp[] = s[] / U->val[U->row[j+1]-1];
            //copy tmp = s
            for(int kk=0; kk< A->ldblock; ++kk){tmp[kk]=s[kk];} 
            //compute inverse of  U->val[(U->row[j+1]-1)]
            if(A->ldblock==1){for(int kk=0; kk< A->ldblock; ++kk){tmp[kk]/= U->val[(U->row[j+1]-1)*U->ldblock+kk];} 
	    }
	    else{
	      //compute matrix multiply s*Uinv;
//ceb fix this here!!
	    }

            //step += pow( L->val[il-1] - tmp, 2 );
            //compute        L->val[il-1] - tmp
            for(int kk=0; kk< A->ldblock; ++kk){L->val[(il-1)*A->ldblock+kk] -= tmp[kk];} 

            //compute l2 norm of result
	    //double result = l2norm(L->val[(il-1)*A.ldblock]);
            //add scalar result to step

            //same as dot product of vector
            //step += l2norm(L->val[(il-1)*A.ldblock], A.ldblock);
            step += data_zdot_mkl(A->ldblock, &(L->val[(il-1)*A->ldblock]), 1, &(L->val[(il-1)*A->ldblock]), 1);

            //L->val[il-1] = tmp;
            for(int kk=0; kk< A->ldblock; ++kk){L->val[(il-1)*A->ldblock+kk]=tmp[kk];} 

        }
        else {            // modify u entry
            for(int kk=0; kk< A->ldblock; ++kk){tmp[kk]=s[kk];} //tmp = s;
            //for(int kk=0; kk< A->ldblock; ++kk){U->val[(iu-1)*A->ldblock+kk] -= tmp[kk];} 
            for(int kk=0; kk<U->ldblock;++kk){step += pow( U->val[(iu-1)*U->ldblock+kk] - tmp[kk], 2 );}
            //step = data_zdot_mkl(A->ldblock, &(U->val[(iu-1)*A->ldblock]), 1, &(U->val[(iu-1)*A->ldblock]), 1);
            for(int kk=0; kk< A->ldblock; ++kk){U->val[(iu-1)*A->ldblock+kk]=tmp[kk];}//U->val[iu-1] = tmp;
        }

      }
    }
    step *= recipAnorm;
    iter++;
    //PARILUDBG("%% PariLUv0_3_iteration = %d step = %e\n", iter, step);
    printf("%% PariLUv0_3_bcsr_iteration = %d step = %e\n", iter, step);
  }

  wend = omp_get_wtime();
  dataType ompwtime = (dataType) (wend-wstart)/((dataType) iter);

  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }

  PARILUDBG("%% PariLU v0.3 bcsr used %d OpenMP threads and required %d iterations, %f wall clock seconds, and an average of %f wall clock seconds per iteration as measured by omp_get_wtime()\n",
    num_threads, iter, wend-wstart, ompwtime );
  PARILUDBG("PariLUv0_3_bcsr_OpenMP = %d \nPariLUv0_3_iter = %d \nPariLUv0_3_wall = %e \nPariLUv0_3_avgWall = %e \n",
    num_threads, iter, wend-wstart, ompwtime );

  //data_zmfree( &Atmp );
  data_zmfree( &LU );

  free(s);
  free(sp);
  free(tmp);

  log->sweeps = iter;
  log->tol = tol;
  log->A_Frobenius = Anorm;
  log->precond_generation_time = wend-wstart;
  log->initial_residual = Ares;
  log->initial_nonlinear_residual = Anonlinres;
  log->omp_num_threads = num_threads;

}
