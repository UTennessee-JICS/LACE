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
data_PariLU_v0_3_bcsr(data_d_matrix* A,
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
  int itermax=100;
  PARILUDBG("PariLU_v0_3_bcsr_tol = %e\n", tol);
  printf("PariLU_v0_3_bcsr_tol = %e\n", tol);
  int num_threads = 0;
  dataType wstart, wend;

  //These arrays need to be static so that they can be private vars in omp parallel region
  dataType s[A->ldblock];
  dataType sp[A->ldblock];
  dataType tmp[A->ldblock];
  dataType Uinv[A->ldblock];

  dataType step = FLT_MAX;
  dataType Anorm = 0.0;
  dataType recipAnorm = 0.0;

  dataType one = 1.0;
  dataType zero = 0.0;
  int layout = LAPACK_ROW_MAJOR;

  data_zfrobenius(*A, &Anorm);
  PARILUDBG("PariLUv0_3_bcsr_Anorm = %e\n", Anorm);
  printf("PariLUv0_3_bcsr_Anorm = %e\n", Anorm);
  recipAnorm = 1.0/Anorm;

  wstart = omp_get_wtime();
  data_rowindex(A, &(A->rowidx) );
  while ( (step > tol) && (iter < itermax) ) {
    step = 0.0;
    for(int kk=0; kk< A->ldblock; ++kk){sp[kk]=0.0;} //sp = 0.0; 
    
    #pragma omp parallel
    {
    #pragma omp for private(i, j, il, iu, jl, ju, s, sp, tmp, Uinv) reduction(+:step) //nowait
      for (int k=0; k<A->nnz; k++ ) {	
	//get row index i and column index j for block element in A
	i = A->rowidx[k];
	j = A->col[k];
	//store block element in s
	for(int kk=0; kk< A->ldblock; ++kk){s[kk] = A->val[k*A->ldblock+kk];} //s = A->val[k]
	
	il = L->row[i];//get number of corresponding row in L
	iu = U->row[j];//get number of corresponding column in U (CSC form)
	
	//while still traversing current row i in L and column j in U
	while (il < L->row[i+1] && iu < U->row[j+1]){
	  
	  for(int kk=0; kk< A->ldblock; ++kk){sp[kk]=0.0;} //sp = 0
	  
	  jl = L->col[il];//get number of corresponding column in L
	  ju = U->col[iu];//get number of corresponding row in U (CSC form)
	  
	  if(jl == ju){ //if on diagonal block 
	    //sp=L[il]*U[iu] (small matrix multiply)
	    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
			A->blocksize, A->blocksize, A->blocksize, one,  
			&(L->val[il*A->ldblock]), 
			A->blocksize, &(U->val[iu*A->ldblock]), A->blocksize, 
			zero, sp, A->blocksize);
	    
	    for(int kk=0; kk< A->ldblock; ++kk){s[kk] -= sp[kk];}
	  }
	  
	  //if not on diagonal, increment row index in L
	  il = ( jl <= ju ) ? il+1 : il;
	  //if not on diagonal, increment col index in U
	  iu = ( jl >= ju ) ? iu+1 : iu;
	}
	
	// undo the last operation (it must be the last)
	for(int kk=0; kk< A->ldblock; ++kk){s[kk]+=sp[kk];}
	
	//row greater than column
	if ( i > j ) {     // modify l entry
	  //#pragma omp critical
	  {
	    //compute inverse of U->val[U->row[j+1]-1], 
	    int ipiv[A->blocksize];
	    //copy U 
	    for(int kk=0; kk< A->ldblock; ++kk){
	      Uinv[kk]=U->val[(U->row[j+1]-1)*U->ldblock+kk];
	    }
	    
	    //compute inverse of  U->val[(U->row[j+1]-1)]
	    //put diagonal block in LU form
	    LAPACKE_dgetrf(LAPACK_ROW_MAJOR, U->blocksize, U->blocksize, &(Uinv[0]), 
			   U->blocksize, ipiv);
	    //compute inverse of U
	    LAPACKE_dgetri(LAPACK_ROW_MAJOR, U->blocksize, Uinv, 
			   U->blocksize, ipiv);
	    
	    //tmp[] = s[] / U->val[U->row[j+1]-1];
	    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
			A->blocksize, A->blocksize,  A->blocksize, one, s, 
			A->blocksize, Uinv, A->blocksize, 
			zero, tmp, A->blocksize);
	    
	    //step += pow( L->val[il-1] - tmp, 2 );
	    //compute        L->val[il-1] - tmp
	    for(int kk=0; kk< A->ldblock; ++kk){L->val[(il-1)*A->ldblock+kk] -= tmp[kk];} 
	    //compute l2 norm of result and add scalar result to step
	    step += cblas_dnrm2(L->ldblock, &(L->val[(il-1)*L->ldblock]), one);
	    for(int kk=0; kk< A->ldblock; ++kk){L->val[(il-1)*A->ldblock+kk]=tmp[kk];} 
	  }
	}
	//row less than column
	else {            // modify u entry
	  for(int kk=0; kk< A->ldblock; ++kk){tmp[kk]=s[kk];}
	  //#pragma omp critical
	  {
	    for(int kk=0; kk< U->ldblock; ++kk){U->val[(iu-1)*U->ldblock+kk] -= tmp[kk];} 
	    //multiply (U[iu-1])*(U[iu-1])
	    step += cblas_dnrm2(U->ldblock, &(U->val[(iu-1)*U->ldblock]), one);
	    for(int kk=0; kk< U->ldblock; ++kk){U->val[(iu-1)*U->ldblock+kk]=tmp[kk];}
	  }
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
  printf("%% PariLU v0.3 bcsr used %d OpenMP threads and required %d iterations, %f wall clock seconds, and an average of %f wall clock seconds per iteration as measured by omp_get_wtime()\n",
	 num_threads, iter, wend-wstart, ompwtime );
  PARILUDBG("PariLUv0_3_bcsr_OpenMP = %d \nPariLUv0_3_iter = %d \nPariLUv0_3_wall = %e \nPariLUv0_3_avgWall = %e \n",num_threads, iter, wend-wstart, ompwtime );
  printf("PariLUv0_3_bcsr_OpenMP = %d \nPariLUv0_3_iter = %d \nPariLUv0_3_wall = %e \nPariLUv0_3_avgWall = %e \n",num_threads, iter, wend-wstart, ompwtime );

  data_zmfree( &LU );
  
  log->sweeps = iter;
  log->tol = tol;
  log->A_Frobenius = Anorm;
  log->precond_generation_time = wend-wstart;
  log->initial_residual = Ares;
  log->initial_nonlinear_residual = Anonlinres;
  log->omp_num_threads = num_threads;
}
