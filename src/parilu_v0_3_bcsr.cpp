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
  data_zmconvert(*A, L, Magma_BCSR, Magma_BCSRL);
  
  U->diagorder_type = Magma_VALUE;
  data_zmconvert(*A, U, Magma_BCSR, Magma_BCSCU);
  
  dataType one = 1.0;
  dataType negone = -1.0;
  dataType zero = 0.0;
  dataType Ares = 0.0;
  dataType Anonlinres = 0.0;
  data_d_matrix LU = {Magma_CSR};
  data_zmconvert(*A, &LU, Magma_BCSR, Magma_BCSR);
  data_zilures(*A, *L, *U, &LU, &Ares, &Anonlinres);
  printf("PariLUv0_2_csrilu0_init_res = %e\n", Ares);
  printf("PariLUv0_2_csrilu0_init_nonlinres = %e\n", Anonlinres);
  
  // ParLU element wise
  int i, j;
  int il, iu, jl, ju;
  
  int iter = 0;
  dataType tol = 1.0e-1;
  tol = reduction*Ares;
  printf("tol = %e\n", tol);
  int num_threads = 0;
  
  //dataType s = 0.0;
  data_d_matrix s = {Magma_DENSE};
  s.num_rows = A->blocksize;
  s.num_cols = A->blocksize;
  s.blocksize = A->blocksize;
  s.nnz = s.num_rows*s.num_cols;
  s.true_nnz = s.nnz;
  s.ld = s.num_cols;
  s.major = MagmaRowMajor;
  LACE_CALLOC(s.val, s.nnz);
  //dataType sp = 0.0;
  data_d_matrix sp = {Magma_DENSE};
  sp.num_rows = A->blocksize;
  sp.num_cols = A->blocksize;
  sp.blocksize = A->blocksize;
  sp.nnz = sp.num_rows*sp.num_cols;
  sp.true_nnz = sp.nnz;
  sp.ld = sp.num_cols;
  sp.major = MagmaRowMajor;
  LACE_CALLOC(sp.val, sp.nnz);
  
  //dataType tmp = 0.0;
  data_d_matrix tmp = {Magma_DENSE};
  tmp.num_rows = A->blocksize;
  tmp.num_cols = A->blocksize;
  tmp.blocksize = A->blocksize;
  tmp.nnz = tmp.num_rows*tmp.num_cols;
  tmp.true_nnz = tmp.nnz;
  tmp.ld = tmp.num_cols;
  tmp.major = MagmaRowMajor;
  LACE_CALLOC(tmp.val, tmp.nnz);
  
  data_d_matrix tmpinv = {Magma_DENSE};
  tmpinv.num_rows = A->blocksize;
  tmpinv.num_cols = A->blocksize;
  tmpinv.blocksize = A->blocksize;
  tmpinv.nnz = tmpinv.num_rows*tmpinv.num_cols;
  tmpinv.true_nnz = tmpinv.nnz;
  tmpinv.ld = tmpinv.num_cols;
  tmpinv.major = MagmaRowMajor;
  LACE_CALLOC(tmpinv.val, tmpinv.nnz);
  
  dataType step = FLT_MAX;
  dataType Anorm = 0.0;
  dataType recipAnorm = 0.0;
  
  data_zfrobenius(*A, &Anorm);
  printf("Anorm = %e\n", Anorm);
  recipAnorm = 1.0/Anorm;
  
  dataType wstart = omp_get_wtime();
  data_rowindex(A, &(A->rowidx) );
  while ( step > tol ) {
    step = 0.0;
    
    //sp = 0.0;
    for ( int kk=0; kk< A->ldblock; kk++) {
      sp.val[kk] = 0.0;
    }
    
    #pragma omp parallel 
    {
      #pragma omp for private(i, j, il, iu, jl, ju, s, sp, tmp, tmpinv) reduction(+:step) nowait
      for (int k=0; k<A->nnz; k++ ) {
        i = A->rowidx[k];
        j = A->col[k];
        //s = A->val[k];
        for ( int kk=0; kk< A->ldblock; kk++) {
          s.val[kk] = 0.0;
        }
        
        il = L->row[i];
        iu = U->row[j];
        while (il < L->row[i+1] && iu < U->row[j+1])
        {
            //sp = 0.0;
            for ( int kk=0; kk< A->ldblock; kk++) {
              sp.val[kk] = 0.0;
            }
            jl = L->col[il];
            ju = U->col[iu];
        
            // avoid branching when possible
            if ( jl == ju ) {
              //sp = ( jl == ju ) ? L->val[il] * U->val[iu] : sp;
              data_dgemm_mkl( MagmaRowMajor, MagmaNoTrans, MagmaNoTrans, 
                A->blocksize, A->blocksize, A->blocksize, 
                one, &(L->val[il*A->ldblock]), A->blocksize, 
                &(U->val[iu*A->ldblock]), A->blocksize,
                zero, sp.val, A->blocksize );
              
              //s = ( jl == ju ) ? s-sp : s;
              data_domatadd_mkl( MagmaRowMajor, MagmaNoTrans, MagmaNoTrans,
                A->blocksize, A->blocksize,
                one, s.val,  A->blocksize,
                negone, sp.val, A->blocksize,
                s.val,  A->blocksize );
            }            
            il = ( jl <= ju ) ? il+1 : il;
            iu = ( jl >= ju ) ? iu+1 : iu;
        }
        // undo the last operation (it must be the last)
        //s += sp;
        data_domatadd_mkl( MagmaRowMajor, MagmaNoTrans, MagmaNoTrans,
          A->blocksize, A->blocksize,
          one, s.val,  A->blocksize,
          one, sp.val, A->blocksize,
          s.val,  A->blocksize );
        
        if ( i > j ) {     // modify l entry
            //tmp = s / U->val[U->row[j+1]-1];
            sp.val = &(U->val[(U->row[j+1]-1)*A->ldblock]);
            data_inverse( &sp, &tmpinv);
            data_dgemm_mkl( MagmaRowMajor, MagmaNoTrans, MagmaNoTrans, 
                A->blocksize, A->blocksize, A->blocksize, 
                one, s.val, A->blocksize, 
                tmpinv.val, A->blocksize,
                zero, tmp.val, A->blocksize );
            for ( int kk=0; kk< A->ldblock; kk++) {
              //step += pow( L->val[il-1] - tmp, 2 );
              step += pow( L->val[((il-1)*A->ldblock)+kk] - tmp.val[kk], 2);
              //L->val[il-1] = tmp; 
              L->val[((il-1)*A->ldblock)+kk] = tmp.val[kk];
            }
        }    
        else {            // modify u entry
            //tmp = s;
            for ( int kk=0; kk< A->ldblock; kk++) {
              //step += pow( U->val[iu-1] - tmp, 2 );
              step += pow( U->val[((iu-1)*A->ldblock)+kk] - tmp.val[kk], 2);
              //U->val[iu-1] = tmp;
              U->val[((iu-1)*A->ldblock)+kk] = s.val[kk];
            }
        }
      }
    }
    step *= recipAnorm;
    iter++;
    printf("%% iteration = %d step = %e\n", iter, step);
  }
  dataType wend = omp_get_wtime();
  dataType ompwtime = (dataType) (wend-wstart)/((dataType) iter);
  
  #pragma omp parallel 
  {
    num_threads = omp_get_num_threads();
  }
  
  printf("%% PariLU v0.3 used %d OpenMP threads and required %d iterations, %f wall clock seconds, and an average of %f wall clock seconds per iteration as measured by omp_get_wtime()\n", 
    num_threads, iter, wend-wstart, ompwtime );
  printf("PariLUv0_3_OpenMP = %d \nPariLUv0_3_iter = %d \nPariLUv0_3_wall = %e \nPariLUv0_3_avgWall = %e \n", 
    num_threads, iter, wend-wstart, ompwtime );
  //data_zmfree( &Atmp );
  data_zmfree( &LU );
  
  log->sweeps = iter;
  log->precond_generation_time = wend-wstart;
  log->initial_residual = Ares;
  log->initial_nonlinear_residual = Anonlinres;
  
}
