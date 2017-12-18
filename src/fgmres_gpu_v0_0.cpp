
#include "../include/sparse.h"
#include <mkl.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include "cublas.h"
#include "cublas_v2.h"
/**
    Purpose
    -------

    Solves a system of linear equations
       A * X = B
    where A is a complex sparse matrix stored in the GPU memory.
    X and B are complex vectors stored on the GPU memory.
    This is a CPU implementation of a basic GMRES based on the discriptions in
    Saad, Yousef. Iterative methods for sparse linear systems.
    Society for Industrial and Applied Mathematics, 2003.
    and
    Kelley, C. T. Iterative methods for linear and nonlinear equations.
    Society for Industrial and Applied Mathematics, Philadelphia, 1995.
    MR 96d 65002.


    Arguments
    ---------

    @param[in]
    A           data_d_matrix*
                descriptor for matrix A

    @param[in]
    b           data_d_matrix*
                RHS b vector

    @param[in,out]
    x           data_d_matrix*
                solution approximation

    @param[in]
    L           data_d_matrix*
                descriptor for matrix A

    @param[in]
    U           data_d_matrix*
                descriptor for matrix A

    @param[in,out]
    solver_par  data_d_solver_par*
                solver parameters

    @param[in]
    precond_par data_d_preconditioner*
                preconditioner

    @ingroup datasparse_linsolvers
    ********************************************************************/


#define USE_CUDA 1


void cublasCheck(cublasStatus_t status, const char *fn_name)
{
  if(status != CUBLAS_STATUS_SUCCESS)
  {
    fprintf(stderr,"cublas error returned %d from %s. exiting...\n", status, fn_name);
    exit(EXIT_FAILURE);
  }
}


extern "C"
int
data_fgmres_gpu(
    data_d_matrix *A, data_d_matrix *b, data_d_matrix *x0,
    data_d_matrix *L, data_d_matrix *U,
    data_d_gmres_param *gmres_par,
    data_d_gmres_log *gmres_log )
{
    printf("%% data_fgmres begin\n");
    dataType wstart = omp_get_wtime();
    dataType zero = 0.0;
    dataType one = 1.0;
    dataType negone = -1.0;

    // initialize
    data_int_t info = DEV_NOTCONVERGED;
    data_int_t n = A->num_rows;
    data_int_t search_max = gmres_par->search_max;
    data_int_t search_directions = 0;
    dataType rtol = gmres_par->rtol;
    dataType rnorm2 = 0.0;
    dataType bnorm = 0.0;
    dataType beta = 0.0;
    dataType delta = 0.0;
    dataType gamma = 0.0;

    // preconditioning
    // for mkl_dcsrtrsv
    char cvar, cvar1, cvar2;
    data_d_matrix LU = {Magma_CSR};
    data_zmlumerge( *L, *U, &LU );
    int* ia;
    int* ja;
    LACE_CALLOC( ia, (LU.num_rows+1) );
    LACE_CALLOC( ja, LU.nnz );

    int chunk = 1;
    int maxThreads = 0;

    //cuda vars
    cublasHandle_t handle;
    cublasStatus_t status;
    dataType* d_uval;
    dataType* d_rval;
    dataType* d_xval;
    dataType* d_yval;
    dataType* d_zval;
    dataType* d_krylov_val;
    dataType* d_alpha_val;
    dataType* d_A_val;
    dataType* d_Minvvj_val;
    dataType d_scalar;
    status = cublasCreate(&handle);//, "cublasCreate";
    status = cublasInit();

    cudaMalloc((void**)&d_uval,(n)*sizeof(dataType));
    cudaMalloc((void**)&d_rval,(n)*sizeof(dataType));
    cudaMalloc((void**)&d_xval,(n)*sizeof(dataType));
    cudaMalloc((void**)&d_yval,(n)*sizeof(dataType));
    cudaMalloc((void**)&d_zval,(n)*sizeof(dataType));
    //cudaMalloc((void**)&d_A_val,(A->nnz)*sizeof(dataType));
    cudaMalloc((void**)&d_krylov_val,(n*search_max)*sizeof(dataType));
    cudaMalloc((void**)&d_alpha_val,(search_max)*sizeof(dataType));    
    cudaMalloc((void**)&d_Minvvj_val,(n*search_max)*sizeof(dataType));   
    //cudaMemcpy(d_A_val,A->val,(A->nnz)*sizeof(dataType),cudaMemcpyHostToDevice);

    //#if USE_CUDA
    //cudaMemcpy(d_ia,ja,(LU.num_rows+1)*sizeof(int),cudaMemcpyHostToDevice);
    //cudaMemcpy(d_ja,ja,(LU.nnz+1)*sizeof(int),cudaMemcpyHostToDevice);  
    //#else
    #pragma omp parallel
    {
      maxThreads = omp_get_max_threads();
      chunk = n/maxThreads;
      #pragma omp for nowait schedule(static,chunk) 
      #pragma simd  
      #pragma vector aligned
      #pragma vector vecremainder
      #pragma nounroll_and_jam
      for (int i=0; i<LU.num_rows+1; i++) {
        ia[i] = LU.row[i] + 1;
      }
      #pragma omp for schedule(static,chunk) nowait
      #pragma simd 
      #pragma vector aligned
      #pragma vector vecremainder
      #pragma nounroll_and_jam
      for (int i=0; i<LU.nnz; i++) {
        ja[i] = LU.col[i] + 1;
      }
    }
    //#endif

    data_d_matrix tmp={Magma_DENSE};
    data_zvinit( &tmp, n, 1, zero );

    // store preconditioned vectors for flexibility
    data_d_matrix Minvvj={Magma_DENSE};
    data_zvinit( &Minvvj, n, search_max, zero );
    //cublasSetVector(n*search_max,sizeof(dataType),Minvvj.val,1,d_Minvvj_val,1);  

    // Partrsv
    dataType ptrsv_tol = 1.0; //1.0e-10;
    int ptrsv_iter = 0;

    // alocate solution and residual vectors
    data_d_matrix x={Magma_DENSE};
    data_zmconvert( *x0, &x, Magma_DENSE, Magma_DENSE );
    cublasSetVector(n,sizeof(dataType),x.val,1,d_xval,1);

    data_d_matrix r={Magma_DENSE};
    data_zvinit( &r, n, 1, zero );

    // Krylov subspace
    data_d_matrix krylov={Magma_DENSE};
    data_zvinit( &krylov, n, search_max+1, zero );
    krylov.major = MagmaColMajor;

    data_d_matrix z={Magma_DENSE};
    data_zvinit( &z, n, 1, zero );
    cublasSetVector(n,sizeof(dataType),z.val,1,d_zval,1);

    // Reorthogonalize
    dataType eps = nextafter(0.0,1.0);
    dataType normav = 0.0;
    dataType normav2 = 0.0;
    dataType hr = 0.0;

    // Hessenberg Matrix for Arnoldi iterations
    data_d_matrix h={Magma_DENSE};
    data_zvinit( &h, search_max+1, search_max, zero );
    h.major = MagmaColMajor;

    data_d_matrix u={Magma_DENSE};
    data_zvinit( &u, n, 1, zero );
    cublasSetVector(n,sizeof(dataType),u.val,1,d_uval,1);

    // Givens rotation vectors
    data_d_matrix givens={Magma_DENSE};
    data_zvinit( &givens, search_max+1, 1, zero );
    givens.major = MagmaColMajor;
    data_d_matrix givens_cos={Magma_DENSE};
    data_zvinit( &givens_cos, search_max, 1, zero );
    givens_cos.major = MagmaColMajor;
    data_d_matrix givens_sin={Magma_DENSE};
    data_zvinit( &givens_sin, search_max, 1, zero );
    givens_sin.major = MagmaColMajor;

    // Coefficents for update to solution vector
    data_d_matrix alpha={Magma_DENSE};
    data_zvinit( &alpha, search_max, 1, zero );
    alpha.major = MagmaColMajor;

    // initial residual
    data_z_spmv( negone, A, &x, zero, &r );
    data_zaxpy( n, one, b->val, 1, r.val, 1);
//ceb replace the above with cublas calls


    cublasSetVector(n,sizeof(dataType), r.val,1, d_rval,1); 
    status = cublasDnrm2(handle, n, d_rval, 1, &rnorm2);
    printf("rnorm2 = %e; tol = %e; rtol = %e;\n", rnorm2, rtol, rtol*rnorm2 );
    if ( gmres_par->tol_type == 1 ) {
      rtol = rtol*rnorm2;
    }
    if (rnorm2 < rtol ) {
      info = 0;
      return info;
    }

    // fill first column of Kylov subspace for Arnoldi iteration
    d_scalar = 1.0/rnorm2;
    cublasDscal(handle,n,&d_scalar, d_rval,1);
    status = cublasDcopy(handle,n,d_rval,1,&d_krylov_val[0*krylov.ld],1);
    givens.val[0] = rnorm2;

    gmres_log->search_directions = search_directions;
    gmres_log->solve_time = 0.0;
    gmres_log->initial_residual = rnorm2;

    // GMRES search direction
    //while ( (rnorm2 > rtol) && (search < search_max) ) {
    for ( int search = 0; search < search_max; search++ ) {

      int search1 = search + 1;

      //reset vectors
      data_zmfree( &u );
      data_zvinit( &u, n, 1, zero );
      data_zmfree( &tmp );
      data_zvinit( &tmp, n, 1, zero );

      cublasGetVector(n,sizeof(dataType),&d_krylov_val[search*krylov.ld],1,
		      &krylov.val[search*krylov.ld],1);

#if 1 //Want to minimize copy of data to device by copying only one vector per iteration
      //rather than all vectors at once later on in the loop
//cublasGetVector(n,sizeof(dataType),&d_Minvvj_val[search*krylov.ld],1,&Minvvj.val[search*krylov.ld],1);
#endif  

//ceb Krylov.val is modified here
      if ( gmres_par->user_csrtrsv_choice == 0 ) {

        // Apply preconditioner to krylov.val[idx(A->col[j],search,krylov.ld)]
        cvar1='L';
        cvar='N';
        cvar2='U';
//ceb can we replace this with a cuda call on d_krylov_val
        mkl_dcsrtrsv( &cvar1, &cvar, &cvar2, &n, LU.val, ia, ja,
          &(krylov.val[idx(0,search,krylov.ld)]), tmp.val );

        cvar1='U';
        cvar='N';
        cvar2='N';
//ceb can we replace this with a cuda call on d_krylov_val
        mkl_dcsrtrsv( &cvar1, &cvar, &cvar2, &n, LU.val, ia, ja,
          tmp.val, &(Minvvj.val[idx(0,search,krylov.ld)]) );

      }
      else if ( gmres_par->user_csrtrsv_choice == 1 ) {
//ceb can we replace this with a cuda call on d_krylov_val
        data_parcsrtrsv( MagmaLower, L->storage_type, L->diagorder_type,
          L->num_rows, L->val, L->row, L->col,
          &(krylov.val[idx(0,search,krylov.ld)]), tmp.val,
          ptrsv_tol, &ptrsv_iter );
        printf("ParCSRTRSV_L(%d) = %d;\n", search1, ptrsv_iter);

//ceb can we replace this with a cuda call on d_krylov_val
        data_parcsrtrsv( MagmaUpper, U->storage_type, U->diagorder_type,
          U->num_rows, U->val, U->row, U->col,
          tmp.val, &(Minvvj.val[idx(0,search,krylov.ld)]),
          ptrsv_tol, &ptrsv_iter );
        printf("ParCSRTRSV_U(%d) = %d;\n", search1, ptrsv_iter);
      }
      else if ( gmres_par->user_csrtrsv_choice == 2 ) {
        for ( int i=0; i<Minvvj.ld; i++ ) {
//ceb can we replace this with a cuda call on d_krylov_val
          Minvvj.val[idx(i,search,Minvvj.ld)] = krylov.val[idx(i,search,krylov.ld)];
        }
      }

      cublasSetVector(n,sizeof(dataType),&krylov.val[search*krylov.ld],1,
		      &d_krylov_val[search*krylov.ld],1);

#if USE_CUDA

//cublasSetVector(n,sizeof(dataType),&Minvvj.val[search*krylov.ld],1,&d_Minvvj_val[search*krylov.ld],1);  
//cublasSetVector(n*search1,sizeof(dataType),Minvvj.val,1,d_Minvvj_val,1);  
#endif



#if 0
      //ceb
      //difficulty on cuda conversion due to indirect indexing A->col[j] into Minvvj
      //need to write custom cuda kernel to handle this
#else
      #pragma omp parallel
      #pragma omp for schedule(static,chunk) nowait
      #pragma simd 
      #pragma vector aligned
      #pragma vector vecremainder
      #pragma nounroll_and_jam
      for ( int i=0; i<n; i++ ) {
        for ( int j=A->row[i]; j<A->row[i+1]; j++ ) {
          u.val[i] = u.val[i] + A->val[j] * Minvvj.val[idx(A->col[j],search,krylov.ld)];
        }
      } 
#endif
      //ceb
      cublasSetVector(n,sizeof(dataType),u.val,1,d_uval,1);
      //l2nrm
      status = cublasDnrm2(handle, n, d_uval, 1, &normav);

      // Modified Gram-Schmidt
      // need to keep history of krylov vectors to produce current vector
      for ( int j=0; j <= search; j++ ) {
        status = cublasDdot(handle, n, d_uval,1, &d_krylov_val[j*krylov.ld],1,&d_scalar);
        h.val[idx(j,search,h.ld)] = d_scalar;
        d_scalar= -h.val[idx(j,search,h.ld)];
        status = cublasDaxpy(handle,n,&d_scalar,&d_krylov_val[j*krylov.ld],1,d_uval,1);
      }
      //l2nrm
      status = cublasDnrm2(handle, n, d_uval, 1, &h.val[idx((search1),search, h.ld)]);
      normav2 = h.val[idx((search1),search,h.ld)];

//cublasSetVector(n*search_max,sizeof(dataType),Minvvj.val,1,d_Minvvj_val,1);  

      // Reorthogonalize?
      hr = (normav + 0.001*normav2) - normav;
      if ( ( gmres_par->reorth == 0 && hr <= eps ) || gmres_par->reorth == 2 ) {
        printf("Reorthogonalize %d\n", search);
        for ( int j=0; j <= search; j++ ) {
          hr = 0.0;
          //dot product
          status=cublasDdot(handle,n, d_uval,1, &d_krylov_val[j*krylov.ld],1,&hr);
          h.val[idx(j,search,h.ld)] = h.val[idx(j,search,h.ld)] + hr;       
          d_scalar=-hr;
          status = cublasDaxpy(handle,n, &d_scalar, &d_krylov_val[j*krylov.ld],1, d_uval,1);
        }

        //l2nrm
        status = cublasDnrm2(handle, n, d_uval, 1, &h.val[idx((search1),search, h.ld)]);
      }

      // Watch out for happy breakdown
      if ( fabs(h.val[idx((search1),search,h.ld)]) > 0 ) {
        d_scalar=1.0/h.val[idx((search1),search,h.ld)];
        status = cublasDcopy(handle, n, d_uval,1,  &d_krylov_val[search1*krylov.ld],1);
        status = cublasDscal(handle, n, &d_scalar, &d_krylov_val[search1*krylov.ld],1);
      }
      else {
        printf("%%\t******* happy breakdown **********\n");
      }

#if 1 //ceb Need to figure out how to avoid copying all vectors at once to the GPU here
      cublasSetVector(n*search,sizeof(dataType),Minvvj.val,1,d_Minvvj_val,1);  
      cublasSetVector(n,sizeof(dataType),&Minvvj.val[search*krylov.ld],1,
		      &d_Minvvj_val[search*krylov.ld],1);  
#endif


#if 0
      if (gmres_par->monitorOrthog == 1) {
        //ceb To compute orthogonality error here, we need to copy krylov data back 
	//    from host to device
	cublasGetVector(n,sizeof(dataType),&d_krylov_val[search1*krylov.ld],1,
			&krylov.val[search1*krylov.

        // Monitor Orthogonality Error of Krylov search Space
        dataType ortherr = 0.0;
        int imax = 0;
        data_orthogonality_error( &krylov, &ortherr, &imax, (search1) );
        //data_orthogonality_error( &Minvvj, &ortherr, &imax, (search1) );
        if ( gmres_par->user_csrtrsv_choice == 0 ) {
          printf("FGMRES_mkltrsv_ortherr(%d) = %.16e;\n", search1, ortherr);
        }
        else {
          printf("FGMRES_partrsv_ortherr(%d) = %.16e;\n", search1, ortherr);
        }

        data_orthogonality_error_incremental( &krylov, &ortherr, &imax, (search1) );
        if ( gmres_par->user_csrtrsv_choice == 0 ) {
          printf("FGMRES_mkltrsv_ortherr_inc(%d) = %.16e;\n", search1, ortherr);
        }
        else {
          printf("FGMRES_partrsv_ortherr_inc(%d) = %.16e;\n", search1, ortherr);
        }

      }
#endif

      // Givens rotations
      for ( int j = 0; j<search; j++ ) {
        delta = h.val[idx(j,search,h.ld)];
        h.val[idx(j,search,h.ld)] =
          givens_cos.val[j]*delta
          - givens_sin.val[j]*h.val[idx(j+1,search,h.ld)];

        h.val[idx(j+1,search,h.ld)] =
          givens_sin.val[j]*delta
          + givens_cos.val[j]*h.val[idx(j+1,search,h.ld)];
      }

      gamma = sqrt(
        h.val[idx(search,search,h.ld)]*h.val[idx(search,search,h.ld)] +
        h.val[idx((search1),search,h.ld)]*h.val[idx((search1),search,h.ld)] );
      //printf("gamma = %e\n", gamma);
      if ( gamma > 0.0 ) {
        givens_cos.val[search] = h.val[idx(search,search,h.ld)]/gamma;
        givens_sin.val[search] = -h.val[idx((search1),search,h.ld)]/gamma;
        h.val[idx(search,search,h.ld)] =
          givens_cos.val[search]*h.val[idx(search,search,h.ld)] -
          givens_sin.val[search]*h.val[idx((search1),search,h.ld)];
        h.val[idx((search1),search,h.ld)] = 0.0;
        delta = givens.val[search];
        givens.val[search] =
          givens_cos.val[search]*delta - givens_sin.val[search]*givens.val[(search1)];
        givens.val[(search1)] =
          givens_sin.val[search]*delta + givens_cos.val[search]*givens.val[(search1)];
      }



      //printf("%%======= FGMRES search %d fabs(givens.val[(%d+1)]) = %.16e =======\n", search, search, fabs(givens.val[(search1)]));
      if ( gmres_par->user_csrtrsv_choice == 0 ) {
        printf("FGMRES_mkltrsv_search(%d) = %.16e;\n", search1, fabs(givens.val[(search1)]));
      }
      else {
        printf("FGMRES_partrsv_search(%d) = %.16e;\n", search1, fabs(givens.val[(search1)]));
      }

      // update the solution

      // solve the least squares problem
      if ( fabs(givens.val[(search1)]) < rtol  || (search == (search_max-1)) ) {
#if 0
	//need to write custom cuda kernel for this nested loop with decrementing index
#else
        GMRESDBG(" !!!!!!! update the solution !!!!!!!\n");
        for ( int i = 0; i <= search; i++ ) {
          alpha.val[i] = givens.val[i]/h.val[idx(i,i,h.ld)];
        }
        for ( int j = search; j > 0; j-- ) {
          for (int i = j-1; i > -1; i-- ) {
            alpha.val[i] = alpha.val[i]
             - h.val[idx(i,j,h.ld)]*alpha.val[j]/h.val[idx(i,i,h.ld)];
          }
#endif
        }

        // use preconditioned vectors to form the update (GEMV)
        cublasSetVector(search_max,sizeof(dataType),alpha.val,1,d_alpha_val,1);        
        int row=n;
        int col=search_max;  
        d_scalar=1.0;
  
        status = cublasDgemv( handle, CUBLAS_OP_N,
                    row, col, 
                    &d_scalar, d_Minvvj_val, row,
                    d_alpha_val, 1, 
                    &d_scalar, d_zval, 1);

        status = cublasDaxpy(handle,n,&d_scalar,d_zval,1,d_xval,1);

        gmres_log->search_directions = search1;
        dataType wend = omp_get_wtime();
        gmres_log->solve_time = (wend-wstart);
        gmres_log->final_residual = fabs(givens.val[(search1)]);

        break;
      }

    }
        
    //retrieve x result to host
    cublasGetVector(n,sizeof(dataType),d_xval,1,x.val,1);

    fflush(stdout);
    data_zmfree( x0 );
    data_zmconvert( x, x0, Magma_DENSE, Magma_DENSE );


    if (gmres_log->final_residual > rtol) {
      info = 0;
    }

    data_zmfree( &x );
    data_zmfree( &r );
    data_zmfree( &krylov );
    data_zmfree( &h );
    data_zmfree( &u );
    data_zmfree( &givens );
    data_zmfree( &givens_cos );
    data_zmfree( &givens_sin );
    data_zmfree( &alpha );
    data_zmfree( &z );

    data_zmfree( &LU );
    free( ia );
    free( ja );
    data_zmfree( &tmp );
    data_zmfree( &Minvvj );

#if USE_CUDA
    cudaFree(d_uval);
    cudaFree(d_rval);
    cudaFree(d_xval);
    cudaFree(d_yval);
    cudaFree(d_zval);
    cudaFree(d_krylov_val);
    cudaFree(d_alpha_val);
    cudaFree(d_Minvvj_val);
    cublasShutdown();
#endif

    return info;
}
