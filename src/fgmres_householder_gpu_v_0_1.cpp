//#define DEBUG_GMRES
//ceb fgmres_householder_gpu_v_0_1.cpp 
//    This port to GPU is a partial conversion using only CUBLAS routines.
//    This partial conversion requires many copies to and from the GPU devices 
//    to go back and forth from CPU and GPU functions

#include "../include/sparse.h"
#include "../include/cuda_tools.h"
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


extern "C"
int
data_fgmres_householder_gpu_v_0_1(
    data_d_matrix *A, data_d_matrix *b, data_d_matrix *x0,
    data_d_matrix *L, data_d_matrix *U,
    data_d_gmres_param *gmres_par,
    data_d_gmres_log *gmres_log )
{

    printf("%% data_fgmres_householder begin\n");
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
    dataType residual = 0.0;
    data_int_t search = 0; // search directions

    
    int span;
    //cuda vars
    cublasHandle_t handle;
    cublasStatus_t status;
    dataType* d_uval;
    dataType* d_qval;
    dataType* d_rval;
    dataType* d_xval;
    dataType* d_yval;
    dataType* d_zval;
    dataType* d_krylov_val;
    dataType* d_alpha_val;
    dataType* d_A_val;
    dataType* d_Minvvj_val;
    dataType* d_precondq_val;
    dataType d_scalar;
    dataType snrm2;
    //cublasCheckCore(cublasCreate(&handle),__FILE__,__LINE__);//ceb
    cublasCheck(cublasCreate(&handle));
    cublasCheck(cublasInit());

#if 0//test managed
    cudaCheck(cudaMalloc((void**)&d_uval,(n)*sizeof(dataType)));
    cudaCheck(cudaMalloc((void**)&d_qval,(n)*sizeof(dataType)));
    cudaCheck(cudaMalloc((void**)&d_rval,(n)*sizeof(dataType)));
    cudaCheck(cudaMalloc((void**)&d_xval,(n)*sizeof(dataType)));
    cudaCheck(cudaMalloc((void**)&d_yval,(n)*sizeof(dataType)));
    cudaCheck(cudaMalloc((void**)&d_zval,(n)*sizeof(dataType)));
    //cudaMalloc((void**)&d_A_val,(A->nnz)*sizeof(dataType));
    cudaCheck(cudaMalloc((void**)&d_krylov_val,(n*(search_max+1))*sizeof(dataType)));
    cudaCheck(cudaMalloc((void**)&d_alpha_val,(search_max+1)*sizeof(dataType)));    
    cudaCheck(cudaMalloc((void**)&d_Minvvj_val,(n*(search_max))*sizeof(dataType)));   
    cudaCheck(cudaMalloc((void**)&d_precondq_val,(n*(search_max+1))*sizeof(dataType))); 
#else
    cudaCheck(cudaMallocManaged((void**)&d_uval,(n)*sizeof(dataType)));
    cudaCheck(cudaMallocManaged((void**)&d_qval,(n)*sizeof(dataType)));
    cudaCheck(cudaMallocManaged((void**)&d_rval,(n)*sizeof(dataType)));
    cudaCheck(cudaMallocManaged((void**)&d_xval,(n)*sizeof(dataType)));
    cudaCheck(cudaMallocManaged((void**)&d_yval,(n)*sizeof(dataType)));
    cudaCheck(cudaMallocManaged((void**)&d_zval,(n)*sizeof(dataType)));
    //cudaMalloc((void**)&d_A_val,(A->nnz)*sizeof(dataType));
    cudaCheck(cudaMallocManaged((void**)&d_krylov_val,(n*(search_max+1))*sizeof(dataType)));
    cudaCheck(cudaMallocManaged((void**)&d_alpha_val,(search_max+1)*sizeof(dataType)));    
    cudaCheck(cudaMallocManaged((void**)&d_Minvvj_val,(n*(search_max))*sizeof(dataType)));   
    cudaCheck(cudaMallocManaged((void**)&d_precondq_val,(n*(search_max+1))*sizeof(dataType)));
#endif

    cudaDeviceProp prop;
    int device=0;
    cudaGetDeviceProperties(&prop,device);
    //printf("device name=%s \n",prop.name);
    //printf("warpSize=%d \n",prop.warpSize);
    printf("maxThreadsPerBlock=%d \n",prop.maxThreadsPerBlock);
    int nthreads = prop.maxThreadsPerBlock;
    //printf("multiProcessorCount=%d \n",prop.multiProcessorCount);
    printf("n=%d search_max=%d\n",n,search_max);


    // preconditioning
    // for mkl_dcsrtrsv
    char cvar, cvar1, cvar2;
    data_d_matrix LU = {Magma_CSR};
    data_zmlumerge( *L, *U, &LU );
    int* ia;
    int* ja;
    LACE_CALLOC( ia, (LU.num_rows+1) );
    LACE_CALLOC( ja, LU.nnz );

    #pragma omp parallel
    {
      #pragma omp for nowait
      for (int i=0; i<LU.num_rows+1; i++) {
        ia[i] = LU.row[i] + 1;
      }
      #pragma omp for nowait
      for (int i=0; i<LU.nnz; i++) {
        ja[i] = LU.col[i] + 1;
      }
    }

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
    data_d_matrix r={Magma_DENSE};
    data_zvinit( &r, n, 1, zero );
    data_d_matrix q={Magma_DENSE};
    data_zvinit( &q, n, 1, zero );

    // Krylov subspace
    data_d_matrix krylov={Magma_DENSE};
    data_zvinit( &krylov, n, search_max+1, zero );
    krylov.major = MagmaColMajor;
    data_d_matrix z={Magma_DENSE};
    data_zvinit( &z, n, 1, zero );
    data_d_matrix precondq={Magma_DENSE};
    data_zvinit( &precondq, n, search_max+1, zero );
    precondq.major = MagmaColMajor;

    // Reorthogonalize
    //dataType eps = nextafter(0.0,1.0);
    //dataType normav = 0.0;
    //dataType normav2 = 0.0;
    //dataType hr = 0.0;

    // Hessenberg Matrix for Arnoldi iterations
    data_d_matrix h={Magma_DENSE};
    data_zvinit( &h, search_max+1, search_max, zero );
    h.major = MagmaColMajor;
    data_d_matrix u={Magma_DENSE};
    data_zvinit( &u, n, 1, zero );

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
    dataType eps = nextafter(0.0,1.0);

    // Coefficents for update to solution vector
    data_d_matrix alpha={Magma_DENSE};
    data_zvinit( &alpha, search_max, 1, zero );
    alpha.major = MagmaColMajor;
    cublasCheck(cublasDscal(handle,n,&zero,d_alpha_val,1));

    // initial residual
    data_z_spmv( negone, A, &x, zero, &r );
    data_zaxpy( n, one, b->val, 1, r.val, 1);

#if 1//USE_CUDA
    cublasCheck(cublasSetVector(n,sizeof(dataType), x.val,1, d_xval,1));
    cublasCheck(cublasSetVector(n,sizeof(dataType), z.val,1, d_zval,1));
    cublasCheck(cublasSetVector(n,sizeof(dataType), u.val,1, d_uval,1));
    cublasCheck(cublasSetVector(n,sizeof(dataType), r.val,1, d_rval,1)); 
    cublasCheck(cublasDnrm2(handle, n, d_rval, 1, &rnorm2));
#else
    rnorm2 = data_dnrm2( n, r.val, 1 );
#endif

    printf("rnorm2 = %e; tol = %e; rtol = %e;\n", rnorm2, rtol, rtol*rnorm2 );
    if ( gmres_par->tol_type == 1 ) {
      rtol = rtol*rnorm2;
    }
    if (rnorm2 < rtol ) {
      info = 0;
      return info;
    }

#if 1//USE_CUDA
    cublasCheck(cublasDcopy(handle,n,d_rval,1,d_krylov_val,1));

    //printf("d_rval[0]=%e\n",d_rval[0]);
    //printf("d_krylov_val[0]=%e\n",d_krylov_val[0]);


    dataType k1norm2;

    //#if 0//not malloc managed
    ////it is inefficient to copy this back to the host to perform the following two operations
    //cublasCheck(cublasGetVector(n,sizeof(dataType), d_krylov_val,1, krylov.val,1),"cublasGetVector ln 244");
    ////get sign mysgn(T v) {return T(v >= T(0)) - T(v < T(0));}

    //dataType dd = mysgn(krylov.val[0+0*krylov.ld])*rnorm2;
    //krylov.val[0+0*krylov.ld] = krylov.val[0+0*krylov.ld] + dd;

    ////copy back to coprocessor
    //cublasCheck(cublasSetVector(n,sizeof(dataType), krylov.val,1, d_krylov_val,1),"cublasSetVector ln 249");
    //#else

    //get sign mysgn(T v) {return T(v >= T(0)) - T(v < T(0));}
    cudaDeviceSynchronize();
    dataType dd = mysgn(d_krylov_val[0+0*krylov.ld])*rnorm2;
    d_krylov_val[0+0*krylov.ld] = d_krylov_val[0+0*krylov.ld] + dd;

    //#endif 


    cublasCheck(cublasDnrm2(handle, n, d_krylov_val, 1, &k1norm2));
    d_scalar = 1.0/k1norm2;
    cublasCheck(cublasDscal(handle,n,&d_scalar, d_krylov_val,1));

    cudaDeviceSynchronize();

cublasCheck(cublasGetVector(n,sizeof(dataType), d_krylov_val,1, krylov.val,1));


#else
//   // fill first column of Kylov subspace for Arnoldi iteration
//    for ( int i=0; i<n; i++ ) {
//      //krylov.val[idx(i,0,krylov.ld)] = r.val[i]/rnorm2;
//      krylov.val[idx(i,0,krylov.ld)] = r.val[i];
//    }
//    dataType dd = mysgn(krylov.val[idx(0,0,krylov.ld)])*rnorm2;
//    krylov.val[idx(0,0,krylov.ld)] = krylov.val[idx(0,0,krylov.ld)] + dd;
//    dataType k1norm2 = data_dnrm2( n, krylov.val, 1 );
//    for ( int i=0; i<n; ++i ) {
//      krylov.val[idx(i,0,krylov.ld)] = krylov.val[idx(i,0,krylov.ld)]/k1norm2;
//    }
#endif


#if 1//USE_CUDA
    d_scalar=-1.0/dd;
    cublasCheck(cublasDcopy(handle,n,d_rval,1,d_qval,1));
    cublasCheck(cublasDscal(handle,n,&d_scalar, d_qval,1));
    cublasCheck(cublasDcopy(handle,n,d_qval,1,d_precondq_val,1));
cublasCheck(cublasGetVector(n,sizeof(dataType), d_qval,1, q.val,1));
cublasCheck(cublasGetVector(n,sizeof(dataType), d_precondq_val,1, precondq.val,1));
#else
//for ( int i=0; i<n; ++i ) {
//    q.val[i] = -r.val[i]/dd;
//    precondq.val[idx(i,0,precondq.ld)] = q.val[i];
//  }
#endif


    for ( int i=0; i<n; ++i ) {
      //GMRESDBG("q.val[%d] = %e\n", i, q.val[i]);
      GMRESDBG("q.val[%d] = %e\n", i, d_qval[i]);
    }
    //givens.val[0] = rnorm2;
    givens.val[0] = -dd;

    search = 0;

    gmres_log->search_directions = search_directions;
    gmres_log->solve_time = 0.0;
    gmres_log->initial_residual = rnorm2;

    // GMRES search direction
    for ( int search = 0; search < search_max; search++ ) {
      int search1 = search + 1;
      data_zmfree( &u );
      data_zvinit( &u, n, 1, zero );
      data_zmfree( &tmp );
      data_zvinit( &tmp, n, 1, zero );

      for ( int i=0; i<krylov.ld; i++ ) {
        GMRESDBG("\tkrylov.val[idx(%d,%d,%d)] = %e\n", i, search, krylov.ld, krylov.val[idx(i,search,krylov.ld)]);
      }

      if ( gmres_par->user_csrtrsv_choice == 0 ) {
        // Apply preconditioner to krylov.val[idx(A->col[j],search,krylov.ld)]
        cvar1='L';
        cvar='N';
        cvar2='U';
        mkl_dcsrtrsv( &cvar1, &cvar, &cvar2, &n, LU.val, ia, ja,
          q.val, tmp.val );
        cvar1='U';
        cvar='N';
        cvar2='N';
        mkl_dcsrtrsv( &cvar1, &cvar, &cvar2, &n, LU.val, ia, ja,
          tmp.val, &(Minvvj.val[idx(0,search,Minvvj.ld)]) );
      }
      else {
        data_parcsrtrsv( MagmaLower, L->storage_type, L->diagorder_type,
          L->num_rows, L->val, L->row, L->col,
          q.val, tmp.val,
          ptrsv_tol, &ptrsv_iter );
        printf("ParCSRTRSV_L(%d) = %d;\n", search+1, ptrsv_iter);

        data_parcsrtrsv( MagmaUpper, U->storage_type, U->diagorder_type,
          U->num_rows, U->val, U->row, U->col,
          tmp.val, &(Minvvj.val[idx(0,search,krylov.ld)]),
          ptrsv_tol, &ptrsv_iter );
        printf("ParCSRTRSV_U(%d) = %d;\n", search+1, ptrsv_iter);
      }

      for ( int i=0; i<Minvvj.ld; i++ ) {
        GMRESDBG("Minvvj.val[idx(%d,%d,%d)] = %e\n",
          i, search, Minvvj.ld, Minvvj.val[idx(i,search,Minvvj.ld)]);
      }

#if 1//USE_CUDA
      cublasCheck(cublasSetVector(n,sizeof(dataType),&krylov.val[search*krylov.ld],1,
		      &d_krylov_val[search*krylov.ld],1));

      cublasCheck(cublasSetVector(n,sizeof(dataType),&Minvvj.val[search*Minvvj.ld],1,
		      &d_Minvvj_val[search*Minvvj.ld],1));
#endif

#if 0//USE_CUDA
      //ceb
      //difficulty on cuda conversion due to indirect indexing A->col[j] into Minvvj
      //need to write custom cuda kernel to handle this
      //#pragma omp parallel for
      cudaDeviceSynchronize();
        for ( int i=0; i<n; i++ ) {
          for ( int j=A->row[i]; j<A->row[i+1]; j++ ) {
            //u.val[i] = u.val[i] + A->val[j]*Minvvj.val[A->col[j]];
            d_krylov_val[i+search1*krylov.ld] = d_krylov_val[i+search1*krylov.ld] + A->val[j]*Minvvj.val[A->col[j]+search*Minvvj.ld];
          }
        }
      cudaDeviceSynchronize();

#else
      //#pragma omp parallel for
        for ( int i=0; i<n; i++ ) {
          for ( int j=A->row[i]; j<A->row[i+1]; j++ ) {
            //u.val[i] = u.val[i] + A->val[j]*Minvvj.val[A->col[j]];
            krylov.val[idx(i,search1,krylov.ld)] = krylov.val[idx(i,search1,krylov.ld)] + A->val[j]*Minvvj.val[idx(A->col[j],search,Minvvj.ld)];
          }
        }
#endif
      
      for ( int j=0; j <= search1; j++ ) {
        for ( int i=0; i<krylov.ld; i++ ) {
          GMRESDBG("krylov.val[idx(%d,%d,%d)] = %e\n", i, j, krylov.ld, krylov.val[idx(i,j,krylov.ld)]);
        }
      }

      // Householder Transformations

      for ( int j=0; j <= search; ++j ) {
        dataType sum = 0.0;

#if 1//USE_CUDA
	cublasCheck(cublasSetVector(n*(search_max+1),sizeof(dataType), krylov.val,1, d_krylov_val,1));

        span = n-j;
        cublasCheck(cublasDdot(handle, span, &d_krylov_val[j+j*krylov.ld],1, 
		       &d_krylov_val[j+search1*krylov.ld],1,&sum));

        d_scalar = -2.0*sum;
        cublasCheck(cublasDaxpy(handle, span, &d_scalar, &d_krylov_val[j+j*krylov.ld],1, 
			&d_krylov_val[j+search1*krylov.ld],1));

	cublasCheck(cublasGetVector(n*(search_max+1),sizeof(dataType), d_krylov_val,1, krylov.val,1));
#else
        for ( int i=j; i<n; ++i ) {
          sum = sum + krylov.val[idx(i,j,krylov.ld)]*krylov.val[idx(i,search1,krylov.ld)];
        }
        for ( int jj=j; jj < n; ++jj ) {
          krylov.val[idx(jj,search1,krylov.ld)] = krylov.val[idx(jj,search1,krylov.ld)] - 2.0*sum*krylov.val[idx(jj,j,krylov.ld)];
        }
#endif
      }
      for ( int j=0; j <= search1; ++j ) {
        for ( int i=0; i<n; ++i ) {
          GMRESDBG("krylov.val[idx(%d,%d,%d)] = %e\n", i, j, krylov.ld, krylov.val[idx(i,j,krylov.ld)]);
        }
      }

      if ( search < n ) {
#if 1//USE_CUDA
        dataType snrm2;
        span=n-search;
        cublasCheck(cublasDnrm2(handle, span, &d_krylov_val[search1+search1*krylov.ld], 1, &snrm2));

	cublasCheck(cublasGetVector(n*(search_max+1),sizeof(dataType),d_krylov_val,1,krylov.val,1));
        dd = mysgn(krylov.val[idx(search1,search1,krylov.ld)])*snrm2;
        krylov.val[idx(search1,search1,krylov.ld)] = krylov.val[idx(search1,search1,krylov.ld)] + dd;
	cublasCheck(cublasSetVector(n,sizeof(dataType),&krylov.val[search1*krylov.ld],1, &d_krylov_val[search1*krylov.ld],1));

        cublasCheck(cublasDnrm2(handle, span, &d_krylov_val[search1+search1*krylov.ld], 1, &snrm2));
        d_scalar=1.0/snrm2;
        cublasCheck(cublasDscal(handle, span, &d_scalar, &d_krylov_val[search1+search1*krylov.ld], 1));

	cublasCheck(cublasGetVector(n*(search_max+1),sizeof(dataType),d_krylov_val,1,krylov.val,1));

        data_zmfree( &q );
        data_zvinit( &q, n, 1, zero );
        q.val[search1] = 1.0;

	cublasCheck(cublasSetVector(n,sizeof(dataType), q.val,1, d_qval,1));

        for (int j=search1; j>=0; --j) {
          dataType sum = 0.0;
          span=n-j;
          cublasCheck(cublasDdot(handle, span, &d_qval[j],1, &d_krylov_val[j+j*krylov.ld],1,&sum));
          d_scalar = -2.0*sum;
	  cublasCheck(cublasDaxpy(handle, span, &d_scalar, &d_krylov_val[j+j*krylov.ld],1, &d_qval[j],1));
	  cublasCheck(cublasGetVector(span,sizeof(dataType), &d_qval[j],1, &q.val[j],1));

        }

	cublasCheck(cublasDcopy(handle, n, d_qval,1,  &d_precondq_val[search1*precondq.ld],1));        cublasCheck(cublasGetVector(n,sizeof(dataType), &d_precondq_val[search1*precondq.ld],1, &precondq.val[search1*precondq.ld],1));

#else
        dataType snrm2 = data_dnrm2( (n-search), &(krylov.val[idx(search1,search1,krylov.ld)]), 1 );
        dd = mysgn(krylov.val[idx(search1,search1,krylov.ld)])*snrm2;
        GMRESDBG("dd = %e  %d\n", dd, __LINE__);
        krylov.val[idx(search1,search1,krylov.ld)] = krylov.val[idx(search1,search1,krylov.ld)] + dd;
        snrm2 = data_dnrm2( (n-search), &(krylov.val[idx(search1,search1,krylov.ld)]), 1 );
        for ( int i=search1; i<n; ++i ) {
          krylov.val[idx(i,search1,krylov.ld)] = krylov.val[idx(i,search1,krylov.ld)]/snrm2;
        }
        for ( int j=0; j <= search1; ++j ) {
          for ( int i=0; i<n; ++i ) {
            GMRESDBG("krylov.val[idx(%d,%d,%d)] = %e\n", i, j, krylov.ld, krylov.val[idx(i,j,krylov.ld)]);
          }
        }

        data_zmfree( &q );
        data_zvinit( &q, n, 1, zero );
        q.val[search1] = 1.0;
        for (int j=search1; j>=0; --j) {
          dataType sum = 0.0;
          for ( int i=j; i<n; ++i ) {
            sum = sum + krylov.val[idx(i,j,krylov.ld)]*q.val[i];
          }
          for ( int jj=j; jj < n; ++jj ) {
            q.val[jj] = q.val[jj] - 2.0*sum*krylov.val[idx(jj,j,krylov.ld)];
          }
        }
        for ( int i=0; i<n; ++i ) {
          precondq.val[idx(i,search1,precondq.ld)] = q.val[i];
        }
#endif
      }


#if 1//USE_CUDA
      //cublasCheck(cublasGetVector(n*(search_max+1),sizeof(dataType), d_precondq_val,1, precondq.val,1),"cublasGetVector ln 420");
      // Monitor Orthogonality Error of Krylov search Space
      dataType ortherr = 0.0;
      int imax = 0;
      data_orthogonality_error( &precondq, &ortherr, &imax, search1 );
      if ( gmres_par->user_csrtrsv_choice == 0 ) {
        printf("FGMRES_Householders_mkltrsv_ortherr(%d) = %.16e;\n", search+1, ortherr);
      }
      else {
        printf("FGMRES_Householders_partrsv_ortherr(%d) = %.16e;\n", search+1, ortherr);
      }
#else


      // Monitor Orthogonality Error of Krylov search Space
      dataType ortherr = 0.0;
      int imax = 0;
      data_orthogonality_error( &precondq, &ortherr, &imax, search1 );
      if ( gmres_par->user_csrtrsv_choice == 0 ) {
        printf("FGMRES_Householders_mkltrsv_ortherr(%d) = %.16e;\n", search+1, ortherr);
      }
      else {
        printf("FGMRES_Householders_partrsv_ortherr(%d) = %.16e;\n", search+1, ortherr);
      }
#endif


      // Apply Givens rotations
      for ( int j = 0; j<search; ++j ) {
        dataType temp = givens_cos.val[j]*krylov.val[idx(j,search1,krylov.ld)]
          + givens_sin.val[j]*krylov.val[idx(j+1,search1,krylov.ld)];
        krylov.val[idx(j+1,search1,krylov.ld)] = -givens_sin.val[j]*krylov.val[idx(j,search1,krylov.ld)]
          + givens_cos.val[j]*krylov.val[idx(j+1,search1,krylov.ld)];
        krylov.val[idx(j,search1,krylov.ld)] = temp;
        GMRESDBG("\n\tApply Givens rotations search = %d temp = %e\n", j, temp);
      }

      for ( int j=0; j <= search1; ++j ) {
        for ( int i=0; i<n; ++i ) {
          GMRESDBG("krylov.val[idx(%d,%d,%d)] = %e\n", i, j, krylov.ld, krylov.val[idx(i,j,krylov.ld)]);
        }
      }

      if ( search < n ) {
        GMRESDBG("%e %e %e %e %e\n", krylov.val[idx(search,search1,krylov.ld)], -dd, eps, givens_cos.val[search], givens_sin.val[search] )
        // form search-th rotation matrix
        givens_rotation(krylov.val[idx(search,search1,krylov.ld)], -dd, eps, &givens_cos.val[search], &givens_sin.val[search] );

        for ( int i=0; i<n; ++i ) {
          GMRESDBG("givens_cos.val[%d] = %e\n", i, givens_cos.val[i]);
        }

        for ( int i=0; i<n; ++i ) {
          GMRESDBG("givens_sin.val[%d] = %e\n", i, givens_sin.val[i]);
        }

        krylov.val[idx(search,search1,krylov.ld)] = 
	  givens_cos.val[search]*krylov.val[idx(search,search1,krylov.ld)]
          - givens_sin.val[search]*dd;

        // approximate residual norm
        givens.val[search1] = -givens_sin.val[search]*givens.val[search];
        givens.val[search] = givens_cos.val[search]*givens.val[search];
        residual = fabs(givens.val[search1]); // /rnorm2;
      }

      for ( int i=0; i<givens.ld; ++i ) {
        GMRESDBG("givens.val[%d] = %e\n", i, givens.val[i]);
      }

      //printf("%%======= FGMRES search %d fabs(givens.val[(%d+1)]) = %.16e =======\n", search, search, fabs(givens.val[(search+1)]));
#if 1 //ceb
      if ( gmres_par->user_csrtrsv_choice == 0 ) {
        printf("FGMRES_Householders_mkltrsv_search(%d) = %.16e;\n", search+1, fabs(givens.val[(search+1)]));
      }
      else {
        printf("FGMRES_Householders_partrsv_search(%d) = %.16e;\n", search+1, fabs(givens.val[(search+1)]));
      }
#endif

      // update the solution
      // solve the least squares problem
      if ( fabs(givens.val[(search+1)]) < rtol  || (search == (search_max-1)) ) {
        GMRESDBG(" !!!!!!! update the solution %d!!!!!!!\n",0);

#if 0//USE_CUDA
        //need to invert krylov_val vector to do daxpy 
        //
#else
        for ( int i = 0; i <= search; ++i ) {
          alpha.val[i] = givens.val[i]/krylov.val[idx(i,i+1,krylov.ld)];
        }
#endif

        for ( int j = search; j > 0; --j ) {
          for (int i = j-1; i >= 0; --i ) {
            alpha.val[i] = alpha.val[i]
             - krylov.val[idx(i,j+1,krylov.ld)]*alpha.val[j]/krylov.val[idx(i,i+1,krylov.ld)];
          }
        }

        for ( int i=0; i<n; ++i ) {
          GMRESDBG("alpha.val[%d] = %e\n", i, alpha.val[i]);
        }

        // use preconditioned vectors to form the update (GEMV)
#if 1//USE_CUDA

        cublasCheck(cublasSetVector(search1,sizeof(dataType),alpha.val,1,d_alpha_val,1));        
        cublasCheck(cublasSetVector(n*search1,sizeof(dataType),Minvvj.val,1,d_Minvvj_val,1));
#endif

#if 1//USE_CUDA

#if 0
        d_scalar=1.0;
	//z[i] = z[i] + Minvvj[i,j]*alpha[j]
        cublasCheck(cublasDgemv( handle, CUBLAS_OP_N,
				 n, Minvvj.ld, 
				 &d_scalar, d_Minvvj_val, n,
				 d_alpha_val, 1, 
				 &d_scalar, d_zval, 1),"cublasDgemv");
#else
        // use preconditioned vectors to form the update (GEMV)
        for (int j = 0; j <= search; j++ ) {
	  //d_scalar=d_alpha_val[j];
          cublasCheck(cublasGetVector(1,sizeof(dataType),&d_alpha_val[j],1,&d_scalar,1));        

	//d_scalar=alpha.val[j];
	  cublasCheck(cublasDaxpy(handle, n, &d_scalar, &d_Minvvj_val[0+j*Minvvj.ld],1,d_zval,1));
          //cublasCheck(cublasSetVector(1,sizeof(dataType),&alpha.val[j],1,&d_alpha_val[j],1),"cublasSetVector ln 523");        
  	  //cublasCheck(cublasDaxpy(handle, n, &d_alpha_val[j], &d_Minvvj_val[0+j*Minvvj.ld],1,d_zval,1),"cublasDaxpy");

          //for (int i = 0; i < n; i++ ) {
	  //z.val[i] = z.val[i] + Minvvj.val[idx(i,j,Minvvj.ld)]*alpha.val[j];
          //}
        }
#endif


#else

        // use preconditioned vectors to form the update (GEMV)
        for (int j = 0; j <= search; j++ ) {
          for (int i = 0; i < n; i++ ) {
            //z.val[i] = z.val[i] + krylov.val[idx(i,j,krylov.ld)]*alpha.val[j];
            z.val[i] = z.val[i] + Minvvj.val[idx(i,j,Minvvj.ld)]*alpha.val[j];
          }
        }
    cublasCheck(cublasSetVector(n,sizeof(dataType),z.val,1,d_zval,1));

#endif

#if 1//USE_CUDA
	//x = x+z
        d_scalar=1.0;
        cublasCheck(cublasDaxpy(handle,n,&d_scalar,d_zval,1,d_xval,1));
        //cublasCheck(cublasGetVector(n,sizeof(dataType),d_xval,1,x.val,1),"cublasGetVector ln 601");

#else
        for (int i = 0; i < n; i++ ) {
          x.val[i] = x.val[i] + z.val[i];
        }
#endif

        gmres_log->search_directions = search+1;
        dataType wend = omp_get_wtime();
        gmres_log->solve_time = (wend-wstart);
        gmres_log->final_residual = fabs(givens.val[(search+1)]);

        break;
      }
    }
#if 1//USE_CUDA
    //retrieve x result to host
    cublasCheck(cublasGetVector(n,sizeof(dataType),d_xval,1,x.val,1));
#else

#endif

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
    data_zmfree( &precondq );
 
    cudaCheck(cudaFree(d_uval));
    cudaCheck(cudaFree(d_qval));
    cudaCheck(cudaFree(d_rval));
    cudaCheck(cudaFree(d_xval));
    cudaCheck(cudaFree(d_yval));
    cudaCheck(cudaFree(d_zval));
    cudaCheck(cudaFree(d_krylov_val));
    cudaCheck(cudaFree(d_alpha_val)); 
    cudaCheck(cudaFree(d_Minvvj_val));
    cudaCheck(cudaFree(d_precondq_val));

    cublasCheck(cublasDestroy(handle));
    cublasCheck(cublasShutdown());

    return info;
}
