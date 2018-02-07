//#define DEBUG_GMRES
//ceb fgmres_householder_gpu_v_0_3.cpp
//    This port to GPU is a conversion of a CPU algorithm using CUBLAS and CUSPARSE routines
//    and custom kernels to reduce memory transfer as much as possible.


#include "../include/sparse.h"
#include "../include/cuda_tools.h"
#include <mkl.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include "cublas.h"
#include "cublas_v2.h"
#include "cusparse.h"
#include "cusparse_v2.h"

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



__global__ void assign_krylov(int n, int search, int search1,
                              dataType* krylov_val, int krylov_ld,
                              dataType* A_val, int* A_row, int* A_col,
                              dataType* Minvvj_val, int Minvvj_ld)
{
  int i,j;
  int tid = threadIdx.x;
  int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int nthreads= blockDim.x; //need to include num_blocks multplier in next version
  if (n<nthreads){nthreads=n;}
  int stride =  (n + nthreads/2)/nthreads;
  int start_idx=globalIdx*stride;
  int end_idx=(globalIdx+1)*stride;
  if (end_idx>n) end_idx=n;
  
 
//printf("globalIdx=%d, stride=%d start_idx=%d, end_idx=%d\n",globalIdx, stride, start_idx, end_idx);

    if(start_idx<n){
    //for ( i=0; i<n; i++ ) {
    for ( i=start_idx; i<end_idx; i++ ) {
      for ( j=A_row[i]; j<A_row[i+1]; j++ ) {
        krylov_val[i+search1*krylov_ld] = krylov_val[i+search1*krylov_ld]
          + A_val[j]*Minvvj_val[A_col[j]+search*Minvvj_ld];
      }
    }
  }
  __syncthreads();
}


__global__ void assign_alpha(int n, dataType* alpha_val, dataType* givens_val, dataType* tmp_alpha_val,
                             dataType* krylov_val, int krylov_ld)
{
  int i,j;
  int tid = threadIdx.x;
  int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int nthreads= blockDim.x; //need to include num_blocks multplier in next version
  if (n<nthreads){nthreads=n;}
  int stride = stride =  (n + nthreads/2)/nthreads;
  int start_idx=globalIdx*stride;
  int end_idx=(globalIdx+1)*stride;
  if (end_idx>n) end_idx=n;
  //if(globalIdx==0){printf("search=%d nthreads=%d strid=%d\n",search, blockDim.x,stride);}
  //printf("globalIdx=%d, start_idx=%d, end_idx=%d\n",globalIdx,start_idx,end_idx);
 
  if(start_idx<n){
    for ( int i = start_idx; i <= end_idx; ++i ) {
      alpha_val[i] = givens_val[i]/krylov_val[i+(i+1)*krylov_ld];
      //use givens val vector as a temp alpha val to allow parallelization in next loop
      tmp_alpha_val[i]=alpha_val[i];
    }
  }
  __syncthreads();

  //cublasCheck(cublasDcopy(handle,search_max,d_alpha_val,1,d_tmp_val,1));  
  //__syncthreads();
 
  //if(start_idx<n){
  if(globalIdx==0){
    //ceb This loop has dependence on alpha_val[j] so cannot be simply parallelized
    for ( int j = n; j > 0; --j ) {
    //for ( int j = end_idx; j > start_idx; --j ) {
      for (int i = j-1; i >= 0; --i ) {
        alpha_val[i] = alpha_val[i]
          - krylov_val[i+(j+1)*krylov_ld]*alpha_val[j]/krylov_val[i+(i+1)*krylov_ld];
          //- krylov_val[i+(j+1)*krylov_ld]*tmp_alpha_val[j]/krylov_val[i+(i+1)*krylov_ld];
      }
    }
  }
  __syncthreads();
}


extern "C"
int
data_fgmres_householder_gpu_v_0_3(
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
    dataType* d_bval;
    dataType* d_xval;
    dataType* d_yval;
    dataType* d_zval;
    dataType* d_krylov_val;
    dataType* d_alpha_val;
    dataType* d_givens_val;
    dataType* d_tmp_val;
    dataType* d_A_val;
    int* d_A_row;
    int* d_A_col;
    dataType* d_LU_val;
    int* d_LU_row;
    int* d_LU_col;
    dataType* d_Minvvj_val;
    dataType* d_precondq_val;
    dataType d_scalar;
    dataType snrm2;
    cublasCheck(cublasCreate(&handle));
    cublasCheck(cublasInit());

    int structural_zero;
    int numerical_zero;
    cusparseStatus_t cusparse_status;
    cusparseHandle_t cusparse_handle;
    cusparseCheck(cusparseCreate(&cusparse_handle));
    const cusparseOperation_t trans = CUSPARSE_OPERATION_NON_TRANSPOSE;

    cudaCheck(cudaMalloc((void**)&d_uval,(n)*sizeof(dataType)));
    cudaCheck(cudaMalloc((void**)&d_qval,(n)*sizeof(dataType)));
    cudaCheck(cudaMalloc((void**)&d_rval,(n)*sizeof(dataType)));
    cudaCheck(cudaMalloc((void**)&d_bval,(n)*sizeof(dataType)));
    cudaCheck(cudaMalloc((void**)&d_xval,(n)*sizeof(dataType)));
    cudaCheck(cudaMalloc((void**)&d_yval,(n)*sizeof(dataType)));
    cudaCheck(cudaMalloc((void**)&d_zval,(n)*sizeof(dataType)));
    cudaCheck(cudaMalloc((void**)&d_A_val,(A->nnz)*sizeof(dataType)));
    cudaCheck(cudaMalloc((void**)&d_A_row,(A->num_rows+1)*sizeof(int)));
    cudaCheck(cudaMalloc((void**)&d_A_col,(A->nnz)*sizeof(int)));
    cudaCheck(cudaMalloc((void**)&d_krylov_val,(n*(search_max+1))*sizeof(dataType)));
    cudaCheck(cudaMalloc((void**)&d_alpha_val,(search_max)*sizeof(dataType)));
    cudaCheck(cudaMalloc((void**)&d_givens_val,(search_max)*sizeof(dataType)));
    cudaCheck(cudaMalloc((void**)&d_tmp_val,(n)*sizeof(dataType)));    
    cudaCheck(cudaMalloc((void**)&d_Minvvj_val,(n*(search_max))*sizeof(dataType)));   
    cudaCheck(cudaMalloc((void**)&d_precondq_val,(n*(search_max+1))*sizeof(dataType))); 

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

    cudaCheck(cudaMalloc((void**)&d_LU_val,(LU.nnz)*sizeof(dataType)));
    cudaCheck(cudaMalloc((void**)&d_LU_row,(LU.num_rows+1)*sizeof(int)));
    cudaCheck(cudaMalloc((void**)&d_LU_col,(LU.nnz)*sizeof(int)));

    cublasCheck(cublasSetVector(LU.num_rows+1,sizeof(int),LU.row,1,d_LU_row,1));
    cublasCheck(cublasSetVector(LU.nnz,sizeof(int),LU.col,1,d_LU_col,1));
    cublasCheck(cublasSetVector(LU.nnz,sizeof(dataType),LU.val,1,d_LU_val,1));

    data_d_matrix tmp={Magma_DENSE};
    data_zvinit( &tmp, n, 1, zero );
    cublasCheck(cublasDscal(handle,n,&zero,d_tmp_val,1));

    // store preconditioned vectors for flexibility
    cublasCheck(cublasDscal(handle,n*search_max,&zero,d_Minvvj_val,1));
    int Minvvj_ld = n;

    // Partrsv
    dataType ptrsv_tol = 1.0; //1.0e-10;
    int ptrsv_iter = 0;

    // alocate solution and residual vectors
    data_d_matrix x={Magma_DENSE};
    data_zmconvert( *x0, &x, Magma_DENSE, Magma_DENSE );
    cublasCheck(cublasSetVector(n,sizeof(dataType),x.val,1,d_xval,1));

    cublasCheck(cublasDscal(handle,n,&zero,d_rval,1));

    cublasCheck(cublasDscal(handle,n,&zero,d_qval,1));

    // Krylov subspace
    cublasCheck(cublasDscal(handle,n*(search_max+1),&zero,d_krylov_val,1));
    int krylov_ld = n;

    cublasCheck(cublasDscal(handle,n,&zero,d_zval,1));

    int precondq_ld = n;
    cublasCheck(cublasDscal(handle,n*(search_max+1),&zero,d_precondq_val,1));

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
    cublasCheck(cublasDscal(handle,n,&zero,d_uval,1));

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
    cublasCheck(cublasDscal(handle,search_max,&zero,d_alpha_val,1));



#if 1
    //if we are not changing LU, we should be able to move these prepatory operations outside of the iteration loop

    const cusparseOperation_t transLU = CUSPARSE_OPERATION_NON_TRANSPOSE;//no transpose
    cusparseSolvePolicy_t policy = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    cusparseMatDescr_t descrL = 0;
    cusparseCheck(cusparseCreateMatDescr(&descrL));
    csrsv2Info_t infoL=0;
    cusparseCreateCsrsv2Info(&infoL);

    cusparseSetMatDiagType(descrL, CUSPARSE_DIAG_TYPE_UNIT);
    cusparseSetMatFillMode(descrL, CUSPARSE_FILL_MODE_LOWER);
    int pBufferSize;
    void *pBuffer = 0;

    cusparseCheck(cusparseDcsrsv2_bufferSize(cusparse_handle, transLU,
                                             LU.num_rows, LU.nnz, descrL,
                                             d_LU_val, d_LU_row, d_LU_col,
					     infoL, &pBufferSize));
     cudaMalloc((void**)&pBuffer, pBufferSize);
     // perform analysis
     cusparseCheck(cusparseDcsrsv2_analysis(cusparse_handle, transLU,
				            LU.num_rows, LU.nnz, descrL,
					    d_LU_val, d_LU_row, d_LU_col,
					    infoL, policy, pBuffer));
     // L has unit diagonal, so no structural zero is reported.
     cusparseCheck(cusparseXcsrsv2_zeroPivot(cusparse_handle, infoL, &structural_zero));
     if (CUSPARSE_STATUS_ZERO_PIVOT == cusparse_status)
        { printf("L(%d,%d) is missing\n", structural_zero, structural_zero); }
     //for u solve
     csrsv2Info_t infoU=0;
     cusparseCheck(cusparseCreateCsrsv2Info(&infoU));
     cusparseMatDescr_t descrU = 0;
     cusparseCheck(cusparseCreateMatDescr(&descrU));
     cusparseSetMatDiagType(descrU, CUSPARSE_DIAG_TYPE_NON_UNIT);
     cusparseSetMatFillMode(descrU, CUSPARSE_FILL_MODE_UPPER);
     cusparseCheck(cusparseDcsrsv2_bufferSize(cusparse_handle, transLU,
					      LU.num_rows, LU.nnz, descrU,
					      d_LU_val, d_LU_row, d_LU_col,
                                              infoU, &pBufferSize));
     // perform analysis
    cusparseCheck(cusparseDcsrsv2_analysis(cusparse_handle, transLU,
                                           LU.num_rows, LU.nnz, descrU,
                                           d_LU_val, d_LU_row, d_LU_col,
                                           infoU, policy, pBuffer));
    cusparseCheck(cusparseXcsrsv2_zeroPivot(cusparse_handle, infoU, &structural_zero));
    if (CUSPARSE_STATUS_ZERO_PIVOT == cusparse_status)
       { printf("U(%d,%d) is missing\n", structural_zero, structural_zero); }

 
#endif

    // initial residual
    cusparseMatDescr_t descrA = 0;
    cusparseCheck(cusparseCreateMatDescr(&descrA));

    cublasCheck(cublasSetVector(A->num_rows+1,sizeof(int),A->row,1,d_A_row,1));
    cublasCheck(cublasSetVector(A->nnz,sizeof(int),A->col,1,d_A_col,1));
    cublasCheck(cublasSetVector(A->nnz,sizeof(dataType),A->val,1,d_A_val,1));

    //Ax=r
    cusparseCheck(cusparseDcsrmv(cusparse_handle, trans,
                                 A->num_rows, A->num_cols, A->nnz, &one, descrA,
                                 d_A_val, d_A_row, d_A_col,
                                 d_xval, &one, d_rval));
    cublasCheck(cublasSetVector(n,sizeof(dataType), b->val,1, d_bval,1));
    cublasCheck(cublasDaxpy(handle, n, &one, d_bval,1, d_rval,1));
    cublasCheck(cublasDnrm2(handle, n, d_rval, 1, &rnorm2));

    printf("rnorm2 = %e; tol = %e; rtol = %e;\n", rnorm2, rtol, rtol*rnorm2 );
    if ( gmres_par->tol_type == 1 ) {
      rtol = rtol*rnorm2;
    }
    if (rnorm2 < rtol ) {
      info = 0;
      return info;
    }

    // fill first column of Kylov subspace for Arnoldi iteration
    cublasCheck(cublasDcopy(handle,n,d_rval,1,d_krylov_val,1));

    //it is inefficient to copy this back to the host to perform the following two operations
    cublasCheck(cublasGetVector(1,sizeof(dataType), &d_krylov_val[idx(0,0,krylov_ld)],1, &d_scalar,1));

    //get sign mysgn(T v) {return T(v >= T(0)) - T(v < T(0));}
    dataType dd = mysgn(d_scalar)*rnorm2;
    d_scalar = d_scalar + dd;
    //copy back to coprocessor
    cublasCheck(cublasSetVector(1,sizeof(dataType), &d_scalar, 1, &d_krylov_val[0], 1));

    dataType k1norm2;
    cublasCheck(cublasDnrm2(handle, n, d_krylov_val, 1, &k1norm2));
    d_scalar = 1.0/k1norm2;
    cublasCheck(cublasDscal(handle,n,&d_scalar, d_krylov_val,1));


    //q=-r/dd
    d_scalar=-1.0/dd;
    cublasCheck(cublasDcopy(handle,n,d_rval,1,d_qval,1));
    cublasCheck(cublasDscal(handle,n,&d_scalar, d_qval,1));
    //precondq=q
    cublasCheck(cublasDcopy(handle,n,d_qval,1,d_precondq_val,1));


#if 0
    for ( int i=0; i<n; ++i ) {
      GMRESDBG("q.val[%d] = %e\n", i, q.val[i]);
      //GMRESDBG("q.val[%d] = %e\n", i, d_qval[i]);
    }
#endif

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
      cublasCheck(cublasDscal(handle,n,&zero,d_uval,1));

      data_zmfree( &tmp );
      data_zvinit( &tmp, n, 1, zero );
      cublasCheck(cublasDscal(handle,n,&zero,d_tmp_val,1));
#if 0
      for ( int i=0; i<krylov.ld; i++ ) {
        GMRESDBG("\tkrylov.val[idx(%d,%d,%d)] = %e\n", i, search, krylov.ld, krylov.val[idx(i,search,krylov.ld)]);
      }
#endif

      if ( gmres_par->user_csrtrsv_choice == 0 ) {
        // Apply preconditioner to krylov.val[idx(A->col[j],search,krylov.ld)]

        cusparseCheck(cusparseDcsrsv2_solve(cusparse_handle,
                              transLU,
                              LU.num_rows,
                              LU.nnz,
                              &one,
                              descrL,
                              d_LU_val,
                              d_LU_row,
                              d_LU_col,
                              infoL,
                              d_qval,
                              d_tmp_val,
                              policy,
                              pBuffer));

        cusparseCheck(cusparseDcsrsv2_solve(cusparse_handle,
                              transLU,
                              LU.num_rows,
                              LU.nnz,
                              &one,
                              descrU,
                              d_LU_val,
                              d_LU_row,
                              d_LU_col,
                              infoU,
                              d_tmp_val,
                              &d_Minvvj_val[idx(0,search,Minvvj_ld)],
                              policy,
                              pBuffer));

      }
      else {
        cusparseCheck(cusparseDcsrsv2_solve(cusparse_handle,
                              transLU,
                              LU.num_rows,
                              LU.nnz,
                              &one,
                              descrL,
                              d_LU_val,
                              d_LU_row,
                              d_LU_col,
                              infoL,
                              d_qval,
                              d_tmp_val,
                              policy,
                              pBuffer));

        cusparseCheck(cusparseDcsrsv2_solve(cusparse_handle,
                              transLU,
                              LU.num_rows,
                              LU.nnz,
                              &one,
                              descrU,
                              d_LU_val,
                              d_LU_row,
                              d_LU_col,
                              infoU,
                              d_tmp_val,
                              &d_Minvvj_val[idx(0,search,Minvvj_ld)],
                              policy,
                              pBuffer));

      }
#if 0
      for ( int i=0; i<Minvvj.ld; i++ ) {
        GMRESDBG("Minvvj.val[idx(%d,%d,%d)] = %e\n",
          i, search, Minvvj.ld, Minvvj.val[idx(i,search,Minvvj.ld)]);
      }
#endif

      dim3 threadsPerBlock(nthreads);
      dim3 numBlocks(1);
      assign_krylov<<<numBlocks,threadsPerBlock,nthreads*sizeof(double)>>>
        (n,search,search1,d_krylov_val,krylov_ld,d_A_val,d_A_row,d_A_col,d_Minvvj_val,Minvvj_ld);

      // Householder Transformations
      for ( int j=0; j <= search; ++j ) {
        dataType sum = 0.0;


        span = n-j;
        cublasCheck(cublasDdot(handle, span, &d_krylov_val[j+j*krylov_ld],1, 
		       &d_krylov_val[j+search1*krylov_ld],1,&sum));

        d_scalar = -2.0*sum;
        cublasCheck(cublasDaxpy(handle, span, &d_scalar, &d_krylov_val[j+j*krylov_ld],1, 
			&d_krylov_val[j+search1*krylov_ld],1));

      }

#if 0
      for ( int j=0; j <= search1; ++j ) {
        for ( int i=0; i<n; ++i ) {
          GMRESDBG("krylov.val[idx(%d,%d,%d)] = %e\n", i, j, krylov.ld, krylov.val[idx(i,j,krylov.ld)]);
        }
      }
#endif
      if ( search < n ) {

        dataType snrm2;
        span=n-search;
        cublasCheck(cublasDnrm2(handle, span, &d_krylov_val[search1+search1*krylov_ld], 1, &snrm2));
        cublasCheck(cublasGetVector(1,sizeof(dataType),&d_krylov_val[idx(search1,search1,krylov_ld)],1,
                                    &d_scalar,1));
        dd = mysgn(d_scalar)*snrm2;
        d_scalar=d_scalar+dd;
        cublasCheck(cublasSetVector(1,sizeof(dataType),&d_scalar,1, 
			&d_krylov_val[idx(search1,search1,krylov_ld)],1));



        cublasCheck(cublasDnrm2(handle, span, 
				&d_krylov_val[search1+search1*krylov_ld], 1, &snrm2));
        d_scalar=1.0/snrm2;
        cublasCheck(cublasDscal(handle, span, &d_scalar, 
		    &d_krylov_val[search1+search1*krylov_ld], 1));


        cublasCheck(cublasDscal(handle,n,&zero,d_qval,1));
	cublasCheck(cublasSetVector(1,sizeof(dataType), &one,1, &d_qval[search1],1));

        for (int j=search1; j>=0; --j) {
          dataType sum = 0.0;
          span=n-j;
          cublasCheck(cublasDdot(handle, span, &d_qval[j],1, &d_krylov_val[j+j*krylov_ld],1,&sum));
          d_scalar = -2.0*sum;
	  cublasCheck(cublasDaxpy(handle, span, &d_scalar, 
		      &d_krylov_val[j+j*krylov_ld],1, &d_qval[j],1));

        }

	cublasCheck(cublasDcopy(handle, n, d_qval,1,  &d_precondq_val[search1*precondq_ld],1));        


      }

#if 0
cublasCheck(cublasGetVector(n,sizeof(dataType), &d_precondq_val[search1*precondq.ld],1, &precondq.val[search1*precondq.ld],1));
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
      for( int j=0; j<search; ++j ) {
        dataType c=givens_cos.val[j];
        dataType s=givens_sin.val[j];
        int span=search;
        
        cublasCheck(cublasDrot(handle,1, 
			       &d_krylov_val[idx(j,search1,krylov_ld)],1,
                               &d_krylov_val[idx(j+1,search1,krylov_ld)],1,
			       &c, &s));
        
      }

#if 0
      for ( int j=0; j <= search1; ++j ) {
        for ( int i=0; i<n; ++i ) {
          GMRESDBG("krylov.val[idx(%d,%d,%d)] = %e\n", i, j, krylov.ld, krylov.val[idx(i,j,krylov.ld)]);
        }
      }
#endif

      if ( search < n ) {
        // form search-th rotation matrix
        dataType a;//=krylov.val[idx(search,search1,krylov.ld)];
        cublasCheck(cublasGetVector(1, sizeof(dataType), 
			&d_krylov_val[idx(search,search1,krylov_ld)],1,&a,1));
        dataType b=-dd;
        d_scalar = a;
        //a and b are modified by function, hence we are using temporary vars rather than
        //  passing in krylov_val
        cublasCheck(cublasDrotg(handle, &a, &b, &givens_cos.val[search], &givens_sin.val[search]));

        dataType tmp = givens_cos.val[search]*d_scalar - givens_sin.val[search]*dd;

        cublasSetVector(1, sizeof(dataType), &tmp,1, &d_krylov_val[idx(search,search1,krylov_ld)],1);

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

      cublasCheck(cublasSetVector(search_max,sizeof(dataType),givens.val,1,d_givens_val,1));

      dim3 threadsPerBlock(nthreads);
      dim3 numBlocks(1);
      assign_alpha<<<numBlocks,threadsPerBlock,nthreads*sizeof(double)>>>
		(search, d_alpha_val, d_givens_val, d_tmp_val, d_krylov_val, krylov_ld);

        // use preconditioned vectors to form the update (GEMV)
        d_scalar=1.0;
	//z[i] = z[i] + Minvvj[i,j]*alpha[j]
        cublasCheck(cublasDgemv( handle, CUBLAS_OP_N,
				 n, search_max, 
				 &d_scalar, d_Minvvj_val, Minvvj_ld,
				 d_alpha_val, 1, 
				 &d_scalar, d_zval, 1));

	//x = x+z
        d_scalar=1.0;
        cublasCheck(cublasDaxpy(handle,n,&d_scalar,d_zval,1,d_xval,1));

        gmres_log->search_directions = search+1;
        dataType wend = omp_get_wtime();
        gmres_log->solve_time = (wend-wstart);
        gmres_log->final_residual = fabs(givens.val[(search+1)]);

        break;
      }
    }

    //retrieve x result to host
    cublasCheck(cublasGetVector(n,sizeof(dataType),d_xval,1,x.val,1));

    data_zmconvert( x, x0, Magma_DENSE, Magma_DENSE );

    if (gmres_log->final_residual > rtol) {
      info = 0;
    }

    data_zmfree( &x );
    //data_zmfree( &r );
    //data_zmfree( &krylov );
    data_zmfree( &h );
    //data_zmfree( &u );
    data_zmfree( &givens );
    data_zmfree( &givens_cos );
    data_zmfree( &givens_sin );
    //data_zmfree( &alpha );
    //data_zmfree( &z );
    data_zmfree( &LU );
    //free( ia );
    //free( ja );
    data_zmfree( &tmp );
    //data_zmfree( &Minvvj );
    //data_zmfree( &precondq );
 
    cudaCheck(cudaFree(d_uval));
    cudaCheck(cudaFree(d_qval));
    cudaCheck(cudaFree(d_rval));
    cudaCheck(cudaFree(d_bval));
    cudaCheck(cudaFree(d_xval));
    cudaCheck(cudaFree(d_yval));
    cudaCheck(cudaFree(d_zval));
    cudaCheck(cudaFree(d_krylov_val));
    cudaCheck(cudaFree(d_alpha_val));
    cudaCheck(cudaFree(d_tmp_val));
    cudaCheck(cudaFree(d_A_val));
    cudaCheck(cudaFree(d_A_row));
    cudaCheck(cudaFree(d_A_col));
    cudaCheck(cudaFree(d_LU_val));
    cudaCheck(cudaFree(d_LU_row));
    cudaCheck(cudaFree(d_LU_col));
    cudaCheck(cudaFree(d_Minvvj_val));
    cudaCheck(cudaFree(d_precondq_val));
    cudaCheck(cudaFree(pBuffer));

    cublasCheck(cublasDestroy(handle));
    cublasCheck(cublasShutdown());

    return info;
}
