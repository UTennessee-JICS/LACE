/*
 *  -- LACE (version 0.0) --
 *     Univ. of Tennessee, Knoxville
 *
 *     @author Stephen Wood, Chad Burdyshaw
 *
 */
#include "../include/sparse.h"
#include <mkl.h>
#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>

// gpu version of parilu_v0_3
// current version assumes single block, nnz's are split amongst available threads in block
__global__ void
lu_kernel(dataType * A_val,
  int *              A_rowidx,
  int *              A_col,
  int                A_nnz,
  dataType *         L_val,
  int *              L_row,
  int *              L_col,
  dataType *         U_val,
  int *              U_row,
  int *              U_col,
  dataType *         global_step)
{
  extern __shared__ dataType local_step[];

  int i, j, k;
  int il, iu, jl, ju;
  dataType s, tmp;
  int tid       = threadIdx.x;
  int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

  local_step[tid] = 0.0;
  global_step[0]  = 0;
  dataType sp      = 0;
  int num_elements = A_nnz;
  int start_idx    = 0;
  int end_idx      = A_nnz;
  int nthreads     = blockDim.x; // need to include num_blocks multplier in next version
  if (num_elements < nthreads) nthreads = num_elements; int stride = num_elements / nthreads;

  // splits nnz's among available GPU threads
  {
    start_idx = globalIdx * stride;
    end_idx   = (globalIdx + 1) * stride;
    if (globalIdx + 1 == nthreads) end_idx = num_elements;
    for (k = start_idx; k < end_idx; k++) {
      i = A_rowidx[k];
      j = A_col[k];
      s = A_val[k];

      il = L_row[i];
      iu = U_row[j];

      while (il < L_row[i + 1] && iu < U_row[j + 1]) {
        sp = 0.0;
        jl = L_col[il];
        ju = U_col[iu];

        // avoid branching
        sp = ( jl == ju ) ? L_val[il] * U_val[iu] : sp;
        s  = ( jl == ju ) ? s - sp : s;
        il = ( jl <= ju ) ? il + 1 : il;
        iu = ( jl >= ju ) ? iu + 1 : iu;
      }
      // undo the last operation (it must be the last)
      s += sp;

      if (i > j) { // modify l entry
        tmp = s / U_val[U_row[j + 1] - 1];
        local_step[tid] += pow(L_val[il - 1] - tmp, 2);
        L_val[il - 1]    = tmp;
      } else { // modify u entry
        tmp = s;
        local_step[tid] += pow(U_val[iu - 1] - tmp, 2);
        U_val[iu - 1]    = tmp;
      }
    }
  }

  // need to synch and reduce step accumulation
  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    // printf("stride=%d\n",stride);
    if (tid < stride) {
      local_step[tid] += local_step[tid + stride];
    }
    // __syncthreads();//may not need to sync here. Does not change result.
  }

  if (tid == 0) {
    // if odd block size we need to collect data from tid=blockDim.x
    if (blockDim.x % 2 > 0) {
      local_step[tid] += local_step[blockDim.x - 1];
    }
    global_step[0] = local_step[tid];
    // printf("tid=%d kernel global_step=%e\n",tid,global_step[0]);
  }
} // lu_kernel

extern "C"
void
data_PariLU_v0_3_gpu(data_d_matrix * A,
  data_d_matrix *                    L,
  data_d_matrix *                    U,
  dataType                           reduction,
  data_d_preconditioner_log *        log,
  int                                nthreads)
{
  // Separate the lower and upper elements into L and U respectively.
  L->diagorder_type = Magma_UNITY;
  data_zmconvert(*A, L, Magma_CSR, Magma_CSRL);

  U->diagorder_type = Magma_VALUE;
  data_zmconvert(*A, U, Magma_CSR, Magma_CSCU);

  dataType Ares       = 0.0;
  dataType Anonlinres = 0.0;
  data_d_matrix LU    = { Magma_CSR };
  data_zmconvert(*A, &LU, Magma_CSR, Magma_CSR);
  data_zilures(*A, *L, *U, &LU, &Ares, &Anonlinres);
  PARILUDBG("GPU PariLUv0_3_csrilu0_init_res = %e\n", Ares);
  PARILUDBG("GPU PariLUv0_3_csrilu0_init_nonlinres = %e\n", Anonlinres);
  // printf("\nGPU PariLUv0_3_csrilu0_init_res = %e\n", Ares);
  // printf("GPU PariLUv0_3_csrilu0_init_nonlinres = %e\n", Anonlinres);

  // ParLU element wise

  int iter     = 0;
  dataType tol = reduction * Ares;
  PARILUDBG("GPU PariLU_v0_3_tol = %e\n", tol);
  // printf("GPU PariLU_v0_3_tol = %e\n", tol);
  int num_threads = 0;

  dataType step[1];
  step[0] = FLT_MAX;
  dataType sp         = 0.0;
  dataType Anorm      = 0.0;
  dataType recipAnorm = 0.0;

  data_zfrobenius(*A, &Anorm);

  PARILUDBG("GPU PariLUv0_3_Anorm = %e\n", Anorm);
  // printf("GPU PariLUv0_3_Anorm = %e\n\n", Anorm);
  // printf("A->nnz=%d\n",A->nnz);

  recipAnorm = 1.0 / Anorm;

  dataType wstart;
  wstart = 0;
  // wstart = omp_get_wtime();

  data_rowindex(A, &(A->rowidx) );

  dataType * dA_val;
  int * dA_rowidx;
  int * dA_col;
  dataType * dL_val;
  int * dL_row;
  int * dL_col;
  dataType * dU_val;
  int * dU_row;
  int * dU_col;
  dataType * d_step;
  dataType * d_sp;

  // create memory space on the GPU device
  cudaMalloc((void **) &dA_val, (A->nnz + 1) * sizeof(dataType));
  cudaMalloc((void **) &dA_rowidx, (A->nnz + 1) * sizeof(int));
  cudaMalloc((void **) &dA_col, (A->nnz + 1) * sizeof(int));
  cudaMalloc((void **) &dL_val, (L->nnz + 1) * sizeof(dataType));
  cudaMalloc((void **) &dL_row, (L->nnz + 1) * sizeof(int));
  cudaMalloc((void **) &dL_col, (L->nnz + 1) * sizeof(int));
  cudaMalloc((void **) &dU_val, (U->nnz + 1) * sizeof(dataType));
  cudaMalloc((void **) &dU_row, (U->nnz + 1) * sizeof(int));
  cudaMalloc((void **) &dU_col, (U->nnz + 1) * sizeof(int));
  cudaMalloc((void **) &d_step, sizeof(dataType));

  // copy data from host to device
  cudaMemcpy(dA_rowidx, A->rowidx, A->nnz * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dA_col, A->col, A->nnz * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dA_val, A->val, A->nnz * sizeof(dataType), cudaMemcpyHostToDevice);

  cudaMemcpy(dL_row, L->row, (L->nnz + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dL_col, L->col, (L->nnz + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dL_val, L->val, L->nnz * sizeof(dataType), cudaMemcpyHostToDevice);

  cudaMemcpy(dU_row, U->row, (U->nnz + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dU_col, U->col, (U->nnz + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dU_val, U->val, U->nnz * sizeof(dataType), cudaMemcpyHostToDevice);

  while (step[0] > tol) {
    // reset sweep residual
    step[0] = 0.0;

    dim3 threadsPerBlock(nthreads);
    dim3 numBlocks(1);
    // call lu_kernel and send variables numBlocks, threadsPerBlock, and size of array for d_step residual accumulator to kernel
    lu_kernel << < numBlocks, threadsPerBlock, nthreads * sizeof(double) >> > (dA_val, dA_rowidx, dA_col, A->nnz,
    dL_val, dL_row, dL_col,
    dU_val, dU_row, dU_col,
    d_step);
    // copy d_step residual accumulator back to host
    cudaMemcpy(step, d_step, sizeof(dataType), cudaMemcpyDeviceToHost);

    step[0] *= recipAnorm;

    iter++;
    PARILUDBG("%% GPU PariLUv0_3_iteration = %d step = %e\n", iter, step[0]);
  }
  // copy L and U from device to host
  cudaMemcpy(L->val, dL_val, L->nnz * sizeof(dataType), cudaMemcpyDeviceToHost);
  cudaMemcpy(U->val, dU_val, U->nnz * sizeof(dataType), cudaMemcpyDeviceToHost);

  dataType ompwtime;
  dataType wend;
  wend     = 0;
  ompwtime = 0;

  // wend= omp_get_wtime();
  // ompwtime= (dataType) (wend-wstart)/((dataType) iter);

  PARILUDBG(
    "%% GPU PariLU v0.3 used %d OpenMP threads and required %d iterations, %f wall clock seconds, and an average of %f wall clock seconds per iteration as measured by omp_get_wtime()\n", num_threads, iter, wend - wstart,
    ompwtime);
  PARILUDBG("GPU PariLUv0_3_OpenMP = %d \nPariLUv0_3_iter = %d \nPariLUv0_3_wall = %e \nPariLUv0_3_avgWall = %e \n",
    num_threads, iter, wend - wstart, ompwtime);
  // printf("%% GPU PariLU v0.3 used %d OpenMP threads and required %d iterations, %f wall clock seconds, and an average of %f wall clock seconds per iteration as measured by omp_get_wtime()\n", num_threads, iter, wend-wstart, ompwtime );
  // printf("GPU PariLUv0_3_OpenMP = %d \nPariLUv0_3_iter = %d \nPariLUv0_3_wall = %e \nPariLUv0_3_avgWall = %e \n\n", num_threads, iter, wend-wstart, ompwtime );

  log->sweeps      = iter;
  log->tol         = tol;
  log->A_Frobenius = Anorm;
  log->precond_generation_time    = wend - wstart;
  log->initial_residual           = Ares;
  log->initial_nonlinear_residual = Anonlinres;
  log->omp_num_threads = num_threads;
} // data_PariLU_v0_3_gpu
