#ifndef CUDA_TOOLS_H
#define CUDA_TOOLS_H

#include "cublas.h"
#include "cublas_v2.h"
#include "cusparse.h"
#include "cusparse_v2.h"

//======================================
// GPU tools
//======================================

//included from cublas.h
//static const char* 
//cudaGetErrorString(cudaError_t status);

const char* 
cublasGetErrorString(cublasStatus_t status);

const char* 
cusparseGetErrorString(cusparseStatus_t status);

void 
cudaCheckCore(cudaError_t code, const char* file, int line);

void 
cublasCheckCore(cublasStatus_t code, const char* file, int line);

void 
cusparseCheckCore(cusparseStatus_t code, const char* file, int line);

#define cudaCheck( func )  cudaCheckCore( (func), __FILE__, __LINE__);
#define cublasCheck( func ) cublasCheckCore( (func), __FILE__, __LINE__);
#define cusparseCheck( func ) cusparseCheckCore( (func), __FILE__, __LINE__);

#endif /* CUDA_TOOLS_H */

