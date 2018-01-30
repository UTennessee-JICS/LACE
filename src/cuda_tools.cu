/*
    -- LACE (version 0.0) --
       Univ. of Tennessee, Knoxville

       @author Chad Burdyshaw
*/
#include "../include/sparse.h"
#include "../include/cuda_tools.h"

//#include "cublas.h"
//#include "cublas_v2.h"
//#include "cusparse.h"
//#include "cusparse_v2.h"

#if 0 //defined in cublas.h
const char* cudaGetErrorString(cudaError_t status)
{
  char* status_string;
  if(status != cudaSuccess)
  {
    switch (status)
    {
        case cudaErrorInvalidValue:
            status_string = "cudaErrorInvalidValue";
            break;
        case cudaErrorMemoryAllocation:
            status_string = "cudaErrorMemoryAllocation";
            break;
        case cudaErrorInvalidDevicePointer:
            status_string = "cudaErrorInvalidDevicePointer";
            break;
        default:
            status_string = "<unknown>";
    }
  }
  return status_string;
}
#endif

const char* cublasGetErrorString(cublasStatus_t status)
{
  char* status_string;
  if(status != CUBLAS_STATUS_SUCCESS)
  {
    switch (status)
    {
        case CUBLAS_STATUS_SUCCESS:
            status_string = "CUBLAS_STATUS_SUCCESS";
            break;
        case CUBLAS_STATUS_NOT_INITIALIZED:
            status_string = "CUBLAS_STATUS_NOT_INITIALIZED";
            break;
        case CUBLAS_STATUS_ALLOC_FAILED:
            status_string = "CUBLAS_STATUS_ALLOC_FAILED";
            break;
        case CUBLAS_STATUS_INVALID_VALUE:
            status_string = "CUBLAS_STATUS_INVALID_VALUE";
            break;
        case CUBLAS_STATUS_ARCH_MISMATCH:
            status_string = "CUBLAS_STATUS_ARCH_MISMATCH";
            break;
        case CUBLAS_STATUS_MAPPING_ERROR:
            status_string = "CUBLAS_STATUS_MAPPING_ERROR";
            break;
        case CUBLAS_STATUS_EXECUTION_FAILED:
            status_string = "CUBLAS_STATUS_EXECUTION_FAILED";
            break;
        case CUBLAS_STATUS_INTERNAL_ERROR:
            status_string = "CUBLAS_STATUS_INTERNAL_ERROR";
            break;
        default:
            status_string = "<unknown>";
    }
  }
  return status_string;
}


const char* cusparseGetErrorString(cusparseStatus_t status)
{
  char* status_string;
  if(status != CUSPARSE_STATUS_SUCCESS)
  {
    switch (status)
    {
        case CUSPARSE_STATUS_SUCCESS:
            status_string = "CUSPARSE_STATUS_SUCCESS";
            break;
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            status_string = "CUSPARSE_STATUS_NOT_INITIALIZED";
            break;
        case CUSPARSE_STATUS_ALLOC_FAILED:
            status_string = "CUSPARSE_STATUS_ALLOC_FAILED";
            break;
        case CUSPARSE_STATUS_INVALID_VALUE:
            status_string = "CUSPARSE_STATUS_INVALID_VALUE";
            break;
        case CUSPARSE_STATUS_ARCH_MISMATCH:
            status_string = "CUSPARSE_STATUS_ARCH_MISMATCH";
            break;
        case CUSPARSE_STATUS_MAPPING_ERROR:
            status_string = "CUSPARSE_STATUS_MAPPING_ERROR";
            break;
        case CUSPARSE_STATUS_EXECUTION_FAILED:
            status_string = "CUSPARSE_STATUS_EXECUTION_FAILED";
            break;
        case CUSPARSE_STATUS_INTERNAL_ERROR:
            status_string = "CUSPARSE_STATUS_INTERNAL_ERROR";
            break;
        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            status_string = "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
            break;
        default:
            status_string = "<unknown>";
    }
  }
  return status_string;
}


void cudaCheckCore(cudaError_t code, const char* file, int line) {
   if (code != cudaSuccess) {
      fprintf(stderr,"Cuda Error %d : %s %s %d\n", code, cudaGetErrorString(code), file, line);
      exit(code);
   }
}
void cublasCheckCore(cublasStatus_t code, const char* file, int line) {
   if (code != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr,"CuBlas Error %d : %s %s %d\n", code, cublasGetErrorString(code), file, line);
      exit(code);
   }
}

void cusparseCheckCore(cusparseStatus_t code, const char* file, int line) {
   if (code != CUSPARSE_STATUS_SUCCESS) {
      fprintf(stderr,"CuSparse Error %d : %s %s %d\n", code, cusparseGetErrorString(code), file, line);
      exit(code);
   }
}

