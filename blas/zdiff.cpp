#include "../include/sparse.h"
#include <mkl.h>

extern "C"
int
data_zdiff_csr(
    data_d_matrix *A,
    data_d_matrix *B,
    data_d_matrix *C,
    dataType *res,
    dataType *nonlinres )
{
  dataType tmp, tmp2;
  int i,j,k;
  *res = 0.0;
  *nonlinres = 0.0;

  if ( A->storage_type != B->storage_type ) {
    data_d_matrix B_a = {A->storage_type};
    data_zmconvert( *B, &B_a, B->storage_type, A->storage_type );
    data_zmconvert( *A, C, Magma_CSR, Magma_CSR );
    for (i=0; i<A->num_rows; i++) {
        for (j=A->row[i]; j<A->row[i+1]; j++) {
            int localcol = A->col[j];
            for( k=B_a.row[i]; k<B_a.row[i+1]; k++){
                if(B_a.col[k] == localcol){
                    tmp2 = A->val[j] - B_a.val[k];
                    C->val[k] = tmp2;
                    (*nonlinres) = (*nonlinres) + tmp2 * tmp2;
                    break;
                }
            }
        }
    }
    for(i=0; i<C->num_rows; i++){
        for(j=C->row[i]; j<C->row[i+1]; j++){
            tmp = C->val[j];
            (*res) = (*res) + tmp * tmp;
        }
    }
    (*nonlinres) =  sqrt((*nonlinres));
    (*res) =  sqrt((*res));
  }
  else {
    data_zmconvert( *A, C, Magma_CSR, Magma_CSR );
    for (i=0; i<A->num_rows; i++) {
        for (j=A->row[i]; j<A->row[i+1]; j++) {
            int localcol = A->col[j];
            for( k=B->row[i]; k<B->row[i+1]; k++){
                if(B->col[k] == localcol){
                    tmp2 = A->val[j] - B->val[k];
                    C->val[k] = tmp2;
                    (*nonlinres) = (*nonlinres) + tmp2 * tmp2;
                    break;
                }
            }
        }
    }
    for(i=0; i<C->num_rows; i++){
        for(j=C->row[i]; j<C->row[i+1]; j++){
            tmp = C->val[j];
            (*res) = (*res) + tmp * tmp;
        }
    }
    (*nonlinres) =  sqrt((*nonlinres));
    (*res) =  sqrt((*res));
  }

  return DEV_SUCCESS;
}

int
data_zdiff_magnitude_csr(
    data_d_matrix *A,
    data_d_matrix *B,
    dataType *res)
{
  int i,j,k;
  *res = 0.0;

  if ( A->storage_type != B->storage_type ) {
    data_d_matrix C = {A->storage_type};
    data_zmconvert( *B, &C, B->storage_type, A->storage_type );
    for (i=0; i<A->num_rows; i++) {
        for (j=A->row[i]; j<A->row[i+1]; j++) {
            int localcol = A->col[j];
            for( k=C.row[i]; k<C.row[i+1]; k++){
                if(C.col[k] == localcol){
                    (*res) =  (*res) + (A->val[j] - C.val[k]);
                    break;
                }
            }
        }
    }
    data_zmfree( &C );
  }
  else {
    for (i=0; i<A->num_rows; i++) {
        for (j=A->row[i]; j<A->row[i+1]; j++) {
            int localcol = A->col[j];
            for( k=B->row[i]; k<B->row[i+1]; k++){
                if(B->col[k] == localcol){
                    (*res) =  (*res) + (A->val[j] - B->val[k]);
                    break;
                }
            }
        }
    }
  }

  return DEV_SUCCESS;
}

extern "C"
int
data_zsubtract_csr(
    data_d_matrix *A,
    data_d_matrix *B )
{
  int i,j,k;
  for (i=0; i<A->num_rows; i++) {
      for (j=A->row[i]; j<A->row[i+1]; j++) {
          int localcol = A->col[j];
          for( k=B->row[i]; k<B->row[i+1]; k++){
              if(B->col[k] == localcol){
                  A->val[j] = A->val[j] - B->val[k];
                  break;
              }
          }
      }
  }

  return DEV_SUCCESS;
}

extern "C"
int
data_zsubtract_guided_csr(
  data_d_matrix *A,
  data_d_matrix *B,
  data_d_matrix *C,
  dataType *step )
{
  int i,j,k,l;
  dataType tmp = 0.0;
  for (i=0; i<A->num_rows; i++) {
    for (j=A->row[i]; j<A->row[i+1]; j++) {
      int localcol = A->col[j];
      for( k=B->row[i]; k<B->row[i+1]; k++){
        if(B->col[k] == localcol){
          for( l=C->row[i]; l<C->row[i+1]; l++){
            if(C->col[l] == localcol){
              tmp = B->val[k] - C->val[l];
              (*step) = (*step) + pow( A->val[j] - tmp, 2 );
              //A->val[j] = B->val[k] - C->val[l];
              A->val[j] = tmp;
              break;
            }
          }
          break;
        }
      }
    }
  }

  return DEV_SUCCESS;
}

extern "C"
int
data_zdiagdivide_csr(
  data_d_matrix *A,
  data_d_matrix *B )
{
  int i,j,k;
  dataType div = 1.0;
  for (i=0; i<A->num_rows; i++) {
    for( k=B->row[i]; k<B->row[i+1]; k++){
      if(B->col[k] == i) { // localcol){
        //A->val[j] = A->val[j] / B->val[k];
        div = 1.0/ B->val[k];
        break;
      }
    }
    for (j=A->row[i]; j<A->row[i+1]; j++) {
      A->val[j] *= div;
      //int localcol = A->col[j];
      //for( k=B->row[i]; k<B->row[i+1]; k++){
      //  if(B->col[k] == i) { // localcol){
      //    A->val[j] = A->val[j] / B->val[k];
      //  }
      //}
    }
  }

  return DEV_SUCCESS;
}

extern "C"
int
data_zset_csr(
    data_d_matrix *A,
    data_d_matrix *B )
{
  int i,j,k;
  for (i=0; i<A->num_rows; i++) {
      for (j=A->row[i]; j<A->row[i+1]; j++) {
          int localcol = A->col[j];
          for( k=B->row[i]; k<B->row[i+1]; k++){
              if(B->col[k] == localcol){
                  A->val[j] = B->val[k];
              }
          }
      }
  }

  return DEV_SUCCESS;
}
