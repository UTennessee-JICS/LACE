/*


*/


#include <stdio.h>
#include "../include/sparse.h"

/**
    Purpose
    -------
    Extracts the major diagonal from a matrix
    
    Arguments
    ---------

    @param[in]
    A           data_d_matrix
                sparse matrix A

    @param[out]
    B           data_d_matrix*
                copy of A in new format
    ********************************************************************/

extern "C" 
int
data_zmextractdiag(
    data_d_matrix A,
    data_d_matrix *B )
{
    int info = 0;
    
    data_zmfree( B );
    //dataType one = dataType(1.0);
    //dataType zero = dataType(0.0);
    
    //B->val = NULL;
    //B->col = NULL;
    //B->row = NULL;
    //B->rowidx = NULL;
    //B->list = NULL;
    //B->blockinfo = NULL;
    //B->diag = NULL;
    
    //int rowlimit = A.num_rows;
    //int collimit = A.num_cols;
    //if (A.pad_rows > 0 && A.pad_cols > 0) {
    //   rowlimit = A.pad_rows;
    //   collimit = A.pad_cols;
    //}
    
    // CSR to anything
    if ( A.storage_type == Magma_CSR 
      || A.storage_type == Magma_CSRL 
      || A.storage_type == Magma_CSRU )
    {
      // fill in information for B
      B->storage_type = Magma_CSR;
      B->major = MagmaRowMajor;
      B->num_cols = 1;
      B->max_nnz_row = 1;
      B->diameter = 1;
    
      int count = 0;
      for(int i=0; i < A.num_rows; i++) {
        for(int j=A.row[i]; j < A.row[i+1]; j++) {
          if ( A.col[j] == i ) {
            printf("row=%d, col=%d, val=%e \n", i, A.col[j], A.val[j]);
            count++;
          } 
        }
      }
      
      B->nnz = count;
      B->num_rows = count;
      B->true_nnz = B->nnz;
      B->val = (dataType*) calloc( B->nnz, sizeof(dataType) );
      B->row = (int*) calloc( (B->nnz+1), sizeof(int) );
      B->col = (int*) calloc( B->nnz, sizeof(int) );
      count = 0;
      for(int i=0; i < A.num_rows; i++) {
        for(int j=A.row[i]; j < A.row[i+1]; j++) {
          if ( A.col[j] == i ) {
            printf("row=%d, col=%d, val=%e \n", i, A.col[j], A.val[j]);
            B->col[count] = A.col[j];
            B->val[count] = A.val[j];
            B->row[count] = count;
            count++;
          } 
        }
      }
      B->row[count] = count;
    }
    
    return info;
}    