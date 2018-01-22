/*
 *
 *
 */


#include <stdio.h>
#include "../include/sparse.h"

/**
 *  Purpose
 *  -------
 *  Extracts the major diagonal from a matrix
 *
 *  Arguments
 *  ---------
 *
 *  @param[in]
 *  A           data_d_matrix
 *              sparse matrix A
 *
 *  @param[out]
 *  B           data_d_matrix*
 *              copy of A in new format
 ********************************************************************/

extern "C"
int
data_zmextractdiag(
  data_d_matrix   A,
  data_d_matrix * B)
{
  int info = 0;

  data_zmfree(B);
  // dataType one = dataType(1.0);
  // dataType zero = dataType(0.0);

  // B->val = NULL;
  // B->col = NULL;
  // B->row = NULL;
  // B->rowidx = NULL;
  // B->list = NULL;
  // B->blockinfo = NULL;
  // B->diag = NULL;

  // int rowlimit = A.num_rows;
  // int collimit = A.num_cols;
  // if (A.pad_rows > 0 && A.pad_cols > 0) {
  //   rowlimit = A.pad_rows;
  //   collimit = A.pad_cols;
  // }

  // CSR
  if (A.storage_type == Magma_CSR ||
    A.storage_type == Magma_CSRL ||
    A.storage_type == Magma_CSRU)
  {
    // fill in information for B
    B->storage_type = Magma_CSR;
    B->major        = MagmaRowMajor;
    B->num_cols     = A.num_cols;
    B->max_nnz_row  = 1;
    B->diameter     = 1;

    int count = 0;
    for (int i = 0; i < A.num_rows; i++) {
      for (int j = A.row[i]; j < A.row[i + 1]; j++) {
        if (A.col[j] == i) {
          printf("row=%d, col=%d, val=%e \n", i, A.col[j], A.val[j]);
          count++;
        } else {
          count++;
        }
      }
    }

    B->nnz      = count;
    B->num_rows = count;
    B->true_nnz = B->nnz;
    // B->val = (dataType*) calloc( B->nnz, sizeof(dataType) );
    // B->row = (int*) calloc( (B->nnz+1), sizeof(int) );
    // B->col = (int*) calloc( B->nnz, sizeof(int) );
    LACE_CALLOC(B->val, B->nnz);
    LACE_CALLOC ( B->row, ( (B->nnz + 1) );
    LACE_CALLOC(B->col, B->nnz);
    count = 0;
    for (int i = 0; i < A.num_rows; i++) {
      for (int j = A.row[i]; j < A.row[i + 1]; j++) {
        if (A.col[j] == i) {
          printf("row=%d, col=%d, val=%e \n", i, A.col[j], A.val[j]);
          B->col[count] = A.col[j];
          B->val[count] = A.val[j];
          B->row[count] = count;
          count++;
        } else {
          printf("zero on diagonal row=%d, col=%d, val=%e \n", i, i, 0.0);
          B->col[count] = i;
          B->val[count] = 0.0;
          B->row[count] = count;
          count++;
        }
      }
    }
    B->row[count] = count;
    } else if (A.storage_type == Magma_BCSR ||
    A.storage_type == Magma_BCSRL ||
    A.storage_type == Magma_BCSRU) // BCSR
    {
      // fill in information for B
      B->storage_type = Magma_BCSR;
      B->major = MagmaRowMajor;
      B->num_cols = A.num_cols;
      B->max_nnz_row = 1;
      B->diameter = 1;

      B->blocksize = A.blocksize;
      B->ldblock = A.ldblock;
      // printf("%s %d B->ldblock=%d\n", __FILE__, __LINE__, B->ldblock);
      B->numblocks = -1;

      int count = 0;
      for (int i = 0; i < A.num_rows; i++) {
        for (int j = A.row[i]; j < A.row[i + 1]; j++) {
          if (A.col[j] == i) {
 // printf("row=%d, col=%d, val=%e \n", i, A.col[j], A.val[j]);
            count++;
          }
        }
      }

      B->nnz = count * B->ldblock;
      B->num_rows = count;
      B->true_nnz = B->nnz;
      B->numblocks = count;
      // B->val = (dataType*) calloc( B->nnz, sizeof(dataType) );
      // B->row = (int*) calloc( (B->num_rows+1), sizeof(int) );
      // B->col = (int*) calloc( B->numblocks, sizeof(int) );
      LACE_CALLOC(B->val, B->nnz);
      LACE_CALLOC(B->row, (B->num_rows + 1) );
      LACE_CALLOC(B->col, B->numblocks);
      count = 0;
      for (int i = 0; i < A.num_rows; i++) {
        for (int j = A.row[i]; j < A.row[i + 1]; j++) {
          if (A.col[j] == i) {
 // printf("+row=%d, col=%d, val=%e \n", i, A.col[j], A.val[j]);
            B->col[count] = A.col[j];
            // B->val[count] = A.val[j];
            for (int k = 0; k < B->ldblock; k++) {
              B->val[count * B->ldblock + k] = A.val[j * A.ldblock + k];
 // printf("%e ", A.val[j*A.ldblock+k]);
 // if ((k+1)%A.blocksize==0)
 // printf("\n");
            }
            B->row[count] = count;
            count++;
          }
        }
      }
      B->row[count] = count;
    }

    return info;
    }
