/*
    -- DEV (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
       @author Hartwig Anzt
       @author Stephen Wood
*/
#include <stdio.h>
#include <algorithm>
#include <vector>
#include <utility>  // pair
#include <cstring>  // memset
#include <cassert>  // for assert
#include "../include/sparse.h"
#include <mkl.h>


/**
    Purpose
    -------
    Returns true if first element of a is less than first element of b.
    Ignores second element. Used for sorting pairs,
    std::pair< int, dataType >, of column indices and values.
*/
static bool compare_first(
    const std::pair< int, dataType >& a,
    const std::pair< int, dataType >& b )
{
    return (a.first < b.first);
}


/**
    Purpose
    -------

    Helper function to compress CSR containing zero-entries.


    Arguments
    ---------

    @param[in]
    val         dataType**
                input val pointer to compress

    @param[in]
    row         int**
                input row pointer to modify

    @param[in]
    col         int**
                input col pointer to compress

    @param[in]
    valn        dataType**
                output val pointer

    @param[out]
    rown        int**
                output row pointer

    @param[out]
    coln        int**
                output col pointer

    @param[out]
    n           int*
                number of rows in matrix

    @param[in]
    queue       data_queue_t
                Queue to execute in.

    @ingroup datasparse_zaux
    ********************************************************************/

extern "C" int
data_z_csr_compressor(
    dataType ** val,
    int ** row,
    int ** col,
    dataType ** valn,
    int ** rown,
    int ** coln,
    int *n )
{
    int info = 0;
    
    int i,j, nnz_new=0, (*row_nnz)=NULL, nnz_this_row;
    //CHECK( data_index_malloc_cpu( &(row_nnz), (*n) ));
    //CHECK( data_index_malloc_cpu( rown, *n+1 ));
    row_nnz = (int*) calloc( (*n), sizeof(int) );
    *rown = (int*) calloc( (*n+1), sizeof(int) );
    for( i=0; i<*n; i++ ) {
        (*rown)[i] = nnz_new;
        nnz_this_row = 0;
        for( j=(*row)[i]; j<(*row)[i+1]; j++ ) {
            if ( DEV_D_REAL((*val)[j]) != 0 ) {
                nnz_new++;
                nnz_this_row++;
            }
        }
        row_nnz[i] = nnz_this_row;
    }
    (*rown)[*n] = nnz_new;

    //CHECK( data_zmalloc_cpu( valn, nnz_new ));
    //CHECK( data_index_malloc_cpu( coln, nnz_new ));
    *valn = (dataType*) calloc( nnz_new, sizeof(dataType) );
    *coln = (int*) calloc( nnz_new, sizeof(int) );
    nnz_new = 0;
    for( i=0; i<*n; i++ ) {
        for( j=(*row)[i]; j<(*row)[i+1]; j++ ) {
            if ( DEV_D_REAL((*val)[j]) != 0 ) {
                (*valn)[nnz_new]= (*val)[j];
                (*coln)[nnz_new]= (*col)[j];
                nnz_new++;
            }
        }
    }


//cleanup:
    if ( info != 0 ) {
        free( valn );
        free( coln );
        free( rown );
    }
    free( row_nnz );
    row_nnz = NULL;
    return info;
}

extern "C" 
int
data_rowindex(
    data_d_matrix *A,
    int **rowidx ) 
{
  int info = 0;
  (*rowidx) = (int*) calloc( A->nnz, sizeof(int) );
  int rowlimit = A->num_rows;
  if (A->pad_rows > 0 && A->pad_cols > 0) {
     rowlimit = A->pad_rows;
  }
  for(int i=0; i < rowlimit; i++) {
      for(int j=A->row[i]; j < A->row[i+1]; j++) {
          (*rowidx)[j] = i;
          DEV_PRINTF("%d ", i);
      }
      DEV_PRINTF("\n");
  }
  return info;
}

/**
    Purpose
    -------

    Converter between different sparse storage formats.

    Arguments
    ---------

    @param[in]
    A           data_d_matrix
                sparse matrix A

    @param[out]
    B           data_d_matrix*
                copy of A in new format

    @param[in]
    old_format  data_storage_t
                original storage format

    @param[in]
    new_format  data_storage_t
                new storage format

    @ingroup datasparse_zaux
    ********************************************************************/

extern "C" 
int
data_zmconvert(
    data_d_matrix A,
    data_d_matrix *B,
    data_storage_t old_format,
    data_storage_t new_format )
{
	
    int info = 0;
    //data_zmfree( B );
    //int *length=NULL;
    
    //data_d_matrix hA={Magma_CSR}, hB={Magma_CSR};
    //data_d_matrix A_d={Magma_CSR}, B_d={Magma_CSR};
    //int *row_tmp=NULL, *col_tmp=NULL;
    //dataType *val_tmp = NULL;
    //int *row_tmp2=NULL, *col_tmp2=NULL;
    //dataType *val_tmp2 = NULL;
    //dataType *transpose=NULL;
    //int *nnz_per_row=NULL;
    dataType one = dataType(1.0);
    dataType zero = dataType(0.0);
    
    B->val = NULL;
    B->col = NULL;
    B->row = NULL;
    B->rowidx = NULL;
    B->list = NULL;
    B->blockinfo = NULL;
    B->diag = NULL;
    
    int rowlimit = A.num_rows;
    int collimit = A.num_cols;
    if (A.pad_rows > 0 && A.pad_cols > 0) {
       rowlimit = A.pad_rows;
       collimit = A.pad_cols;
    }
      
    
    // CSR to anything
    if ( old_format == Magma_CSR 
      || old_format == Magma_CSRL 
      || old_format == Magma_CSRU )
    {
        // CSR to CSR
        if ( new_format == Magma_CSR ) {
            // fill in information for B
            B->storage_type = Magma_CSR;
            B->major = MagmaRowMajor;
            B->fill_mode = A.fill_mode;
            B->num_rows = A.num_rows; 
            B->num_cols = A.num_cols;
            B->pad_rows = A.pad_rows; 
            B->pad_cols = A.pad_cols;
            B->nnz = A.nnz;
            B->true_nnz = A.true_nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
    
            //CHECK( data_zmalloc_cpu( &B->val, A.nnz ));
            //CHECK( data_index_malloc_cpu( &B->row, rowlimit+1 ));
            //CHECK( data_index_malloc_cpu( &B->col, A.nnz ));
            B->val = (dataType*) calloc( A.nnz, sizeof(dataType) );
            B->row = (int*) calloc( (rowlimit+1), sizeof(int) );
            B->col = (int*) calloc( A.nnz, sizeof(int) );
            
            for( int i=0; i < A.nnz; i++) {
                B->val[i] = A.val[i];
                B->col[i] = A.col[i];
            }
            for( int i=0; i < rowlimit+1; i++) {
                B->row[i] = A.row[i];
            }
        }
        // CSR to CUCSR
        else if ( new_format == Magma_CUCSR ){
            //CHECK(data_zmconvert(A, B, Magma_CSR, Magma_CSR, queue));
            data_zmconvert(A, B, Magma_CSR, Magma_CSR );
            B->storage_type = Magma_CUCSR;
        }
        
        // CSR to CSRL
        else if ( new_format == Magma_CSRL ) {
            // fill in information for B
            B->storage_type = Magma_CSRL;
            B->major = MagmaRowMajor;
            B->fill_mode = MagmaLower;
            B->num_rows = A.num_rows; 
            B->num_cols = A.num_cols;
            B->pad_rows = A.pad_rows; 
            B->pad_cols = A.pad_cols;
            B->diameter = A.diameter;
    
            int numzeros=0;
            for( int i=0; i < rowlimit; i++) {
                for( int j=A.row[i]; j < A.row[i+1]; j++) {
                    if ( A.col[j] < i) {
                        numzeros++;
                    }
                    else if ( A.col[j] == i &&
                                    B->diagorder_type != Magma_NODIAG ) {
                        numzeros++;
                    }
                }
            }
            B->nnz = numzeros;
            B->true_nnz = numzeros;
            //CHECK( data_zmalloc_cpu( &B->val, numzeros ));
            //CHECK( data_index_malloc_cpu( &B->row, rowlimit+1 ));
            //CHECK( data_index_malloc_cpu( &B->col, numzeros ));
            B->val = (dataType*) calloc( B->nnz, sizeof(dataType) );
            B->row = (int*) calloc( (rowlimit+1), sizeof(int) );
            B->col = (int*) calloc( B->nnz, sizeof(int) );
            
            numzeros=0;
            for( int i=0; i < rowlimit; i++) {
                B->row[i]=numzeros;
                for( int j=A.row[i]; j < A.row[i+1]; j++) {
                    // diagonal omitted by default
                    if ( A.col[j] < i) {
                        B->val[numzeros] = A.val[j];
                        B->col[numzeros] = A.col[j];
                        numzeros++;
                    }
                    // add option of including diagonal with unit value
                    else if ( A.col[j] == i &&
                                    B->diagorder_type == Magma_UNITY) {
                        B->val[numzeros] = one;
                        B->col[numzeros] = A.col[j];
                        numzeros++;
                    }
                    // add option of including diagonal
                    else if ( A.col[j] == i &&
                                    B->diagorder_type == Magma_VALUE) {
                        B->val[numzeros] = A.val[j];
                        B->col[numzeros] = A.col[j];
                        numzeros++;
                    }
                    // add option of including diagonal with zero value
                    else if ( A.col[j] == i &&
                                    B->diagorder_type == Magma_ZERO) {
                        B->val[numzeros] = zero;
                        B->col[numzeros] = A.col[j];
                        numzeros++;
                    }
                    // explicit option to omit diagonal
                    else if ( A.col[j] == i &&
                                    B->diagorder_type == Magma_NODIAG) {

                    }
                }
            }
            B->row[rowlimit] = numzeros;
        }
    
        // CSR to CSRU
        else if (  new_format == Magma_CSRU ) {
            // fill in information for B
            B->storage_type = Magma_CSRU;
            B->major = MagmaRowMajor;
            B->fill_mode = MagmaUpper;
            B->num_rows = A.num_rows; 
            B->num_cols = A.num_cols;
            B->pad_rows = A.pad_rows; 
            B->pad_cols = A.pad_cols;
            B->diameter = A.diameter;
            
            int numzeros=0;
            for( int i=0; i < rowlimit; i++) {
                for( int j=A.row[i]; j < A.row[i+1]; j++) {
                    if ( A.col[j] > i ) {
                        numzeros++;
                    }
                    else if ( A.col[j] == i &&
                                    B->diagorder_type != Magma_NODIAG) {
                        numzeros++;
                    }
                }
            }
            B->nnz = numzeros;
            B->true_nnz = numzeros;
            //CHECK( data_zmalloc_cpu( &B->val, numzeros ));
            //CHECK( data_index_malloc_cpu( &B->row, rowlimit+1 ));
            //CHECK( data_index_malloc_cpu( &B->col, numzeros ));
            B->val = (dataType*) calloc( B->nnz, sizeof(dataType) );
            B->row = (int*) calloc( (rowlimit+1), sizeof(int) );
            B->col = (int*) calloc( B->nnz, sizeof(int) );
            
            numzeros=0;
            for( int i=0; i < rowlimit; i++) {
                B->row[i]=numzeros;
                for( int j=A.row[i]; j < A.row[i+1]; j++) {
                    if ( A.col[j] > i) {
                        B->val[numzeros] = A.val[j];
                        B->col[numzeros] = A.col[j];
                        numzeros++;
                    }
                    else if ( A.col[j] == i &&
                                    B->diagorder_type == Magma_UNITY) {
                        B->val[numzeros] = DEV_D_MAKE(one, zero);
                        B->col[numzeros] = A.col[j];
                        numzeros++;
                    }
                    // explicit option of including diagonal
                    else if ( A.col[j] == i &&
                                    B->diagorder_type == Magma_VALUE) {
                        B->val[numzeros] = A.val[j];
                        B->col[numzeros] = A.col[j];
                        numzeros++;
                    }
                    // explicit option of including diagonal with zero value
                    else if ( A.col[j] == i &&
                                    B->diagorder_type == Magma_ZERO) {
                        B->val[numzeros] = DEV_D_MAKE(zero, zero);
                        B->col[numzeros] = A.col[j];
                        numzeros++;
                    }
                    // explicit option to omit diagonal
                    else if ( A.col[j] == i &&
                                    B->diagorder_type == Magma_NODIAG) {

                    }
                    // diagonal included by default
                    else if ( A.col[j] == i ) {
                        B->val[numzeros] = A.val[j];
                        B->col[numzeros] = A.col[j];
                        numzeros++;
                    }
                }
            }
            B->row[rowlimit] = numzeros;
        }
        
        // CSR to CSRD (diagonal elements first)
        else if ( new_format == Magma_CSRD ) {
            // fill in information for B
            B->storage_type = Magma_CSRD;
            B->major = MagmaRowMajor;
            B->fill_mode = A.fill_mode;
            B->num_rows = A.num_rows; 
            B->num_cols = A.num_cols;
            B->pad_rows = A.pad_rows; 
            B->pad_cols = A.pad_cols;
            B->nnz = A.nnz;
            B->true_nnz = A.true_nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
    
            //CHECK( data_zmalloc_cpu( &B->val, A.nnz ));
            //CHECK( data_index_malloc_cpu( &B->row, rowlimit+1 ));
            //CHECK( data_index_malloc_cpu( &B->col, A.nnz ));
            B->val = (dataType*) calloc( A.nnz, sizeof(dataType) );
            B->row = (int*) calloc( (rowlimit+1), sizeof(int) );
            B->col = (int*) calloc( A.nnz, sizeof(int) );
            
            for(int i=0; i < rowlimit; i++) {
                int count = 1;
                for(int j=A.row[i]; j < A.row[i+1]; j++) {
                    if ( A.col[j] == i ) {
                        B->col[A.row[i]] = A.col[j];
                        B->val[A.row[i]] = A.val[j];
                    } else {
                        B->col[A.row[i]+count] = A.col[j];
                        B->val[A.row[i]+count] = A.val[j];
                        count++;
                    }
                }
            }
            for( int i=0; i < rowlimit+1; i++) {
                B->row[i] = A.row[i];
            }
        }
                    
        // CSR to COO
        else if ( new_format == Magma_COO ) {
            //CHECK( data_zmconvert( A, B, Magma_CSR, Magma_CSR, queue ));
            data_zmconvert( A, B, Magma_CSR, Magma_CSR );
            B->storage_type = Magma_COO;
            B->major = MagmaRowMajor;
    
            free( B->row );
            //CHECK( data_index_malloc_cpu( &B->row, A.nnz ));
            B->row = (int*) calloc( A.nnz, sizeof(int) );
            
            for(int i=0; i < rowlimit; i++) {
                for(int j=A.row[i]; j < A.row[i+1]; j++) {
                    B->row[j] = i;
                }
            }
        }
                
        // CSR to CSRCOO
        else if ( new_format == Magma_CSRCOO ) {
            //CHECK( data_zmconvert( A, B, Magma_CSR, Magma_CSR, queue ));
            data_zmconvert( A, B, Magma_CSR, Magma_CSR );
            B->storage_type = Magma_CSRCOO;
            B->major = MagmaRowMajor;
    
            //CHECK( data_index_malloc_cpu( &B->rowidx, A.nnz ));
            B->rowidx = (int*) calloc( A.nnz, sizeof(int) );
            
            for(int i=0; i < rowlimit; i++) {
                for(int j=A.row[i]; j < A.row[i+1]; j++) {
                    B->rowidx[j] = i;
                }
            }
        }
                
        // CSR to DENSE
        else if ( new_format == Magma_DENSE ) {
            //printf( "%% Conversion to DENSE: " );
            // fill in information for B
            B->storage_type = Magma_DENSE;
            B->major = MagmaRowMajor;
            B->fill_mode = A.fill_mode;
            B->num_rows = A.num_rows; 
            B->num_cols = A.num_cols;
            B->pad_rows = A.pad_rows; 
            B->pad_cols = A.pad_cols;
            B->nnz = rowlimit*collimit;
            B->true_nnz = A.true_nnz;
            B->ld = A.num_cols;
            
            // conversion
            B->val = (dataType*) calloc( B->nnz, sizeof(dataType) );
            
            for(int i=0; i < rowlimit; i++ ) {
                for(int j=A.row[i]; j < A.row[i+1]; j++ )
                    B->val[i * (A.num_cols) + A.col[j] ] = A.val[ j ];
            }
    
            //printf( "done\n" );
        }
        
        // CSR to DENSED
        else if ( new_format == Magma_DENSED ) {
            //printf( "%% Conversion to DENSED: " );
            // fill in information for B
            B->num_rows = A.num_rows; 
            B->num_cols = A.num_cols;
            B->pad_rows = A.pad_rows; 
            B->pad_cols = A.pad_cols;
            B->true_nnz = A.true_nnz;
            B->ld = A.ld;
            
            B->storage_type = Magma_DENSED;
            B->diagorder_type = Magma_VALUE;
            B->nnz = MIN(rowlimit, collimit);
            B->val = (dataType*) calloc( B->nnz, sizeof(dataType) );
            
            // conversion
            for(int i=0; i < rowlimit; i++ ) {
              for(int j=A.row[i]; j < A.row[i+1]; j++) {
                if ( A.col[j] == i ) {
                  B->val[ i ] = A.val[ j ];
                }
              }
            }
    
            //printf( "done\n" );
        }
        
        // CSR to CSC
        else if ( new_format == Magma_CSC ) {
            data_zmtranspose(A, B);
            B->storage_type = Magma_CSC;
            B->major = MagmaColMajor;
        }
        // CSR to CSCL
        else if ( new_format == Magma_CSCL ) {
            data_d_matrix C = {Magma_CSR}; 
            data_zmconvert(A, &C, Magma_CSR, Magma_CSRL );
            data_zmtranspose(C, B);
            B->storage_type = Magma_CSCL;
            B->major = MagmaColMajor;
            data_zmfree( &C );
        }
        // CSR to CSCU
        else if ( new_format == Magma_CSCU ) {
            data_d_matrix C = {Magma_CSRU}; 
            data_zmconvert(A, &C, Magma_CSR, Magma_CSRU );
            data_zmtranspose(C, B);
            B->storage_type = Magma_CSCU;
            B->major = MagmaColMajor;
            data_zmfree( &C );
        }
        // CSR to CSCCOO
        else if ( new_format == Magma_CSCCOO ) {
            data_zmconvert( A, B, Magma_CSR, Magma_CSR );
            
            B->rowidx = (int*) calloc( A.nnz, sizeof(int) );
            
            for(int i=0; i < rowlimit; i++) {
                for(int j=A.row[i]; j < A.row[i+1]; j++) {
                    B->rowidx[j] = i;
                }
            }
            data_zmtranspose(A, B);
            B->storage_type = Magma_CSCCOO;
            B->major = MagmaColMajor;
          
        }
        // CSR to BCSR
        else if ( new_format == Magma_BCSR ) {
          if (B->blocksize > 0) {
            B->storage_type = Magma_BCSR;
            B->nnz = A.nnz;
            B->numblocks = 0;
            B->true_nnz = A.nnz;
            B->ldblock = B->blocksize*B->blocksize;
            B->numblocks = -1;
            
            // One based indexing is associated with column major storage in 
            //    mkl_dcsrbsr!
            // Using zero based indexing. 
            // Handle column major - row major transform separately.
            int job[6] = { 0, 0, 0, 0, 0, -1 }; 
            mkl_dcsrbsr(job, &A.num_rows, &B->blocksize, &B->ldblock, 
              A.val, A.col, A.row, NULL, NULL, &B->numblocks, &info);
            
            B->num_rows = (A.num_rows + B->blocksize - 1)/B->blocksize;
            B->num_cols = (A.num_cols + B->blocksize - 1)/B->blocksize;
            LACE_CALLOC(B->val, B->numblocks*B->ldblock);
            LACE_CALLOC(B->row, (B->num_rows+1));
            LACE_CALLOC(B->col, B->numblocks);
            
            job[5] = 1;
            mkl_dcsrbsr(job, &A.num_rows, &B->blocksize, &B->ldblock, 
              A.val, A.col, A.row, B->val, B->col, B->row, &info);
            
            B->nnz = B->numblocks*B->ldblock;
            
          }
          else {
             printf("error: conversion from %d to %d requires blocksize to be set.\n",
               old_format, new_format);
             printf("\tB->blocksize currently = %d .  B->blocksize > 0 required.\n",
               B->blocksize);
             info = DEV_ERR_NOT_SUPPORTED;
             exit(-1);
          }
        }
        
        // CSR to BCSRL
        
        
        
        // CSR to BCSRU
        
        
        
        // CSR to BCSRCOO
        
        
        
        // CSR to BCSC
        
        
        
        // CSR to BCSCL
        
        
        
        // CSR to BCSCU
        
        
        
        // CSR to BCSCCOO
        
 
    }
    
    // anything to CSR
    else if ( new_format == Magma_CSR ) {
    	  //printf("\n <<< new format == DEV_CSR old format =%d\n", old_format);
        // CSRU/CSRCSCU to CSR
        if ( old_format == Magma_CSRU ) {
            //CHECK( data_zmconvert( A, B, Magma_CSR, Magma_CSR, queue ));
            data_zmconvert( A, B, Magma_CSR, Magma_CSR );
        }
        
        // CUCSR to CSR
        else if ( old_format == Magma_CUCSR ){
            //CHECK(data_zmconvert(A, B, Magma_CSR, Magma_CSR, queue));
            data_zmconvert(A, B, Magma_CSR, Magma_CSR );
        }
                
        //// CSRD to CSR (diagonal elements first)
        //else if ( old_format == Magma_CSRD ) {
        //    //CHECK( data_zmconvert( A, B, Magma_CSR, Magma_CSR, queue ));
        //    data_zmconvert( A, B, Magma_CSR, Magma_CSR, queue );
        //    for( int i=0; i < A.num_rows; i++) {
        //        data_zindexsortval(
        //        B->col,
        //        B->val,
        //        B->row[i],
        //        B->row[i+1]-1,
        //        queue );
        //    }
        //}
        
        // CSRCOO to CSR
        else if ( old_format == Magma_CSRCOO ) {
            //CHECK( data_zmconvert( A, B, Magma_CSR, Magma_CSR, queue ));
            data_zmconvert( A, B, Magma_CSR, Magma_CSR );
        }
        

        
        // DENSE to CSR
        else if ( old_format == Magma_DENSE ) {
            //printf( "%% Conversion to CSR: " ); 
            // fill in information for B
            B->storage_type = Magma_CSR;
            B->major = MagmaRowMajor;
            B->fill_mode = A.fill_mode;
            B->num_rows = A.num_rows; 
            B->num_cols = A.num_cols;
            B->pad_rows = A.pad_rows; 
            B->pad_cols = A.pad_cols;
            B->nnz = A.nnz;
            B->true_nnz = A.true_nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
    
            // conversion
            B->nnz=0;
            for( int i=0; i<(A.num_rows)*(A.num_cols); i++ ) {
                if ( DEV_D_REAL(A.val[i]) != zero )
                    (B->nnz)++;
            }
            //CHECK( data_zmalloc_cpu( &B->val, B->nnz));
            //CHECK( data_index_malloc_cpu( &B->row, rowlimit+1 ));
            //CHECK( data_index_malloc_cpu( &B->col, B->nnz ));
            B->val = (dataType*) calloc( B->nnz, sizeof(dataType) );
            B->row = (int*) calloc( (rowlimit+1), sizeof(int) );
            B->col = (int*) calloc( B->nnz, sizeof(int) );
            
            int i = 0;
            int j = 0;
            int k = 0;
    
            for(i=0; i<(rowlimit)*(collimit); i++)
            {
                if ( i%(collimit) == 0 )
                {
                    (B->row)[k] = j;
                    k++;
                }
                if ( DEV_D_REAL(A.val[i]) != 0 )
                {
                    (B->val)[j] = A.val[i];
                    (B->col)[j] = i%(collimit);
                    j++;
                }
            }
            (B->row)[rowlimit]=B->nnz;
    
            //printf( "done\n" );
        }
        
        // BCSR to CSR
        else if ( old_format == Magma_BCSR ) {
            
            if (A.blocksize > 0) {
              //printf("\nBSCSR to CSR A.numblocks=%d\n", A.numblocks);
              B->storage_type = Magma_BCSR;
              B->nnz = A.numblocks*A.ldblock;
              B->major = MagmaRowMajor;
              B->numblocks = 0;
              B->blocksize = 0;
              B->true_nnz = A.nnz;
              B->ldblock = 0;
              
              // One based indexing is associated with column major storage in 
              //    mkl_dcsrbsr!
              // Using zero based indexing. 
              // Handle column major - row major transform separately.
              int job[6] = { 1, 0, 0, 0, 0, 1 }; 
              
              B->num_rows = A.num_rows*A.blocksize;
              B->num_cols = A.num_cols*A.blocksize;
              //LACE_CALLOC(B->val, B->nnz);
              //LACE_CALLOC(B->row, (B->num_rows+1));
              //LACE_CALLOC(B->col, B->nnz);
              B->val = (dataType*) calloc( B->nnz, sizeof(dataType) );
              B->row = (int*) calloc( (B->num_rows+1), sizeof(int) );
              B->col = (int*) calloc( B->nnz, sizeof(int) );
              
              //printf("\nBSCSR to CSR A.numblocks=%d\n", A.numblocks);
              //printf("\nBSCSR to CSR A.blocksize=%d\n", A.blocksize);
              //printf("\nBSCSR to CSR B->num_rows=%d\n", B->num_rows);
              //fflush(stdout);
              mkl_dcsrbsr(job, &A.num_rows, &A.blocksize, &A.ldblock, 
                B->val, B->col, B->row, A.val, A.col, A.row, &info);
              //printf("\ninfo_bsrcsr=%d\n",info);
              //fflush(stdout);
            }
            else {
               printf("error: conversion from %d to %d requires blocksize to be set.\n",
                 old_format, new_format);
               printf("\tB->blocksize currently = %d .  A.blocksize > 0 required.\n",
                 A.blocksize);
               info = DEV_ERR_NOT_SUPPORTED;
               exit(-1);
            }
            
        }
        
        // COO to CSR
        else if ( old_format == Magma_COO ) {
            
            ///////////////
            std::vector< std::pair< int, dataType > > rowval;
            
            B->nnz = A.nnz;
            B->true_nnz = A.true_nnz;
            B->num_rows = A.num_rows; 
            B->num_cols = A.num_cols;
            B->pad_rows = A.pad_rows; 
            B->pad_cols = A.pad_cols;
            B->major = MagmaRowMajor;
            B->row = (int*) calloc( (rowlimit+1), sizeof(int) );
            B->col = (int*) calloc( A.nnz, sizeof(int) );
            B->val = (dataType*) calloc( A.nnz, sizeof(dataType) );
            
            // original code from  Nathan Bell and Michael Garland
            for (int i = 0; i < rowlimit; i++)
                (B->row)[i] = 0;
            
            for (int i = 0; i < A.nnz; i++)
                (B->row)[A.row[i]]++;
            
            // cumulative sum the nnz per row to get row[]
            int cumsum;
            cumsum = 0;
            for(int i = 0; i < rowlimit; i++) {
                int temp = (B->row)[i];
                (B->row)[i] = cumsum;
                cumsum += temp;
            }
            (B->row)[rowlimit] = A.nnz;
            
            // write Aj,Ax into Bj,Bx
            for(int i = 0; i < A.nnz; i++) {
                int row_ = A.row[i];
                int dest = (B->row)[row_];
                (B->col)[dest] = A.col[i];
                (B->val)[dest] = A.val[i];
                (B->row)[row_]++;
            }
            
            int last;
            last = 0;
            for(int i = 0; i <= rowlimit; i++) {
                int temp  = (B->row)[i];
                (B->row)[i] = last;
                last = temp;
            }
            
            (B->row)[rowlimit] = A.nnz;
            
            // sort column indices within each row
            // copy into vector of pairs (column index, value), sort by column index, then copy back
            for (int k=0; k < rowlimit; ++k) {
                int kk  = (B->row)[k];
                int len = (B->row)[k+1] - (B->row)[k];
                rowval.resize( len );
                for( int i=0; i < len; ++i ) {
                    rowval[i] = std::make_pair( (B->col)[kk+i], (B->val)[kk+i] );
                }
                std::sort( rowval.begin(), rowval.end(), compare_first );
                for( int i=0; i < len; ++i ) {
                    (B->col)[kk+i] = rowval[i].first;
                    (B->val)[kk+i] = rowval[i].second;
                }
            }
            ///////////////
        }
        
        // CSC to CSR
        else if (old_format == Magma_CSC ) {
          data_zmconvert(A, B, Magma_CSR, Magma_CSC );
          B->storage_type = Magma_CSR;
          B->major = MagmaRowMajor;
        }
        else if (old_format == Magma_CSCU ) {
          data_zmconvert(A, B, Magma_CSR, Magma_CSC );
          B->storage_type = new_format;
          B->major = MagmaRowMajor;
        }
        else if (old_format == Magma_CSCL ) {
          data_zmconvert(A, B, Magma_CSR, Magma_CSC );
          B->storage_type = new_format;
          B->major = MagmaRowMajor;
        }
        
        else {
            printf("error: format not supported %d to %d.\n",
              old_format, new_format);
            //datablasSetKernelStream( queue );
            info = DEV_ERR_NOT_SUPPORTED;
        }
    }
    // CSC to anything
    else if ( old_format == Magma_CSC 
        || old_format == Magma_CSCD
        || old_format == Magma_CSCU 
        || old_format == Magma_CSCL
        || old_format == Magma_CSCCOO ) {
      if (new_format == Magma_CSCL ) {
          data_zmconvert(A, B, Magma_CSR, Magma_CSRL );
          B->storage_type = Magma_CSCL;
          B->major = MagmaColMajor;
      }
      else if (new_format == Magma_CSCU ) {
          data_zmconvert(A, B, Magma_CSR, Magma_CSRU );
          B->storage_type = Magma_CSCU;
          B->major = MagmaColMajor;
      }
      else if (new_format == Magma_CSRU ) {
          data_zmconvert(A, B, Magma_CSC, Magma_CSRU );
          B->storage_type = Magma_CSRU;
          B->major = MagmaRowMajor;
      }
      else if (new_format == Magma_CSRCOO ) {
          data_zmconvert(A, B, Magma_CSR, Magma_CSRCOO );
          B->storage_type = Magma_CSCCOO;
          B->major = MagmaColMajor;
      }
      else if (new_format == Magma_DENSE ) {
          data_zmconvert(A, B, Magma_CSR, Magma_DENSE );
          B->storage_type = Magma_DENSE;
          B->major = MagmaColMajor;
      }
    }
    // DENSE to anything DENSE
    else if ( ( old_format == Magma_DENSE 
        || old_format == Magma_DENSEL 
        || old_format == Magma_DENSEU 
        || old_format == Magma_DENSED ) && 
      ( new_format == Magma_DENSE 
        || new_format == Magma_DENSEL 
        || new_format == Magma_DENSEU 
        || new_format == Magma_DENSED ) ) {
      
      B->num_rows = A.num_rows; 
      B->num_cols = A.num_cols;
      B->pad_rows = A.pad_rows; 
      B->pad_cols = A.pad_cols;
      B->nnz = A.nnz;
      B->true_nnz = A.true_nnz;
      B->ld = A.ld;
      if (B->major != MagmaRowMajor && B->major != MagmaColMajor) {
        //printf("B->major = %d : it has not been set\n", B->major);
        B->major = A.major;
      }
      
      if ( new_format == Magma_DENSE ) {
          //printf("\n%% dense to dense ");
          B->storage_type = Magma_DENSE;
          B->fill_mode = A.fill_mode;
          
          B->val = (dataType*) calloc( rowlimit*collimit, sizeof(dataType) );
          
          if (A.major == MagmaRowMajor) {
              //printf("A is row major ");
              if (B->major == MagmaRowMajor) { 
                  for(int i=0; i < A.num_rows; i++ ) {
                      for(int j=0; j < A.num_cols; j++ )
                          B->val[ i * (A.ld) + j ] = A.val[ i * (A.ld) + j ];
                  }
              }
              else {
                  for(int j=0; j < A.num_cols; j++ ) {
                      for(int i=0; i < A.num_rows; i++ )
                          B->val[ i + j * (A.ld) ] = A.val[ i * (A.ld) + j ];
                  }
              }
          }
          else {
              //printf("A is column major ");
              if (B->major == MagmaRowMajor) { 
                  for(int i=0; i < A.num_rows; i++ ) {
                      for(int j=0; j < A.num_cols; j++ )
                          B->val[ i * (A.ld) + j ] = A.val[ i + j * (A.ld) ];
                  }
              }
              else { 
                  for(int j=0; j < A.num_cols; j++ ) {
                      for(int i=0; i < A.num_rows; i++ ) 
                          B->val[ i + j * (A.ld) ] = A.val[ i + j * (A.ld) ];
                  } 
              }
          }

                    
          if (A.pad_rows > 0 && A.pad_cols > 0) {
              if (B->major == MagmaRowMajor) { 
                  for ( int i = A.num_rows; i < A.pad_rows; i++ ) {
                      B->val[ i*A.ld + i ] = one;
                  }
              }
              else {
                  for ( int i = A.num_cols; i < A.pad_cols; i++ ) {
                      B->val[ i + i*A.ld ] = one;
                  }
              }
          }
          //printf( "done\n" );
      }
      else if ( new_format == Magma_DENSEL ) {
          //printf("\n%% dense to denseL ");
          B->storage_type = Magma_DENSEL;
          B->fill_mode = MagmaLower;
          
          B->val = (dataType*) calloc( rowlimit*collimit, sizeof(dataType) );
          
          if (A.major == MagmaRowMajor) {
              //printf("A is Row major B->diagorder_type = %d\n", B->diagorder_type);
              if (B->major == MagmaRowMajor) {
                  //printf("B is row major B->diagorder_type = %d ", B->diagorder_type);
                  for(int i=0; i < A.num_rows; i++ ) {
                      for(int j=0; j < i; j++ )
                          B->val[ i * (A.ld) + j ] = A.val[ i * (A.ld) + j ];
                      if ( B->diagorder_type == Magma_VALUE )
                        B->val[ i * (A.ld) + i ] = A.val[ i * (A.ld) + i ];
                      else if ( B->diagorder_type == Magma_UNITY )
                        B->val[ i * (A.ld) + i ] = one;
                      else if ( B->diagorder_type == Magma_NODIAG )
                        B->val[ i * (A.ld) + i ] = zero;
                  }
              }
              else {
                  //printf("B is Col major B->diagorder_type = %d ", B->diagorder_type);
                  for(int j=0; j < A.num_cols; j++ ) {
                      if ( B->diagorder_type == Magma_VALUE )
                        B->val[ j + j * (A.ld) ] = A.val[ j * (A.ld) + j ];
                      else if ( B->diagorder_type == Magma_UNITY )
                        B->val[ j + j * (A.ld) ] = one;
                      else if ( B->diagorder_type == Magma_NODIAG )
                        B->val[ j + j * (A.ld) ] = zero;
                      for(int i=j+1; i < A.num_rows; i++ )
                          B->val[ i + j * (A.ld) ] = A.val[ i * (A.ld) + j ];
                  }
              }
          }
          else {
              //printf("A is Col major B->diagorder_type = %d ", B->diagorder_type);
              if (B->major == MagmaRowMajor) {
                  //printf("B is row major B->diagorder_type = %d ", B->diagorder_type);
                  for(int j=0; j < A.num_cols; j++ ) {
                      if ( B->diagorder_type == Magma_VALUE )
                        B->val[ j * (A.ld) + j ] = A.val[ j + j * (A.ld) ];
                      else if ( B->diagorder_type == Magma_UNITY )
                        B->val[ j * (A.ld) + j ] = one;
                      else if ( B->diagorder_type == Magma_NODIAG )
                        B->val[ j * (A.ld) + j ] = zero;  
                      for(int i=j+1; i < A.num_rows; i++ ) 
                          B->val[ i * (A.ld) + j ] = A.val[ i + j * (A.ld) ];
                  }
              }
              else {
                //printf("B is Col major B->diagorder_type = %d ", B->diagorder_type);
                for(int j=0; j < A.num_cols; j++ ) {
                    if ( B->diagorder_type == Magma_VALUE )
                      B->val[ j + j * (A.ld) ] = A.val[ j + j * (A.ld) ];
                    else if ( B->diagorder_type == Magma_UNITY )
                      B->val[ j + j * (A.ld) ] = one;
                    else if ( B->diagorder_type == Magma_NODIAG )
                      B->val[ j + j * (A.ld) ] = zero;  
                    for(int i=j+1; i < A.num_rows; i++ ) 
                        B->val[ i + j * (A.ld) ] = A.val[ i + j * (A.ld) ];
                  }
              }
          }
          
                    
          if (A.pad_rows > 0 && A.pad_cols > 0) {
              if (B->major == MagmaRowMajor) { 
                  for ( int i = A.num_rows; i < A.pad_rows; i++ ) {
                      B->val[ i*A.ld + i ] = one;
                  }
              }
              else {
                  for ( int i = A.num_cols; i < A.pad_cols; i++ ) {
                      B->val[ i + i*A.ld ] = one;
                  }
              }
          }
          //printf( "done\n" );
      }
      else if ( new_format == Magma_DENSEU ) {
          //printf("\n%% dense* to denseU ");
          B->storage_type = Magma_DENSEU;
          B->fill_mode = MagmaUpper;
          
          B->val = (dataType*) calloc( rowlimit*collimit, sizeof(dataType) );
          
          if (A.major == MagmaRowMajor) {
              //printf("A is row major B->diagorder_type = %d ", B->diagorder_type);
              if (B->major == MagmaRowMajor) {
                  //printf("B is Row major B->diagorder_type = %d\n", B->diagorder_type);
                  for(int i=0; i < A.num_rows; i++ ) {
                      if ( i < A.num_cols ) {
                          if ( B->diagorder_type == Magma_VALUE )
                            B->val[ i * (A.ld) + i ] = A.val[ i * (A.ld) + i ];
                          else if ( B->diagorder_type == Magma_UNITY )
                            B->val[ i * (A.ld) + i ] = one;
                          else if ( B->diagorder_type == Magma_NODIAG )
                            B->val[ i * (A.ld) + i ] = zero;
                          for(int j=i+1; j < A.num_cols; j++ )
                              B->val[ i * (A.ld) + j ] = A.val[ i * (A.ld) + j ];   
                      }
                  }
              }
              else {
                  //printf("B is Col major B->diagorder_type = %d\n", B->diagorder_type);
                  for(int i=0; i < A.num_rows; i++ ) {
                      if ( i < A.num_cols ) {
                          if ( B->diagorder_type == Magma_VALUE )
                            B->val[ i + i * (A.ld) ] = A.val[ i * (A.ld) + i ];
                          else if ( B->diagorder_type == Magma_UNITY )
                            B->val[ i + i * (A.ld)] = one;
                          else if ( B->diagorder_type == Magma_NODIAG )
                            B->val[ i + i * (A.ld) ] = zero;
                          for(int j=i+1; j < A.num_cols; j++ )
                              B->val[ i + j * (A.ld) ] = A.val[ i * (A.ld) + j ];   
                      }
                  }
              }
          }
          else {
              //printf("A is Col major B->diagorder_type = %d\n", B->diagorder_type);
              if (B->major == MagmaRowMajor) {
                  //printf("B is Row major B->diagorder_type = %d\n", B->diagorder_type);
                  for(int j=0; j < A.num_cols; j++ ) {
                      if ( j < A.num_rows ) {
                          if ( B->diagorder_type == Magma_VALUE )
                            B->val[ j * (A.ld) + j ] = A.val[ j + j * (A.ld) ];
                          else if ( B->diagorder_type == Magma_UNITY )
                            B->val[ j * (A.ld) + j ] = one;
                          else if ( B->diagorder_type == Magma_NODIAG )
                            B->val[ j * (A.ld) + j ] = zero; 
                          for(int i=0; i < j; i++ ) 
                              B->val[ i * (A.ld) + j ] = A.val[ i + j * (A.ld) ];  
                      }
                  }
              }
              else {
                  //printf("B is Col major B->diagorder_type = %d\n", B->diagorder_type);
                  for(int j=0; j < A.num_cols; j++ ) {
                      if ( j < A.num_rows ) {
                          if ( B->diagorder_type == Magma_VALUE )
                            B->val[ j + j * (A.ld) ] = A.val[ j + j * (A.ld) ];
                          else if ( B->diagorder_type == Magma_UNITY )
                            B->val[ j + j * (A.ld) ] = one;
                          else if ( B->diagorder_type == Magma_NODIAG )
                            B->val[ j + j * (A.ld) ] = zero; 
                          for(int i=j+1; i < A.num_rows; i++ ) 
                              B->val[ i + j * (A.ld) ] = A.val[ i + j * (A.ld) ];  
                      }
                  }
              }
          }
          
          if ( B->diagorder_type != Magma_NODIAG ) {
            if (A.pad_rows > 0 && A.pad_cols > 0) {
              //printf("\n U padded!!!\n");
                if (B->major == MagmaRowMajor) { 
                    for ( int i = A.num_rows; i < A.pad_rows; i++ ) {
                        B->val[ i*A.ld + i ] = one;
                    }
                }
                else {
                    for ( int i = A.num_cols; i < A.pad_cols; i++ ) {
                        B->val[ i + i*A.ld ] = one;
                    }
                }
            }
          }
          //printf( "done\n" );
        
      }
      else if ( new_format == Magma_DENSED ) {
          //printf("\n%%dense to denseD ");
          B->storage_type = Magma_DENSED;
          B->diagorder_type = Magma_VALUE;
          B->nnz = MIN(rowlimit, collimit);
          B->val = (dataType*) calloc( B->nnz, sizeof(dataType) );
          
          for(int i=0; i < B->nnz; i++ ) {
              B->val[ i ] = A.val[ i * (A.ld) + i ];
          }
          
      }
      
    }
    // BCSR to BCSRL
    else if ( ( old_format == Magma_BCSR ) && ( new_format == Magma_BCSRL ) ) {
        // fill in information for B
        B->storage_type = Magma_BCSR;
        B->major = MagmaRowMajor;
        B->fill_mode = MagmaLower;
        B->num_rows = A.num_rows; 
        B->num_cols = A.num_cols;
        B->pad_rows = A.pad_rows; 
        B->pad_cols = A.pad_cols;
        B->diameter = A.diameter;
        
        B->blocksize = A.blocksize;
        B->ldblock = A.ldblock;
        B->numblocks = -1;
    
        int numblocks=0;
        for( int i=0; i < rowlimit; i++) {
            for( int j=A.row[i]; j < A.row[i+1]; j++) {
                if ( A.col[j] < i) {
                    numblocks++;
                }
                else if ( A.col[j] == i &&
                                B->diagorder_type != Magma_NODIAG ) {
                    numblocks++;
                }
            }
        }
        B->numblocks = numblocks;
        B->nnz = numblocks*B->ldblock;
        //CHECK( data_zmalloc_cpu( &B->val, numzeros ));
        //CHECK( data_index_malloc_cpu( &B->row, rowlimit+1 ));
        //CHECK( data_index_malloc_cpu( &B->col, numzeros ));
        B->val = (dataType*) calloc( B->nnz, sizeof(dataType) );
        B->row = (int*) calloc( (rowlimit+1), sizeof(int) );
        B->col = (int*) calloc( B->numblocks, sizeof(int) );
        
        numblocks=0;
        for( int i=0; i < rowlimit; i++) {
            B->row[i]=numblocks;
            for( int j=A.row[i]; j < A.row[i+1]; j++) {
                // diagonal omitted by default
                if ( A.col[j] < i) {
                    for (int k=0; k< B->ldblock; k++) {
                        B->val[numblocks*B->ldblock+k] = A.val[j*B->ldblock+k];
                    }
                    B->col[numblocks] = A.col[j];
                    numblocks++;
                }
                // add option of including diagonal with unit value
                else if ( A.col[j] == i &&
                                B->diagorder_type == Magma_UNITY) {
                    //for (int k=0; k< B->ldblock; k++) {
                    //    B->val[numblocks*B->ldblock+k] = one;
                    //}
                    for (int k=0; k<B->blocksize; k++) {
                        B->val[numblocks*B->ldblock+k*B->blocksize+k] = one;
                    }
                    B->col[numblocks] = A.col[j];
                    numblocks++;
                }
                // add option of including diagonal
                else if ( A.col[j] == i &&
                                B->diagorder_type == Magma_VALUE) {
                    for (int k=0; k< B->ldblock; k++) {
                        B->val[numblocks*B->ldblock+k] = A.val[j*B->ldblock+k];
                    }
                    B->col[numblocks] = A.col[j];
                    numblocks++;
                }
                // add option of including diagonal with zero value
                else if ( A.col[j] == i &&
                                B->diagorder_type == Magma_ZERO) {
                    for (int k=0; k< B->ldblock; k++) {
                        B->val[numblocks*B->ldblock+k] = zero;
                    }
                    B->col[numblocks] = A.col[j];
                    numblocks++;
                }
                // explicit option to omit diagonal
                else if ( A.col[j] == i &&
                                B->diagorder_type == Magma_NODIAG) {

                }
            }
        }
        B->row[rowlimit] = numblocks;
    }
    // BCSR to BCSRU
    else if ( ( old_format == Magma_BCSR ) && ( new_format == Magma_BCSRU ) ) {
        // fill in information for B
        B->storage_type = Magma_BCSRU;
        B->major = MagmaRowMajor;
        B->fill_mode = MagmaUpper;
        B->num_rows = A.num_rows; 
        B->num_cols = A.num_cols;
        B->pad_rows = A.pad_rows; 
        B->pad_cols = A.pad_cols;
        B->diameter = A.diameter;
        
        B->blocksize = A.blocksize;
        B->ldblock = A.ldblock;
        B->numblocks = -1;
        
        int numblocks=0;
        for( int i=0; i < rowlimit; i++) {
            for( int j=A.row[i]; j < A.row[i+1]; j++) {
                if ( A.col[j] > i ) {
                    numblocks++;
                }
                else if ( A.col[j] == i &&
                                B->diagorder_type != Magma_NODIAG) {
                    numblocks++;
                }
            }
        }
        B->numblocks = numblocks;
        B->nnz = numblocks*B->ldblock;
        //CHECK( data_zmalloc_cpu( &B->val, numzeros ));
        //CHECK( data_index_malloc_cpu( &B->row, rowlimit+1 ));
        //CHECK( data_index_malloc_cpu( &B->col, numzeros ));
        B->val = (dataType*) calloc( B->nnz, sizeof(dataType) );
        B->row = (int*) calloc( (rowlimit+1), sizeof(int) );
        B->col = (int*) calloc( B->numblocks, sizeof(int) );
        
        numblocks=0;
        for( int i=0; i < rowlimit; i++) {
            B->row[i]=numblocks;
            for( int j=A.row[i]; j < A.row[i+1]; j++) {
                if ( A.col[j] > i) {
                    for (int k=0; k< B->ldblock; k++) {
                        B->val[numblocks*B->ldblock+k] = A.val[j*B->ldblock+k];
                    }
                    B->col[numblocks] = A.col[j];
                    numblocks++;
                }
                else if ( A.col[j] == i &&
                                B->diagorder_type == Magma_UNITY) {
                    //for (int k=0; k< B->ldblock; k++) {
                    //    B->val[numblocks*B->ldblock+k] = one;
                    //}
                    for (int k=0; k<B->blocksize; k++) {
                        B->val[numblocks*B->ldblock+k*B->blocksize+k] = one;
                    }
                    B->col[numblocks] = A.col[j];
                    numblocks++;
                }
                // explicit option of including diagonal
                else if ( A.col[j] == i &&
                                B->diagorder_type == Magma_VALUE) {
                    for (int k=0; k< B->ldblock; k++) {
                        B->val[numblocks*B->ldblock+k] = A.val[j*B->ldblock+k];
                    }
                    B->col[numblocks] = A.col[j];
                    numblocks++;
                }
                // explicit option of including diagonal with zero value
                else if ( A.col[j] == i &&
                                B->diagorder_type == Magma_ZERO) {
                    for (int k=0; k< B->ldblock; k++) {
                        B->val[numblocks*B->ldblock+k] = zero;
                    }
                    B->col[numblocks] = A.col[j];
                    numblocks++;
                }
                // explicit option to omit diagonal
                else if ( A.col[j] == i &&
                                B->diagorder_type == Magma_NODIAG) {
    
                }
                // diagonal included by default
                else if ( A.col[j] == i ) {
                    for (int k=0; k< B->ldblock; k++) {
                        B->val[numblocks*B->ldblock+k] = A.val[j*B->ldblock+k];
                    }
                    B->col[numblocks] = A.col[j];
                    numblocks++;
                }
            }
        }
        B->row[rowlimit] = numblocks;
    }
    
    // BCSR to BCSCU
    else if ( ( old_format == Magma_BCSR ) && ( new_format == Magma_BCSCU ) ) {
        data_d_matrix C = {Magma_BCSR}; 
        data_zmconvert(A, &C, Magma_BCSR, Magma_BCSRU );
        data_zmtranspose(C, B);
        B->storage_type = Magma_BCSCU;
        B->major = MagmaColMajor;
        data_zmfree( &C );
    }
    
    else {
        printf("error: conversion not supported %d to %d.\n",
          old_format, new_format);
        info = DEV_ERR_NOT_SUPPORTED;
    }

//cleanup:    
    if ( info != 0 ) {
      free( B );
    }
    return info;
}

int
data_zcheckupperlower(
  data_d_matrix * A ) {

  int info = 0;
  
  if (A->storage_type == Magma_CSRL) {
    for( int i=0; i < A->num_rows; i++) {
      for( int j=A->row[i]; j < A->row[i+1]; j++) {
        if ( A->col[j] > i ) {
          printf("%d, %d : %e \n", i, A->col[j], A->val[j]);
          info = -1;
        }
      }
    }
  }
  else if (A->storage_type == Magma_CSRU) {
    for( int i=0; i < A->num_rows; i++) {
      for( int j=A->row[i]; j < A->row[i+1]; j++) {
        if ( A->col[j] < i ) {
          printf("%d, %d : %e \n", i, A->col[j], A->val[j]);
          info = -1;
        }
      }
    }
  }
  
  return info;
  
}

extern "C" 
int
data_zmcopy(
    data_d_matrix A,
    data_d_matrix *B )
{
	
    printf("data_zmcopy\n");
    int info = 0;
    //dataType one = dataType(1.0);
    //dataType zero = dataType(0.0);
    
    B->val = NULL;
    B->col = NULL;
    B->row = NULL;
    B->rowidx = NULL;
    B->list = NULL;
    B->blockinfo = NULL;
    B->diag = NULL;
    
    int rowlimit = A.num_rows;
    //int collimit = A.num_cols;
    if (A.pad_rows > 0 && A.pad_cols > 0) {
       rowlimit = A.pad_rows;
       //collimit = A.pad_cols;
    }
      
    
    // CSR
    if ( A.storage_type == Magma_CSR 
      || A.storage_type == Magma_CSRL 
      || A.storage_type == Magma_CSRU )
    {
      printf("copying CSR\n");
      // fill in information for B
      B->storage_type = A.storage_type;
      B->major = A.major;
      B->fill_mode = A.fill_mode;
      B->num_rows = A.num_rows; 
      B->num_cols = A.num_cols;
      B->pad_rows = A.pad_rows; 
      B->pad_cols = A.pad_cols;
      B->nnz = A.nnz;
      B->true_nnz = A.true_nnz;
      B->max_nnz_row = A.max_nnz_row;
      B->diameter = A.diameter;
      
      B->val = (dataType*) calloc( A.nnz, sizeof(dataType) );
      B->row = (int*) calloc( (rowlimit+1), sizeof(int) );
      B->col = (int*) calloc( A.nnz, sizeof(int) );
      
      for( int i=0; i < A.nnz; i++) {
          B->val[i] = A.val[i];
          B->col[i] = A.col[i];
      }
      for( int i=0; i < rowlimit+1; i++) {
          B->row[i] = A.row[i];
      }
        
    }
    
    // BCSR 
    else if ( A.storage_type == Magma_BCSR 
      || A.storage_type == Magma_BCSRL 
      || A.storage_type == Magma_BCSRU ) {
        printf("copying BCSR\n");
        // fill in information for B
        B->storage_type = A.storage_type;
        B->major = A.major;
        B->fill_mode = A.fill_mode;
        B->num_rows = A.num_rows; 
        B->num_cols = A.num_cols;
        B->pad_rows = A.pad_rows; 
        B->pad_cols = A.pad_cols;
        B->diameter = A.diameter;
        
        B->blocksize = A.blocksize;
        B->ldblock = A.ldblock;
        printf("%s %d B->ldblock=%d\n", __FILE__, __LINE__, B->ldblock);
        B->numblocks = A.numblocks;
        B->nnz = A.nnz;
        //CHECK( data_zmalloc_cpu( &B->val, numzeros ));
        //CHECK( data_index_malloc_cpu( &B->row, rowlimit+1 ));
        //CHECK( data_index_malloc_cpu( &B->col, numzeros ));
        B->val = (dataType*) calloc( B->nnz, sizeof(dataType) );
        B->row = (int*) calloc( (rowlimit+1), sizeof(int) );
        B->col = (int*) calloc( B->numblocks, sizeof(int) );
        
        for( int i=0; i < A.numblocks; i++) {
          B->col[i] = A.col[i];
          for (int k=0; k< B->ldblock; k++) {
              B->val[i*B->ldblock+k] = A.val[i*B->ldblock+k];
          }
        }
        for( int i=0; i < rowlimit+1; i++) {
            B->row[i] = A.row[i];
        }
        
    }
    
    if ( info != 0 ) {
      free( B );
    }
    return info;
}