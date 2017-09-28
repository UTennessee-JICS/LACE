/*
    -- DEV (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
       @author Hartwig Anzt
       @author Mark Gates
       @author Stephen Wood

*/
#include <stdio.h>
#include "../include/sparse.h"

#ifdef DEV_WITH_MKL
    #define MKL_Complex8  dataFloatComplex
    #define MKL_Complex16 dataDoubleComplex
    #include <mkl_spblas.h>
    #include <mkl_trans.h>
#endif

/*******************************************************************************
    Purpose
    -------
    Transposes a matrix stored in CSR format on the CPU host.


    Arguments
    ---------
    @param[in]
    n_rows      data_int_t
                number of rows in input matrix

    @param[in]
    n_cols      data_int_t
                number of columns in input matrix

    @param[in]
    nnz         data_int_t
                number of nonzeros in input matrix

    @param[in]
    values      dataDoubleComplex*
                value array of input matrix

    @param[in]
    rowptr      data_index_t*
                row pointer of input matrix

    @param[in]
    colind      data_index_t*
                column indices of input matrix

    @param[in]
    new_n_rows  data_index_t*
                number of rows in transposed matrix

    @param[in]
    new_n_cols  data_index_t*
                number of columns in transposed matrix

    @param[in]
    new_nnz     data_index_t*
                number of nonzeros in transposed matrix

    @param[in]
    new_values  dataDoubleComplex**
                value array of transposed matrix

    @param[in]
    new_rowptr  data_index_t**
                row pointer of transposed matrix

    @param[in]
    new_colind  data_index_t**
                column indices of transposed matrix


    @ingroup datasparse_zaux
*******************************************************************************/

extern "C" int
z_transpose_csr(
    data_d_matrix A,
    data_d_matrix *B )
{
    data_int_t info = 0;
    data_int_t workaround = 0;
    
    B->nnz = A.nnz;
    B->true_nnz = A.true_nnz;
    B->fill_mode = A.fill_mode;
    B->num_rows = A.num_cols; 
    B->num_cols = A.num_rows;
    B->pad_rows = A.pad_cols; 
    B->pad_cols = A.pad_rows;
    int rowlimit = A.num_rows;
    //int collimit = A.num_rows;
    if (A.pad_rows > 0 && A.pad_cols > 0) {
       rowlimit = A.pad_rows;
       //collimit = A.pad_rows;
    }
    B->row = (int*) malloc( (rowlimit+1)*sizeof(int) );
    B->col = (int*) malloc( A.nnz*sizeof(int) );
    B->val = (dataType*) malloc( A.nnz*sizeof(dataType) );
    B->storage_type = A.storage_type;
    
    // this workaround should resolve the problem with the 1 indexing in case of MKL
    // we check whether first element in A.rowptr is 1
    if( A.row[0] == 1 ){
        // increment to create one-based indexing of the array parameters
        #pragma omp parallel 
        {
        #pragma omp for simd nowait
        for (data_int_t i=0; i<A.nnz; i++) {
            A.col[i] -= 1;
        }
        #pragma omp for simd nowait
        for (data_int_t i=0; i<rowlimit+1; i++) {
            A.row[i] -= 1;	
        }
        }
        workaround = 1;       
    }
    //CHECK( data_zmtransfer( A, B, A.memory_location, A.memory_location, queue) );
    
    // easier to keep names straight if we convert CSR to CSC,
    // which is the same as tranposing CSR.
    // dataDoubleComplex *csc_values=NULL;
    // data_index_t *csc_colptr=NULL, *csc_rowind=NULL;
    
    // i, j are actual row & col indices (0 <= i < nrows, 0 <= j < ncols).
    // k is index into col and values (0 <= k < nnz).
    data_int_t i, j, k, total;
    
    // CHECK( data_zmalloc_cpu( &csc_values, nnz ) );
    // CHECK( data_index_malloc_cpu( &csc_colptr, n_cols + 1 ) );
    // CHECK( data_index_malloc_cpu( &csc_rowind, nnz ) );
    
    // example matrix
    // [ x x 0 x ]
    // [ x 0 x x ]
    // [ x x 0 0 ]
    // rowptr = [ 0 3 6, 8 ]
    // colind = [ 0 1 3; 0 2 3; 0 1 ]
    
    // sum up nnz in each original column
    // colptr = [ 3 2 1 2, X ]
    #pragma omp parallel private (j) 
    {
    #pragma omp for simd nowait
    for( j=0; j < rowlimit+1; j++ ) {
        B->row[ j ] = 0;
    }
    }
    for( k=0; k < A.nnz; k++ ) {
        B->row[ A.col[k]+1 ]++;
    }
    
    // running sum to convert to new colptr
    // colptr = [ 0 3 5 6, 8 ]
    total = 0;
    for( j=0; j < rowlimit+1; j++ ) {
        total = total + B->row[ j ];
        B->row[ j ] = total;
    }
    if (A.pad_rows != 0) {
        //printf("total=%d A.true_nnz=%d\n", total, A.true_nnz);
        assert( total == A.true_nnz );
    }
    else { 
        //printf("total=%d A.nnz=%d\n", total, A.nnz);
        assert( total == A.nnz );
    }
    
    // copy row indices and values
    // this increments colptr until it effectively shifts left one
    // colptr = [ 3 5 6 8, 8 ]
    // rowind = [ 0 1 2; 0 2; 1; 0 1 ]
    for( i=0; i < rowlimit; i++ ) {
        for( k=A.row[i]; k < A.row[i+1]; k++ ) {
            j = A.col[k];
            B->col[ B->row[ j ] ] = i;
            B->val[ B->row[ j ] ] = A.val[k];
            B->row[ j ]++;
        }
    }
    if (A.pad_rows != 0) {
        //printf("total=%d A.true_nnz=%d\n", total, A.true_nnz);
        assert( B->row[ rowlimit-1 ] == A.true_nnz );
    }
    else { 
        //printf("total=%d A.nnz=%d\n", total, A.nnz);
        assert( B->row[ rowlimit-1 ] == A.nnz );
    }
    
    // shift colptr right one
    // colptr = [ 0 3 5 6, 8 ]
    for( j=rowlimit-1; j > 0; j-- ) {
        B->row[j] = B->row[j-1];
    }
    B->row[0] = 0;
    
    if( workaround == 1 ){
        // increment to create one-based indexing of the array parameters
        #pragma omp parallel 
        {
        #pragma omp for simd nowait
        for (data_int_t it=0; it<B->nnz; it++) {
            A.col[it] += 1;
            B->col[it] += 1;
        }
        #pragma omp for simd nowait
        for (data_int_t it=0; it<rowlimit+1; it++) {
            A.row[it] += 1;	
            B->row[it] += 1;	
        }
        }
    } 
   
    
//cleanup:
    return info;
}

extern "C" int
z_transpose_bcsr(
    data_d_matrix A,
    data_d_matrix *B )
{
    
    data_int_t info = 0;
    data_int_t workaround = 0;
    
    B->nnz = A.nnz;
    B->true_nnz = A.true_nnz;
    B->fill_mode = A.fill_mode;
    B->num_rows = A.num_cols; 
    B->num_cols = A.num_rows;
    B->pad_rows = A.pad_cols; 
    B->pad_cols = A.pad_rows;
    B->diameter = A.diameter;
    
    B->blocksize = A.blocksize;
    B->ldblock = A.ldblock;
    B->numblocks = A.numblocks;
    
    int rowlimit = A.num_rows;
    //int collimit = A.num_rows;
    if (A.pad_rows > 0 && A.pad_cols > 0) {
       rowlimit = A.pad_rows;
       //collimit = A.pad_rows;
    }
    B->row = (int*) malloc( (rowlimit+1)*sizeof(int) );
    B->col = (int*) malloc( A.nnz*sizeof(int) );
    B->val = (dataType*) malloc( A.nnz*sizeof(dataType) );
    B->storage_type = A.storage_type;
    
    // this workaround should resolve the problem with the 1 indexing in case of MKL
    // we check whether first element in A.rowptr is 1
    if( A.row[0] == 1 ){
        // increment to create one-based indexing of the array parameters
        #pragma omp parallel 
        {
        #pragma omp for simd nowait
        for (data_int_t i=0; i<A.numblocks; i++) {
            A.col[i] -= 1;
        }
        #pragma omp for simd nowait
        for (data_int_t i=0; i<rowlimit+1; i++) {
            A.row[i] -= 1;	
        }
        }
        workaround = 1;       
    }
    //CHECK( data_zmtransfer( A, B, A.memory_location, A.memory_location, queue) );
    
    // easier to keep names straight if we convert CSR to CSC,
    // which is the same as tranposing CSR.
    // dataDoubleComplex *csc_values=NULL;
    // data_index_t *csc_colptr=NULL, *csc_rowind=NULL;
    
    // i, j are actual row & col indices (0 <= i < nrows, 0 <= j < ncols).
    // k is index into col and values (0 <= k < nnz).
    data_int_t i, j, k, total;
    
    // CHECK( data_zmalloc_cpu( &csc_values, nnz ) );
    // CHECK( data_index_malloc_cpu( &csc_colptr, n_cols + 1 ) );
    // CHECK( data_index_malloc_cpu( &csc_rowind, nnz ) );
    
    // example matrix
    // [ x x 0 x ]
    // [ x 0 x x ]
    // [ x x 0 0 ]
    // rowptr = [ 0 3 6, 8 ]
    // colind = [ 0 1 3; 0 2 3; 0 1 ]
    
    // sum up nnz in each original column
    // colptr = [ 3 2 1 2, X ]
    #pragma omp parallel private (j) 
    {
    #pragma omp for simd nowait
    for( j=0; j < rowlimit+1; j++ ) {
        B->row[ j ] = 0;
    }
    }
    
    DEV_PRINTF("\nA.num_rows=%d\n", A.num_rows );
    for( k=0; k < A.numblocks; k++ ) {
        DEV_PRINTF("A.col[k]+1=%d\n", A.col[k]+1);
        B->row[ A.col[k]+1 ]++;
    }
    
    // running sum to convert to new colptr
    // colptr = [ 0 3 5 6, 8 ]
    total = 0;
    for( j=0; j < rowlimit+1; j++ ) {
        total = total + B->row[ j ];
        B->row[ j ] = total;
    }
    if (A.pad_rows != 0) {
        //printf("total=%d A.true_nnz=%d\n", total, A.true_nnz);
        assert( total == A.numblocks );
    }
    else { 
        //printf("total=%d A.nnz=%d\n", total, A.nnz);
        assert( total == A.numblocks );
    }
    
    // copy row indices and values
    // this increments colptr until it effectively shifts left one
    // colptr = [ 3 5 6 8, 8 ]
    // rowind = [ 0 1 2; 0 2; 1; 0 1 ]
    for( i=0; i < rowlimit; i++ ) {
        for( k=A.row[i]; k < A.row[i+1]; k++ ) {
            j = A.col[k];
            B->col[ B->row[ j ] ] = i;
            //B->val[ B->row[ j ] ] = A.val[k];
            // transpose dense block
            //for (int ii=0; ii < B->blocksize; ii++) {
            //    for (int jj=0; jj < B->blocksize; jj++) {
            //        B->val[B->row[j]*B->ldblock+ii*B->blocksize+jj] = A.val[k*B->ldblock+jj*B->blocksize+ii];
            //    }
            //}
            // do not tanspose dense block
            for ( int kk=0; kk< B->ldblock; kk++) {
              B->val[B->row[j]*B->ldblock+kk] = A.val[k*B->ldblock+kk];
            }
            B->row[ j ]++;
        }
    }
    if (A.pad_rows != 0) {
        //printf("total=%d A.true_nnz=%d\n", total, A.true_nnz);
        assert( B->row[ rowlimit-1 ] == A.numblocks );
    }
    else { 
        //printf("total=%d A.nnz=%d\n", total, A.nnz);
        assert( B->row[ rowlimit-1 ] == A.numblocks );
    }
    
    // shift colptr right one
    // colptr = [ 0 3 5 6, 8 ]
    for( j=rowlimit-1; j > 0; j-- ) {
        B->row[j] = B->row[j-1];
    }
    B->row[0] = 0;
    
    if ( workaround == 1 ) {
        // increment to create one-based indexing of the array parameters
        #pragma omp parallel 
        {
        #pragma omp for simd nowait
        for (data_int_t it=0; it<B->numblocks; it++) {
            A.col[it] += 1;
            B->col[it] += 1;
        }
        #pragma omp for simd nowait
        for (data_int_t it=0; it<rowlimit+1; it++) {
            A.row[it] += 1;	
            B->row[it] += 1;	
        }
        }
    } 
    
    
//cleanup:
    return info;
}

extern "C" 
int
z_transpose_dense(
    data_d_matrix A,
    data_d_matrix *B )
{
    int info = 0;
    
    B->storage_type = A.storage_type;
    B->fill_mode    = A.fill_mode;
    B->num_rows     = A.num_cols;
    B->num_cols     = A.num_rows;
    B->nnz          = A.nnz;
    B->true_nnz     = A.true_nnz;
    B->pad_rows     = A.pad_cols;
    B->pad_cols     = A.pad_rows;
    //B->ld           = A.ld;
    
    if ( B->val != NULL )
      free( B->val );
    //if (A.pad_rows > 0 && A.pad_cols > 0)
    //  B->val = (dataType*) malloc( A.pad_rows*A.pad_cols*sizeof(dataType) );
    //else
    //  B->val = (dataType*) malloc( A.num_rows*A.num_cols*sizeof(dataType) );
    
    int rowlimit = B->num_rows;
    int collimit = B->num_cols;
    if (A.pad_rows > 0 && A.pad_cols > 0) {
       rowlimit = B->pad_rows;
       collimit = B->pad_cols;
    }
    B->val = (dataType*) malloc( rowlimit*collimit*sizeof(dataType) );
    
    if (A.major == MagmaRowMajor) {
      for ( int i = 0; i < rowlimit; i++ ) {
        for ( int j = 0; j < collimit; j++ ) {
          B->val[ i + j*rowlimit ] = A.val[ i*collimit + j ];
        }
      }
      //if (A.pad_rows > 0 && A.pad_cols > 0) {
      //  for ( int i = 0; i < A.pad_rows; i++ ) {
      //    for ( int j = 0; j < A.pad_cols; j++ ) {
      //      B->val[ i + j*A.pad_rows ] = A.val[ i*A.pad_cols + j ];
      //    }
      //  }
      //}
      //else {
      //  for ( int i = 0; i < A.num_rows; i++ ) {
      //    for ( int j = 0; j < A.num_cols; j++ ) {
      //      B->val[ i + j*A.num_rows ] = A.val[ i*A.num_cols + j ];
      //    }
      //  }
      //}
      B->major = MagmaRowMajor;
      B->ld = rowlimit;
    }
    else if (A.major == MagmaColMajor) {
      for ( int j = 0; j < collimit; j++ ) {
        for ( int i = 0; i < rowlimit; i++ ) {
          B->val[ i*collimit + j ] = A.val[ i + j*rowlimit ];
        }
      }
      //if (A.pad_rows > 0 && A.pad_cols > 0) {
      //  for ( int j = 0; j < A.num_cols; j++ ) {
      //    for ( int i = 0; i < A.num_rows; i++ ) {
      //      B->val[ i*A.pad_cols + j ] = A.val[ i + j*A.pad_rows ];
      //    }
      //  }
      //}
      //else {
      //  for ( int j = 0; j < A.num_cols; j++ ) {
      //    for ( int i = 0; i < A.num_rows; i++ ) {
      //      B->val[ i*A.num_cols + j ] = A.val[ i + j*A.num_rows ];
      //    }
      //  }
      //}
      B->major = MagmaColMajor;
      B->ld = collimit;
      //printf("%s line %d\n", __FILE__, __LINE__);
    }

    //cleanup:
    return info;
}

/*******************************************************************************
    Purpose
    -------

    Interface to transpose.

    Arguments
    ---------

    @param[in]
    A           data_d_matrix
                input matrix (CSR)

    @param[out]
    B           data_d_matrix*
                output matrix (CSR)
    @param[in]
    queue       data_queue_t
                Queue to execute in.

    @ingroup datasparse_zaux
*******************************************************************************/
    
    
extern "C" int
data_zmtranspose(
    data_d_matrix A, data_d_matrix *B )
{
    data_int_t info = 0;
    
    if (A.storage_type == Magma_CSR 
      || A.storage_type == Magma_CSRD
      || A.storage_type == Magma_CSRL 
      || A.storage_type == Magma_CSRU 
      || A.storage_type == Magma_CSRCOO
      || A.storage_type == Magma_CSCD
      || A.storage_type == Magma_CSCL 
      || A.storage_type == Magma_CSCU 
      || A.storage_type == Magma_CSCCOO )
      z_transpose_csr( A, B );
    else if (A.storage_type == Magma_BCSR
      || A.storage_type == Magma_BCSRD
      || A.storage_type == Magma_BCSRL
      || A.storage_type == Magma_BCSRU
      || A.storage_type == Magma_BCSC
      || A.storage_type == Magma_BCSCD
      || A.storage_type == Magma_BCSCL
      || A.storage_type == Magma_BCSCU )
      z_transpose_bcsr( A, B );
    else if (A.storage_type == Magma_DENSE
      || A.storage_type == Magma_DENSEL
      || A.storage_type == Magma_DENSEU )
      z_transpose_dense( A, B );
    
//cleanup:
    return info;
}