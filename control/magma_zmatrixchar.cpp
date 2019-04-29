/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
       @author Hartwig Anzt
       @author Stephen Wood
*/
#include <stdio.h>
#include "../include/sparse_types.h"

#define THRESHOLD 10e-99



/**
    Purpose
    -------

    Checks the maximal number of nonzeros in a row of matrix A.
    Inserts the data into max_nnz_row.


    Arguments
    ---------

    @param[in,out]
    A           data_d_matrix*
                sparse matrix

    @ingroup datasparse_zaux
    ********************************************************************/

extern "C"
int
data_zrowentries(
    data_d_matrix *A )
{
    int info = 0;

    int *length=NULL;
    int i,j, maxrowlength=0;

    // CSR
    if ( A->storage_type == Magma_CSR || A->storage_type == Magma_CSC ) {
        // length = (int*) malloc( A->num_rows*sizeof(int) );
        LACE_CALLOC( length, A->num_rows );
        for( i=0; i<A->num_rows; i++ ) {
            length[i] = A->row[i+1]-A->row[i];
            if (length[i] > maxrowlength)
                 maxrowlength = length[i];
        }
        A->max_nnz_row = maxrowlength;
    }
    // Dense
    else if ( A->storage_type == Magma_DENSE ) {
        //length = (int*) malloc( A->num_rows*sizeof(int) );
        LACE_CALLOC( length, A->num_rows );

        for( i=0; i<A->num_rows; i++ ) {
            length[i] = 0;
            for( j=0; j<A->num_cols; j++ ) {
                if ( ( A->val[i*A->num_cols + j] ) != 0. )
                    length[i]++;
                }
            if (length[i] > maxrowlength)
                 maxrowlength = length[i];
        }
        A->max_nnz_row = maxrowlength;
    }

//cleanup:
    free( length );
    return info;
}


/**
    Purpose
    -------

    Computes the diameter of a sparse matrix and stores the value in diameter.


    Arguments
    ---------

    @param[in,out]
    A           data_d_matrix*
                sparse matrix

    @ingroup datasparse_zaux
    ********************************************************************/
extern "C"
int
data_zdiameter(
    data_d_matrix *A )
{
    int info = 0;

    int i, j, tmp,  *dim=NULL, maxdim=0;

    // CSR, BCSR
    if ( A->storage_type == Magma_CSR || A->storage_type == Magma_CSC ||
	 A->storage_type == Magma_BCSR || A->storage_type == Magma_BCSC ) {
         //dim = (int*) malloc( A->num_rows*sizeof(int) );

        LACE_CALLOC( dim, A->num_rows );
        for( i=0; i<A->num_rows; i++ ) {
            dim[i] = 0;
            for( j=A->row[i]; j<A->row[i+1]; j++ ) {
                 tmp = abs( i - A->col[j] );
                 if ( tmp > dim[i] )
                     dim[i] = tmp;
            }
            if ( dim[i] > maxdim )
                 maxdim = dim[i];
        }
       A->diameter = maxdim;
   }
    // Dense
    else if ( A->storage_type == Magma_DENSE ) {
        // dim = (int*) malloc( A->num_rows*sizeof(int) );
        LACE_CALLOC( dim, A->num_rows );
        for( i=0; i<A->num_rows; i++ ) {
            dim[i] = 0;
            for( j=0; j<A->num_cols; j++ ) {
                if ( ( A->val[i*A->num_cols + j] ) !=  0.0 ) {
                    tmp = abs( i -j );
                    if ( tmp > dim[i] )
                        dim[i] = tmp;
                }
            }
            if ( dim[i] > maxdim )
                 maxdim = dim[i];
        }
        A->diameter = maxdim;
    }
    // ELLPACK
    else if ( A->storage_type == Magma_ELLPACKT ) {
        // dim = (int*) malloc( A->num_rows*sizeof(int) );
        LACE_CALLOC( dim, A->num_rows );
        for( i=0; i<A->num_rows; i++ ) {
            dim[i] = 0;
            for( j=i*A->max_nnz_row; j<(i+1)*A->max_nnz_row; j++ ) {
                if ( ( A->val[j] ) > THRESHOLD ) {
                    tmp = abs( i - A->col[j] );
                    if ( tmp > dim[i] )
                        dim[i] = tmp;
                }
            }
            if ( dim[i] > maxdim )
                 maxdim = dim[i];
        }
        A->diameter = maxdim;
    }
    // ELL
    else if ( A->storage_type == Magma_ELL ) {
        printf("error:format not supported.\n");
        info = DEV_ERR_NOT_SUPPORTED;
    }
//cleanup:
    free( dim );
    return info;
}
