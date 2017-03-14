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
#include "../include/sparse.h"

/**
    Purpose
    -------

    Scales a matrix.

    Arguments
    ---------

    @param[in,out]
    A           data_d_matrix*
                input/output matrix

    @param[in]
    scaling     data_scale_t
                scaling type (unit rownorm / unit diagonal)

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" 
int
data_zmscale(
    data_d_matrix *A,
    data_scale_t scaling )
{
    data_int_t info = 0;
    
    dataType *tmp=NULL;
    
    data_d_matrix CSRA={Magma_CSR};
    
    if( A->num_rows != A->num_cols && scaling != Magma_NOSCALE ){
        printf("%% warning: non-square matrix.\n");
        printf("%% Fallback: no scaling.\n");
        scaling = Magma_NOSCALE;
    } 
        
   
    if ( A->storage_type == Magma_CSRCOO ) {
        if ( scaling == Magma_NOSCALE ) {
            // no scale
            ;
        }
        else if( A->num_rows == A->num_cols ){
            if ( scaling == Magma_UNITROW ) {
                // scale to unit rownorm
                tmp = (dataType*) calloc( A->num_rows, sizeof(dataType) );
                for( data_int_t z=0; z<A->num_rows; z++ ) {
                    dataType s = 0.0;
                    for( data_int_t f=A->row[z]; f<A->row[z+1]; f++ )
                        s+= A->val[f]*A->val[f];
                    tmp[z] = 1.0/sqrt( s );
                }        
                for( data_int_t z=0; z<A->nnz; z++ ) {
                    A->val[z] = A->val[z] * tmp[A->col[z]] * tmp[A->rowidx[z]];
                }
            }
            else if (scaling == Magma_UNITDIAG ) {
                // scale to unit diagonal
                tmp = (dataType*) calloc( A->num_rows, sizeof(dataType) );
                for( data_int_t z=0; z<A->num_rows; z++ ) {
                    dataType s = 0.0;
                    for( data_int_t f=A->row[z]; f<A->row[z+1]; f++ ) {
                        if ( A->col[f]== z ) {
                            // add some identity matrix
                            //A->val[f] = A->val[f] +  MAGMA_Z_MAKE( 100000.0, 0.0 );
                            s = A->val[f];
                        }
                    }
                    if ( s == 0.0 ){
                        printf("%%error: zero diagonal element.\n");
                        info = DEV_ERR;
                    }
                    tmp[z] = 1.0/sqrt( s );
                }
                for( data_int_t z=0; z<A->nnz; z++ ) {
                    A->val[z] = A->val[z] * tmp[A->col[z]] * tmp[A->rowidx[z]];
                }
            }
            else {
                printf( "%%error: scaling not supported.\n" );
                info = DEV_ERR_NOT_SUPPORTED;
            }
        }
        else {
            printf( "%%error: scaling not supported.\n" );
            info = DEV_ERR_NOT_SUPPORTED;
        }
    }
    else {
        data_storage_t A_storage = A->storage_type;
        data_zmconvert( *A, &CSRA, A->storage_type, Magma_CSRCOO );

        data_zmscale( &CSRA, scaling );

        data_zmfree( A );
        data_zmconvert( CSRA, A, Magma_CSRCOO, A_storage );
    }
    
//cleanup:
    free( tmp );
    data_zmfree( &CSRA );
    return info;
}


/**
    Purpose
    -------

    Adds a multiple of the Identity matrix to a matrix: A = A+add * I

    Arguments
    ---------

    @param[in,out]
    A           data_d_matrix*
                input/output matrix

    @param[in]
    add         dataType
                scaling for the identity matrix
                
    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" 
int
data_zmdiagadd(
    data_d_matrix *A,
    dataType add )
{
    data_int_t info = 0;
    
    data_d_matrix CSRA={Magma_CSR};
    
    if ( A->storage_type == Magma_CSRCOO ) {
        for( data_int_t z=0; z<A->nnz; z++ ) {
            if ( A->col[z]== A->rowidx[z] ) {
                // add some identity matrix
                A->val[z] = A->val[z] +  add;
            }
        }
    }
    else {
        data_storage_t A_storage = A->storage_type;
        data_zmconvert( *A, &CSRA, A->storage_type, Magma_CSRCOO );

        data_zmdiagadd( &CSRA, add );

        data_zmfree( A );
        data_zmconvert( CSRA, A, Magma_CSRCOO, A_storage );
    }
    
//cleanup:
    data_zmfree( &CSRA );
    return info;
}
