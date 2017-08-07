/*
    -- LACE (version 0.0) --
       Univ. of Tennessee, Knoxville
       
       @author Stephen Wood

*/
#include "../include/sparse.h"
#include <mkl.h>
#include <float.h> 
#include <stdlib.h>
#include <stdio.h>

/**
    Purpose
    -------

    Assess the orthogonality of the upper-left search x search portion of a matrix.
    

    Arguments
    ---------

    @param[in]
    krylov      data_d_matrix*
                descriptor for matrix krylov

    @param[in,out]
    ortherr     dataType *
                Infinity norm of orthogonality error

    @param[in,out]
    imax        int*
                row where maximum row sum is found

    @param[in]
    search      int
                extent of the matrix to be assesed

    @ingroup datasparse_orthogonality
    ********************************************************************/


extern "C" 
void
data_orthogonality_error( data_d_matrix* krylov,  
  dataType* ortherr,
  int* imax,
  int search ) 
{
  
  dataType zero = 0.0;
  dataType one = 1.0;
  dataType negone = -1.0;
     
  data_d_matrix eye={Magma_DENSE};
  data_zvinit( &eye, search, search, zero );
  for (int e=0; e<eye.ld; e++) {
    eye.val[idx(e,e,eye.ld)] = 1.0; 
  }
  
  data_dgemm_mkl( krylov->major, MagmaTrans, MagmaNoTrans,
    search, search, search,
    one, krylov->val, krylov->ld,
    krylov->val, krylov->ld,
    negone, eye.val, eye.ld );
  
  data_infinity_norm( &eye, imax, ortherr ); 
  
  data_zmfree( &eye );
  
}
