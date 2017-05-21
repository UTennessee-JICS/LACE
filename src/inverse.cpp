/*
    -- LACE (version 0.0) --
       Univ. of Tennessee, Knoxville
       
       @author Stephen Wood

*/
#include "../include/sparse.h"
#include <mkl.h>
#include <stdlib.h>
#include <stdio.h>
#include <algorithm>
#include <ctime> 

extern "C"
int
data_inverse( data_d_matrix* A, data_d_matrix* Ainv )
{
  int info = 0;
  
  DEV_CHECKPT
  
  printf("A:\n");
  data_zdisplay_dense( A );
  
  Ainv->num_rows = A->num_rows;
  Ainv->num_cols = A->num_cols;
  Ainv->ld = A->num_cols;
  Ainv->nnz = A->nnz;
  LACE_CALLOC(Ainv->val, Ainv->nnz);
  printf("Ainv:\n");
  data_zdisplay_dense( Ainv );
  
  // parLU
  data_d_matrix L = {Magma_DENSE};
  data_d_matrix U = {Magma_DENSE};
  L.diagorder_type = Magma_UNITY;
  L.fill_mode = MagmaLower;
  U.diagorder_type = Magma_VALUE;
  U.fill_mode = MagmaUpper;
  data_ParLU_v1_0(A, &L, &U);
  printf("L:\n");
  data_zdisplay_dense( &L );
  printf("U:\n");
  data_zdisplay_dense( &U );
  
  // mxm identity matrix
  //data_d_matrix Eye = {Magma_DENSE};
  //Eye.num_rows = A->num_rows;
  //Eye.num_cols = A->num_cols;
  //Eye.ld = A->num_cols;
  //Eye.nnz = A->nnz;
  //LACE_CALLOC(Eye.val, Eye.num_rows);
  //for (int i=0; i<Eye.num_rows; i++ ) {
  //  Eye.val[i+i*Eye.ld] = 1.0; 
  //  //Ainv->val[i+i*Ainv->ld] = 1.0; 
  //}
  //printf("Eye:\n");
  //data_zdisplay_dense( &Eye );
  
  // data_partrsv_dot for each column of inverse
  dataType ptrsv_tol = 1.0e-15;
  int ptrsv_iter = 0;
  
  data_d_matrix y = {Magma_DENSE};
  y.num_rows = A->num_rows;
  y.num_cols = 1;
  y.ld = 1;
  y.nnz = y.num_rows;
  LACE_CALLOC(y.val, y.num_rows);
  
  
  data_d_matrix e = {Magma_DENSE};
  e.num_rows = A->num_rows;
  e.num_cols = 1;
  e.ld = 1;
  e.nnz = e.num_rows;
  LACE_CALLOC(e.val, e.num_rows);
  
  
  data_d_matrix f = {Magma_DENSE};
  f.num_rows = A->num_rows;
  f.num_cols = 1;
  f.ld = 1;
  f.nnz = f.num_rows;
  LACE_CALLOC(f.val, f.num_rows);
  
  L.diagorder_type = Magma_UNITY;
  L.fill_mode = MagmaLower;
  U.diagorder_type = Magma_VALUE;
  U.fill_mode = MagmaUpper;
  
  for ( int col=0; col<A->num_cols; col++) {
    e.val[col] = 1.0;
    //for ( int i=0; i<e.num_rows; i++ ) {
    //  printf("e[%d]=%e\n", i, e.val[i]);
    //}
    printf("col=%d forward ", col);
    data_partrsv_dot( MagmaRowMajor, MagmaLower, Magma_DENSEL, Magma_UNITY,
      L.num_rows, L.val, L.ld, e.val, 1, y.val, 1, 
      ptrsv_tol, &ptrsv_iter );
    printf("backward ");
    
    // TODO: inspect and correct indexing so that column vectors are writen correctly
    //data_partrsv_dot( MagmaRowMajor, MagmaUpper, Magma_DENSEU, Magma_VALUE,
    //  U.num_rows, U.val, U.ld, y.val, 1, &(Ainv->val[col]), Ainv->ld, 
    //  ptrsv_tol, &ptrsv_iter );
    
    data_partrsv_dot( MagmaRowMajor, MagmaUpper, Magma_DENSEU, Magma_VALUE,
      U.num_rows, U.val, U.ld, y.val, 1, f.val, 1, 
      ptrsv_tol, &ptrsv_iter );
    printf("done.\n");
    
    for ( int i=0; i<f.num_rows; i++ ) {
      //printf("f[%d]=%e\n", i, f.val[i]);
      Ainv->val[col + i*Ainv->ld] = f.val[i];
    }
    
    e.val[col] = 0.0;
    
    //printf("Ainv:\n");
    //data_zdisplay_dense( Ainv );
  }
  
  
  
  
  printf("Ainv:\n");
  data_zdisplay_dense( Ainv );
  
  //for ( int col=0; col<A->num_cols; col++) {
  //  data_partrsv_dot( MagmaRowMajor, MagmaLower, Magma_DENSEL, Magma_UNITY,
  //    L.num_rows, L.val, L.ld, Ainv->val[col*Ainv->ld], 1, y.val, 1, 
  //    ptrsv_tol, &ptrsv_iter );
  //  data_partrsv_dot( MagmaRowMajor, MagmaUpper, Magma_DENSEU, Magma_VALUE,
  //    U.num_rows, U.val, U.ld, y.val, 1, Eye.val[col*Eye.ld], 1, 
  //    ptrsv_tol, &ptrsv_iter );
  //}
  //
  //Eye.major = MagmaColMajor;
  //Ainv->major = MagmaRowMajor;
  //data_zmconvert( Eye, Ainv, Magma_DENSE, Magma_DENSE );
  
  
  data_zmfree( &L );
  data_zmfree( &U );
  data_zmfree( &e );
  data_zmfree( &y );
  
  return info;
}