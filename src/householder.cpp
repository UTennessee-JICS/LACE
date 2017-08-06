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

    Generate Householder reflection.
    G. W. Stewart, "Matrix Algorithms, Volume 1", SIAM, 1998.
    [u,nu] = housegen(x).
    H = I - uu' with Hx = -+ nu e_1
    nu = norm(x).  

    Arguments
    ---------

    @param[in]
    n           int
                number of elements in X

    @param[in,out]
    X           dataType *
                vector to be reflected

    @param[in,out]
    u           dataType *
                reflected vector

    @param[in]
    nu          dataType *
                norm of X

    @ingroup datasparse_orthogonality
    ********************************************************************/


extern "C" 
void
data_housegen( int n,
  dataType* X,  
  dataType* u,
  dataType* nu ) 
{
  
  dataType eps = nextafter(0.0,1.0);
    
  //LACE_CALLOC( u, n );
  for (int i=0; i<n; i++) {
    u[i] = X[i];
  }
  
  *nu = data_dnrm2( n, X, 1 );
  if ( (*nu) < eps ) {
    u[0] = sqrt(2);
    return;
  }
  for (int i=0; i<n; i++) {
    u[i] = X[i]/(*nu);
  }
  if ( u[0] >= 0 ) {
     u[0] = u[0] + 1.0;
     (*nu) = -(*nu);
  }
  else {
    u[0] = u[0] - 1.0; 
  }
  dataType tmp = 1.0/sqrt(fabs(u[0]));
  for (int i=0; i<n; i++) {
    u[i] = u[i]*tmp;
  }
  
}


/**
    Purpose
    -------

    Householder triangularization.  [U,R] = hqrd(X);
    Generators of Householder reflections stored in U->
    H_k = I - U(:,k)*U(:,k)'.
    prod(H_m ... H_1)X = [R; 0]
    where m = min(size(X))
    G. W. Stewart, "Matrix Algorithms, Volume 1", SIAM, 1998.

    Arguments
    ---------

    @param[in]
    n           int
                number of rows elements in X

    @param[in,out]
    X           dataType *
                vector to be reflected

    @param[in,out]
    u           dataType *
                reflected vector

    @param[in]
    nu          dataType *
                norm of X

    @ingroup datasparse_orthogonality
    ********************************************************************/


extern "C" 
void
data_hqrd( data_d_matrix* X,  
  data_d_matrix* U,
  data_d_matrix* R ) 
{
  printf("data_hqrd begin\n X\n");
  data_zdisplay_dense( X );
  
  int length = 0;
  int n = X->num_rows;
  int p = X->num_cols;
  dataType* v;
  LACE_CALLOC( v, n );
  for ( int k=0; k<MIN(n,p); k++ ) {
    length = n - k;
    data_housegen( length, &(X->val[idx(k,k,X->ld)]), &(U->val[idx(k,k,U->ld)]), &(R->val[idx(k,k,R->ld)]) );
    for ( int j=k+1; j<p; j++ ) {
      dataType colsum = 0.0;
      for ( int i=k; i<n; i++ ) {
        printf("U[%d,%d] = %e X[%d,%d] = %e\n", i, k, U->val[idx(i,k,U->ld)], i, j, X->val[idx(i,j,X->ld)] );
        colsum = colsum + U->val[idx(i,k,U->ld)] * X->val[idx(i,j,X->ld)];
      }
      v[j] = colsum;
      printf("k=%d v[%d] = %e\n", k, j, v[j] );
    }
    for ( int i=k; i<n; i++ ) {
      length = n - k;
      for ( int j=k+1; j<p; j++ ) {
        X->val[idx(i,j,X->ld)] = X->val[idx(i,j,X->ld)] - U->val[idx(i,k,U->ld)] * v[j];
        //R->val[idx(i,j,X->ld)] = X->val[idx(i,j,X->ld)];
      }
    }
    for ( int j=k+1; j<p; j++ ) {
      printf("R[%d,%d] = %e\n", k, j, R->val[idx(k,j,X->ld)] );
      //X->val[idx(i,j,X->ld)] = X->val[idx(i,j,X->ld)] - U->val[idx(i,k,U->ld)] * v[j];
      R->val[idx(k,j,X->ld)] = X->val[idx(k,j,X->ld)];
    }
    
  }
  
}

extern "C"
void
data_housegen_matrix( int n,
  dataType* u,
  int uld,
  dataType* x,
  int xld,
  dataType* H,
  int Hld )
{
  dataType negone = -1.0;
  dataType zero = 0.0;
  dataType one = 1.0;
  dataType* tmp;
  LACE_CALLOC( tmp, n );
  dataType tmp2;
  //  H = @(u,x) x - u*(u'*x);
  
  // u'*x
  for (int i=0; i<n; i++) {
    for (int j=0; j<n; j++) {
      tmp[i] = tmp[i] + u[j]*x[idx(j,i,xld)];
    }
  }
  
  for (int i=0; i<n; i++) {
    printf("u[%d] = %e tmp[%d] = %e\n", i, u[i], i, tmp[i]);
  } 
  
  // x - u*(u'*x)
  printf("temp2=\n");
  for (int i=0; i<n; i++) {
    for (int j=0; j<n; j++) {
      H[idx(i,j,Hld)] = x[idx(i,j,xld)] - u[i]*tmp[j];
      printf("%e  ", u[i]*tmp[j]);
    }
    printf("\n");
  }
  
}
