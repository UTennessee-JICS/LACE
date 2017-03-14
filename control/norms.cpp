
#include <stdio.h>
#include "../include/sparse.h"


extern "C" 
int
data_zfrobenius_csr(
    data_d_matrix A,
    dataType *res )
{
    int i,j;
    *res = 0.0;
    
    for (i=0; i<A.num_rows; i++) {
        for (j=A.row[i]; j<A.row[i+1]; j++) {
            (*res) = (*res) + A.val[j] * A.val[j];
        }
    }

    (*res) =  sqrt((*res));

    return DEV_SUCCESS;
}


extern "C" 
int
data_zfrobenius_dense(
    data_d_matrix A,
    dataType *res )
{
    int i,j;
    *res = 0.0;
    
    if (A.major == MagmaRowMajor ) {
      for (i=0; i<A.num_rows; i++) {
        for (j=0; j<A.num_cols; j++) {
          (*res) = (*res) + A.val[ i*A.ld + j ] * A.val[ i*A.ld + j ];
        }
      }
    }
    else {
      for (j=0; j<A.num_cols; j++) {
        for (i=0; i<A.num_rows; i++) {
          (*res) = (*res) + A.val[ i + j*A.ld ] * A.val[ i + j*A.ld ];
        }
      }
    }

    (*res) =  sqrt((*res));

    return DEV_SUCCESS;
}

extern "C" 
int
data_zfrobenius(
    data_d_matrix A,
    dataType *res )
{
  int info = 0;
  if ( A.storage_type == Magma_CSR 
      || A.storage_type == Magma_CSRD
      || A.storage_type == Magma_CSRL 
      || A.storage_type == Magma_CSRU 
      || A.storage_type == Magma_CSRCOO
      || A.storage_type == Magma_CSCD
      || A.storage_type == Magma_CSCL 
      || A.storage_type == Magma_CSCU 
      || A.storage_type == Magma_CSCCOO ) {
    info = data_zfrobenius_csr(A, res); 
  }
  else if ( A.storage_type == Magma_DENSE
      || A.storage_type == Magma_DENSEL
      || A.storage_type == Magma_DENSEU ) {
    info = data_zfrobenius_dense(A, res);    
  }
  
  return info;
}


extern "C" 
int
data_zfrobenius_diff_csr(
    data_d_matrix A,
    data_d_matrix B,
    dataType *res )
{
    dataType tmp2;
    int i,j,k;
    *res = 0.0;
    
    for (i=0; i<A.num_rows; i++) {
        for (j=A.row[i]; j<A.row[i+1]; j++) {
            int localcol = A.col[j];
            for( k=B.row[i]; k<B.row[i+1]; k++){
                if(B.col[k] == localcol){
                    tmp2 = A.val[j] - B.val[k];
                    (*res) = (*res) + tmp2 * tmp2;
                }
            }
        }
    }

    (*res) =  sqrt((*res));

    return DEV_SUCCESS;
}

extern "C" 
int
data_zfrobenius_diff_dense(
    data_d_matrix A,
    data_d_matrix B,
    dataType *res )
{
    dataType tmp2;
    int i,j;
    *res = 0.0;
    
    if (A.major == MagmaRowMajor ) {
      for (i=0; i<A.num_rows; i++) {
        for (j=0; j<A.num_cols; j++) {
          tmp2 = A.val[ i*A.ld + j ] - B.val[ i*A.ld + j ];
          (*res) = (*res) + tmp2 * tmp2;
        }
      }
    }
    else {
      for (j=0; j<A.num_cols; j++) {
        for (i=0; i<A.num_rows; i++) {
          tmp2 = A.val[ i + j*A.ld ] - B.val[ i + j*A.ld ];
          (*res) = (*res) + tmp2 * tmp2;
        }
      }
    }

    (*res) =  sqrt((*res));

    return DEV_SUCCESS;
}

extern "C" 
int
data_zfrobenius_diff(
    data_d_matrix A,
    data_d_matrix B,
    dataType *res )
{
  int info = 0;
  if ( A.storage_type == Magma_CSR 
      || A.storage_type == Magma_CSRD
      || A.storage_type == Magma_CSRL 
      || A.storage_type == Magma_CSRU 
      || A.storage_type == Magma_CSRCOO
      || A.storage_type == Magma_CSCD
      || A.storage_type == Magma_CSCL 
      || A.storage_type == Magma_CSCU 
      || A.storage_type == Magma_CSCCOO ) {
    info = data_zfrobenius_diff_csr(A, B, res); 
  }
  else if ( A.storage_type == Magma_DENSE
      || A.storage_type == Magma_DENSEL
      || A.storage_type == Magma_DENSEU ) {
    info = data_zfrobenius_diff_dense(A, B, res);    
  }
  
  return info;
}

extern "C" 
int
data_zfrobenius_LUresidual( 
  data_d_matrix A,
  data_d_matrix L,
  data_d_matrix U,
  dataType *res)
{
  
  // fill diagonal of L
  if (L.diagorder_type == Magma_NODIAG) {
    #pragma omp parallel  
    #pragma omp for nowait
    for (int i=0; i<A.num_rows; i++) {
      L.val[ i*A.ld + i ] = 1.0;
      //U.val[ i*A.ld + i ] = D.val[ i ];
    }
  }
  
  // Check ||A-LU||_Frobenius
  dataType alpha = 1.0;
  dataType beta = 0.0;
  data_d_matrix B = {Magma_DENSE};
  B.num_rows = A.num_rows;
  B.num_cols = A.num_cols;
  B.ld = A.ld;
  B.nnz = A.nnz;
  B.val = (dataType*) calloc( B.num_rows*B.num_cols, sizeof(dataType) );
  if (U.major == MagmaRowMajor) {
    data_dgemm_mkl( L.major, MagmaNoTrans, MagmaNoTrans, 
      A.num_rows, A.num_rows, A.num_cols, 
      alpha, L.val, L.ld, U.val, U.ld, 
      beta, B.val, B.ld );
  }
  else {
    data_d_matrix C = {Magma_DENSEU};
    C.major = MagmaRowMajor;
    //C.diagorder_type = U.diagorder_type;
    C.diagorder_type = Magma_VALUE;
    //printf("before data_zmconvert(U, &C, Magma_DENSEU, Magma_DENSEU);\n");
    //data_zdisplay_dense( &U );
    data_zmconvert(U, &C, Magma_DENSEU, Magma_DENSEU);
    //data_zmconvert(U, &C, Magma_DENSE, Magma_DENSE);
    //printf("after data_zmconvert(U, &C, Magma_DENSEU, Magma_DENSEU);\n");
    //data_zdisplay_dense( &C );
    data_dgemm_mkl( L.major, MagmaNoTrans, MagmaNoTrans, 
      A.num_rows, A.num_rows, A.num_cols, 
      alpha, L.val, L.ld, C.val, C.ld, 
      beta, B.val, B.ld );
      data_zmfree( &C );
  }
  data_zfrobenius_diff(A, B, res);
  
  data_zmfree( &B );
  return DEV_SUCCESS;
}

extern "C" 
int
data_zfrobenius_inplaceLUresidual( 
  data_d_matrix A,
  data_d_matrix LU,
  dataType *res)
{
  
  // Separate L and U
  data_d_matrix L = {Magma_DENSEL};
  L.diagorder_type = Magma_UNITY;
  data_zmconvert(LU, &L, Magma_DENSE, Magma_DENSEL);
  //data_zdisplay_dense( &L );
  
  data_d_matrix U = {Magma_DENSEU};
  U.diagorder_type = Magma_VALUE;
  data_zmconvert(LU, &U, Magma_DENSE, Magma_DENSEU);
  
  // Check ||A-LU||_Frobenius
  dataType alpha = 1.0;
  dataType beta = 0.0;
  data_d_matrix B = {Magma_DENSE};
  B.num_rows = A.num_rows;
  B.num_cols = A.num_cols;
  B.ld = A.ld;
  B.nnz = A.nnz;
  B.val = (dataType*) calloc( B.num_rows*B.num_cols, sizeof(dataType) );
  data_dgemm_mkl( L.major, MagmaNoTrans, MagmaNoTrans, 
    A.num_rows, A.num_rows, A.num_cols, 
    alpha, L.val, L.ld, U.val, U.ld, 
    beta, B.val, B.ld );
  data_zfrobenius_diff(A, B, res);
  
  data_zmfree( &L );
  data_zmfree( &U );
  data_zmfree( &B );
  return DEV_SUCCESS;
}


int
data_zilures(
    data_d_matrix A,
    data_d_matrix L,
    data_d_matrix U,
    data_d_matrix *LU,
    dataType *res,
    dataType *nonlinres )
{
	int info = 0;
  *res = 0.0;
  *nonlinres = 0.0;

    dataType tmp;
    int i, j, k;
    
    dataType one = 1.0;

    data_d_matrix LL={Magma_CSR}, L_d={Magma_CSR}, U_d={Magma_CSR}, LU_d={Magma_CSR};
    
    if( L.row[1]==1 ){        // lower triangular with unit diagonal
    	//printf("L lower triangular.\n");
        LL.diagorder_type = Magma_UNITY;
        data_zmconvert( L, &LL, Magma_CSR, Magma_CSRL );
    }
    else if ( L.row[1]==0 ){ // strictly lower triangular
    	printf("L strictly lower triangular.\n");
        data_zmconvert( L, &LL, Magma_CSR, Magma_CSR );
        free( LL.col );
        free( LL.val );
        LL.nnz = L.nnz+L.num_rows;
        LL.val = (dataType*) malloc( LL.nnz*sizeof(dataType) );
        LL.col = (int*) malloc( LL.nnz*sizeof(int) );
        int z=0;
        for (i=0; i < L.num_rows; i++) {
            LL.row[i] = z;
            for (j=L.row[i]; j < L.row[i+1]; j++) {
                LL.val[z] = L.val[j];
                LL.col[z] = L.col[j];
                z++;
            }
            // add unit diagonal
            LL.val[z] = 1.0;
            LL.col[z] = i;
            z++;
        }
        LL.row[LL.num_rows] = z;
    }
    else {
        printf("error: L neither lower nor strictly lower triangular!\n");
    }

    data_zmconvert( LL, &L_d, Magma_CSR, Magma_CSR );
    if ( U.storage_type == Magma_CSR || U.storage_type == Magma_CSRU ) {
        //printf("U is CSR\n");
        data_zmconvert( U, &U_d, Magma_CSR, Magma_CSR );
    }
    else if ( U.storage_type == Magma_CSC || U.storage_type == Magma_CSCU ) {
        //printf("U is CSC\n");
        data_zmconvert( U, &U_d, Magma_CSC, Magma_CSR );
    }
    data_zmfree( &LL );
    
    data_z_spmm( one, L_d, U_d, &LU_d );
    data_zmconvert( LU_d, LU, Magma_CSR, Magma_CSR );
    data_zrowentries( LU );
    data_zdiameter( LU );
    
    data_zmfree( &L_d );
    data_zmfree( &U_d );
    data_zmfree( &LU_d );

    // compute Frobenius norm of A-LU
    for(i=0; i<A.num_rows; i++){
    	for(j=A.row[i]; j<A.row[i+1]; j++){
            int lcol = A.col[j];
            for(k=LU->row[i]; k<LU->row[i+1]; k++){
                if( LU->col[k] == lcol ){
                    tmp =  LU->val[k] -  A.val[j];
                    LU->val[k] = tmp;
                    (*nonlinres) = (*nonlinres) + tmp*tmp;
                    break;
                }
            }
        }
    }

    for(i=0; i<LU->num_rows; i++){
        for(j=LU->row[i]; j<LU->row[i+1]; j++){
            tmp = LU->val[j];
            (*res) = (*res) + tmp * tmp;
        }
    }

    (*res) =  sqrt((*res));
    (*nonlinres) =  sqrt((*nonlinres));

//cleanup:
    if( info !=0 ){
        data_zmfree( LU );
    }
    return info;
}

extern "C" 
int
data_maxfabs_csr(
    data_d_matrix A,
    int *imax,
    int *jmax,
    dataType *max )
{
    int i,j;
    *max = 0.0;
    dataType tmp = 0.0;
    
    for (i=0; i<A.num_rows; i++) {
        for (j=A.row[i]; j<A.row[i+1]; j++) {
            tmp = fabs( A.val[j] );
            if ( (*max) < tmp ) {
              (*max) = tmp;
              (*imax) = i;
              (*jmax) = A.col[j];
            }
        }
    }

    return DEV_SUCCESS;
}