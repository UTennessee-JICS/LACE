
#include <stdio.h>
#include "../include/sparse.h"

extern "C"
dataType
data_dnrm2(
    data_int_t n,
    dataType* x,
    data_int_t incx )
{
    return cblas_dnrm2( n, x, incx);
}



extern "C"
int
data_zfrobenius_csr(
    data_d_matrix A,
    dataType *res )
{
  int i,j,k;
    *res = 0.0;

    if(A.storage_type == Magma_BCSR){

       for (i=0; i<A.num_rows; i++) {
           for (j=A.row[i]; j<A.row[i+1]; j++) {
	     for (k=0; k< A.ldblock; ++k ) {
               (*res) = (*res) + A.val[j*A.ldblock+k] * A.val[j*A.ldblock+k];
	     }
           }
       }

    }
    else{

       for (i=0; i<A.num_rows; i++) {
           for (j=A.row[i]; j<A.row[i+1]; j++) {
               (*res) = (*res) + A.val[j] * A.val[j];
           }
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
      || A.storage_type == Magma_CSCCOO 
      || A.storage_type == Magma_BCSR ) {
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

  int rowlimit = A.num_rows;
  int collimit = A.num_cols;
  if (A.pad_rows > 0 && A.pad_cols > 0) {
     rowlimit = A.pad_rows;
     collimit = A.pad_cols;
  }

  // fill diagonal of L
  if (L.diagorder_type == Magma_NODIAG) {
    #pragma omp parallel
    #pragma omp for nowait
    for (int i=0; i<rowlimit; i++) {
      L.val[ i*A.ld + i ] = 1.0;
    }
  }

  // Check ||A-LU||_Frobenius
  dataType alpha = 1.0;
  dataType beta = 0.0;
  data_d_matrix B = {Magma_DENSE};
  B.num_rows = A.num_rows;
  B.num_cols = A.num_cols;
  B.pad_rows = A.pad_rows;
  B.pad_cols = A.pad_cols;
  B.ld = A.ld;
  B.nnz = A.nnz;
  // B.val = (dataType*) calloc( rowlimit*collimit, sizeof(dataType) );
  LACE_CALLOC( B.val, (rowlimit*collimit) );
  if (U.major == MagmaRowMajor) {
    data_dgemm_mkl( L.major, MagmaNoTrans, MagmaNoTrans,
      rowlimit, rowlimit, collimit,
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
      rowlimit, rowlimit, collimit,
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

  int rowlimit = A.num_rows;
  int collimit = A.num_cols;
  if (A.pad_rows > 0 && A.pad_cols > 0) {
     rowlimit = A.pad_rows;
     collimit = A.pad_cols;
  }

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
  B.pad_rows = A.pad_rows;
  B.pad_cols = A.pad_cols;
  B.ld = A.ld;
  B.nnz = A.nnz;
  // B.val = (dataType*) calloc( B.num_rows*B.num_cols, sizeof(dataType) );
  LACE_CALLOC( B.val, (rowlimit*collimit) );
  data_dgemm_mkl( L.major, MagmaNoTrans, MagmaNoTrans,
    rowlimit, rowlimit, collimit,
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
        LACE_CALLOC( LL.val, LL.nnz );
        LACE_CALLOC( LL.col, LL.nnz );
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
    //include only LU elements that correspond with nonzero elements in A
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

    // compute Frobenius norm of A-LU
    //include all LU elements
    for(i=0; i<LU->num_rows; i++){
        #pragma nounroll
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
data_zilures_bcsr(
    data_d_matrix A,
    data_d_matrix L,
    data_d_matrix U,
    data_d_matrix *LU,
    dataType *res,
    dataType *nonlinres )
{
  //multiply L and U into single matrix LU, subtract A from LU and return the l2 norm of the sum
  // of the elements of the subtraction.
  int info = 0;
  *res = 0.0;
  *nonlinres = 0.0;
  dataType* tmp = NULL;
  //dataType tmp[A.ldblock];
  //printf("A.ldblock=%d\n",A.ldblock);
  LACE_CALLOC(tmp,A.ldblock);
  int i, j, k;
  dataType one = 1.0;
  
  data_d_matrix LL={Magma_BCSR}, L_d={Magma_BCSR}, U_d={Magma_BCSR}, LU_d={Magma_BCSR};
  
  if( L.row[1]==1 ){        // lower triangular with unit diagonal
    //printf("L lower triangular.\n");
    LL.diagorder_type = Magma_UNITY;
    data_zmconvert( L, &LL, Magma_BCSR, Magma_BCSRL );
  }
  else if ( L.row[1]==0 ){ // strictly lower triangular
    //printf("L strictly lower triangular.\n");
    data_zmconvert( L, &LL, Magma_BCSR, Magma_BCSR );
    free( LL.col );
    free( LL.val );
    LL.nnz = L.nnz+L.num_rows;
    LACE_CALLOC( LL.val, LL.nnz*LL.ldblock );
    LACE_CALLOC( LL.col, LL.nnz );
    int z=0;
    for (i=0; i < L.num_rows; i++) {
      LL.row[i] = z;
      for (j=L.row[i]; j < L.row[i+1]; j++) {
	for(int zz=0; zz<LL.ldblock; ++zz){LL.val[z*LL.ldblock+zz] = L.val[j*LL.ldblock+zz];}
	LL.col[z] = L.col[j];
	z++;
      }
      // add unit diagonal
      for(int zz=0; zz<LL.blocksize;++zz){LL.val[z*LL.ldblock+zz*LL.blocksize+zz] = 1.0;}
      LL.col[z] = i;
      z++;
    }
    LL.row[LL.num_rows] = z;
  }
  else {
    printf("error: L neither lower nor strictly lower triangular!\n");
  }
  
  data_zmconvert( LL, &L_d, Magma_BCSR, Magma_BCSR );

 if ( U.storage_type == Magma_BCSR || U.storage_type == Magma_BCSRU ) {
    //printf("U is BCSR\n");
    data_zmconvert( U, &U_d, Magma_BCSR, Magma_BCSR );
  }
  else if ( U.storage_type == Magma_BCSC || U.storage_type == Magma_BCSCU ) {
    //printf("U is BCSC\n");
    data_zmconvert( U, &U_d, Magma_BCSC, Magma_BCSR );
  }
  
  data_zmfree( &LL );
  data_z_spmm( one, L_d, U_d, &LU_d );
  data_zmconvert( LU_d, LU, Magma_BCSR, Magma_BCSR );
  data_zrowentries( LU );
  data_zdiameter( LU );
  
  data_zmfree( &L_d );
  data_zmfree( &U_d );
  data_zmfree( &LU_d );

  // Compute Nonlinear Residual of A-LU
  // Compute Frobenius norm of A-LU
  // Only computes difference between values in A and in LU where fill pattern overlaps 
  for(i=0; i<A.num_rows; i++){
    for(j=A.row[i]; j<A.row[i+1]; j++){
      int lcol = A.col[j];
      for(k=LU->row[i]; k<LU->row[i+1]; k++){
	//if there is a matching row,col pair in A
	if( LU->col[k] == lcol ){
	  //tmp =  LU->val[k] -  A.val[j];
	  for(int kk=0; kk< A.ldblock; ++kk){
	    tmp[kk]= LU->val[k*A.ldblock+kk]-A.val[j*A.ldblock+kk];
	  }
	  for(int kk=0; kk< A.ldblock; ++kk){
	    LU->val[k*A.ldblock+kk] = tmp[kk];
	  }
	  for(int kk=0; kk< A.ldblock; ++kk){
	    (*nonlinres) = (*nonlinres) + tmp[kk]*tmp[kk];
	  }
	  break;
	}
      }
    }
  }

  // Compute Normal Residual of A-LU
  // Compute Frobenius norm of A-LU
  // Adds norm of LU values where fill does not overlap with A
  // This 
  for(i=0; i<LU->num_rows; i++){
    for(j=LU->row[i]; j<LU->row[i+1]; j++){
      //tmp = LU->val[j];
      for(int kk=0; kk< A.ldblock; ++kk){
	tmp[kk]= LU->val[j*A.ldblock+kk];
      }
      for(int kk=0; kk< A.ldblock; ++kk){(*res) = (*res) + tmp[kk] * tmp[kk];}
    }
  }
  
  (*res) =  sqrt((*res));
  (*nonlinres) =  sqrt((*nonlinres));
  //cleanup:
  free(tmp);
  if( info !=0 ){
    data_zmfree( LU );
  }
  return info;
}


//ceb infinity norm for dense or elemental sparse matrix
extern "C"
int
data_infinity_norm(
  data_d_matrix *A,
  int *imax,
  dataType *max )
{
  int i,j;
  *max = 0.0;
  dataType tmp = 0.0;

  for (i=0; i<A->num_rows; i++) {
    tmp = 0.0;
    for (j=0; j<A->num_cols; j++) {
      tmp += fabs( A->val[i*A->ld+j] );
    }
    if ( (*max) < tmp ) {
      (*max) = tmp;
      (*imax) = i;
    }
  }

  return DEV_SUCCESS;
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
	  //tmp = fabs( A.val[j] );
	  for(int kk=0;kk<A.ldblock;++kk){ 
	    tmp = fabs( A.val[j*A.ldblock+kk] );
            if ( (*max) < tmp ) {
              (*max) = tmp;
              (*imax) = i;
              (*jmax) = A.col[j];
            }
	  }
        }
    }

    return DEV_SUCCESS;
}


extern "C"
int
data_norm_diff_vec(
  data_d_matrix* A,
  data_d_matrix* B,
  dataType* norm )
{
  int info = 0;
  assert(A->nnz == B->nnz);
  for (int i=0; i<A->nnz; i++) {
    (*norm) = (*norm) + pow( (A->val[i] - B->val[i]), 2 );
    //printf("data_norm_diff_vec %d %e %e %e\n", i, A->val[i], B->val[i], (A->val[i] - B->val[i]) );
  }
  (*norm) = sqrt((*norm));

  return info;
}

//extern "C"
// template <class T>
// inline int
// sgn(T v) {
//     return (v > T(0)) - (v < T(0));
// }
