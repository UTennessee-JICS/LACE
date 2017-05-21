/*

*/

#include "include/sparse.h"
#include <stdio.h>
#include <stdlib.h>
#include <libgen.h>
#include <string.h>
#include <vector> 
#include <omp.h>
#include <float.h> 
#include "math.h"

#define size 128

//+++++++
#define EXPECT_ITERABLE_DOUBLE_EQ( TYPE, ref, target) \
{ \
const TYPE& _ref(ref); \
const TYPE& _target(target); \
TYPE::const_iterator tarIter   = _target.begin(); \
TYPE::const_iterator refIter = _ref.begin(); \
unsigned int i = 0; \
while(refIter != _ref.end()) { \
    if ( tarIter == _target.end() ) { \
        ADD_FAILURE() << #target \
            " has a smaller length than " #ref ; \
        break; \
    } \
    EXPECT_DOUBLE_EQ(* refIter, * tarIter) \
        << "Vectors " #ref  " (refIter) " \
           "and " #target " (tarIter) " \
           "differ at index " << i; \
    ++refIter; ++tarIter; ++i; \
} \
EXPECT_TRUE( tarIter == _target.end() ) \
    << #ref " has a smaller length than " \
       #target ; \
}
//+++++++

//+++++++
#define EXPECT_ARRAY_DOUBLE_EQ( length, ref, target) \
{ \
  unsigned int i = 0; \
  for(i=0; i<length; i++) { \
    EXPECT_DOUBLE_EQ(ref[i], target[i]) \
    << "Arrays " #ref  " and " #target \
           "differ at index " << i; \
  } \
}
//+++++++ 

//+++++++
#define EXPECT_ARRAY_INT_EQ( length, ref, target) \
{ \
  unsigned int i = 0; \
  for(i=0; i<length; i++) { \
    EXPECT_EQ(ref[i], target[i]) \
    << "Arrays " #ref  " and " #target \
           "differ at index " << i; \
  } \
}
//+++++++ 


int main(int argc, char* argv[])
{
  
  // begin with a square matrix A
  char* sparse_filename;
  char* rhs_filename;
  char* sparse_basename;
  char sparse_name[256];
  char* output_dir;
  char output_basename[256];
  //char output_L[256];
  //char output_U[256];
  
  int permute = 0;
  
  data_scale_t scaling  = Magma_NOSCALE;
  
  
  // for mkl_dcsrtrsv
	char cvar, cvar1, cvar2;
  
  // for timing of MKL csriLU0 and FGMRES
  dataType wcsrtrsvstart = 0.0;
  dataType wcsrtrsvend = 0.0;
  dataType parwcsrtrsvtime = 0.0;
  
  
  // Partrsv
  int dim = 20;
  int tile = 100;
  dataType ptrsv_tol = 1.0e-15;
  int ptrsv_iter = 0;
  
  if (argc < 4) {
    printf("Usage %s <mdim> <tile size> <tol>", argv[0] );
    //printf("[diagonal scaling] [abs/rel] [GMRES_tolerance] [restart] ");
    //printf("[maxiter] [precond_choice] [reduction]\n");
    return 1;
  }
  else {
    dim = atoi( argv[1] );
    tile = atoi( argv[2] );
    ptrsv_tol = atof(argv[3]); 
    
    printf("Matrix dim is %d \n", dim ); 
    printf("tile size is %d \n", tile ); 
    printf("reading ptrsv_tol %e\n", ptrsv_tol);
    
  }
  
	data_d_matrix A = {Magma_DENSE};
  //lapack_int LAPACKE_dlarnv (lapack_int idist , lapack_int * iseed , lapack_int n , double * x );
  A.num_rows = dim;
  A.num_cols = dim;
  A.nnz = dim*dim;
  A.true_nnz = A.nnz;
  A.ld = dim;
  A.major = MagmaRowMajor;
  int ione = 1;
  int ISEED[4] = {0,0,0,1};
  A.val = (dataType*) calloc( A.nnz, sizeof(dataType) );
  CHECK( LAPACKE_dlarnv( ione, ISEED, A.nnz, A.val ) );
  for ( int i = 0; i<A.num_rows; i++ ) {
    for ( int j = 0; j<A.num_cols; j++ ) {
      if (i == j) {
        A.val[ i*A.ld + j ] += 1.0e3; 
      }
    }
  }
  //data_zdisplay_dense( &A );
  data_d_matrix B = {Magma_DENSE};
  data_zmconvert( A, &B, Magma_DENSE, Magma_DENSE );
  
  DEV_CHECKPT
  
  data_d_matrix Ainv = {Magma_DENSE};
  Ainv.major = MagmaRowMajor;
  
  data_inverse( &A, &Ainv );
  
  dataType one = 1.0;
  dataType zero = 0.0;
  data_dgemm_mkl( MagmaRowMajor, MagmaNoTrans, MagmaNoTrans, 
    A.num_rows, A.num_cols, A.num_rows, 
    one, A.val, A.ld, Ainv.val, Ainv.ld,
    zero, B.val, B.ld );
  
  printf("A*Ainv:\n");
  data_zdisplay_dense( &B );
  
  int num_threads = 0;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  printf("omp_get_num_threads = %d\n", num_threads);
    

  data_zmfree( &A );
	data_zmfree( &Ainv );
  data_zmfree( &B );

  
  char filename[] = "testing/matrices/bcsr_test.mtx";
  data_d_matrix C = {Magma_CSR};
  data_z_csr_mtx( &C, filename ); 
  DEV_CHECKPT
  data_zprint_csr( C );
  data_d_matrix D = {Magma_BCSR};
  D.blocksize = 5;
  data_zmconvert( C, &D, Magma_CSR, Magma_BCSR );
  DEV_CHECKPT
  data_zprint_bcsr( &D );
  
  // TODO: create a matrix copy function
  //data_zmcopy( D, &Dinv );
  
  // TODO: create a function to extract diagonal elements (values or blocks of values)
  
  data_d_matrix Dinv = {Magma_BCSR};
  Dinv.blocksize = D.blocksize;
  data_zmconvert( C, &Dinv, Magma_CSR, Magma_BCSR );
  
  
  data_d_matrix bhandle = {Magma_DENSE};
  bhandle.num_rows = D.blocksize;
  bhandle.num_cols = D.blocksize;
  bhandle.blocksize = D.blocksize;
  bhandle.nnz = bhandle.num_rows*bhandle.num_cols;
  bhandle.true_nnz = bhandle.nnz;
  bhandle.ld = bhandle.num_cols;
  bhandle.major = MagmaRowMajor;
  //LACE_CALLOC(bhandle.val, bhandle.nnz);
  
  
  data_d_matrix binvhandle = {Magma_DENSE};
  binvhandle.num_rows = D.blocksize;
  binvhandle.num_cols = D.blocksize;
  binvhandle.blocksize = D.blocksize;
  binvhandle.nnz = binvhandle.num_rows*binvhandle.num_cols;
  binvhandle.true_nnz = binvhandle.nnz;
  binvhandle.ld = binvhandle.num_cols;
  binvhandle.major = MagmaRowMajor;
  //LACE_CALLOC(binvhandle.val, binvhandle.nnz);
  
  data_d_matrix binvcheck = {Magma_DENSE};
  binvcheck.num_rows = D.blocksize;
  binvcheck.num_cols = D.blocksize;
  binvcheck.blocksize = D.blocksize;
  binvcheck.nnz = binvcheck.num_rows*binvcheck.num_cols;
  binvcheck.true_nnz = binvcheck.nnz;
  binvcheck.ld = binvcheck.num_cols;
  binvcheck.major = MagmaRowMajor;
  LACE_CALLOC(binvcheck.val, binvcheck.nnz);
  
  DEV_CHECKPT
  
  //TODO: create a wrapper routine to invert each block in a bcsr matrix
  // exemplar use is to preform block diagonal scaling:
  // first extract diagonal blocks from matrix A 
  //    data_zmextract_diagonal( A, &Adiag, Magma_BCSR, Magma_BCSR );
  // then invert each block
  //    data_inverse_bcsr( &Adiag, &Adiaginv );  
  // then apply scaling
  
  for (int i=0; i<D.num_rows; i++ ) {
    printf("row %d:\n", i);
    for (int j=D.row[i]; j<D.row[i+1]; j++) {
      printf("block %d bcol %d\n", j, D.col[j]);
  
      bhandle.val = &D.val[j*D.ldblock];
      binvhandle.val = &Dinv.val[j*Dinv.ldblock];
      
      //data_zprint_dense( bhandle );
      //DEV_CHECKPT
      //data_zprint_dense( binvhandle );
      
      data_inverse( &bhandle, &binvhandle );
      
      data_dgemm_mkl( MagmaRowMajor, MagmaNoTrans, MagmaNoTrans, 
        bhandle.num_rows, bhandle.num_cols, binvhandle.num_cols, 
        one, bhandle.val, bhandle.ld, binvhandle.val, binvhandle.ld,
        zero, binvcheck.val, binvcheck.ld );
  
      printf("bhandle*binvhandle block %d bcol %d : \n", j, D.col[j]);
      data_zdisplay_dense( &binvcheck );
      
      //data_zprint_dense( bhandle );
      //DEV_CHECKPT
      //data_zprint_dense( binvhandle );
  
    }
  }
  
  //testing::InitGoogleTest(&argc, argv);
  //return RUN_ALL_TESTS();
  
  
  //data_zmfree( &A );
  //data_zmfree( &Ainv );
  //data_zmfree( &B );
  data_zmfree( &C );
  DEV_CHECKPT
  data_zmfree( &D );
  DEV_CHECKPT
  data_zmfree( &Dinv );
  DEV_CHECKPT
  //data_zmfree( &bhandle );
  DEV_CHECKPT
  //data_zmfree( &binvhandle );
  DEV_CHECKPT
  data_zmfree( &binvcheck );
  
  printf("done\n");
  fflush(stdout); 
  return 0;
  
}