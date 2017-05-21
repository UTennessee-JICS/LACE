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
        A.val[ i*A.ld + j ] = 1.0e2; 
      }
    }
  }
  
  //data_zdisplay_dense( &A );
  data_d_matrix B = {Magma_DENSE};
  data_zmconvert( A, &B, Magma_DENSE, Magma_DENSE );
	
  DEV_CHECKPT
  
	// setup expected answer
	data_d_matrix y_expect = {Magma_DENSE};
  y_expect.num_rows = A.num_rows;
  y_expect.num_cols = 1;
  y_expect.ld = 1;
  y_expect.nnz = y_expect.num_rows;
  //LACE_CALLOC(y_mkl.val, y_mkl.num_rows);
  y_expect.val = (dataType*) calloc( (y_expect.num_rows), sizeof(dataType) );
  #pragma omp parallel 
  {
    #pragma omp for nowait
    for (int i=0; i<A.num_rows; i++) {
      y_expect.val[i] = 1.0;
    }
  }
  
  // Setup rhs 
  data_d_matrix rhs_vector = {Magma_DENSE};
  rhs_vector.num_rows = A.num_rows;
  rhs_vector.num_cols = 1;
  rhs_vector.ld = 1;
  rhs_vector.nnz = A.num_rows;
  rhs_vector.val = (dataType*) calloc( A.num_rows, sizeof(dataType) );
  
  //void cblas_dgemv (const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE trans, 
  //  const MKL_INT m, const MKL_INT n, const double alpha, const double *a, 
  //  const MKL_INT lda, const double *x, const MKL_INT incx, const double beta, 
  //  double *y, const MKL_INT incy);
  data_trans_t trans = MagmaNoTrans;
  dataType one = 1.0;
  dataType zero = 0.0;
  dataType negone = -1.0;
  cblas_dgemv( cblas_order_const(A.major), cblas_trans_const(trans), A.num_rows, A.num_cols,
    one, A.val, A.ld, y_expect.val, 1, zero, rhs_vector.val, 1 );
  
 
  DEV_CHECKPT
  
  // parLU
  data_d_matrix L = {Magma_DENSE};
  data_d_matrix U = {Magma_DENSE};
  L.diagorder_type = Magma_UNITY;
  L.fill_mode = MagmaLower;
  U.diagorder_type = Magma_VALUE;
  U.fill_mode = MagmaUpper;
  data_ParLU_v1_0(&A, &L, &U);
  printf("L:\n");
  data_zdisplay_dense( &L );
  printf("U:\n");
  data_zdisplay_dense( &U );

  DEV_CHECKPT
  
  data_d_matrix y = {Magma_DENSE};
  y.num_rows = A.num_rows;
  y.num_cols = 1;
  y.ld = 1;
  y.nnz = y.num_rows;
  LACE_CALLOC(y.val, y.num_rows);
  
  DEV_CHECKPT
  
  wcsrtrsvstart = omp_get_wtime();
  data_partrsv_dot( MagmaRowMajor, MagmaLower, Magma_DENSEL, Magma_UNITY,
    L.num_rows, L.val, L.ld, rhs_vector.val, 1, y.val, 1, 
    ptrsv_tol, &ptrsv_iter );
  wcsrtrsvend = omp_get_wtime();

	parwcsrtrsvtime = wcsrtrsvend - wcsrtrsvstart;
	
  //dataType error = 0.0;
  //data_norm_diff_vec( &y, &y_expect, &error );
  //printf("ptrsv_tol = %e ptrsv_iter = %d\n", ptrsv_tol, ptrsv_iter );
  //printf("y error = %.16e\n", error);
  //
  //DEV_CHECKPT
  //
  //dataType* Ay;
  ////LACE_CALLOC(Ay, L.num_rows);
  //Ay = (dataType*) calloc( (L.num_rows), sizeof(dataType) );
  //cblas_dgemv( cblas_order_const(L.major), cblas_trans_const(trans), L.num_rows, L.num_cols,
  //  one, L.val, L.ld, y.val, 1, zero, Ay, 1 );
  //cblas_daxpy(L.num_rows, negone, rhs_vector.val, 1, Ay, 1 );
  //
  //error = dataType(0.0);
  //for (int i=0; i<L.num_rows; i++) {
  //  //printf("Ay[%d] = %e\n", i, Ay[i]);
  //  error += pow(Ay[i], 2);
  //}
  //error = sqrt(error);
  //printf("system errors:\n\terror = %.16e\n", error);
  //printf("Par time : %e\n", 
  //  parwcsrtrsvtime); 
  
  // improve y with error and solve again
  //printf("\n===============================================================\n");
  //cblas_daxpy(L.num_rows, one, Ay, 1, y.val, 1 );
  //
  //wcsrtrsvstart = omp_get_wtime();
  //data_partrsv_dot( MagmaRowMajor, MagmaLower, Magma_DENSEL, Magma_UNITY,
  //  L.num_rows, L.val, L.ld, rhs_vector.val, 1, y.val, 1, 
  //  ptrsv_tol, &ptrsv_iter );
  //wcsrtrsvend = omp_get_wtime();
  //
	//parwcsrtrsvtime = wcsrtrsvend - wcsrtrsvstart;
	//
  //error = 0.0;
  //data_norm_diff_vec( &y, &y_expect, &error );
  //printf("ptrsv_tol = %e ptrsv_iter = %d\n", ptrsv_tol, ptrsv_iter );
  //printf("y error = %e\n", error);
  //
  //DEV_CHECKPT
  //
  //free( Ay );
  ////LACE_CALLOC(Ay, L.num_rows);
  //Ay = (dataType*) calloc( (L.num_rows), sizeof(dataType) );
  //cblas_dgemv( cblas_order_const(L.major), cblas_trans_const(trans), L.num_rows, L.num_cols,
  //  one, L.val, L.ld, y.val, 1, zero, Ay, 1 );
  //cblas_daxpy(L.num_rows, negone, rhs_vector.val, 1, Ay, 1 );
  //
  //error = dataType(0.0);
  //for (int i=0; i<L.num_rows; i++) {
  //  //printf("Ay[%d] = %e\n", i, Ay[i]);
  //  error += pow(Ay[i], 2);
  //}
  //error = sqrt(error);
  //printf("system errors:\n\terror = %e\n", error);
  printf("Par time : %e\n", 
    parwcsrtrsvtime); 
 
  getchar();

//------------------------------------------------------------------------------  
  
  //data_d_matrix U = {Magma_DENSEU};
  ////U.diagorder_type = Magma_VALUE;
  //U.diagorder_type = Magma_UNITY;
  //U.fill_mode = MagmaUpper;
  //U.major = MagmaRowMajor;
  //
  //data_zmconvert( A, &U, Magma_DENSE, Magma_DENSEU );
  //
  //// Setup rhs for U
  //cblas_dgemv( cblas_order_const(U.major), cblas_trans_const(trans), U.num_rows, U.num_cols,
  //  one, U.val, U.ld, y_expect.val, 1, zero, rhs_vector.val, 1 );
  //
  //DEV_CHECKPT

  data_d_matrix x = {Magma_DENSE};
  x.num_rows = A.num_rows;
  x.num_cols = 1;
  x.ld = 1;
  x.nnz = x.num_rows;
  LACE_CALLOC(x.val, x.num_rows);
  
  DEV_CHECKPT
  
  wcsrtrsvstart = omp_get_wtime();
  data_partrsv_dot( MagmaRowMajor, MagmaUpper, Magma_DENSEU, Magma_VALUE,
    U.num_rows, U.val, U.ld, y.val, 1, x.val, 1, 
    ptrsv_tol, &ptrsv_iter );
  wcsrtrsvend = omp_get_wtime();

	parwcsrtrsvtime = wcsrtrsvend - wcsrtrsvstart;
	
  dataType error = 0.0;
  data_norm_diff_vec( &x, &y_expect, &error );
  printf("ptrsv_tol = %e ptrsv_iter = %d\n", ptrsv_tol, ptrsv_iter );
  printf("x error = %e, num_rows = %d, x error/num_rows = %e\n", 
    error, A.num_rows, error/((dataType)A.num_rows));
  
  DEV_CHECKPT

  dataType* Ax;
  //LACE_CALLOC(Ax, U.num_rows);
  Ax = (dataType*) calloc( (U.num_rows), sizeof(dataType) );
  cblas_dgemv( cblas_order_const(L.major), cblas_trans_const(trans), A.num_rows, A.num_cols,
    one, A.val, A.ld, x.val, 1, zero, Ax, 1 );
  cblas_daxpy(U.num_rows, negone, rhs_vector.val, 1, Ax, 1 );
  
  error = dataType(0.0);
  for (int i=0; i<U.num_rows; i++) {
    printf("Ax[%d] = %e\n", i, Ax[i]);
    error += pow(Ax[i], 2);
  }
  error = sqrt(error);
  //printf("system errors:\n\terror = %e\n", error);
  printf("system error = %e, num_rows = %d, system error/num_rows = %e\n", 
    error, A.num_rows, error/((dataType)A.num_rows));
  printf("Par time : %e\n", 
    parwcsrtrsvtime); 
  
  
  int num_threads = 0;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  printf("omp_get_num_threads = %d\n", num_threads);
    

  data_zmfree( &A );
	data_zmfree( &rhs_vector );
  data_zmfree( &y );
  data_zmfree( &y_expect );
  data_zmfree( &x );
  data_zmfree( &L );
  data_zmfree( &U );

  //free( Ay );
  free( Ax );

  
  //testing::InitGoogleTest(&argc, argv);
  //return RUN_ALL_TESTS();
  
  printf("done\n");
  fflush(stdout); 
  return 0;
  
}