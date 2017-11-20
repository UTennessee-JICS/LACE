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

#define MY_EXPECT_ARRAY_DOUBLE_EQ( length, ref, target) \
{ \
  unsigned int i = 0; \
  for(i=0; i<length; i++) { \
    if (ref[i] - target[i] > 1.0e-13) \
      printf("Arrays ref[%d] = %.16e  target[%d] = %.16e" \
        " differ\n", i, ref[i], i, target[i] );  \
  } \
}

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

  dataType zero = 0.0;
  dataType one = 1.0;
  dataType negone = -1.0;

  if (argc < 2) {
    printf("Usage %s <matrix> \n", argv[0] );
    return 1;
  }
  else {
    sparse_filename = argv[1];
    sparse_basename = basename( sparse_filename );
    char *ext;
    ext = strrchr( sparse_basename, '.');
    strncpy( sparse_name, sparse_basename, int(ext - sparse_basename) );
    printf( "File %s basename %s name %s \n",
      sparse_filename, sparse_basename, sparse_name );
  }
  data_d_matrix Adense = {Magma_DENSE};
  CHECK( data_z_dense_mtx( &Adense, MagmaColMajor, sparse_filename ) );

  DEV_CHECKPT
  printf( "Adense num_rows = %d num_cols = %d ld =%d\n",
    Adense.num_rows, Adense.num_cols, Adense.ld );
  data_zdisplay_dense( &Adense );

  dataType* v;
  LACE_CALLOC( v, Adense.num_rows );
  dataType nu = 0;
  data_housegen( Adense.num_rows, Adense.val, v, &nu );
  for ( int i=0; i<Adense.num_rows; i++ ) {
    printf( "Adense.val[%d] = %e\n", i, Adense.val[idx(i,0,Adense.ld)] );
  }
  for ( int i=0; i<Adense.num_rows; i++ ) {
    printf( "v.val[%d] = %e\n", i, v[i] );
  }
  printf( "nu =%e\n", nu );

  data_d_matrix U = {Magma_DENSE};
  data_zvinit( &U, Adense.num_rows, Adense.num_cols, zero );
  int m = MIN( Adense.num_rows, Adense.num_cols );
  data_d_matrix R = {Magma_DENSE};
  data_zvinit( &R, m, m, zero );

  data_hqrd( &Adense, &U, &R );


  printf("Adense\n");
  data_zdisplay_dense( &Adense );
  data_d_matrix U_check = {Magma_DENSE};
  CHECK( data_z_dense_mtx( &U_check, MagmaColMajor, "magic7_Householder_U.mtx" ) );
  printf("U\n");
  data_zdisplay_dense( &U );
  MY_EXPECT_ARRAY_DOUBLE_EQ( U.nnz, U_check.val, U.val );
  data_d_matrix R_check = {Magma_DENSE};
  CHECK( data_z_dense_mtx( &R_check, MagmaColMajor, "magic7_Householder_R.mtx" ) );
  printf("R\n");
  data_zdisplay_dense( &R );
  MY_EXPECT_ARRAY_DOUBLE_EQ( R.nnz, R_check.val, R.val );


  data_d_matrix Q = {Magma_DENSE};
  data_zvinit( &Q, R.num_rows, R.num_cols, zero );
  data_d_matrix identity = {Magma_DENSE};
  data_zvinit( &identity, R.num_rows, R.num_cols, zero );
  for (int e=0; e<identity.ld; e++) {
    identity.val[idx(e,e,identity.ld)] = 1.0;
  }

  data_house_apply( &U, &identity, &Q );

  printf("Q\n");
  data_zdisplay_dense( &Q );

  printf("U\n");
  data_zdisplay_dense( &U );

  data_d_matrix QR = {Magma_DENSE};
  data_zvinit( &QR, R.num_rows, R.num_cols, zero );

  data_dgemm_mkl( MagmaColMajor, MagmaNoTrans, MagmaNoTrans,
    R.num_rows, R.num_rows, R.num_rows,
    one, Q.val, Q.ld,
    R.val, R.ld,
    zero, QR.val, QR.ld );

  printf("QR\n");
  data_zdisplay_dense( &QR );
  data_d_matrix A_check = {Magma_DENSE};
  CHECK( data_z_dense_mtx( &A_check, MagmaColMajor, sparse_filename ) );

  MY_EXPECT_ARRAY_DOUBLE_EQ( A_check.nnz, A_check.val, QR.val );

  dataType org_infinityNorm = 0.0;
  dataType qr_infinityNorm = 0.0;
  dataType ortherr = 0.0;
  int imax = -1;

  data_d_matrix diff = {Magma_DENSE};
  data_zvinit( &diff, A_check.num_rows, A_check.num_cols, zero );
  printf( "A_check.major = %d\n", A_check.major );
  data_domatadd( one, &A_check, MagmaNoTrans,
    negone, &QR, MagmaNoTrans,
    &diff );

  printf("A-QR\n");
  data_zdisplay_dense( &diff );

  data_infinity_norm( &A_check, &imax, &org_infinityNorm );
  printf( "org_infinityNorm = %e\n", org_infinityNorm);

  data_infinity_norm( &diff, &imax, &qr_infinityNorm );
  printf( "qr_infinityNorm = %e\n", qr_infinityNorm );
  qr_infinityNorm = qr_infinityNorm/org_infinityNorm;
  printf( "qr_infinityNorm = %e\n", qr_infinityNorm );

  printf( "Q.num_cols = %d Q.ld = %d\n", Q.num_cols, Q.ld );
  data_orthogonality_error( &Q, &ortherr, &imax, Q.num_cols );
  printf( "ortherr = %e\n", ortherr );

  data_zmfree( &Adense );
  data_zmfree( &U );
  data_zmfree( &R );
  data_zmfree( &Q );
  data_zmfree( &QR );
  data_zmfree( &A_check );
  data_zmfree( &diff );

  free( v );

  //testing::InitGoogleTest(&argc, argv);
  //return RUN_ALL_TESTS();

  printf("done\n");
  fflush(stdout);
  return 0;

}
