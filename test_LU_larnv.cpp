/*

*/

#include <stdlib.h>
#include <stdio.h>
#include <libgen.h>
#include <string>
//#include <gtest/gtest.h>
//#include <gmock/gmock.h>
#include <vector>
#include <omp.h>

#include "include/mmio.h"
#include "include/sparse_types.h"

int main(int argc, char* argv[])
{

  // begin with a square dense matrix A
  //char* sparse_filename;
  //char* sparse_basename;
  char sparse_name[256];
  char* output_dir;
  char output_basename[256];
  char output_L[256];
  char output_U[256];
  int dim = 20;
  int tile = 100;
  
  if (argc < 4) {
    printf("Usage %s <mdim> <tile size> <output directory>\n", argv[0] );
    return 1;
  }
  else {
    dim = atoi( argv[1] );
    tile = atoi( argv[2] );
    output_dir = argv[3];
    printf("Matrix dim is %d \n", dim ); 
    printf("tile size is %d \n", tile ); 
    printf("Output directory is %s\n", output_dir );
    strcpy( output_basename, output_dir );
    strcat( output_basename, sparse_name );
    printf("Output file base name is %s\n", output_basename );
  }
  //char sparse_filename[] = "testing/matrices/Trefethen_20.mtx";
  //data_d_matrix Asparse = {Magma_CSR};
  //data_z_csr_mtx( &Asparse, sparse_filename );
  //data_d_matrix A = {Magma_DENSE};
  //data_zmconvert( Asparse, &A, Magma_CSR, Magma_DENSE ); 
  //data_d_matrix B = {Magma_DENSE};
  //data_zmconvert( Asparse, &B, Magma_CSR, Magma_DENSE ); 
  ////data_zdisplay_dense( &A );
  //data_zmfree( &Asparse );
  
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
  
  // =========================================================================
  // MKL LU with no pivoting (Benchmark)
  // =========================================================================
  printf("%% MKL LU with no pivoting (Benchmark)\n");
  data_d_matrix Amkl = {Magma_DENSE};
  data_zmconvert(A, &Amkl, Magma_DENSE, Magma_DENSE);
  
  dataType wstart = omp_get_wtime();
  data_LUnp_mkl( &Amkl );
  dataType wend = omp_get_wtime();
  printf("%% MKL LU with no pivoting required %f wall clock seconds as measured by omp_get_wtime()\n", wend-wstart );
  
  dataType Amkldiff = 0.0;
  data_zfrobenius_inplaceLUresidual(A, Amkl, &Amkldiff);
  printf("MKL_LUnp_res = %e\n", Amkldiff);
  strcpy( output_L, output_basename );
  strcat( output_L, "_LUmkl.mtx" );
  //strcpy( output_U, output_basename );
  //strcat( output_U, "_Umkl.mtx" );
  data_zwrite_dense( Amkl, output_L );
  //data_zwrite_dense( Umkl, output_U );
  data_zmfree( &Amkl );
  fflush(stdout); 

  
  
  data_d_matrix L = {Magma_DENSEL};
  data_d_matrix U = {Magma_DENSEU};
  dataType Adiff = 0.0;

  // =========================================================================
  // ParLU v0.0
  // =========================================================================
  printf("%% ParLU v0.0\n");
  // Separate the strictly lower and upper elements 
  // into L, and U respectively.
  L = {Magma_DENSEL};
  U = {Magma_DENSEU};
  data_ParLU_v0_0( &A, &L, &U);
  // Check ||A-LU||_Frobenius
  data_zfrobenius_LUresidual(A, L, U, &Adiff);
  printf("ParLUv0_0_res = %e\n", Adiff);
  fflush(stdout); 
  strcpy( output_L, output_basename );
  strcat( output_L, "_LparLUv0_0.mtx" );
  strcpy( output_U, output_basename );
  strcat( output_U, "_UparLUv0_0.mtx" );
  data_zwrite_dense( L, output_L );
  data_zwrite_dense( U, output_U );
  data_zmfree( &L );
  data_zmfree( &U );
  fflush(stdout); 
  // =========================================================================
  data_zmfree( &A );
  data_zmconvert( B, &A, Magma_DENSE, Magma_DENSE );  // reset A 
  // =========================================================================
  // ParLU v0.1
  // =========================================================================
  //
  // Separate the strictly lower and upper, elements 
  // into L, U respectively.
  // Convert U to column major storage.
  printf("%% ParLU v0.1\n");
  L = {Magma_DENSEL};
  U = {Magma_DENSEU};
  data_ParLU_v0_1( &A, &L, &U);
  // Check ||A-LU||_Frobenius
  data_zfrobenius_LUresidual(A, L, U, &Adiff);
  printf("ParLUv0_1_res = %e\n", Adiff);
  fflush(stdout); 
  strcpy( output_L, output_basename );
  strcat( output_L, "_LparLUv0_1.mtx" );
  strcpy( output_U, output_basename );
  strcat( output_U, "_UparLUv0_1.mtx" );
  data_zwrite_dense( L, output_L );
  data_zwrite_dense( U, output_U );
  data_zmfree( &L );
  data_zmfree( &U );
  fflush(stdout); 
  // =========================================================================
  data_zmfree( &A );
  data_zmconvert( B, &A, Magma_DENSE, Magma_DENSE );  // reset A 
  // =========================================================================
  // ParLU v1.0
  // =========================================================================
  //
  // Separate the strictly lower, upper elements 
  // into L and U respectively.
  printf("%% ParLU v1.0\n");
  L = {Magma_DENSEL};
  U = {Magma_DENSEU};
  // ParLU with dot products replacing summations
  data_ParLU_v1_0( &A, &L, &U);
  // Check ||A-LU||_Frobenius
  data_zfrobenius_LUresidual(A, L, U, &Adiff);
  printf("ParLUv1_0_res = %e\n", Adiff);
  fflush(stdout); 
  strcpy( output_L, output_basename );
  strcat( output_L, "_LparLUv1_0.mtx" );
  strcpy( output_U, output_basename );
  strcat( output_U, "_UparLUv1_0.mtx" );
  data_zwrite_dense( L, output_L );
  data_zwrite_dense( U, output_U );
  data_zmfree( &L );
  data_zmfree( &U );
  fflush(stdout); 
  // =========================================================================
  data_zmfree( &A );
  data_zmconvert( B, &A, Magma_DENSE, Magma_DENSE );  // reset A 
  // =========================================================================
  // ParLU v1.1
  // =========================================================================
  //
  // Separate the strictly lower, upper elements 
  // into L and U respectively.
  // Convert U to column major storage.
  printf("%% ParLU v1.1\n");
  L = {Magma_DENSEL};
  U = {Magma_DENSEU};
  // ParLU with dot products replacing summations
  data_ParLU_v1_1( &A, &L, &U);
  // Check ||A-LU||_Frobenius
  data_zfrobenius_LUresidual(A, L, U, &Adiff);
  printf("ParLUv1_1_res = %e\n", Adiff);
  fflush(stdout); 
  strcpy( output_L, output_basename );
  strcat( output_L, "_LparLUv1_1.mtx" );
  strcpy( output_U, output_basename );
  strcat( output_U, "_UparLUv1_1.mtx" );
  data_zwrite_dense( L, output_L );
  data_zwrite_dense( U, output_U );
  data_zmfree( &L );
  data_zmfree( &U );
  fflush(stdout);
  // =========================================================================

  data_zmfree( &A );
  data_zmconvert( B, &A, Magma_DENSE, Magma_DENSE );  // reset A 
  // =========================================================================
  // ParLU v1.2
  // =========================================================================
  //
  // Separate the strictly lower, strictly upper, and diagonal elements 
  // into L, U, and D respectively.
  // ParLU with dot products and a tiled access pattern
  printf("%% ParLU v1.2\n");
  L = {Magma_DENSEL};
  U = {Magma_DENSEU};
  //int tile = 8;
  // ParLU with dot products replacing summations
  data_ParLU_v1_2( &A, &L, &U, tile);
  // Check ||A-LU||_Frobenius
  data_zfrobenius_LUresidual(A, L, U, &Adiff);
  printf("ParLUv1_2_res = %e\n", Adiff);
  fflush(stdout); 
  strcpy( output_L, output_basename );
  strcat( output_L, "_LparLUv1_2.mtx" );
  strcpy( output_U, output_basename );
  strcat( output_U, "_UparLUv1_2.mtx" );
  data_zwrite_dense( L, output_L );
  data_zwrite_dense( U, output_U );
  data_zmfree( &L );
  data_zmfree( &U );
  fflush(stdout); 
  // =========================================================================

  data_zmfree( &A );
  data_zmconvert( B, &A, Magma_DENSE, Magma_DENSE );  // reset A 
  // =========================================================================
  // ParLU v1.2c
  // =========================================================================
  //
  // Separate the strictly lower, strictly upper, and diagonal elements 
  // into L, U, and D respectively.
  // ParLU with dot products and a tiled access pattern
  printf("%% ParLU v1.2c\n");
  L = {Magma_DENSEL};
  U = {Magma_DENSEU};
  //int tile = 8;
  // ParLU with dot products replacing summations
  data_ParLU_v1_2c( &A, &L, &U, tile);
  // Check ||A-LU||_Frobenius
  data_zfrobenius_LUresidual(A, L, U, &Adiff);
  printf("ParLUv1_2c_res = %e\n", Adiff);
  fflush(stdout); 
  strcpy( output_L, output_basename );
  strcat( output_L, "_LparLUv1_2c.mtx" );
  strcpy( output_U, output_basename );
  strcat( output_U, "_UparLUv1_2c.mtx" );
  data_zwrite_dense( L, output_L );
  data_zwrite_dense( U, output_U );
  data_zmfree( &L );
  data_zmfree( &U );
  fflush(stdout); 
  // =========================================================================

  
  data_zmfree( &A );
  data_zmconvert( B, &A, Magma_DENSE, Magma_DENSE );  // reset A 
  // =========================================================================
  // ParLU v1.3
  // =========================================================================
  //
  // Separate the strictly lower, strictly upper, and diagonal elements 
  // into L, U, and D respectively.
  // Convert U to column major storage.
  printf("%% ParLU v1.3\n");
  L = {Magma_DENSEL};
  U = {Magma_DENSEU};
  //tile = 8;
  // ParLU with dot products replacing summations
  data_ParLU_v1_3( &A, &L, &U, tile);
  // Check ||A-LU||_Frobenius
  data_zfrobenius_LUresidual(A, L, U, &Adiff);
  printf("ParLUv1_3_res = %e\n", Adiff);
  fflush(stdout); 
  strcpy( output_L, output_basename );
  strcat( output_L, "_LparLUv1_3.mtx" );
  strcpy( output_U, output_basename );
  strcat( output_U, "_UparLUv1_3.mtx" );
  data_zwrite_dense( L, output_L );
  data_zwrite_dense( U, output_U );
  data_zmfree( &L );
  data_zmfree( &U );
  fflush(stdout); 
  // =========================================================================
  

  data_zmfree( &A );
  data_zmconvert( B, &A, Magma_DENSE, Magma_DENSE );  // reset A 

  // =========================================================================
  // ParLU v2.0
  // =========================================================================
  //
  // Separate the strictly lower, strictly upper, and diagonal elements 
  // into L, U, and D respectively.
  // ParLU with matrix-vector products and a tiled access pattern
  printf("%% ParLU v2.0\n");
  L = {Magma_DENSEL};
  U = {Magma_DENSEU};
  //tile = 8;
  // ParLU with matrix-vector products replacing summations
  data_ParLU_v2_0( &A, &L, &U, tile);
  // Check ||A-LU||_Frobenius
  data_zfrobenius_LUresidual(A, L, U, &Adiff);
  printf("ParLUv2_0_res = %e\n", Adiff);
  fflush(stdout); 
  strcpy( output_L, output_basename );
  strcat( output_L, "_LparLUv2_0.mtx" );
  strcpy( output_U, output_basename );
  strcat( output_U, "_UparLUv2_0.mtx" );
  data_zwrite_dense( L, output_L );
  data_zwrite_dense( U, output_U );
  data_zmfree( &L );
  data_zmfree( &U );
  fflush(stdout); 
  // =========================================================================
  data_zmfree( &A );
  data_zmconvert( B, &A, Magma_DENSE, Magma_DENSE );  // reset A 
 
  // =========================================================================
  // ParLU v2.1
  // =========================================================================
  //
  // Separate the strictly lower, strictly upper, and diagonal elements 
  // into L, U, and D respectively.
  // Use a calloc'd workspace for all threads
  printf("%% ParLU v2.1\n");
  L = {Magma_DENSEL};
  U = {Magma_DENSEU};
  //tile = 8;
  // ParLU with matrix-vector products replacing summations
  data_ParLU_v2_1( &A, &L, &U, tile);
  // Check ||A-LU||_Frobenius
  data_zfrobenius_LUresidual(A, L, U, &Adiff);
  printf("ParLUv2_1_res = %e\n", Adiff);
  fflush(stdout); 
  strcpy( output_L, output_basename );
  strcat( output_L, "_LparLUv2_1.mtx" );
  strcpy( output_U, output_basename );
  strcat( output_U, "_UparLUv2_1.mtx" );
  data_zwrite_dense( L, output_L );
  data_zwrite_dense( U, output_U );
  data_zmfree( &L );
  data_zmfree( &U );
  fflush(stdout); 
  // =========================================================================
  
  data_zmfree( &A );
  data_zmconvert( B, &A, Magma_DENSE, Magma_DENSE );  // reset A 
  // =========================================================================
  // ParLU v3.0
  // =========================================================================
  //
  // Separate the strictly lower, strictly upper, and diagonal elements 
  // into L, U, and D respectively.
  // ParLU with matrix-matrix products and a tiled access pattern
  printf("%% ParLU v3.0\n");
  L = {Magma_DENSEL};
  U = {Magma_DENSEU};
  //tile = 8;
  // ParLU with matrix-matrix products replacing summations
  data_ParLU_v3_0( &A, &L, &U, tile);
  // Check ||A-LU||_Frobenius
  data_zfrobenius_LUresidual(A, L, U, &Adiff);
  printf("ParLUv3_0_res = %e\n", Adiff);
  fflush(stdout); 
  strcpy( output_L, output_basename );
  strcat( output_L, "_LparLUv3_0.mtx" );
  strcpy( output_U, output_basename );
  strcat( output_U, "_UparLUv3_0.mtx" );
  data_zwrite_dense( L, output_L );
  data_zwrite_dense( U, output_U );
  data_zmfree( &L );
  data_zmfree( &U );
  fflush(stdout);
  // =========================================================================
  data_zmfree( &A );
  data_zmconvert( B, &A, Magma_DENSE, Magma_DENSE );  // reset A 
  
  // =========================================================================
  // ParLU v3.1
  // =========================================================================
  //
  // Separate the strictly lower, strictly upper, and diagonal elements 
  // into L, U, and D respectively.
  // ParLU with matrix-matrix products and a tiled access pattern
  printf("%% ParLU v3.1\n");
  L = {Magma_DENSEL};
  U = {Magma_DENSEU};
  // ParLU with matrix-matrix products replacing summations
  data_ParLU_v3_1( &A, &L, &U, tile);
  // Check ||A-LU||_Frobenius
  data_zfrobenius_LUresidual(A, L, U, &Adiff);
  printf("ParLUv3_1_res = %e\n", Adiff);
  fflush(stdout); 
  strcpy( output_L, output_basename );
  strcat( output_L, "_LparLUv3_1.mtx" );
  strcpy( output_U, output_basename );
  strcat( output_U, "_UparLUv3_1.mtx" );
  data_zwrite_dense( L, output_L );
  data_zwrite_dense( U, output_U );
  data_zmfree( &L );
  data_zmfree( &U );
  fflush(stdout);
  // =========================================================================
  data_zmfree( &A );
  data_zmconvert( B, &A, Magma_DENSE, Magma_DENSE );  // reset A 
  
  // =========================================================================
  // ParLU v3.1
  // =========================================================================
  //
  // Separate the strictly lower, strictly upper, and diagonal elements 
  // into L, U, and D respectively.
  // Convert U to column major storage.
  
  // ParLU with matrix-matrix products and a tiled access pattern
  
  // Check A-LU
  // Check ||A-LU||_Frobenius
  // =========================================================================

  data_zmfree( &B );
  //testing::InitGoogleTest(&argc, argv);
  //return RUN_ALL_TESTS();
  return 0;
  
}