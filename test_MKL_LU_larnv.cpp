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
  //char sparse_name[256];
  char output_dir[256];
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
    //output_dir = argv[3];
    strcpy( output_dir, argv[3] );
    printf("Matrix dim is %d \n", dim ); 
    printf("tile size is %d \n", tile ); 
    printf("Output directory is %s\n", output_dir );
    strcpy( output_basename, output_dir );
    strcat( output_basename, "LU_larnv_" );
    strcat( output_basename, argv[1] );
    strcat( output_basename, "_" );
    strcat( output_basename, argv[2] );
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
  // =========================================================================

  data_zmfree( &A );
  //data_zmfree( &B );
  
  //testing::InitGoogleTest(&argc, argv);
  //return RUN_ALL_TESTS();
  return 0;
  
}