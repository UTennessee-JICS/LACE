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
  char* sparse_filename;
  char* sparse_basename;
  char sparse_name[256];
  char* output_dir;
  char output_basename[256];
  char output_L[256];
  //char output_U[256];
  int tile = 100;

  if (argc < 4) {
    printf("Usage %s <matrix> <tile size> <output directory>\n", argv[0] );
    return 1;
  }
  else {
    sparse_filename = argv[1];
    tile = atoi( argv[2] );
    output_dir = argv[3];
    sparse_basename = basename( sparse_filename );
    char *ext;
    ext = strrchr( sparse_basename, '.');
    strncpy( sparse_name, sparse_basename, int(ext - sparse_basename) );
    printf("File %s basename %s name %s \n",
      sparse_filename, sparse_basename, sparse_name );
    printf("tile size is %d \n", tile );
    printf("Output directory is %s\n", output_dir );
    strcpy( output_basename, output_dir );
    strcat( output_basename, sparse_name );
    printf("Output file base name is %s\n", output_basename );
  }
  //char sparse_filename[] = "testing/matrices/Trefethen_20.mtx";
  data_d_matrix Asparse = {Magma_CSR};
  data_z_csr_mtx( &Asparse, sparse_filename );
  data_d_matrix A = {Magma_DENSE};
  data_zmconvert( Asparse, &A, Magma_CSR, Magma_DENSE );
  data_d_matrix B = {Magma_DENSE};
  data_zmconvert( Asparse, &B, Magma_CSR, Magma_DENSE );
  //data_zdisplay_dense( &A );
  data_zmfree( &Asparse );

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
  data_zmfree( &B );

  //testing::InitGoogleTest(&argc, argv);
  //return RUN_ALL_TESTS();
  return 0;

}
