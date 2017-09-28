/*

*/
#define DEBUG_P
#include "include/sparse.h"
#include <stdio.h>
#include <stdlib.h>
#include <libgen.h>
#include <string.h>
#include <vector>
#include <omp.h>
#include <float.h> 
#include "math.h"


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
  
  if (argc < 2) {
    printf("Usage %s <matrix>\n", argv[0] );
    return 1;
  }
  else {
    sparse_filename = argv[1];
    sparse_basename = basename( sparse_filename );
    char *ext;
    ext = strrchr( sparse_basename, '.');
    strncpy( sparse_name, sparse_basename, int(ext - sparse_basename) );
    printf("File %s basename %s name %s \n", 
      sparse_filename, sparse_basename, sparse_name );
    //printf("Output directory is %s\n", output_dir );
    //strcpy( output_basename, output_dir );
    //strcat( output_basename, sparse_name );
    //printf("Output file base name is %s\n", output_basename );
  }
	data_d_matrix Asparse = {Magma_CSR};
  CHECK( data_z_csr_mtx( &Asparse, sparse_filename ) ); 
	
  DEV_CHECKPT
  
  data_d_matrix diagfactor[1] = {Magma_DENSE};
  diagfactor[0].num_rows = Asparse.num_rows;
  diagfactor[0].num_cols = 1;
  diagfactor[0].nnz = Asparse.num_rows;
  diagfactor[0].true_nnz = Asparse.num_rows;
  diagfactor[0].ld = 1;
  diagfactor[0].major = MagmaRowMajor;
  LACE_CALLOC(diagfactor[0].val, diagfactor[0].nnz);
  
  
  data_scale_t scale[1] = {Magma_UNITDIAG};
  
  //data_side_t side = MagmaLeft;
  //data_side_t side = MagmaRight;
  data_side_t side[1] = {MagmaBothSides};
  DEV_CHECKPT
  data_zmscale_generate( 1, scale, side, &Asparse, diagfactor );
  DEV_CHECKPT
  for (int i = 0; i<Asparse.num_rows; i++ ) {
    printf("diagfactor[%d]=%e\n", i, diagfactor[0].val[i] ); 
  }
  DEV_CHECKPT
  data_zmscale_apply( 1, side, diagfactor, &Asparse );
  DEV_CHECKPT
  data_zwrite_csr_mtx( Asparse, MagmaRowMajor, "testing/matrices/diagcsr_scaled.mtx" );
  DEV_CHECKPT
	data_zmfree( &Asparse );
	CHECK( data_z_csr_mtx( &Asparse, sparse_filename ) );  
	DEV_CHECKPT
  data_d_matrix diagfactors[2] = {{ Magma_DENSE} , { Magma_DENSE }};
  
  diagfactors[0].num_rows = Asparse.num_rows;
  diagfactors[0].num_cols = 1;
  diagfactors[0].nnz = Asparse.num_rows;
  diagfactors[0].true_nnz = Asparse.num_rows;
  diagfactors[0].ld = 1;
  diagfactors[0].major = MagmaRowMajor;
  //LACE_CALLOC(diagfactors[0].val, diagfactors[0].nnz);
  posix_memalign( (void**) &(diagfactors[0].val), 64, (diagfactors[0].nnz*sizeof(dataType)) );
  #pragma omp parallel 
  {
    #pragma omp for schedule(monotonic:dynamic,4096) 
    for (int p = 0; p<diagfactors[0].nnz; p++) {
      diagfactors[0].val[p] = 0.0;
    }
  }
    
  diagfactors[1].num_rows = Asparse.num_rows;
  diagfactors[1].num_cols = 1;
  diagfactors[1].nnz = Asparse.num_rows;
  diagfactors[1].true_nnz = Asparse.num_rows;
  diagfactors[1].ld = 1;
  diagfactors[1].major = MagmaRowMajor;
  LACE_CALLOC(diagfactors[1].val, diagfactors[1].nnz);
  
  data_scale_t scales[2] = { Magma_UNITROW, Magma_UNITCOL };
  data_side_t sides[2] = { MagmaLeft, MagmaRight};
  DEV_CHECKPT
  data_zmscale_generate( 2, scales, sides, &Asparse, diagfactors );
  DEV_CHECKPT
  data_zmscale_apply( 2, sides, diagfactors, &Asparse );
  DEV_CHECKPT
  data_zwrite_csr_mtx( Asparse, MagmaRowMajor, "testing/matrices/rAc_csr_scaled.mtx" );
  DEV_CHECKPT
  
	data_zmfree( &Asparse );
	data_zmfree( diagfactor );
	
  //testing::InitGoogleTest(&argc, argv);
  //return RUN_ALL_TESTS();
  
  printf("done\n");
  fflush(stdout); 
  return 0;
  
}