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
  
  data_d_matrix Adiag = {Magma_CSR};
  
  data_zprint_csr( Asparse );
  
  data_zmextractdiag( Asparse, &Adiag );
  DEV_CHECKPT
  data_zprint_csr( Adiag );
  
  
  char filename[] = "testing/matrices/bcsr_test.mtx";
  data_d_matrix B = {Magma_CSR};
  data_z_csr_mtx( &B, filename ); 
  data_zprint_csr( B );
  
  // estimate condition number of B
  
  data_d_matrix C = {Magma_BCSR};
  C.blocksize = 5;
  data_zmconvert( B, &C, Magma_CSR, Magma_BCSR );
  printf("C.ldblock=%d\n", C.ldblock);
  
  // estimate condition number of C
  
  data_zprint_bcsr( &C );
  
  data_d_matrix Cdiag = {Magma_BCSR};
  Cdiag.blocksize = 5;
  
  data_zmextractdiag( C, &Cdiag );
  printf("Cdiag.ldblock=%d\n", Cdiag.ldblock);
  
  DEV_CHECKPT
  data_zprint_bcsr( &Cdiag );
  
  // copy diagonal block matrix into Cdiaginv
  data_d_matrix Cdiaginv = {Magma_BCSR};
  Cdiaginv.blocksize = 5;
  data_zmcopy(Cdiag, &Cdiaginv);
  printf("Cdiaginv.ldblock=%d\n", Cdiaginv.ldblock);
  
  DEV_CHECKPT
  data_zprint_bcsr( &Cdiaginv );
  getchar();
  DEV_CHECKPT
  // calculate block inverses
  data_inverse_bcsr(&Cdiag, &Cdiaginv);
  
  // block multiply Cdiag and Cdiaginv
  
  // check for identity blocks
  data_diagbcsr_mult_bcsr( &Cdiaginv, &C );
  
  // block multiply C and Cdiaginv
  // estimate condition number
  
  data_zmfree( &Asparse );
	data_zmfree( &Adiag );
	
  //testing::InitGoogleTest(&argc, argv);
  //return RUN_ALL_TESTS();
  
  printf("done\n");
  fflush(stdout); 
  return 0;
  
}