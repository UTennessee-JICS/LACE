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
  char output_L[256];
  char output_U[256];

  // for PariLU0
  //dataType user_precond_reduction = 1.0e-15;
  dataType user_precond_reduction = 1.0e-1;
  data_d_preconditioner_log parilu_log;

  if (argc < 3) {
    printf("Usage %s <matrix> <output directory>\n", argv[0] );
    return 1;
  }
  else {
    sparse_filename = argv[1];
    output_dir = argv[2];
    sparse_basename = basename( sparse_filename );
    char *ext;
    ext = strrchr( sparse_basename, '.');
    strncpy( sparse_name, sparse_basename, int(ext - sparse_basename) );
    printf("File %s basename %s name %s \n",
      sparse_filename, sparse_basename, sparse_name );
    printf("Output directory is %s\n", output_dir );
    strcpy( output_basename, output_dir );
    strcat( output_basename, sparse_name );
    printf("Output file base name is %s\n", output_basename );
  }
  data_d_matrix Asparse = {Magma_CSR};
  CHECK( data_z_csr_mtx( &Asparse, sparse_filename ) );

  DEV_CHECKPT

  data_d_matrix A = {Magma_CSR};
  data_zmconvert( Asparse, &A, Magma_CSR, Magma_CSR );
  //data_zdisplay_dense( &A );
  //data_zmfree( &Asparse );

  // =========================================================================
  // MKL csrilu0  (Benchmark)
  // =========================================================================
  printf("%% MKL csrilu0 (Benchmark)\n");
  data_d_matrix Amkl = {Magma_CSR};
  data_zmconvert(Asparse, &Amkl, Magma_CSR, Magma_CSR);

  dataType wstart = omp_get_wtime();
  CHECK( data_dcsrilu0_mkl( &Amkl ) );
  dataType wend = omp_get_wtime();
  printf("%% MKL csrilu0 required %f wall clock seconds as measured by omp_get_wtime()\n", wend-wstart );


  data_d_matrix Lmkl = {Magma_CSRL};
  Lmkl.diagorder_type = Magma_UNITY;
  data_zmconvert(Amkl, &Lmkl, Magma_CSR, Magma_CSRL);
  printf("test if Lmkl is lower: ");
  data_zcheckupperlower( &Lmkl );
  printf(" done.\n");
  data_d_matrix Umkl = {Magma_CSRU};
  Umkl.diagorder_type = Magma_VALUE;
  data_zmconvert(Amkl, &Umkl, Magma_CSR, Magma_CSRU);
  printf("test if Umkl is upper: ");
  data_zcheckupperlower( &Umkl );
  printf(" done.\n");
  data_d_matrix LUmkl = {Magma_CSR};
  data_zmconvert(Amkl, &LUmkl, Magma_CSR, Magma_CSR);

  dataType Amklres = 0.0;
  dataType Amklnonlinres = 0.0;
  data_zilures( A, Lmkl, Umkl, &LUmkl, &Amklres, &Amklnonlinres);

  printf("MKL_csrilu0_res = %e\n", Amklres);
  printf("MKL_csrilu0_nonlinres = %e\n", Amklnonlinres);
  strcpy( output_L, output_basename );
  strcat( output_L, "_Lmkl.mtx" );
  strcpy( output_U, output_basename );
  strcat( output_U, "_Umkl.mtx" );
  data_zwrite_csr_mtx( Lmkl, Lmkl.major, output_L );
  data_zwrite_csr_mtx( Umkl, Umkl.major, output_U );
  data_zmfree( &Amkl );
  //data_zmfree( &Lmkl );
  //data_zmfree( &Umkl );
  fflush(stdout);
  // =========================================================================

  // =========================================================================
  // PariLU v0.0
  // =========================================================================
  //printf("%% PariLU v0.0 to 5 sweeps\n");
  //// Separate the strictly lower and upper elements
  //// into L, and U respectively.
  //data_d_matrix L5 = {Magma_CSRL};
  //data_d_matrix U5 = {Magma_CSCU};
  //data_PariLU_v0_0( &A, &L5, &U5);
  //
  //printf("test if L is lower: ");
  //data_zcheckupperlower( &L5 );
  //printf(" done.\n");
  //printf("test if U is lower: ");
  //data_zcheckupperlower( &U5 );
  //printf(" done.\n");
  //// Check ||A-LU||_Frobenius
  dataType Ares = 0.0;
  dataType Anonlinres = 0.0;
  data_d_matrix LU = {Magma_CSR};
  data_zmconvert(A, &LU, Magma_CSR, Magma_CSR);
  //data_zilures(A, L5, U5, &LU, &Ares, &Anonlinres);
  //printf("PariLUv0_0-5_csrilu0_res = %e\n", Ares);
  //printf("PariLUv0_0-5_csrilu0_nonlinres = %e\n", Anonlinres);
  ////data_zmfree( &L );
  ////data_zmfree( &U );
  ////data_zmfree( &LU );
  //fflush(stdout);
  //
  //data_d_matrix Ldiff = {Magma_CSRL};
  //dataType Lres = 0.0;
  //dataType Lnonlinres = 0.0;
  //data_zdiff_csr(&Lmkl, &L5, &Ldiff, &Lres, &Lnonlinres );
  ////data_zwrite_csr( &Ldiff );
  //printf("L_res = %e\n", Lres);
  //printf("L_nonlinres = %e\n", Lnonlinres);
  //fflush(stdout);
  //
  //data_d_matrix Udiff = {Magma_CSRU};
  //dataType Ures = 0.0;
  //dataType Unonlinres = 0.0;
  //data_zdiff_csr(&Umkl, &U5, &Udiff, &Ures, &Unonlinres );
  ////data_zwrite_csr( &Udiff );
  //printf("U_res = %e\n", Ures);
  //printf("U_nonlinres = %e\n", Unonlinres);
  //fflush(stdout);
  //dataType vmaxA = 0.0;
  //int imaxA = 0;
  //int jmaxA = 0;
  //data_maxfabs_csr(Ldiff, &imaxA, &jmaxA, &vmaxA);
  //printf("max(fabs(Ldiff)) = (%d,%d) %e\n", imaxA, jmaxA, vmaxA);
  //data_maxfabs_csr(Udiff, &imaxA, &jmaxA, &vmaxA);
  //printf("max(fabs(Udiff)) = (%d,%d) %e\n", imaxA, jmaxA, vmaxA);
  //
  //printf("test if Ldiff is lower: ");
  //data_zcheckupperlower( &Ldiff );
  //printf(" done.\n");
  //printf("test if Udiff is lower: ");
  //data_zcheckupperlower( &Udiff );
  //printf(" done.\n");
  //
  //strcpy( output_L, output_basename );
  //strcat( output_L, "_L5pariLUv0_0.mtx" );
  //strcpy( output_U, output_basename );
  //strcat( output_U, "_U5pariLUv0_0.mtx" );
  //data_zwrite_csr_mtx( L5, L5.major, output_L );
  //data_zwrite_csr_mtx( U5, U5.major, output_U );
  //strcpy( output_L, output_basename );
  //strcat( output_L, "_L5pariLUv0_0_diff.mtx" );
  //strcpy( output_U, output_basename );
  //strcat( output_U, "_U5pariLUv0_0_diff.mtx" );
  //data_zwrite_csr_mtx( Ldiff, Ldiff.major, output_L );
  //data_zwrite_csr_mtx( Udiff, Udiff.major, output_U );
  //data_zmfree( &Ldiff );
  //data_zmfree( &Udiff );
  //
  ////data_zmfree( &L );
  ////data_zmfree( &U );
  //data_zmfree( &LU );

  // =========================================================================
  // PariLU v0.3
  // =========================================================================
  printf("%% PariLU v0.3 BCSR\n");

  data_d_matrix A_BCSR = {Magma_BCSR};
  A_BCSR.blocksize = 5;
  data_zmconvert( Asparse, &A_BCSR, Magma_CSR, Magma_BCSR );
  //data_zprint_bcsr( &A_BCSR );

  // Separate the strictly lower and upper elements
  // into L, and U respectively.
  data_d_matrix L = {Magma_BCSRL};
  //L = {Magma_BCSRL};
  //L.diagorder_type = Magma_UNITY;
  //data_zmconvert(A, &L, Magma_CSR, Magma_CSRL);
  data_d_matrix U = {Magma_BCSCU};
  //U = {Magma_BCSCU};
  //U.diagorder_type = Magma_VALUE;
  //data_zmconvert(A, &U, Magma_CSR, Magma_CSRU);
  user_precond_reduction = 1.0e-4;
  data_PariLU_v0_3( &A_BCSR, &L, &U, user_precond_reduction, &parilu_log );
  DEV_CHECKPT
  // Check ||A-LU||_Frobenius
  Ares = 0.0;
  Anonlinres = 0.0;
  LU = {Magma_CSR};
  DEV_CHECKPT
  data_zmconvert(A_BCSR, &LU, Magma_BCSR, Magma_CSR);

  data_zilures_bcsr(Asparse, L, U, &LU, &Ares, &Anonlinres);
  DEV_CHECKPT
  printf("PariLUv0_3_csrilu0_res = %e\n", Ares);
  printf("PariLUv0_3_csrilu0_nonlinres = %e\n", Anonlinres);
  fflush(stdout);

  data_zmfree( &LU );


  strcpy( output_L, output_basename );
  strcat( output_L, "_Lbcsr.mtx" );
  strcpy( output_U, output_basename );
  strcat( output_U, "_Ubcsr.mtx" );
  data_zwrite_csr_mtx( L, L.major, output_L );
  data_zwrite_csr_mtx( U, U.major, output_U );


  data_zmfree( &Asparse );
  data_zmfree( &A );
  data_zmfree( &A_BCSR );
  data_zmfree( &L );
  data_zmfree( &U );
  data_zmfree( &Amkl );
  data_zmfree( &Lmkl );
  data_zmfree( &Umkl );


  //testing::InitGoogleTest(&argc, argv);
  //return RUN_ALL_TESTS();


  printf("done\n");
  fflush(stdout);
  return 0;

}
