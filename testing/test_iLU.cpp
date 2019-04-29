
#include <stdlib.h>
#include <stdio.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <vector>
#include <omp.h>

#include "mmio.h"
#include "sparse.h"
#include "container_tests.h"
#include "test_cmd_line.h"

#define USE_CUDA 0

class iLUTest: public
  ::testing::Test
{
protected:
  iLUTest() {}

  // per-test-case set-up
  static void SetUpTestCase() {
    printf("seting up\n");
    fflush(stdout);

    printf("my_argc = %d\n", my_argc);
    for (int i=0; i< my_argc; ++i) {
      printf("my_argv[%d] = %s\n", i, my_argv[i]);
    }
    fflush(stdout);

    block_size = new int();

    //char default_matrix[] = "matrices/Trefethen_20.mtx"; // 19 x 19
    //char default_matrix[] = "matrices/ciprian.mtx"; // 8 x 8
    //char default_matrix[] = "matrices/sparisty2x2_test.mtx";
    //char default_matrix[] = "matrices/sparsity6x6_dense.mtx";
    //char default_matrix[] = "matrices/30p30n.mtx"; // 211685 x 211685
    //char default_matrix[] = "matrices/olafu/olafu.mtx"; // 16146 x 16146
    //char default_matrix[] = "matrices/paper1_matrices/ani5_crop.mtx";//12561 x 12561
 char default_matrix[] = "matrices/fidap001.mtx"; //216 x 216
    //char default_matrix[] = "matrices/fidapm05.mtx"; //42 x 42
    //char default_matrix[] = "matrices/steam3.mtx"; // 80 x 80
    (*block_size) = 1; 

    char* matrix_name = NULL;
    tile_size = new int();
    (*tile_size) = 8;

    matchfactor = new int();
    (*matchfactor)=1000.0;

    // parse command line arguments
    if (my_argc>1) {
      int count = 1;
      while (count < my_argc) {
        if ( (strcmp(my_argv[count], "--matrix") == 0)
            && count+1 < my_argc ) {
          matrix_name = my_argv[count+1];
          count = count + 2;
        }
        else if ( (strcmp(my_argv[count], "--tile") == 0)
            && count+1 < my_argc ) {
          (*tile_size) = atoi(my_argv[count+1]);
          count = count + 2;
        }
        else {
          count++;
        }
      }
    }

    // load A matrix
    if (matrix_name == NULL ) {
      matrix_name = default_matrix;
    }
    printf("A will be read from %s\n", matrix_name);
    A = new data_d_matrix();
    A->storage_type = Magma_CSR;
    
    CHECK( data_z_csr_mtx( A, matrix_name ) );


#if 1
  printf("\nA:\n");
  data_zprint_csr(*A);
#endif
    
    // =========================================================================
    // MKL csrilu0  (Benchmark)
    // =========================================================================
    printf("%% MKL csrilu0  (Benchmark)\n");

    Lmkl = new data_d_matrix();
    Umkl = new data_d_matrix();

    data_d_matrix Amkl = {Magma_CSR};
    data_zmconvert((*A), &Amkl, Magma_CSR, Magma_CSR);

    dataType wstart = omp_get_wtime();
    //this function modifies Amkl
    //convert matrix to ILU0 form
    CHECK( data_dcsrilu0_mkl( &Amkl ) );

    dataType wend = omp_get_wtime();
    printf("%% MKL csrilu0 required %f wall clock seconds as measured by omp_get_wtime()\n", wend-wstart );

    //separate Amkl(in LU form) into L and U form

    Lmkl->storage_type = Magma_CSRL;
    Lmkl->diagorder_type = Magma_UNITY;

    data_zmconvert(Amkl, Lmkl, Magma_CSR, Magma_CSRL);

    printf("test if Lmkl is lower: ");
    data_zcheckupperlower( Lmkl );
    printf(" done.\n");

    Umkl->storage_type = Magma_CSRU;
    Umkl->diagorder_type = Magma_VALUE;
    data_zmconvert(Amkl, Umkl, Magma_CSR, Magma_CSRU);
    printf("test if Umkl is upper: ");
    data_zcheckupperlower( Umkl );
    printf(" done.\n");
    //copy A to LU
    data_d_matrix LUmkl = {Magma_CSR};
    data_zmconvert(Amkl, &LUmkl, Magma_CSR, Magma_CSR);


#if 1
  printf("L mkl:\n");
  data_zprint_csr(*Lmkl);
  printf("\nU mkl:\n");
  data_zprint_csr(*Umkl);
#endif
    
    Amklres = new dataType();
    (*Amklres) = 0.0;
    Amklnonlinres = new dataType();
    (*Amklnonlinres) = 0.0;
    // Check ||A-LU||_Frobenius for the whole and restricted to A's sparsity pattern
    data_zilures( (*A), (*Lmkl), (*Umkl), &LUmkl, Amklres, Amklnonlinres);
    printf("MKL_csrilu0_res = %e\n", (*Amklres));
    printf("MKL_csrilu0_nonlinres = %e\n", (*Amklnonlinres));

    dataType tol=1.0e-12;
    if( (*Amklres)<tol) (*Amklres)=tol;
    if((*Amklnonlinres)<tol)(*Amklnonlinres)=tol;
    
    data_zmfree( &LUmkl );

    data_zmfree( &Amkl );
    printf("\n");
    fflush(stdout);
  }

  // per-test-case tear-down
  static void TearDownTestCase() {
    data_zmfree( A );
    data_zmfree( Lmkl );
    data_zmfree( Umkl );
    delete A;
    delete Lmkl;
    delete Umkl;
    delete Amklres;
    delete Amklnonlinres;
    if (tile_size != NULL ) {
      delete tile_size;
    }
    if (block_size != NULL ) {
      delete block_size;
    }
    if (matchfactor != NULL ) {
      delete matchfactor;
    }
  }

  // per-test set-up and tear-down
  virtual void SetUp() {}
  virtual void TearDown() {}

  // shared by all tests
  static data_d_matrix* A;
  static data_d_matrix* Lmkl;
  static data_d_matrix* Umkl;
  static dataType* Amklres;
  static dataType* Amklnonlinres;
  static int* tile_size;
  static int* block_size;
  static int* matchfactor;
};

data_d_matrix* iLUTest::A = NULL;
data_d_matrix* iLUTest::Lmkl = NULL;
data_d_matrix* iLUTest::Umkl = NULL;
dataType* iLUTest::Amklres = NULL;
dataType* iLUTest::Amklnonlinres = NULL;
int* iLUTest::tile_size = NULL;
int* iLUTest::block_size = NULL;
int* iLUTest::matchfactor = NULL;



#if 0
TEST_F(iLUTest, PariLUv0_0) {
  // =========================================================================
  // PariLU v0.0
  // =========================================================================
  printf("%% PariLU v0.0\n");

  data_d_matrix L = {Magma_CSRL};
  data_d_matrix U = {Magma_CSRU};
  // Separate the strictly lower and upper elements
  // into L, and U respectively.
  data_PariLU_v0_0( A, &L, &U);
  // Check ||A-LU||_Frobenius
  dataType Ares = 0.0;
  dataType Anonlinres = 0.0;
  data_d_matrix LU = {Magma_CSR};
  data_zmconvert((*A), &LU, Magma_CSR, Magma_CSR);
  data_zilures((*A), L, U, &LU, &Ares, &Anonlinres);

  printf("PariLU_v0_0-5_csrilu0_res = %e\n", Ares);
  printf("PariLU_v0_0-5_csrilu0_nonlinres = %e\n", Anonlinres);
  fflush(stdout);

  EXPECT_LE( Ares, (*Amklres)*(*matchfactor) );
  EXPECT_LE( Anonlinres, (*Amklnonlinres)*(*matchfactor) );
  printf("\n");
  
#if 0
  printf("L:\n");
  data_zprint_csr(L);
  printf("\nU:\n");
  data_zprint_csr(U);
#endif
  
  data_zmfree( &L );
  data_zmfree( &U );
  data_zmfree( &LU );
  // =========================================================================
}
#endif

#if 0
TEST_F(iLUTest, PariLUv0_3) {
  // =========================================================================
  // PariLU v0.3
  // =========================================================================
  printf("%% PariLU v0.3\n");

  data_d_matrix L = {Magma_CSRL};
  data_d_matrix U = {Magma_CSRU};
  //data_d_matrix U = {Magma_CSCU};
  dataType reduction = 1.0e-20;
  data_d_preconditioner_log parilu_log;
  
  // Separate the strictly lower and upper elements
  // into L, and U respectively.
  data_PariLU_v0_3( A, &L, &U, reduction, &parilu_log );
  
  // Check ||A-LU||_Frobenius
  dataType Ares = 0.0;
  dataType Anonlinres = 0.0;
  data_d_matrix LU = {Magma_CSR};
  data_zmconvert((*A), &LU, Magma_CSR, Magma_CSR);
  data_zilures((*A), L, U, &LU, &Ares, &Anonlinres);
  printf("PariLU_v0_3_omp_num_threads = %d\n", parilu_log.omp_num_threads );
  printf("PariLU_v0_3_sweeps = %d\n", parilu_log.sweeps );
  printf("PariLU_v0_3_tol = %e\n", parilu_log.tol );
  printf("PariLU_v0_3_A_Frobenius = %e\n", parilu_log.A_Frobenius );
  printf("PariLU_v0_3_generation_time = %e\n", parilu_log.precond_generation_time );
  printf("PariLU_v0_3_initial_residual = %e\n", parilu_log.initial_residual );
  printf("PariLU_v0_3_initial_nonlinear_residual = %e\n", parilu_log.initial_nonlinear_residual );
  printf("PariLU_v0_3_csrilu0_res = %e\n", Ares);
  printf("PariLU_v0_3_csrilu0_nonlinres = %e\n", Anonlinres);
  
#if 0
  printf("L:\n");
  data_zprint_csr(L);
  printf("\nU:\n");
  data_zprint_csr(U);
#endif

  fflush(stdout);

  EXPECT_LE( Ares, (*Amklres)*(*matchfactor) );
  EXPECT_LE( Anonlinres, (*Amklnonlinres)*(*matchfactor) );

  data_zmfree( &L );
  data_zmfree( &U );
  data_zmfree( &LU );
  // =========================================================================
}
#endif

#if 0
TEST_F(iLUTest, PariLUv0_3_bcsr) {

  // =========================================================================
  // PariLU v0.3
  // =========================================================================
  printf("%% PariLU v0.3 BCSR\n");
  data_d_matrix A_BCSR = {Magma_BCSR};
  data_d_matrix L = {Magma_BCSRL};
  //data_d_matrix U = {Magma_BCSRU};
  data_d_matrix U = {Magma_BCSCU};
  data_d_matrix LU = {Magma_BCSR};

  //copy A csr to A BCSR
  //do this twice to get ordered rows
  A_BCSR.blocksize= *block_size;
  
  data_zmconvert((*A), &A_BCSR, Magma_CSR, Magma_BCSR);
  sort_csr_rows(&A_BCSR);
  data_rowindex(&A_BCSR, &(A_BCSR.rowidx) );

  //printf("A_BCSR:\n");
  //data_zprint_bcsr(&A_BCSR);

#if 0
  printf("L:\n");
  data_zprint_bcsr(&L);
  printf("\nU:\n");
  data_zprint_bcsr(&U);
#endif

  dataType reduction = 1.0e-20;
  data_d_preconditioner_log parilu_log;

  // Separate the strictly lower and upper elements
  // into L, and U respectively.
  //write bcsr version for this function
  //ultimately work this in to a single function 
  data_PariLU_v0_3_bcsr( &A_BCSR, &L, &U, reduction, &parilu_log );
  
#if 0
  printf("L:\n");
  data_zprint_bcsr(&L);
  printf("\nU:\n");
  data_zprint_bcsr(&U);
#endif
  
  // Check ||A-LU||_Frobenius
  dataType Ares = 0.0;
  dataType Anonlinres = 0.0;
  LU.blocksize=A_BCSR.blocksize;//set blocksize for LU
  
  data_zmconvert((*A), &LU, Magma_CSR, Magma_BCSR);

  data_zilures_bcsr(A_BCSR, L, U, &LU, &Ares, &Anonlinres);

  printf("PariLU_v0_3_bcsr_omp_num_threads = %d\n", parilu_log.omp_num_threads );
  printf("PariLU_v0_3_bcsr_sweeps = %d\n", parilu_log.sweeps );
  printf("PariLU_v0_3_bcsr_tol = %e\n", parilu_log.tol );
  printf("PariLU_v0_3_bcsr_A_Frobenius = %e\n", parilu_log.A_Frobenius );
  printf("PariLU_v0_3_bcsr_generation_time = %e\n", parilu_log.precond_generation_time );
  printf("PariLU_v0_3_bcsr_initial_residual = %e\n", parilu_log.initial_residual );
  printf("PariLU_v0_3_bcsr_initial_nonlinear_residual = %e\n", parilu_log.initial_nonlinear_residual );
  printf("PariLU_v0_3_bcsr_csrilu0_res = %e\n", Ares);
  printf("PariLU_v0_3_bcsr_csrilu0_nonlinres = %e\n", Anonlinres);

  fflush(stdout);

  EXPECT_LE( Ares, (*Amklres)*(*matchfactor) );
  EXPECT_LE( Anonlinres, (*Amklnonlinres)*(*matchfactor) );

  data_zmfree( &A_BCSR);
  data_zmfree( &L );
  data_zmfree( &U );
  data_zmfree( &LU );
  // =========================================================================
}
#endif

#if 1
TEST_F(iLUTest, ILU0_bcsr_v1_0) {

  // =========================================================================
  // ILU0 bcsr v1.0
  // =========================================================================
  printf("%% ILU0 v1.0 BCSR\n");
  data_d_matrix A_BCSR = {Magma_BCSR};
  data_d_matrix L = {Magma_BCSRL};
  //data_d_matrix U = {Magma_BCSRU};
  data_d_matrix U = {Magma_BCSCU};
  data_d_matrix LU = {Magma_BCSR};

  //copy A csr to A BCSR
  //do this twice to get ordered rows
  A_BCSR.blocksize= *block_size;
  data_zmconvert((*A), &A_BCSR, Magma_CSR, Magma_BCSR);
  sort_csr_rows(&A_BCSR);
  data_rowindex(&A_BCSR, &(A_BCSR.rowidx) );

  //printf("A_BCSR:\n");
  //data_zprint_bcsr(&A_BCSR);

  dataType reduction = 1.0e-20;
  data_d_preconditioner_log parilu_log;

  // Separate the strictly lower and upper elements
  // into L, and U respectively.
  //write bcsr version for this function
  //ultimately work this in to a single function 
  data_ILU0_bcsr_v1_0( &A_BCSR, &L, &U, reduction, &parilu_log );
  
#if 0
  printf("L bcsr:\n");
  data_zprint_bcsr(&L);
  printf("\nU bcsr:\n");
  data_zprint_bcsr(&U);
#endif
  
  // Check ||A-LU||_Frobenius
  dataType Ares = 0.0;
  dataType Anonlinres = 0.0;
  LU.blocksize=A_BCSR.blocksize;//set blocksize for LU
  data_zmconvert((*A), &LU, Magma_CSR, Magma_BCSR);
  data_zilures_bcsr(A_BCSR, L, U, &LU, &Ares, &Anonlinres);

  printf("ILU_v1_0_bcsr_omp_num_threads = %d\n", parilu_log.omp_num_threads );
  printf("ILU_v1_0_bcsr_A_Frobenius = %e\n", parilu_log.A_Frobenius );
  printf("ILU_v1_0_bcsr_generation_time = %e\n", parilu_log.precond_generation_time );
  printf("ILU_v1_0_bcsr_initial_residual = %e\n", parilu_log.initial_residual );
  printf("ILU_v1_0_bcsr_initial_nonlinear_residual = %e\n", parilu_log.initial_nonlinear_residual );
  printf("ILU_v1_0_bcsr_csrilu0_res = %e\n", Ares);
  printf("ILU_v1_0_bcsr_csrilu0_nonlinres = %e\n", Anonlinres);
  
  fflush(stdout);

  EXPECT_LE( Ares, (*Amklres)*(*matchfactor) );
  EXPECT_LE( Anonlinres, (*Amklnonlinres)*(*matchfactor) );

  data_zmfree( &A_BCSR);
  data_zmfree( &L );
  data_zmfree( &U );
  data_zmfree( &LU );
  // =========================================================================
}
#endif

#if 1
TEST_F(iLUTest, ILUK) {

  // =========================================================================
  // ILU0 bcsr v1.0
  // =========================================================================
  printf("%% ILUK\n");
  data_d_matrix A_BCSR = {Magma_BCSR};
  data_d_matrix A_k= {Magma_BCSR};
  data_d_matrix L = {Magma_BCSRL};
  data_d_matrix U = {Magma_BCSRU};
  //data_d_matrix U = {Magma_BCSCU};
  data_d_matrix LU = {Magma_BCSR};

  //copy A csr to A BCSR
  //do this twice to get ordered rows
  A_BCSR.blocksize= *block_size;
  data_zmconvert(*A, &A_BCSR, Magma_CSR, Magma_BCSR);
  sort_csr_rows(&A_BCSR);
  data_rowindex(&A_BCSR, &(A_BCSR.rowidx) );

  //printf("A_BCSR:\n");
  //data_zprint_bcsr(&A_BCSR);

  int k=2;
  dataType iluk_start, iluk_end;
  iluk_start = omp_get_wtime();
  data_ILUK(&A_BCSR, &A_k, k);
  iluk_end = omp_get_wtime();
  dataType iluk_time =   (dataType) (iluk_end - iluk_start);
  printf("A nnz= %d\n", A_BCSR.nnz);
  printf("Bandwidth A    = %d\n", A_BCSR.diameter);
  printf("Bandwidth A_new= %d\n", A_k.diameter);
  printf("A_k nnz= %d\n", A_k.nnz);

  sort_csr_rows(&A_k);
  data_rowindex(&A_k, &(A_k.rowidx) );

#if 1 //note: writes pattern and only first value in subblock 
  data_zwrite_csr_mtx(A_BCSR, MagmaRowMajor,"A_orig.dat");
  data_zwrite_csr_mtx(A_k, MagmaRowMajor,"A_k.dat");
#endif

  dataType reduction = 1.0e-20;
  data_d_preconditioner_log ilu_log;

  // Separate the strictly lower and upper elements
  // into L, and U respectively.
  //write bcsr version for this function
  //ultimately work this in to a single function 
DEV_CHECKPT
   data_ILU0_bcsr_v1_0( &A_k, &L, &U, reduction, &ilu_log );
DEV_CHECKPT
  
#if 0
  printf("L bcsr:\n");
  data_zprint_bcsr(&L);
  printf("\nU bcsr:\n");
  data_zprint_bcsr(&U);
#endif
  
  // Check ||A-LU||_Frobenius
  dataType Ares = 0.0;
  dataType Anonlinres = 0.0;
  LU.blocksize=A_k.blocksize;//set blocksize for LU
  data_zmconvert( *A, &LU, Magma_CSR, Magma_BCSR);
  data_zilures_bcsr( A_k, L, U, &LU, &Ares, &Anonlinres);

  printf("ILU(%d)_v1_0_bcsr_omp_num_threads = %d\n",k, ilu_log.omp_num_threads );
  printf("ILU(%d)_v1_0_bcsr_A_Frobenius = %e\n",k, ilu_log.A_Frobenius );
  printf("ILU(%d)_generation_time = %e\n",k,  iluk_time);
  printf("ILU(%d)_v1_0_bcsr_generation_time = %e\n",k, ilu_log.precond_generation_time );
  printf("ILU(%d)_v1_0_bcsr_initial_residual = %e\n",k, ilu_log.initial_residual );
  printf("ILU(%d)_v1_0_bcsr_initial_nonlinear_residual = %e\n",k, ilu_log.initial_nonlinear_residual );
  printf("ILU(%d)_v1_0_bcsr_csrilu0_res = %e\n",k, Ares);
  printf("ILU(%d)_v1_0_bcsr_csrilu0_nonlinres = %e\n",k, Anonlinres);
  
  fflush(stdout);

  EXPECT_LE( Ares, (*Amklres)*(*matchfactor) );
  EXPECT_LE( Anonlinres, (*Amklnonlinres)*(*matchfactor) );

  data_zmfree( &L );
  data_zmfree( &U );
  data_zmfree( &LU );

  data_zmfree( &A_BCSR);
  data_zmfree( &A_k );

  // =========================================================================

}
#endif





#if 1
TEST_F(iLUTest, ILU_CuthillMcKee) {

  // =========================================================================
  // ILU0 bcsr v1.0
  // =========================================================================
  printf("%%  ILU_CuthillMcKee\n");
  data_d_matrix A_BCSR = {Magma_BCSR};
  data_d_matrix A_k= {Magma_BCSR};
  data_d_matrix A_reorder= {Magma_BCSR};
  data_d_matrix L = {Magma_BCSRL};
  data_d_matrix U = {Magma_BCSRU};
  //data_d_matrix U = {Magma_BCSCU};
  data_d_matrix LU = {Magma_BCSR};
  int* P;
  int* Pinv;
  int k=0;
  //copy A csr to A BCSR
  //do this twice to get ordered rows
  A_BCSR.blocksize= *block_size;
  printf("A_BCSR.blocksize=%d\n",A_BCSR.blocksize);
  data_zmconvert(*A, &A_BCSR, Magma_CSR, Magma_BCSR);
  sort_csr_rows(&A_BCSR);
  data_rowindex(&A_BCSR, &(A_BCSR.rowidx) );
  printf("A_BCSR.num_rows=%d\n", A_BCSR.num_rows); 
  data_zmcopy(A_BCSR,&A_reorder);
  data_zdiameter(&A_BCSR);
  printf("A_BCSR->diameter=%d\n",A_BCSR.diameter);

    
#if 0
  printf("A bcsr:\n");
  data_zprint_bcsr(&A_BCSR);
  //printf("\nU bcsr:\n");
  //data_zprint_bcsr(&U);
#endif

#if 0  
  //printf("A_BCSR:\n");
  //data_zprint_bcsr(&A_BCSR);
  dataType iluk_start, iluk_end;
  iluk_start = omp_get_wtime();
  data_ILUK(&A_BCSR, &A_k, k);
  iluk_end = omp_get_wtime();
  dataType iluk_time =   (dataType) (iluk_end - iluk_start);
  printf("A nnz= %d\n", A_BCSR.nnz);
  printf("Bandwidth A    = %d\n", A_BCSR.diameter);
  printf("Bandwidth A_new= %d\n", A_k.diameter);
  printf("A_k nnz= %d\n", A_k.nnz);

  sort_csr_rows(&A_k);
  data_rowindex(&A_k, &(A_k.rowidx) );
#endif

#if 1
DEV_CHECKPT
    int reorder =2;//RCM
  LACE_CALLOC(P, A_BCSR.num_rows);
  LACE_CALLOC(Pinv, A_BCSR.num_rows);
  data_sparse_reorder(&A_reorder, P, Pinv, reorder);
  data_zdiameter(&A_reorder);
  printf("A_reorder->diameter=%d\n",A_reorder.diameter);
  //for(int i=0; i<A_reorder.num_rows; ++i)
  //{
  // printf("P[%d]: %d\n",i,P[i]);
  //}
DEV_CHECKPT
#endif

  printf("A_reorder.blocksize=%d\n",A_reorder.blocksize);
  
  
#if 1 //note: writes pattern and only first value in subblock 
  data_zwrite_csr_mtx(A_BCSR, MagmaRowMajor,"A_orig.dat");
  data_zwrite_csr_mtx(A_reorder, MagmaRowMajor,"A_reorder.dat");
#endif
DEV_CHECKPT




  
  dataType reduction = 1.0e-20;
  data_d_preconditioner_log ilu_log;

  // Separate the strictly lower and upper elements
  // into L, and U respectively.
  //write bcsr version for this function
  //ultimately work this in to a single function 
DEV_CHECKPT
   data_ILU0_bcsr_v1_0( &A_reorder, &L, &U, reduction, &ilu_log );
DEV_CHECKPT
  
#if 0
  printf("L bcsr:\n");
  data_zprint_bcsr(&L);
  printf("\nU bcsr:\n");
  data_zprint_bcsr(&U);
#endif
  
  // Check ||A-LU||_Frobenius
  dataType Ares = 0.0;
  dataType Anonlinres = 0.0;
DEV_CHECKPT
  LU.blocksize=A_reorder.blocksize;//set blocksize for LU
DEV_CHECKPT
  data_zmconvert( *A, &LU, Magma_CSR, Magma_BCSR);
DEV_CHECKPT
  data_zilures_bcsr( A_reorder, L, U, &LU, &Ares, &Anonlinres);
DEV_CHECKPT

  printf("ILU(%d)_v1_0_bcsr_omp_num_threads = %d\n",k, ilu_log.omp_num_threads );
  printf("ILU(%d)_v1_0_bcsr_A_Frobenius = %e\n",k, ilu_log.A_Frobenius );
  //printf("ILU(%d)_generation_time = %e\n",k,  iluk_time);
  printf("ILU(%d)_v1_0_bcsr_generation_time = %e\n",k, ilu_log.precond_generation_time );
  printf("ILU(%d)_v1_0_bcsr_initial_residual = %e\n",k, ilu_log.initial_residual );
  printf("ILU(%d)_v1_0_bcsr_initial_nonlinear_residual = %e\n",k, ilu_log.initial_nonlinear_residual );
  printf("ILU(%d)_v1_0_bcsr_csrilu0_res = %e\n",k, Ares);
  printf("ILU(%d)_v1_0_bcsr_csrilu0_nonlinres = %e\n",k, Anonlinres);
  
  fflush(stdout);

  EXPECT_LE( Ares, (*Amklres)*(*matchfactor) );
  EXPECT_LE( Anonlinres, (*Amklnonlinres)*(*matchfactor) );

  data_zmfree( &L );
  data_zmfree( &U );
  data_zmfree( &LU );

  data_zmfree( &A_BCSR);
  data_zmfree( &A_reorder );

  // =========================================================================

}
#endif





#if USE_CUDA
TEST_F(iLUTest, PariLUv0_3_gpu) {
  // =========================================================================
  // PariLU v0.3 GPU
  // =========================================================================
  printf("%% PariLU v0.3 GPU\n");
  data_d_matrix L = {Magma_CSRL};
  //data_d_matrix U = {Magma_CSRU};
  data_d_matrix U = {Magma_CSCU};
  dataType reduction = 1.0e-15;
  data_d_preconditioner_log parilu_log;
  data_PariLU_v0_3_gpu( A, &L, &U, reduction, &parilu_log, *tile_size);
  // Check ||A-LU||_Frobenius
  dataType Ares = 0.0;
  dataType Anonlinres = 0.0;
  data_d_matrix LU = {Magma_CSR};
  data_zmconvert((*A), &LU, Magma_CSR, Magma_CSR);
  data_zilures((*A), L, U, &LU, &Ares, &Anonlinres);
  printf("PariLU_v0_3_omp_num_threads = %d\n", parilu_log.omp_num_threads );
  printf("PariLU_v0_3_sweeps = %d\n", parilu_log.sweeps );
  printf("PariLU_v0_3_tol = %e\n", parilu_log.tol );
  printf("PariLU_v0_3_A_Frobenius = %e\n", parilu_log.A_Frobenius );
  printf("PariLU_v0_3_generation_time = %e\n", parilu_log.precond_generation_time );
  printf("PariLU_v0_3_initial_residual = %e\n", parilu_log.initial_residual );
  printf("PariLU_v0_3_initial_nonlinear_residual = %e\n", parilu_log.initial_nonlinear_residual );
  fflush(stdout);

  EXPECT_LE( Ares, (*Amklres)*(*matchfactor) );
  EXPECT_LE( Anonlinres, (*Amklnonlinres)*(*matchfactor) );
  data_zmfree( &L );
  data_zmfree( &U );
  data_zmfree( &LU );
  // =========================================================================
}
#endif
