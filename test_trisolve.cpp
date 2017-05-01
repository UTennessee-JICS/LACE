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
  dataType mklwcsrtrsvtime = 0.0;
  dataType parwcsrtrsvtime = 0.0;
  
  if (argc < 3) {
    printf("Usage %s <matrix> <rhs vector> [permute]", argv[0] );
    //printf("[diagonal scaling] [abs/rel] [GMRES_tolerance] [restart] ");
    //printf("[maxiter] [precond_choice] [reduction]\n");
    return 1;
  }
  else {
    sparse_filename = argv[1];
    rhs_filename = argv[2];
    //output_dir = argv[3];
    sparse_basename = basename( sparse_filename );
    char *ext;
    ext = strrchr( sparse_basename, '.');
    strncpy( sparse_name, sparse_basename, int(ext - sparse_basename) );
    printf("File %s basename %s name %s \n", 
      sparse_filename, sparse_basename, sparse_name );
    printf("rhs vector name is %s \n", rhs_filename ); 
    //printf("Output directory is %s\n", output_dir );
    //strcpy( output_basename, output_dir );
    //strcat( output_basename, sparse_name );
    //printf("Output file base name is %s\n", output_basename );
  }
	data_d_matrix Asparse = {Magma_CSR};
  CHECK( data_z_csr_mtx( &Asparse, sparse_filename ) );
	data_d_matrix rhs_vector = {Magma_DENSE};
	rhs_vector.major = MagmaRowMajor;
	
	// Setup rhs
	if ( strcmp( rhs_filename, "ONES" ) == 0 ) {
	  printf("creating a vector of %d ones for the rhs.\n", Asparse.num_rows);
	  rhs_vector.num_rows = Asparse.num_rows;
    rhs_vector.num_cols = 1;
    rhs_vector.ld = 1;
    rhs_vector.nnz = Asparse.num_rows;
    rhs_vector.val = (dataType*) calloc( Asparse.num_rows, sizeof(dataType) );
    #pragma omp parallel 
    {
      #pragma omp for nowait
      for (int i=0; i<Asparse.num_rows; i++) {
        rhs_vector.val[i] = 1.0;
      }
    }
	}
	else {
	  
	  CHECK( data_z_dense_mtx( &rhs_vector, rhs_vector.major, rhs_filename ) );
  }
  
  if (argc > 3) {
    permute = atoi(argv[3]); 
    printf("reading permute %d\n", permute);
  }
  
  DEV_CHECKPT
  
  data_d_matrix L = {Magma_CSRL};
  L.diagorder_type = Magma_UNITY;
  //L.diagorder_type = Magma_VALUE;
  L.fill_mode = MagmaLower;
  data_zmconvert( Asparse, &L, Magma_CSR, Magma_CSRL );
  int* il;
  int* jl;
  il = (int*) calloc( (L.num_rows+1), sizeof(int) );
  jl = (int*) calloc( L.nnz, sizeof(int) );
  
  // TODO: create indexing wrapper functions 
  #pragma omp parallel 
  {
    #pragma omp for nowait
    for (int i=0; i<L.num_rows+1; i++) {
    	il[i] = L.row[i] + 1;	
    }
    #pragma omp for nowait
    for (int i=0; i<L.nnz; i++) {
    	jl[i] = L.col[i] + 1;
    }
  }
  DEV_CHECKPT
  
  data_d_matrix y = {Magma_DENSE};
  y.num_rows = Asparse.num_rows;
  y.num_cols = 1;
  y.ld = 1;
  y.nnz = y.num_rows;
  LACE_CALLOC(y.val, y.num_rows);
  
  
  data_d_matrix y_mkl = {Magma_DENSE};
  y_mkl.num_rows = Asparse.num_rows;
  y_mkl.num_cols = 1;
  y_mkl.ld = 1;
  y_mkl.nnz = y_mkl.num_rows;
  LACE_CALLOC(y_mkl.val, y_mkl.num_rows);
  
  
  //for (int i=0; i<L.row[4+1]; i++) {
  //  printf("L.val[%d] = %e\n", i, L.val[i] );
  //}
  
  cvar1='L';
  cvar='N';
  cvar2='U';
  wcsrtrsvstart = omp_get_wtime();
  mkl_dcsrtrsv(&cvar1, &cvar, &cvar2, &L.num_rows, 
    L.val, il, jl, rhs_vector.val, y_mkl.val);
	wcsrtrsvend = omp_get_wtime();
	mklwcsrtrsvtime = wcsrtrsvend - wcsrtrsvstart;
	
  //for (int i=0; i<L.row[4+1]; i++) {
  //  printf("L.val[%d] = %e\n", i, L.val[i] );
  //}
  
  //for (int i=0; i<L.num_rows; i++) {
  //  printf("y_mkl[%d] = %e\n", i, y_mkl.val[i] );
  //}
  //for (int i=0; i<Asparse.num_rows; i++) {
  //  printf("rhs_vector[%d] = %e\n", i, rhs_vector.val[i] );
  //}
  
  if (permute == 0) { 
    wcsrtrsvstart = omp_get_wtime();
    data_forward_solve( &L, &y, &rhs_vector );
    wcsrtrsvend = omp_get_wtime();
  }
  else if (permute == 1) {
    wcsrtrsvstart = omp_get_wtime();
    data_forward_solve_permute( &L, &y, &rhs_vector );
    wcsrtrsvend = omp_get_wtime();
  }
	parwcsrtrsvtime = wcsrtrsvend - wcsrtrsvstart;

  //printf("i:\ty_mkl\ty\tdiff\n");
  //for (int i=0; i<L.num_rows; i++) {
  ////for (int i=0; i<10; i++) {
  //  //if (y_mkl.val[i] -  y.val[i] > 1.e-8) {
  //  printf("%d:\t%e\t%e\t%e", i, y_mkl.val[i], y.val[i], y_mkl.val[i] -  y.val[i]  );
  //  if (y_mkl.val[i] -  y.val[i] > 1.e15) {  
  //     printf("\tbad\n");
  //  }
  //  else {
  //    printf("\n");
  //  }
  //  //}
  //  //printf("y[%d] = %e\n", i, y.val[i] );
  //  //printf("diff[%d] = %e\n", i, y_mkl.val[i] -  y.val[i] );
  //}
  dataType error = 0.0;
  data_norm_diff_vec( &y, &y_mkl, &error );
  printf("y error = %e\n", error);
  
  //void mkl_dcsrgemv (const char *transa , const MKL_INT *m , const double *a , 
  //  const MKL_INT *ia , const MKL_INT *ja , const double *x , double *y );
  double* Ay_mkl;
  LACE_CALLOC(Ay_mkl, L.num_rows);
  mkl_dcsrgemv(&cvar, &L.num_rows, L.val, il, jl, y_mkl.val, Ay_mkl );
  //void cblas_daxpy (const MKL_INT n, const double a, const double *x, 
  //  const MKL_INT incx, double *y, const MKL_INT incy);
  double negone = -1.0;
  cblas_daxpy(L.num_rows, negone, rhs_vector.val, 1, Ay_mkl, 1 );
  
  
  double* Ay;
  LACE_CALLOC(Ay, L.num_rows);
  mkl_dcsrgemv(&cvar, &L.num_rows, L.val, il, jl, y.val, Ay );
  cblas_daxpy(L.num_rows, negone, rhs_vector.val, 1, Ay, 1 );
  
  double error_mkl = 0.0;
  error = 0.0;
  for (int i=0; i<L.num_rows; i++) {
    //printf("%d:\t%e\t%e\t%e\n", i, Ay[i], Ay_mkl[i], Ay_mkl[i]  -  Ay[i]  );
    error_mkl += pow(Ay_mkl[i], 2);
    error += pow(Ay[i], 2);
  }
  error_mkl = sqrt(error_mkl);
  error = sqrt(error);
  printf("system errors:\n\t error_mkl = %e\terror = %e\n", error_mkl, error);
  printf("MKL time : %e Par time : %e  permute = %d\n", 
    mklwcsrtrsvtime, parwcsrtrsvtime, permute); 
  
  
  data_d_matrix U = {Magma_CSRU};
  U.diagorder_type = Magma_VALUE;
  U.fill_mode = MagmaUpper;
  data_zmconvert( Asparse, &U, Magma_CSR, Magma_CSRU );
  int* iu;
  int* ju;
  iu = (int*) calloc( (U.num_rows+1), sizeof(int) );
  ju = (int*) calloc( U.nnz, sizeof(int) );
  
  // TODO: create indexing wrapper functions 
  #pragma omp parallel 
  {
    #pragma omp for nowait
    for (int i=0; i<U.num_rows+1; i++) {
    	iu[i] = U.row[i] + 1;	
    }
    #pragma omp for nowait
    for (int i=0; i<U.nnz; i++) {
    	ju[i] = U.col[i] + 1;
    }
  }
  DEV_CHECKPT
  
  data_d_matrix x = {Magma_DENSE};
  x.num_rows = Asparse.num_rows;
  x.num_cols = 1;
  x.ld = 1;
  x.nnz = x.num_rows;
  LACE_CALLOC(x.val, x.num_rows);
  
  
  data_d_matrix x_mkl = {Magma_DENSE};
  x_mkl.num_rows = Asparse.num_rows;
  x_mkl.num_cols = 1;
  x_mkl.ld = 1;
  x_mkl.nnz = x_mkl.num_rows;
  LACE_CALLOC(x_mkl.val, x_mkl.num_rows);
  
  cvar1='U';
  cvar='N';
  cvar2='N';
  wcsrtrsvstart = omp_get_wtime();
  mkl_dcsrtrsv(&cvar1, &cvar, &cvar2, &U.num_rows, 
    U.val, iu, ju, rhs_vector.val, x_mkl.val);
  wcsrtrsvend = omp_get_wtime();
	mklwcsrtrsvtime = wcsrtrsvend - wcsrtrsvstart;
	
	if (permute == 0) {
    wcsrtrsvstart = omp_get_wtime();
    data_backward_solve( &U, &x, &rhs_vector );
    wcsrtrsvend = omp_get_wtime();
  }
  else if (permute == 1) {
    wcsrtrsvstart = omp_get_wtime();
    data_backward_solve_permute( &U, &x, &rhs_vector );
    wcsrtrsvend = omp_get_wtime();
  }
	parwcsrtrsvtime = wcsrtrsvend - wcsrtrsvstart;
  
  error = 0.0;
  data_norm_diff_vec( &x, &x_mkl, &error );
  printf("x error = %e\n", error);
  
  //void mkl_dcsrgemv (const char *transa , const MKL_INT *m , const double *a , 
  //  const MKL_INT *ia , const MKL_INT *ja , const double *x , double *y );
  double* Ax_mkl;
  LACE_CALLOC(Ax_mkl, L.num_rows);
  mkl_dcsrgemv(&cvar, &L.num_rows, L.val, il, jl, x_mkl.val, Ax_mkl );
  //void cblas_daxpy (const MKL_INT n, const double a, const double *x, 
  //  const MKL_INT incx, double *y, const MKL_INT incy);
  cblas_daxpy(L.num_rows, negone, rhs_vector.val, 1, Ax_mkl, 1 );
  
  
  double* Ax;
  LACE_CALLOC(Ax, L.num_rows);
  mkl_dcsrgemv(&cvar, &L.num_rows, L.val, il, jl, x.val, Ax );
  cblas_daxpy(L.num_rows, negone, rhs_vector.val, 1, Ax, 1 );
  
  error_mkl = 0.0;
  error = 0.0;
  for (int i=0; i<L.num_rows; i++) {
    //printf("%d:\t%e\t%e\t%e\n", i, Ax[i], Ax_mkl[i], Ax_mkl[i]  -  Ax[i]  );
    error_mkl += pow(Ax_mkl[i], 2);
    error += pow(Ax[i], 2);
  }
  error_mkl = sqrt(error_mkl);
  error = sqrt(error);
  printf("system errors:\n\t error_mkl = %e\terror = %e\n", error_mkl, error);
  printf("MKL time : %e Par time : %e  permute = %d\n", 
    mklwcsrtrsvtime, parwcsrtrsvtime, permute); 
  
  
  int num_threads = 0;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  printf("omp_get_num_threads = %d\n", num_threads);
  
  data_zmfree( &Asparse );
	data_zmfree( &rhs_vector );
  data_zmfree( &y );
  data_zmfree( &y_mkl );
  data_zmfree( &x );
  data_zmfree( &x_mkl );
  data_zmfree( &L );
  data_zmfree( &U );
  
  
  
  //testing::InitGoogleTest(&argc, argv);
  //return RUN_ALL_TESTS();
  
  printf("done\n");
  fflush(stdout); 
  return 0;
  
}