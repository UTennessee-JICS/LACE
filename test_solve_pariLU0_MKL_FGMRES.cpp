
// begun from:
// http://sep.stanford.edu/sep/claudio/Research/Prst_ExpRefl/ShtPSPI/intel/mkl/10.0.3.020/examples/solver/source/dcsrilu0_exampl1.c

/********************************************************************************
                                INTEL CONFIDENTIAL
     Copyright(C) 2006-2008 Intel Corporation. All Rights Reserved.
     The source code contained  or  described herein and all documents related to
     the source code ("Material") are owned by Intel Corporation or its suppliers
     or licensors.  Title to the  Material remains with  Intel Corporation or its
     suppliers and licensors. The Material contains trade secrets and proprietary
     and  confidential  information of  Intel or its suppliers and licensors. The
     Material  is  protected  by  worldwide  copyright  and trade secret laws and
     treaty  provisions. No part of the Material may be used, copied, reproduced,
     modified, published, uploaded, posted, transmitted, distributed or disclosed
     in any way without Intel's prior express written permission.
     No license  under any  patent, copyright, trade secret or other intellectual
     property right is granted to or conferred upon you by disclosure or delivery
     of the Materials,  either expressly, by implication, inducement, estoppel or
     otherwise.  Any  license  under  such  intellectual property  rights must be
     express and approved by Intel in writing.
  
  *******************************************************************************
    Content:
    Intel MKL example of RCI Flexible Generalized Minimal RESidual method with
    ILU0 Preconditioner
  *******************************************************************************
  
  ---------------------------------------------------------------------------
    Example program for solving non-degenerate system of equations.
    Full functionality of RCI FGMRES solver is exploited. Example shows how
    ILU0 preconditioner accelerates the solver by reducing the number of
    iterations.
  ---------------------------------------------------------------------------*/

#include "include/sparse.h"
#include <stdio.h>
#include <stdlib.h>
#include <libgen.h>
#include <string.h>
#include <vector>
#include <omp.h>
#include <float.h> 
#include "math.h"
#include "mkl_blas.h"
#include "mkl_spblas.h"
#include "mkl_rci.h"

//#define N 4
#define size 128

int main(int argc, char* argv[])
{
/*---------------------------------------------------------------------------
   Define arrays for the upper triangle of the coefficient matrix
   Compressed sparse row storage is used for sparse representation
  ---------------------------------------------------------------------------*/
	//MKL_INT ia[5]={1,4,7,10,13};
	//MKL_INT ja[12]={1,2,3,1,2,4,1,3,4,2,3,4};
	//double A[12]={4.,-1.,-1.,-1.,4.,-1.,-1.,4.,-1.,-1.,-1.,4.};
	
	// begin with a square matrix A
  char* sparse_filename;
  char* rhs_filename;
  char* sparse_basename;
  char sparse_name[256];
  char* output_dir;
  char output_basename[256];
  //char output_L[256];
  //char output_U[256];
  
  data_scale_t scaling  = Magma_NOSCALE;
  
  // for PariLU0
  dataType user_precond_reduction = 1.0e-15;
  
  // for FGMRES
  int user_restart = 150;
  int user_maxiter = 150;
  int user_gmres_tol_type = 1;
  int user_precond_choice = 0;
  dataType user_rel_tol = 1.0e-6;
  
  // for timing of MKL csriLU0 and FGMRES
  dataType wcsrilu0start = 0.0;
  dataType wcsrilu0end = 0.0;
  dataType ompwcsrilu0time = 0.0;
  dataType wfgmresstart = 0.0;
  dataType wfgmresend = 0.0;
  dataType ompwfgmrestime = 0.0;
  
  if (argc < 4) {
    printf("Usage %s <matrix> <rhs vector> <output directory> ", argv[0] );
    printf("[diagonal scaling] [abs/rel] [GMRES_tolerance] [restart] ");
    printf("[maxiter] [precond_choice] [reduction]\n");
    return 1;
  }
  else {
    sparse_filename = argv[1];
    rhs_filename = argv[2];
    output_dir = argv[3];
    sparse_basename = basename( sparse_filename );
    char *ext;
    ext = strrchr( sparse_basename, '.');
    strncpy( sparse_name, sparse_basename, int(ext - sparse_basename) );
    printf("File %s basename %s name %s \n", 
      sparse_filename, sparse_basename, sparse_name );
    printf("rhs vector name is %s \n", rhs_filename ); 
    printf("Output directory is %s\n", output_dir );
    strcpy( output_basename, output_dir );
    strcat( output_basename, sparse_name );
    printf("Output file base name is %s\n", output_basename );
  }
	data_d_matrix Asparse = {Magma_CSR};
  CHECK( data_z_csr_mtx( &Asparse, sparse_filename ) );
	data_d_matrix rhs_vector = {Magma_DENSE};
	rhs_vector.major = MagmaRowMajor;
	data_d_matrix scaling_factors = {Magma_DENSE};
	scaling_factors.major = MagmaRowMajor;
	
	data_d_matrix A_org = {Magma_CSR};
	CHECK( data_zmconvert( Asparse, &A_org, Magma_CSR, Magma_CSR ) );
	
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
  
  
	data_d_matrix rhs_org = {Magma_DENSE};
	rhs_org.major = MagmaRowMajor;
	CHECK( data_zmconvert( rhs_vector, &rhs_org, Magma_DENSE, Magma_DENSE ) );
  
  // Optionally Scale matrix 
  if ( argc >= 5 ) {
    if ( strcmp( argv[4], "UNITROW" ) == 0 ) {
      printf("rescaling UNITROW ");
      scaling = Magma_UNITROW;
      data_zmscale_matrix_rhs( &Asparse, &rhs_vector, &scaling_factors, Magma_UNITROW );
      //data_zwrite_csr( &Asparse );
      printf("done.\n");
    }
    else if ( strcmp( argv[4], "UNITDIAG" ) == 0 ) {
      printf("rescaling UNITDIAG ");
      scaling = Magma_UNITDIAG;
      data_zmscale_matrix_rhs( &Asparse, &rhs_vector, &scaling_factors, Magma_UNITDIAG );
      //data_zwrite_csr( &Asparse );
      printf("done.\n");
    }
    else if ( strcmp( argv[4], "UNITROWCOL" ) == 0 ) {
      printf("rescaling UNITROWCOL\n");
      scaling = Magma_UNITROWCOL;
      data_zmscale_matrix_rhs( &Asparse, &rhs_vector, &scaling_factors, Magma_UNITROWCOL );
      //data_zwrite_csr( &Asparse );
    }
    else if ( strcmp( argv[4], "UNITDIAGCOL" ) == 0 ) {
      printf("rescaling UNITDIAGCOL\n");
      scaling = Magma_UNITDIAGCOL;
      data_zmscale_matrix_rhs( &Asparse, &rhs_vector, &scaling_factors, Magma_UNITDIAGCOL );
      //data_zwrite_csr( &Asparse );
    }
  }
  
  // Select absolute=0 or relative=1 tolerance stopping citeria for FGMRES
  if ( argc >= 6 ) {
    user_gmres_tol_type = atoi( argv[5] );
  }
  
  // Set tolerance for stopping citeria for FGMRES
  if ( argc >= 7 ) {
    user_rel_tol = atof( argv[6] );
  }
  
  // Set restarts
  if ( argc >= 8 ) {
    user_restart = atoi( argv[7] );
  }
  
  // Set max iterations
  if ( argc >= 9 ) {
    user_maxiter = atoi( argv[8] );
  }
  
  // Set precond choice: 0 = PariLU0, 1 = MKL csriLU0
  if ( argc >= 10 ) {
    user_precond_choice = atoi( argv[9] );
  }
  
  // Set PariLU0 iterative improvement tolerance
  if ( argc >= 11 ) {
    user_precond_reduction = atof( argv[10] );
  }
  
  printf("diagonal scaling = %s : %d\n", argv[4], scaling );
  printf("user_gmres_tol_type = %d\n", user_gmres_tol_type);
  printf("user_rel_tol = %e\n", user_rel_tol);
  printf("user_restart = %d\n", user_restart);
  printf("user_maxiter = %d\n", user_maxiter);
  printf("user_precond_choice = %d\n", user_precond_choice);
  printf("user_precond_reduction = %e\n", user_precond_reduction);

  // for MKL's GMRES
  int N = Asparse.num_rows;
  printf("N = %d\n", N);
  int* ia;
  int* ja;
  dataType* A;
  ia = (int*) calloc( (Asparse.num_rows+1), sizeof(int) );
  ja = (int*) calloc( Asparse.nnz, sizeof(int) );
  A = (dataType*) calloc( Asparse.nnz, sizeof(dataType) );
  // TODO: create indexing wrapper functions 
  #pragma omp parallel 
  {
    #pragma omp for nowait
    for (int i=0; i<Asparse.num_rows+1; i++) {
    	ia[i] = Asparse.row[i] + 1;	
    }
    #pragma omp for nowait
    for (int i=0; i<Asparse.nnz; i++) {
    	ja[i] = Asparse.col[i] + 1;
    	A[i] = Asparse.val[i];
    }
  }
  
  // pariLUv0.2
  data_d_matrix L = {Magma_CSRL};
  data_d_matrix U = {Magma_CSCU};
  data_d_matrix LU = {Magma_CSR};
  //dataType user_precond_reduction = 1.0e-15;
	
/*---------------------------------------------------------------------------
   Allocate storage for the ?par parameters and the solution/rhs/residual vectors
  ---------------------------------------------------------------------------*/
	MKL_INT ipar[size];
	double dpar[size];
	// Array of size ((2*ipar[14] + 1)*n + ipar[14]*(ipar[14] + 9)/2 + 1).
	//double tmp[N*(2*N+1)+(N*(N+9))/2+1];
	//double trvec[N], bilu0[Asparse.nnz];
	//double expected_solution[N];
	//double rhs[N], b[N];
	//double computed_solution[N];
	//double residual[N];
	
	double* tmp;
	double* trvec;
	double* bilu0;
	double* bilu0MKL;
	double* expected_solution;
	double* rhs;
	double* b;
	double* computed_solution;
	double* residual;
	double* b_scaled;
	double* residual_scaled;
  //tmp = (double*) calloc( ( N*(2*N+1)+(N*(N+9))/2+1 ), sizeof(double) );
  tmp = (double*) calloc( ( N*(2*user_restart+1)+(user_restart*(user_restart+9))/2+1 ), sizeof(double) );
  trvec = (double*) calloc( N, sizeof(double) );
  bilu0 = (double*) calloc( Asparse.nnz, sizeof(double) );
  bilu0MKL = (double*) calloc( Asparse.nnz, sizeof(double) );
  expected_solution = (double*) calloc( N, sizeof(double) );
  rhs = (double*) calloc( N, sizeof(double) );
  b = (double*) calloc( N, sizeof(double) );
  computed_solution = (double*) calloc( N, sizeof(double) );
  residual = (double*) calloc( N, sizeof(double) );
  
  //if ( scaling != Magma_NOSCALE ) {
    b_scaled = (double*) calloc( N, sizeof(double) );
    residual_scaled = (double*) calloc( N, sizeof(double) );
  //}

	MKL_INT matsize=Asparse.nnz, incx=1; //, ref_nit=2;
	//double ref_norm2=7.772387E+0, nrm2;
	double ref_norm2 = FLT_MAX, nrm2 = FLT_MAX;
	double final_residual_nrm2 = FLT_MAX;
	//double solution_error_nrm2;
	double tol_gmres_res = 1.0e-10;
  dataType Ares = 0.0;
  dataType Anonlinres = 0.0;
  //double dvar_bad = 0.0;
	
/*---------------------------------------------------------------------------
   Some additional variables to use with the RCI (P)FGMRES solver
  ---------------------------------------------------------------------------*/
	MKL_INT itercount=0,ierr=0;
	MKL_INT RCI_request=0, i=0, ivar=0;
	double dvar = FLT_MAX;
	double dvar_scaled = FLT_MAX;
	char cvar, cvar1, cvar2;

	printf("--------------------------------------------------------\n");
	printf("The FULLY ADVANCED example RCI FGMRES with ILU0 preconditioner\n");
	printf("to solve the non-degenerate algebraic system of linear equations\n");
	printf("--------------------------------------------------------\n\n");
/*---------------------------------------------------------------------------
   Initialize variables and the right hand side through matrix-vector product
  ---------------------------------------------------------------------------*/
	ivar=N;
	cvar='N';
	//for(i=0;i<N;i++)
	//{
	//	expected_solution[i]=1.0;
	//}
	//mkl_dcsrgemv(&cvar, &ivar, A, ia, ja, expected_solution, rhs);
	DEV_CHECKPT
	//dcopy(&ivar, rhs_vector.val, &i, rhs, &i);
	#pragma omp parallel 
  {
    #pragma omp for nowait
    for (int i=0; i<Asparse.num_rows; i++) {
      rhs[i] = rhs_vector.val[i];
    }
  }
	DEV_CHECKPT
/*---------------------------------------------------------------------------
   Save the right-hand side in vector b for future use
  ---------------------------------------------------------------------------*/
	i=1;
	dcopy(&ivar, rhs, &i, b, &i);
	DEV_CHECKPT
/*---------------------------------------------------------------------------
   Initialize the initial guess
  ---------------------------------------------------------------------------*/
	/*
	//debugging daxpy when compiled with -lmkl_intel_thread
	for(i=0;i<N;i++)
	{
		computed_solution[i]=1.0;
		residual[i] = 1.0;
	}
 	i = 1;
	dvar = 1.0;
	daxpy(&ivar, &dvar, computed_solution, &i, residual, &i);
	for(i=0;i<MIN(N,100);i++)
	{
		if ( residual[i] - 2.0 > 1e-16 )
 			printf("%d %e\n", i, residual[i]);
	}
  //getchar();
  */
	//normal use case
	#pragma omp parallel 
  {
    #pragma omp for nowait
	  for(i=0;i<N;i++)
	  {
	  	computed_solution[i] = 0.0;
	  	//computed_solution[i] = rhs[i];
	  	residual[i] = 0.0;
	  }
	}
	
	
	/*---------------------------------------------------------------------------
	   Initialize the solver
	  ---------------------------------------------------------------------------*/
	//for (int i=0; i<size; i++) {
  //  printf("ipar[%d] = %d\n", i, ipar[i] ); 
  //}  
	dfgmres_init(&ivar, computed_solution, rhs, &RCI_request, ipar, dpar, tmp);
	//printf("after dfgmres_init on line %d RCI_request = %d\n", __LINE__, RCI_request);
	//for (int i=0; i<size; i++) {
  //  printf("ipar[%d] = %d\n", i, ipar[i] ); 
  //}
	if (RCI_request!=0) goto FAILED;

/*---------------------------------------------------------------------------
   Calculate ILU0 preconditioner.
                        !ATTENTION!
   DCSRILU0 routine uses some IPAR, DPAR set by DFGMRES_INIT routine.
   Important for DCSRILU0 default entries set by DFGMRES_INIT are
   ipar[1] = 6 - output of error messages to the screen,
   ipar[5] = 1 - allow output of errors,
   ipar[30]= 0 - abort DCSRILU0 calculations if routine meets zero diagonal element.
  
   If ILU0 is going to be used out of MKL FGMRES context, than the values
   of ipar[1], ipar[5], ipar[30], dpar[30], and dpar[31] should be user
   provided before the DCSRILU0 routine call.
  
   In this example, specific for DCSRILU0 entries are set in turn:
   ipar[30]= 1 - change small diagonal value to that given by dpar[31],
   dpar[30]= 1.E-20 instead of the default value set by DFGMRES_INIT.
                    It is a small value to compare a diagonal entry with it.
   dpar[31]= 1.E-16 instead of the default value set by DFGMRES_INIT.
                    It is the target value of the diagonal value if it is
                    small as compared to dpar[30] and the routine should change
                    it rather than abort DCSRILU0 calculations.
  ---------------------------------------------------------------------------*/
  ipar[1] = 6;
  ipar[5] = 1;
	ipar[30]=1;
	dpar[30]=1.E-20;
	dpar[31]=1.E-16;
	      
	//MKL's iLU0
	wcsrilu0start = omp_get_wtime();
  dcsrilu0(&ivar, A, ia, ja, bilu0MKL, ipar, dpar, &ierr);
  wcsrilu0end = omp_get_wtime();
  ompwcsrilu0time = (dataType) (wcsrilu0end-wcsrilu0start);
  ref_norm2=dnrm2(&matsize, bilu0MKL, &incx );
  
  // PariLUv0.2
  //Asparse.num_rows = N; 
  //Asparse.num_cols = N; 
  //Asparse.nnz = matsize;
  //Asparse.row = (int*) calloc( (Asparse.num_rows+1), sizeof(int) );
  //Asparse.col = (int*) calloc( Asparse.nnz, sizeof(int) );
  //Asparse.val = (dataType*) calloc( Asparse.nnz, sizeof(dataType) );
  //#pragma omp parallel 
  //#pragma omp for nowait
  //for (int i=0; i<Asparse.nnz; i++) {
  //	Asparse.val[i] = A[i];
  //}
  //
  //// TODO: create indexing wrapper functions 
  //#pragma omp parallel 
  //{
  //  #pragma omp for nowait
  //  for (int i=0; i<Asparse.num_rows+1; i++) {
  //  	Asparse.row[i] = ia[i] - 1;	
  //  }
  //  #pragma omp for nowait
  //  for (int i=0; i<Asparse.nnz; i++) {
  //  	Asparse.col[i] = ja[i] - 1;
  //  }
  //}
  //data_zwrite_csr( &Asparse );
  
  
  if ( user_precond_choice == 0 ) {
    // TODO: return an int from parilu factorizations and check it
    data_PariLU_v0_2( &Asparse, &L, &U, user_precond_reduction );
    
    data_zilures(Asparse, L, U, &LU, &Ares, &Anonlinres);
    printf("PariLUv0_2_csrilu0_res = %e\n", Ares);
    printf("PariLUv0_2_csrilu0_nonlinres = %e\n", Anonlinres);
    
    data_zmlumerge( L, U, &LU );
    
    #pragma omp parallel 
    {
      #pragma omp for nowait
      for (int i=0; i<LU.nnz; i++) {
      	bilu0[i] = LU.val[i];     // Use PariLU factorization
      }
    }
  }
  else if ( user_precond_choice == 1 ) {
    #pragma omp parallel 
    {
      #pragma omp for nowait
      for (int i=0; i<Asparse.nnz; i++) {
      	bilu0[i] = bilu0MKL[i]; // Use MKL's csrilu0 factorization
      }
    }
  }
  else if ( user_precond_choice == 2 ) {
    #pragma omp parallel 
    {
      #pragma omp for nowait
      for (int i=0; i<Asparse.nnz; i++) {
      	bilu0[i] = Asparse.val[i]; // Use tril(A) + triu(A) factorization
      }
    }
  }
  nrm2=dnrm2(&matsize, bilu0, &incx );
  //nrm2=dnrm2(&matsize, Asparse.val, &incx );
  
  ierr = 0;
        
  
  //for (int i=0; i<size; i++) {
  //  printf("ipar[%d] = %d\n", i, ipar[i] ); 
  //}
  //for (int i=0; i<size; i++) {
  //  printf("dpar[%d] = %e\n", i, dpar[i] );
  //}
  //printf("N*(2*N+1)+(N*(N+9))/2+1 = %d \n", N*(2*N+1)+(N*(N+9))/2+1);
  //printf("((2*ipar[14] + 1)*n + ipar[14]*(ipar[14] + 9)/2 + 1) = %d \n", 
  //  ((2*ipar[14] + 1)*N + ipar[14]*(ipar[14] + 9)/2 + 1) );
  
	if (ierr!=0)
	{
	  printf("Preconditioner dcsrilu0 has returned the ERROR code %d", ierr);
	  goto FAILED1;
	}

	/*---------------------------------------------------------------------------
	  Set the desired parameters:
	  do the restart after 2 iterations
	  LOGICAL parameters:
	  do not do the stopping test for the maximal number of iterations
	  do the Preconditioned iterations of FGMRES method
	  Set parameter ipar[10] for preconditioner call. For this example,
    it reduces the number of iterations.
	  DOUBLE PRECISION parameters
	  set the relative tolerance to 1.0D-3 instead of default value 1.0D-6
      NOTE. Preconditioner may increase the number of iterations for an
      arbitrary case of the system and initial guess and even ruin the
      convergence. It is user's responsibility to use a suitable preconditioner
      and to apply it skillfully.
     ---------------------------------------------------------------------------*/
	
	ipar[4]=user_maxiter;
	ipar[7]=1;
	//if ( scaling == Magma_NOSCALE) {
	//  ipar[8]=1;
	//}
	//else {
	//  ipar[8]=0;
	//}
	ipar[8] = user_gmres_tol_type;
	ipar[9]=1;
	ipar[10]=1;
	//ipar[11]=1;
	ipar[14]=user_restart;
	dpar[0]=user_rel_tol;
	
	printf("ipar[4]=%d\n", ipar[4]);
	printf("ipar[14]=%d\n", ipar[14]);
	

	/*---------------------------------------------------------------------------
	   Check the correctness and consistency of the newly set parameters
	  ---------------------------------------------------------------------------*/
	dfgmres_check(&ivar, computed_solution, rhs, &RCI_request, ipar, dpar, tmp);
	//printf("after dfgmres_check on line %d RCI_request = %d\n", __LINE__, RCI_request);
	
	
	//for (int ii = 0; ii<100; ii++) 
	//  printf("%e\t%e\t%e\n", computed_solution[ii], rhs[ii], residual[ii]);
	
	mkl_dcsrgemv(&cvar, &ivar, A, ia, ja, computed_solution, residual);
	//for (int ii = 0; ii<100; ii++) 
	//  printf("%e\t%e\t%e\n", computed_solution[ii], rhs[ii], residual[ii]);
	i=1;
	dvar=dnrm2(&ivar,residual,&i);
	//printf("\n ==== Initial norm of residual = %e ==== \n", dvar );
	dvar=-1.0;
	daxpy(&ivar, &dvar, rhs, &i, residual, &i);
	//for (int ii = 0; ii<100; ii++) 
	//  printf("%e\t%e\t%e\n", computed_solution[ii], rhs[ii], residual[ii]);
	dvar=dnrm2(&ivar,residual,&i);
	//printf("\n ==== Initial norm of residual = %e ==== \n", dvar );
	//for (int ii = 0; ii<100; ii++) 
	//  printf("%e\t%e\n", rhs[ii], residual[ii]);
	dvar=dnrm2(&ivar,rhs,&i);
	//printf("\n ==== Initial norm of rhs = %e ==== \n", dvar );
	//printf("\n ==== rhs/dvar = %e ==== \n", dvar/dvar_bad );
	dvar=dnrm2(&ivar,b,&i);
	//printf("\n ==== Initial norm of b = %e ==== \n", dvar );
	
	if (RCI_request!=0) goto FAILED;
	/*---------------------------------------------------------------------------
	   Print the info about the RCI FGMRES method
	  ---------------------------------------------------------------------------*/
	printf("Some info about the current run of RCI FGMRES method:\n\n");
	if (ipar[7])
	{
		printf("As ipar[7]=%d, the automatic test for the maximal number of iterations will be\n", ipar[7]);
		printf("performed\n");
	}
	else
	{
		printf("As ipar[7]=%d, the automatic test for the maximal number of iterations will be\n", ipar[7]);
		printf("skipped\n");
	}
	printf("+++\n");
	if (ipar[8])
	{
		printf("As ipar[8]=%d, the automatic residual test will be performed\n", ipar[8]);
	}
	else
	{
		printf("As ipar[8]=%d, the automatic residual test will be skipped\n", ipar[8]);
	}
	printf("+++\n");
	if (ipar[9])
	{
		printf("As ipar[9]=%d the user-defined stopping test will be requested via\n", ipar[9]);
		printf("RCI_request=2\n");
	}
	else
	{
		printf("As ipar[9]=%d, the user-defined stopping test will not be requested, thus,\n", ipar[9]);
		printf("RCI_request will not take the value 2\n");
	}
	printf("+++\n");
	if (ipar[10])
	{
		printf("As ipar[10]=%d, the Preconditioned FGMRES iterations will be performed, thus,\n", ipar[10]);
		printf("the preconditioner action will be requested via RCI_request=3\n");
	}
	else
	{
		printf("As ipar[10]=%d, the Preconditioned FGMRES iterations will not be performed,\n", ipar[10]);
		printf("thus, RCI_request will not take the value 3\n");
	}
	printf("+++\n");
	if (ipar[11])
	{
		printf("As ipar[11]=%d, the automatic test for the norm of the next generated vector is\n", ipar[11]);
		printf("not equal to zero up to rounding and computational errors will be performed,\n");
		printf("thus, RCI_request will not take the value 4\n");
	}
	else
	{
		printf("As ipar[11]=%d, the automatic test for the norm of the next generated vector is\n", ipar[11]);
		printf("not equal to zero up to rounding and computational errors will be skipped,\n");
		printf("thus, the user-defined test will be requested via RCI_request=4\n");
	}
	printf("+++\n\n");
	printf("\n\niter\tdpar[4]/dpar[2]\tdpar[6]\n");
	/*---------------------------------------------------------------------------
	   Compute the solution by RCI (P)FGMRES solver with preconditioning
	   Reverse Communication starts here
	  ---------------------------------------------------------------------------*/
	  wfgmresstart = omp_get_wtime();
ONE:  dfgmres(&ivar, computed_solution, rhs, &RCI_request, ipar, dpar, tmp);
	//printf("after dfgmres on line %d RCI_request = %d\n", __LINE__, RCI_request);
	/*---------------------------------------------------------------------------
	   If RCI_request=0, then the solution was found with the required precision
	  ---------------------------------------------------------------------------*/
	if (RCI_request==0) goto COMPLETE;
	/*---------------------------------------------------------------------------
	   If RCI_request=1, then compute the vector A*tmp[ipar[21]-1]
	   and put the result in vector tmp[ipar[22]-1]
	  ---------------------------------------------------------------------------
	   NOTE that ipar[21] and ipar[22] contain FORTRAN style addresses,
	   therefore, in C code it is required to subtract 1 from them to get C style
	   addresses
	  ---------------------------------------------------------------------------*/
	if (RCI_request==1)
	{
	  //printf("RCI_request = %d line = %d\n", RCI_request, __LINE__);
		mkl_dcsrgemv(&cvar, &ivar, A, ia, ja, &tmp[ipar[21]-1], &tmp[ipar[22]-1]);
		goto ONE;
	}
	/*---------------------------------------------------------------------------
	   If RCI_request=2, then do the user-defined stopping test
	   The residual stopping test for the computed solution is performed here
	  ---------------------------------------------------------------------------
	   NOTE: from this point vector b[N] is no longer containing the right-hand
	   side of the problem! It contains the current FGMRES approximation to the
	   solution. If you need to keep the right-hand side, save it in some other
	   vector before the call to dfgmres routine. Here we saved it in vector
	   rhs[N]. The vector b is used instead of rhs to preserve the
	   original right-hand side of the problem and guarantee the proper
	   restart of FGMRES method. Vector b will be altered when computing the
	   residual stopping criterion!
	  ---------------------------------------------------------------------------*/
	if (RCI_request==2)
	{
	  printf("RCI_request = %d line = %d\n", RCI_request, __LINE__);
		/* Request to the dfgmres_get routine to put the solution into b[N] via ipar[12]
		  ---------------------------------------------------------------------------
		   WARNING: beware that the call to dfgmres_get routine with ipar[12]=0 at this stage may
		   destroy the convergence of the FGMRES method, therefore, only advanced users should
		   exploit this option with care */
		ipar[12]=1;
		/* Get the current FGMRES solution in the vector b[N] */
	  //printf("before get ipar[12] = %d\n", ipar[12] );
	  //printf("before get ipar[13] = %d\n", ipar[13] );
		dfgmres_get(&ivar, computed_solution, b, &RCI_request, ipar, dpar, tmp, &itercount);
	  //printf("after get RCI_request = %d line = %d\n", RCI_request, __LINE__);
		/* Compute the current true residual via MKL (Sparse) BLAS routines */
		mkl_dcsrgemv(&cvar, &ivar, A, ia, ja, b, residual);
		dvar=-1.0E0;
		i=1;
		daxpy(&ivar, &dvar, rhs, &i, residual, &i);
		dvar=dnrm2(&ivar,residual,&i);
		//printf("dvar = %e\n", dvar );
	  //printf("dpar[4] = %e\n", dpar[4] );
	  //printf("dpar[5] = %e\n", dpar[5] );
		//if (dvar<1.0E-3) goto COMPLETE;
		if ( scaling == Magma_NOSCALE ) {
		  
		  if (dvar<tol_gmres_res) goto COMPLETE;
		  else goto ONE;
		}
		else {
		  printf("calculating residual in original system for scaling %s\n", argv[4]);
		  if ( scaling == Magma_UNITROWCOL 
          || scaling == Magma_UNITDIAGCOL ) {
        printf("rescaling computed solution %s\n", argv[4]);
        DEV_CHECKPT
        for ( int ii=0; ii<N; ii++ ) {
          b_scaled[ii] = b[ii] * scaling_factors.val[ii];
        }
        DEV_CHECKPT
		    mkl_dcsrgemv(&cvar, &ivar, A_org.val, ia, ja, b_scaled, residual_scaled);
		    DEV_CHECKPT
      }
      else {
        DEV_CHECKPT
		    mkl_dcsrgemv(&cvar, &ivar, A_org.val, ia, ja, b, residual_scaled);
		    DEV_CHECKPT
      }
          
		  dvar_scaled=-1.0E0;
		  i=1;
		  daxpy(&ivar, &dvar_scaled, rhs_org.val, &i, residual_scaled, &i);
		  DEV_CHECKPT
		  dvar_scaled=dnrm2(&ivar,residual_scaled,&i);
		  DEV_CHECKPT
	    //printf("dvar = %e\n", dvar );
	    //printf("dpar[4] = %e\n", dpar[4] );
	    //printf("dpar[5] = %e\n", dpar[5] );
		  //if (dvar<1.0E-3) goto COMPLETE;
		  
		  if (dvar_scaled<tol_gmres_res) goto COMPLETE;
		  else goto ONE;
		}
	}
	/*---------------------------------------------------------------------------
	   If RCI_request=3, then apply the preconditioner on the vector
	   tmp[ipar[21]-1] and put the result in vector tmp[ipar[22]-1]
	  ---------------------------------------------------------------------------
	   NOTE that ipar[21] and ipar[22] contain FORTRAN style addresses,
	   therefore, in C code it is required to subtract 1 from them to get C style
	   addresses
	   Here is the recommended usage of the result produced by ILU0 routine
       via standard MKL Sparse Blas solver routine mkl_dcsrtrsv.
      ---------------------------------------------------------------------------*/
	if (RCI_request==3)
	{
	  //printf("RCI_request = %d line = %d\n", RCI_request, __LINE__);
		cvar1='L';
		cvar='N';
		cvar2='U';
		mkl_dcsrtrsv(&cvar1,&cvar,&cvar2,&ivar,bilu0,ia,ja,&tmp[ipar[21]-1],trvec);
		cvar1='U';
		cvar='N';
		cvar2='N';
		mkl_dcsrtrsv(&cvar1,&cvar,&cvar2,&ivar,bilu0,ia,ja,trvec,&tmp[ipar[22]-1]);
		goto ONE;
	}
	/*---------------------------------------------------------------------------
	   If RCI_request=4, then check if the norm of the next generated vector is
	   not zero up to rounding and computational errors. The norm is contained
	   in dpar[6] parameter
	  ---------------------------------------------------------------------------*/
	if (RCI_request==4)
	{
	  //printf("RCI_request = %d line = %d\n", RCI_request, __LINE__);
	  //printf("dpar[2] = %e\n", dpar[2] );
	  //printf("dpar[4] = %e\n", dpar[4] );
	  //printf("dpar[4]/dpar[2] = %e\n", dpar[4]/dpar[2] );
	  //printf("dpar[6] = %e\n", dpar[6] );
	  //printf("itercount = %d\n", itercount );
	  
	  //printf("itercount = %d dpar[4]/dpar[2] = %e dpar[6] = %e\n", 
	  //  itercount, dpar[4]/dpar[2], dpar[6] );
	  
	  printf("%d\t%e\t%e\n", 
	    itercount, dpar[4]/dpar[2], dpar[6] );
		if (dpar[6]<1.0E-14) goto COMPLETE;
		else goto ONE;
	}
	/*---------------------------------------------------------------------------
	   If RCI_request=anything else, then dfgmres subroutine failed
	   to compute the solution vector: computed_solution[N]
	  ---------------------------------------------------------------------------*/
	else
	{
	  printf("RCI_request = %d\n", RCI_request);
		goto FAILED;
	}
	/*---------------------------------------------------------------------------
	   Reverse Communication ends here
	   Get the current iteration number and the FGMRES solution (DO NOT FORGET to
	   call dfgmres_get routine as computed_solution is still containing
	   the initial guess!). Request to dfgmres_get to put the solution
	   into vector computed_solution[N] via ipar[12]
	  ---------------------------------------------------------------------------*/
COMPLETE:   ipar[12]=0;
	dfgmres_get(&ivar, computed_solution, rhs, &RCI_request, ipar, dpar, tmp, &itercount);
  wfgmresend = omp_get_wtime();
  ompwfgmrestime = (dataType) (wfgmresend-wfgmresstart);
	/*---------------------------------------------------------------------------
	   Print solution vector: computed_solution[N] and the number of iterations: itercount
	  ---------------------------------------------------------------------------*/
	printf("The system has been solved \n");

	//printf("\nThe following solution has been obtained: \n");
	
	// Scale the computed solution if necessary
  if ( argc >= 5 ) {
    //if ( ( strcmp( argv[4], "UNITROWCOL" ) == 0 ) 
    //      || ( strcmp( argv[4], "UNITDIAGCOL" ) == 0 ) ) {
    //  printf("rescaling computed solution %s\n", argv[4]);
    //  for ( int i=0; i<N; i++ ) {
    //    computed_solution[i] = computed_solution[i] * scaling_factors.val[i];
    //  }
    //}
    if ( scaling == Magma_UNITROWCOL 
          || scaling == Magma_UNITDIAGCOL ) {
      printf("rescaling computed solution %s\n", argv[4]);
      for ( int i=0; i<N; i++ ) {
        computed_solution[i] = computed_solution[i] * scaling_factors.val[i];
      }
    }
  }
  
  //for(i=0;i<N;i++) {
	//	printf("computed_solution_%s(%d) = %e;\n", argv[4], i+1, computed_solution[i]);
	//}
	
	
	final_residual_nrm2 = dnrm2(&ivar, residual, &incx );
  
	//solution_error_nrm2 = 0.0;
	//for (i=0;i<N;i++)
	//{
	//  solution_error_nrm2 += pow( (expected_solution[i] - computed_solution[i]), 2 );
	//	//printf("computed_solution[%d]=%e\n",i,computed_solution[i]);
	//}
	//solution_error_nrm2 = sqrt( solution_error_nrm2 );
	//printf("\nThe expected solution is: \n");
	//for (i=0;i<N;i++)
	//{
	//	printf("expected_solution[%d]=%e\n",i,expected_solution[i]);
	//}
	printf("\nNumber of iterations: %d\n" ,itercount);
	printf("\nfinal residual nrm2: %e\n" ,final_residual_nrm2);
	printf("\ndvar: %e\n" ,dvar);
	printf("RCI_request = %d line = %d\n", RCI_request, __LINE__);
	printf("Euclidean norm of initial residual dpar[2] = %e\n", dpar[2] );
	printf("Euclidean norm of current residual dpar[4] = %e\n", dpar[4] );
	printf("dpar[4]/dpar[2] = %e\n", dpar[4]/dpar[2] );
	printf("Euclidean norm of generated vector dpar[6] = %e\n", dpar[6] );
	//printf("\nsoution error_nrm2: %e\n" ,solution_error_nrm2);
	printf("\npreconditioner fabs(ref_norm2-nrm2) = %e\n", fabs(ref_norm2-nrm2) );
	printf("\n");
	printf("csrilu0_wall = %e\n", ompwcsrilu0time);
	printf("fgmres_wall = %e\n", ompwfgmrestime);

	
	cvar='N';
	mkl_dcsrgemv(&cvar, &ivar, A_org.val, ia, ja, computed_solution, residual);
	dvar=-1.0E0;
	i=1;
	daxpy(&ivar, &dvar, rhs_org.val, &i, residual, &i);
	final_residual_nrm2 = dnrm2(&ivar,residual,&i);
	printf("\nfinal residual nrm2 from original system: %e\n" ,final_residual_nrm2);
	
	free( ia );
	free( ja );
	free( A );
  free( tmp );
	free( trvec );
	free( bilu0 );
	free( bilu0MKL );
	free( expected_solution );
	free( rhs );
	free( b );
	free( computed_solution );
	free( residual );
	data_zmfree( &Asparse );
	data_zmfree( &rhs_vector );
	data_zmfree( &A_org );
	data_zmfree( &rhs_org );
	data_zmfree( &L );
	data_zmfree( &U );
	data_zmfree( &LU );
	//if ( scaling != Magma_NOSCALE ) {
	  free( b_scaled );
	  free( residual_scaled );
	//}
	return 0;
	
	//if(itercount==ref_nit && fabs(ref_norm2-nrm2)<1.e-6) {
	//if(final_residual_nrm2<1.e-6 && solution_error_nrm2<1.e-6 && fabs(ref_norm2-nrm2)<1.e-6) {
	if(final_residual_nrm2<1.e-6 && fabs(ref_norm2-nrm2)<1.e-6) {
	  printf("--------------------------------------------------------\n");
	  printf("C example of FGMRES with ILU0 preconditioner \n");
	  printf("has successfully PASSED all stages of computations\n");
	  printf("--------------------------------------------------------\n");
    free( ia );
	  free( ja );
	  free( A );
    free( tmp );
	  free( trvec );
	  free( bilu0 );
	  free( bilu0MKL );
	  free( expected_solution );
	  free( rhs );
	  free( b );
	  free( computed_solution );
	  free( residual );
	  data_zmfree( &Asparse );
	  data_zmfree( &rhs_vector );
	  data_zmfree( &A_org );
	  data_zmfree( &rhs_org );
	  data_zmfree( &L );
	  data_zmfree( &U );
	  data_zmfree( &LU );
	  return 0;
	}
	else
	{
	  printf("Probably, the preconditioner was computed incorrectly:\n");
	  printf("Either the preconditioner norm %e differs from the expected norm %e\n",nrm2,ref_norm2);
	  printf("and/or the final_residual_nrm2 %e is greater than %e\n",final_residual_nrm2,1.e-6);
	  //printf("and/or the solution_error_nrm2 %e is greater than %e\n",solution_error_nrm2,1.e-6);
	  //printf("and/or the number of iterations %d differs from the expected number %d\n",itercount,ref_nit);
	  printf("-------------------------------------------------------------------\n");
	  printf("Unfortunately, FGMRES+ILU0 C example has FAILED\n");
	  printf("-------------------------------------------------------------------\n");
    free( ia );
	  free( ja );
	  free( A );
    free( tmp );
	  free( trvec );
	  free( bilu0 );
	  free( bilu0MKL );
	  free( expected_solution );
	  free( rhs );
	  free( b );
	  free( computed_solution );
	  free( residual );
	  data_zmfree( &Asparse );
	  data_zmfree( &rhs_vector );
	  data_zmfree( &A_org );
	  data_zmfree( &rhs_org );
	  data_zmfree( &L );
	  data_zmfree( &U );
	  data_zmfree( &LU );
	  return 0;
	}
FAILED:
  wfgmresend = omp_get_wtime();
  ompwfgmrestime = (dataType) (wfgmresend-wfgmresstart);
	final_residual_nrm2 = dnrm2(&ivar, residual, &incx );
	printf("The solver has returned the ERROR code %d \n", RCI_request);
	printf("\nNumber of iterations: %d\n" ,itercount);
	printf("\nfinal residual nrm2: %e\n" ,final_residual_nrm2);
	printf("\ndvar: %e\n" ,dvar);
	printf("RCI_request = %d line = %d\n", RCI_request, __LINE__);
	printf("Euclidean norm of initial residual dpar[2] = %e\n", dpar[2] );
	printf("Euclidean norm of current residual dpar[4] = %e\n", dpar[4] );
	printf("dpar[4]/dpar[2] = %e\n", dpar[4]/dpar[2] );
	printf("Euclidean norm of generated vector dpar[6] = %e\n", dpar[6] );
	printf("\npreconditioner fabs(ref_norm2-nrm2) = %e\n", fabs(ref_norm2-nrm2) );
	printf("\n");
	printf("csrilu0_wall = %e\n", ompwcsrilu0time);
	printf("fgmres_wall = %e\n", ompwfgmrestime);
FAILED1:
	printf("-------------------------------------------------------------------\n");
	printf("Unfortunately, FGMRES + ParILU0 C example has FAILED\n");
	printf("-------------------------------------------------------------------\n");
  free( ia );
	free( ja );
	free( A );
  free( tmp );
	free( trvec );
	free( bilu0 );
	free( bilu0MKL );
	free( expected_solution );
	free( rhs );
	free( b );
	free( computed_solution );
	free( residual );
	data_zmfree( &Asparse );
	data_zmfree( &rhs_vector );
	data_zmfree( &A_org );
	data_zmfree( &rhs_org );
	data_zmfree( &L );
	data_zmfree( &U );
	data_zmfree( &LU );
	data_zmfree( &scaling_factors );
	return 0;
}
