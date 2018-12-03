
#include "sparse.h"
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <libgen.h>
#include <string.h>
#include <vector>
#include <omp.h>
#include "math.h"
#include "mkl_blas.h"
#include "mkl_spblas.h"
#include "mkl_rci.h"

#define size 128

//#define DEBUG_MKL 
#ifdef DEBUG_MKL
  #define MKL_PRINTF(...){ printf(__VA_ARGS__); fflush(stdout); }
#else
  #define MKL_PRINTF(...){ (void)0; }
#endif

extern "C"
int
data_MKL_FGMRES(
    data_d_matrix *A, data_d_matrix *x0, data_d_matrix *rhs_vector,
    data_d_gmres_param *solverParam )
{
  int info = 0;

  int N = A->num_rows;
  printf("N = %d\n", N);
  int* ia;
  int* ja;
  LACE_CALLOC(ia, (A->num_rows+1) );
  LACE_CALLOC(ja, A->nnz);
  // TODO: create indexing wrapper functions
  #pragma omp parallel
  {
    #pragma omp for nowait
    for (int i=0; i<A->num_rows+1; i++) {
      ia[i] = A->row[i] + 1;
    }
    #pragma omp for nowait
    for (int i=0; i<A->nnz; i++) {
      ja[i] = A->col[i] + 1;
    }
  }

  data_d_matrix L = {Magma_CSRL};
  data_d_matrix U = {Magma_CSCU};
  data_d_matrix LU = {Magma_CSR};

/*---------------------------------------------------------------------------
   Allocate storage for the ?par parameters and the solution/rhs/residual vectors
  ---------------------------------------------------------------------------*/
  MKL_INT ipar[size];
  double dpar[size];

  double* tmp;
  double* trvec;
  double* bilu0;
  double* bilu0MKL;
  double* expected_solution;
  double* rhs;
  double* b;
  double* computed_solution;
  double* residual;

  tmp = (double*) calloc( ( N*(2*solverParam->restart_max+1)+(solverParam->restart_max*(solverParam->restart_max+9))/2+1 ), sizeof(double) );
  trvec = (double*) calloc( N, sizeof(double) );
  bilu0 = (double*) calloc( A->nnz, sizeof(double) );
  bilu0MKL = (double*) calloc( A->nnz, sizeof(double) );
  expected_solution = (double*) calloc( N, sizeof(double) );
  rhs = (double*) calloc( N, sizeof(double) );
  b = (double*) calloc( N, sizeof(double) );
  computed_solution = (double*) calloc( N, sizeof(double) );
  residual = (double*) calloc( N, sizeof(double) );

  MKL_INT matsize=A->nnz, incx=1;
  double ref_norm2, nrm2;
  double final_residual_nrm2;
  dataType Ares = 0.0;
  dataType Anonlinres = 0.0;
  double dvar_bad = 0.0;

/*---------------------------------------------------------------------------
   Some additional variables to use with the RCI (P)FGMRES solver
  ---------------------------------------------------------------------------*/
  MKL_INT itercount=0, ierr=0;
  MKL_INT RCI_request, i, ivar;
  double dvar;
  char cvar,cvar1,cvar2;

  printf("--------------------------------------------------------\n");
  printf("The FULLY ADVANCED example RCI FGMRES with ILU0 preconditioner\n");
  printf("to solve the non-degenerate algebraic system of linear equations\n");
  printf("--------------------------------------------------------\n\n");
/*---------------------------------------------------------------------------
   Initialize variables and the right hand side through matrix-vector product
  ---------------------------------------------------------------------------*/
  ivar=N;
  cvar='N';
  #pragma omp parallel
  {
    #pragma omp for nowait
    for (int i=0; i<A->num_rows; i++) {
      rhs[i] = rhs_vector->val[i];
    }
  }
/*---------------------------------------------------------------------------
   Save the right-hand side in vector b for future use
  ---------------------------------------------------------------------------*/
  i=1;
  dcopy(&ivar, rhs, &i, b, &i);
/*---------------------------------------------------------------------------
   Initialize the initial guess
  ---------------------------------------------------------------------------*/
  for(i=0;i<N;i++)
  {
    computed_solution[i]=x0->val[i];
    residual[i] = 0.0;
  }
  /*---------------------------------------------------------------------------
     Initialize the solver
    ---------------------------------------------------------------------------*/
  dfgmres_init(&ivar, computed_solution, rhs, &RCI_request, ipar, dpar, tmp);
  MKL_PRINTF("after dfgmres_init on line %d RCI_request = %d\n", __LINE__, RCI_request);
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

  
#if 0 //Use parilu preconditioner
  data_PariLU_v0_2( A, &L, &U, solverParam->parilu_reduction );
  MKL_PRINTF("%s %d A->num_rows = %d\n", __FILE__, __LINE__, A->num_rows);

  data_zilures(*A, L, U, &LU, &Ares, &Anonlinres);
  MKL_PRINTF("%s %d A->num_rows = %d\n", __FILE__, __LINE__, A->num_rows);
  MKL_PRINTF("PariLUv0_2_csrilu0_res = %e\n", Ares);
  MKL_PRINTF("PariLUv0_2_csrilu0_nonlinres = %e\n", Anonlinres);

  data_zmlumerge( L, U, &LU );
  
  #pragma omp parallel
  {
    #pragma omp for nowait
    for (int i=0; i<LU.nnz; i++) {
      bilu0[i] = LU.val[i];
    }
  }
#else //use MKL preconditioner
  //MKL's iLU0
  dcsrilu0(&ivar, A->val, ia, ja, bilu0MKL, ipar, dpar, &ierr);
  ref_norm2=dnrm2(&matsize, bilu0MKL, &incx );
  MKL_PRINTF("%s %d A->num_rows = %d\n", __FILE__, __LINE__, A->num_rows);
  
  #pragma omp parallel
  {
    #pragma omp for nowait
    for (int i=0; i<A->nnz; i++) {
      bilu0[i] = bilu0MKL[i];
    }
  } 
#endif
  
  nrm2=dnrm2(&matsize, bilu0, &incx );

  ierr = 0;

  if (ierr!=0)
  {
    MKL_PRINTF("Preconditioner dcsrilu0 has returned the ERROR code %d", ierr);
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

  ipar[4]=solverParam->search_max;
  ipar[7]=1;
  ipar[8]=1;
  ipar[9]=1;
  ipar[10]=solverParam->precondition;
  ipar[14]=solverParam->restart_max;
  if (solverParam->tol_type == 0) {
    dpar[0]=0.0;
    dpar[1]=solverParam->rtol;
  }
  else {
    dpar[0]=solverParam->rtol;
    dpar[1]=0.0;
  }

  MKL_PRINTF("ipar[4]=%d\n", ipar[4]);
  MKL_PRINTF("ipar[14]=%d\n", ipar[14]);

  /*---------------------------------------------------------------------------
     Check the correctness and consistency of the newly set parameters
    ---------------------------------------------------------------------------*/
  dfgmres_check(&ivar, computed_solution, rhs, &RCI_request, ipar, dpar, tmp);
  MKL_PRINTF("after dfgmres_check on line %d RCI_request = %d\n", __LINE__, RCI_request);

  mkl_dcsrgemv(&cvar, &ivar, A->val, ia, ja, computed_solution, residual);
  i=1;
  dvar=dnrm2(&ivar,residual,&i);
  MKL_PRINTF("\n ==== Initial norm of residual = %e ==== \n", dvar );
  dvar=-1.0;
  daxpy(&ivar, &dvar, rhs, &i, residual, &i);
  dvar=dnrm2(&ivar,residual,&i);
  MKL_PRINTF("\n ==== Initial residual = %e ==== \n", dvar );
  dvar=dnrm2(&ivar,rhs,&i);
  MKL_PRINTF("\n ==== Initial norm of rhs = %e ==== \n", dvar );
  dvar=dnrm2(&ivar,b,&i);
  MKL_PRINTF("\n ==== Initial norm of b = %e ==== \n", dvar );

  if (RCI_request!=0) goto FAILED;
  /*---------------------------------------------------------------------------
     Print the info about the RCI FGMRES method
    ---------------------------------------------------------------------------*/
  MKL_PRINTF("Some info about the current run of RCI FGMRES method:\n\n");
  if (ipar[7])
  {
    MKL_PRINTF("As ipar[7]=%d, the automatic test for the maximal number of iterations will be\n", ipar[7]);
    MKL_PRINTF("performed\n");
  }
  else
  {
    MKL_PRINTF("As ipar[7]=%d, the automatic test for the maximal number of iterations will be\n", ipar[7]);
    MKL_PRINTF("skipped\n");
  }
  MKL_PRINTF("+++\n");
  if (ipar[8])
  {
    MKL_PRINTF("As ipar[8]=%d, the automatic residual test will be performed\n", ipar[8]);
  }
  else
  {
    MKL_PRINTF("As ipar[8]=%d, the automatic residual test will be skipped\n", ipar[8]);
  }
  MKL_PRINTF("+++\n");
  if (ipar[9])
  {
    MKL_PRINTF("As ipar[9]=%d the user-defined stopping test will be requested via\n", ipar[9]);
    MKL_PRINTF("RCI_request=2\n");
  }
  else
  {
    MKL_PRINTF("As ipar[9]=%d, the user-defined stopping test will not be requested, thus,\n", ipar[9]);
    MKL_PRINTF("RCI_request will not take the value 2\n");
  }
  MKL_PRINTF("+++\n");
  if (ipar[10])
  {
    MKL_PRINTF("As ipar[10]=%d, the Preconditioned FGMRES iterations will be performed, thus,\n", ipar[10]);
    MKL_PRINTF("the preconditioner action will be requested via RCI_request=3\n");
  }
  else
  {
    MKL_PRINTF("As ipar[10]=%d, the Preconditioned FGMRES iterations will not be performed,\n", ipar[10]);
    MKL_PRINTF("thus, RCI_request will not take the value 3\n");
  }
  MKL_PRINTF("+++\n");
  if (ipar[11])
  {
    MKL_PRINTF("As ipar[11]=%d, the automatic test for the norm of the next generated vector is\n", ipar[11]);
    MKL_PRINTF("not equal to zero up to rounding and computational errors will be performed,\n");
    MKL_PRINTF("thus, RCI_request will not take the value 4\n");
  }
  else
  {
    MKL_PRINTF("As ipar[11]=%d, the automatic test for the norm of the next generated vector is\n", ipar[11]);
    MKL_PRINTF("not equal to zero up to rounding and computational errors will be skipped,\n");
    MKL_PRINTF("thus, the user-defined test will be requested via RCI_request=4\n");
  }
  MKL_PRINTF("+++\n\n");
  /*---------------------------------------------------------------------------
     Compute the solution by RCI (P)FGMRES solver with preconditioning
     Reverse Communication starts here
    ---------------------------------------------------------------------------*/

ONE:  dfgmres(&ivar, computed_solution, rhs, &RCI_request, ipar, dpar, tmp);

  MKL_PRINTF("after dfgmres on line %d RCI_request = %d\n", __LINE__, RCI_request);
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
    MKL_PRINTF("RCI_request = %d line = %d\n", RCI_request, __LINE__);
    mkl_dcsrgemv(&cvar, &ivar, A->val, ia, ja, &tmp[ipar[21]-1], &tmp[ipar[22]-1]);
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
    MKL_PRINTF("RCI_request = %d line = %d\n", RCI_request, __LINE__);
    /* Request to the dfgmres_get routine to put the solution into b[N] via ipar[12]
      ---------------------------------------------------------------------------
       WARNING: beware that the call to dfgmres_get routine with ipar[12]=0 at this stage may
       destroy the convergence of the FGMRES method, therefore, only advanced users should
       exploit this option with care */
    ipar[12]=1;
    /* Get the current FGMRES solution in the vector b[N] */
    MKL_PRINTF("before get ipar[12] = %d\n", ipar[12] );
    MKL_PRINTF("before get ipar[13] = %d\n", ipar[13] );
    dfgmres_get(&ivar, computed_solution, b, &RCI_request, ipar, dpar, tmp, &itercount);
    MKL_PRINTF("after get RCI_request = %d line = %d\n", RCI_request, __LINE__);
    /* Compute the current true residual via MKL (Sparse) BLAS routines */
    mkl_dcsrgemv(&cvar, &ivar, A->val, ia, ja, b, residual);
    dvar=-1.0E0;
    i=1;
    daxpy(&ivar, &dvar, rhs, &i, residual, &i);
    dvar=dnrm2(&ivar,residual,&i);
    MKL_PRINTF("dvar = %e\n", dvar );
    MKL_PRINTF("dpar[4] = %e\n", dpar[4] );
    MKL_PRINTF("dpar[5] = %e\n", dpar[5] );
    if (dvar<solverParam->rtol) goto COMPLETE;
    else goto ONE;
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
    MKL_PRINTF("RCI_request = %d line = %d\n", RCI_request, __LINE__);
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
    MKL_PRINTF("RCI_request = %d line = %d\n", RCI_request, __LINE__);
    MKL_PRINTF("dpar[2] = %e\n", dpar[2] );
    MKL_PRINTF("dpar[4] = %e\n", dpar[4] );
    MKL_PRINTF("dpar[4]/dpar[2] = %e\n", dpar[4]/dpar[2] );
    MKL_PRINTF("dpar[6] = %e\n", dpar[6] );
    MKL_PRINTF("itercount = %d\n", itercount );
    if (dpar[6]<1.0E-14) goto COMPLETE;
    else goto ONE;
  }
  /*---------------------------------------------------------------------------
     If RCI_request=anything else, then dfgmres subroutine failed
     to compute the solution vector: computed_solution[N]
    ---------------------------------------------------------------------------*/
  else
  {
    MKL_PRINTF("RCI_request = %d\n", RCI_request);
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
  /*---------------------------------------------------------------------------
     Print solution vector: computed_solution[N] and the number of iterations: itercount
    ---------------------------------------------------------------------------*/
  MKL_PRINTF("The system has been solved \n");

  final_residual_nrm2 = dnrm2(&ivar, residual, &incx );
  MKL_PRINTF("\nNumber of iterations: %d\n" ,itercount);
  MKL_PRINTF("\nfinal residual nrm2: %e\n" ,final_residual_nrm2);
  MKL_PRINTF("\ndvar: %e\n" ,dvar);
  MKL_PRINTF("\npreconditioner fabs(ref_norm2-nrm2) = %e\n", fabs(ref_norm2-nrm2) );
  MKL_PRINTF("\n");

  for (int pi=0; pi<A->num_rows; ++pi) {
    x0->val[pi] = computed_solution[pi];
  }

  // if(final_residual_nrm2<solverParam->rtol && fabs(ref_norm2-nrm2)<1.e-6) {
    // printf("--------------------------------------------------------\n");
    // printf("C example of FGMRES with ILU0 preconditioner \n");
    // printf("has successfully PASSED all stages of computations\n");
    // printf("--------------------------------------------------------\n");
  free( ia );
  free( ja );
  free( tmp );
  free( trvec );
  free( bilu0 );
  free( bilu0MKL );
  free( expected_solution );
  free( rhs );
  free( b );
  free( computed_solution );
  free( residual );
  data_zmfree( &L );
  data_zmfree( &U );
  data_zmfree( &LU );
  return 0;
  // }
  // else
  // {
  //   printf("Probably, the preconditioner was computed incorrectly:\n");
  //   printf("Either the preconditioner norm %e differs from the expected norm %e\n",nrm2,ref_norm2);
  //   printf("and/or the final_residual_nrm2 %e is greater than %e\n",final_residual_nrm2,solverParam->rtol);
  //   printf("-------------------------------------------------------------------\n");
  //   printf("Unfortunately, FGMRES+ILU0 C example has FAILED\n");
  //   printf("-------------------------------------------------------------------\n");
  //   free( ia );
  //   free( ja );
  //   free( tmp );
  //   free( trvec );
  //   free( bilu0 );
  //   free( bilu0MKL );
  //   free( expected_solution );
  //   free( rhs );
  //   free( b );
  //   free( computed_solution );
  //   free( residual );
  //   data_zmfree( &L );
  //   data_zmfree( &U );
  //   data_zmfree( &LU );
  //   return 0;
  // }
FAILED:
  printf("The solver has returned the ERROR code %d \n", RCI_request);
FAILED1:
  printf("-------------------------------------------------------------------\n");
  printf("Unfortunately, FGMRES + ParILU0 C example has FAILED\n");
  printf("-------------------------------------------------------------------\n");
  free( ia );
  free( ja );
  free( tmp );
  free( trvec );
  free( bilu0 );
  free( bilu0MKL );
  free( expected_solution );
  free( rhs );
  free( b );
  free( computed_solution );
  free( residual );
  data_zmfree( &L );
  data_zmfree( &U );
  data_zmfree( &LU );

  return info;
}
