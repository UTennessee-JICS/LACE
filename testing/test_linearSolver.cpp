
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

const dataType zero = 0.0;
const dataType one = 1.0;
const dataType negone = -1.0;

class LinearSolverTest: public
  ::testing::Test
{
protected:
  LinearSolverTest() {}

  // per-test-case set-up
  static void SetUpTestCase() {
    printf("%% seting up\n");
    fflush(stdout);

    printf("%% my_argc = %d\n", my_argc);
    for (int i=0; i< my_argc; ++i) {
      printf("%% my_agv[%d] = %s\n", i, my_argv[i]);
    }
    fflush(stdout);

    char default_matrix[] = "matrices/Trefethen_20.mtx";
    char* matrix_name = NULL;

    char default_rhs[] = "ONES";
    char* rhs_name = NULL;

    char default_initialGuess[] = "ZEROS";
    char* initialGuess_name = NULL;

    tolerance = new dataType();
    (*tolerance) = 1.0e-3;

    // parse command line arguments
    if (my_argc>1) {
      int count = 1;
      while (count < my_argc) {
        if ( (strcmp(my_argv[count], "--matrix") == 0)
            && count+1 < my_argc ) {
          matrix_name = my_argv[count+1];
          count = count + 2;
        }
        else if ( (strcmp(my_argv[count], "--rhs") == 0)
            && count+1 < my_argc ) {
          matrix_name = my_argv[count+1];
          count = count + 2;
        }
        else if ( (strcmp(my_argv[count], "--initialGuess") == 0)
            && count+1 < my_argc ) {
          initialGuess_name = my_argv[count+1];
          count = count + 2;
        }
        else if ( (strcmp(my_argv[count], "--tolerance") == 0)
            && count+1 < my_argc ) {
          (*tolerance) = atof(my_argv[count+1]);
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
    if (rhs_name == NULL ) {
      rhs_name = default_rhs;
    }
    if (initialGuess_name == NULL ) {
      initialGuess_name = default_initialGuess;
    }
    printf("A will be read from %s\n", matrix_name);
    A = new data_d_matrix();
    A->storage_type = Magma_CSR;
    CHECK( data_z_csr_mtx( A, matrix_name ) );

    rhs_vector = new data_d_matrix();
    rhs_vector->storage_type = Magma_DENSE;
  	rhs_vector->major = MagmaRowMajor;
    if ( strcmp( rhs_name, "ONES" ) == 0 ) {
      printf("%% creating a vector of %d ones for the rhs.\n", A->num_rows);
      CHECK( data_zvinit( rhs_vector, A->num_rows, 1, one ) );
    }
    else {
      printf("%% RHS will be read from %s\n", rhs_name);
      CHECK( data_z_dense_mtx( rhs_vector, rhs_vector->major, rhs_name ) );
    }

    initialGuess_vector = new data_d_matrix();
    initialGuess_vector->storage_type = Magma_DENSE;
  	initialGuess_vector->major = MagmaRowMajor;
    if ( strcmp( initialGuess_name, "ZEROS" ) == 0 ) {
      printf("%% creating a vector of %d zeros for the initial guess.\n", A->num_rows);
      CHECK( data_zvinit( initialGuess_vector, A->num_rows, 1, zero ) );
    }
    else {
      printf("%% initial guess will be read from %s\n", initialGuess_name);
      CHECK( data_z_dense_mtx( initialGuess_vector, initialGuess_vector->major, initialGuess_name ) );
    }

  }

  // per-test-case tear-down
  static void TearDownTestCase() {
    data_zmfree( A );
    data_zmfree( rhs_vector );
    delete A;
    delete rhs_vector;
    delete tolerance;
  }

  // per-test set-up and tear-down
  virtual void SetUp() {}
  virtual void TearDown() {}

  // shared by all tests
  static data_d_matrix* A;
  static data_d_matrix* rhs_vector;
  static data_d_matrix* initialGuess_vector;
  static dataType* tolerance;
};

data_d_matrix* LinearSolverTest::A = NULL;
data_d_matrix* LinearSolverTest::rhs_vector = NULL;
data_d_matrix* LinearSolverTest::initialGuess_vector = NULL;
dataType* LinearSolverTest::tolerance = NULL;

TEST_F(LinearSolverTest, MKLFGMRESnonPreconditioned) {
  printf("%% MKL FGMRES non-preconditioned\n");

  // store initial guess in solution_vector
  data_d_matrix solution_vector = {Magma_DENSE};
  CHECK( data_zmconvert((*initialGuess_vector), &solution_vector, Magma_DENSE, Magma_DENSE) );

  data_z_gmres_param solverParam;

  solverParam.search_max = 2000;
  solverParam.restart_max = 2000;
  solverParam.tol_type = 0;
  solverParam.rtol = (*LinearSolverTest::tolerance);
  solverParam.precondition = 0;

  // solve
  data_MKL_FGMRES( A, &solution_vector, rhs_vector, &solverParam );

  // print solver summary


  // caclulate residual
  dataType residual = 0.0;
  data_d_matrix r={Magma_DENSE};
  data_zvinit( &r, A->num_rows, 1, zero );

  data_z_spmv( negone, A, &solution_vector, zero, &r );
  data_zaxpy( A->num_rows, one, rhs_vector->val, 1, r.val, 1);
  for (int i=0; i<A->num_rows; ++i) {
    GMRESDBG("r.val[%d] = %.16e\n", i, r.val[i]);
  }
  residual = data_dnrm2( A->num_rows, r.val, 1 );
  printf("%% external check of rnorm2 = %.16e;\n\n", residual);

  fflush(stdout);

  EXPECT_LE( residual, (*LinearSolverTest::tolerance) );

  data_zmfree( &solution_vector );
  data_zmfree( &r );

}

TEST_F(LinearSolverTest, MKLFGMRESPreconditioned) {
  printf("%% MKL FGMRES non-preconditioned\n");

  // store initial guess in solution_vector
  data_d_matrix solution_vector = {Magma_DENSE};
  CHECK( data_zmconvert((*initialGuess_vector), &solution_vector, Magma_DENSE, Magma_DENSE) );

  data_z_gmres_param solverParam;

  solverParam.search_max = 2000;
  solverParam.restart_max = 2000;
  solverParam.tol_type = 0;
  solverParam.rtol = (*LinearSolverTest::tolerance);
  solverParam.precondition = 1;

  // solve
  data_MKL_FGMRES( A, &solution_vector, rhs_vector, &solverParam );

  // print solver summary


  // caclulate residual
  dataType residual = 0.0;
  data_d_matrix r={Magma_DENSE};
  data_zvinit( &r, A->num_rows, 1, zero );

  data_z_spmv( negone, A, &solution_vector, zero, &r );
  data_zaxpy( A->num_rows, one, rhs_vector->val, 1, r.val, 1);
  for (int i=0; i<A->num_rows; ++i) {
    GMRESDBG("r.val[%d] = %.16e\n", i, r.val[i]);
  }
  residual = data_dnrm2( A->num_rows, r.val, 1 );
  printf("%% external check of rnorm2 = %.16e;\n\n", residual);

  fflush(stdout);

  EXPECT_LE( residual, (*LinearSolverTest::tolerance) );

  data_zmfree( &solution_vector );
  data_zmfree( &r );

}

TEST_F(LinearSolverTest, MKLFGMRESPreconditionedRestart) {
  printf("%% MKL FGMRES preconditioned restarted\n");

  // store initial guess in solution_vector
  data_d_matrix solution_vector = {Magma_DENSE};
  CHECK( data_zmconvert((*initialGuess_vector), &solution_vector, Magma_DENSE, Magma_DENSE) );

  data_z_gmres_param solverParam;

  solverParam.search_max = 2000;
  solverParam.restart_max = 20;
  solverParam.tol_type = 0;
  solverParam.rtol = (*LinearSolverTest::tolerance);
  solverParam.precondition = 1;

  // solve
  data_MKL_FGMRES( A, &solution_vector, rhs_vector, &solverParam );

  // print solver summary


  // caclulate residual
  dataType residual = 0.0;
  data_d_matrix r={Magma_DENSE};
  data_zvinit( &r, A->num_rows, 1, zero );

  data_z_spmv( negone, A, &solution_vector, zero, &r );
  data_zaxpy( A->num_rows, one, rhs_vector->val, 1, r.val, 1);
  for (int i=0; i<A->num_rows; ++i) {
    GMRESDBG("r.val[%d] = %.16e\n", i, r.val[i]);
  }
  residual = data_dnrm2( A->num_rows, r.val, 1 );
  printf("%% external check of rnorm2 = %.16e;\n\n", residual);

  fflush(stdout);

  EXPECT_LE( residual, (*LinearSolverTest::tolerance) );

  data_zmfree( &solution_vector );
  data_zmfree( &r );

}

TEST_F(LinearSolverTest, FGMRESPreconditioned) {
  printf("%% FGMRES preconditioned\n");

  // store initial guess in solution_vector
  data_d_matrix solution_vector = {Magma_DENSE};
  CHECK( data_zmconvert((*initialGuess_vector), &solution_vector, Magma_DENSE, Magma_DENSE) );

  data_z_gmres_param solverParam;
	data_d_gmres_log gmresLog;

  solverParam.search_max = 2000;
  solverParam.restart_max = 2000;
  solverParam.tol_type = 0;
  solverParam.rtol = (*LinearSolverTest::tolerance);
  solverParam.precondition = 1;
  solverParam.user_csrtrsv_choice = 0;

  gmresLog.restarts = 0;

  int maxthreads = 0;
  int numprocs = 0;
  #pragma omp parallel
  {
    maxthreads = omp_get_max_threads();
    numprocs = omp_get_num_procs();
  }

  printf("maxthreads = %d numprocs = %d\n", maxthreads, numprocs );

  // generate preconditioner
  data_d_matrix L = {Magma_CSRL};
  data_d_matrix U = {Magma_CSCU};
  data_d_preconditioner_log parilu_log;

  // PariLU is efficient when L is CSRL and U is CSCU
  // data_PariLU_v0_3 is hard coded to expect L is CSRL and U is CSCU
  data_PariLU_v0_3( A, &L, &U, solverParam.parilu_reduction, &parilu_log );
  printf("PariLU_v0_3_sweeps = %d\n", parilu_log.sweeps );
  printf("PariLU_v0_3_tol = %e\n", parilu_log.tol );
  printf("PariLU_v0_3_A_Frobenius = %e\n", parilu_log.A_Frobenius );
  printf("PariLU_v0_3_generation_time = %e\n", parilu_log.precond_generation_time );
  printf("PariLU_v0_3_initial_residual = %e\n", parilu_log.initial_residual );
  printf("PariLU_v0_3_initial_nonlinear_residual = %e\n", parilu_log.initial_nonlinear_residual );
  printf("PariLU_v0_3_omp_num_threads = %d\n", parilu_log.omp_num_threads );

  data_d_matrix Ucsr = {Magma_CSRU};
  CHECK( data_zmconvert( U, &Ucsr, Magma_CSC, Magma_CSR ) );
  Ucsr.storage_type = Magma_CSRU;
  Ucsr.fill_mode = MagmaUpper;

  omp_set_num_threads(numprocs);
  #pragma omp parallel
  {
    maxthreads = omp_get_max_threads();
    numprocs = omp_get_num_procs();
  }
  printf("maxthreads = %d numprocs = %d\n", maxthreads, numprocs );
  data_fgmres( A, rhs_vector, &solution_vector, &L, &Ucsr, &solverParam, &gmresLog );

  for (int i=0; i<A->num_rows; i++) {
    GMRESDBG("x.val[%d] = %.16e\n", i, x.val[i]);
  }

  data_d_matrix r={Magma_DENSE};
  data_zvinit( &r, A->num_rows, 1, zero );
  data_z_spmv( negone, A, &solution_vector, zero, &r );
  data_zaxpy( A->num_rows, one, rhs_vector->val, 1, r.val, 1);
  for (int i=0; i<A->num_rows; i++) {
    GMRESDBG("r.val[%d] = %.16e\n", i, r.val[i]);
  }
  dataType residual = 0.0;
  residual = data_dnrm2( A->num_rows, r.val, 1 );
  printf("%% external check of rnorm2 = %.16e;\n\n", residual);

  printf("gmres_search_directions = %d;\n", gmresLog.search_directions );
  printf("gmres_solve_time = %e;\n", gmresLog.solve_time );
  printf("gmres_initial_residual = %e;\n", gmresLog.initial_residual );
  printf("gmres_final_residual = %e;\n", gmresLog.final_residual );

  printf("\n\n");
  printf("%% ################################################################################\n");
  // printf("%% Matrix: %s\n%% \t%d -by- %d with %d non-zeros\n",
  //   sparse_filename, A->num_rows, A->num_cols, A->nnz );
  printf("%% Solver: FGMRES\n");
  printf("%% \trestarts: %d\n", gmresLog.restarts );
  printf("%% \tsearch directions: %d\n", gmresLog.search_directions );
  printf("%% \tsolve time [s]: %e\n", gmresLog.solve_time );
  printf("%% \tinitial residual: %e\n", gmresLog.initial_residual );
  printf("%% \tfinal residual: %e\n", gmresLog.final_residual );
  printf("%% ################################################################################\n");
  printf("\n\n");
  printf("%% Done.\n");
  fflush(stdout);

  EXPECT_LE( residual, (*LinearSolverTest::tolerance) );

  data_zmfree( &solution_vector );
  data_zmfree( &r );

}

TEST_F(LinearSolverTest, FGMRESPreconditionedRestart) {
  printf("%% FGMRES preconditioned restarted\n");

  // store initial guess in solution_vector
  data_d_matrix solution_vector = {Magma_DENSE};
  CHECK( data_zmconvert((*initialGuess_vector), &solution_vector, Magma_DENSE, Magma_DENSE) );

  data_z_gmres_param solverParam;
	data_d_gmres_log gmresLog;

  solverParam.search_max = 20;
  solverParam.restart_max = 100;
  solverParam.tol_type = 0;
  solverParam.rtol = (*LinearSolverTest::tolerance);
  solverParam.precondition = 1;
  solverParam.user_csrtrsv_choice = 0;

  gmresLog.restarts = 0;

  int maxthreads = 0;
  int numprocs = 0;
  #pragma omp parallel
  {
    maxthreads = omp_get_max_threads();
    numprocs = omp_get_num_procs();
  }

  printf("maxthreads = %d numprocs = %d\n", maxthreads, numprocs );

  // generate preconditioner
  data_d_matrix L = {Magma_CSRL};
  data_d_matrix U = {Magma_CSCU};
  data_d_preconditioner_log parilu_log;

  // PariLU is efficient when L is CSRL and U is CSCU
  // data_PariLU_v0_3 is hard coded to expect L is CSRL and U is CSCU
  data_PariLU_v0_3( A, &L, &U, solverParam.parilu_reduction, &parilu_log );
  printf("PariLU_v0_3_sweeps = %d\n", parilu_log.sweeps );
  printf("PariLU_v0_3_tol = %e\n", parilu_log.tol );
  printf("PariLU_v0_3_A_Frobenius = %e\n", parilu_log.A_Frobenius );
  printf("PariLU_v0_3_generation_time = %e\n", parilu_log.precond_generation_time );
  printf("PariLU_v0_3_initial_residual = %e\n", parilu_log.initial_residual );
  printf("PariLU_v0_3_initial_nonlinear_residual = %e\n", parilu_log.initial_nonlinear_residual );
  printf("PariLU_v0_3_omp_num_threads = %d\n", parilu_log.omp_num_threads );

  data_d_matrix Ucsr = {Magma_CSRU};
  CHECK( data_zmconvert( U, &Ucsr, Magma_CSC, Magma_CSR ) );
  Ucsr.storage_type = Magma_CSRU;
  Ucsr.fill_mode = MagmaUpper;

  omp_set_num_threads(numprocs);
  #pragma omp parallel
  {
    maxthreads = omp_get_max_threads();
    numprocs = omp_get_num_procs();
  }
  printf("maxthreads = %d numprocs = %d\n", maxthreads, numprocs );
  data_fgmres_restart( A, rhs_vector, &solution_vector, &L, &Ucsr, &solverParam, &gmresLog );

  for (int i=0; i<A->num_rows; i++) {
    GMRESDBG("x.val[%d] = %.16e\n", i, x.val[i]);
  }

  data_d_matrix r={Magma_DENSE};
  data_zvinit( &r, A->num_rows, 1, zero );
  data_z_spmv( negone, A, &solution_vector, zero, &r );
  data_zaxpy( A->num_rows, one, rhs_vector->val, 1, r.val, 1);
  for (int i=0; i<A->num_rows; i++) {
    GMRESDBG("r.val[%d] = %.16e\n", i, r.val[i]);
  }
  dataType residual = 0.0;
  residual = data_dnrm2( A->num_rows, r.val, 1 );
  printf("%% external check of rnorm2 = %.16e;\n\n", residual);

  printf("gmres_search_directions = %d;\n", gmresLog.search_directions );
  printf("gmres_solve_time = %e;\n", gmresLog.solve_time );
  printf("gmres_initial_residual = %e;\n", gmresLog.initial_residual );
  printf("gmres_final_residual = %e;\n", gmresLog.final_residual );

  printf("\n\n");
  printf("%% ################################################################################\n");
  // printf("%% Matrix: %s\n%% \t%d -by- %d with %d non-zeros\n",
  //   sparse_filename, A->num_rows, A->num_cols, A->nnz );
  printf("%% Solver: FGMRES\n");
  printf("%% \trestarts: %d\n", gmresLog.restarts );
  printf("%% \tsearch directions: %d\n", gmresLog.search_directions );
  printf("%% \tsolve time [s]: %e\n", gmresLog.solve_time );
  printf("%% \tinitial residual: %e\n", gmresLog.initial_residual );
  printf("%% \tfinal residual: %e\n", gmresLog.final_residual );
  printf("%% ################################################################################\n");
  printf("\n\n");
  printf("%% Done.\n");
  fflush(stdout);

  EXPECT_LE( residual, (*LinearSolverTest::tolerance) );

  data_zmfree( &solution_vector );
  data_zmfree( &r );

}

TEST_F(LinearSolverTest, FGMRESHouseholderPreconditioned) {
  printf("%% FGMRES Householder preconditioned\n");

  // store initial guess in solution_vector
  data_d_matrix solution_vector = {Magma_DENSE};
  CHECK( data_zmconvert((*initialGuess_vector), &solution_vector, Magma_DENSE, Magma_DENSE) );

  data_z_gmres_param solverParam;
  data_d_gmres_log gmresLog;

  solverParam.search_max = 2000;
  solverParam.restart_max = 0;
  solverParam.tol_type = 0;
  solverParam.rtol = (*LinearSolverTest::tolerance);
  solverParam.precondition = 1;
  solverParam.user_csrtrsv_choice = 0;

  gmresLog.restarts = 0;

  int maxthreads = 0;
  int numprocs = 0;
  #pragma omp parallel
  {
    maxthreads = omp_get_max_threads();
    numprocs = omp_get_num_procs();
  }

  printf("maxthreads = %d numprocs = %d\n", maxthreads, numprocs );

  // generate preconditioner
  data_d_matrix L = {Magma_CSRL};
  data_d_matrix U = {Magma_CSCU};
  data_d_preconditioner_log parilu_log;

  // PariLU is efficient when L is CSRL and U is CSCU
  // data_PariLU_v0_3 is hard coded to expect L is CSRL and U is CSCU
  data_PariLU_v0_3( A, &L, &U, solverParam.parilu_reduction, &parilu_log );
  printf("PariLU_v0_3_sweeps = %d\n", parilu_log.sweeps );
  printf("PariLU_v0_3_tol = %e\n", parilu_log.tol );
  printf("PariLU_v0_3_A_Frobenius = %e\n", parilu_log.A_Frobenius );
  printf("PariLU_v0_3_generation_time = %e\n", parilu_log.precond_generation_time );
  printf("PariLU_v0_3_initial_residual = %e\n", parilu_log.initial_residual );
  printf("PariLU_v0_3_initial_nonlinear_residual = %e\n", parilu_log.initial_nonlinear_residual );
  printf("PariLU_v0_3_omp_num_threads = %d\n", parilu_log.omp_num_threads );

  data_d_matrix Ucsr = {Magma_CSRU};
  CHECK( data_zmconvert( U, &Ucsr, Magma_CSC, Magma_CSR ) );
  Ucsr.storage_type = Magma_CSRU;
  Ucsr.fill_mode = MagmaUpper;

  omp_set_num_threads(numprocs);
  #pragma omp parallel
  {
    maxthreads = omp_get_max_threads();
    numprocs = omp_get_num_procs();
  }
  printf("maxthreads = %d numprocs = %d\n", maxthreads, numprocs );
  data_fgmres_householder( A, rhs_vector, &solution_vector, &L, &Ucsr, &solverParam, &gmresLog );

  for (int i=0; i<A->num_rows; i++) {
    GMRESDBG("x.val[%d] = %.16e\n", i, x.val[i]);
  }

  data_d_matrix r={Magma_DENSE};
  data_zvinit( &r, A->num_rows, 1, zero );
  data_z_spmv( negone, A, &solution_vector, zero, &r );
  data_zaxpy( A->num_rows, one, rhs_vector->val, 1, r.val, 1);
  for (int i=0; i<A->num_rows; i++) {
    GMRESDBG("r.val[%d] = %.16e\n", i, r.val[i]);
  }
  dataType residual = 0.0;
  residual = data_dnrm2( A->num_rows, r.val, 1 );
  printf("%% external check of rnorm2 = %.16e;\n\n", residual);

  printf("gmres_search_directions = %d;\n", gmresLog.search_directions );
  printf("gmres_solve_time = %e;\n", gmresLog.solve_time );
  printf("gmres_initial_residual = %e;\n", gmresLog.initial_residual );
  printf("gmres_final_residual = %e;\n", gmresLog.final_residual );

  printf("\n\n");
  printf("%% ################################################################################\n");
  // printf("%% Matrix: %s\n%% \t%d -by- %d with %d non-zeros\n",
  //   sparse_filename, A->num_rows, A->num_cols, A->nnz );
  printf("%% Solver: FGMRES\n");
  printf("%% \trestarts: %d\n", gmresLog.restarts );
  printf("%% \tsearch directions: %d\n", gmresLog.search_directions );
  printf("%% \tsolve time [s]: %e\n", gmresLog.solve_time );
  printf("%% \tinitial residual: %e\n", gmresLog.initial_residual );
  printf("%% \tfinal residual: %e\n", gmresLog.final_residual );
  printf("%% ################################################################################\n");
  printf("\n\n");
  printf("%% Done.\n");
  fflush(stdout);

  EXPECT_LE( residual, (*LinearSolverTest::tolerance) );

  data_zmfree( &solution_vector );
  data_zmfree( &r );

}

TEST_F(LinearSolverTest, FGMRESHouseholderPreconditionedRestart) {
  printf("%% FGMRES Householder preconditioned restarted\n");

  // store initial guess in solution_vector
  data_d_matrix solution_vector = {Magma_DENSE};
  CHECK( data_zmconvert((*LinearSolverTest::initialGuess_vector), &solution_vector, Magma_DENSE, Magma_DENSE) );

  data_z_gmres_param solverParam;
	data_d_gmres_log gmresLog;

  solverParam.tol_type = 0;
  solverParam.rtol = (*LinearSolverTest::tolerance);
  solverParam.search_max = 20;
  solverParam.restart_max = 20;
  solverParam.reorth = 0;
  solverParam.precondition = 1;
  solverParam.parilu_reduction = 1.0e-15;
  solverParam.monitorOrthog = 1;
  solverParam.user_csrtrsv_choice = 0;

  gmresLog.restarts = 0;

  int maxthreads = 0;
  int numprocs = 0;
  #pragma omp parallel
  {
    maxthreads = omp_get_max_threads();
    numprocs = omp_get_num_procs();
  }

  printf("maxthreads = %d numprocs = %d\n", maxthreads, numprocs );

  // generate preconditioner
  data_d_matrix L = {Magma_CSRL};
  data_d_matrix U = {Magma_CSCU};
  data_d_preconditioner_log parilu_log;

  // PariLU is efficient when L is CSRL and U is CSCU
  // data_PariLU_v0_3 is hard coded to expect L is CSRL and U is CSCU
  data_PariLU_v0_3( LinearSolverTest::A, &L, &U, solverParam.parilu_reduction, &parilu_log );
  printf("PariLU_v0_3_sweeps = %d\n", parilu_log.sweeps );
  printf("PariLU_v0_3_tol = %e\n", parilu_log.tol );
  printf("PariLU_v0_3_A_Frobenius = %e\n", parilu_log.A_Frobenius );
  printf("PariLU_v0_3_generation_time = %e\n", parilu_log.precond_generation_time );
  printf("PariLU_v0_3_initial_residual = %e\n", parilu_log.initial_residual );
  printf("PariLU_v0_3_initial_nonlinear_residual = %e\n", parilu_log.initial_nonlinear_residual );
  printf("PariLU_v0_3_omp_num_threads = %d\n", parilu_log.omp_num_threads );

  data_d_matrix Ucsr = {Magma_CSRU};
  CHECK( data_zmconvert( U, &Ucsr, Magma_CSC, Magma_CSR ) );
  Ucsr.storage_type = Magma_CSRU;
  Ucsr.fill_mode = MagmaUpper;

  omp_set_num_threads(numprocs);
  #pragma omp parallel
  {
    maxthreads = omp_get_max_threads();
    numprocs = omp_get_num_procs();
  }
  printf("maxthreads = %d numprocs = %d\n", maxthreads, numprocs );
  data_fgmres_householder_restart( LinearSolverTest::A, LinearSolverTest::rhs_vector, &solution_vector, &L, &Ucsr, &solverParam, &gmresLog );

  for (int i=0; i<A->num_rows; i++) {
    GMRESDBG("x.val[%d] = %.16e\n", i, x.val[i]);
  }

  data_d_matrix r={Magma_DENSE};
  data_zvinit( &r, A->num_rows, 1, zero );
  data_z_spmv( negone, A, &solution_vector, zero, &r );
  data_zaxpy( A->num_rows, one, rhs_vector->val, 1, r.val, 1);
  for (int i=0; i<A->num_rows; i++) {
    GMRESDBG("r.val[%d] = %.16e\n", i, r.val[i]);
  }
  dataType residual = 0.0;
  residual = data_dnrm2( A->num_rows, r.val, 1 );
  printf("%% external check of rnorm2 = %.16e;\n\n", residual);

  printf("gmres_search_directions = %d;\n", gmresLog.search_directions );
  printf("gmres_solve_time = %e;\n", gmresLog.solve_time );
  printf("gmres_initial_residual = %e;\n", gmresLog.initial_residual );
  printf("gmres_final_residual = %e;\n", gmresLog.final_residual );

  printf("\n\n");
  printf("%% ################################################################################\n");
  // printf("%% Matrix: %s\n%% \t%d -by- %d with %d non-zeros\n",
  //   sparse_filename, A->num_rows, A->num_cols, A->nnz );
  printf("%% Solver: FGMRES\n");
  printf("%% \trestarts: %d\n", gmresLog.restarts );
  printf("%% \tsearch directions: %d\n", gmresLog.search_directions );
  printf("%% \tsolve time [s]: %e\n", gmresLog.solve_time );
  printf("%% \tinitial residual: %e\n", gmresLog.initial_residual );
  printf("%% \tfinal residual: %e\n", gmresLog.final_residual );
  printf("%% ################################################################################\n");
  printf("\n\n");
  printf("%% Done.\n");
  fflush(stdout);

  EXPECT_LE( residual, (*LinearSolverTest::tolerance) );

  data_zmfree( &solution_vector );
  data_zmfree( &r );

}
