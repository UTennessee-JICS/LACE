
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
  solverParam. precondition = 0;

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
