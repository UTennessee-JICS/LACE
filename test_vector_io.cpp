/*
 *
 #mac osx
 * g++ -isystem ${GTEST_DIR}/include -I${GTEST_DIR} -pthread -c ${GTEST_DIR}/src/gtest-all.cc
 * ar -rv libgtest.a gtest-all.o
 * g++ -std=c++11 -isystem ${GTEST_DIR}/include -pthread example_02.cpp libgtest.a -o example_02
 * g++ -std=c++11 -isystem ${GTEST_DIR}/include -isystem ${GMOCK_DIR}/include -pthread example_02.cpp libgtest.a libgmock.a -o example_02
 *
 #beacon
 * icpc -isystem ${GTEST_DIR}/include -I${GTEST_DIR} -pthread -c ${GTEST_DIR}/src/gtest-all.cc
 * ar -rv libgtest.a gtest-all.o
 * icpc -std=c++11 -isystem ${GTEST_DIR}/include -pthread example_02.cpp libgtest.a -o example_02
 *
 * ./example_02 --gtest_output="xml:report.xml"
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "include/mmio.h"
#include "include/sparse_types.h"

TEST(ReadColumn, WorksForMTXformat){
  dataType dense_valcheck[30] = { -8.223278758419077e-07, -1.733487757205144e-04,
                                  -1.880576133127482e-05, -2.678447499564818e-06, -1.285945782907584e-10,
                                  -1.160995150204430e-06, -2.042039430122352e-04, -2.715480591341862e-05,
                                  -3.804810692217438e-06, -1.346459050102805e-10, -2.498054174520410e-06,
                                  -2.609630608689073e-05, -4.216491255061494e-06, -7.410581788269939e-06,
                                  -8.830494084456525e-11, -7.299998974472012e-07, -1.533448376344399e-04,
                                  -1.513584767969564e-05, -2.284506549425184e-06, -1.154634463391157e-10,
                                  -3.074453995008234e-06,  7.156446011472396e-05,  6.609656789659304e-06,
                                  -6.443018197531840e-06, -8.824406759446290e-11, -3.910292961279981e-06,
                                  -2.327167539879282e-05, -5.225388422848710e-06, -1.126227737281539e-05,
                                  -9.714290066846522e-11 };

  // data_order_t A_order = MagmaRowMajor;
  data_d_matrix A = { Magma_DENSE };

  A.major = MagmaRowMajor;
  char dense_filename[] = "testing/vectors/io_column_test.mtx";
  data_z_dense_mtx(&A, A.major, dense_filename);

  EXPECT_EQ(30, A.num_rows);
  EXPECT_EQ(1, A.num_cols);
  EXPECT_EQ(30, A.nnz);

  dataType dense_val[30];
  for (int i = 0; i < A.nnz; i++) {
    dense_val[i] = A.val[i];
  }

  EXPECT_THAT(dense_val, testing::ContainerEq(dense_valcheck));
  data_zmfree(&A);
}

TEST(ReadRow, WorksForMTXformat){
  dataType dense_valcheck[30] = { -8.223278758419077e-07, -1.733487757205144e-04,
                                  -1.880576133127482e-05, -2.678447499564818e-06, -1.285945782907584e-10,
                                  -1.160995150204430e-06, -2.042039430122352e-04, -2.715480591341862e-05,
                                  -3.804810692217438e-06, -1.346459050102805e-10, -2.498054174520410e-06,
                                  -2.609630608689073e-05, -4.216491255061494e-06, -7.410581788269939e-06,
                                  -8.830494084456525e-11, -7.299998974472012e-07, -1.533448376344399e-04,
                                  -1.513584767969564e-05, -2.284506549425184e-06, -1.154634463391157e-10,
                                  -3.074453995008234e-06,  7.156446011472396e-05,  6.609656789659304e-06,
                                  -6.443018197531840e-06, -8.824406759446290e-11, -3.910292961279981e-06,
                                  -2.327167539879282e-05, -5.225388422848710e-06, -1.126227737281539e-05,
                                  -9.714290066846522e-11 };

  // data_order_t A_order = MagmaRowMajor;
  data_d_matrix A = { Magma_DENSE };

  A.major = MagmaRowMajor;
  char dense_filename[] = "testing/vectors/io_row_test.mtx";
  data_z_dense_mtx(&A, A.major, dense_filename);

  EXPECT_EQ(1, A.num_rows);
  EXPECT_EQ(30, A.num_cols);
  EXPECT_EQ(30, A.nnz);

  dataType dense_val[30];
  for (int i = 0; i < A.nnz; i++) {
    dense_val[i] = A.val[i];
  }

  EXPECT_THAT(dense_val, testing::ContainerEq(dense_valcheck));
  data_zmfree(&A);
}


int
main(int argc, char * argv[])
{
  data_storage_t A_dense_storage = Magma_DENSE;
  data_order_t A_order = MagmaRowMajor;

  int dense_nrow, dense_ncol, dense_nnz;
  dataType * dense_val;
  char dense_filename[] = "testing/vectors/io_row_test.mtx";

  read_z_dense_from_mtx(&A_dense_storage, &dense_nrow, &dense_ncol, &dense_nnz,
    A_order, &dense_val, dense_filename);
  data_zprint_dense_mtx(dense_nrow, dense_ncol, dense_nnz, A_order, &dense_val);

  data_d_matrix D = { Magma_DENSE };
  data_z_dense_mtx(&D, A_order, dense_filename);
  data_zprint_dense(D);

  free(dense_val);
  data_zmfree(&D);

  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
