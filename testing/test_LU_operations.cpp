
#include <stdlib.h>
#include <stdio.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <vector>
#include <omp.h>

#include "mmio.h"
#include "sparse_types.h"
#include "container_tests.h"

TEST(Operations_LU, transpose) {
  dataType dense_valcheck[35] = {
    -8.223278758419077e-07, -1.733487757205144e-04, -1.880576133127482e-05, -2.678447499564818e-06, -1.285945782907584e-10,
    -1.160995150204430e-06, -2.042039430122352e-04, -2.715480591341862e-05, -3.804810692217438e-06, -1.346459050102805e-10,
    -2.498054174520410e-06, -2.609630608689073e-05, -4.216491255061494e-06, -7.410581788269939e-06, -8.830494084456525e-11,
    -7.299998974472012e-07, -1.533448376344399e-04, -1.513584767969564e-05, -2.284506549425184e-06, -1.154634463391157e-10,
    -3.074453995008234e-06,  7.156446011472396e-05,  6.609656789659304e-06, -6.443018197531840e-06, -8.824406759446290e-11,
    -3.910292961279981e-06, -2.327167539879282e-05, -5.225388422848710e-06, -1.126227737281539e-05, -9.714290066846522e-11,
    -4.653465034868186e-06, -1.710451419456480e-05, -5.879559848378777e-06, -1.330832780896548e-05,  1.389558807025642e-11};

  dataType dense_rowMajorTransposeCheck[35] = {
    -8.2232787584190771e-07, -1.1609951502044301e-06, -2.4980541745204101e-06, -7.2999989744720115e-07, -3.0744539950082341e-06, -3.9102929612799807e-06, -4.6534650348681863e-06,
    -1.7334877572051439e-04, -2.0420394301223519e-04, -2.6096306086890729e-05, -1.5334483763443989e-04,  7.1564460114723958e-05, -2.3271675398792820e-05, -1.7104514194564800e-05,
    -1.8805761331274822e-05, -2.7154805913418619e-05, -4.2164912550614941e-06, -1.5135847679695639e-05,  6.6096567896593036e-06, -5.2253884228487098e-06, -5.8795598483787767e-06,
    -2.6784474995648182e-06, -3.8048106922174378e-06, -7.4105817882699386e-06, -2.2845065494251841e-06, -6.4430181975318399e-06, -1.1262277372815390e-05, -1.3308327808965481e-05,
    -1.2859457829075839e-10, -1.3464590501028049e-10, -8.8304940844565251e-11, -1.1546344633911570e-10, -8.8244067594462898e-11, -9.7142900668465223e-11,  1.3895588070256420e-11};

  char dense_filename[] = "matrices/io_dense_test.mtx";

  data_d_matrix A = {Magma_DENSE};
  A.major = MagmaRowMajor;
  data_z_dense_mtx( &A, MagmaRowMajor, dense_filename );

  EXPECT_EQ(7, A.num_rows);
  EXPECT_EQ(5, A.num_cols);
  EXPECT_EQ(35, A.nnz);
  EXPECT_ARRAY_DOUBLE_EQ(A.nnz, A.val, dense_valcheck);

  data_d_matrix B = {Magma_DENSE};
  data_zmtranspose(A, &B);

  EXPECT_EQ(5, B.num_rows);
  EXPECT_EQ(7, B.num_cols);
  EXPECT_EQ(35, B.nnz);
  EXPECT_ARRAY_DOUBLE_EQ(B.nnz, B.val, dense_rowMajorTransposeCheck);

  // // data_zdisplay_dense( &A );
  // // data_zdisplay_dense( &B );

  // printf("make A column major\n");
  A.major = MagmaColMajor;
  data_int_t tmp = A.num_cols;
  A.num_cols = A.num_rows;
  A.num_rows = tmp;
  data_zmfree( &B );
  data_zmtranspose(A, &B);

  EXPECT_EQ(7, B.num_rows);
  EXPECT_EQ(5, B.num_cols);
  EXPECT_EQ(35, B.nnz);
  EXPECT_ARRAY_DOUBLE_EQ(B.nnz, B.val, dense_rowMajorTransposeCheck);

  // // data_zdisplay_dense( &A );
  // // data_zdisplay_dense( &B );

  data_d_matrix C = {Magma_DENSE};
  data_zmtranspose(B, &C);

  EXPECT_EQ(5, C.num_rows);
  EXPECT_EQ(7, C.num_cols);
  EXPECT_EQ(35, C.nnz);
  EXPECT_ARRAY_DOUBLE_EQ(C.nnz, C.val, dense_valcheck);

  // // data_zdisplay_dense( &A );
  // // data_zdisplay_dense( &C );

  data_zmfree( &A );
  data_zmfree( &B );
  data_zmfree( &C );

}

TEST(Operations_LU, extract_diag) {
  dataType diag_valcheck[5] = {
    -8.2232787584190771e-07,
    -2.0420394301223519e-04,
    -4.2164912550614941e-06,
    -2.2845065494251841e-06,
    -8.8244067594462898e-11};

  char dense_filename[] = "matrices/io_dense_test.mtx";

  data_d_matrix A = {Magma_DENSE};
  A.major = MagmaRowMajor;
  data_z_dense_mtx( &A, MagmaRowMajor, dense_filename );

  data_d_matrix E = {Magma_DENSE};
  data_zmconvert(A, &E, Magma_DENSE, Magma_DENSED);

  EXPECT_EQ(7, E.num_rows);
  EXPECT_EQ(5, E.num_cols);
  EXPECT_EQ(5, E.nnz);
  EXPECT_EQ(Magma_DENSED, E.storage_type);
  EXPECT_ARRAY_DOUBLE_EQ(E.nnz, E.val, diag_valcheck);

  // data_zprint_dense( E );
  data_zmfree( &A );
  data_zmfree( &E );

}

TEST(Operations_LU, extract_lower) {
  dataType lower_valcheck[35] = {
     0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00,
    -1.1609951502044301e-06,  0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00,
    -2.4980541745204101e-06, -2.6096306086890729e-05,  0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00,
    -7.2999989744720115e-07, -1.5334483763443989e-04, -1.5135847679695639e-05,  0.0000000000000000e+00,  0.0000000000000000e+00,
    -3.0744539950082341e-06,  7.1564460114723958e-05,  6.6096567896593036e-06, -6.4430181975318399e-06,  0.0000000000000000e+00,
    -3.9102929612799807e-06, -2.3271675398792820e-05, -5.2253884228487098e-06, -1.1262277372815390e-05, -9.7142900668465223e-11,
    -4.6534650348681863e-06, -1.7104514194564800e-05, -5.8795598483787767e-06, -1.3308327808965481e-05,  1.3895588070256420e-11};

  dataType lower_dunity_valcheck[35] = {
     1.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00,
    -1.1609951502044301e-06,  1.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00,
    -2.4980541745204101e-06, -2.6096306086890729e-05,  1.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00,
    -7.2999989744720115e-07, -1.5334483763443989e-04, -1.5135847679695639e-05,  1.0000000000000000e+00,  0.0000000000000000e+00,
    -3.0744539950082341e-06,  7.1564460114723958e-05,  6.6096567896593036e-06, -6.4430181975318399e-06,  1.0000000000000000e+00,
    -3.9102929612799807e-06, -2.3271675398792820e-05, -5.2253884228487098e-06, -1.1262277372815390e-05, -9.7142900668465223e-11,
    -4.6534650348681863e-06, -1.7104514194564800e-05, -5.8795598483787767e-06, -1.3308327808965481e-05,  1.3895588070256420e-11};

  char dense_filename[] = "matrices/io_dense_test.mtx";

  data_d_matrix A = {Magma_DENSE};
  A.major = MagmaRowMajor;
  data_z_dense_mtx( &A, MagmaRowMajor, dense_filename );

  data_d_matrix L = {Magma_DENSE};
  data_zmconvert(A, &L, Magma_DENSE, Magma_DENSEL);

  EXPECT_EQ(7, L.num_rows);
  EXPECT_EQ(5, L.num_cols);
  EXPECT_EQ(35, L.nnz);
  EXPECT_EQ(Magma_DENSEL, L.storage_type);
  EXPECT_EQ(MagmaLower, L.fill_mode);
  EXPECT_ARRAY_DOUBLE_EQ(L.nnz, L.val, lower_valcheck);

  // data_zdisplay_dense( &A );
  // data_zdisplay_dense( &L );

  data_zmfree( &L );
  L.diagorder_type = Magma_UNITY;
  data_zmconvert(A, &L, Magma_DENSE, Magma_DENSEL);

  EXPECT_EQ(7, L.num_rows);
  EXPECT_EQ(5, L.num_cols);
  EXPECT_EQ(35, L.nnz);
  EXPECT_EQ(Magma_DENSEL, L.storage_type);
  EXPECT_EQ(MagmaLower, L.fill_mode);
  EXPECT_EQ(Magma_UNITY, L.diagorder_type);
  EXPECT_ARRAY_DOUBLE_EQ(L.nnz, L.val, lower_dunity_valcheck);

  // data_zdisplay_dense( &A );
  // data_zdisplay_dense( &L );

  data_zmfree( &A );
  data_zmfree( &L );
}

TEST(Operations_LU, extract_upper) {
  dataType upper_valcheck[35] = {
    0.0000000000000000e+00, -1.7334877572051439e-04, -1.8805761331274822e-05, -2.6784474995648182e-06, -1.2859457829075839e-10,
    0.0000000000000000e+00,  0.0000000000000000e+00, -2.7154805913418619e-05, -3.8048106922174378e-06, -1.3464590501028049e-10,
    0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00, -7.4105817882699386e-06, -8.8304940844565251e-11,
    0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00, -1.1546344633911570e-10,
    0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00,
    0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00,
    0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00};

  dataType upper_dvalue_valcheck[35] = {
    -8.2232787584190771e-07, -1.7334877572051439e-04, -1.8805761331274822e-05, -2.6784474995648182e-06, -1.2859457829075839e-10,
     0.0000000000000000e+00, -2.0420394301223519e-04, -2.7154805913418619e-05, -3.8048106922174378e-06, -1.3464590501028049e-10,
     0.0000000000000000e+00,  0.0000000000000000e+00, -4.2164912550614941e-06, -7.4105817882699386e-06, -8.8304940844565251e-11,
     0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00, -2.2845065494251841e-06, -1.1546344633911570e-10,
     0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00, -8.8244067594462898e-11,
     0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00,
     0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00};

  char dense_filename[] = "matrices/io_dense_test.mtx";

  data_d_matrix A = {Magma_DENSE};
  A.major = MagmaRowMajor;
  data_z_dense_mtx( &A, MagmaRowMajor, dense_filename );

  data_d_matrix U = {Magma_DENSE};
  data_zmconvert(A, &U, Magma_DENSE, Magma_DENSEU);

  EXPECT_EQ(7, U.num_rows);
  EXPECT_EQ(5, U.num_cols);
  EXPECT_EQ(35, U.nnz);
  EXPECT_EQ(Magma_DENSEU, U.storage_type);
  EXPECT_EQ(MagmaUpper, U.fill_mode);
  EXPECT_ARRAY_DOUBLE_EQ(U.nnz, U.val, upper_valcheck);

  // data_zdisplay_dense( &A );
  // data_zdisplay_dense( &U );

  data_zmfree( &U );
  U.diagorder_type = Magma_VALUE;
  data_zmconvert(A, &U, Magma_DENSE, Magma_DENSEU);

  EXPECT_EQ(7, U.num_rows);
  EXPECT_EQ(5, U.num_cols);
  EXPECT_EQ(35, U.nnz);
  EXPECT_EQ(Magma_DENSEU, U.storage_type);
  EXPECT_EQ(MagmaUpper, U.fill_mode);
  EXPECT_EQ(Magma_VALUE, U.diagorder_type);
  EXPECT_ARRAY_DOUBLE_EQ(U.nnz, U.val, upper_dvalue_valcheck);

  // data_zdisplay_dense( &A );
  // data_zdisplay_dense( &U );

  data_zmfree( &A );
  data_zmfree( &U );
}
