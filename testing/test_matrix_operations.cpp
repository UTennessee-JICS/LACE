
#include <stdlib.h>
#include <stdio.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <vector>

#include "mmio.h"
#include "sparse_types.h"
#include "container_tests.h"

TEST(Operations_dense, dotproduct) {
  dataType dense_valcheck[35] = {
    -8.223278758419077e-07, -1.733487757205144e-04, -1.880576133127482e-05, -2.678447499564818e-06, -1.285945782907584e-10,
    -1.160995150204430e-06, -2.042039430122352e-04, -2.715480591341862e-05, -3.804810692217438e-06, -1.346459050102805e-10,
    -2.498054174520410e-06, -2.609630608689073e-05, -4.216491255061494e-06, -7.410581788269939e-06, -8.830494084456525e-11,
    -7.299998974472012e-07, -1.533448376344399e-04, -1.513584767969564e-05, -2.284506549425184e-06, -1.154634463391157e-10,
    -3.074453995008234e-06,  7.156446011472396e-05,  6.609656789659304e-06, -6.443018197531840e-06, -8.824406759446290e-11,
    -3.910292961279981e-06, -2.327167539879282e-05, -5.225388422848710e-06, -1.126227737281539e-05, -9.714290066846522e-11,
    -4.653465034868186e-06, -1.710451419456480e-05, -5.879559848378777e-06, -1.330832780896548e-05,  1.389558807025642e-11};

  data_d_matrix A = {Magma_DENSE};
  A.major = MagmaRowMajor;
  char dense_filename[] = "matrices/io_dense_test.mtx";
  data_z_dense_mtx( &A, A.major, dense_filename );

  EXPECT_EQ(7, A.num_rows);
  EXPECT_EQ(5, A.num_cols);
  EXPECT_EQ(35, A.nnz);

  dataType dense_val[35];
  for (int i=0; i<A.nnz; i++) {
    dense_val[i] = A.val[i];
  }

  EXPECT_THAT(dense_val, testing::ContainerEq(dense_valcheck));

  data_d_matrix B = {Magma_DENSE};
  B.major = MagmaRowMajor;
  data_z_dense_mtx( &B, B.major, dense_filename );

  dataType dotanswer = 0;

  dotanswer = data_zdot( 3, A.val, 1, B.val, 1 );
  EXPECT_THAT( dotanswer, testing::DoubleEq(3.040413092618546e-08));

  dotanswer = data_zdot( 5, A.val, 1, B.val, 1 );
  EXPECT_THAT( dotanswer, testing::DoubleEq(3.041130500720993e-08));

  dotanswer = data_zdot( 5, &A.val[5], 1, &B.val[5], 1 );
  EXPECT_THAT( dotanswer, testing::DoubleEq(4.245245832010018e-08));

  dotanswer = data_zdot( 7, A.val, 5, B.val, 5 );
  EXPECT_THAT( dotanswer, testing::DoubleEq(5.519470262449527e-11));

  dotanswer = data_zdot_mkl( 3, A.val, 1, B.val, 1 );
  EXPECT_THAT( dotanswer, testing::DoubleEq(3.040413092618546e-08));

  dotanswer = data_zdot_mkl( 5, A.val, 1, B.val, 1 );
  EXPECT_THAT( dotanswer, testing::DoubleEq(3.041130500720993e-08));

  dotanswer = data_zdot_mkl( 5, &A.val[5], 1, &B.val[5], 1 );
  EXPECT_THAT( dotanswer, testing::DoubleEq(4.245245832010018e-08));

  dotanswer = data_zdot_mkl( 7, A.val, 5, B.val, 5 );
  EXPECT_THAT( dotanswer, testing::DoubleEq(5.519470262449527e-11));

  data_zmfree( &A );
  data_zmfree( &B );
}

TEST(Operations_dense, dgemv) {
  dataType dense_valcheck[35] = {
    -8.223278758419077e-07, -1.733487757205144e-04, -1.880576133127482e-05, -2.678447499564818e-06, -1.285945782907584e-10,
    -1.160995150204430e-06, -2.042039430122352e-04, -2.715480591341862e-05, -3.804810692217438e-06, -1.346459050102805e-10,
    -2.498054174520410e-06, -2.609630608689073e-05, -4.216491255061494e-06, -7.410581788269939e-06, -8.830494084456525e-11,
    -7.299998974472012e-07, -1.533448376344399e-04, -1.513584767969564e-05, -2.284506549425184e-06, -1.154634463391157e-10,
    -3.074453995008234e-06,  7.156446011472396e-05,  6.609656789659304e-06, -6.443018197531840e-06, -8.824406759446290e-11,
    -3.910292961279981e-06, -2.327167539879282e-05, -5.225388422848710e-06, -1.126227737281539e-05, -9.714290066846522e-11,
    -4.653465034868186e-06, -1.710451419456480e-05, -5.879559848378777e-06, -1.330832780896548e-05,  1.389558807025642e-11};

  // TODO: set correct check values for tests
  dataType y_valcheck[7] = { 3.0411305007209926e-08, 3.592031602185543e-08, 4.6249601130439481e-09,
    2.6873500236137029e-08, -1.2510125678839689e-08, 4.1657648084751323e-09, 3.1150885255224793e-09 };
  dataType ytrans_valcheck[5] = { 1.2056625233510966e-09, 3.9962358391134244e-08, 6.0492468682242077e-09,
    3.5379316921156661e-09, 2.2691614026598202e-14};
  dataType alpha = 1.0;
  dataType beta = 0.0;
  dataType y[7];
  dataType ytrans[5];

  data_d_matrix A = {Magma_DENSE};
  A.major = MagmaRowMajor;
  char dense_filename[] = "matrices/io_dense_test.mtx";
  data_z_dense_mtx( &A, A.major, dense_filename );

  data_d_matrix B = {Magma_DENSE};
  data_z_dense_mtx( &B, MagmaRowMajor, dense_filename );

  EXPECT_EQ(7, A.num_rows);
  EXPECT_EQ(5, A.num_cols);
  EXPECT_EQ(35, A.nnz);

  dataType dense_val[35];
  for (int i=0; i<A.nnz; i++) {
    dense_val[i] = A.val[i];
  }

  EXPECT_THAT(dense_val, testing::ContainerEq(dense_valcheck));

  data_dgemv_mkl( A.major, MagmaNoTrans, 7, 5, alpha, A.val, A.ld,
    B.val, 1, beta, y, 1);

  EXPECT_THAT(y, testing::ContainerEq(y_valcheck));
  //EXPECT_ARRAY_DOUBLE_EQ(7, y, y_valcheck);

  data_dgemv_mkl( A.major, MagmaTrans, 7, 5, alpha, A.val, A.ld,
    B.val, 1, beta, ytrans, 1);

  EXPECT_THAT(ytrans, testing::ContainerEq(ytrans_valcheck));
  EXPECT_ARRAY_DOUBLE_EQ(5, ytrans, ytrans_valcheck);

  data_zmfree( &A );
  data_zmfree( &B );
}

TEST(Operations_dense, dgemm) {
  dataType dense_valcheck[35] = {
    -8.223278758419077e-07, -1.733487757205144e-04, -1.880576133127482e-05, -2.678447499564818e-06, -1.285945782907584e-10,
    -1.160995150204430e-06, -2.042039430122352e-04, -2.715480591341862e-05, -3.804810692217438e-06, -1.346459050102805e-10,
    -2.498054174520410e-06, -2.609630608689073e-05, -4.216491255061494e-06, -7.410581788269939e-06, -8.830494084456525e-11,
    -7.299998974472012e-07, -1.533448376344399e-04, -1.513584767969564e-05, -2.284506549425184e-06, -1.154634463391157e-10,
    -3.074453995008234e-06,  7.156446011472396e-05,  6.609656789659304e-06, -6.443018197531840e-06, -8.824406759446290e-11,
    -3.910292961279981e-06, -2.327167539879282e-05, -5.225388422848710e-06, -1.126227737281539e-05, -9.714290066846522e-11,
    -4.653465034868186e-06, -1.710451419456480e-05, -5.879559848378777e-06, -1.330832780896548e-05,  1.389558807025642e-11};

  dataType dgemm_valcheck[49] = {
     3.041130500720992e-08,  3.592031602185544e-08,  4.624960113043948e-09,  2.687350023613703e-08, -1.251012567883969e-08,  4.165764808475131e-09,  3.115088525522480e-09,
     3.592031602185544e-08,  4.245245832010018e-08,  5.474562692281120e-09,  3.173417113290265e-08, -1.476514499152824e-08,  4.941452949922816e-09,  3.708505866656276e-09,
     4.624960113043948e-09,  5.474562692281120e-09,  7.599529869919749e-10,  4.084307091064051e-09, -1.840010950188961e-09,  7.225657202084352e-10,  5.814028100217863e-10,
     2.687350023613703e-08,  3.173417113290265e-08,  4.084307091064051e-09,  2.374948498415365e-08, -1.105711980669046e-08,  3.676265228636337e-09,  2.745681065308330e-09,
    -1.251012567883969e-08, -1.476514499152824e-08, -1.840010950188961e-09, -1.105711980669046e-08,  5.216124265257941e-09, -1.615377836063668e-09, -1.162884534112968e-09,
     4.165764808475131e-09,  4.941452949922816e-09,  7.225657202084352e-10,  3.676265228636337e-09, -1.615377836063668e-09,  7.110048427111224e-10,  5.968521768761003e-10,
     3.115088525522480e-09,  3.708505866656276e-09,  5.814028100217863e-10,  2.745681065308330e-09, -1.162884534112968e-09,  5.968521768761003e-10,  5.258999557445544e-10  };

  dataType dgemmsub1_valcheck[9] = {
     3.040413092618546e-08, 3.591012503615323e-08, 4.605111258771481e-09,
     3.591012503615323e-08, 4.243798173567843e-08, 5.446366831445670e-09,
     4.605111258771481e-09, 5.446366831445669e-09, 7.050362645435391e-10 };

  dataType dgemmsub2_valcheck[9] = {
     3.608307402733330e-10, 5.208577847070427e-10, 9.914318247056162e-11,
     5.208577847070427e-10, 7.518600686171765e-10, 1.426938625022729e-10,
     9.914318247056162e-11, 1.426938625022729e-10, 7.269552095244590e-11 };

  dataType dgemmsub3_valcheck[9] = {
     7.050362645435391e-10,  4.067377568423501e-09, -1.887757463512875e-09,
     4.067377568423501e-09,  2.374426601396595e-08, -1.107183892397098e-08,
    -1.887757463512875e-09, -1.107183892397098e-08,  5.174611781756428e-09 };

  dataType dgemmsub4_valcheck[9] = {
     7.269552095244590e-11,  8.074969201992913e-11,  1.987695327135737e-11,
     8.074969201992913e-11,  2.343128551706463e-10, -8.532364110283319e-11,
     1.987695327135736e-11, -8.532364110283319e-11,  8.520004637860279e-11 };

  dataType alpha = 1.0;
  dataType beta = 0.0;

  data_d_matrix A = {Magma_DENSE};
  A.major = MagmaRowMajor;
  char dense_filename[] = "matrices/io_dense_test.mtx";
  data_z_dense_mtx( &A, A.major, dense_filename );
  data_d_matrix B = {Magma_DENSE};
  data_z_dense_mtx( &B, MagmaRowMajor, dense_filename );

  EXPECT_EQ(7, A.num_rows);
  EXPECT_EQ(5, A.num_cols);
  EXPECT_EQ(35, A.nnz);

  dataType dense_val[35];
  for (int i=0; i<A.nnz; i++) {
    dense_val[i] = A.val[i];
  }

  EXPECT_THAT(dense_val, testing::ContainerEq(dense_valcheck));

  // TODO: write a create matrix method that initializes structure
  data_d_matrix C = {Magma_DENSE};
  C.num_rows = 7;
  C.num_cols = 7;
  C.ld = 7;
  C.nnz = 49;
  C.val = (dataType*) malloc( C.num_rows*C.num_cols*sizeof(dataType) );

  printf("\nfull matrices\nA.ld=%d, B.ld=%d, C.ld=%d\n", A.ld, B.ld, C.ld);
  data_dgemm_mkl( A.major, MagmaNoTrans, MagmaTrans, 7, 7, 5,
    alpha, A.val, A.ld,
    B.val, B.ld, beta, C.val, C.ld);
  dataType dgemm_val[49];
  for (int i=0; i<C.nnz; i++) {
    dgemm_val[i] = C.val[i];
  }

  EXPECT_ARRAY_DOUBLE_EQ(49, dgemm_val, dgemm_valcheck);
  //EXPECT_THAT(dgemm_val, testing::ContainerEq(dgemm_valcheck));

  data_d_matrix D = {Magma_DENSE};
  D.num_rows = 3;
  D.num_cols = 3;
  D.ld = 3;
  D.nnz = 9;
  D.val = (dataType*) malloc( D.num_rows*D.num_cols*sizeof(dataType) );

  printf("\nsub matrices 1\nA.ld=%d, B.ld=%d, C.ld=%d\n", A.ld, B.ld, D.ld);
  data_dgemm_mkl( A.major, MagmaNoTrans, MagmaTrans, 3, 3, 3,
    alpha, A.val, A.ld,
    B.val, B.ld, beta, D.val, D.ld);

  dataType dgemmsub_val[9];
  for (int i=0; i<D.nnz; i++) {
    dgemmsub_val[i] = D.val[i];
  }

  EXPECT_ARRAY_DOUBLE_EQ(9, dgemmsub_val, dgemmsub1_valcheck);

  printf("\nsub matrices 2\nA.ld=%d, B.ld=%d, C.ld=%d\n", A.ld, B.ld, D.ld);
  data_dgemm_mkl( A.major, MagmaNoTrans, MagmaTrans, 3, 3, 3,
    alpha, &A.val[2], A.ld,
    &B.val[2], B.ld, beta, D.val, D.ld);
  for (int i=0; i<D.nnz; i++) {
    dgemmsub_val[i] = D.val[i];
  }
  EXPECT_ARRAY_DOUBLE_EQ(9, dgemmsub_val, dgemmsub2_valcheck);

  printf("\nsub matrices 3\nA.ld=%d, B.ld=%d, C.ld=%d\n", A.ld, B.ld, D.ld);
  data_dgemm_mkl( A.major, MagmaNoTrans, MagmaTrans, 3, 3, 3,
    alpha, &A.val[2*A.ld], A.ld,
    &B.val[2*B.ld], B.ld, beta, D.val, D.ld);
  for (int i=0; i<D.nnz; i++) {
    dgemmsub_val[i] = D.val[i];
  }
  EXPECT_ARRAY_DOUBLE_EQ(9, dgemmsub_val, dgemmsub3_valcheck);

  printf("\nsub matrices 4\nA.ld=%d, B.ld=%d, C.ld=%d\n", A.ld, B.ld, D.ld);
  data_dgemm_mkl( A.major, MagmaNoTrans, MagmaTrans, 3, 3, 3,
    alpha, &A.val[2*A.ld+2], A.ld,
    &B.val[2*B.ld+2], B.ld, beta, D.val, D.ld);
  for (int i=0; i<D.nnz; i++) {
    dgemmsub_val[i] = D.val[i];
  }
  EXPECT_ARRAY_DOUBLE_EQ(9, dgemmsub_val, dgemmsub4_valcheck);

  data_zmfree( &A );
  data_zmfree( &B );
  data_zmfree( &C );
  data_zmfree( &D );
}
