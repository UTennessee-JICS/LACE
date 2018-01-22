#include <stdlib.h>
#include <stdio.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "mmio.h"
#include "sparse_types.h"
#include "container_tests.h"


TEST(ReadCOO, WorksForMTXformat){
  int coo_rowcheck[14]      = { 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 3, 4 };
  int coo_colcheck[14]      = { 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4 };
  dataType coo_valcheck[14] = { -948.1011349, -7178501.646,  4.731272996,
                                35742.61854,   23349.69309, -24613410.87, -3005.164596, 12934346.29,
                                4.731272996,   35670.21095, -3120.860678, -3250082.045, 29953.98635,
                                -10035133.8 };

  data_storage_t A_storage = Magma_COO;
  int nrow, ncol, nnz;
  int * row, * col;
  dataType * val;
  char filename[] = "matrices/io_test.mtx";

  read_z_coo_from_mtx(&A_storage, &nrow, &ncol, &nnz, &val, &row, &col, filename);

  EXPECT_EQ(4, nrow);
  EXPECT_EQ(4, ncol);
  EXPECT_EQ(14, nnz);

  int coo_row[14];
  for (int i = 0; i < nnz; i++) {
    coo_row[i] = row[i] + 1;
  }
  int coo_col[14];
  for (int i = 0; i < nnz; i++) {
    coo_col[i] = col[i] + 1;
  }
  dataType coo_val[14];
  for (int i = 0; i < nnz; i++) {
    coo_val[i] = val[i];
  }

  EXPECT_THAT(coo_row, testing::ContainerEq(coo_rowcheck));
  EXPECT_THAT(coo_col, testing::ContainerEq(coo_colcheck));
  EXPECT_THAT(coo_val, testing::ContainerEq(coo_valcheck));
}

TEST(ReadCOOconvertCSR, WorksForMTXformat){
  int coo_rowcheck[14]      = { 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 3, 4 };
  int coo_colcheck[14]      = { 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4 };
  dataType coo_valcheck[14] = { -948.1011349, -7178501.646,  4.731272996,
                                35742.61854,   23349.69309, -24613410.87, -3005.164596, 12934346.29,
                                4.731272996,   35670.21095, -3120.860678, -3250082.045, 29953.98635,
                                -10035133.8 };

  int csr_rowcheck[5]       = { 0, 3, 6, 10, 14 };
  int csr_colcheck[14]      = { 0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3 };
  dataType csr_valcheck[14] = { -948.1011349,  23349.69309, 4.731272996,
                                -7178501.646, -24613410.87, 35670.21095, 4.731272996, -3005.164596,
                                -3120.860678,  29953.98635, 35742.61854, 12934346.29, -3250082.045,
                                -10035133.8 };
  char filename[] = "matrices/io_test.mtx";

  data_d_matrix A = { Magma_COO };

  data_z_coo_mtx(&A, filename);

  EXPECT_EQ(4, A.num_rows);
  EXPECT_EQ(4, A.num_cols);
  EXPECT_EQ(14, A.nnz);

  int coo_row[14];
  for (int i = 0; i < A.nnz; i++) {
    coo_row[i] = A.row[i] + 1;
  }
  int coo_col[14];
  for (int i = 0; i < A.nnz; i++) {
    coo_col[i] = A.col[i] + 1;
  }
  dataType coo_val[14];
  for (int i = 0; i < A.nnz; i++) {
    coo_val[i] = A.val[i];
  }

  EXPECT_THAT(coo_row, testing::ContainerEq(coo_rowcheck));
  EXPECT_THAT(coo_col, testing::ContainerEq(coo_colcheck));
  EXPECT_THAT(coo_val, testing::ContainerEq(coo_valcheck));

  data_d_matrix B = { Magma_CSR };
  data_zmconvert(A, &B, Magma_COO, Magma_CSR);

  EXPECT_EQ(4, B.num_rows);
  EXPECT_EQ(4, B.num_cols);
  EXPECT_EQ(14, B.nnz);

  int csr_row[5];
  for (int i = 0; i < (B.num_rows + 1); i++) {
    csr_row[i] = B.row[i];
  }
  int csr_col[14];
  for (int i = 0; i < B.nnz; i++) {
    csr_col[i] = B.col[i];
  }
  dataType csr_val[14];
  for (int i = 0; i < B.nnz; i++) {
    csr_val[i] = B.val[i];
  }

  EXPECT_THAT(csr_row, testing::ContainerEq(csr_rowcheck));
  EXPECT_THAT(csr_col, testing::ContainerEq(csr_colcheck));
  EXPECT_THAT(csr_val, testing::ContainerEq(csr_valcheck));
  data_zmfree(&A);
  data_zmfree(&B);
}

TEST(ReadCOOtoCSR, WorksForMTXformat){
  int csr_rowcheck[5]       = { 0, 3, 6, 10, 14 };
  int csr_colcheck[14]      = { 0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3 };
  dataType csr_valcheck[14] = { -948.1011349,  23349.69309, 4.731272996,
                                -7178501.646, -24613410.87, 35670.21095, 4.731272996, -3005.164596,
                                -3120.860678,  29953.98635, 35742.61854, 12934346.29, -3250082.045,
                                -10035133.8 };

  data_storage_t A_storage = Magma_CSR;
  int nrow, ncol, nnz;
  int * row, * col;
  dataType * val;
  char filename[] = "matrices/io_test.mtx";

  read_z_csr_from_mtx(&A_storage, &nrow, &ncol, &nnz, &val, &row, &col, filename);

  EXPECT_EQ(4, nrow);
  EXPECT_EQ(4, ncol);
  EXPECT_EQ(14, nnz);

  int csr_row[5];
  for (int i = 0; i < nrow + 1; i++) {
    csr_row[i] = row[i];
  }
  int csr_col[14];
  for (int i = 0; i < nnz; i++) {
    csr_col[i] = col[i];
  }
  dataType csr_val[14];
  for (int i = 0; i < nnz; i++) {
    csr_val[i] = val[i];
  }

  EXPECT_THAT(csr_row, testing::ContainerEq(csr_rowcheck));
  EXPECT_THAT(csr_col, testing::ContainerEq(csr_colcheck));
  EXPECT_THAT(csr_val, testing::ContainerEq(csr_valcheck));
}

TEST(ReadDense, WorksForMTXformat){
  dataType dense_valcheck[35] = { -8.223278758419077e-07, -1.733487757205144e-04,
                                  -1.880576133127482e-05, -2.678447499564818e-06, -1.285945782907584e-10,
                                  -1.160995150204430e-06, -2.042039430122352e-04, -2.715480591341862e-05,
                                  -3.804810692217438e-06, -1.346459050102805e-10, -2.498054174520410e-06,
                                  -2.609630608689073e-05, -4.216491255061494e-06, -7.410581788269939e-06,
                                  -8.830494084456525e-11, -7.299998974472012e-07, -1.533448376344399e-04,
                                  -1.513584767969564e-05, -2.284506549425184e-06, -1.154634463391157e-10,
                                  -3.074453995008234e-06,  7.156446011472396e-05,  6.609656789659304e-06,
                                  -6.443018197531840e-06, -8.824406759446290e-11, -3.910292961279981e-06,
                                  -2.327167539879282e-05, -5.225388422848710e-06, -1.126227737281539e-05,
                                  -9.714290066846522e-11, -4.653465034868186e-06, -1.710451419456480e-05,
                                  -5.879559848378777e-06, -1.330832780896548e-05, 1.389558807025642e-11 };

  // data_order_t A_order = MagmaRowMajor;
  data_d_matrix A = { Magma_DENSE };

  A.major = MagmaRowMajor;
  char dense_filename[] = "matrices/io_dense_test.mtx";
  data_z_dense_mtx(&A, A.major, dense_filename);

  EXPECT_EQ(7, A.num_rows);
  EXPECT_EQ(5, A.num_cols);
  EXPECT_EQ(35, A.nnz);

  dataType dense_val[35];
  for (int i = 0; i < A.nnz; i++) {
    dense_val[i] = A.val[i];
  }

  EXPECT_THAT(dense_val, testing::ContainerEq(dense_valcheck));
  data_zmfree(&A);
}

TEST(ReadCOOtoCSRtoDense, WorksForMTXformat){
  int csr_rowcheck[5]       = { 0, 3, 6, 10, 14 };
  int csr_colcheck[14]      = { 0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3 };
  dataType csr_valcheck[14] = { -948.1011349,  23349.69309, 4.731272996,
                                -7178501.646, -24613410.87, 35670.21095, 4.731272996, -3005.164596,
                                -3120.860678,  29953.98635, 35742.61854, 12934346.29, -3250082.045,
                                -10035133.8 };
  dataType dense_valcheck[16] = { -948.1011349,  23349.69309,  4.731272996,
                                  0.0,          -7178501.646, -24613410.87, 35670.21095,          0.0,
                                  4.731272996,
                                  -3005.164596,
                                  -3120.860678,  29953.98635,  35742.61854, 12934346.29, -3250082.045,
                                  -10035133.8 };

  char filename[] = "matrices/io_test.mtx";
  data_d_matrix A = { Magma_CSR };

  data_z_csr_mtx(&A, filename);

  EXPECT_EQ(4, A.num_rows);
  EXPECT_EQ(4, A.num_cols);
  EXPECT_EQ(14, A.nnz);

  int csr_row[5];
  for (int i = 0; i < A.num_rows + 1; i++) {
    csr_row[i] = A.row[i];
  }
  int csr_col[14];
  for (int i = 0; i < A.nnz; i++) {
    csr_col[i] = A.col[i];
  }
  dataType csr_val[14];
  for (int i = 0; i < A.nnz; i++) {
    csr_val[i] = A.val[i];
  }

  EXPECT_THAT(csr_row, testing::ContainerEq(csr_rowcheck));
  EXPECT_THAT(csr_col, testing::ContainerEq(csr_colcheck));
  EXPECT_THAT(csr_val, testing::ContainerEq(csr_valcheck));

  data_d_matrix B = { Magma_DENSE };
  data_zmconvert(A, &B, Magma_CSR, Magma_DENSE);

  EXPECT_EQ(4, B.num_rows);
  EXPECT_EQ(4, B.num_cols);
  EXPECT_EQ(14, B.true_nnz);

  dataType dense_val[16];
  for (int i = 0; i < B.num_rows * B.num_cols; i++) {
    dense_val[i] = B.val[i];
  }

  EXPECT_THAT(dense_val, testing::ContainerEq(dense_valcheck));
  EXPECT_ARRAY_DOUBLE_EQ(B.num_rows * B.num_cols, dense_valcheck, B.val);
  data_zmfree(&A);
  data_zmfree(&B);
}

TEST(PadCSR, WorksForMTXformat){
  int csr_rowcheck[5]       = { 0, 3, 6, 10, 14 };
  int csr_colcheck[14]      = { 0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3 };
  dataType csr_valcheck[14] = { -948.1011349,  23349.69309, 4.731272996,
                                -7178501.646, -24613410.87, 35670.21095, 4.731272996, -3005.164596,
                                -3120.860678,  29953.98635, 35742.61854, 12934346.29, -3250082.045,
                                -10035133.8 };
  int pad_csr_rowcheck[7]       = { 0, 3, 6, 10, 14, 15, 16 };
  int pad_csr_colcheck[16]      = { 0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5 };
  dataType pad_csr_valcheck[16] = { -948.1011349,  23349.69309, 4.731272996,
                                    -7178501.646, -24613410.87, 35670.21095, 4.731272996, -3005.164596,
                                    -3120.860678,  29953.98635, 35742.61854, 12934346.29, -3250082.045,
                                    -10035133.8,           1.0, 1.0 };

  char filename[] = "matrices/io_test.mtx";
  data_d_matrix A = { Magma_CSR };

  data_z_csr_mtx(&A, filename);

  EXPECT_EQ(4, A.num_rows);
  EXPECT_EQ(4, A.num_cols);
  EXPECT_EQ(14, A.nnz);

  int csr_row[5];
  for (int i = 0; i < A.num_rows + 1; i++) {
    csr_row[i] = A.row[i];
  }
  int csr_col[14];
  for (int i = 0; i < A.nnz; i++) {
    csr_col[i] = A.col[i];
  }
  dataType csr_val[14];
  for (int i = 0; i < A.nnz; i++) {
    csr_val[i] = A.val[i];
  }

  EXPECT_THAT(csr_row, testing::ContainerEq(csr_rowcheck));
  EXPECT_THAT(csr_col, testing::ContainerEq(csr_colcheck));
  EXPECT_THAT(csr_val, testing::ContainerEq(csr_valcheck));

  data_z_pad_csr(&A, 3);

  EXPECT_EQ(4, A.num_rows);
  EXPECT_EQ(4, A.num_cols);
  EXPECT_EQ(6, A.pad_rows);
  EXPECT_EQ(6, A.pad_cols);
  EXPECT_EQ(16, A.nnz);
  EXPECT_EQ(14, A.true_nnz);

  int pad_csr_row[7];
  for (int i = 0; i < A.pad_rows + 1; i++) {
    pad_csr_row[i] = A.row[i];
  }
  int pad_csr_col[16];
  for (int i = 0; i < A.nnz; i++) {
    pad_csr_col[i] = A.col[i];
  }
  dataType pad_csr_val[16];
  for (int i = 0; i < A.nnz; i++) {
    pad_csr_val[i] = A.val[i];
  }

  EXPECT_THAT(pad_csr_row, testing::ContainerEq(pad_csr_rowcheck));
  EXPECT_THAT(pad_csr_col, testing::ContainerEq(pad_csr_colcheck));
  EXPECT_THAT(pad_csr_val, testing::ContainerEq(pad_csr_valcheck));

  data_zmfree(&A);
}

TEST(PadDense, WorksForMTXformat){
  int csr_rowcheck[5]       = { 0, 3, 6, 10, 14 };
  int csr_colcheck[14]      = { 0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3 };
  dataType csr_valcheck[14] = { -948.1011349,  23349.69309, 4.731272996,
                                -7178501.646, -24613410.87, 35670.21095, 4.731272996, -3005.164596,
                                -3120.860678,  29953.98635, 35742.61854, 12934346.29, -3250082.045,
                                -10035133.8 };
  dataType dense_valcheck[16] = { -948.1011349,  23349.69309,  4.731272996,
                                  0.0,          -7178501.646, -24613410.87, 35670.21095,          0.0,
                                  4.731272996,
                                  -3005.164596,
                                  -3120.860678,  29953.98635,  35742.61854, 12934346.29, -3250082.045,
                                  -10035133.8 };
  dataType pad_valcheck[36] = { -948.1011349,  23349.69309,  4.731272996,         0.0, 0.0, 0.0,
                                -7178501.646, -24613410.87,  35670.21095,         0.0, 0.0, 0.0,
                                4.731272996,  -3005.164596, -3120.860678, 29953.98635, 0.0, 0.0,
                                35742.61854,   12934346.29, -3250082.045, -10035133.8, 0.0, 0.0,
                                0.0,                   0.0,          0.0,         0.0, 1.0, 0.0,
                                0.0,                   0.0,          0.0,         0.0, 0.0, 1.0 };

  char filename[] = "matrices/io_test.mtx";
  data_d_matrix A = { Magma_CSR };

  data_z_csr_mtx(&A, filename);

  EXPECT_EQ(4, A.num_rows);
  EXPECT_EQ(4, A.num_cols);
  EXPECT_EQ(14, A.nnz);

  int csr_row[5];
  for (int i = 0; i < A.num_rows + 1; i++) {
    csr_row[i] = A.row[i];
  }
  int csr_col[14];
  for (int i = 0; i < A.nnz; i++) {
    csr_col[i] = A.col[i];
  }
  dataType csr_val[14];
  for (int i = 0; i < A.nnz; i++) {
    csr_val[i] = A.val[i];
  }

  EXPECT_THAT(csr_row, testing::ContainerEq(csr_rowcheck));
  EXPECT_THAT(csr_col, testing::ContainerEq(csr_colcheck));
  EXPECT_THAT(csr_val, testing::ContainerEq(csr_valcheck));

  data_d_matrix B = { Magma_DENSE };
  data_zmconvert(A, &B, Magma_CSR, Magma_DENSE);

  EXPECT_EQ(4, B.num_rows);
  EXPECT_EQ(4, B.num_cols);
  EXPECT_EQ(14, B.true_nnz);

  dataType dense_val[16];
  for (int i = 0; i < B.num_rows * B.num_cols; i++) {
    dense_val[i] = B.val[i];
  }

  EXPECT_THAT(dense_val, testing::ContainerEq(dense_valcheck));

  data_z_pad_dense(&B, 3);

  EXPECT_EQ(6, B.pad_rows);
  EXPECT_EQ(6, B.pad_cols);
  EXPECT_EQ(14, B.true_nnz);
  dataType pad_val[36];
  for (int i = 0; i < B.pad_rows * B.pad_cols; i++) {
    pad_val[i] = B.val[i];
  }
  EXPECT_THAT(pad_val, testing::ContainerEq(pad_valcheck));

  data_zmfree(&A);
  data_zmfree(&B);
}

TEST(TransposeDense, WorksForMTXformat){
  int csr_rowcheck[5]       = { 0, 3, 6, 10, 14 };
  int csr_colcheck[14]      = { 0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3 };
  dataType csr_valcheck[14] = { -948.1011349,  23349.69309, 4.731272996,
                                -7178501.646, -24613410.87, 35670.21095, 4.731272996, -3005.164596,
                                -3120.860678,  29953.98635, 35742.61854, 12934346.29, -3250082.045,
                                -10035133.8 };
  dataType dense_valcheck[16] = { -948.1011349,  23349.69309,  4.731272996,
                                  0.0,          -7178501.646, -24613410.87, 35670.21095,          0.0,
                                  4.731272996,
                                  -3005.164596,
                                  -3120.860678,  29953.98635,  35742.61854, 12934346.29, -3250082.045,
                                  -10035133.8 };
  dataType pad_valcheck[36] = { -948.1011349,  23349.69309,  4.731272996,         0.0, 0.0, 0.0,
                                -7178501.646, -24613410.87,  35670.21095,         0.0, 0.0, 0.0,
                                4.731272996,  -3005.164596, -3120.860678, 29953.98635, 0.0, 0.0,
                                35742.61854,   12934346.29, -3250082.045, -10035133.8, 0.0, 0.0,
                                0.0,                   0.0,          0.0,         0.0, 1.0, 0.0,
                                0.0,                   0.0,          0.0,         0.0, 0.0, 1.0 };
  dataType trans_valcheck[36] = { -948.1011349, -7178501.646,  4.731272996,  35742.61854, 0.0, 0.0,
                                  23349.69309,  -24613410.87, -3005.164596,  12934346.29, 0.0, 0.0,
                                  4.731272996,   35670.21095, -3120.860678, -3250082.045, 0.0, 0.0,
                                  0.0,                   0.0,  29953.98635,  -10035133.8, 0.0, 0.0,
                                  0.0,                   0.0,          0.0,          0.0, 1.0, 0.0,
                                  0.0,                   0.0,          0.0,          0.0, 0.0, 1.0 };

  char filename[] = "matrices/io_test.mtx";
  data_d_matrix A = { Magma_CSR };

  data_z_csr_mtx(&A, filename);

  EXPECT_EQ(4, A.num_rows);
  EXPECT_EQ(4, A.num_cols);
  EXPECT_EQ(14, A.nnz);

  int csr_row[5];
  for (int i = 0; i < A.num_rows + 1; i++) {
    csr_row[i] = A.row[i];
  }
  int csr_col[14];
  for (int i = 0; i < A.nnz; i++) {
    csr_col[i] = A.col[i];
  }
  dataType csr_val[14];
  for (int i = 0; i < A.nnz; i++) {
    csr_val[i] = A.val[i];
  }

  EXPECT_THAT(csr_row, testing::ContainerEq(csr_rowcheck));
  EXPECT_THAT(csr_col, testing::ContainerEq(csr_colcheck));
  EXPECT_THAT(csr_val, testing::ContainerEq(csr_valcheck));

  data_d_matrix B = { Magma_DENSE };
  data_zmconvert(A, &B, Magma_CSR, Magma_DENSE);

  EXPECT_EQ(4, B.num_rows);
  EXPECT_EQ(4, B.num_cols);
  EXPECT_EQ(14, B.true_nnz);

  dataType dense_val[16];
  for (int i = 0; i < B.num_rows * B.num_cols; i++) {
    dense_val[i] = B.val[i];
  }

  EXPECT_THAT(dense_val, testing::ContainerEq(dense_valcheck));

  data_z_pad_dense(&B, 3);

  EXPECT_EQ(6, B.pad_rows);
  EXPECT_EQ(6, B.pad_cols);
  EXPECT_EQ(14, B.true_nnz);
  dataType pad_val[36];
  for (int i = 0; i < B.pad_rows * B.pad_cols; i++) {
    pad_val[i] = B.val[i];
  }
  EXPECT_THAT(pad_val, testing::ContainerEq(pad_valcheck));

  data_d_matrix C = { Magma_DENSE };
  data_zmtranspose(B, &C);

  EXPECT_EQ(6, C.pad_rows);
  EXPECT_EQ(6, C.pad_cols);
  EXPECT_EQ(14, C.true_nnz);
  dataType trans_val[36];
  for (int i = 0; i < C.pad_rows * C.pad_cols; i++) {
    trans_val[i] = C.val[i];
  }
  EXPECT_THAT(trans_val, testing::ContainerEq(trans_valcheck));

  data_zmfree(&A);
  data_zmfree(&B);
  data_zmfree(&C);
}

TEST(TransposeCSR, WorksForMTXformat){
  int csr_rowcheck[5]       = { 0, 3, 6, 10, 14 };
  int csr_colcheck[14]      = { 0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3 };
  dataType csr_valcheck[14] = { -948.1011349,  23349.69309, 4.731272996,
                                -7178501.646, -24613410.87, 35670.21095, 4.731272996, -3005.164596,
                                -3120.860678,  29953.98635, 35742.61854, 12934346.29, -3250082.045,
                                -10035133.8 };
  int trans_rowcheck[5]       = { 0, 4, 8, 12, 14 };
  int trans_colcheck[14]      = { 0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3 };
  dataType trans_valcheck[14] = { -948.1011349, -7178501.646,  4.731272996,
                                  35742.61854,   23349.69309, -24613410.87, -3005.164596, 12934346.29,
                                  4.731272996,   35670.21095, -3120.860678, -3250082.045, 29953.98635,
                                  -10035133.8 };

  data_d_matrix A = { Magma_CSR };
  char filename[] = "matrices/io_test.mtx";

  data_z_csr_mtx(&A, filename);

  EXPECT_EQ(4, A.num_rows);
  EXPECT_EQ(4, A.num_cols);
  EXPECT_EQ(14, A.nnz);

  int csr_row[5];
  for (int i = 0; i < A.num_rows + 1; i++) {
    csr_row[i] = A.row[i];
  }
  int csr_col[14];
  for (int i = 0; i < A.nnz; i++) {
    csr_col[i] = A.col[i];
  }
  dataType csr_val[14];
  for (int i = 0; i < A.nnz; i++) {
    csr_val[i] = A.val[i];
  }

  EXPECT_THAT(csr_row, testing::ContainerEq(csr_rowcheck));
  EXPECT_THAT(csr_col, testing::ContainerEq(csr_colcheck));
  EXPECT_THAT(csr_val, testing::ContainerEq(csr_valcheck));

  data_d_matrix B = { Magma_CSR };
  data_zmtranspose(A, &B);

  EXPECT_EQ(4, B.num_rows);
  EXPECT_EQ(4, B.num_cols);
  EXPECT_EQ(14, B.nnz);

  int trans_row[5];
  for (int i = 0; i < B.num_rows + 1; i++) {
    trans_row[i] = B.row[i];
  }
  int trans_col[14];
  for (int i = 0; i < B.nnz; i++) {
    trans_col[i] = A.col[i];
  }
  dataType trans_val[14];
  for (int i = 0; i < B.nnz; i++) {
    trans_val[i] = B.val[i];
  }

  EXPECT_THAT(trans_row, testing::ContainerEq(trans_rowcheck));
  EXPECT_THAT(trans_col, testing::ContainerEq(trans_colcheck));
  EXPECT_THAT(trans_val, testing::ContainerEq(trans_valcheck));
}

TEST(CSRtoCSC, WorksForMTXformat){
  int csr_rowcheck[5]       = { 0, 3, 6, 10, 14 };
  int csr_colcheck[14]      = { 0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3 };
  dataType csr_valcheck[14] = { -948.1011349,  23349.69309, 4.731272996,
                                -7178501.646, -24613410.87, 35670.21095, 4.731272996, -3005.164596,
                                -3120.860678,  29953.98635, 35742.61854, 12934346.29, -3250082.045,
                                -10035133.8 };
  int trans_rowcheck[5]       = { 0, 4, 8, 12, 14 };
  int trans_colcheck[14]      = { 0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3 };
  dataType trans_valcheck[14] = { -948.1011349, -7178501.646,  4.731272996,
                                  35742.61854,   23349.69309, -24613410.87, -3005.164596, 12934346.29,
                                  4.731272996,   35670.21095, -3120.860678, -3250082.045, 29953.98635,
                                  -10035133.8 };

  data_d_matrix A = { Magma_CSR };
  char filename[] = "matrices/io_test.mtx";

  data_z_csr_mtx(&A, filename);

  EXPECT_EQ(4, A.num_rows);
  EXPECT_EQ(4, A.num_cols);
  EXPECT_EQ(14, A.nnz);

  int csr_row[5];
  for (int i = 0; i < A.num_rows + 1; i++) {
    csr_row[i] = A.row[i];
  }
  int csr_col[14];
  for (int i = 0; i < A.nnz; i++) {
    csr_col[i] = A.col[i];
  }
  dataType csr_val[14];
  for (int i = 0; i < A.nnz; i++) {
    csr_val[i] = A.val[i];
  }

  EXPECT_THAT(csr_row, testing::ContainerEq(csr_rowcheck));
  EXPECT_THAT(csr_col, testing::ContainerEq(csr_colcheck));
  EXPECT_THAT(csr_val, testing::ContainerEq(csr_valcheck));

  data_d_matrix B = { Magma_CSC };
  data_zmconvert(A, &B, Magma_CSR, Magma_CSC);

  EXPECT_EQ(4, B.num_rows);
  EXPECT_EQ(4, B.num_cols);
  EXPECT_EQ(14, B.nnz);

  int trans_row[5];
  for (int i = 0; i < B.num_rows + 1; i++) {
    trans_row[i] = B.row[i];
  }
  int trans_col[14];
  for (int i = 0; i < B.nnz; i++) {
    trans_col[i] = A.col[i];
  }
  dataType trans_val[14];
  for (int i = 0; i < B.nnz; i++) {
    trans_val[i] = B.val[i];
  }

  EXPECT_THAT(trans_row, testing::ContainerEq(trans_rowcheck));
  EXPECT_THAT(trans_col, testing::ContainerEq(trans_colcheck));
  EXPECT_THAT(trans_val, testing::ContainerEq(trans_valcheck));
}


TEST(CSRtoCSRL, WorksForMTXformat){
  int csr_rowcheck[5]       = { 0, 3, 6, 10, 14 };
  int csr_colcheck[14]      = { 0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3 };
  dataType csr_valcheck[14] = { -948.1011349,  23349.69309, 4.731272996,
                                -7178501.646, -24613410.87, 35670.21095, 4.731272996, -3005.164596,
                                -3120.860678,  29953.98635, 35742.61854, 12934346.29, -3250082.045,
                                -10035133.8 };
  int csrl_rowcheck[5]       = { 0, 1, 3, 6, 10 };
  int csrl_colcheck[10]      = { 0, 1, 2, 0, 1, 2, 0, 1, 2, 3 };
  dataType csrl_valcheck[10] = { -948.1011349, -7178501.646, -24613410.87,
                                 4.731272996,  -3005.164596, -3120.860678,35742.61854, 12934346.29,
                                 -3250082.045, -10035133.8 };

  data_d_matrix A = { Magma_CSR };
  char filename[] = "matrices/io_test.mtx";

  data_z_csr_mtx(&A, filename);

  EXPECT_EQ(4, A.num_rows);
  EXPECT_EQ(4, A.num_cols);
  EXPECT_EQ(14, A.nnz);

  int csr_row[5];
  for (int i = 0; i < A.num_rows + 1; i++) {
    csr_row[i] = A.row[i];
  }
  int csr_col[14];
  for (int i = 0; i < A.nnz; i++) {
    csr_col[i] = A.col[i];
  }
  dataType csr_val[14];
  for (int i = 0; i < A.nnz; i++) {
    csr_val[i] = A.val[i];
  }

  EXPECT_THAT(csr_row, testing::ContainerEq(csr_rowcheck));
  EXPECT_THAT(csr_col, testing::ContainerEq(csr_colcheck));
  EXPECT_THAT(csr_val, testing::ContainerEq(csr_valcheck));

  data_d_matrix B = { Magma_CSRL };
  B.diagorder_type = Magma_VALUE;
  data_zmconvert(A, &B, Magma_CSR, Magma_CSRL);

  EXPECT_EQ(4, B.num_rows);
  EXPECT_EQ(4, B.num_cols);
  EXPECT_EQ(10, B.nnz);

  int csrl_row[5];
  for (int i = 0; i < B.num_rows + 1; i++) {
    csrl_row[i] = B.row[i];
  }
  int csrl_col[10];
  for (int i = 0; i < B.nnz; i++) {
    csrl_col[i] = A.col[i];
  }
  dataType csrl_val[10];
  for (int i = 0; i < B.nnz; i++) {
    csrl_val[i] = B.val[i];
  }

  EXPECT_THAT(csrl_row, testing::ContainerEq(csrl_rowcheck));
  EXPECT_THAT(csrl_col, testing::ContainerEq(csrl_colcheck));
  EXPECT_THAT(csrl_val, testing::ContainerEq(csrl_valcheck));
}

TEST(CSRtoCSRU, WorksForMTXformat){
  int csr_rowcheck[5]       = { 0, 3, 6, 10, 14 };
  int csr_colcheck[14]      = { 0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3 };
  dataType csr_valcheck[14] = { -948.1011349,  23349.69309, 4.731272996,
                                -7178501.646, -24613410.87, 35670.21095, 4.731272996, -3005.164596,
                                -3120.860678,  29953.98635, 35742.61854, 12934346.29, -3250082.045,
                                -10035133.8 };
  int csru_rowcheck[5]      = { 0, 3, 5, 7, 8 };
  int csru_colcheck[8]      = { 0, 1, 2, 0, 1, 2, 0, 1 };
  dataType csru_valcheck[8] = { -948.1011349, 23349.69309,  4.731272996,
                                -24613410.87, 35670.21095, -3120.860678, 29953.98635, -10035133.8 };

  data_d_matrix A = { Magma_CSR };
  char filename[] = "matrices/io_test.mtx";

  data_z_csr_mtx(&A, filename);

  EXPECT_EQ(4, A.num_rows);
  EXPECT_EQ(4, A.num_cols);
  EXPECT_EQ(14, A.nnz);

  int csr_row[5];
  for (int i = 0; i < A.num_rows + 1; i++) {
    csr_row[i] = A.row[i];
  }
  int csr_col[14];
  for (int i = 0; i < A.nnz; i++) {
    csr_col[i] = A.col[i];
  }
  dataType csr_val[14];
  for (int i = 0; i < A.nnz; i++) {
    csr_val[i] = A.val[i];
  }

  EXPECT_THAT(csr_row, testing::ContainerEq(csr_rowcheck));
  EXPECT_THAT(csr_col, testing::ContainerEq(csr_colcheck));
  EXPECT_THAT(csr_val, testing::ContainerEq(csr_valcheck));

  data_d_matrix B = { Magma_CSRU };
  data_zmconvert(A, &B, Magma_CSR, Magma_CSRU);

  EXPECT_EQ(4, B.num_rows);
  EXPECT_EQ(4, B.num_cols);
  EXPECT_EQ(8, B.nnz);

  int csru_row[5];
  for (int i = 0; i < B.num_rows + 1; i++) {
    csru_row[i] = B.row[i];
  }
  int csru_col[8];
  for (int i = 0; i < B.nnz; i++) {
    csru_col[i] = A.col[i];
  }
  dataType csru_val[8];
  for (int i = 0; i < B.nnz; i++) {
    csru_val[i] = B.val[i];
  }

  EXPECT_THAT(csru_row, testing::ContainerEq(csru_rowcheck));
  EXPECT_THAT(csru_col, testing::ContainerEq(csru_colcheck));
  EXPECT_THAT(csru_val, testing::ContainerEq(csru_valcheck));
}

dataType *
getValCSR(data_d_matrix * A, int r, int c)
{
  dataType * ptr = NULL;

  for (int j = A->row[r]; j < A->row[r + 1]; ++j) {
    if (j == c) {
      ptr = &(A->val[j]);
      break;
    }
  }
  return ptr;
}
