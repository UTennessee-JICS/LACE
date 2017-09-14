#
#
CC=g++-6
#
#
VPATH=blas control include src testing
CFLAGS=-std=c++11 -g -Wall -O2 -fno-unsafe-math-optimizations -fopenmp
LDFLAGS=-isystem ${GTEST_DIR}/include -pthread libgtest.a
LDFLAGS2=-isystem ${GTEST_DIR}/include -isystem ${GMOCK_DIR}/include -pthread libgmock.a
SOURCES=example_01.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=exampleGoogleTest_01

all: gtest-all.o gmock-all.o exampleGoogleTest_01 exampleGoogleTest_02 \
  test_matrix_io test_vector_io \
	test_matrix_ops test_spmatrix_ops test_blockspmatrix_ops \
	test_LU_ops test_LU test_LU_larnv test_MKL_LU \
	test_iLU_ops test_iLU test_MKL_iLU post_iLU \
	test_MKL_iLU0_FGMRES test_pariLU0_MKL_FGMRES \
	test_read_pariLU0_MKL_FGMRES test_read_rhs_pariLU0_MKL_FGMRES \
	test_solve_pariLU0_MKL_FGMRES \
	test_trisolve \
	test_solve_pariLU0_partrsv_MKL_FGMRES \
	test_dense_trisolve

exampleGoogleTest_01: example_01.cpp libgtest.a
	$(CC) $(LDFLAGS) $(CFLAGS) example_01.cpp -o $@

exampleGoogleTest_02: example_02.cpp libgmock.a
	$(CC) $(LDFLAGS2) $(CFLAGS) example_02.cpp -o $@

test_matrix_io: test_matrix_io.cpp libgmock.a
	$(CC) $(LDFLAGS2) $(CFLAGS) \
	-L${MKLROOT}/lib -I${MKLROOT}/include \
	test_matrix_io.cpp control/magma_zmio.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp \
	-lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lstdc++ -lm \
	-o $@

test_vector_io: test_vector_io.cpp libgmock.a
	$(CC) $(LDFLAGS2) $(CFLAGS) \
	-L${MKLROOT}/lib -I${MKLROOT}/include \
	test_vector_io.cpp control/magma_zmio.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp \
	-lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lstdc++ -lm \
	-o $@

test_matrix_ops: test_matrix_operations.cpp libgmock.a
	$(CC) $(LDFLAGS2) $(CFLAGS) \
	-L${MKLROOT}/lib -I${MKLROOT}/include \
	test_matrix_operations.cpp control/constants.cpp control/magma_zmio.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp blas/zdot.cpp blas/zgemv.cpp blas/zgemm.cpp \
	-lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lstdc++ -lm \
	-o $@

test_spmatrix_ops: test_spmatrix_operations.cpp libgmock.a
	$(CC) $(LDFLAGS2) $(CFLAGS) \
	-L${MKLROOT}/lib -I${MKLROOT}/include \
	test_spmatrix_operations.cpp control/constants.cpp control/magma_zmio.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp control/sparse_sub.cpp control/sparse_tilepattern.cpp \
	blas/zdot.cpp blas/zgemv.cpp blas/zgemm.cpp blas/zspmm.cpp \
	-lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lstdc++ -lm \
	-o $@

test_blockspmatrix_ops: test_blockspmatrix_operations.cpp libgmock.a
	$(CC) $(LDFLAGS2) $(CFLAGS) \
	-L${MKLROOT}/lib -I${MKLROOT}/include \
	test_blockspmatrix_operations.cpp control/constants.cpp control/magma_zmio.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp control/sparse_sub.cpp control/sparse_tilepattern.cpp \
	blas/zdot.cpp blas/zgemv.cpp blas/zgemm.cpp blas/zspmm.cpp \
	-lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lstdc++ -lm \
	-o $@

test_LU_ops: test_LU_operations.cpp libgmock.a
	$(CC) $(LDFLAGS2) $(CFLAGS) \
	-L${MKLROOT}/lib -I${MKLROOT}/include \
	test_LU_operations.cpp control/constants.cpp control/magma_zmio.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp control/magma_zmatrixchar.cpp control/norms.cpp \
	control/sparse_sub.cpp control/sparse_tilepattern.cpp \
	blas/zdiff.cpp blas/zdot.cpp blas/zgemv.cpp blas/zgemm.cpp \
	blas/zcsrilu0.cpp blas/zlunp.cpp blas/zspmm.cpp \
	-lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lstdc++ -lm \
	-o $@

test_LU: test_LU.cpp
	$(CC) $(CFLAGS) \
	-L${MKLROOT}/lib -I${MKLROOT}/include \
	test_LU.cpp control/constants.cpp control/magma_zmio.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp control/magma_zmatrixchar.cpp control/norms.cpp \
	control/sparse_sub.cpp control/sparse_tilepattern.cpp \
	blas/zdiff.cpp blas/zdot.cpp blas/zgemv.cpp blas/zgemm.cpp \
	blas/zcsrilu0.cpp blas/zlunp.cpp blas/zspmm.cpp \
	src/parlu_v0_0.cpp src/parlu_v0_1.cpp \
	src/parlu_v1_0.cpp src/parlu_v1_1.cpp src/parlu_v1_2.cpp src/parlu_v1_3.cpp \
	src/parlu_v2_0.cpp src/parlu_v3_0.cpp \
	-lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lstdc++ -lm \
	-o $@

test_LU_larnv: test_LU_larnv.cpp
	$(CC) $(CFLAGS) \
	-L${MKLROOT}/lib -I${MKLROOT}/include \
	test_LU_larnv.cpp control/constants.cpp control/magma_zmio.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp control/magma_zmatrixchar.cpp control/norms.cpp \
	control/sparse_sub.cpp control/sparse_tilepattern.cpp \
	blas/zdiff.cpp blas/zdot.cpp blas/zgemv.cpp blas/zgemm.cpp \
	blas/zcsrilu0.cpp blas/zlunp.cpp blas/zspmm.cpp \
	src/parlu_v0_0.cpp src/parlu_v0_1.cpp \
	src/parlu_v1_0.cpp src/parlu_v1_1.cpp src/parlu_v1_2.cpp src/parlu_v1_2c.cpp src/parlu_v1_3.cpp \
	src/parlu_v2_0.cpp src/parlu_v2_1.cpp src/parlu_v3_0.cpp src/parlu_v3_1.cpp \
	-lmkl_intel_lp64 -lmkl_sequential -lmkl_core -liomp5 -lpthread -lstdc++ -lm -ldl \
	-o $@

test_MKL_LU: test_MKL_LU.cpp
	$(CC) $(CFLAGS) \
	-L${MKLROOT}/lib -I${MKLROOT}/include \
	test_MKL_LU.cpp control/constants.cpp control/magma_zmio.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp control/magma_zmatrixchar.cpp control/norms.cpp \
	control/sparse_sub.cpp control/sparse_tilepattern.cpp \
	blas/zdiff.cpp blas/zdot.cpp blas/zgemv.cpp blas/zgemm.cpp \
	blas/zcsrilu0.cpp blas/zlunp.cpp blas/zspmm.cpp \
	-lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lstdc++ -lm \
	-o $@

test_MKL_LU_larnv: test_MKL_LU_larnv.cpp
	$(CC) $(CFLAGS) \
	-L${MKLROOT}/lib -I${MKLROOT}/include \
	test_MKL_LU_larnv.cpp control/constants.cpp control/magma_zmio.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp control/magma_zmatrixchar.cpp control/norms.cpp \
	control/sparse_sub.cpp control/sparse_tilepattern.cpp \
	blas/zdiff.cpp blas/zdot.cpp blas/zgemv.cpp blas/zgemm.cpp \
	blas/zcsrilu0.cpp blas/zlunp.cpp blas/zspmm.cpp \
	-lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lstdc++ -lm \
	-o $@

test_iLU_ops: test_iLU_operations.cpp libgmock.a
	$(CC) $(LDFLAGS2) $(CFLAGS) \
	-L${MKLROOT}/lib -I${MKLROOT}/include \
	test_iLU_operations.cpp control/constants.cpp control/magma_zmio.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp control/magma_zmatrixchar.cpp control/norms.cpp \
	control/sparse_sub.cpp control/sparse_tilepattern.cpp \
	blas/zdiff.cpp blas/zdot.cpp blas/zgemv.cpp blas/zgemm.cpp \
	blas/zcsrilu0.cpp blas/zlunp.cpp blas/zspmm.cpp \
	-lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lstdc++ -lm \
	src/parilu_v0_0.cpp src/parilu_v3_0.cpp -o $@

test_iLU: test_iLU.cpp libgmock.a
	$(CC) $(LDFLAGS2) $(CFLAGS) \
	-L${MKLROOT}/lib -I${MKLROOT}/include \
	test_iLU.cpp control/constants.cpp control/magma_zmio.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp control/magma_zmatrixchar.cpp control/norms.cpp \
	control/sparse_sub.cpp control/sparse_tilepattern.cpp \
	control/magma_zmscale.cpp \
	blas/zdiff.cpp blas/zdot.cpp blas/zgemv.cpp blas/zgemm.cpp \
	blas/zcsrilu0.cpp blas/zlunp.cpp blas/zspmm.cpp \
	src/parilu_v0_0.cpp src/parilu_v0_1.cpp src/parilu_v0_2.cpp \
	 src/parilu_v0_3.cpp  src/parilu_v0_4.cpp \
	src/parilu_v3_0.cpp src/parilu_v3_1.cpp \
	src/parilu_v3_2.cpp src/parilu_v3_5.cpp src/parilu_v3_6.cpp \
	src/parilu_v3_7.cpp src/parilu_v3_8.cpp src/parilu_v3_9.cpp  \
	-lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lstdc++ -lm \
	src/parilu_v4_0.cpp -o $@

test_MKL_iLU: test_MKL_iLU.cpp libgmock.a
	$(CC) $(LDFLAGS2) $(CFLAGS) \
	-L${MKLROOT}/lib -I${MKLROOT}/include \
	test_MKL_iLU.cpp control/constants.cpp control/magma_zmio.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp control/magma_zmatrixchar.cpp control/norms.cpp \
	control/sparse_sub.cpp control/sparse_tilepattern.cpp \
	control/magma_zmscale.cpp \
	blas/zdiff.cpp blas/zdot.cpp blas/zgemv.cpp blas/zgemm.cpp \
	blas/zcsrilu0.cpp blas/zlunp.cpp blas/zspmm.cpp \
	-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lpthread -lstdc++ -lm \
	-o $@

post_iLU: post_iLU.cpp libgmock.a
	$(CC) $(LDFLAGS2) $(CFLAGS) \
	-L${MKLROOT}/lib -I${MKLROOT}/include \
	post_iLU.cpp control/constants.cpp control/magma_zmio.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp control/magma_zmatrixchar.cpp control/norms.cpp \
	blas/zdiff.cpp blas/zdot.cpp blas/zgemv.cpp blas/zgemm.cpp \
	blas/zcsrilu0.cpp blas/zlunp.cpp blas/zspmm.cpp \
	-lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lstdc++ -lm \
	-o $@

test_MKL_iLU0_FGMRES: test_MKL_iLU0_FGMRES.cpp
	$(CC) $(CFLAGS) \
	-L${MKLROOT}/lib -I${MKLROOT}/include \
	-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lstdc++ -lm -ldl \
	test_MKL_iLU0_FGMRES.cpp -o $@

test_pariLU0_MKL_FGMRES: test_pariLU0_MKL_FGMRES.cpp
	$(CC) $(CFLAGS) \
	-L${MKLROOT}/lib -I${MKLROOT}/include \
	control/constants.cpp control/magma_zmio.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp control/magma_zmatrixchar.cpp control/norms.cpp \
	control/magma_zmlumerge.cpp \
	blas/zdiff.cpp blas/zdot.cpp blas/zgemv.cpp blas/zgemm.cpp \
	blas/zcsrilu0.cpp blas/zlunp.cpp blas/zspmm.cpp \
	src/parilu_v0_2.cpp \
	-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lstdc++ -lm -ldl \
	test_pariLU0_MKL_FGMRES.cpp -o $@

test_read_pariLU0_MKL_FGMRES: test_read_pariLU0_MKL_FGMRES.cpp
	$(CC) $(CFLAGS) \
	-L${MKLROOT}/lib -I${MKLROOT}/include \
	control/constants.cpp control/magma_zmio.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp control/magma_zmatrixchar.cpp control/norms.cpp \
	control/magma_zmlumerge.cpp control/magma_zmscale.cpp \
	blas/zdiff.cpp blas/zdot.cpp blas/zgemv.cpp blas/zgemm.cpp \
	blas/zcsrilu0.cpp blas/zlunp.cpp blas/zspmm.cpp \
	src/parilu_v0_2.cpp \
	-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lstdc++ -lm -ldl \
	test_read_pariLU0_MKL_FGMRES.cpp -o $@

test_read_rhs_pariLU0_MKL_FGMRES: test_read_rhs_pariLU0_MKL_FGMRES.cpp
	$(CC) $(CFLAGS) \
	-L${MKLROOT}/lib -I${MKLROOT}/include \
	control/constants.cpp control/magma_zmio.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp control/magma_zmatrixchar.cpp control/norms.cpp \
	control/magma_zmlumerge.cpp control/magma_zmscale.cpp \
	blas/zdiff.cpp blas/zdot.cpp blas/zgemv.cpp blas/zgemm.cpp \
	blas/zcsrilu0.cpp blas/zlunp.cpp blas/zspmm.cpp \
	src/parilu_v0_2.cpp \
	test_read_rhs_pariLU0_MKL_FGMRES.cpp \
	-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lstdc++ -lm -ldl \
	-o $@

test_solve_pariLU0_MKL_FGMRES: test_solve_pariLU0_MKL_FGMRES.cpp
	$(CC) $(CFLAGS) \
	-L${MKLROOT}/lib -I${MKLROOT}/include \
	control/constants.cpp control/magma_zmio.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp control/magma_zmatrixchar.cpp control/norms.cpp \
	control/magma_zmlumerge.cpp control/magma_zmscale.cpp \
	blas/zdiff.cpp blas/zdot.cpp blas/zgemv.cpp blas/zgemm.cpp \
	blas/zcsrilu0.cpp blas/zlunp.cpp blas/zspmm.cpp \
	src/parilu_v0_3.cpp \
	test_solve_pariLU0_MKL_FGMRES.cpp \
	-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lstdc++ -lm -ldl \
	-o $@

test_solve_pariLU0_MKL_FGMRES_knl: test_solve_pariLU0_MKL_FGMRES.cpp
	$(CC) $(CFLAGSKNL) \
	-L${MKLROOT}/lib -I${MKLROOT}/include \
	control/constants.cpp control/magma_zmio.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp control/magma_zmatrixchar.cpp control/norms.cpp \
	control/magma_zmlumerge.cpp control/magma_zmscale.cpp \
	blas/zdiff.cpp blas/zdot.cpp blas/zgemv.cpp blas/zgemm.cpp \
	blas/zcsrilu0.cpp blas/zlunp.cpp blas/zspmm.cpp \
	src/parilu_v0_3.cpp \
	test_solve_pariLU0_MKL_FGMRES.cpp \
	-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lstdc++ -lm -ldl \
	-o $@

test_trisolve: test_trisolve.cpp
	$(CC) $(CFLAGS) \
	-L${MKLROOT}/lib -I${MKLROOT}/include \
	control/constants.cpp control/magma_zmio.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp control/magma_zmatrixchar.cpp control/norms.cpp \
	control/magma_zmlumerge.cpp control/magma_zmscale.cpp \
	blas/zdiff.cpp blas/zdot.cpp blas/zgemv.cpp blas/zgemm.cpp \
	blas/zcsrilu0.cpp blas/zlunp.cpp blas/zspmm.cpp \
	src/parilu_v0_3.cpp \
	src/trisolve.cpp \
	test_trisolve.cpp \
	-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lstdc++ -lm -ldl \
	-o $@

test_dense_trisolve: test_dense_trisolve.cpp
	$(CC) $(CFLAGS) \
	-L${MKLROOT}/lib -I${MKLROOT}/include \
	control/constants.cpp control/magma_zmio.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp control/magma_zmatrixchar.cpp control/norms.cpp \
	control/magma_zmlumerge.cpp control/magma_zmscale.cpp \
	blas/zdiff.cpp blas/zdot.cpp blas/zgemv.cpp blas/zgemm.cpp \
	blas/zcsrilu0.cpp blas/zlunp.cpp blas/zspmm.cpp \
	src/parlu_v1_0.cpp \
	src/trisolve.cpp \
	test_dense_trisolve.cpp \
	-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lstdc++ -lm -ldl \
	-o $@

test_dense_inverse: test_dense_inverse.cpp
	$(CC) $(CFLAGS) \
	-L${MKLROOT}/lib -I${MKLROOT}/include \
	control/constants.cpp control/magma_zmio.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp control/magma_zmatrixchar.cpp control/norms.cpp \
	control/magma_zmlumerge.cpp control/magma_zmscale.cpp \
	blas/zdiff.cpp blas/zdot.cpp blas/zgemv.cpp blas/zgemm.cpp \
	blas/zcsrilu0.cpp blas/zlunp.cpp blas/zspmm.cpp \
	src/parlu_v1_0.cpp \
	src/trisolve.cpp \
	src/inverse.cpp \
	test_dense_inverse.cpp \
	-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lstdc++ -lm -ldl \
	-o $@

test_extract_diag: test_extract_diag.cpp
	$(CC) $(CFLAGS) \
	-L${MKLROOT}/lib -I${MKLROOT}/include \
	control/constants.cpp control/magma_zmio.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp control/magma_zmatrixchar.cpp control/norms.cpp \
	control/magma_zmlumerge.cpp control/magma_zmscale.cpp \
	control/extract_diag.cpp \
	blas/zdiff.cpp blas/zdot.cpp blas/zgemv.cpp blas/zgemm.cpp \
	blas/zcsrilu0.cpp blas/zlunp.cpp blas/zspmm.cpp \
	blas/zbcsrmultbcsr.cpp \
	test_extract_diag.cpp \
	src/parlu_v1_0.cpp \
	src/trisolve.cpp \
	src/inverse.cpp \
	-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lstdc++ -lm -ldl \
	-o $@

test_scaling: test_scaling.cpp
	$(CC) $(CFLAGS) \
	-L${MKLROOT}/lib -I${MKLROOT}/include \
	control/constants.cpp control/magma_zmio.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp control/magma_zmatrixchar.cpp control/norms.cpp \
	control/magma_zmlumerge.cpp control/magma_zmscale.cpp \
	control/extract_diag.cpp \
	blas/zdiff.cpp blas/zdot.cpp blas/zgemv.cpp blas/zgemm.cpp \
	blas/zcsrilu0.cpp blas/zlunp.cpp blas/zspmm.cpp \
	blas/zbcsrmultbcsr.cpp \
	test_scaling.cpp \
	src/parlu_v1_0.cpp \
	src/trisolve.cpp \
	src/inverse.cpp \
	-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lstdc++ -lm -ldl \
	-o $@

test_iLU_bcsr: test_iLU_bcsr.cpp
	$(CC) $(CFLAGS) \
	-L${MKLROOT}/lib -I${MKLROOT}/include \
	control/constants.cpp control/magma_zmio.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp control/magma_zmatrixchar.cpp control/norms.cpp \
	control/magma_zmlumerge.cpp control/magma_zmscale.cpp \
	control/extract_diag.cpp \
	blas/zdiff.cpp blas/zdot.cpp blas/zgemv.cpp blas/zgemm.cpp \
	blas/zcsrilu0.cpp blas/zlunp.cpp blas/zspmm.cpp \
	blas/zbcsrmultbcsr.cpp blas/zdomatadd.cpp \
	test_iLU_bcsr.cpp \
	src/parilu_v0_0.cpp \
	src/parlu_v1_0.cpp \
	src/parilu_v0_3_bcsr.cpp \
	src/trisolve.cpp \
	src/inverse.cpp \
	-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lstdc++ -lm -ldl \
	-o $@

test_solve_pariLU0_partrsv_MKL_FGMRES: test_solve_pariLU0_partrsv_MKL_FGMRES.cpp
	$(CC) $(CFLAGS) \
	-L${MKLROOT}/lib -I${MKLROOT}/include \
	control/constants.cpp control/magma_zmio.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp control/magma_zmatrixchar.cpp control/norms.cpp \
	control/magma_zmlumerge.cpp control/magma_zmscale.cpp \
	blas/zdiff.cpp blas/zdot.cpp blas/zgemv.cpp blas/zgemm.cpp \
	blas/zcsrilu0.cpp blas/zlunp.cpp blas/zspmm.cpp \
	src/parilu_v0_3.cpp \
	src/trisolve.cpp \
	test_solve_pariLU0_partrsv_MKL_FGMRES.cpp \
	-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lstdc++ -lm -ldl \
	-o $@

test_solve_GMRES_basic: test_solve_GMRES_basic.cpp
	$(CC) $(CFLAGS) \
	-L${MKLROOT}/lib -I${MKLROOT}/include \
	control/constants.cpp control/magma_zmio.cpp control/init.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp control/magma_zmatrixchar.cpp control/norms.cpp \
	control/magma_zmlumerge.cpp control/magma_zmscale.cpp \
	blas/zdiff.cpp blas/zdot.cpp blas/zgemv.cpp blas/zgemm.cpp \
	blas/zcsrilu0.cpp blas/zlunp.cpp blas/zaxpy.cpp blas/zspmv.cpp blas/zspmm.cpp \
	src/parilu_v0_3.cpp \
	src/trisolve.cpp \
	src/gmres_basic.cpp \
	test_solve_GMRES_basic.cpp \
	-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lstdc++ -lm -ldl \
	-o $@

test_solve_GMRES_reorth: test_solve_GMRES_reorth.cpp
	$(CC) $(CFLAGS) \
	-L${MKLROOT}/lib -I${MKLROOT}/include \
	control/constants.cpp control/magma_zmio.cpp control/init.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp control/magma_zmatrixchar.cpp control/norms.cpp \
	control/magma_zmlumerge.cpp control/magma_zmscale.cpp \
	blas/zdiff.cpp blas/zdot.cpp blas/zgemv.cpp blas/zgemm.cpp \
	blas/zcsrilu0.cpp blas/zlunp.cpp blas/zaxpy.cpp blas/zspmv.cpp blas/zspmm.cpp \
	src/parilu_v0_3.cpp \
	src/trisolve.cpp \
	src/gmres_reorth.cpp \
	test_solve_GMRES_reorth.cpp \
	-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lstdc++ -lm -ldl \
	-o $@

test_solve_GMRES_precond: test_solve_GMRES_precond.cpp
	$(CC) $(CFLAGS) \
	-L${MKLROOT}/lib -I${MKLROOT}/include \
	control/constants.cpp control/magma_zmio.cpp control/init.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp control/magma_zmatrixchar.cpp control/norms.cpp \
	control/magma_zmlumerge.cpp control/magma_zmscale.cpp \
	blas/zdiff.cpp blas/zdot.cpp blas/zgemv.cpp blas/zgemm.cpp \
	blas/zcsrilu0.cpp blas/zlunp.cpp blas/zaxpy.cpp blas/zspmv.cpp blas/zspmm.cpp \
	src/parilu_v0_3.cpp \
	src/trisolve.cpp \
	src/gmres_precond.cpp \
	test_solve_GMRES_precond.cpp \
	-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lstdc++ -lm -ldl \
	-o $@

test_solve_FGMRES: test_solve_FGMRES.cpp
	$(CC) $(CFLAGS) \
	-L${MKLROOT}/lib -I${MKLROOT}/include \
	control/constants.cpp control/magma_zmio.cpp control/init.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp control/magma_zmatrixchar.cpp control/norms.cpp \
	control/magma_zmlumerge.cpp control/magma_zmscale.cpp \
	blas/zdiff.cpp blas/zdot.cpp blas/zgemv.cpp blas/zgemm.cpp \
	blas/zcsrilu0.cpp blas/zlunp.cpp blas/zaxpy.cpp blas/zspmv.cpp blas/zspmm.cpp \
	src/parilu_v0_3.cpp \
	src/trisolve.cpp \
	src/orthogonality_error.cpp \
	src/fgmres.cpp \
	test_solve_FGMRES.cpp \
	-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lstdc++ -lm -ldl \
	-o $@

test_solve_GMRES_basic_orthog: test_solve_GMRES_basic_orthog.cpp
	$(CC) $(CFLAGS) \
	-L${MKLROOT}/lib -I${MKLROOT}/include \
	control/constants.cpp control/magma_zmio.cpp control/init.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp control/magma_zmatrixchar.cpp control/norms.cpp \
	control/magma_zmlumerge.cpp control/magma_zmscale.cpp \
	blas/zdiff.cpp blas/zdot.cpp blas/zgemv.cpp blas/zgemm.cpp \
	blas/zcsrilu0.cpp blas/zlunp.cpp blas/zaxpy.cpp blas/zspmv.cpp blas/zspmm.cpp \
	src/parilu_v0_3.cpp \
	src/trisolve.cpp \
	src/gram_schmidt.cpp \
	src/orthogonality_error.cpp \
	src/gmres_basic_orthog.cpp \
	test_solve_GMRES_basic_orthog.cpp \
	-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lstdc++ -lm -ldl \
	-o $@

test_householder: test_householder.cpp
	$(CC) $(CFLAGS) \
	-L${MKLROOT}/lib -I${MKLROOT}/include \
	control/constants.cpp control/magma_zmio.cpp control/init.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp control/magma_zmatrixchar.cpp control/norms.cpp \
	control/magma_zmlumerge.cpp control/magma_zmscale.cpp \
	blas/zdiff.cpp blas/zdot.cpp blas/zgemv.cpp blas/zgemm.cpp \
	blas/zcsrilu0.cpp blas/zlunp.cpp blas/zaxpy.cpp blas/zspmv.cpp blas/zspmm.cpp \
	blas/zdomatadd.cpp \
	src/parilu_v0_3.cpp \
	src/trisolve.cpp \
	src/orthogonality_error.cpp \
	src/householder.cpp \
	test_householder.cpp \
	-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lstdc++ -lm -ldl \
	-o $@

test_solve_GMRES_basic_householder_orthog: test_solve_GMRES_basic_householder_orthog.cpp
	$(CC) $(CFLAGS) \
	-L${MKLROOT}/lib -I${MKLROOT}/include \
	control/constants.cpp control/magma_zmio.cpp control/init.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp control/magma_zmatrixchar.cpp control/norms.cpp \
	control/magma_zmlumerge.cpp control/magma_zmscale.cpp \
	blas/zdiff.cpp blas/zdot.cpp blas/zgemv.cpp blas/zgemm.cpp \
	blas/zcsrilu0.cpp blas/zlunp.cpp blas/zaxpy.cpp blas/zspmv.cpp blas/zspmm.cpp \
	src/parilu_v0_3.cpp \
	src/trisolve.cpp \
	src/gram_schmidt.cpp \
	src/givens.cpp \
	src/orthogonality_error.cpp \
	src/householder.cpp \
	src/gmres_householder_orthog.cpp \
	test_solve_GMRES_basic_householder_orthog.cpp \
	-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lstdc++ -lm -ldl \
	-o $@
	
test_solve_GMRES_householder_precond: test_solve_GMRES_householder_precond.cpp
	$(CC) $(CFLAGS) \
	-L${MKLROOT}/lib -I${MKLROOT}/include \
	control/constants.cpp control/magma_zmio.cpp control/init.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp control/magma_zmatrixchar.cpp control/norms.cpp \
	control/magma_zmlumerge.cpp control/magma_zmscale.cpp \
	blas/zdiff.cpp blas/zdot.cpp blas/zgemv.cpp blas/zgemm.cpp \
	blas/zcsrilu0.cpp blas/zlunp.cpp blas/zaxpy.cpp blas/zspmv.cpp blas/zspmm.cpp \
	src/parilu_v0_3.cpp \
	src/trisolve.cpp \
	src/givens.cpp \
	src/orthogonality_error.cpp \
	src/gmres_householder_precond.cpp \
	test_solve_GMRES_householder_precond.cpp \
	-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lstdc++ -lm -ldl \
	-o $@

test_solve_FGMRES_householder: test_solve_FGMRES_householder.cpp
	$(CC) $(CFLAGS) \
	-L${MKLROOT}/lib -I${MKLROOT}/include \
	control/constants.cpp control/magma_zmio.cpp control/init.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp control/magma_zmatrixchar.cpp control/norms.cpp \
	control/magma_zmlumerge.cpp control/magma_zmscale.cpp \
	blas/zdiff.cpp blas/zdot.cpp blas/zgemv.cpp blas/zgemm.cpp \
	blas/zcsrilu0.cpp blas/zlunp.cpp blas/zaxpy.cpp blas/zspmv.cpp blas/zspmm.cpp \
	src/parilu_v0_3.cpp \
	src/trisolve.cpp \
	src/givens.cpp \
	src/orthogonality_error.cpp \
	src/fgmres_householder.cpp \
	test_solve_FGMRES_householder.cpp \
	-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lstdc++ -lm -ldl \
	-o $@

test_malloc: test_malloc.cpp
	$(CC) $(CFLAGS) \
	-L${MKLROOT}/lib -I${MKLROOT}/include \
	control/magma_zfree.cpp \
	test_malloc.cpp \
	-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lstdc++ -lm -ldl \
	-o $@

libgtest.a: gtest-all.o
	ar -rv libgtest.a gtest-all.o

libgmock.a: gtest-all.o gmock-all.o
	ar -rv libgmock.a gtest-all.o gmock-all.o

gtest-all.o: ${GTEST_DIR}/src/gtest-all.cc
	$(CC) -isystem ${GTEST_DIR}/include -I${GTEST_DIR} \
        -isystem ${GMOCK_DIR}/include -I${GMOCK_DIR} \
        -pthread -c ${GTEST_DIR}/src/gtest-all.cc

gmock-all.o: ${GMOCK_DIR}/src/gmock-all.cc
	$(CC) -isystem ${GTEST_DIR}/include -I${GTEST_DIR} \
        -isystem ${GMOCK_DIR}/include -I${GMOCK_DIR} \
        -pthread -c ${GMOCK_DIR}/src/gmock-all.cc

clean:
	rm exampleGoogleTest_01 exampleGoogleTest_02 test_matrix_io test_vector_io \
				test_matrix_ops test_spmatrix_ops test_blockspmatrix_ops \
				test_LU_ops test_LU test_LU_larnv test_MKL_LU \
				test_iLU_ops test_iLU test_MKL_iLU post_iLU test_MKL_iLU0_FGMRES \
				test_pariLU0_MKL_FGMRES test_read_pariLU0_MKL_FGMRES \
				test_read_rhs_pariLU0_MKL_FGMRES test_solve_pariLU0_MKL_FGMRES \

cleanall:
	rm *.o exampleGoogleTest_01 exampleGoogleTest_02 test_matrix_io \
				test_vector_io test_matrix_ops test_spmatrix_ops test_blockspmatrix_ops \
				test_LU_ops \
				test_LU test_MKL_LU test_iLU_ops test_iLU test_LU_larnv test_MKL_iLU post_iLU \
				test_MKL_iLU0_FGMRES test_pariLU0_MKL_FGMRES \
				test_read_pariLU0_MKL_FGMRES test_read_rhs_pariLU0_MKL_FGMRES \
				test_solve_pariLU0_MKL_FGMRES \
				libgtest.a libgmock.a
