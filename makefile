#
#
# ACF compiler selection
CXX			 	 = icpc
CXXFLAGS	 = -std=c++11 -Wall -g -O3 -qopenmp -qopt-assume-safe-padding -qopt-report=5 -xAVX
#
#
# Mac compiler override to use OpenMP
#CXX				 = g++-6
#CXXFLAGS	 = -std=c++11 -Wall -g -O3 -fopenmp -pthread
#
#
#
CPPFLAGS  += -I$(GTEST_DIR)/include -I$(GMOCK_DIR)/include \
	-I$(MKLROOT)/include -I./include
LIBDIR		?= -L$(MKLROOT)/lib
LIB 			?= -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -liomp5 -lpthread -lstdc++ -lm -ldl
LIBS			 = $(LIBDIR) $(LIB)
#SOURCES=example_01.cpp
#OBJECTS=$(SOURCES:.cpp=.o)
#EXECUTABLE=exampleGoogleTest_01

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
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -lpthread $^ -o $@

exampleGoogleTest_02: example_02.cpp libgmock.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -lpthread $^ -o $@

test_matrix_io: test_matrix_io.cpp libgmock.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) \
	control/magma_zmio.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp \
	$(LIBS) \
	$^ -o $@

test_vector_io: test_vector_io.cpp libgmock.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) \
	control/magma_zmio.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp \
	$(LIBS) \
	$^ -o $@

test_matrix_ops: test_matrix_operations.cpp libgmock.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) \
	control/constants.cpp control/magma_zmio.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp blas/zdot.cpp blas/zgemv.cpp blas/zgemm.cpp \
	$(LIBS) \
	$^ -o $@

test_spmatrix_ops: test_spmatrix_operations.cpp libgmock.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) \
	control/constants.cpp control/magma_zmio.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp control/sparse_sub.cpp control/sparse_tilepattern.cpp \
	blas/zdot.cpp blas/zgemv.cpp blas/zgemm.cpp blas/zspmm.cpp \
	$(LIBS) \
	$^ -o $@

test_blockspmatrix_ops: test_blockspmatrix_operations.cpp libgmock.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) \
	control/constants.cpp control/magma_zmio.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp control/sparse_sub.cpp control/sparse_tilepattern.cpp \
	blas/zdot.cpp blas/zgemv.cpp blas/zgemm.cpp blas/zspmm.cpp \
	$(LIBS) \
	$^ -o $@

test_LU_ops: test_LU_operations.cpp libgmock.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) \
	control/constants.cpp control/magma_zmio.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp control/magma_zmatrixchar.cpp control/norms.cpp \
	control/sparse_sub.cpp control/sparse_tilepattern.cpp \
	blas/zdiff.cpp blas/zdot.cpp blas/zgemv.cpp blas/zgemm.cpp \
	blas/zcsrilu0.cpp blas/zlunp.cpp blas/zspmm.cpp \
	$(LIBS) \
	$^ -o $@

test_LU: test_LU.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) \
	test_LU.cpp control/constants.cpp control/magma_zmio.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp control/magma_zmatrixchar.cpp control/norms.cpp \
	control/sparse_sub.cpp control/sparse_tilepattern.cpp \
	blas/zdiff.cpp blas/zdot.cpp blas/zgemv.cpp blas/zgemm.cpp \
	blas/zcsrilu0.cpp blas/zlunp.cpp blas/zspmm.cpp \
	src/parlu_v0_0.cpp src/parlu_v0_1.cpp \
	src/parlu_v1_0.cpp src/parlu_v1_1.cpp src/parlu_v1_2.cpp src/parlu_v1_3.cpp \
	src/parlu_v2_0.cpp src/parlu_v3_0.cpp \
	$(LIBS) \
	-o $@

test_LU_larnv: test_LU_larnv.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) \
	test_LU_larnv.cpp control/constants.cpp control/magma_zmio.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp control/magma_zmatrixchar.cpp control/norms.cpp \
	control/sparse_sub.cpp control/sparse_tilepattern.cpp \
	blas/zdiff.cpp blas/zdot.cpp blas/zgemv.cpp blas/zgemm.cpp \
	blas/zcsrilu0.cpp blas/zlunp.cpp blas/zspmm.cpp \
	src/parlu_v0_0.cpp src/parlu_v0_1.cpp \
	src/parlu_v1_0.cpp src/parlu_v1_1.cpp src/parlu_v1_2.cpp src/parlu_v1_2c.cpp src/parlu_v1_3.cpp \
	src/parlu_v2_0.cpp src/parlu_v2_1.cpp src/parlu_v3_0.cpp src/parlu_v3_1.cpp \
	$(LIBS) \
	-o $@

test_MKL_LU: test_MKL_LU.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) \
	test_MKL_LU.cpp control/constants.cpp control/magma_zmio.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp control/magma_zmatrixchar.cpp control/norms.cpp \
	control/sparse_sub.cpp control/sparse_tilepattern.cpp \
	blas/zdiff.cpp blas/zdot.cpp blas/zgemv.cpp blas/zgemm.cpp \
	blas/zcsrilu0.cpp blas/zlunp.cpp blas/zspmm.cpp \
	$(LIBS) \
	-o $@

test_MKL_LU_larnv: test_MKL_LU_larnv.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) \
	test_MKL_LU_larnv.cpp control/constants.cpp control/magma_zmio.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp control/magma_zmatrixchar.cpp control/norms.cpp \
	control/sparse_sub.cpp control/sparse_tilepattern.cpp \
	blas/zdiff.cpp blas/zdot.cpp blas/zgemv.cpp blas/zgemm.cpp \
	blas/zcsrilu0.cpp blas/zlunp.cpp blas/zspmm.cpp \
	$(LIBS) \
	-o $@

test_iLU_ops: test_iLU_operations.cpp libgmock.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) \
	control/constants.cpp control/magma_zmio.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp control/magma_zmatrixchar.cpp control/norms.cpp \
	control/sparse_sub.cpp control/sparse_tilepattern.cpp \
	blas/zdiff.cpp blas/zdot.cpp blas/zgemv.cpp blas/zgemm.cpp \
	blas/zcsrilu0.cpp blas/zlunp.cpp blas/zspmm.cpp \
	src/parilu_v0_0.cpp src/parilu_v3_0.cpp \
	$(LIBS) \
	$^ -o $@

test_iLU: test_iLU.cpp libgmock.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) \
	control/constants.cpp control/magma_zmio.cpp \
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
	src/parilu_v4_0.cpp \
	$(LIBS) \
	$^ -o $@

test_MKL_iLU: test_MKL_iLU.cpp libgmock.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) \
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
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) \
	post_iLU.cpp control/constants.cpp control/magma_zmio.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp control/magma_zmatrixchar.cpp control/norms.cpp \
	blas/zdiff.cpp blas/zdot.cpp blas/zgemv.cpp blas/zgemm.cpp \
	blas/zcsrilu0.cpp blas/zlunp.cpp blas/zspmm.cpp \
	$(LIBS) \
	-o $@

test_MKL_iLU0_FGMRES: test_MKL_iLU0_FGMRES.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) \
	test_MKL_iLU0_FGMRES.cpp \
	$(LIBS) \
	-o $@

test_pariLU0_MKL_FGMRES: test_pariLU0_MKL_FGMRES.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) \
	control/constants.cpp control/magma_zmio.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp control/magma_zmatrixchar.cpp control/norms.cpp \
	control/magma_zmlumerge.cpp \
	blas/zdiff.cpp blas/zdot.cpp blas/zgemv.cpp blas/zgemm.cpp \
	blas/zcsrilu0.cpp blas/zlunp.cpp blas/zspmm.cpp \
	src/parilu_v0_2.cpp \
	$(LIBS) \
	test_pariLU0_MKL_FGMRES.cpp -o $@

test_read_pariLU0_MKL_FGMRES: test_read_pariLU0_MKL_FGMRES.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) \
	control/constants.cpp control/magma_zmio.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp control/magma_zmatrixchar.cpp control/norms.cpp \
	control/magma_zmlumerge.cpp control/magma_zmscale.cpp \
	blas/zdiff.cpp blas/zdot.cpp blas/zgemv.cpp blas/zgemm.cpp \
	blas/zcsrilu0.cpp blas/zlunp.cpp blas/zspmm.cpp \
	src/parilu_v0_2.cpp \
	$(LIBS) \
	test_read_pariLU0_MKL_FGMRES.cpp -o $@

test_read_rhs_pariLU0_MKL_FGMRES: test_read_rhs_pariLU0_MKL_FGMRES.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) \
	control/constants.cpp control/magma_zmio.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp control/magma_zmatrixchar.cpp control/norms.cpp \
	control/magma_zmlumerge.cpp control/magma_zmscale.cpp \
	blas/zdiff.cpp blas/zdot.cpp blas/zgemv.cpp blas/zgemm.cpp \
	blas/zcsrilu0.cpp blas/zlunp.cpp blas/zspmm.cpp \
	src/parilu_v0_2.cpp \
	test_read_rhs_pariLU0_MKL_FGMRES.cpp \
	$(LIBS) \
	-o $@

test_solve_pariLU0_MKL_FGMRES: test_solve_pariLU0_MKL_FGMRES.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) \
	control/constants.cpp control/magma_zmio.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp control/magma_zmatrixchar.cpp control/norms.cpp \
	control/magma_zmlumerge.cpp control/magma_zmscale.cpp \
	blas/zdiff.cpp blas/zdot.cpp blas/zgemv.cpp blas/zgemm.cpp \
	blas/zcsrilu0.cpp blas/zlunp.cpp blas/zspmm.cpp \
	src/parilu_v0_3.cpp \
	test_solve_pariLU0_MKL_FGMRES.cpp \
	$(LIBS) \
	-o $@

test_solve_pariLU0_MKL_FGMRES_knl: test_solve_pariLU0_MKL_FGMRES.cpp
	$(CXX) $(CXXFLAGSKNL) \
	control/constants.cpp control/magma_zmio.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp control/magma_zmatrixchar.cpp control/norms.cpp \
	control/magma_zmlumerge.cpp control/magma_zmscale.cpp \
	blas/zdiff.cpp blas/zdot.cpp blas/zgemv.cpp blas/zgemm.cpp \
	blas/zcsrilu0.cpp blas/zlunp.cpp blas/zspmm.cpp \
	src/parilu_v0_3.cpp \
	test_solve_pariLU0_MKL_FGMRES.cpp \
	$(LIBS) \
	-o $@

test_trisolve: test_trisolve.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) \
	control/constants.cpp control/magma_zmio.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp control/magma_zmatrixchar.cpp control/norms.cpp \
	control/magma_zmlumerge.cpp control/magma_zmscale.cpp \
	blas/zdiff.cpp blas/zdot.cpp blas/zgemv.cpp blas/zgemm.cpp \
	blas/zcsrilu0.cpp blas/zlunp.cpp blas/zspmm.cpp \
	src/parilu_v0_3.cpp \
	src/trisolve.cpp \
	test_trisolve.cpp \
	$(LIBS) \
	-o $@

test_dense_trisolve: test_dense_trisolve.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) \
	control/constants.cpp control/magma_zmio.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp control/magma_zmatrixchar.cpp control/norms.cpp \
	control/magma_zmlumerge.cpp control/magma_zmscale.cpp \
	blas/zdiff.cpp blas/zdot.cpp blas/zgemv.cpp blas/zgemm.cpp \
	blas/zcsrilu0.cpp blas/zlunp.cpp blas/zspmm.cpp \
	src/parlu_v1_0.cpp \
	src/trisolve.cpp \
	test_dense_trisolve.cpp \
	$(LIBS) \
	-o $@

test_dense_inverse: test_dense_inverse.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) \
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
	$(LIBS) \
	-o $@

test_extract_diag: test_extract_diag.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) \
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
	$(LIBS) \
	-o $@

test_scaling: test_scaling.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) \
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
	$(LIBS) \
	-o $@

test_iLU_bcsr: test_iLU_bcsr.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) \
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
	$(LIBS) \
	-o $@

test_solve_pariLU0_partrsv_MKL_FGMRES: test_solve_pariLU0_partrsv_MKL_FGMRES.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) \
	control/constants.cpp control/magma_zmio.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp control/magma_zmatrixchar.cpp control/norms.cpp \
	control/magma_zmlumerge.cpp control/magma_zmscale.cpp \
	blas/zdiff.cpp blas/zdot.cpp blas/zgemv.cpp blas/zgemm.cpp \
	blas/zcsrilu0.cpp blas/zlunp.cpp blas/zspmm.cpp \
	src/parilu_v0_3.cpp \
	src/trisolve.cpp \
	test_solve_pariLU0_partrsv_MKL_FGMRES.cpp \
	$(LIBS) \
	-o $@

test_solve_GMRES_basic: test_solve_GMRES_basic.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) \
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
	$(LIBS) \
	-o $@

test_solve_GMRES_reorth: test_solve_GMRES_reorth.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) \
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
	$(LIBS) \
	-o $@

test_solve_GMRES_precond: test_solve_GMRES_precond.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) \
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
	$(LIBS) \
	-o $@

test_solve_FGMRES: test_solve_FGMRES.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) \
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
	$(LIBS) \
	-o $@

test_solve_GMRES_basic_orthog: test_solve_GMRES_basic_orthog.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) \
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
	$(LIBS) \
	-o $@

test_householder: test_householder.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) \
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
	$(LIBS) \
	-o $@

test_solve_GMRES_basic_householder_orthog: test_solve_GMRES_basic_householder_orthog.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) \
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
	$(LIBS) \
	-o $@

test_solve_GMRES_householder_precond: test_solve_GMRES_householder_precond.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) \
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
	$(LIBS) \
	-o $@

test_solve_FGMRES_householder: test_solve_FGMRES_Householder.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) \
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
	test_solve_FGMRES_Householder.cpp \
	$(LIBS) \
	-o $@

test_solve_FGMRES_householder_opt1: test_solve_FGMRES_Householder.cpp
		$(CXX) $(CPPFLAGS) $(CXXFLAGS) \
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
		src/fgmres_householder_opt1.cpp \
		test_solve_FGMRES_Householder.cpp \
		$(LIBS) \
		-o $@

test_solve_FGMRES_householder_opt2: test_solve_FGMRES_Householder.cpp
		$(CXX) $(CPPFLAGS) $(CXXFLAGS) \
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
		src/fgmres_householder_opt2.cpp \
		test_solve_FGMRES_Householder.cpp \
		$(LIBS) \
		-o $@

test_solve_FGMRES_householder_opt3: test_solve_FGMRES_Householder.cpp
		$(CXX) $(CPPFLAGS) $(CXXFLAGS) \
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
		src/fgmres_householder_opt3.cpp \
		test_solve_FGMRES_Householder.cpp \
		$(LIBS) \
		-o $@

test_solve_FGMRES_householder_opt4: test_solve_FGMRES_Householder.cpp
		$(CXX) $(CPPFLAGS) $(CXXFLAGS) \
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
		src/fgmres_householder_opt4.cpp \
		test_solve_FGMRES_Householder.cpp \
		$(LIBS) \
		-o $@

test_solve_FGMRES_precond: test_solve_FGMRES.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) \
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
	$(LIBS) \
	-o $@

test_solve_FGMRES_householder_precond: test_solve_FGMRES_Householder.cpp
		$(CXX) $(CPPFLAGS) $(CXXFLAGS) \
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
		src/fgmres_householder_precond.cpp \
		test_solve_FGMRES_Householder.cpp \
		$(LIBS) \
		-o $@

test_solve_FGMRES_householder_restart: test_solve_FGMRES_Householder_restart.cpp
		$(CXX) $(CPPFLAGS) $(CXXFLAGS) \
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
		src/fgmres_householder_restart.cpp \
		test_solve_FGMRES_Householder_restart.cpp \
		$(LIBS) \
		-o $@

test_solve_FGMRES_restart: test_solve_FGMRES_restart.cpp
		$(CXX) $(CPPFLAGS) $(CXXFLAGS) \
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
		src/fgmres_restart.cpp \
		test_solve_FGMRES_restart.cpp \
		$(LIBS) \
		-o $@

test_solve_FGMRES_deflated_restart: test_solve_FGMRES_deflated_restart.cpp
		$(CXX) $(CPPFLAGS) $(CXXFLAGS) \
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
		src/fgmres_deflated_restart.cpp \
		test_solve_FGMRES_deflated_restart.cpp \
		$(LIBS) \
		-o $@

test_malloc: test_malloc.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) \
	control/magma_zfree.cpp \
	test_malloc.cpp \
	$(LIBS) \
	-o $@

test_orthogError: test_orthogError.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) \
	control/constants.cpp control/magma_zmio.cpp control/init.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp control/magma_zmatrixchar.cpp control/norms.cpp \
	blas/zdiff.cpp blas/zdot.cpp blas/zgemv.cpp blas/zgemm.cpp \
	blas/zcsrilu0.cpp blas/zlunp.cpp blas/zaxpy.cpp blas/zspmv.cpp blas/zspmm.cpp \
	src/orthogonality_error.cpp \
	test_orthogError.cpp \
	$(LIBS) \
	-o $@


GTEST_HEADERS = $(GTEST_DIR)/include/gtest/*.h \
	$(GTEST_DIR)/include/gtest/internal/*.h

GMOCK_HEADERS = $(GMOCK_DIR)/include/gmock/*.h \
	$(GMOCK_DIR)/include/gmock/internal/*.h \
	$(GTEST_HEADERS)

GTEST_SRCS_ = $(GTEST_DIR)/src/*.cc $(GTEST_DIR)/src/*.h $(GTEST_HEADERS)
GMOCK_SRCS_ = $(GMOCK_DIR)/src/*.cc $(GMOCK_HEADERS)

gtest-all.o : $(GTEST_SRCS_)
	$(CXX) $(CPPFLAGS) -I$(GTEST_DIR) $(CXXFLAGS) -c \
		$(GTEST_DIR)/src/gtest-all.cc

gtest_main.o : $(GTEST_SRCS_)
	$(CXX) $(CPPFLAGS) -I$(GTEST_DIR) $(CXXFLAGS) -c \
		$(GTEST_DIR)/src/gtest_main.cc

libgtest.a : gtest-all.o
	$(AR) $(ARFLAGS) $@ $^

gtest_main.a : gtest-all.o gtest_main.o
	$(AR) $(ARFLAGS) $@ $^

gmock-all.o : $(GMOCK_SRCS_)
	$(CXX) $(CPPFLAGS) -I$(GTEST_DIR) -I$(GMOCK_DIR) $(CXXFLAGS) \
		-c $(GMOCK_DIR)/src/gmock-all.cc

gmock_main.o : $(GMOCK_SRCS_)
	$(CXX) $(CPPFLAGS) -I$(GTEST_DIR) -I$(GMOCK_DIR) $(CXXFLAGS) \
		-c $(GMOCK_DIR)/src/gmock_main.cc

libgmock.a : gmock-all.o gtest-all.o
	$(AR) $(ARFLAGS) $@ $^

gmock_main.a : gmock-all.o gtest-all.o gmock_main.o
	$(AR) $(ARFLAGS) $@ $^

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
