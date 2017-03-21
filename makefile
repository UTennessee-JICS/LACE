# 
#
CC=g++-6
#
#
VPATH=blas control include src testing
CFLAGS=-std=c++11 -g -Wall -O0 -fopenmp 
LDFLAGS=-isystem ${GTEST_DIR}/include -pthread libgtest.a
LDFLAGS2=-isystem ${GTEST_DIR}/include -isystem ${GMOCK_DIR}/include -pthread libgmock.a
SOURCES=example_01.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=exampleGoogleTest_01

all: gtest-all.o gmock-all.o exampleGoogleTest_01 exampleGoogleTest_02 \
  test_matrix_io test_vector_io \
	test_matrix_ops test_spmatrix_ops \
	test_LU_ops test_LU test_MKL_LU \
	test_iLU_ops test_iLU test_MKL_iLU post_iLU \
	test_MKL_iLU0_FGMRES test_pariLU0_MKL_FGMRES \
	test_read_pariLU0_MKL_FGMRES

exampleGoogleTest_01: example_01.cpp libgtest.a
	$(CC) $(LDFLAGS) $(CFLAGS) example_01.cpp -o $@
	
exampleGoogleTest_02: example_02.cpp libgmock.a
	$(CC) $(LDFLAGS2) $(CFLAGS) example_02.cpp -o $@	
	
test_matrix_io: test_matrix_io.cpp libgmock.a
	$(CC) $(LDFLAGS2) $(CFLAGS) \
	-L${MKLROOT}/lib -I${MKLROOT}/include \
	-lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lstdc++ -lm \
	test_matrix_io.cpp control/magma_zmio.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp -o $@	
	
test_vector_io: test_vector_io.cpp libgmock.a
	$(CC) $(LDFLAGS2) $(CFLAGS) \
	-L${MKLROOT}/lib -I${MKLROOT}/include \
	-lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lstdc++ -lm \
	test_vector_io.cpp control/magma_zmio.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp -o $@		

test_matrix_ops: test_matrix_operations.cpp libgmock.a
	$(CC) $(LDFLAGS2) $(CFLAGS) \
	-L${MKLROOT}/lib -I${MKLROOT}/include \
	-lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lstdc++ -lm \
	test_matrix_operations.cpp control/constants.cpp control/magma_zmio.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp blas/zdot.cpp blas/zgemv.cpp blas/zgemm.cpp -o $@	
	
test_spmatrix_ops: test_spmatrix_operations.cpp libgmock.a
	$(CC) $(LDFLAGS2) $(CFLAGS) \
	-L${MKLROOT}/lib -I${MKLROOT}/include \
	-lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lstdc++ -lm \
	test_spmatrix_operations.cpp control/constants.cpp control/magma_zmio.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp control/sparse_sub.cpp control/sparse_tilepattern.cpp \
	blas/zdot.cpp blas/zgemv.cpp blas/zgemm.cpp blas/zspmm.cpp -o $@	

test_LU_ops: test_LU_operations.cpp libgmock.a
	$(CC) $(LDFLAGS2) $(CFLAGS) \
	-L${MKLROOT}/lib -I${MKLROOT}/include \
	-lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lstdc++ -lm \
	test_LU_operations.cpp control/constants.cpp control/magma_zmio.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp control/magma_zmatrixchar.cpp control/norms.cpp \
	control/sparse_sub.cpp control/sparse_tilepattern.cpp \
	blas/zdiff.cpp blas/zdot.cpp blas/zgemv.cpp blas/zgemm.cpp \
	blas/zcsrilu0.cpp blas/zlunp.cpp blas/zspmm.cpp \
	-o $@	
	
test_LU: test_LU.cpp libgmock.a
	$(CC) $(LDFLAGS2) $(CFLAGS) \
	-L${MKLROOT}/lib -I${MKLROOT}/include \
	-lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lstdc++ -lm \
	test_LU.cpp control/constants.cpp control/magma_zmio.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp control/magma_zmatrixchar.cpp control/norms.cpp \
	control/sparse_sub.cpp control/sparse_tilepattern.cpp \
	blas/zdiff.cpp blas/zdot.cpp blas/zgemv.cpp blas/zgemm.cpp \
	blas/zcsrilu0.cpp blas/zlunp.cpp blas/zspmm.cpp \
	src/parlu_v0_0.cpp src/parlu_v0_1.cpp \
	src/parlu_v1_0.cpp src/parlu_v1_1.cpp src/parlu_v1_2.cpp src/parlu_v1_3.cpp \
	src/parlu_v2_0.cpp src/parlu_v3_0.cpp -o $@		
	
test_MKL_LU: test_MKL_LU.cpp libgmock.a
	$(CC) $(LDFLAGS2) $(CFLAGS) \
	-L${MKLROOT}/lib -I${MKLROOT}/include \
	-lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lstdc++ -lm \
	test_MKL_LU.cpp control/constants.cpp control/magma_zmio.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp control/magma_zmatrixchar.cpp control/norms.cpp \
	control/sparse_sub.cpp control/sparse_tilepattern.cpp \
	blas/zdiff.cpp blas/zdot.cpp blas/zgemv.cpp blas/zgemm.cpp \
	blas/zcsrilu0.cpp blas/zlunp.cpp blas/zspmm.cpp \
	-o $@		
	
test_iLU_ops: test_iLU_operations.cpp libgmock.a
	$(CC) $(LDFLAGS2) $(CFLAGS) \
	-L${MKLROOT}/lib -I${MKLROOT}/include \
	-lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lstdc++ -lm \
	test_iLU_operations.cpp control/constants.cpp control/magma_zmio.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp control/magma_zmatrixchar.cpp control/norms.cpp \
	control/sparse_sub.cpp control/sparse_tilepattern.cpp \
	blas/zdiff.cpp blas/zdot.cpp blas/zgemv.cpp blas/zgemm.cpp \
	blas/zcsrilu0.cpp blas/zlunp.cpp blas/zspmm.cpp \
	src/parilu_v0_0.cpp src/parilu_v3_0.cpp -o $@		
	
test_iLU: test_iLU.cpp libgmock.a
	$(CC) $(LDFLAGS2) $(CFLAGS) \
	-L${MKLROOT}/lib -I${MKLROOT}/include \
	-lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lstdc++ -lm \
	test_iLU.cpp control/constants.cpp control/magma_zmio.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp control/magma_zmatrixchar.cpp control/norms.cpp \
	control/sparse_sub.cpp control/sparse_tilepattern.cpp \
	control/magma_zmscale.cpp \
	blas/zdiff.cpp blas/zdot.cpp blas/zgemv.cpp blas/zgemm.cpp \
	blas/zcsrilu0.cpp blas/zlunp.cpp blas/zspmm.cpp \
	src/parilu_v0_0.cpp src/parilu_v0_1.cpp src/parilu_v0_2.cpp \
	src/parilu_v3_0.cpp src/parilu_v3_1.cpp \
	src/parilu_v3_2.cpp src/parilu_v3_5.cpp src/parilu_v3_6.cpp \
	src/parilu_v3_7.cpp src/parilu_v3_8.cpp src/parilu_v3_9.cpp  \
	src/parilu_v4_0.cpp -o $@		
	
test_MKL_iLU: test_MKL_iLU.cpp libgmock.a
	$(CC) $(LDFLAGS2) $(CFLAGS) \
	-L${MKLROOT}/lib -I${MKLROOT}/include \
	-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lpthread -lstdc++ -lm \
	test_MKL_iLU.cpp control/constants.cpp control/magma_zmio.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp control/magma_zmatrixchar.cpp control/norms.cpp \
	control/sparse_sub.cpp control/sparse_tilepattern.cpp \
	control/magma_zmscale.cpp \
	blas/zdiff.cpp blas/zdot.cpp blas/zgemv.cpp blas/zgemm.cpp \
	blas/zcsrilu0.cpp blas/zlunp.cpp blas/zspmm.cpp \
	-o $@			
	
post_iLU: post_iLU.cpp libgmock.a
	$(CC) $(LDFLAGS2) $(CFLAGS) \
	-L${MKLROOT}/lib -I${MKLROOT}/include \
	-lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lstdc++ -lm \
	post_iLU.cpp control/constants.cpp control/magma_zmio.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp control/magma_zmatrixchar.cpp control/norms.cpp \
	blas/zdiff.cpp blas/zdot.cpp blas/zgemv.cpp blas/zgemm.cpp \
	blas/zcsrilu0.cpp blas/zlunp.cpp blas/zspmm.cpp \
	-o $@		

test_MKL_iLU0_FGMRES: test_MKL_iLU0_FGMRES.cpp
	$(CC) $(CFLAGS) \
	-L${MKLROOT}/lib -I${MKLROOT}/include \
	-lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lstdc++ -lm \
	test_MKL_iLU0_FGMRES.cpp -o $@		
	
test_pariLU0_MKL_FGMRES: test_pariLU0_MKL_FGMRES.cpp
	$(CC) $(CFLAGS) \
	-L${MKLROOT}/lib -I${MKLROOT}/include \
	-lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lstdc++ -lm \
	control/constants.cpp control/magma_zmio.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp control/magma_zmatrixchar.cpp control/norms.cpp \
	control/magma_zmlumerge.cpp \
	blas/zdiff.cpp blas/zdot.cpp blas/zgemv.cpp blas/zgemm.cpp \
	blas/zcsrilu0.cpp blas/zlunp.cpp blas/zspmm.cpp \
	src/parilu_v0_2.cpp \
	test_pariLU0_MKL_FGMRES.cpp -o $@	
	
test_read_pariLU0_MKL_FGMRES: test_read_pariLU0_MKL_FGMRES.cpp
	$(CC) $(CFLAGS) \
	-L${MKLROOT}/lib -I${MKLROOT}/include \
	-lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lstdc++ -lm \
	control/constants.cpp control/magma_zmio.cpp \
	control/mmio.cpp control/magma_zmconverter.cpp control/magma_zmtranspose.cpp \
	control/magma_zfree.cpp control/magma_zmatrixchar.cpp control/norms.cpp \
	control/magma_zmlumerge.cpp control/magma_zmscale.cpp \
	blas/zdiff.cpp blas/zdot.cpp blas/zgemv.cpp blas/zgemm.cpp \
	blas/zcsrilu0.cpp blas/zlunp.cpp blas/zspmm.cpp \
	src/parilu_v0_2.cpp \
	test_read_pariLU0_MKL_FGMRES.cpp -o $@	
	
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
				test_matrix_ops test_spmatrix_ops test_LU_ops test_LU test_MKL_LU \
				test_iLU_ops test_iLU test_MKL_iLU post_iLU test_MKL_iLU0_FGMRES \
				test_read_pariLU0_MKL_FGMRES \
				test_pariLU0_MKL_FGMRES
	
cleanall:
	rm *.o exampleGoogleTest_01 exampleGoogleTest_02 test_matrix_io \ 
				test_vector_io test_matrix_ops test_spmatrix_ops test_LU_ops \
				test_LU test_MKL_LU test_iLU_ops test_iLU test_MKL_iLU post_iLU \
				test_MKL_iLU0_FGMRES test_pariLU0_MKL_FGMRES \
				test_read_pariLU0_MKL_FGMRES \
				libgtest.a libgmock.a
				
