/*
 * make -f makefile_beacon test_iLU -B
 */

#include <stdlib.h>
#include <stdio.h>
#include <libgen.h>
#include <string>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <vector>
#include <omp.h>


#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "include/mmio.h"
#include "include/sparse_types.h"

int
main(int argc, char * argv[])
{
  // begin with a square matrix A
  char * sparse_filename;
  char * sparse_basename;
  char sparse_name[256];
  char * output_dir;
  char output_basename[256];
  int tile = 100;

  if (argc < 4) {
    printf("Usage %s <matrix> <tile size> <output directory>\n", argv[0]);
    return 1;
  } else   {
    sparse_filename = argv[1];
    tile            = atoi(argv[2]);
    output_dir      = argv[3];
    sparse_basename = basename(sparse_filename);
    char * ext;
    ext = strrchr(sparse_basename, '.');
    strncpy(sparse_name, sparse_basename, int(ext - sparse_basename) );
    printf("File %s basename %s name %s \n",
      sparse_filename, sparse_basename, sparse_name);
    printf("tile size is %d \n", tile);
    printf("Output directory is %s\n", output_dir);
    strcpy(output_basename, output_dir);
    strcat(output_basename, sparse_name);
    printf("Output file base name is %s\n", output_basename);
  }
  // char sparse_filename[] = "testing/matrices/dRdQ_sm.mtx";
  // char sparse_filename[] = "testing/matrices/30p30n.mtx";
  // char sparse_filename[] = "testing/matrices/paper1_matrices/ani5_crop.mtx";
  data_d_matrix Asparse = { Magma_CSR };
  data_z_csr_mtx(&Asparse, sparse_filename);
  data_d_matrix A = { Magma_CSR };
  data_zmconvert(Asparse, &A, Magma_CSR, Magma_CSR);
  // data_d_matrix B = {Magma_CSR};
  // data_zmconvert( Asparse, &B, Magma_CSR, Magma_CSR );
  // data_zdisplay_dense( &A );
  // data_zmfree( &Asparse );

  dataType vmaxA = 0.0;
  int imaxA      = 0;
  int jmaxA      = 0;
  data_maxfabs_csr(A, &imaxA, &jmaxA, &vmaxA);
  printf("imaxA = %d\n", imaxA);
  printf("jmaxA = %d\n", jmaxA);
  printf("vmaxA = %e\n", vmaxA);
  printf("max(fabs(A)) = (%d,%d) %e\n", imaxA, jmaxA, vmaxA);

  A.storage_type = Magma_CSRU;
  data_zcheckupperlower(&A);

  data_zmfree(&Asparse);
  data_zmfree(&A);
  // data_zmfree( &B );
  // data_zmfree( &Amkl );
  // data_zmfree( &Lmkl );
  // data_zmfree( &Umkl );
  // testing::InitGoogleTest(&argc, argv);
  // return RUN_ALL_TESTS();
  return 0;
} // main
