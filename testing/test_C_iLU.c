#include <stdio.h>

#include "mmio.h"
//#include "sparse.h"
//#include "container_tests.h"

int
main(int argc, char * argv[])
{
  printf("C program using LACE.\n\n");
  printf("argc = %d\n", argc);
    for (int i = 0; i < argc; ++i) {
      printf("agv[%d] = %s\n", i, argv[i]);
    }
    fflush(stdout);

    char default_matrix[] = "matrices/Trefethen_20.mtx";
    char * matrix_name    = NULL;
    int* tile_size    = (int*) malloc(sizeof(int));
    (*tile_size) = 8;

    // parse command line arguments
    if (argc > 1) {
      int count = 1;
      while (count < argc) {
        if ( (strcmp(argv[count], "--matrix") == 0) &&
          count + 1 < argc)
        {
          matrix_name = argv[count + 1];
          count       = count + 2;
        } else if ( (strcmp(argv[count], "--tile") == 0) &&
          count + 1 < argc)
        {
          (*tile_size) = atoi(argv[count + 1]);
          count        = count + 2;
        } else   {
          count++;
        }
      }
    }

    // load A matrix
    if (matrix_name == NULL) {
      matrix_name = default_matrix;
    }
    printf("A will be read from %s\n", matrix_name);
    data_d_matrix A = { Magma_CSR };
    CHECK(data_z_csr_mtx((&A), matrix_name) );

    data_zprint_csr(A);
    data_zmfree(&A);

}
