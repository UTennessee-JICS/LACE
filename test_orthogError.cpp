#include <stdlib.h>
#include <stdio.h>
#include "include/sparse.h"

void someFunction (int **data) {
  //*data = (int*) malloc (sizeof (int));
  LACE_CALLOC( *data, 1 );
}

void useData (int *data) {
  printf ("%p\t%d\n", data, (*data));
}

void someMatrixFunction (data_d_matrix *A) {
  //A->val = (dataType*) malloc(sizeof(dataType));
  LACE_CALLOC( A->val, 1 );
}

void useMatrixData (data_d_matrix *A) {
  printf ("%p\t%e\n", A->val, A->val[0]);
}


int main () {

  dataType zero = 0.0;
  dataType one = 1.0;
  dataType negone = -1.0;
  int n = 10;
  int searchMax = 6;

  data_d_matrix A = {Magma_DENSE};
  data_zvinit( &A, n, searchMax, zero );

  for (int i=0; i<searchMax; ++i) {
    A.val[idx(i,i,A.ld)] = one;
  }
  A.val[idx(0,1,A.ld)] = -0.5;
  A.val[idx(1,1,A.ld)] = negone;
  A.val[idx(2,3,A.ld)] = 3.0;
  A.val[idx(4,3,A.ld)] = -0.1;

  dataType ortherr = 0.0;
  int imax = 0;

  for (int i=0; i<searchMax; ++i) {
    data_orthogonality_error( &A, &ortherr, &imax, (i+1) );
    printf("ortherr(%d) \t\t= %.16e;\n", i+1, ortherr);
    data_orthogonality_error_incremental( &A, &ortherr, &imax, (i+1) );
    printf("ortherr_inc(%d) \t\t= %.16e;\n", i+1, ortherr);
  }

  data_zmfree( &A );

  return 0;
}
