
#include <stdio.h>
#include "../include/sparse.h"


extern "C"
void
data_sparse_subvector( int sub_mbegin, int sub_nbegin,
  data_d_matrix* A, dataType* subvector ) {

  //printf("sub_mbegin=%d sub_mbegin+sub_m=%d\n", sub_mbegin, sub_mbegin+sub_m );
  //printf("sub_nbegin=%d sub_nbegin+sub_n=%d\n", sub_nbegin, sub_nbegin+sub_n );
  //for(int i=sub_mbegin; i < sub_mbegin+sub_m; i++ ) {
    //printf("row=%d A->row[i]=%d A->col[A->row[i]]=%d A->col[A->row[i+1]-1]=%d \n", i, A->row[i], A->col[A->row[i]], A->col[A->row[i+1]-1] );
    //printf("A->col[A->row[i]] >= sub_nbegin %d  A->col[A->row[i]] < sub_nbegin+sub_n %d\n", A->col[A->row[i]] >= sub_nbegin, A->col[A->row[i]] < sub_nbegin+sub_n );
    for (int j=A->row[sub_mbegin]; j < A->row[sub_mbegin+1]; j++ ) {
      //printf("\tj=%d A->col[j]=%d A->val[j]=%e\n", j, A->col[j], A->val[j] );
      //printf("\tA->col[j] >= sub_nbegin = %d  A->col[j] < sub_nbegin+sub_n = %d \n", A->col[j] >= sub_nbegin, A->col[j] < sub_nbegin+sub_n );
      //if ( A->col[j] >= sub_nbegin && A->col[j] < sub_nbegin+sub_n ) {
        //printf("adding A->val[j]=%e from (%d, %d) to [%d]\n",
        //  A->val[j], sub_mbegin, A->col[j], A->col[j]-sub_nbegin );
        subvector[ A->col[j]-sub_nbegin ] = A->val[ j ]; // rowmajor
      //}
    }
  //}

}

extern "C"
void
data_sparse_subvector_lowerupper( int sub_mbegin, int sub_nbegin,
  data_d_matrix* A, dataType* subvector ) {

  //printf("sub_mbegin=%d sub_mbegin+sub_m=%d\n", sub_mbegin, sub_mbegin+sub_m );
  //printf("sub_nbegin=%d sub_nbegin+sub_n=%d\n", sub_nbegin, sub_nbegin+sub_n );
  //for(int i=sub_mbegin; i < sub_mbegin+sub_m; i++ ) {
    //printf("row=%d A->row[i]=%d A->col[A->row[i]]=%d A->col[A->row[i+1]-1]=%d \n", i, A->row[i], A->col[A->row[i]], A->col[A->row[i+1]-1] );
    //printf("A->col[A->row[i]] >= sub_nbegin %d  A->col[A->row[i]] < sub_nbegin+sub_n %d\n", A->col[A->row[i]] >= sub_nbegin, A->col[A->row[i]] < sub_nbegin+sub_n );
    for (int j=A->row[sub_mbegin]; j < A->row[sub_mbegin+1]; j++ ) {
      //printf("\tj=%d A->col[j]=%d A->val[j]=%e\n", j, A->col[j], A->val[j] );
      //printf("\tA->col[j] >= sub_nbegin = %d  A->col[j] < sub_nbegin+sub_n = %d \n", A->col[j] >= sub_nbegin, A->col[j] < sub_nbegin+sub_n );
      if (A->col[j] != sub_mbegin) {
        //printf("adding A->val[j]=%e from (%d, %d) to [%d, %d]\n",
        //  A->val[j], i, A->col[j], i-sub_mbegin, A->col[j]-sub_nbegin );
        subvector[ A->col[j]-sub_nbegin ] = A->val[ j ]; // rowmajor
      }
    }
  //}

}

extern "C"
void
data_sparse_subdense( int sub_m, int sub_n, int sub_mbegin, int sub_nbegin,
  data_d_matrix* A, dataType* subdense ) {

  //printf("sub_mbegin=%d sub_mbegin+sub_m=%d\n", sub_mbegin, sub_mbegin+sub_m );
  //printf("sub_nbegin=%d sub_nbegin+sub_n=%d\n", sub_nbegin, sub_nbegin+sub_n );
  for(int i=sub_mbegin; i < sub_mbegin+sub_m; i++ ) {
    //printf("row=%d A->row[i]=%d A->col[A->row[i]]=%d A->col[A->row[i+1]-1]=%d \n", i, A->row[i], A->col[A->row[i]], A->col[A->row[i+1]-1] );
    //printf("A->col[A->row[i]] >= sub_nbegin %d  A->col[A->row[i]] < sub_nbegin+sub_n %d\n", A->col[A->row[i]] >= sub_nbegin, A->col[A->row[i]] < sub_nbegin+sub_n );
    for (int j=A->row[i]; j < A->row[i+1]; j++ ) {
      //printf("\tj=%d A->col[j]=%d A->val[j]=%e\n", j, A->col[j], A->val[j] );
      //printf("\tA->col[j] >= sub_nbegin = %d  A->col[j] < sub_nbegin+sub_n = %d \n", A->col[j] >= sub_nbegin, A->col[j] < sub_nbegin+sub_n );
      if ( A->col[j] >= sub_nbegin && A->col[j] < sub_nbegin+sub_n ) {
        //printf("adding A->val[j]=%e from (%d, %d) to [%d, %d]\n",
        //  A->val[j], i, A->col[j], i-sub_mbegin, A->col[j]-sub_nbegin );
        subdense[ (i-sub_mbegin) * sub_n + A->col[j]-sub_nbegin ] = A->val[ j ]; // rowmajor
      }
    }
  }

}

extern "C"
void
data_sparse_subdense_lowerupper( int sub_m, int sub_n, int sub_mbegin, int sub_nbegin,
  data_d_matrix* A, dataType* subdense ) {

  //printf("sub_mbegin=%d sub_mbegin+sub_m=%d\n", sub_mbegin, sub_mbegin+sub_m );
  //printf("sub_nbegin=%d sub_nbegin+sub_n=%d\n", sub_nbegin, sub_nbegin+sub_n );
  for(int i=sub_mbegin; i < sub_mbegin+sub_m; i++ ) {
    //printf("row=%d A->row[i]=%d A->col[A->row[i]]=%d A->col[A->row[i+1]-1]=%d \n", i, A->row[i], A->col[A->row[i]], A->col[A->row[i+1]-1] );
    //printf("A->col[A->row[i]] >= sub_nbegin %d  A->col[A->row[i]] < sub_nbegin+sub_n %d\n", A->col[A->row[i]] >= sub_nbegin, A->col[A->row[i]] < sub_nbegin+sub_n );
    for (int j=A->row[i]; j < A->row[i+1]; j++ ) {
      //printf("\tj=%d A->col[j]=%d A->val[j]=%e\n", j, A->col[j], A->val[j] );
      //printf("\tA->col[j] >= sub_nbegin = %d  A->col[j] < sub_nbegin+sub_n = %d \n", A->col[j] >= sub_nbegin, A->col[j] < sub_nbegin+sub_n );
      if ( A->col[j] >= sub_nbegin && A->col[j] < sub_nbegin+sub_n && A->col[j] != i ) {
        //printf("adding A->val[j]=%e from (%d, %d) to [%d, %d]\n",
        //  A->val[j], i, A->col[j], i-sub_mbegin, A->col[j]-sub_nbegin );
        subdense[ (i-sub_mbegin) * sub_n + A->col[j]-sub_nbegin ] = A->val[ j ]; // rowmajor
      }
    }
  }

}

extern "C"
void
data_sparse_subsparse(
  int sub_m,
  int sub_n,
  int sub_mbegin,
  int sub_nbegin,
  data_d_matrix* A,
  int* rowtmp,
  int* rowindxtmp,
  int* valindxtmp,
  int* coltmp,
  int* nnztmp ) {

  (*nnztmp) = 0;
  //printf("sub_mbegin=%d sub_mbegin+sub_m=%d\n", sub_mbegin, sub_mbegin+sub_m );
  //printf("sub_nbegin=%d sub_nbegin+sub_n=%d\n", sub_nbegin, sub_nbegin+sub_n );
  for(int i=sub_mbegin; i < sub_mbegin+sub_m; i++ ) {
    //printf("row=%d A->row[i]=%d A->col[A->row[i]]=%d A->col[A->row[i+1]-1]=%d \n", i, A->row[i], A->col[A->row[i]], A->col[A->row[i+1]-1] );
    //printf("A->col[A->row[i]] >= sub_nbegin %d  A->col[A->row[i]] < sub_nbegin+sub_n %d\n", A->col[A->row[i]] >= sub_nbegin, A->col[A->row[i]] < sub_nbegin+sub_n );
    rowtmp[i-sub_mbegin] = (*nnztmp);
    for (int j=A->row[i]; j < A->row[i+1]; j++ ) {
      //printf("\tj=%d A->col[j]=%d A->val[j]=%e\n", j, A->col[j], A->val[j] );
      //printf("\tA->col[j] >= sub_nbegin = %d  A->col[j] < sub_nbegin+sub_n = %d \n", A->col[j] >= sub_nbegin, A->col[j] < sub_nbegin+sub_n );
      if ( A->col[j] >= sub_nbegin && A->col[j] < sub_nbegin+sub_n ) {
        //printf("adding A->val[j]=%e from (%d, %d) to [%d, %d]\n",
        //  A->val[j], i, A->col[j], i-sub_mbegin, A->col[j]-sub_nbegin );
        rowindxtmp[(*nnztmp)] = i;
        coltmp[(*nnztmp)] = A->col[j];
        valindxtmp[(*nnztmp)] = j;
        (*nnztmp)++;
      }
    }
  }
  rowtmp[sub_m] = (*nnztmp);
  //printf("\nDONE forming temp row and col arrays.\n");
  //printf("csr tmp :\n");
  //for (int k=0; k<sub_m+1; k++) {
  //  printf("\t%d\n", rowtmp[k] );
  //}
  //for (int k=0; k<(*nnztmp); k++) {
  //  printf("\t%d %d %d %e\n", rowindxtmp[k], coltmp[k], valindxtmp[k], A->val[ valindxtmp[k] ] );
  //}

}

extern "C"
void
data_sparse_subsparse_lowerupper(
  int sub_m,
  int sub_n,
  int sub_mbegin,
  int sub_nbegin,
  data_d_matrix* A,
  int* rowtmp,
  int* rowindxtmp,
  int* valindxtmp,
  int* coltmp,
  int* nnztmp ) {

  (*nnztmp) = 0;
  //printf("sub_mbegin=%d sub_mbegin+sub_m=%d\n", sub_mbegin, sub_mbegin+sub_m );
  //printf("sub_nbegin=%d sub_nbegin+sub_n=%d\n", sub_nbegin, sub_nbegin+sub_n );
  for(int i=sub_mbegin; i < sub_mbegin+sub_m; i++ ) {
    //printf("row=%d A->row[i]=%d A->col[A->row[i]]=%d A->col[A->row[i+1]-1]=%d \n", i, A->row[i], A->col[A->row[i]], A->col[A->row[i+1]-1] );
    //printf("A->col[A->row[i]] >= sub_nbegin %d  A->col[A->row[i]] < sub_nbegin+sub_n %d\n", A->col[A->row[i]] >= sub_nbegin, A->col[A->row[i]] < sub_nbegin+sub_n );
    rowtmp[i-sub_mbegin] = (*nnztmp);
    for (int j=A->row[i]; j < A->row[i+1]; j++ ) {
      //printf("\tj=%d A->col[j]=%d A->val[j]=%e\n", j, A->col[j], A->val[j] );
      //printf("\tA->col[j] >= sub_nbegin = %d  A->col[j] < sub_nbegin+sub_n = %d \n", A->col[j] >= sub_nbegin, A->col[j] < sub_nbegin+sub_n );
      if ( A->col[j] >= sub_nbegin && A->col[j] < sub_nbegin+sub_n && A->col[j] != i ) {
        //printf("adding A->val[j]=%e from (%d, %d) to [%d, %d]\n",
        //  A->val[j], i, A->col[j], i-sub_mbegin, A->col[j]-sub_nbegin );
        rowindxtmp[(*nnztmp)] = i;
        coltmp[(*nnztmp)] = A->col[j];
        valindxtmp[(*nnztmp)] = j;
        (*nnztmp)++;
      }
    }
  }
  rowtmp[sub_m] = (*nnztmp);
  //printf("\nDONE forming temp row and col arrays.\n");
  //printf("csr tmp :\n");
  //for (int k=0; k<sub_m+1; k++) {
  //  printf("\t%d\n", rowtmp[k] );
  //}
  //for (int k=0; k<(*nnztmp); k++) {
  //  printf("\t%d %d %d %e\n", rowindxtmp[k], coltmp[k], valindxtmp[k], A->val[ valindxtmp[k] ] );
  //}

}

extern "C"
void
data_sparse_subsparse_cs(
  int sub_m,
  int sub_n,
  int sub_mbegin,
  int sub_nbegin,
  data_d_matrix* A,
  data_d_matrix* Asub ) {

  data_zmfree( Asub );
  Asub->num_rows = sub_m;
  Asub->num_cols = sub_n;
  Asub->major = A->major;
  Asub->storage_type = A->storage_type;
  Asub->row = (int*) calloc( (sub_m+1), sizeof(int) );

  //printf("\n data_sparse_subsparse_cs \n");
  int rowindxtmp[sub_m*sub_n];
  int coltmp[sub_n*sub_m];
  int valindxtmp[sub_n*sub_m];
  Asub->nnz = 0;
  //printf("sub_mbegin=%d sub_mbegin+sub_m=%d\n", sub_mbegin, sub_mbegin+sub_m );
  //printf("sub_nbegin=%d sub_nbegin+sub_n=%d\n", sub_nbegin, sub_nbegin+sub_n );
  for(int i=sub_mbegin; i < sub_mbegin+sub_m; i++ ) {
    //printf("row=%d A->row[i]=%d A->col[A->row[i]]=%d A->col[A->row[i+1]-1]=%d \n", i, A->row[i], A->col[A->row[i]], A->col[A->row[i+1]-1] );
    //printf("A->col[A->row[i]] >= sub_nbegin %d  A->col[A->row[i]] < sub_nbegin+sub_n %d\n", A->col[A->row[i]] >= sub_nbegin, A->col[A->row[i]] < sub_nbegin+sub_n );
    Asub->row[i-sub_mbegin] = Asub->nnz;
    for (int j=A->row[i]; j < A->row[i+1]; j++ ) {
      //printf("\tj=%d A->col[j]=%d A->val[j]=%e\n", j, A->col[j], A->val[j] );
      //printf("\tA->col[j] >= sub_nbegin = %d  A->col[j] < sub_nbegin+sub_n = %d \n", A->col[j] >= sub_nbegin, A->col[j] < sub_nbegin+sub_n );
      if ( A->col[j] >= sub_nbegin && A->col[j] < sub_nbegin+sub_n ) {
        //printf("adding A->val[j]=%e from (%d, %d) to [%d, %d]\n",
        //  A->val[j], i, A->col[j], i-sub_mbegin, A->col[j]-sub_nbegin );
        rowindxtmp[Asub->nnz] = i;
        coltmp[Asub->nnz] = A->col[j];
        valindxtmp[Asub->nnz] = j;
        Asub->nnz++;
      }
    }
  }
  Asub->row[sub_m] = Asub->nnz;

  Asub->val = (dataType*) calloc( Asub->nnz, sizeof(dataType) );
  Asub->rowidx = (int*) calloc( Asub->nnz, sizeof(int) );
  Asub->col = (int*) calloc( Asub->nnz, sizeof(int) );

  //printf("\nDONE forming temp row and col arrays.\n");
  //printf("csr tmp :\n");
  //for (int k=0; k<sub_m+1; k++) {
  //  printf("\t%d\n", Asub->row[k] );
  //}
  //printf("\nAsub->nnz=%d\n", Asub->nnz);
  for (int k=0; k<Asub->nnz; k++) {
    Asub->val[k] = A->val[ valindxtmp[k] ];
    Asub->rowidx[k] = rowindxtmp[k] - sub_mbegin;
    Asub->col[k] = coltmp[k] - sub_nbegin;
    //printf("\t%d %d %d %e\n", Asub->rowidx[k], Asub->col[k], valindxtmp[k], A->val[ valindxtmp[k] ] );
  }


}

int
data_sparse_subsparse_cs_lowerupper(
  int sub_m,
  int sub_n,
  int sub_mbegin,
  int sub_nbegin,
  data_d_matrix* A,
  data_d_matrix* Asub ) {

  int info = 0;
  data_zmfree( Asub );
  Asub->num_rows = sub_m;
  Asub->num_cols = sub_n;
  Asub->major = A->major;
  Asub->storage_type = A->storage_type;
  int nnz = 0;
  for(int i=sub_mbegin; i < sub_mbegin+sub_m; i++ ) {
    for (int j=A->row[i]; j < A->row[i+1]; j++ ) {
      if ( A->col[j] >= sub_nbegin && A->col[j] < sub_nbegin+sub_n && A->col[j] != i ) {
        nnz++;
      }
    }
  }

  if ( nnz > 0 ) {
    Asub->nnz = nnz;
    Asub->row = (int*) malloc( (sub_m+1)*sizeof(int) );
    Asub->col = (int*) malloc( nnz*sizeof(int) );
    Asub->val = (dataType*) malloc( nnz*sizeof(dataType) );

    nnz = 0;
    for(int i=sub_mbegin; i < sub_mbegin+sub_m; i++ ) {
      Asub->row[i-sub_mbegin] = nnz;
      for (int j=A->row[i]; j < A->row[i+1]; j++ ) {
        if ( A->col[j] >= sub_nbegin && A->col[j] < sub_nbegin+sub_n && A->col[j] != i ) {
          Asub->col[nnz] = ( A->col[j] - sub_nbegin );
          Asub->val[nnz] = A->val[j];
          nnz++;
        }
      }
    }
    Asub->row[sub_m] = Asub->nnz;
  }
  else {
    data_zmfree( Asub );
    printf("empty sparse sub matrix sub_m=%d sub_n%d sub_mbegin=%d sub_nbegin=%d\n",
      sub_m, sub_n, sub_mbegin, sub_nbegin);
    info = 1;
  }
  return info;
}

int
data_sparse_subsparse_spmm(
  int tile,
  int span,
  int ti,
  int tj,
  Int3* tiles,
  data_d_matrix* L,
  data_d_matrix* U,
  data_d_matrix* C ) {

    data_d_matrix Lsub = {Magma_CSR};
    int infol = data_sparse_subsparse_cs_lowerupper( tile, span, ti, tiles->a[2], L, &Lsub );

    data_d_matrix Usub = {Magma_CSR};
    int infou = data_sparse_subsparse_cs_lowerupper( span, tile, tiles->a[2], tj, U, &Usub );

    if (infol != 0 || infou != 0) {
      printf("==== ti=%d tj=%d span=%d\n", ti, tj, span);
      printf("infol=%d infou=%d\n", infol, infou);
      fflush(stdout);
      exit(1);
    }

    // calculate update
    //data_d_matrix C = {Magma_CSR};

    dataType cone = 1.0;
    int infospmm = data_z_spmm( cone, Lsub, Usub, C );

    if ( infospmm != 0 ) {
      //printf("***\t*** Empty C ***\t***\n");
      C->num_rows = tile;
      C->num_cols = 1;
      C->nnz = 1;
      C->true_nnz = 1;
      C->row = (int*) calloc( (tile+1), sizeof(int) );
      C->col = (int*) calloc( 1, sizeof(int) );
      C->val = (dataType*) calloc( 1, sizeof(dataType) );
    }

    data_zmfree( &Lsub );
    data_zmfree( &Usub );

    return infospmm;
}



#define CALL_AND_CHECK_STATUS(function, error_message) do { \
  if(function != SPARSE_STATUS_SUCCESS)                     \
  {                                                         \
  printf(error_message); fflush(0);                         \
  info = 1;                                                 \
  goto cleanup;                                             \
  }                                                         \
} while(0)

int
data_sparse_subsparse_cs_lowerupper_handle(
  int sub_m,
  int sub_n,
  int sub_mbegin,
  int sub_nbegin,
  int uplo,
  data_d_matrix* A,
  data_d_matrix* Asub,
  sparse_matrix_t* Asub_handle ) {


  int info = 0;
  //printf("data_sparse_subsparse_cs_lowerupper_handle begin:\n");

  Asub->num_rows = sub_m;
  Asub->num_cols = sub_n;
  Asub->major = A->major;
  Asub->storage_type = A->storage_type;
  int nnz = 0;
  if (uplo == MagmaLower) { // L
    for(int i=sub_mbegin; i < sub_mbegin+sub_m; i++ ) {
      for (int j=A->row[i]; j < A->row[i+1]; j++ ) {
        if ( A->col[j] >= sub_nbegin && A->col[j] < sub_nbegin+sub_n && A->col[j] < i ) {
          nnz++;
        }
      }
    }
  }
  else { // U
    for(int i=sub_mbegin; i < sub_mbegin+sub_m; i++ ) {
      for (int j=A->row[i]; j < A->row[i+1]; j++ ) {
        if ( A->col[j] >= sub_nbegin && A->col[j] < sub_nbegin+sub_n && A->col[j] > i ) {
          nnz++;
        }
      }
    }
  }

  if ( nnz > 0 ) {
    Asub->nnz = nnz;
    Asub->row = (int*) malloc( (sub_m+1)*sizeof(int) );
    Asub->col = (int*) malloc( nnz*sizeof(int) );
    Asub->val = (dataType*) malloc( nnz*sizeof(dataType) );

    nnz = 0;
    if (uplo == MagmaLower) { // L
      for(int i=sub_mbegin; i < sub_mbegin+sub_m; i++ ) {
        Asub->row[i-sub_mbegin] = nnz;
        for (int j=A->row[i]; j < A->row[i+1]; j++ ) {
          if ( A->col[j] >= sub_nbegin && A->col[j] < sub_nbegin+sub_n && A->col[j] < i ) {
            Asub->col[nnz] = ( A->col[j] - sub_nbegin );
            Asub->val[nnz] = A->val[j];
            nnz++;
          }
        }
      }
    }
    else { // U
      for(int i=sub_mbegin; i < sub_mbegin+sub_m; i++ ) {
        Asub->row[i-sub_mbegin] = nnz;
        for (int j=A->row[i]; j < A->row[i+1]; j++ ) {
          if ( A->col[j] >= sub_nbegin && A->col[j] < sub_nbegin+sub_n && A->col[j] > i ) {
            Asub->col[nnz] = ( A->col[j] - sub_nbegin );
            Asub->val[nnz] = A->val[j];
            nnz++;
          }
        }
      }
    }
    Asub->row[sub_m] = Asub->nnz;
    //printf("data_sparse_subsparse_cs_lowerupper_handle Asub set:\n");

    sparse_index_base_t indexing = SPARSE_INDEX_BASE_ZERO;
    CALL_AND_CHECK_STATUS( mkl_sparse_d_create_csr( Asub_handle, indexing,
      Asub->num_rows, Asub->num_cols, Asub->row,
      Asub->row+1, Asub->col, Asub->val ),
      "Error after MKL_SPARSE_D_CREATE_CSR, csrA\n");

    //printf("data_sparse_subsparse_cs_lowerupper_handle handle created:\n");
  }
  else {
    printf("empty sparse sub matrix sub_m=%d sub_n%d sub_mbegin=%d sub_nbegin=%d\n",
      sub_m, sub_n, sub_mbegin, sub_nbegin);
    info = 1;
  }

cleanup:
  return info;
}


int
data_sparse_subsparse_spmm_handle(
  int tile,
  int span,
  int ti,
  int tj,
  Int3* tiles,
  sparse_matrix_t* Lsub,
  sparse_matrix_t* Usub,
  data_d_matrix* C ) {

    dataType cone = 1.0;
    int infospmm = data_z_spmm_handle( cone, Lsub, Usub, C );

    if ( infospmm != 0 ) {
      //printf("***\t*** Empty C ***\t***\n");
      C->num_rows = tile;
      C->num_cols = 1;
      C->nnz = 1;
      C->true_nnz = 1;
      C->row = (int*) calloc( (tile+1), sizeof(int) );
      C->col = (int*) calloc( 1, sizeof(int) );
      C->val = (dataType*) calloc( 1, sizeof(dataType) );
    }

    //data_zmfree( &Lsub );
    //data_zmfree( &Usub );

    return infospmm;
}


int
data_sparse_subsparse_spmm_batches(
  int to_update,
  int tile,
  dataType alpha,
  std::vector<sparse_matrix_t>* L_handles,
  std::vector<sparse_matrix_t>* U_handles,
  std::vector<int>* lbatch,
  std::vector<int>* ubatch,
  data_d_matrix* C ) {

  int info = 0;
  int infospmm = 0;
  int infoadd = 0;
  sparse_index_base_t indexing = SPARSE_INDEX_BASE_ZERO;
  dataType *values_C = NULL;
  int *pointerB_C = NULL, *pointerE_C = NULL, *columns_C = NULL;
  int rows_C, cols_C, nnz_C;

  dataType cone = 1.0;
  sparse_matrix_t csrC = NULL;
  sparse_matrix_t csrSum = NULL;
  sparse_matrix_t csrTmp = NULL;
  struct matrix_descr descr_csrTmp;
  descr_csrTmp.type = SPARSE_MATRIX_TYPE_GENERAL;
  descr_csrTmp.mode = SPARSE_FILL_MODE_LOWER;
  descr_csrTmp.diag = SPARSE_DIAG_NON_UNIT;

  data_d_matrix csrTmp_m = {Magma_CSR};
  csrTmp_m.num_rows = tile;
  csrTmp_m.num_cols = tile;
  csrTmp_m.nnz = tile;
  csrTmp_m.true_nnz = tile;
  csrTmp_m.row = (int*) calloc( (tile+1), sizeof(int) );
  csrTmp_m.col = (int*) calloc( tile, sizeof(int) );
  csrTmp_m.val = (dataType*) calloc( tile, sizeof(dataType) );

  info = mkl_sparse_d_create_csr( &csrTmp, indexing,
      csrTmp_m.num_rows, csrTmp_m.num_cols, csrTmp_m.row,
      csrTmp_m.row+1, csrTmp_m.col, csrTmp_m.val );
  printf("temp created info = %d\n", info);
  //info = mkl_sparse_copy( csrTmp, descr_csrTmp, &csrSum );

  data_d_matrix csrSum_m = {Magma_CSR};
  csrSum_m.num_rows = tile;
  csrSum_m.num_cols = tile;
  csrSum_m.nnz = tile;
  csrSum_m.true_nnz = tile;
  csrSum_m.row = (int*) calloc( (tile+1), sizeof(int) );
  csrSum_m.col = (int*) calloc( tile, sizeof(int) );
  csrSum_m.val = (dataType*) calloc( tile, sizeof(dataType) );

  info = mkl_sparse_d_create_csr( &csrSum, indexing,
      csrSum_m.num_rows, csrSum_m.num_cols, csrTmp_m.row,
      csrSum_m.row+1, csrSum_m.col, csrSum_m.val );
  printf("sum created info = %d\n", info);

  printf("to_update = %d, int(lbatch->size())=%d\n", to_update, int(lbatch->size()) );
  for (int b=0; b<int(lbatch->size()); b++) {
    printf("\t(*lbatch)[b]=%d (*ubatch)[b]=%d\n", (*lbatch)[b], (*ubatch)[b] );
    //infospmm += data_z_spmm_batch( cone,
    //  &((*L_handles)[ (*lbatch)[b] ]), &((*U_handles)[ (*ubatch)[b] ]), &csrC );
    infospmm = mkl_sparse_spmm( SPARSE_OPERATION_NON_TRANSPOSE,
      ((*L_handles)[ (*lbatch)[b] ]), ((*U_handles)[ (*ubatch)[b] ]), &csrC );

    mkl_sparse_d_export_csr( ((*L_handles)[ (*lbatch)[b] ]), &indexing,
      &rows_C, &cols_C,
      &pointerB_C, &pointerE_C, &columns_C, &values_C );
    printf("\tL Handle rows_C =%d cols_C = %d pointerE_C[ rows_C-1 ] = %d\n",
      rows_C, cols_C, pointerE_C[ rows_C-1 ]);
    printf("\tvals:\n\t");
    for (int pi=0; pi<pointerE_C[ rows_C-1 ]; pi++) {
      printf("\t%e", values_C[pi]);
    }
    printf("\n");

    mkl_sparse_d_export_csr( ((*U_handles)[ (*ubatch)[b] ]), &indexing,
      &rows_C, &cols_C,
      &pointerB_C, &pointerE_C, &columns_C, &values_C );
    printf("\tU Handle rows_C =%d cols_C = %d pointerE_C[ rows_C-1 ] = %d\n",
      rows_C, cols_C, pointerE_C[ rows_C-1 ]);
    printf("\tvals:\n\t");
    for (int pi=0; pi<pointerE_C[ rows_C-1 ]; pi++) {
      printf("\t%e", values_C[pi]);
    }
    printf("\n");

    mkl_sparse_d_export_csr( csrC, &indexing,
      &rows_C, &cols_C,
      &pointerB_C, &pointerE_C, &columns_C, &values_C );
    printf("\tC rows_C =%d cols_C = %d pointerE_C[ rows_C-1 ] = %d\n",
      rows_C, cols_C, pointerE_C[ rows_C-1 ]);
    printf("\tvals:\n\t");
    for (int pi=0; pi<pointerE_C[ rows_C-1 ]; pi++) {
      printf("\t%e", values_C[pi]);
    }
    printf("\n");

    printf("batch %d infospmm = %d\n", b, infospmm );

    if ( infospmm == SPARSE_STATUS_SUCCESS ) {
      printf("+\tadd");
      //sparse_status_t mkl_sparse_copy (const sparse_matrix_t source,
      //  struct matrix_descr descr, sparse_matrix_t *dest);
      mkl_sparse_copy( csrSum, descr_csrTmp, &csrTmp );
      //mkl_sparse_d_add (sparse_operation_t operation, const sparse_matrix_t A, dataType alpha, const sparse_matrix_t B, sparse_matrix_t *C);
      infoadd += mkl_sparse_d_add( SPARSE_OPERATION_NON_TRANSPOSE, csrC, cone, csrTmp, &csrSum );
      printf(" %d  done\n", infoadd);
      mkl_sparse_d_export_csr( csrSum, &indexing,
        &rows_C, &cols_C,
        &pointerB_C, &pointerE_C, &columns_C, &values_C );
      printf("\t running SUM rows_C =%d cols_C = %d pointerE_C[ rows_C-1 ] = %d\n",
        rows_C, cols_C, pointerE_C[ rows_C-1 ]);
      printf("\tvals:\n\t");
      for (int pi=0; pi<pointerE_C[ rows_C-1 ]; pi++) {
        printf("\t%e", values_C[pi]);
      }
      printf("\n");
    }
    else if ( b > 0 ) {
      info += infospmm;
      infospmm = SPARSE_STATUS_SUCCESS;
    }
  }

  if ( infoadd == 0 ) {
    mkl_sparse_d_export_csr( csrSum, &indexing,
      &rows_C, &cols_C,
      &pointerB_C, &pointerE_C, &columns_C, &values_C );
    printf("\t___ C rows_C =%d cols_C = %d pointerE_C[ rows_C-1 ] = %d\n",
      rows_C, cols_C, pointerE_C[ rows_C-1 ]);
    printf("\tvals:\n\t");
    for (int pi=0; pi<pointerE_C[ rows_C-1 ]; pi++) {
      printf("\t%e", values_C[pi]);
    }
    printf("\n");
  }

  printf("infospmm = %d, infoadd = %d\n", infospmm, infoadd );

 printf("SPARSE_STATUS_SUCCESS = %d SPARSE_STATUS_NOT_INITIALIZED=%d\n",
   SPARSE_STATUS_SUCCESS, SPARSE_STATUS_NOT_INITIALIZED);

 printf("SPARSE_STATUS_ALLOC_FAILED=%d SPARSE_STATUS_INVALID_VALUE=%d SPARSE_STATUS_EXECUTION_FAILED=%d SPARSE_STATUS_INTERNAL_ERROR=%d SPARSE_STATUS_NOT_SUPPORTED=%d\n",
   SPARSE_STATUS_ALLOC_FAILED, SPARSE_STATUS_INVALID_VALUE, SPARSE_STATUS_EXECUTION_FAILED, SPARSE_STATUS_INTERNAL_ERROR, SPARSE_STATUS_NOT_SUPPORTED);

  if ( infospmm == SPARSE_STATUS_SUCCESS && infoadd == SPARSE_STATUS_SUCCESS ) {
    sparse_index_base_t indexing = SPARSE_INDEX_BASE_ZERO;
    infospmm = mkl_sparse_d_export_csr( csrSum, &indexing,
      &rows_C, &cols_C,
      &pointerB_C, &pointerE_C, &columns_C, &values_C );

    // ensure column indices are in ascending order in every row
    //printf( "\n RESULTANT MATRIX C:\nrow# : (value, column) (value, column)\n" );
    int ii = 0;
    int coltmp;
    dataType valtmp;
    for( int i = 0; i < rows_C; i++ ) {
      //printf("row#%d:", i); fflush(0);
      for( int j = pointerB_C[i]; j < pointerE_C[i]; j++ ) {
        //printf(" (%e, %6d)", values_C[ii], columns_C[ii] ); fflush(0);
        if ( j+1 < pointerE_C[i] && columns_C[ii] > columns_C[ii+1]) {
          //printf("\nSWAP!!!\n");
          valtmp = values_C[ii];
          values_C[ii] = values_C[ii+1];
          values_C[ii+1] = valtmp;
          coltmp = columns_C[ii];
          columns_C[ii] = columns_C[ii+1];
          columns_C[ii+1] = coltmp;
        }
        ii++;
      }
      //printf( "\n" );
    }
    //printf( "_____________________________________________________________________  \n" );

    nnz_C = pointerE_C[ rows_C-1 ];

    // fill in information for C
    C->storage_type = Magma_CSR;
    //C->sym = A.sym;
    C->diagorder_type = Magma_VALUE;
    C->fill_mode = MagmaFull;
    C->num_rows = rows_C;
    C->num_cols = cols_C;
    C->nnz = nnz_C;
    C->true_nnz = nnz_C;
    // memory allocation
    //CHECK( magma_zmalloc( &C->dval, nnz_C ));
    C->val = (dataType*) malloc( nnz_C*sizeof(dataType) );
     for( int i=0; i<nnz_C; i++) {
      C->val[i] = values_C[i] * alpha;
    }
    //CHECK( magma_index_malloc( &C->drow, rows_C + 1 ));
    C->row = (int*) malloc ( (rows_C+1)*sizeof(int));
     for( int i=0; i<rows_C; i++) {
      C->row[i] = pointerB_C[i];
    }
    C->row[rows_C] = pointerE_C[rows_C-1];
    //CHECK( magma_index_malloc( &C->dcol, nnz_C ));
    C->col = (int*) malloc ( (nnz_C)*sizeof(int));
     for( int i=0; i<nnz_C; i++) {
      C->col[i] = columns_C[i];
    }

  }
  else {
    //printf("***\t*** Empty C ***\t***\n");
    C->num_rows = tile;
    C->num_cols = 1;
    C->nnz = 1;
    C->true_nnz = 1;
    C->row = (int*) calloc( (tile+1), sizeof(int) );
    C->col = (int*) calloc( 1, sizeof(int) );
    C->val = (dataType*) calloc( 1, sizeof(dataType) );
  }

  data_zmfree( &csrTmp_m );

  return infospmm;
}
