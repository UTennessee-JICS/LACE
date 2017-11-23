/*
 *  -- LACE (version 0.0) --
 *     Univ. of Tennessee, Knoxville
 *
 *     @author Stephen Wood
 *
 */
#include "../include/sparse.h"
#include <mkl.h>
#include <stdlib.h>
#include <stdio.h>

extern "C"
void
data_PariLU_v3_1(data_d_matrix * A, data_d_matrix * L, data_d_matrix * U, int tile)
{
  data_zdiameter(A);
  // printf("A:\n");
  // data_zwrite_csr( A );
  data_z_pad_csr(A, tile);
  printf("A padded by %d tile:\n", tile);
  // data_zwrite_csr( A );

  std::vector<Int3> tiles;
  data_sparse_tilepattern(tile, tile, &tiles, A);
  printf("\nA's active tiles:\n");
  for (int i = 0; i < int(tiles.size()); i++) {
    printf("[%d, %d : %d]\n", tiles[i].a[0], tiles[i].a[1], tiles[i].a[2]);
  }
  // getchar();

  // Separate the strictly lower, strictly upper, and diagonal elements
  // into L, U, and D respectively.
  L->diagorder_type = Magma_UNITY;
  data_zmconvert(*A, L, Magma_CSR, Magma_CSRL);
  data_d_matrix Lc = { Magma_CSR };
  Lc.diagorder_type = Magma_UNITY;
  data_zmconvert(*A, &Lc, Magma_CSR, Magma_CSRL);
  // printf("L done.\n");
  // data_zwrite_csr( L );

  U->diagorder_type = Magma_VALUE;
  // store U in column major
  // U->major = MagmaColMajor;
  U->major = MagmaRowMajor;
  data_zmconvert(*A, U, Magma_CSR, Magma_CSRU);
  data_d_matrix Uc = { Magma_CSR };
  Uc.diagorder_type = Magma_VALUE;
  data_zmconvert(*A, &Uc, Magma_CSR, Magma_CSRU);
  // printf("U done.\n");
  // data_zwrite_csr( U );

  data_d_matrix D = { Magma_DENSED };
  data_zmconvert(*A, &D, Magma_CSR, Magma_DENSED);
  // data_zprint_dense( D );
  data_d_matrix Dc = { Magma_DENSED };
  data_zmconvert(*A, &Dc, Magma_CSR, Magma_DENSED);
  // printf("D done.\n");

  // Set diagonal elements to the recipricol
  #pragma omp parallel
  #pragma omp for nowait
  for (int i = 0; i < D.nnz; i++) {
    D.val[ i ] = 1.0 / D.val[ i ];
  }

  int row_limit = A->num_rows;
  // int col_limit = A->num_cols;
  if (A->pad_rows > 0 && A->pad_cols > 0) {
    row_limit = A->pad_rows;
    // col_limit = A->pad_cols;
  }

  // ParLU element wise
  int iter       = 0;
  dataType tmp   = 0.0;
  dataType step  = 1.0;
  dataType tol   = 1.0e-15;
  dataType Anorm = 0.0;

  int num_threads = 0;

  dataType cone  = 1.0;
  dataType czero = 0.0;

  int ii = 0, jj = 0;

  data_zfrobenius(*A, &Anorm);
  printf("%% Anorm = %e\n", Anorm);
  printf("%% A->diameter=%d\n", A->diameter);

  dataType wstart   = omp_get_wtime();
  while (step > tol && iter < 1000) {
    step = 0.0;
    #pragma omp parallel private(ii, jj, tmp)
    {
      num_threads = omp_get_num_threads();
      #pragma omp for schedule(static,1) reduction(+:step) nowait
      for (int i = 0; i < int(tiles.size()); i++) {
        // printf("[%d, %d : %d]\n", tiles[i].a[0], tiles[i].a[1], tiles[i].a[2] );
        // TODO : check if this is an active tile
        // TODO : iterate over a list of active tiles
        int ti = tiles[i].a[0];
        int tj = tiles[i].a[1];
        // int minrow = MAX(0, ti - A->diameter);
        // int mincol = MAX(0, tj - A->diameter);
        // int maxrow = ti + tile;
        int maxcol = tj + tile;
        int span   = MIN( (ti + tile - 0), (tj + tile - 0) );

        dataType * Lblock;
        Lblock = (dataType *) calloc(tile * span, sizeof(dataType) );
        data_sparse_subdense_lowerupper(tile, span, ti, 0, L, Lblock);
        // printf("Lbkock:\n");
        // for (int ip=0; ip<tile; ip++) {
        //  for (int jp=0; jp<span; jp++) {
        //    printf("%e ", Lblock[ip*tile + jp]);
        //  }
        //  printf("\n");
        // }
        // printf("\n");

        dataType * Ublock;
        Ublock = (dataType *) calloc(span * tile, sizeof(dataType) );
        data_sparse_subdense_lowerupper(span, tile, 0, tj, U, Ublock);
        // printf("Ubkock:\n");
        // for (int ip=0; ip<span; ip++) {
        //  for (int jp=0; jp<tile; jp++) {
        //    printf("%e ", Ublock[ip*span + jp]);
        //  }
        //  printf("\n");
        // }
        // printf("\n");

        dataType Ctile2[tile * tile];

        // caLculate update
        data_dgemm_mkl(L->major,
          MagmaNoTrans, MagmaNoTrans,
          tile, tile, span,
          cone, Lblock, span,
          Ublock, tile,
          czero, Ctile2, tile);

        free(Lblock);
        free(Ublock);
        // printf("Ctile2:\n");
        // for (int ip=0; ip<tile; ip++) {
        //  for (int jp=0; jp<tile; jp++) {
        //    printf("%e ", Ctile2[ip*tile + jp]);
        //  }
        //  printf("\n");
        // }
        // printf("\n");
        // printf("Ctile2 done.\n");

        // upper tiles and tiles along major diagonal
        if (ti <= tj) {
          // begin updating values of U
          for (int i = ti; i < ti + tile; i++) {
            // #pragma omp simd
            for (int j = Uc.row[i]; j < Uc.row[i + 1]; j++) {
              if (U->col[j] >= tj && U->col[j] < maxcol && U->col[j] != i) {
                ii = i - ti;
                jj = U->col[j] - tj;
                tmp         = Uc.val[ j ] - Ctile2[ ii * tile + jj ];
                step       += pow(U->val[ j ] - tmp, 2);
                U->val[ j ] = tmp;
              }
            }
          }
          if (ti == tj) {
            // maxcol = tj + tile;
            // begin updating major diagonal values of U
            // #pragma omp simd
            for (int i = ti; i < ti + tile; i++) {
              for (int j = U->row[i]; j < U->row[i + 1]; j++) {
                if (U->col[j] == i) {
                  ii = i - ti;
                  jj = U->col[j] - tj;
                  // TODO : unnecessary, update U->val[ j ] directly,
                  // temporary L and U blocks do not use the major diagonals
                  tmp        = 1.0 / ( Dc.val[ i ] - Ctile2[ ii * tile + jj ] );
                  step      += pow(D.val[ i ] - tmp, 2);
                  D.val[ i ] = tmp;
                }
              }
            }
            // begin updating values of L
            for (int i = ti; i < ti + tile; i++) {
              // #pragma omp simd
              for (int j = Lc.row[i]; j < Lc.row[i + 1]; j++) {
                if (L->col[j] >= tj && L->col[j] < maxcol && L->col[j] != i) {
                  ii = i - ti;
                  jj = L->col[j] - tj;
                  tmp         = ( (Lc.val[ j ] - Ctile2[ ii * tile + jj ]) * D.val[ L->col[j] ] );
                  step       += pow(L->val[ j ] - tmp, 2);
                  L->val[ j ] = tmp;
                }
              }
            }
          }
          // done updating values for tiles in the upper triangle and on the major diagonal
        }
        // strictly lower tiles
        else {
          // begin updating values of L
          for (int i = ti; i < ti + tile; i++) {
            // #pragma omp simd
            for (int j = Lc.row[i]; j < Lc.row[i + 1]; j++) {
              if (Lc.col[j] >= tj && L->col[j] < maxcol && L->col[j] != i) {
                ii = i - ti;
                jj = L->col[j] - tj;
                tmp         = ( (Lc.val[ j ] - Ctile2[ ii * tile + jj ]) * D.val[ L->col[j] ] );
                step       += pow(L->val[ j ] - tmp, 2);
                L->val[ j ] = tmp;
              }
            }
          }
          // done updating values for strictly lower tiles
        }
      }
    }
    step /= Anorm;
    iter++;
    printf("%% iteration = %d step = %e\n", iter, step);
  }
  dataType wend     = omp_get_wtime();
  dataType ompwtime = (dataType) (wend - wstart) / ((dataType) iter);

  #pragma omp parallel
  #pragma omp for nowait
  for (int i = 0; i < row_limit; i++) {
    for (int j = U->row[i]; j < U->row[i + 1]; j++) {
      if (U->col[j] == i) {
        U->val[ j ] = 1.0 / D.val[ i ];
      }
    }
  }


  printf(
    "%% PariLU v3.1 used %d OpenMP threads and required %d iterations, %f wall clock seconds, and an average of %f wall clock seconds per iteration as measured by omp_get_wtime()\n",
    num_threads, iter, wend - wstart, ompwtime);
  fflush(stdout);

  data_zmfree(&D);
  data_zmfree(&Dc);
  data_zmfree(&Lc);
  data_zmfree(&Uc);
} // data_PariLU_v3_1
