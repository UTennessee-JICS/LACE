
#include <stdlib.h>
#include <stdio.h>
#include "../include/sparse.h"
#include <vector>
#include <climits>

void
data_sparse_tilepattern( int sub_m, int sub_n, std::vector<Int3>* tiles, 
  data_d_matrix* A ) {

  int activetile = 0;
  int tilerowbegin = INT_MAX;
  
  int num_tile_cols = ceil( A->num_cols/sub_n );
  int* minrow;
  minrow = (int*) malloc( num_tile_cols*sizeof(int) );
  for (int t=0; t<num_tile_cols; t++) {
    minrow[t] = INT_MAX;
  }
  
  for (int sub_mcounter=0; sub_mcounter < A->num_rows; sub_mcounter+=sub_m ) { 
    for (int sub_ncounter=0; sub_ncounter < A->num_cols; sub_ncounter+=sub_n ) { 
      activetile = 0;
      for (int i=sub_mcounter; i < sub_mcounter+sub_m; i++ ) {
        //printf("row=%d A->row[i]=%d A->col[A->row[i]]=%d A->col[A->row[i+1]-1]=%d \n", i, A->row[i], A->col[A->row[i]], A->col[A->row[i+1]-1] );
        //printf("A->col[A->row[i]] >= sub_ncounter %d  A->col[A->row[i]] < sub_ncounter+sub_n %d\n", A->col[A->row[i]] >= sub_ncounter, A->col[A->row[i]] < sub_ncounter+sub_n );
        
        if (activetile==0) {
          for (int j=A->row[i]; j < A->row[i+1]; j++ ) {
            //printf("\tj=%d A->col[j]=%d A->val[j]=%e\n", j, A->col[j], A->val[j] );
            //printf("\tA->col[j] >= sub_ncounter = %d  A->col[j] < sub_ncounter+sub_n = %d \n", A->col[j] >= sub_ncounter, A->col[j] < sub_ncounter+sub_n );
            if ( A->col[j] >= sub_ncounter && A->col[j] < sub_ncounter+sub_n && A->col[j] != i) {
              //printf("adding A->val[j]=%e from (%d, %d) to [%d, %d]\n",
              //  A->val[j], i, A->col[j], i-sub_mcounter, A->col[j]-sub_ncounter );
              activetile = 1;
              break;
            }
          }
        }
        
      }
      if (activetile==1) {
        tilerowbegin = INT_MAX;
        for (int i=sub_mcounter; i < sub_mcounter+sub_m; i++ ) {
          if (A->col[A->row[i]] < tilerowbegin) {
            tilerowbegin = (A->col[A->row[i]]/sub_n)*sub_n;
          }
        }
        if ( sub_mcounter < minrow[ sub_ncounter/sub_n ] ) {
          minrow[ sub_ncounter/sub_n ] = sub_mcounter;
        }
        tiles->push_back({{sub_mcounter, sub_ncounter, tilerowbegin}});
        activetile = 0;
      }
      
    }
  }
  
  for (int t=0; t<int(tiles->size()); t++) {
    if ( (*tiles)[t].a[2] > minrow[ (*tiles)[t].a[1]/sub_n ] ) {
      (*tiles)[t].a[2] = minrow[ (*tiles)[t].a[1]/sub_n ];
    }
  }
  
  free( minrow );
  
}

void
data_sparse_tilepattern_lowerupper( int sub_m, int sub_n, std::vector<Int3>* tiles, 
  data_d_matrix* A ) {

  int activetile = 0;
  int tilerowbegin = INT_MAX;
  
  for (int sub_mcounter=0; sub_mcounter < A->num_rows; sub_mcounter+=sub_m ) { 
    for (int sub_ncounter=0; sub_ncounter < A->num_cols; sub_ncounter+=sub_n ) { 
      activetile = 0;
      for (int i=sub_mcounter; i < sub_mcounter+sub_m; i++ ) {
        //printf("row=%d A->row[i]=%d A->col[A->row[i]]=%d A->col[A->row[i+1]-1]=%d \n", i, A->row[i], A->col[A->row[i]], A->col[A->row[i+1]-1] );
        //printf("A->col[A->row[i]] >= sub_ncounter %d  A->col[A->row[i]] < sub_ncounter+sub_n %d\n", A->col[A->row[i]] >= sub_ncounter, A->col[A->row[i]] < sub_ncounter+sub_n );
        
        if (activetile==0) {
          for (int j=A->row[i]; j < A->row[i+1]; j++ ) {
            //printf("\tj=%d A->col[j]=%d A->val[j]=%e\n", j, A->col[j], A->val[j] );
            //printf("\tA->col[j] >= sub_ncounter = %d  A->col[j] < sub_ncounter+sub_n = %d \n", A->col[j] >= sub_ncounter, A->col[j] < sub_ncounter+sub_n );
            if ( A->col[j] >= sub_ncounter && A->col[j] < sub_ncounter+sub_n && A->col[j] != i) {
              //printf("adding A->val[j]=%e from (%d, %d) to [%d, %d]\n",
              //  A->val[j], i, A->col[j], i-sub_mcounter, A->col[j]-sub_ncounter );
              activetile = 1;
            }
          }
        }
        
      }
      if (activetile==1) {
        tilerowbegin = INT_MAX;
        for (int i=sub_mcounter; i < sub_mcounter+sub_m; i++ ) {
          if (A->col[A->row[i]] < tilerowbegin) {
            tilerowbegin = (A->col[A->row[i]]/sub_n)*sub_n;
          }
        }
        tiles->push_back({{sub_mcounter, sub_ncounter, tilerowbegin}});
        tilerowbegin = INT_MAX;    
        activetile = 0;
      }
      
    }
  }
  
}


void
data_sparse_tilepatterns( int sub_m, int sub_n, std::vector<Int3>* Ltiles, 
  std::vector<Int3>* Utiles, data_d_matrix* A ) {

  int activetile = 0;
  int tilerowbegin = INT_MAX;
  
  int num_tile_cols = ceil( A->num_cols/sub_n );
  int* minrow;
  minrow = (int*) malloc( num_tile_cols*sizeof(int) );
  for (int t=0; t<num_tile_cols; t++) {
    minrow[t] = INT_MAX;
  }
  
  std::vector<Int3> Atiles;
  
  for (int sub_mcounter=0; sub_mcounter < A->num_rows; sub_mcounter+=sub_m ) { 
    for (int sub_ncounter=0; sub_ncounter < A->num_cols; sub_ncounter+=sub_n ) { 
      activetile = 0;
      for (int i=sub_mcounter; i < sub_mcounter+sub_m; i++ ) {
        //printf("row=%d A->row[i]=%d A->col[A->row[i]]=%d A->col[A->row[i+1]-1]=%d \n", i, A->row[i], A->col[A->row[i]], A->col[A->row[i+1]-1] );
        //printf("A->col[A->row[i]] >= sub_ncounter %d  A->col[A->row[i]] < sub_ncounter+sub_n %d\n", A->col[A->row[i]] >= sub_ncounter, A->col[A->row[i]] < sub_ncounter+sub_n );
        
        if (activetile==0) {
          for (int j=A->row[i]; j < A->row[i+1]; j++ ) {
            //printf("\tj=%d A->col[j]=%d A->val[j]=%e\n", j, A->col[j], A->val[j] );
            //printf("\tA->col[j] >= sub_ncounter = %d  A->col[j] < sub_ncounter+sub_n = %d \n", A->col[j] >= sub_ncounter, A->col[j] < sub_ncounter+sub_n );
            if ( A->col[j] >= sub_ncounter && A->col[j] < sub_ncounter+sub_n && A->col[j] != i) {
              //printf("adding A->val[j]=%e from (%d, %d) to [%d, %d]\n",
              //  A->val[j], i, A->col[j], i-sub_mcounter, A->col[j]-sub_ncounter );
              activetile = 1;
              break;
            }
          }
        }
        
      }
      if (activetile==1) {
        tilerowbegin = INT_MAX;
        for (int i=sub_mcounter; i < sub_mcounter+sub_m; i++ ) {
          if (A->col[A->row[i]] < tilerowbegin) {
            tilerowbegin = (A->col[A->row[i]]/sub_n)*sub_n;
          }
        }
        if ( sub_mcounter < minrow[ sub_ncounter/sub_n ] ) {
          minrow[ sub_ncounter/sub_n ] = sub_mcounter;
        }
        Atiles.push_back({{sub_mcounter, sub_ncounter, tilerowbegin}});
        activetile = 0;
      }
      
    }
  }
  
  for (int t=0; t<int(Atiles.size()); t++) {
    int cidx = Atiles[t].a[1]/sub_n;
    if ( cidx > 0 ) {
      if ( Atiles[t].a[2] > minrow[ cidx ] ) {
        Atiles[t].a[2] = minrow[ cidx ];
      }
    }
    if ( Atiles[t].a[2] < 0 ) {
      Atiles[t].a[2] = 0;
    }
  }
  
  free( minrow );
  
  // separate Atiles into strictly lower tiles and upper tiles 
  for (int t=0; t<int(Atiles.size()); t++) {
    if ( Atiles[t].a[0] <= Atiles[t].a[1] ) {
      Utiles->push_back( Atiles[t] );
    }
    else {
      Ltiles->push_back( Atiles[t] );
    }
  }
  
  
}


void
data_sparse_tilepattern_handles( int sub_m, int sub_n, 
  std::vector<Int3>* tiles, 
  std::vector<data_d_matrix>* L_subs,
  std::vector<data_d_matrix>* U_subs,
  std::vector<sparse_matrix_t>* L_handles,
  std::vector<sparse_matrix_t>* U_handles,
  data_d_matrix* A ) {

  int activetile = 0;
  int tilerowbegin = INT_MAX;
  
  int num_tile_cols = ceil( A->num_cols/sub_n );
  int* minrow;
  minrow = (int*) malloc( num_tile_cols*sizeof(int) );
  for (int t=0; t<num_tile_cols; t++) {
    minrow[t] = INT_MAX;
  }
  printf("data_sparse_tilepattern_handles begin:\n");
  
  for (int sub_mcounter=0; sub_mcounter < A->num_rows; sub_mcounter+=sub_m ) { 
    for (int sub_ncounter=0; sub_ncounter < A->num_cols; sub_ncounter+=sub_n ) { 
      activetile = 0;
      for (int i=sub_mcounter; i < sub_mcounter+sub_m; i++ ) {
        //printf("row=%d A->row[i]=%d A->col[A->row[i]]=%d A->col[A->row[i+1]-1]=%d \n", i, A->row[i], A->col[A->row[i]], A->col[A->row[i+1]-1] );
        //printf("A->col[A->row[i]] >= sub_ncounter %d  A->col[A->row[i]] < sub_ncounter+sub_n %d\n", A->col[A->row[i]] >= sub_ncounter, A->col[A->row[i]] < sub_ncounter+sub_n );
        
        if (activetile==0) {
          for (int j=A->row[i]; j < A->row[i+1]; j++ ) {
            //printf("\tj=%d A->col[j]=%d A->val[j]=%e\n", j, A->col[j], A->val[j] );
            //printf("\tA->col[j] >= sub_ncounter = %d  A->col[j] < sub_ncounter+sub_n = %d \n", A->col[j] >= sub_ncounter, A->col[j] < sub_ncounter+sub_n );
            if ( A->col[j] >= sub_ncounter && A->col[j] < sub_ncounter+sub_n && A->col[j] != i) {
              //printf("adding A->val[j]=%e from (%d, %d) to [%d, %d]\n",
              //  A->val[j], i, A->col[j], i-sub_mcounter, A->col[j]-sub_ncounter );
              activetile = 1;
              break;
            }
          }
        }
        
      }
      if (activetile==1) {
        tilerowbegin = INT_MAX;
        for (int i=sub_mcounter; i < sub_mcounter+sub_m; i++ ) {
          if (A->col[A->row[i]] < tilerowbegin) {
            tilerowbegin = (A->col[A->row[i]]/sub_n)*sub_n;
          }
        }
        if ( sub_mcounter < minrow[ sub_ncounter/sub_n ] ) {
          minrow[ sub_ncounter/sub_n ] = sub_mcounter;
        }
        tiles->push_back({{sub_mcounter, sub_ncounter, tilerowbegin}});
        
        activetile = 0;
      }
      
    }
  }
  
  
  printf("data_sparse_tilepattern_handles min row setting begin:\n");
  for (int t=0; t<int((*tiles).size()); t++) {
    int cidx = (*tiles)[t].a[1]/sub_n;
    if ( cidx > 0 ) {
      if ( (*tiles)[t].a[2] > minrow[ cidx ] ) {
        (*tiles)[t].a[2] = minrow[ cidx ];
      }
    }
    if ( (*tiles)[t].a[2] < 0 ) {
      (*tiles)[t].a[2] = 0;
    }
  }
  
  
  printf("data_sparse_tilepattern_handles creating handles begin:\n");
  for(int i=0; i< int(tiles->size()); i++ ) {
    int ti = (*tiles)[i].a[0];
    int tj = (*tiles)[i].a[1];
    int maxrow = ti + sub_m; 
    int maxcol = tj + sub_n; 
    //int span = MIN( (ti+tile - 0), (tj+tile - 0) );
    int span = MIN( (maxrow - (*tiles)[i].a[2]), (maxcol - (*tiles)[i].a[2]) );
    data_d_matrix Lsub = {Magma_CSR}; 
    sparse_matrix_t Lhandle = NULL;
    printf("tile %d span %d\n", i, span);
    
    int infol = data_sparse_subsparse_cs_lowerupper_handle( sub_m, span, ti, (*tiles)[i].a[2], MagmaLower, A, &Lsub, &Lhandle );
    printf("%d L added\n", i);
    
    data_d_matrix Usub = {Magma_CSR}; 
    sparse_matrix_t Uhandle = NULL;
    int infou = data_sparse_subsparse_cs_lowerupper_handle( span, sub_m, (*tiles)[i].a[2], tj, MagmaUpper, A, &Usub, &Uhandle );
    printf("%d U added\n", i);
    
    if (infol != 0 || infou != 0) {
      printf("==== ti=%d tj=%d span=%d\n", ti, tj, span);
      printf("infol=%d infou=%d\n", infol, infou);
      fflush(stdout);
      exit(1);
    }
    L_subs->push_back(Lsub);
    U_subs->push_back(Usub);
    L_handles->push_back(Lhandle);
    U_handles->push_back(Uhandle);
    
  }
  
  printf("%% num tiles = %d, num L_subs = %d, num U_subs = %d, L_handles = %d, U_handles = %d\n",
    int(tiles->size()), int(L_subs->size()), int(U_subs->size()), int(L_handles->size()), int(U_handles->size()) );
  //getchar();
  
  
  free( minrow );
  
  printf("data_sparse_tilepattern_handles Done.\n");
}

void
data_sparse_tilepatterns_handles( int sub_m, int sub_n, 
  std::vector<Int3>* Atiles, 
  std::vector<Int3>* Ltiles, 
  std::vector<Int3>* Utiles, 
  std::vector<data_d_matrix>* L_subs,
  std::vector<data_d_matrix>* U_subs,
  std::vector<sparse_matrix_t>* L_handles,
  std::vector<sparse_matrix_t>* U_handles,
  std::vector< std::vector<int> >* Lbatches,
  std::vector< std::vector<int> >* Ubatches,
  data_d_matrix* A ) {

  int activetile = 0;
  int tilerowbegin = INT_MAX;
  
  int num_tile_cols = ceil( A->num_cols/sub_n );
  int* minrow;
  minrow = (int*) malloc( num_tile_cols*sizeof(int) );
  for (int t=0; t<num_tile_cols; t++) {
    minrow[t] = INT_MAX;
  }
  
  //std::vector<Int3> Atiles;
  
  for (int sub_mcounter=0; sub_mcounter < A->num_rows; sub_mcounter+=sub_m ) { 
    for (int sub_ncounter=0; sub_ncounter < A->num_cols; sub_ncounter+=sub_n ) { 
      activetile = 0;
      for (int i=sub_mcounter; i < sub_mcounter+sub_m; i++ ) {
        //printf("row=%d A->row[i]=%d A->col[A->row[i]]=%d A->col[A->row[i+1]-1]=%d \n", i, A->row[i], A->col[A->row[i]], A->col[A->row[i+1]-1] );
        //printf("A->col[A->row[i]] >= sub_ncounter %d  A->col[A->row[i]] < sub_ncounter+sub_n %d\n", A->col[A->row[i]] >= sub_ncounter, A->col[A->row[i]] < sub_ncounter+sub_n );
        
        if (activetile==0) {
          for (int j=A->row[i]; j < A->row[i+1]; j++ ) {
            //printf("\tj=%d A->col[j]=%d A->val[j]=%e\n", j, A->col[j], A->val[j] );
            //printf("\tA->col[j] >= sub_ncounter = %d  A->col[j] < sub_ncounter+sub_n = %d \n", A->col[j] >= sub_ncounter, A->col[j] < sub_ncounter+sub_n );
            if ( A->col[j] >= sub_ncounter && A->col[j] < sub_ncounter+sub_n && A->col[j] != i) {
              //printf("adding A->val[j]=%e from (%d, %d) to [%d, %d]\n",
              //  A->val[j], i, A->col[j], i-sub_mcounter, A->col[j]-sub_ncounter );
              activetile = 1;
              break;
            }
          }
        }
        
      }
      if (activetile==1) {
        tilerowbegin = INT_MAX;
        for (int i=sub_mcounter; i < sub_mcounter+sub_m; i++ ) {
          if (A->col[A->row[i]] < tilerowbegin) {
            tilerowbegin = (A->col[A->row[i]]/sub_n)*sub_n;
          }
        }
        if ( sub_mcounter < minrow[ sub_ncounter/sub_n ] ) {
          minrow[ sub_ncounter/sub_n ] = sub_mcounter;
        }
        Atiles->push_back({{sub_mcounter, sub_ncounter, tilerowbegin}});
        activetile = 0;
      }
      
    }
  }
  
  for (int t=0; t<int(Atiles->size()); t++) {
    int cidx = (*Atiles)[t].a[1]/sub_n;
    if ( cidx > 0 && cidx < num_tile_cols ) {
      if ( (*Atiles)[t].a[2] > minrow[ cidx ] ) {
        (*Atiles)[t].a[2] = minrow[ cidx ];
      }
    }
    if ( (*Atiles)[t].a[2] < 0 ) {
      (*Atiles)[t].a[2] = 0;
    }
  }
  
  free( minrow );
  
  // separate Atiles into lower tiles and upper tiles 
  for (int t=0; t<int(Atiles->size()); t++) {
    if ( (*Atiles)[t].a[0] <= (*Atiles)[t].a[1] ) {
      Utiles->push_back( (*Atiles)[t] );
      data_d_matrix Usub = {Magma_CSR}; 
      sparse_matrix_t Uhandle = NULL;
      data_sparse_subsparse_cs_lowerupper_handle( 
        sub_m, sub_n, (*Atiles)[t].a[0], (*Atiles)[t].a[1], MagmaUpper, A, &Usub, &Uhandle );
      //printf("%d U added\n", t);
      U_subs->push_back(Usub);
      U_handles->push_back(Uhandle);
    }
    if ( (*Atiles)[t].a[0] >= (*Atiles)[t].a[1] ) {
      Ltiles->push_back( (*Atiles)[t] );
      data_d_matrix Lsub = {Magma_CSR}; 
      sparse_matrix_t Lhandle = NULL;
      data_sparse_subsparse_cs_lowerupper_handle( 
        sub_m, sub_n, (*Atiles)[t].a[0], (*Atiles)[t].a[1], MagmaLower, A, &Lsub, &Lhandle );
      //printf("%d L added\n", t);
      L_subs->push_back(Lsub);
      L_handles->push_back(Lhandle);
    }
  }
  
  
  // build batch lists for L and U  
  // TODO: make lists coherent (only add if the block of L aligns with a block of U)
  //std::vector< std::vector<int> > Lbatches;
  for (int t=0; t<int(Ltiles->size()); t++) {
    std::vector<int> lbtmp;
    std::vector<int> ubtmp;
    for (int r=0; r<int(Ltiles->size()); r++) {
      if ( (*Ltiles)[r].a[0] == (*Ltiles)[t].a[0] ) {
        if ( (*Ltiles)[r].a[1] >= (*Ltiles)[t].a[2] 
          && (*Ltiles)[r].a[1] <= (*Ltiles)[t].a[1] ) { 
          
          for (int ru=0; ru<int(Utiles->size()); ru++) {
            if ( (*Utiles)[ru].a[1] == (*Ltiles)[t].a[1] ) {
              if ( (*Utiles)[ru].a[0] >= (*Ltiles)[t].a[2] 
                && (*Utiles)[ru].a[0] <= (*Ltiles)[t].a[1] ) { 
              
                if ( (*Ltiles)[r].a[1] == (*Utiles)[ru].a[0] ) {  
              
                  printf("L %d (%d,%d) adding L %d (%d,%d)\n", 
                    t, (*Ltiles)[t].a[0], (*Ltiles)[t].a[1], 
                    r, (*Ltiles)[r].a[0], (*Ltiles)[r].a[1] );
                  lbtmp.push_back(r);
                  printf("L %d (%d,%d) adding U %d (%d,%d)\n", 
                    t, (*Ltiles)[t].a[0], (*Ltiles)[t].a[1], 
                    r, (*Utiles)[ru].a[0], (*Utiles)[ru].a[1] );
                  ubtmp.push_back(ru);
                
                }
                
              }
            }
          }
        }
      }
      
    }
    Lbatches->push_back(lbtmp);
    //std::vector<int> ubtmp;
    //for (int r=0; r<int(Utiles->size()); r++) {
    //  if ( (*Utiles)[r].a[1] == (*Ltiles)[t].a[1] ) {
    //    if ( (*Utiles)[r].a[0] >= (*Ltiles)[t].a[2] 
    //      && (*Utiles)[r].a[0] <= (*Ltiles)[t].a[1] ) { 
    //      printf("L %d (%d,%d) adding U %d (%d,%d)\n", 
    //        t, (*Ltiles)[t].a[0], (*Ltiles)[t].a[1], 
    //        r, (*Utiles)[r].a[0], (*Utiles)[r].a[1] );
    //      ubtmp.push_back(r);
    //    }
    //  }
    //  
    //}
    Lbatches->push_back(ubtmp);
  }
  
  //std::vector< std::vector<int> > Ubatches;
  for (int t=0; t<int(Utiles->size()); t++) {
    std::vector<int> lbtmp;
    std::vector<int> ubtmp;
    for (int r=0; r<int(Ltiles->size()); r++) {
      if ( (*Ltiles)[r].a[0] == (*Utiles)[t].a[0] ) {
        if ( (*Ltiles)[r].a[1] >= (*Utiles)[t].a[2] 
          && (*Ltiles)[r].a[1] <= (*Utiles)[t].a[1] ) { 
        
          //printf("U %d (%d,%d) checking L %d (%d,%d)\n", 
          //  t, (*Utiles)[t].a[0], (*Utiles)[t].a[1],
          //  r, (*Ltiles)[r].a[0], (*Ltiles)[r].a[1] );
          for (int ru=0; ru<int(Utiles->size()); ru++) {
            if ( (*Utiles)[ru].a[1] == (*Utiles)[t].a[1] ) {
              if ( (*Utiles)[ru].a[0] >= (*Utiles)[t].a[2] 
                && (*Utiles)[ru].a[0] <= (*Utiles)[t].a[0] ) {
              
                //printf("U %d (%d,%d) checking U %d (%d,%d)\n", 
                //  t, (*Utiles)[t].a[0], (*Utiles)[t].a[1],
                //  r, (*Utiles)[ru].a[0], (*Utiles)[ru].a[1] );
                if ( (*Ltiles)[r].a[1] == (*Utiles)[ru].a[0] ) { 
                  
                  printf("U %d (%d,%d) adding L %d (%d,%d)\n", 
                    t, (*Utiles)[t].a[0], (*Utiles)[t].a[1],
                    r, (*Ltiles)[r].a[0], (*Ltiles)[r].a[1] );
                  lbtmp.push_back(r);
                  printf("U %d (%d,%d) adding U %d (%d,%d)\n", 
                    t, (*Utiles)[t].a[0], (*Utiles)[t].a[1],
                    r, (*Utiles)[ru].a[0], (*Utiles)[ru].a[1] );
                  ubtmp.push_back(ru);
                  
                }
                
              }
            }  
          }
        } 
      }
      
    }
    Ubatches->push_back(lbtmp);
    //std::vector<int> ubtmp;
    //for (int r=0; r<int(Utiles->size()); r++) {
    //  if ( (*Utiles)[r].a[1] == (*Utiles)[t].a[1] ) {
    //    if ( (*Utiles)[r].a[0] >= (*Utiles)[t].a[2] 
    //      && (*Utiles)[r].a[0] <= (*Utiles)[t].a[0] ) {  
    //      printf("U %d (%d,%d) adding L %d (%d,%d)\n", 
    //        t, (*Utiles)[t].a[0], (*Utiles)[t].a[1],
    //        r, (*Utiles)[r].a[0], (*Utiles)[r].a[1] );
    //      ubtmp.push_back(r);
    //    }
    //  }
    //  
    //}
    Ubatches->push_back(ubtmp);
  }
  
  
}
