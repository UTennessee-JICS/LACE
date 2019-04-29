#include "../include/sparse.h"
#include "amd.h"//for SparseSuite AMD reordering function
#include <stdio.h>
#include <stdlib.h>

void print_char_Matrix(int* Ap, int* Ai, int n)
{
#if 0
   printf ("\nInput matrix:  %d-by-%d, with %d entries.\n"
          "   Note that for a symmetric matrix such as this one, only the\n"
          "   strictly lower or upper triangular parts would need to be\n"
          "   passed to AMD, since AMD computes the ordering of A+A'.  The\n"
          "   diagonal entries are also not needed, since AMD ignores them.\n"
          , n, n, Ap[n]) ;
   for (int j = 0 ; j < n ; j++)
   {
       printf ("\nColumn: %d, number of entries: %d, with row indices in"
               " Ai [%d ... %d]:\n    row indices:",
               j, Ap [j+1] - Ap [j], Ap [j], Ap [j+1]-1) ;
       for (int p = Ap [j] ; p < Ap [j+1] ; p++)
       {
           int i = Ai [p] ;
           printf (" %d", i) ;
       }
       printf ("\n") ;
   }
#endif

    char Ac[n][n];
    /* print a character plot of the input matrix.  This is only reasonable because the matrix is small. */
    printf ("\nPlot of input matrix pattern:\n") ;
    for (int j = 0 ; j < n ; j++)
    {
        for (int i = 0 ; i < n ; i++) Ac [i][j] = '.' ;
        for (int p = Ap [j] ; p < Ap [j+1] ; p++)
        {
            int i = Ai [p] ;
            Ac [i][j] = 'X' ;
        }
    }
    printf ("    ") ;
    for (int j = 0 ; j < n ; j++) printf (" %1d", j % 10) ;
    printf ("\n") ;
    for (int i = 0 ; i < n ; i++)
    {
        printf ("%2d: ", i) ;
        for (int j = 0 ; j < n ; j++)
        {
            printf (" %c", Ac [i][j]) ;
        }
        printf ("\n") ;
    }
};


void print_char_permuted_Matrix(int* Ap, int* Ai, int* P, int* Pinv, int n)
{
  char Ac[n][n];
  int i,j,inew,jnew,p;

  for (jnew = 0 ; jnew < n ; jnew++)
  {
    j = P [jnew] ;
    for (inew = 0 ; inew < n ; inew++) Ac [inew][jnew] = '.' ;
    for (p = Ap [j] ; p < Ap [j+1] ; p++)
    {
      inew = Pinv [Ai [p]] ;
      Ac [inew][jnew] = 'X' ;
    }
  }
  printf ("    ") ;
  for (j = 0 ; j < n ; j++) printf (" %1d", j % 10) ;
  printf ("\n") ;
  for (i = 0 ; i < n ; i++)
  {
    printf ("%2d: ", i) ;
    for (j = 0 ; j < n ; j++)
    {
      printf (" %c", Ac [i][j]) ;
    }
    printf ("\n") ;
  }
};


void reorder_csr_indices(data_d_matrix* A, int* P, int* Pinv)
{
  //reorder matrix indices
  int* ia = A->row;
  int* ja = A->col;

  int* ia_new;
  dataType* val_new;
  int* ja_new;
  int ldblock=A->ldblock;

  LACE_CALLOC(ia_new, A->num_rows+1);
  LACE_CALLOC(ja_new,A->nnz);
  LACE_CALLOC(val_new, A->nnz*ldblock);

  ia_new[0]=0;
  /*compute new index offsets for ia2*/
  for(int i=0; i<A->num_rows; ++i)
  {
    int count=0;
    int old_row = Pinv[i];
    for(int j=ia[old_row]; j<ia[old_row+1]; ++j)
    { 
      //ja_new[ia_new[i]+count] = Pinv[ja[j]];//assign new column index
      ja_new[ia_new[i]+count] = P[ja[j]];//assign new column index
      for(int kk=0;kk<ldblock;++kk){
	val_new[(ia_new[i]+count)*ldblock+kk]=A->val[j*ldblock+kk];
      }
      count++;
    }
    ia_new[i+1]=ia_new[i]+count;
  }

  //eliminate old data and replace with new
  free(A->row);
  A->row = ia_new;
  free(A->col);
  A->col = ja_new;
  free(A->val);
  A->val = val_new;
};

//for use in qsort, defines a tuple with matrix col index and value
typedef struct
{
   int index;
   dataType* val;
} element;

//function passed to qsort to compare matrix elements in row
int comparator(const void *p, const void *q)
{
   element *element_p = (element*)p;
   element *element_q = (element*)q;
   return ( element_p->index - element_q->index );
};

//use qsort to take reordered rows and sort by col index number
extern "C"
void sort_csr_rows(data_d_matrix* A)
{
  int ldblock = A->ldblock;
  element* current_row;
  current_row = (element*)calloc(A->num_rows,sizeof(element)); 

  for(int i=0; i<A->num_rows; ++i){
    //load row into entries
    int count=0;
    for(int j=A->row[i]; j<A->row[i+1]; ++j){
       current_row[count].val = (dataType*)calloc(ldblock,sizeof(dataType));
       current_row[count].index=A->col[j];
       for(int kk=0;kk<ldblock;++kk)
	 current_row[count].val[kk]=A->val[j*ldblock+kk];
       count++;
    }

    //sort row
    qsort(current_row, count, sizeof(element), comparator);

    //load sorted values back into array
    count=0;
    for(int j=A->row[i];j<A->row[i+1];++j){
       A->col[j]=current_row[count].index;
       for(int kk=0;kk<ldblock;++kk)
	 {A->val[j*ldblock+kk]=current_row[count].val[kk];}
       free(current_row[count].val);
       count++;
    }
  }
  free(current_row);
};

extern "C"
int data_sparse_reorder(data_d_matrix* A, int* P, int* Pinv, int reorder)
{
  //reorder A

  int* Ap;//[A->num_rows+1];
  int* Ai;//[A->nnz];
  Ap = A->row;
  Ai = A->col;

  /* print the input matrix */
  int n = A->num_rows;
  //int nz = Ap [n] ;
#if 0
  /* print a character plot of the original matrix. */
  printf ("\nPlot of original matrix pattern:\n") ;
  print_char_Matrix(Ap, Ai, n);
#endif
  
  switch (reorder)
  {
     case 1:{ 
        //Cuthill-McKee
        data_reorder_cuthill_mckee(A, P);
     }break;

     case 2:{
        //Reverse Cuthill-McKee
        bool reverse=true;
        data_reorder_cuthill_mckee(A, P, reverse);
     }break;

     case 3:{ 
        //AMD 
        /* get the default parameters, and print them */
        double Control[AMD_CONTROL];
        double Info[AMD_INFO];
        amd_defaults (Control) ;
        amd_control  (Control) ;
DEV_CHECKPT
        int result = amd_order(A->num_rows, Ap, Ai, P, Control, Info);
DEV_CHECKPT
        amd_control(Control);
        //int status = amd_valid(n,n,Ap,Ai);
        //printf ("return value from amd_order: %d (should be %d)\n", result, AMD_OK) ;
        /* print the statistics */
        amd_info (Info);
        if (result != AMD_OK)
        {
           printf ("AMD failed\n") ;
           exit (1) ;
        }
     }break;

     default:{//no reordering
        for(int k=0;k<n;++k){
           P[k] = k ;
           Pinv[k] = k ;
        }
     }
  }
 
  //compute inverse permutation vector
  for (int k = 0 ; k < n ; k++)
  {
    /* row/column j is the kth row/column in the permuted matrix */
     Pinv[P[k]] = k ;
  }

 #if 0
  for (int k = 0 ; k < n ; k++)
  {
    /* row/column j is the kth row/column in the permuted matrix */
    //printf("P[%d]=%d\n",k,P[k]);
     printf("Pinv[%d]=%d\n",k,Pinv[k]);
  }
#endif
  
  //reorder the indices for A->row A->col from permutation vector P
DEV_CHECKPT
  reorder_csr_indices(A,P,Pinv);
  //sort the cols and accociated vals in increasing order for each row
DEV_CHECKPT
  sort_csr_rows(A);
DEV_CHECKPT

#if 0
  /* print a character plot of the permuted matrix. */
  printf ("\nPlot of reordered matrix pattern:\n") ;
  print_char_Matrix(A->row, A->col, A->num_rows);
#endif
  //print actual values of matrix
#if 0
  for(int i=0;i<A->num_rows;++i){
     printf("\nrow[%d]: ",i);
     for(int j=A->row[i];j<A->row[i+1];++j){
        printf(" %d:%2.2e",A->col[j],A->val[j*A->ldblock]);
     }
    printf("\n");
  }
#endif

  return(0);
};


