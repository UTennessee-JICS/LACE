#include "../include/sparse.h"
#include "amd.h"


static void print_char_orig_Matrix(int* Ap, int* Ai, int n)
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


static void print_char_Matrix(int* Ap, int* Ai, int* P, int* Pinv, int n)
{
  char Ac[n][n];
  int i,j,k,inew,jnew,p;

  //compute inverse permutation vector
  for (k = 0 ; k < n ; k++)
  {
    /* row/column j is the kth row/column in the permuted matrix */
    j = P [k] ;
    Pinv [j] = k ;
  }

#if 0
  /* print the permutation vector, P, and compute the inverse permutation */
  printf ("Permutation vector:\n") ;
  for (k = 0 ; k < n ; k++)
  {
    /* row/column j is the kth row/column in the permuted matrix */
    j = P [k] ;
    printf (" %2d", j) ;
  }
  printf ("\n\n") ;
  printf ("Inverse permutation vector:\n") ;
  for (int j = 0 ; j < n ; j++)
  {
    int k = Pinv [j] ;
    printf (" %2d", k) ;
  }
  printf ("\n\n") ;
#endif

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



int data_sparse_reorder(data_d_matrix* A)
{
  //reorder A
  int* Ap;//[A->num_rows+1];
  int* Ai;//[A->nnz];
  int P[A->num_rows];
  int Pinv[A->num_rows];
  Ap = A->row;
  Ai = A->col;

  /* print the input matrix */
  int n = A->num_rows;
  //int nz = Ap [n] ;
  /* print a character plot of the original matrix. */
  printf ("\nPlot of original matrix pattern:\n") ;
  print_char_orig_Matrix(Ap, Ai, n);


//#if 0 //AMD  
  /* get the default parameters, and print them */
  double Control[AMD_CONTROL];
  double Info[AMD_INFO];
  amd_defaults (Control) ;
  amd_control  (Control) ;

  int result = amd_order(A->num_rows, Ap, Ai, P, Control, Info);
  amd_control(Control);
  //int status = amd_valid(n,n,Ap,Ai);
  printf ("return value from amd_order: %d (should be %d)\n", result, AMD_OK) ;
  /* print the statistics */
  amd_info (Info) ;
  if (result != AMD_OK)
  {
    printf ("AMD failed\n") ;
    exit (1) ;
  }
  /* print a character plot of the permuted matrix. */
  printf ("\nPlot of permuted AMD matrix pattern:\n") ;
  print_char_Matrix(Ap, Ai, P, Pinv, n);
//endif


//#if 0 //Cuthill-McKee
   //int* nn_map=(int*)calloc(n+1,sizeof(int));
   data_reorder_cuthill_mckee(A, P);
   printf("Old vertex id -> Reordered vertex id.\n");
   for (int  i=0; i < n; i++ )
   {
     printf("%d  ->  %d.\n",i,P[i]);
   }
  /* print a character plot of the permuted matrix. */
  printf ("\nPlot of permuted Cuthill Mckee matrix pattern:\n") ;
  print_char_Matrix(Ap, Ai, P, Pinv, n);
//#endif


//#if 0 //Reverse Cuthill-McKee
   //int* nn_map=(int*)calloc(n+1,sizeof(int));
   bool reverse=true;
   data_reorder_cuthill_mckee(A, P, reverse);
   printf("Old vertex id -> Reordered vertex id.\n");
   for (int  i=0; i < n; i++ )
    {
      printf("%d  ->  %d.\n",i,P[i]);
    }
  /* print a character plot of the permuted matrix. */
  printf ("\nPlot of permuted Reverse Cuthill Mckee matrix pattern:\n") ;
  print_char_Matrix(Ap, Ai, P, Pinv, n);
//#endif




//  /* print a character plot of the permuted matrix. */
//  printf ("\nPlot of permuted matrix pattern:\n") ;
//  print_char_Matrix(Ap, Ai, P, Pinv, n);
  return(0);
};


