//=============================================================
// 
//  cuthill_mckee.cpp
//  
//  Applies the Cuthill-McKee Alogrithm to sort the nodes in
//  the mesh.
//
//  Written by - Shane Sawyer
//  Modified by - Chad Burdyshaw
//
//=============================================================

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "../include/sparse_types.h"
#include "../include/sparse.h" 
#include "../include/dense_types.h"


struct node
{
  int id;
  int degree;
};

extern "C" int compare_degree ( const void *a, const void *b )
{
  struct node * aa = (struct node*)a;
  struct node * bb = (struct node*)b; 

  if ( aa->degree < bb->degree )
    return 1;
  else if ( aa->degree > bb->degree )
    return -1;
  else
    return 0;
}


//=============================================================
// 
//  Cuthill_McKee()
//
//  Reorders the vertices in the mesh by applying the Cuthill-
//  McKee Algorithm as outlined in Dr. Hyam's dissertation.
//  
//
//=============================================================
extern "C"
void data_reorder_cuthill_mckee (data_d_matrix* A, int* nn_map, bool reverse)
{
  int i,j;//,k;                             // Loop counters.
  int seed;                              // Seed node to process.
  int inode;                             // The new node counter - used to renumber the nodes.
  int start_idx;                         // Starting index in an array.
  int end_idx;                           // Ending index in an array.
  int degree;                            // Degree of the node.
  int maxdegree=0;                       // Maximum nodal degree in the mesh.
  int *R;                                // The R queue that contains the nodes connected to the processing node.
  struct node *Rnode;                    // The R queue cast as collection of node objects used for sorting by degree.
  int *S;                                // The queue containing nodes that need processing.
  int *Queue;                            // The queue of unorderd nodes. A 1 or 0 is stored for every node to indicate
                                         // whether or not a node has been reordered.
  int startR=0;                          // Starting position of the R queue.
  int endR=0;                            // Ending position of the R queue.            Initially the queues are empty,
  int startS=0;                          // Starting position of the S queue.          hence the 0's.
  int endS=0;                            // Ending position of the S queue.
  // For the queues, they are empty if endX-startX is equal to 0.

  // Allocate space to store the new index of the vertices.
  int nn = A->num_rows;
  if ( nn_map == NULL ) { printf("MEMORY ERROR: COULD NOT ALLOCATE 'nn_map'.\n"); exit(1); }

  // Allocate space for Queue and initialize all nodes to 0 to indicate unordered.
  Queue = (int*)malloc((nn+1)*sizeof(int));
  if ( Queue == NULL ) { printf("MEMORY ERROR: COULD NOT ALLOCATE 'Queue'.\n"); exit(1); }

  // To save memory allocations, assume worst case scenario that all nodes will be added to R and
  // S queues at the same time.
  R = (int*)malloc(nn*sizeof(int));
  S = (int*)malloc(nn*sizeof(int));

  if ( R == NULL ) { printf("MEMORY ERROR: COULD NOT ALLOCATE 'R'.\n"); exit(1); }
  if ( S == NULL ) { printf("MEMORY ERROR: COULD NOT ALLOCATE 'S'.\n"); exit(1); }

  for ( i=0; i < nn; i++ )
    {
      Queue[i] = 0;
    }

  //nodes surrounding nodes (grid->nsn) is for each node i: A->col[A->row[i-1]] to A->col[A->row[i]]
  //number of nodes surrounding node (grid->nnsn) is for each node i: A->col[A->row[i]] - A->col[A->row[i-1]]
  int* nnsn= (int*)calloc((nn+1),sizeof(int));
  int* nsn = (int*)calloc((A->nnz),sizeof(int));

  nnsn[0]=0;
  for(i=0; i<nn; ++i)
  {
    for(j=A->row[i]; j<A->row[i+1]; ++j)
    {
      int offset=0;
      if(i>0)offset=nnsn[i-1];

      if(A->col[j] != i)
      {
        nnsn[i]++;
        nsn[offset+nnsn[i]]=A->col[j];
      }
    }  

    if(i>0) nnsn[i]+=nnsn[i-1];
  }


  // Setting the first new node to 1.
  inode = 0;

  // Choose the initial seed node as the current node '1' even though this is probably not the optimal choice.
  /*
  seed = 1;
  nn_map[seed] = inode;
  inode++;
  */

  // Search through the mesh to find the node of highest degree and choose that as the seed node.
  
  seed = 0;                          // Initialize the seed search.
  maxdegree = nnsn[1]-nnsn[0];
  degree = nnsn[0];
  
  for ( i=0; i < nn; i++ )
    {
      if(i>0)degree = nnsn[i] - nnsn[i-1];
      if ( degree > maxdegree )
      {
	maxdegree = degree;
	seed = i;
      }
    }

  // Reorder the seed node.
  nn_map[seed] = inode;
  inode++;

  // Indicate that seed has been reordered.
  Queue[seed] = 1;

  // Start the algorithm in earnest.
  while ( inode < nn )   // There are still nodes to process.
    {

      start_idx=0;
      if(inode>0) start_idx = nnsn[seed-1];
      end_idx = nnsn[seed];

      for ( i=start_idx; i < end_idx; i++ )
	{
	  if ( Queue[nsn[i]] == 0 )
	    {
	      R[endR] = nsn[i];
	      endR++;
	      // Now 'remove' the current node from Queue.
	      Queue[nsn[i]] = 1;
	    }
	}
      
      // Get the degree of the current seed vertex.                        //
      //degree = end_idx - start_idx;                                      // This is handled above.

      // Track the maximum degree.
      //maxdegree = ( degree > maxdegree ) ? degree : maxdegree ;          //

      // Now sort R so that nodes of maximum degree have lower indices (will be popped off sooner).

      // First we make an array of node struct objects and populate it with the nodes that are in the R queue.
      Rnode = (struct node*) malloc( endR*sizeof(struct node) );
      if ( Rnode == NULL ) { printf("MEMORY ERROR: COULD NOT ALLOCATE 'Rnode'.\n"); exit(1); }

      for ( i=startR; i < endR; i++ )
	{
	  Rnode[i].id = R[i];
	  Rnode[i].degree = nnsn[R[i]+1] - nnsn[R[i]];
	}

      // Sort the node objects by degree.
      qsort ( Rnode , (endR-startR), sizeof(struct node), compare_degree );

      // Copy the ordered set out to the R queue.
      for ( i=startR; i < endR; i++ )
	{
	  R[i] = Rnode[i].id;
	}

      // Free up the memory of Rnode.
      free(Rnode);

      // Renumber the nodes in the R queue and pop them off.
      while ( startR < endR )
	{
	  assert( inode < nn );
	  nn_map[R[startR]] = inode;
	  
	  // Increment the new node index.
	  inode++;
	  
	  // 'Pop' the node off and send it to the back of the S queue.
	  S[endS] = R[startR];
	  endS++;  // Move the position back.
	  startR++;  // Move the position up to 'delete' the current node from R.
	}

      // Reset the indices in R.
      startR = 0;
      endR = 0;

      // Choose the next seed node.
      seed = S[startS];
      assert( seed < A->nnz );
      // Pop off the new seed node from S.
      startS++;
    }

  // Now nn_map has the new ordering of the vertices. -> nn_map[i] = j;
  // Here, i is the old node number and j is new number for the vertex.

  // Free up memory used locally in this function and class-member data that will be reallocated by main().
  free(Queue);
  free(R);
  free(S);
  free(nnsn);
  free(nsn);

  if(reverse){
    printf("Reverse Cuthill_McKee(): Finished reordering the vertices. Maximum degree in mesh is <%d>.\n",maxdegree);
  }
  else{
    printf("Cuthill_McKee(): Finished reordering the vertices. Maximum degree in mesh is <%d>.\n",maxdegree);
  }

  // Debug code to print to screen.
#if 0
  printf("Old vertex id -> Reordered vertex id.\n");
  for ( i=0; i < nn; i++ )
    {
      printf("%d  ->  %d.\n",i,nn_map[i]);
    }
#endif
   //renumber A->row A->col
   if(reverse) //reverse numbering
   {
     int counter=nn-1;
     for(i=0; i<=nn/2; ++i)
     {
       int tmp=nn_map[counter];
       nn_map[counter]=nn_map[i];
       nn_map[i]=tmp;
       counter--;
     }
   }

  return;
}
