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
    return -1;//put aa ahead of bb
  else if ( aa->degree > bb->degree )
    return 1;//put bb ahead of aa
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
  int start_idx=0;                         // Starting index in an array.
  int end_idx=0;                           // Ending index in an array.
  int degree;                            // Degree of the node.
  int maxdegree=A->num_rows;             // Maximum nodal degree in the mesh.
  int mindegree=A->num_rows;             // Maximum nodal degree in the mesh.
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

  int* nn_map_inv;
  LACE_CALLOC(nn_map_inv,nn);
  if ( nn_map_inv == NULL ) { printf("MEMORY ERROR: COULD NOT ALLOCATE 'nn_map_inv'.\n"); exit(1); }

  // Allocate space for Queue and initialize all nodes to 0 to indicate unordered.
  //Queue = (int*)malloc((nn+1)*sizeof(int));
  LACE_CALLOC(Queue,nn+1);
  if ( Queue == NULL ) { printf("MEMORY ERROR: COULD NOT ALLOCATE 'Queue'.\n"); exit(1); }

  // To save memory allocations, assume worst case scenario that all nodes will be added to R and
  // S queues at the same time.
  LACE_CALLOC(R,nn);
  LACE_CALLOC(S,nn);
  if ( R == NULL ) { printf("MEMORY ERROR: COULD NOT ALLOCATE 'R'.\n"); exit(1); }
  if ( S == NULL ) { printf("MEMORY ERROR: COULD NOT ALLOCATE 'S'.\n"); exit(1); }

  //nodes surrounding nodes (grid->nsn) is for each node i: A->col[A->row[i-1]] to A->col[A->row[i]]
  //number of nodes surrounding node (grid->nnsn) is for each node i: A->col[A->row[i]] - A->col[A->row[i-1]]
  int* nnsn;
  int* nsn;
  LACE_CALLOC(nnsn,nn);
  LACE_CALLOC(nsn,A->nnz);

  for(i=0; i<nn; ++i){
    for(j=A->row[i]; j<A->row[i+1]; ++j){
      int offset=0;
      if(i>0){offset=nnsn[i-1];}
      if(A->col[j] != i){//add index of nodes surrounding node i to nsn list
        nsn[offset+nnsn[i]]=A->col[j];
        //printf("node=%d nnsn[%d]=%d nsn[%d]=%d\n" ,i, i,nnsn[i], offset+nnsn[i], nsn[offset+nnsn[i]]);
        nnsn[i]++;
      }
    }  
    if(i>0){nnsn[i]+=nnsn[i-1];}
    //printf("nnsn[%d]=%d\n" ,i,nnsn[i]);
  }
  
  // Setting the first new node to 1.
  inode = 0;
  
  // Search through the mesh to find the node of lowest degree and choose that as the seed node.
  
  seed = 0; // Initialize the seed search.
  mindegree = nnsn[seed];
  degree = nnsn[seed];
  
  // Start the algorithm in earnest.
  while ( inode<nn ){ // While there are still nodes to process.

    if(inode>0)
    {
      //Traverse neighboring nodes of current seed node and check to see if they have not already been
      // added to the reordered set.
      start_idx=0;
      if(seed>0) {start_idx = nnsn[seed-1];}
      end_idx = nnsn[seed];
      //printf("inode=%d seed=%d start_idx=%d end_idx=%d\n",inode, seed, start_idx, end_idx);
      for ( i=start_idx; i < end_idx; i++ ){
	//if nodes surrounding seed node not already reordered
	//printf("Queue[nsn[%d]=%d]=%d\n",i,nsn[i], Queue[nsn[i]]);
	if ( Queue[nsn[i]] == 0 ) {
	  R[endR] = nsn[i]; // Add node to reordered list
	  //printf("  1: Add node to R[%d]=%d\n",endR,R[endR]);
	  endR++;           // increment index to reordered list
	  Queue[nsn[i]] = 1;// Now 'remove' the current node from Queue.
	  //printf("  remove node %d from Q \n",nsn[i]);
	}
      }
      // Track the maximum degree.
      // Get the degree of the current seed vertex.                     
      degree = end_idx - start_idx;             
      maxdegree = ( degree > maxdegree ) ? degree : maxdegree ;
    }
    

    if(endR > startR){
      // Now sort R so that nodes of maximum degree have lower indices (will be popped off sooner).
      
      // First we make an array of node struct objects and populate it with the nodes that are in the R queue.
      //This structure in necessary to pass to qsort function to sort nodes in each row
      Rnode = (struct node*) malloc( endR*sizeof(struct node) );
      if ( Rnode == NULL ) { printf("MEMORY ERROR: COULD NOT ALLOCATE 'Rnode'.\n"); exit(1); }
      //printf("startR=%d ,endR=%d\n",startR,endR);
      for ( i=startR; i < endR; i++ ) {
	Rnode[i].id = R[i];
	Rnode[i].degree = nnsn[R[i]] - nnsn[R[i]-1];
	//printf("   Rnode[%d].id=%d Rnode[%d].degree=%d\n",i,Rnode[i].id,i,Rnode[i].degree);
      }
      // Sort the node objects by degree. Sort in ascending order
      qsort( Rnode , (endR-startR), sizeof(struct node), compare_degree );
      // Copy the ordered set out to the R queue.
      for ( i=startR; i < endR; i++ ) {
	R[i] = Rnode[i].id;
	//printf("  Add node %d to R[%d]\n",R[i],i);
      }
      free(Rnode);// Free up the memory of Rnode.
      
      // Renumber the nodes in the R queue and pop them off.
      while ( startR < endR ) {
	assert( inode < nn );
	
	//nn_map[R[startR]] = inode;
	//printf("  nn_map[%d]=%d\n",R[startR],nn_map[R[startR]]);

	//nn_map[inode] = R[startR];
	//printf("  nn_map[%d]=%d\n",inode,nn_map[inode]);
	
	nn_map_inv[inode] = R[startR];
	//printf("  nn_map[%d]=%d\n",inode,nn_map_inv[inode]);
	
	inode++;// Increment the new node index.
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
      //printf("seed=%d startS=%d endS=%d\n",seed,startS,endS);
      assert( seed < A->nnz );
      assert( seed >= 0 );
      // Pop off the new seed node from S.
      startS++;      
    }
    else{
      //printf("R is empty\n");
      //if R is empty
      //find the next seed with minimum degree
      //loop over rows and look for the node with the smallest degree
      mindegree=nn;
      for ( i=0; i < nn; i++ ){
	//printf("Queue[%d]=%d\n",i, Queue[i]);
	if(Queue[i] == 0){
	  if(i>0){degree = nnsn[i] - nnsn[i-1];}
	  //find node (row) with minumum degree
	  if ( degree < mindegree ){
	    mindegree = degree;
	    seed = i;
	    //printf("` seed=%d mindegree=%d\n",seed,mindegree);
	  }
	}
      }
      //printf("seed=%d mindegree=%d\n",seed,mindegree);
      
      //printf("  Add node %d to R[%d]\n",inode,seed);

      //nn_map[seed] = inode;
      //printf("  nn_map[%d]=%d\n",seed,nn_map[seed]);
      
      //nn_map[inode] = seed;
      nn_map_inv[inode] = seed;
      //printf("  nn_map[%d]=%d\n",inode,nn_map_inv[inode]);
      
      // Indicate that seed has been reordered.
      Queue[seed] = 1;
      inode++;
      
    }

    
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
  

  //renumber A->row A->col
  if(reverse) //reverse numbering
    {
      int counter=nn-1;
      int tmp;
      for(i=0; i<nn/2; ++i)
	{
	  //int tmp=nn_map[counter];
	  //int tmp=nn_map[i];
	  tmp=nn_map_inv[i];
	  //nn_map[counter]=nn_map[i];
	  //nn_map[i]=nn_map[counter];
	  nn_map_inv[i]=nn_map_inv[counter];
	  //nn_map[i]=tmp;
	  //nn_map[counter]=tmp;
	  nn_map_inv[counter]=tmp;
	  counter--;
	}
    }


#if 1
  //invert node map
  for ( i=0; i < nn; i++ )
    {
      nn_map[nn_map_inv[i]]=i;
      //nn_map[i]=nn_map_inv[i];
    }
#endif
  
  // Debug code to print to screen.
#if 0
  printf("Old vertex id -> Reordered vertex id.\n");
  for ( i=0; i < nn; i++ )
    {
      //printf("%d  ->  %d.\n",nn_map[i],i);
      printf("%d  ->  %d.\n",nn_map_inv[i],i);
      //printf("%d  ->  %d.\n",i,nn_map[i]);
    }
#endif
  
  return;
}
