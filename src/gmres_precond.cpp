
#include "../include/sparse.h"
#include <mkl.h>
#include <math.h> 
#include <float.h> 
#include <stdlib.h>
#include <stdio.h>

/**
    Purpose
    -------

    Solves a system of linear equations
       A * X = B
    where A is a complex sparse matrix stored in the GPU memory.
    X and B are complex vectors stored on the GPU memory.
    This is a CPU implementation of a basic GMRES based on the discriptions in
    Saad, Yousef. Iterative methods for sparse linear systems. 
    Society for Industrial and Applied Mathematics, 2003.
    and
    Kelley, C. T. Iterative methods for linear and nonlinear equations. 
    Society for Industrial and Applied Mathematics, Philadelphia, 1995. 
    MR 96d 65002.
    

    Arguments
    ---------

    @param[in]
    A           data_d_matrix*
                descriptor for matrix A

    @param[in]
    b           data_d_matrix*
                RHS b vector

    @param[in,out]
    x           data_d_matrix*
                solution approximation
                
    @param[in]
    L           data_d_matrix*
                descriptor for matrix A
                
    @param[in]
    U           data_d_matrix*
                descriptor for matrix A

    @param[in,out]
    solver_par  data_d_solver_par*
                solver parameters

    @param[in]
    precond_par data_d_preconditioner*
                preconditioner

    @ingroup datasparse_linsolvers
    ********************************************************************/

    
    
extern "C" 
int
data_gmres_precond(
    data_d_matrix *A, data_d_matrix *b, data_d_matrix *x0,
    data_d_matrix *L, data_d_matrix *U,
    data_d_gmres_param *gmres_par,
    data_d_gmres_log *gmres_log )
{
    printf("%% data_gmres_precond begin\n");
    dataType wstart = omp_get_wtime();
    dataType zero = 0.0;
    dataType one = 1.0;
    dataType negone = -1.0;
    
    // initialize
    data_int_t info = DEV_NOTCONVERGED;
    data_int_t n = A->num_rows;
    data_int_t search_max = gmres_par->search_max;
    data_int_t search_directions = 0;
    dataType rtol = gmres_par->rtol;
    dataType rnorm2 = 0.0;
    dataType bnorm = 0.0;
    dataType beta = 0.0;
    dataType delta = 0.0;
    dataType gamma = 0.0;
    data_int_t search = 0; // search directions
    
    // preconditioning
    // for mkl_dcsrtrsv 
    char cvar, cvar1, cvar2;
    data_d_matrix LU = {Magma_CSR};
    data_zmlumerge( *L, *U, &LU );
    DEV_CHECKPT;
    int* ia;
    int* ja;
    LACE_CALLOC( ia, (LU.num_rows+1) );
    LACE_CALLOC( ja, LU.nnz );
    
    #pragma omp parallel 
    {
      #pragma omp for nowait
      for (int i=0; i<LU.num_rows+1; i++) {
      	ia[i] = LU.row[i] + 1;	
      }
      #pragma omp for nowait
      for (int i=0; i<LU.nnz; i++) {
      	ja[i] = LU.col[i] + 1;
      }
    }
    
    data_d_matrix tmp={Magma_DENSE};
    data_zvinit( &tmp, n, 1, zero );
    data_d_matrix Minvvj={Magma_DENSE};
    data_zvinit( &Minvvj, n, 1, zero );
    
    // alocate solution and residual vectors 
    data_d_matrix x={Magma_DENSE};
    data_zmconvert( *x0, &x, Magma_DENSE, Magma_DENSE );
    data_d_matrix r={Magma_DENSE};
    data_zvinit( &r, n, 1, zero );
    
    // Kyrylov subspace 
    data_d_matrix krylov={Magma_DENSE};
    data_zvinit( &krylov, n, search_max+1, zero );
    krylov.major = MagmaColMajor;
    data_d_matrix z={Magma_DENSE};
    data_zvinit( &z, n, 1, zero );
    
    // Reorthogonalize
    dataType eps = nextafter(0.0,1.0);
    dataType normav = 0.0;
    dataType normav2 = 0.0;
    dataType hr = 0.0;
    
    // Hessenberg Matrix for Arnoldi iterations
    data_d_matrix h={Magma_DENSE};
    data_zvinit( &h, search_max+1, search_max, zero );
    h.major = MagmaColMajor;
    data_d_matrix u={Magma_DENSE};
    data_zvinit( &u, n, 1, zero );
    
    // Givens rotation vectors
    data_d_matrix givens={Magma_DENSE};
    data_zvinit( &givens, search_max+1, 1, zero );
    givens.major = MagmaColMajor;
    data_d_matrix givens_cos={Magma_DENSE};
    data_zvinit( &givens_cos, search_max, 1, zero );
    givens_cos.major = MagmaColMajor;
    data_d_matrix givens_sin={Magma_DENSE};
    data_zvinit( &givens_sin, search_max, 1, zero );
    givens_sin.major = MagmaColMajor;
    
    // Coefficents for update to solution vector
    data_d_matrix alpha={Magma_DENSE};
    data_zvinit( &alpha, search_max, 1, zero );
    alpha.major = MagmaColMajor;
    
    // initial residual
    data_z_spmv( negone, A, &x, zero, &r );
    data_zaxpy( n, one, b->val, 1, r.val, 1);
    rnorm2 = data_dnrm2( n, r.val, 1 );
    printf("rnorm2 = %e; tol = %e; rtol = %e;\n", rnorm2, rtol, rtol*rnorm2 );
    if ( gmres_par->tol_type == 1 ) {
      rtol = rtol*rnorm2;
    }
    if (rnorm2 < rtol ) {
      info = 0;
      return info;
    }
    
    // fill first column of Kylov subspace for Arnoldi iteration
    for ( int i=0; i<n; i++ ) {
      krylov.val[idx(i,0,krylov.ld)] = r.val[i]/rnorm2;
    }
    givens.val[0] = rnorm2;
    
    search = 0;
    
    gmres_log->search_directions = search_directions;
    gmres_log->solve_time = 0.0;
    gmres_log->initial_residual = rnorm2;
    
    // GMRES search direction
    //while ( (rnorm2 > rtol) && (search < search_max) ) {
    for ( int search = 0; search < search_max; search++ ) { 
     
      data_zmfree( &u );
      data_zvinit( &u, n, 1, zero );
      data_zmfree( &tmp );
      data_zvinit( &tmp, n, 1, zero );
      data_zmfree( &Minvvj );
      data_zvinit( &Minvvj, n, 1, zero );
      
      for ( int i=0; i<krylov.ld; i++ ) {
        GMRESDBG("\tkrylov.val[idx(%d,%d,%d)] = %e\n", i, search, krylov.ld, krylov.val[idx(i,search,krylov.ld)]);
      }
      
      // Apply preconditioner to krylov.val[idx(A->col[j],search,krylov.ld)]
      cvar1='L';
		  cvar='N';
		  cvar2='U';
		  mkl_dcsrtrsv( &cvar1, &cvar, &cvar2, &n, LU.val, ia, ja,
		    &(krylov.val[idx(0,search,krylov.ld)]), tmp.val );
		  cvar1='U';
		  cvar='N';
		  cvar2='N';
		  mkl_dcsrtrsv( &cvar1, &cvar, &cvar2, &n, LU.val, ia, ja, 
		    tmp.val, Minvvj.val );
		  
      for ( int i=0; i<Minvvj.ld; i++ ) {
        GMRESDBG("Minvvj.val[%d] = %e\n", i, Minvvj.val[i]);
      }
      
      //mkl_dcsrmv( "N", &A->num_rows, &A->num_cols,
      //                  &one, "GFNC", A->val,
      //                  A->col, A->row, A->row+1,
      //                  &(krylov.val[idx(0,search,krylov.ld)]), &zero, 
      //                  u.val );
      
      for ( int i=0; i<n; i++ ) {
        for ( int j=A->row[i]; j<A->row[i+1]; j++ ) {
          u.val[i] = u.val[i] + A->val[j]*Minvvj.val[A->col[j]]; 
        }
      }
      normav = data_dnrm2( n, u.val, 1 );
      
      for ( int i=0; i<u.ld; i++ ) {
        GMRESDBG("u.val[%d] = %e\n", i, u.val[i]);
      }
      for ( int j=0; j <= search; j++ ) {
        for ( int i=0; i<krylov.ld; i++ ) {
          GMRESDBG("krylov.val[idx(%d,%d,%d)] = %e\n", i, j, krylov.ld, krylov.val[idx(i,j,krylov.ld)]);
        }
      }
      
      // Modified Gram-Schmidt
      for ( int j=0; j <= search; j++ ) {
        h.val[idx(j,search,h.ld)] = 0.0;
        for ( int i=0; i<n; i++ ) {
          h.val[idx(j,search,h.ld)] = h.val[idx(j,search,h.ld)] +
            krylov.val[idx(i,j,krylov.ld)]*u.val[i];
        }
        for ( int i=0; i<n; i++ ) {
          u.val[i] = u.val[i] 
            - h.val[idx(j,search,h.ld)]*krylov.val[idx(i,j,krylov.ld)];
          GMRESDBG("\tu.val[%d] = %e\n", i, u.val[i]);  
        }
      }
      h.val[idx((search+1),search,h.ld)] = data_dnrm2( n, u.val, 1 );
      normav2 = h.val[idx((search+1),search,h.ld)];
      
      GMRESDBG("h.val[idx(search,search,h.ld)] =%e\n", h.val[idx(search,search,h.ld)]);
      GMRESDBG("h.val[idx((search+1),search,h.ld)] =%e\n", h.val[idx((search+1),search,h.ld)]);  
      
      // Reorthogonalize?
      hr = (normav + 0.001*normav2) - normav;
      if ( ( gmres_par->reorth == 0 && hr <= eps ) || gmres_par->reorth == 2 ) {
        printf("Reorthogonalize %d\n", search);
        for ( int j=0; j <= search; j++ ) {
          hr = 0.0;
          for ( int i=0; i<n; i++ ) {
            hr = hr + krylov.val[idx(i,j,krylov.ld)]*u.val[i];
          }
          h.val[idx(j,search,h.ld)] = h.val[idx(j,search,h.ld)] + hr;
          for ( int i=0; i<n; i++ ) {
            u.val[i] = u.val[i] - hr*krylov.val[idx(i,j,krylov.ld)];
          }
        }
        h.val[idx((search+1),search,h.ld)] = data_dnrm2( n, u.val, 1 );
      }
      
      // Watch out for happy breakdown
      if ( fabs(h.val[idx((search+1),search,h.ld)]) > 0 ) {
         for ( int i=0; i<n; i++ ) {
          krylov.val[idx(i,(search+1),krylov.ld)] = 
            u.val[i]/h.val[idx((search+1),search,h.ld)];
          GMRESDBG("--\tu.val[%d] = %e\n", i, u.val[i]);
          GMRESDBG("--\tkrylov.val[idx(%d,%d,%d)] = %e\n", i,(search+1),krylov.ld, krylov.val[idx(i,(search+1),krylov.ld)]);
        }
      }
      else {
        printf("%%\t******* happy breakdown **********\n"); 
      }
      
      // Givens rotations
      for ( int j = 0; j<search; j++ ) {
        delta = h.val[idx(j,search,h.ld)];
        h.val[idx(j,search,h.ld)] = 
          givens_cos.val[j]*delta 
          - givens_sin.val[j]*h.val[idx(j+1,search,h.ld)]; 
        h.val[idx(j+1,search,h.ld)] =
          givens_sin.val[j]*delta 
          + givens_cos.val[j]*h.val[idx(j+1,search,h.ld)]; 
      }
      
      gamma = sqrt( 
        h.val[idx(search,search,h.ld)]*h.val[idx(search,search,h.ld)] +
        h.val[idx((search+1),search,h.ld)]*h.val[idx((search+1),search,h.ld)] );
      GMRESDBG("gamma = %e\n", gamma);
      if ( gamma > 0.0 ) {
        givens_cos.val[search] = h.val[idx(search,search,h.ld)]/gamma;
        givens_sin.val[search] = -h.val[idx((search+1),search,h.ld)]/gamma;
        h.val[idx(search,search,h.ld)] = 
          givens_cos.val[search]*h.val[idx(search,search,h.ld)] -
          givens_sin.val[search]*h.val[idx((search+1),search,h.ld)];
        h.val[idx((search+1),search,h.ld)] = 0.0;
        delta = givens.val[search];
        givens.val[search] = 
          givens_cos.val[search]*delta - givens_sin.val[search]*givens.val[(search+1)];
        givens.val[(search+1)] = 
          givens_sin.val[search]*delta + givens_cos.val[search]*givens.val[(search+1)];
      }
      for ( int j=0; j <search_max; j++ ) {
        for ( int i=0; i<h.ld; i++ ) {
          GMRESDBG("h.val[idx(%d,%d,%d)] = %e\n", i, j, h.ld, h.val[idx(i,j,h.ld)]);
        }
        GMRESDBG("\n");
      }
      for ( int i=0; i<search_max; i++ ) {
        GMRESDBG("c.val[%d] = %e\n", i, givens_cos.val[i]);
      }
      for ( int i=0; i<search_max; i++ ) {
        GMRESDBG("s.val[%d] = %e\n", i, givens_sin.val[i]);
      }
      for ( int i=0; i<search_max+1; i++ ) {
        GMRESDBG("g.val[%d] = %e\n", i, givens.val[i]);
      }
      
      //printf("======= GMRES search %d fabs(givens.val[(%d+1)]) = %.16e =======\n", search, search, fabs(givens.val[(search+1)]));  
      printf("PGMRES_search(%d) = %.16e;\n", search+1, fabs(givens.val[(search+1)]));  
      // update the solution
      // solve the least squares problem
      if ( fabs(givens.val[(search+1)]) < rtol  || (search == (search_max-1)) ) {
        GMRESDBG(" !!!!!!! update the solution !!!!!!!\n");
        for ( int i = 0; i <= search; i++ ) {
          alpha.val[i] = givens.val[i]/h.val[idx(i,i,h.ld)];
        }
        for ( int j = search; j > 0; j-- ) {
          for (int i = j-1; i > -1; i-- ) {
            alpha.val[i] = alpha.val[i] 
             - h.val[idx(i,j,h.ld)]*alpha.val[j]/h.val[idx(i,i,h.ld)];
          }
        } 
        
        // use preconditioned vectors to form the update (GEMV)
        for (int i = 0; i < n; i++ ) { 
          for (int j = 0; j <= search; j++ ) {
            z.val[i] = z.val[i] + krylov.val[idx(i,j,krylov.ld)]*alpha.val[j]; 
          }
        }
        
        // apply preconditioner to z before updating x
        cvar1='L';
		    cvar='N';
		    cvar2='U';
		    mkl_dcsrtrsv( &cvar1, &cvar, &cvar2, &n, LU.val, ia, ja,
		      z.val, tmp.val );
		    cvar1='U';
		    cvar='N';
		    cvar2='N';
		    mkl_dcsrtrsv( &cvar1, &cvar, &cvar2, &n, LU.val, ia, ja, 
		      tmp.val, Minvvj.val );
		    
        for (int i = 0; i < n; i++ ) {
          x.val[i] = x.val[i] + Minvvj.val[i];
        }
        
        gmres_log->search_directions = search+1;
        dataType wend = omp_get_wtime();
        gmres_log->solve_time = (wend-wstart);
        gmres_log->final_residual = fabs(givens.val[(search+1)]);
        
        break;
      }
        
    }
    
    
    data_zmconvert( x, x0, Magma_DENSE, Magma_DENSE );
    
    if (gmres_log->final_residual > rtol) {
      info = 0;
    }
    
    data_zmfree( &x );
    data_zmfree( &r );
    data_zmfree( &krylov );
    data_zmfree( &h );
    data_zmfree( &u );
    data_zmfree( &givens );
    data_zmfree( &givens_cos );
    data_zmfree( &givens_sin );
    data_zmfree( &alpha );
    
    data_zmfree( &LU );
    free( ia );
    free( ja );
    data_zmfree( &tmp );
    data_zmfree( &Minvvj );
    
    return info;
} 
