//#define DEBUG_GMRES
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
data_fgmres_householder(
    data_d_matrix *A, data_d_matrix *b, data_d_matrix *x0,
    data_d_matrix *L, data_d_matrix *U,
    data_d_gmres_param *gmres_par,
    data_d_gmres_log *gmres_log )
{

    printf("%% data_fgmres_householder begin\n");
    dataType wstart = omp_get_wtime();
    dataType zero = 0.0;
    dataType one = 1.0;
    dataType negone = -1.0;

    // initialize
    data_int_t info = DEV_NOTCONVERGED;
    const data_int_t n = A->num_rows;
    data_int_t search_max = gmres_par->search_max;
    data_int_t search_directions = 0;
    dataType rtol = gmres_par->rtol;
    dataType rnorm2 = 0.0;
    dataType residual = 0.0;

    // strip-mining for efficient memory access
    const int STRIP = 1024;
    const int BINS = (n/STRIP);
    const int endStrip = BINS*STRIP;
    int startStrip = 0;
    // dataType *sumTemp;
    // LACE_CALLOC( sumTemp, BINS );
    dataType sumTemp[BINS] __attribute__((aligned(64)));

    // preconditioning
    // for mkl_dcsrtrsv
    char cvar, cvar1, cvar2;
    data_d_matrix LU = {Magma_CSR};
    data_zmlumerge( *L, *U, &LU );
    int* ia;
    int* ja;
    LACE_CALLOC( ia, (LU.num_rows+1) );
    LACE_CALLOC( ja, LU.nnz );

    #pragma omp parallel
    {
      #pragma omp for simd schedule(simd: static) nowait
      #pragma vector aligned
      #pragma vector vecremainder
      #pragma nounroll_and_jam
      for (int i=0; i<LU.num_rows+1; i++) {
      	ia[i] = LU.row[i] + 1;
      }
      #pragma omp for simd schedule(simd: static) nowait
      #pragma vector aligned
      #pragma vector vecremainder
      #pragma nounroll_and_jam
      for (int i=0; i<LU.nnz; i++) {
      	ja[i] = LU.col[i] + 1;
      }
    }

    // store preconditioned vectors for flexibility
    data_d_matrix Minvvj={Magma_DENSE};
    data_zvinit( &Minvvj, n, search_max, zero );

    // Partrsv
    dataType ptrsv_tol = 1.0; //1.0e-10;
    int ptrsv_iter = 0;

    // alocate solution and residual vectors
    data_d_matrix x={Magma_DENSE};
    data_zmconvert( *x0, &x, Magma_DENSE, Magma_DENSE );
    data_d_matrix r={Magma_DENSE};
    data_zvinit( &r, n, 1, zero );
    data_d_matrix q={Magma_DENSE};
    data_zvinit( &q, n, 1, zero );

    // Krylov subspace
    data_d_matrix krylov={Magma_DENSE};
    data_zvinit( &krylov, n, search_max+1, zero );
    krylov.major = MagmaColMajor;
    data_d_matrix precondq={Magma_DENSE};
    data_zvinit( &precondq, n, search_max+1, zero );
    precondq.major = MagmaColMajor;

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
    dataType eps = nextafter(0.0,1.0);

    // Memory alignment hints for optimized access
    #if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
    	/* GNU GCC/G++. --------------------------------------------- */
      printf("GNU COMPILER\n");
    #endif
    #if (defined(__INTEL_COMPILER) || defined(__ICC))
    	/* INTEL ICC/C++. --------------------------------------------- */
      printf("INTEL COMPILER\n");
      __assume_aligned( r.val, 64 ); // used for initial residual,
                                     // preconditioner application,
                                     // and solution update
      __assume_aligned( q.val, 64 ); // reinitialized each search direction
      //__assume_aligned( sumTemp, 64 ); // strip-mining summation bins
      __assume_aligned( krylov.val, 64 );  // Householder transformed search space
      __assume_aligned( precondq.val, 64 ); // Search vectors
      __assume_aligned( givens.val, 64 ); // Residual approximation
      __assume_aligned( givens_cos.val, 64 ); // Rotation Cosine Coefficents
      __assume_aligned( givens_sin.val, 64 ); // Rotation Sine Coefficents
    #endif

    // initial residual
    //data_z_spmv( negone, A, &x, zero, &r );
    #pragma omp parallel
    #pragma omp for schedule(monotonic:static) nowait
    #pragma vector aligned
    #pragma vector vecremainder
    #pragma nounroll_and_jam
    for ( int i=0; i<n; i++ ) {
      #pragma nounroll_and_jam
      for ( int j=A->row[i]; j<A->row[i+1]; j++ ) {
        r.val[i] = r.val[i] - A->val[j]*x.val[ A->col[j] ];
      }
    }
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
    dataType dd = 0.0;
    dataType k1norm2 = 0.0;
    #pragma omp parallel
    {
      #pragma omp for schedule(monotonic:static) nowait
      for ( int ii=startStrip; ii<endStrip; ii+=STRIP ) {
        #pragma omp simd
        #pragma vector aligned
        #pragma loop_count max(STRIP)
        for ( int i=ii; i<ii+STRIP; ++i ) {
          krylov.val[idx(i,0,krylov.ld)] = r.val[i];
        }
      }
      #pragma omp single
      #pragma omp simd
      #pragma vector aligned
      #pragma loop_count max(STRIP)
      for ( int i=endStrip; i<n; ++i ) {
        krylov.val[idx(i,0,krylov.ld)] = r.val[i];
      }

      #pragma omp barrier
      #pragma omp single
      {
        dd = mysgn(krylov.val[idx(0,0,krylov.ld)])*rnorm2;
        krylov.val[idx(0,0,krylov.ld)] = krylov.val[idx(0,0,krylov.ld)] + dd;
        k1norm2 = 1.0/data_dnrm2( n, krylov.val, 1 );
      }
      #pragma omp barrier
      #pragma omp for schedule(monotonic:static) nowait
      for ( int ii=startStrip; ii<endStrip; ii+=STRIP ) {
        #pragma omp simd
        #pragma vector aligned
        #pragma loop_count max(STRIP)
        for ( int i=ii; i<ii+STRIP; ++i ) {
          krylov.val[idx(i,0,krylov.ld)] = krylov.val[idx(i,0,krylov.ld)]*k1norm2;
        }
      }
      #pragma omp single
      #pragma omp simd
      #pragma vector aligned
      #pragma loop_count max(STRIP)
      for ( int i=endStrip; i<n; ++i ) {
        krylov.val[idx(i,0,krylov.ld)] = krylov.val[idx(i,0,krylov.ld)]*k1norm2;
      }
      #pragma omp barrier
      #pragma omp for schedule(monotonic:static) nowait
      for ( int ii=startStrip; ii<endStrip; ii+=STRIP ) {
        #pragma omp simd
        #pragma vector aligned
        #pragma loop_count max(STRIP)
        for ( int i=ii; i<ii+STRIP; ++i ) {
          q.val[i] = -r.val[i]/dd;
          precondq.val[idx(i,0,precondq.ld)] = q.val[i];
        }
      }
      #pragma omp single
      #pragma omp simd
      #pragma vector aligned
      #pragma loop_count max(STRIP)
      for ( int i=endStrip; i<n; ++i ) {
        q.val[i] = -r.val[i]/dd;
        precondq.val[idx(i,0,precondq.ld)] = q.val[i];
      }
    }
    for ( int i=0; i<n; ++i ) {
      GMRESDBG("q.val[%d] = %e\n", i, q.val[i]);
    }
    givens.val[0] = -dd;

    gmres_log->search_directions = search_directions;
    gmres_log->solve_time = 0.0;
    gmres_log->initial_residual = rnorm2;


    // GMRES search direction
    for ( int search = 0; search < search_max; ++search ) {
      int search1 = search + 1;
      data_zmfree( &r );
      data_zvinit( &r, n, 1, zero );

      for ( int i=0; i<krylov.ld; i++ ) {
        GMRESDBG("\tkrylov.val[idx(%d,%d,%d)] = %e\n", i, search, krylov.ld, krylov.val[idx(i,search,krylov.ld)]);
      }

      if ( gmres_par->user_csrtrsv_choice == 0 ) {
        // Apply preconditioner to krylov.val[idx(A->col[j],search,krylov.ld)]
        cvar1='L';
		    cvar='N';
		    cvar2='U';
		    mkl_dcsrtrsv( &cvar1, &cvar, &cvar2, &n, LU.val, ia, ja,
		      q.val, r.val );
		    cvar1='U';
		    cvar='N';
		    cvar2='N';
		    mkl_dcsrtrsv( &cvar1, &cvar, &cvar2, &n, LU.val, ia, ja,
		      r.val, &(Minvvj.val[idx(0,search,Minvvj.ld)]) );
      }
      else {
		    data_parcsrtrsv( MagmaLower, L->storage_type, L->diagorder_type,
          L->num_rows, L->val, L->row, L->col,
          q.val, r.val,
          ptrsv_tol, &ptrsv_iter );
        printf("ParCSRTRSV_L(%d) = %d;\n", search+1, ptrsv_iter);
        data_parcsrtrsv( MagmaUpper, U->storage_type, U->diagorder_type,
          U->num_rows, U->val, U->row, U->col,
          r.val, &(Minvvj.val[idx(0,search,krylov.ld)]),
          ptrsv_tol, &ptrsv_iter );
        printf("ParCSRTRSV_U(%d) = %d;\n", search+1, ptrsv_iter);
      }

      for ( int i=0; i<Minvvj.ld; i++ ) {
        GMRESDBG("Minvvj.val[idx(%d,%d,%d)] = %e\n",
          i, search, Minvvj.ld, Minvvj.val[idx(i,search,Minvvj.ld)]);
      }

      // Sparse matrix-vector product, mkl_dcsrmv
      #pragma omp parallel
      #pragma omp for schedule(monotonic:static) nowait
      #pragma vector aligned
      #pragma vector vecremainder
      #pragma nounroll_and_jam
      for ( int i=0; i<n; i++ ) {
        #pragma nounroll_and_jam
        for ( int j=A->row[i]; j<A->row[i+1]; j++ ) {
          krylov.val[idx(i,search1,krylov.ld)] = krylov.val[idx(i,search1,krylov.ld)] + A->val[j]*Minvvj.val[idx(A->col[j],search,Minvvj.ld)];
        }
      }

      for ( int j=0; j <= search1; j++ ) {
        for ( int i=0; i<krylov.ld; i++ ) {
          GMRESDBG("krylov.val[idx(%d,%d,%d)] = %e\n", i, j, krylov.ld, krylov.val[idx(i,j,krylov.ld)]);
        }
      }

      // Householder Transformations
      for ( int j=0; j <= search; ++j ) {
        dataType sum = 0.0;
        dataType sumBegin = 0.0;
        dataType sumEnd = 0.0;
        startStrip = (j/STRIP+1)*STRIP;

        //printf( "n=%d, j=%d, startStrip=%d BINS=%d endStrip=%d\n",
        //  n, j, startStrip, BINS, endStrip );
        #pragma omp parallel
        {
          #pragma omp for schedule(monotonic:static) nowait
          for ( int ii=startStrip; ii<endStrip; ii+=STRIP ) {
            int b = ii/STRIP;
            sumTemp[b] = 0.0;
            //printf("ii/STRIP=%d\n", ii/STRIP);
            #pragma omp simd
            #pragma vector aligned
            #pragma vector vecremainder
	    #pragma loop_count max(STRIP)
            for ( int i=ii; i<ii+STRIP; ++i ) {
              sumTemp[b] += krylov.val[idx(i,j,krylov.ld)]*krylov.val[idx(i,search1,krylov.ld)];
            }
          }
          #pragma omp single
          {
            #pragma omp task
            {
              #pragma omp simd
              #pragma vector aligned
              #pragma vector vecremainder
	      #pragma loop_count max(STRIP)
              for ( int i=j; i<startStrip; ++i ) {
                sumBegin += krylov.val[idx(i,j,krylov.ld)]*krylov.val[idx(i,search1,krylov.ld)];
              }
            }
            #pragma omp task
            {
              #pragma omp simd
              #pragma vector aligned
              #pragma vector vecremainder
	      #pragma loop_count max(STRIP)
              for ( int i=endStrip; i<n; ++i ) {
                sumEnd += krylov.val[idx(i,j,krylov.ld)]*krylov.val[idx(i,search1,krylov.ld)];
              }
            }
          }
          #pragma omp for simd schedule(simd: static) reduction(+:sum) nowait
          #pragma nounroll
	  #pragma vector aligned
          #pragma vector vecremainder
	  for ( int b=0; b<BINS; ++b ) {
            sum += sumTemp[b];
          }
          #pragma omp barrier
          #pragma omp single
          sum = 2.0*(sumBegin+sum+sumEnd);
          #pragma omp barrier

          #pragma omp for schedule(monotonic:static) nowait
          for ( int ii=startStrip; ii<endStrip; ii+=STRIP ) {
            #pragma omp simd
            #pragma vector aligned
            #pragma vector vecremainder
	    #pragma loop_count max(STRIP)
            for ( int jj=ii; jj < ii+STRIP; ++jj ) {
              krylov.val[idx(jj,search1,krylov.ld)] = krylov.val[idx(jj,search1,krylov.ld)] - sum*krylov.val[idx(jj,j,krylov.ld)];
            }
          }
          #pragma omp single
          {
            #pragma omp task
            {
              #pragma omp simd
              #pragma vector aligned
              #pragma vector vecremainder
	      #pragma loop_count max(STRIP)
              for ( int jj=j; jj<startStrip; ++jj ) {
                krylov.val[idx(jj,search1,krylov.ld)] = krylov.val[idx(jj,search1,krylov.ld)] - sum*krylov.val[idx(jj,j,krylov.ld)];
              }
            }
            #pragma omp task
            {
              #pragma omp simd
              #pragma vector aligned
              #pragma vector vecremainder
	      #pragma loop_count max(STRIP)
              for ( int jj=endStrip; jj<n; ++jj ) {
                krylov.val[idx(jj,search1,krylov.ld)] = krylov.val[idx(jj,search1,krylov.ld)] - sum*krylov.val[idx(jj,j,krylov.ld)];
              }
            }
          }
          #pragma omp taskwait
        }
      }
      for ( int j=0; j <= search1; ++j ) {
        for ( int i=0; i<n; ++i ) {
          GMRESDBG("krylov.val[idx(%d,%d,%d)] = %e\n", i, j, krylov.ld, krylov.val[idx(i,j,krylov.ld)]);
        }
      }

      if ( search < n ) {
        dataType snrm2 = data_dnrm2( (n-search), &(krylov.val[idx(search1,search1,krylov.ld)]), 1 );
        dd = mysgn(krylov.val[idx(search1,search1,krylov.ld)])*snrm2;
        GMRESDBG("dd = %e  %d\n", dd, __LINE__);
        krylov.val[idx(search1,search1,krylov.ld)] = krylov.val[idx(search1,search1,krylov.ld)] + dd;
        snrm2 = 1.0 / data_dnrm2( (n-search), &(krylov.val[idx(search1,search1,krylov.ld)]), 1 );
        startStrip = (search1/STRIP+1)*STRIP;
        #pragma omp parallel
        {
          #pragma omp for schedule(monotonic:static) nowait
          for ( int ii=startStrip; ii<endStrip; ii+=STRIP ) {
            #pragma omp simd
            #pragma vector aligned
            #pragma vector vecremainder
	    #pragma loop_count max(STRIP)
            for ( int i=ii; i<ii+STRIP; ++i ) {
              krylov.val[idx(i,search1,krylov.ld)] = krylov.val[idx(i,search1,krylov.ld)]*snrm2;
            }
          }
          #pragma omp single
          #pragma omp simd
          #pragma vector aligned
          #pragma vector vecremainder
	  #pragma loop_count max(STRIP)
          for ( int i=search1; i<startStrip; ++i ) {
            krylov.val[idx(i,search1,krylov.ld)] = krylov.val[idx(i,search1,krylov.ld)]*snrm2;
          }
          #pragma omp single
          #pragma omp simd
          #pragma vector aligned
          #pragma vector vecremainder
	  #pragma loop_count max(STRIP)
          for ( int i=endStrip; i<n; ++i ) {
            krylov.val[idx(i,search1,krylov.ld)] = krylov.val[idx(i,search1,krylov.ld)]*snrm2;
          }
        }

        for ( int j=0; j <= search1; ++j ) {
          for ( int i=0; i<n; ++i ) {
            GMRESDBG("krylov.val[idx(%d,%d,%d)] = %e\n", i, j, krylov.ld, krylov.val[idx(i,j,krylov.ld)]);
          }
        }

        data_zmfree( &q );
        data_zvinit( &q, n, 1, zero );
        q.val[search1] = 1.0;

        for (int j=search1; j>=0; --j) {
          dataType sum = 0.0;
          dataType sumBegin = 0.0;
          dataType sumEnd = 0.0;
          startStrip = (j/STRIP+1)*STRIP;
          #pragma omp parallel
          {
            #pragma omp for schedule(monotonic:static) nowait
            for ( int ii=startStrip; ii<endStrip; ii+=STRIP ) {
              int b = ii/STRIP;
              sumTemp[b] = 0.0;
              #pragma omp simd
              #pragma vector aligned
              #pragma vector vecremainder
	      #pragma loop_count max(STRIP)
              for ( int i=ii; i<ii+STRIP; ++i ) {
                sumTemp[b] += krylov.val[idx(i,j,krylov.ld)]*q.val[i];
              }
            }

            #pragma omp single
            {
              #pragma omp task
              {
                #pragma omp simd
                #pragma vector aligned
                #pragma vector vecremainder
		#pragma loop_count max(STRIP)
                for ( int i=j; i<startStrip; ++i ) {
                  sumBegin += krylov.val[idx(i,j,krylov.ld)]*q.val[i];
                }
              }
              #pragma omp task
              {
                #pragma omp simd
                #pragma vector aligned
                #pragma vector vecremainder
		#pragma loop_count max(STRIP)
                for ( int i=endStrip; i<n; ++i ) {
                  sumEnd += krylov.val[idx(i,j,krylov.ld)]*q.val[i];
                }
              }
            }
            #pragma omp taskwait

            #pragma omp for simd schedule(simd: static) reduction(+:sum) nowait
            #pragma nounroll
	    #pragma vector aligned
            #pragma vector vecremainder
            for ( int b=0; b<BINS; ++b ) {
              sum += sumTemp[b];
            }
            #pragma omp barrier
            #pragma omp single
            sum = 2.0*(sumBegin+sum+sumEnd);
            #pragma omp barrier


            #pragma omp for schedule(monotonic:static) nowait
            for ( int ii=startStrip; ii<endStrip; ii+=STRIP ) {
              #pragma omp simd
              #pragma vector aligned
              #pragma vector vecremainder
	      #pragma loop_count max(STRIP)
              for ( int jj=ii; jj < ii+STRIP; ++jj ) {
                q.val[jj] = q.val[jj] - sum*krylov.val[idx(jj,j,krylov.ld)];
              }
            }
            #pragma omp single
            {
              #pragma omp task
              {
                #pragma omp simd
                #pragma vector aligned
                #pragma vector vecremainder
		#pragma loop_count max(STRIP)
                for ( int jj=j; jj<startStrip; ++jj ) {
                  q.val[jj] = q.val[jj] - sum*krylov.val[idx(jj,j,krylov.ld)];
                }
              }
              #pragma omp task
              {
                #pragma omp simd
                #pragma vector aligned
                #pragma vector vecremainder
		#pragma loop_count max(STRIP)
                for ( int jj=endStrip; jj<n; ++jj ) {
                  q.val[jj] = q.val[jj] - sum*krylov.val[idx(jj,j,krylov.ld)];
                }
              }
            }
            #pragma omp taskwait
          }
        }
      }

      for ( int i=0; i<n; ++i ) {
        GMRESDBG("q.val[%d] = %e\n", i, q.val[i]);
      }
      for ( int j=0; j <= search1; ++j ) {
        for ( int i=0; i<n; ++i ) {
          GMRESDBG("krylov.val[idx(%d,%d,%d)] = %e\n", i, j, krylov.ld, krylov.val[idx(i,j,krylov.ld)]);
        }
      }

      if (gmres_par->monitorOrthog == 1) {
        // Monitor Orthogonality Error of Krylov search Space
        #pragma omp parallel
        {
          #pragma omp for schedule(monotonic:static) nowait
          for ( int ii=0; ii<endStrip; ii+=STRIP ) {
            #pragma omp simd
            #pragma vector aligned
            for ( int i=ii; i < ii+STRIP; ++i ) {
              precondq.val[idx(i,search1,precondq.ld)] = q.val[i];
            }
          }
          #pragma omp single
          #pragma omp simd
          #pragma vector aligned
          for ( int i=endStrip; i<n; ++i ) {
            precondq.val[idx(i,search1,precondq.ld)] = q.val[i];
          }
        }
        dataType ortherr = 0.0;
        int imax = 0;
        data_orthogonality_error( &precondq, &ortherr, &imax, search1 );
        if ( gmres_par->user_csrtrsv_choice == 0 ) {
          printf("FGMRES_Householders_mkltrsv_ortherr(%d) = %.16e;\n", search+1, ortherr);
        }
        else {
          printf("FGMRES_Householders_partrsv_ortherr(%d) = %.16e;\n", search+1, ortherr);
        }
      }

      // Apply Givens rotations
      dataType temp0 = 0.0;
      dataType temp1 = 0.0;
      for ( int j = 0; j<search; ++j ) {
        temp0 = krylov.val[idx(j,search1,krylov.ld)];
        temp1 = krylov.val[idx(j+1,search1,krylov.ld)];

        #pragma omp parallel sections num_threads(2)
        {
          #pragma omp section
          {
            krylov.val[idx(j+1,search1,krylov.ld)] = -givens_sin.val[j]*temp0
              + givens_cos.val[j]*temp1;
          }
          #pragma omp section
          {
            krylov.val[idx(j,search1,krylov.ld)] = givens_cos.val[j]*temp0
              + givens_sin.val[j]*temp1;
          }
        }
        GMRESDBG("\n\tApply Givens rotations search = %d temp = %e\n", j, temp);
      }
      for ( int j=0; j <= search1; ++j ) {
        for ( int i=0; i<n; ++i ) {
          GMRESDBG("krylov.val[idx(%d,%d,%d)] = %e\n", i, j, krylov.ld, krylov.val[idx(i,j,krylov.ld)]);
        }
      }

      if ( search < n ) {
        GMRESDBG("%e %e %e %e %e\n", krylov.val[idx(search,search1,krylov.ld)], -dd, eps, givens_cos.val[search], givens_sin.val[search] )
        // form search-th rotation matrix
        givens_rotation(krylov.val[idx(search,search1,krylov.ld)], -dd, eps, &givens_cos.val[search], &givens_sin.val[search] );
        for ( int i=0; i<n; ++i ) {
          GMRESDBG("givens_cos.val[%d] = %e\n", i, givens_cos.val[i]);
        }
        for ( int i=0; i<n; ++i ) {
          GMRESDBG("givens_sin.val[%d] = %e\n", i, givens_sin.val[i]);
        }
        krylov.val[idx(search,search1,krylov.ld)] = givens_cos.val[search]*krylov.val[idx(search,search1,krylov.ld)]
          - givens_sin.val[search]*dd;
        // approximate residual norm
        givens.val[search1] = -givens_sin.val[search]*givens.val[search];
        givens.val[search] = givens_cos.val[search]*givens.val[search];
        residual = fabs(givens.val[search1]);
      }
      for ( int i=0; i<givens.ld; ++i ) {
        GMRESDBG("givens.val[%d] = %e\n", i, givens.val[i]);
      }

      if ( gmres_par->user_csrtrsv_choice == 0 ) {
        printf("FGMRES_Householders_mkltrsv_search(%d) = %.16e;\n", search+1, fabs(givens.val[(search+1)]));
      }
      else {
        printf("FGMRES_Householders_partrsv_search(%d) = %.16e;\n", search+1, fabs(givens.val[(search+1)]));
      }
      // update the solution
      // solve the least squares problem
      if ( fabs(givens.val[(search+1)]) < rtol  || (search == (search_max-1)) ) {
        GMRESDBG(" !!!!!!! update the solution %d!!!!!!!\n",0);
        #pragma omp parallel
        #pragma omp for simd schedule(simd: static) nowait
        #pragma vector aligned
        #pragma vector vecremainder
        #pragma nounroll_and_jam
        for ( int i = 0; i <= search; ++i ) {
          r.val[i] = givens.val[i]/krylov.val[idx(i,i+1,krylov.ld)];
        }
        for ( int j = search; j > 0; --j ) {
          for (int i = j-1; i >= 0; --i ) {
            r.val[i] = r.val[i]
             - krylov.val[idx(i,j+1,krylov.ld)]*r.val[j]/krylov.val[idx(i,i+1,krylov.ld)];
          }
        }
        for ( int i=0; i<n; ++i ) {
          GMRESDBG("r.val[%d] = %e\n", i, r.val[i]);
        }

        // use preconditioned vectors to form the update (GEMV)
        for (int j = 0; j <= search; j++ ) {
          #pragma omp parallel
          #pragma omp for simd schedule(simd: static) nowait
          #pragma vector aligned
          #pragma vector vecremainder
          #pragma nounroll_and_jam
          for (int i = 0; i < n; i++ ) {
            x.val[i] = x.val[i] + Minvvj.val[idx(i,j,Minvvj.ld)]*r.val[j];
          }
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
    data_zmfree( &givens );
    data_zmfree( &givens_cos );
    data_zmfree( &givens_sin );
    data_zmfree( &q );
    data_zmfree( &LU );
    data_zmfree( &Minvvj );
    data_zmfree( &precondq );

    // free( sumTemp );
    free( ia );
    free( ja );

    return info;
}
