


#ifndef SPARSE_TYPES_H
#define SPARSE_TYPES_H

#include "dense_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct alignas(DEV_ALIGN) data_d_matrix
{
    data_storage_t     storage_type;            // matrix format - CSR, ELL, SELL-P
    data_symmetry_t    sym;                     // opt: indicate symmetry
    data_diagorder_t   diagorder_type;          // opt: only needed for factorization matrices
    data_uplo_t        fill_mode;               // fill mode full/lower/upper
    data_int_t         num_rows;                // number of rows
    data_int_t         num_cols;                // number of columns
    data_int_t         nnz;                     // opt: number of nonzeros
    data_int_t         max_nnz_row;             // opt: max number of nonzeros in one row
    data_int_t         diameter;                // opt: max distance of entry from main diagonal
    data_int_t         true_nnz;                // opt: true nnz
    dataType           *val;                    // array containing values
    dataType           *diag;                   // opt: diagonal entries
    data_int_t         *row;                    // row pointer CPU
    data_int_t         *rowidx;                 // opt: array containing row
    data_int_t         *col;                    // array containing col indices
    data_int_t         *list;                   // opt: linked list pointing to next element
    data_int_t         *blockinfo;              // opt: for BCSR format CPU case
    data_int_t         blocksize;               // opt: info for SELL-P/BCSR
    data_int_t         ldblock;                 // opt: info for SELL-P/BCSR
    data_int_t         numblocks;               // opt: info for SELL-P/BCSR
    data_int_t         alignment;               // opt: info for SELL-P/BCSR
    data_order_t       major;                   // opt: row/col major for dense matrices
    data_int_t         ld;                      // opt: leading dimension for dense
    data_int_t         pad_rows;
    data_int_t         pad_cols;
} data_d_matrix;

struct Int3 {
  int a[3];
};

//*****************     solver parameters     ********************************//

typedef struct data_d_solver_par
{
    data_solver_type   solver;                  // solver type
    data_int_t         version;                 // sometimes there are different versions
    dataType           atol;                     // absolute residual stopping criterion
    dataType           rtol;                    // relative residual stopping criterion
    data_int_t         maxiter;                 // upper iteration limit
    data_int_t         restart;                 // for GMRES
    data_ortho_t       ortho;                   // for GMRES
    data_int_t         numiter;                 // feedback: number of needed iterations
    data_int_t         spmv_count;              // feedback: number of needed SpMV - can be different to iteration count
    dataType           init_res;                // feedback: initial residual
    dataType           final_res;               // feedback: final residual
    dataType           iter_res;                // feedback: iteratively computed residual
    real_Double_t      runtime;                 // feedback: runtime needed
    real_Double_t      *res_vec;                // feedback: array containing residuals
    real_Double_t      *timing;                 // feedback: detailed timing
    data_int_t         verbose;                 // print residual ever 'verbose' iterations
    data_int_t         num_eigenvalues;         // number of EV for eigensolvers
    data_int_t         ev_length;               // needed for framework
    dataType           *eigenvalues;            // feedback: array containing eigenvalues
    dataType_ptr       eigenvectors;   // feedback: array containing eigenvectors on DEV
    data_int_t         info;                    // feedback: did the solver converge etc.

    //---------------------------------
    // the input for verbose is:
    // 0 = production mode
    // k>0 = convergence and timing is monitored in *res_vec and *timeing every
    // k-th iteration
    //
    // the output of info is:
    //  0 = convergence (stopping criterion met)
    // -1 = no convergence
    // -2 = convergence but stopping criterion not met within maxiter
    //--------------------------------
} data_d_solver_par;

//************            preconditioner parameters       ********************//

typedef struct data_z_preconditioner
{
    data_solver_type       solver;
    data_solver_type       trisolver;
    data_int_t             levels;
    data_int_t             sweeps;
    data_int_t             pattern;
    data_int_t             bsize;
    data_int_t             offset;
    data_precision         format;
    dataType               atol;
    dataType               rtol;
    data_int_t             maxiter;
    data_int_t             restart;
    data_int_t             numiter;
    data_int_t             spmv_count;
    dataType               init_res;
    dataType               final_res;
    real_Double_t          runtime;                 // feedback: preconditioner runtime needed
    real_Double_t          setuptime;               // feedback: preconditioner setup time needed
    data_d_matrix          M;
    data_d_matrix          L;
    data_d_matrix          LT;
    data_d_matrix          U;
    data_d_matrix          UT;
    data_d_matrix          LD;
    data_d_matrix          UD;
    data_d_matrix          LDT;
    data_d_matrix          UDT;
    data_d_matrix          d;
    data_d_matrix          d2;
    data_d_matrix          work1;
    data_d_matrix          work2;
    data_int_t*            int_array_1;
    data_int_t*            int_array_2;

#if defined(HAVE_PASTIX)
    pastix_data_t*         pastix_data;
    data_int_t*            iparm;
    dataType*              dparm;
#endif
} data_d_preconditioner;

typedef struct data_z_preconditioner_log
{
    data_int_t             sweeps;
    dataType               tol;
    dataType               A_Frobenius;
    dataType               precond_generation_time;
    dataType               initial_residual;
    dataType               initial_nonlinear_residual;
    dataType               residual;
    dataType               nonlinear_residual;
    data_int_t             omp_num_threads;
} data_d_preconditioner_log;


typedef struct data_z_gmres_log
{
    data_int_t             restarts;
    data_int_t             search_directions;
    dataType               solve_time;
    dataType               initial_residual;
    dataType               final_residual;
    dataType               scaled_residual;
    dataType               original_residual;
} data_d_gmres_log;

typedef struct data_z_gmres_param
{
    data_int_t             search_max;	   // max search directions per restart
    data_int_t             tol_type;	     // 0 -- absolute; 1 -- relative
    dataType               rtol;	         // relative residual reduction factor
    data_int_t             reorth;         // 0 -- Brown/Hindmarsh condition (default)
                                           // 1 -- Never reorthogonalize (not recommended)
                                           // 2 -- Always reorthogonalize (not cheap!)
    data_int_t             user_csrtrsv_choice; // 0 -- MKL CSRTRSV
                                                // 1 -- ParCSRTRSV
    data_int_t             monitorOrthog;  // 0 -- do not monitor
                                           // 1 -- monitor
    data_int_t             restart_max;    // max restarts
} data_d_gmres_param;


#ifdef __cplusplus
}
#endif

#endif        //  #ifndef SPARSE_TYPES_H
