#ifndef SPARSE_Z_H
#define SPARSE_Z_H

#include "sparse_types.h"
#include <vector>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define PRECISION_z


#ifdef __cplusplus
extern "C" {
#endif  

//======================================
// I/O
//======================================

int 
read_z_csr_from_mtx( 
    data_storage_t *type, 
    int* n_row, 
    int* n_col, 
    int* nnz, 
    dataType **val, 
    int **row, 
    int **col, 
    const char *filename );

int 
read_z_coo_from_mtx(
    data_storage_t *type,
    int* n_row,
    int* n_col,
    int* nnz,
    dataType **coo_val,
    int **coo_row,
    int **coo_col,
    const char *filename );

int 
read_z_dense_from_mtx(
    data_storage_t *type,
    int* n_row,
    int* n_col,
    int* nnz,
    data_order_t major,
    dataType **val,
    const char *filename );

int
data_z_csr_mtx(
    data_d_matrix *A,
    const char *filename );

int
data_z_coo_mtx(
    data_d_matrix *A,
    const char *filename );

int
data_z_dense_mtx(
    data_d_matrix *A,
    data_order_t major,
    const char *filename );

int
data_zprint_coo_mtx(
    int n_row,
    int n_col,
    int nnz,
    dataType **val,
    int **row,
    int **col );

int
data_zprint_bcsr(
    data_d_matrix* A );

int
data_zprint_coo(
    data_d_matrix A );

int
data_zprint_csr_mtx(
    int n_row,
    int n_col,
    int nnz,
    dataType **val,
    int **row,
    int **col,
    data_order_t MajorType );

int
data_zprint_csr(
    data_d_matrix A );

int
data_zwrite_csr(
    data_d_matrix* A );

int
data_zprint_dense_mtx(
    int n_row,
    int n_col,
    int nnz,
    data_order_t major,
    dataType **val );

int
data_zprint_dense(
    data_d_matrix A );

int
data_zdisplay_dense(
    data_d_matrix* A );

int
data_zwrite_csr_mtx(
    data_d_matrix A,
    data_order_t MajorType,
    const char *filename );

int
data_zwrite_dense_mtx(
    int n_row,
    int n_col,
    int nnz,
    data_order_t major,
    dataType **val,
    const char* filename );

int
data_zwrite_dense(
    data_d_matrix A,
    const char* filename );

//======================================
// control
//======================================

int
data_z_csr_compressor(
    dataType ** val,
    int ** row,
    int ** col,
    dataType ** valn,
    int ** rown,
    int ** coln,
    int *n );

int
data_rowindex(
    data_d_matrix *A,
    int **rowidx );

int
data_zmconvert(
    data_d_matrix A,
    data_d_matrix *B,
    data_storage_t old_format,
    data_storage_t new_format );

int
data_zmtranspose(
    data_d_matrix A, data_d_matrix *B );

int
data_zmfree(
    data_d_matrix *A );

int
data_z_pad_dense(
    data_d_matrix *A,
    int tile_size );

int
data_z_pad_csr(
    data_d_matrix *A,
    int tile_size );

void
data_sparse_subvector( 
  int sub_mbegin, 
  int sub_nbegin, 
  data_d_matrix* A, 
  dataType* subvector );

void
data_sparse_subvector_lowerupper( 
  int sub_mbegin, 
  int sub_nbegin, 
  data_d_matrix* A, 
  dataType* subvector );

void
data_sparse_subdense( 
  int sub_m, 
  int sub_n, 
  int sub_mbegin, 
  int sub_nbegin, 
  data_d_matrix* A, 
  dataType* subdense );

void
data_sparse_subdense_lowerupper( 
  int sub_m, 
  int sub_n, 
  int sub_mbegin, 
  int sub_nbegin, 
  data_d_matrix* A, 
  dataType* subdense );

void
data_sparse_subsparse( 
  int sub_m, 
  int sub_n, 
  int sub_mbegin, 
  int sub_nbegin, 
  data_d_matrix* A, 
  int* rowtmp, 
  int* rowindxtmp, 
  int* colindxtmp, 
  int* coltmp, 
  int* nnztmp );

void
data_sparse_subsparse_lowerupper( 
  int sub_m, 
  int sub_n, 
  int sub_mbegin, 
  int sub_nbegin, 
  data_d_matrix* A, 
  int* rowtmp, 
  int* rowindxtmp, 
  int* colindxtmp, 
  int* coltmp, 
  int* nnztmp );

void
data_sparse_subsparse_cs( 
  int sub_m, 
  int sub_n, 
  int sub_mbegin, 
  int sub_nbegin, 
  data_d_matrix* A, 
  data_d_matrix* Asub );

int
data_sparse_subsparse_cs_lowerupper( 
  int sub_m, 
  int sub_n, 
  int sub_mbegin, 
  int sub_nbegin, 
  data_d_matrix* A, 
  data_d_matrix* Asub );

int
data_sparse_subsparse_cs_lowerupper_handle( 
  int sub_m, 
  int sub_n, 
  int sub_mbegin, 
  int sub_nbegin, 
  int uplo,
  data_d_matrix* A, 
  data_d_matrix* Asub, 
  sparse_matrix_t* Asub_handle );

void
data_sparse_tilepattern( 
  int sub_m, 
  int sub_n, 
  std::vector<Int3>* tiles, 
  data_d_matrix* A );

void
data_sparse_tilepattern_handles( int sub_m, int sub_n, 
  std::vector<Int3>* tiles, 
  std::vector<data_d_matrix>* L_subs,
  std::vector<data_d_matrix>* U_subs,
  std::vector<sparse_matrix_t>* L_handles,
  std::vector<sparse_matrix_t>* U_handles,
  data_d_matrix* A );

void
data_sparse_tilepatterns( int sub_m, 
  int sub_n, 
  std::vector<Int3>* Ltiles, 
  std::vector<Int3>* Utiles, 
  data_d_matrix* A );

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
  data_d_matrix* A );

void
data_sparse_tilepattern_lowerupper( 
  int sub_m, 
  int sub_n, std::vector<Int3>* tiles, 
  data_d_matrix* A );

int
data_zrowentries(
    data_d_matrix *A );

int
data_zdiameter(
    data_d_matrix *A );

int
data_zcheckupperlower(
  data_d_matrix * A );

int
data_zmscale(
    data_d_matrix *A,
    data_scale_t scaling );

int
data_zmscale_matrix_rhs(
    data_d_matrix *A,
    data_d_matrix *b,
    data_d_matrix *sale_factors,
    data_scale_t scaling );

int
data_zmdiagadd(
    data_d_matrix *A,
    dataType add );

int
data_zmscale_generate( 
	  int n, 
	  data_scale_t* scaling, 
	  data_side_t* side, 
	  data_d_matrix* A, 
	  data_d_matrix* scaling_factors );

int
data_zmscale_apply( 
	  int n,  
	  data_side_t* side, 
	  data_d_matrix* scaling_factors, 
	  data_d_matrix* A );

int
data_zdimv( 
    data_d_matrix* vecA, 
    data_d_matrix* vecB );

int
data_zlascl2(
    data_type_t type, int m, int n,
    dataType* dD,
    dataType* dA, 
    int ldda );

int
data_zmlumerge(
    data_d_matrix L,
    data_d_matrix U,
    data_d_matrix *A );


int
data_zmextractdiag(
    data_d_matrix A,
    data_d_matrix *B );

int
data_zmcopy(
    data_d_matrix A,
    data_d_matrix *B );

//======================================
// norms
//======================================

int
data_zfrobenius_csr(
    data_d_matrix A,
    dataType *res );

int
data_zfrobenius_dense(
    data_d_matrix A,
    dataType *res );

int
data_zfrobenius(
    data_d_matrix A,
    dataType *res );

int
data_zfrobenius_diff_csr(
    data_d_matrix A,
    data_d_matrix B,
    dataType *res );

int
data_zfrobenius_diff_dense(
    data_d_matrix A,
    data_d_matrix B,
    dataType *res );

int
data_zfrobenius_diff(
    data_d_matrix A,
    data_d_matrix B,
    dataType *res );

int
data_zfrobenius_LUresidual( 
  data_d_matrix A,
  data_d_matrix L,
  data_d_matrix U,
  dataType *res);

int
data_zfrobenius_inplaceLUresidual( 
  data_d_matrix A,
  data_d_matrix LU,
  dataType *res);

int
data_zilures(
    data_d_matrix A,
    data_d_matrix L,
    data_d_matrix U,
    data_d_matrix *LU,
    dataType *res,
    dataType *nonlinres );

int
data_maxfabs_csr(
    data_d_matrix A,
    int *imax,
    int *jmax,
    dataType *max );

int
data_norm_diff_vec(
  data_d_matrix* A,
  data_d_matrix* B,
  dataType* norm ); 

//======================================
// dense operations
//======================================

dataType
data_zdot(
    int n,
    dataType* dx, int incx,
    dataType* dy, int incy );

dataType
data_zdot_mkl(
    int n,
    dataType* dx, int incx,
    dataType* dy, int incy );

void
data_dgemv_mkl(
    data_order_t layoutA, data_trans_t transA, int m, int n, 
    dataDouble alpha,
    dataDouble_const_ptr A, int ldda,
    dataDouble_const_ptr x, int incx, dataDouble beta,
    dataDouble_ptr y, int incy );

void
data_dgemm_mkl(
    data_order_t layoutA, data_trans_t transA, data_trans_t transB, 
    int m, int n, int k,
    dataDouble alpha, dataDouble_const_ptr A, int lda,
    dataDouble_const_ptr B, int ldb, 
    dataDouble beta, dataDouble_ptr C, int ldc );


//======================================
// sparse operations
//======================================

int
data_z_spmm(
    dataType alpha,
    data_d_matrix A,
    data_d_matrix B,
    data_d_matrix *C);
int
data_z_spmm_handle(
    dataType alpha,
    sparse_matrix_t* A,
    sparse_matrix_t* B,
    data_d_matrix *C);

int
data_sparse_subsparse_spmm( 
  int tile, 
  int span, 
  int ti, 
  int tj, 
  Int3* tiles,
  data_d_matrix* Lsub, 
  data_d_matrix* Usub, 
  data_d_matrix* C );

int
data_sparse_subsparse_spmm_handle( 
  int tile, 
  int span, 
  int ti, 
  int tj, 
  Int3* tiles,
  sparse_matrix_t* L, 
  sparse_matrix_t* U, 
  data_d_matrix* C );

int
data_sparse_subsparse_spmm_batches(
  int to_update,
  int tile,
  dataType alpha,
  std::vector<sparse_matrix_t>* L_handles,
  std::vector<sparse_matrix_t>* U_handles, 
  std::vector<int>* lbatch,
  std::vector<int>* ubatch,
  data_d_matrix* C ); 

int
data_z_spmm_batch(
    dataType alpha,
    sparse_matrix_t* csrA,
    sparse_matrix_t* csrB,
    sparse_matrix_t* csrC );

int
data_zdiff_csr(
    data_d_matrix *A,
    data_d_matrix *B,
    data_d_matrix *C, 
    dataType *res,
    dataType *nonlinres );

int
data_zdiff_magnitude_csr(
    data_d_matrix *A,
    data_d_matrix *B,  
    dataType *res);

int
data_zsubtract_csr(
    data_d_matrix *A,
    data_d_matrix *B );

int
data_zsubtract_guided_csr(
  data_d_matrix *A,
  data_d_matrix *B,
  data_d_matrix *C,
  dataType *step );

int
data_zdiagdivide_csr(
    data_d_matrix *A,
    data_d_matrix *B );

int
data_zset_csr(
    data_d_matrix *A,
    data_d_matrix *B );

int
data_diagbcsr_mult_bcsr( 
  data_d_matrix* diagA, 
  data_d_matrix* A );

//======================================
// dense factorizations
//======================================

void
data_LUnp_mkl( data_d_matrix* A );

void
data_ParLU_v0_0( 
  data_d_matrix* A, 
  data_d_matrix* L, 
  data_d_matrix* U );

void
data_ParLU_v0_1( 
  data_d_matrix* A, 
  data_d_matrix* L, 
  data_d_matrix* U );

void
data_ParLU_v1_0( 
  data_d_matrix* A, 
  data_d_matrix* L, 
  data_d_matrix* U );

void
data_ParLU_v1_1( 
  data_d_matrix* A, 
  data_d_matrix* L, 
  data_d_matrix* U );

void
data_ParLU_v1_2( 
  data_d_matrix* A, 
  data_d_matrix* L, 
  data_d_matrix* U,
  int tile );

void
data_ParLU_v1_2c( 
  data_d_matrix* A, 
  data_d_matrix* L, 
  data_d_matrix* U,
  int tile );

void
data_ParLU_v1_3( 
  data_d_matrix* A, 
  data_d_matrix* L, 
  data_d_matrix* U,
  int tile );

void
data_ParLU_v2_0( 
  data_d_matrix* A, 
  data_d_matrix* L, 
  data_d_matrix* U,
  int tile );

void
data_ParLU_v2_1( 
  data_d_matrix* A, 
  data_d_matrix* L, 
  data_d_matrix* U,
  int tile );

void
data_ParLU_v3_0( 
  data_d_matrix* A, 
  data_d_matrix* L, 
  data_d_matrix* U,
  int tile );

void
data_ParLU_v3_1( 
  data_d_matrix* A, 
  data_d_matrix* L, 
  data_d_matrix* U,
  int tile );

//======================================
// sparse factorizations
//======================================

int
data_dcsrilu0_mkl( data_d_matrix* A );

void
data_PariLU_v0_0( 
  data_d_matrix* A, 
  data_d_matrix* L, 
  data_d_matrix* U );

void
data_PariLU_v0_1( 
  data_d_matrix* A, 
  data_d_matrix* L, 
  data_d_matrix* U );

void
data_PariLU_v0_2( 
  data_d_matrix* A, 
  data_d_matrix* L, 
  data_d_matrix* U,
  dataType reduction );

void
data_PariLU_v0_3( 
  data_d_matrix* A, 
  data_d_matrix* L, 
  data_d_matrix* U,
  dataType reduction,
  data_d_preconditioner_log* log );

void
data_PariLU_v0_4( 
  data_d_matrix* A, 
  data_d_matrix* L, 
  data_d_matrix* U,
  dataType reduction );

void
data_PariLU_v3_0( 
  data_d_matrix* A, 
  data_d_matrix* L, 
  data_d_matrix* U,
  int tile );

void
data_PariLU_v3_1( 
  data_d_matrix* A, 
  data_d_matrix* L, 
  data_d_matrix* U,
  int tile );

void
data_PariLU_v3_2( 
  data_d_matrix* A, 
  data_d_matrix* L, 
  data_d_matrix* U,
  int tile );

void
data_PariLU_v3_5( 
  data_d_matrix* A, 
  data_d_matrix* L, 
  data_d_matrix* U,
  int tile );

void
data_PariLU_v3_6( 
  data_d_matrix* A, 
  data_d_matrix* L, 
  data_d_matrix* U,
  int tile );

void
data_PariLU_v3_7( 
  data_d_matrix* A, 
  data_d_matrix* L, 
  data_d_matrix* U,
  int tile );

void
data_PariLU_v3_8( 
  data_d_matrix* A, 
  data_d_matrix* L, 
  data_d_matrix* U,
  int tile );


void
data_PariLU_v3_9( 
  data_d_matrix* A, 
  data_d_matrix* L, 
  data_d_matrix* U,
  int tile );

int
data_PariLU_v4_0( 
  data_d_matrix* A, 
  data_d_matrix* L, 
  data_d_matrix* U ); 

//======================================
// Tri-solves
//======================================

int
data_forward_solve( 
  data_d_matrix* L, 
  data_d_matrix* x, 
  data_d_matrix* rhs,
  const dataType tol, 
  int *iter ); 

int
data_forward_solve_permute( 
  data_d_matrix* L, 
  data_d_matrix* x, 
  data_d_matrix* rhs,
  const dataType tol, 
  int *iter ); 

int
data_backward_solve( 
  data_d_matrix* L, 
  data_d_matrix* x, 
  data_d_matrix* rhs,
  const dataType tol, 
  int *iter ); 

int
data_backward_solve_permute( 
  data_d_matrix* L, 
  data_d_matrix* x, 
  data_d_matrix* rhs,
  const dataType tol, 
  int *iter ); 

int
data_parcsrtrsv( 
  const data_uplo_t uplo, 
  const data_storage_t storage, 
  const data_diagorder_t diag, 
  const int num_rows, 
  const dataType *Aval, 
  const int *row, 
  const int *col, 
  const dataType *rhsval, 
  dataType *yval,
  const dataType tol, 
  int *iter ); 

int
data_partrsv( 
  const data_order_t major, 
  const data_uplo_t uplo, 
  const data_storage_t storage, 
  const data_diagorder_t diag, 
  const int num_rows, 
  const dataType *Aval, 
  const int lda, 
  const dataType *rhsval, 
  const int incr,
  dataType *yval,
  const int incx,
  const dataType tol, 
  int *iter );


int
data_partrsv_dot( 
  const data_order_t major, 
  const data_uplo_t uplo, 
  const data_storage_t storage, 
  const data_diagorder_t diag, 
  const int num_rows, 
  dataType *Aval, 
  const int lda, 
  const dataType *rhsval, 
  const int incr,
  dataType *yval,
  const int incx,
  const dataType tol, 
  int *iter );

//======================================
// Form an inverse matrix
//======================================

int
data_inverse( 
  data_d_matrix* A, 
  data_d_matrix* Ainv );

int
data_inverse_bcsr( 
  data_d_matrix* A, 
  data_d_matrix* Ainv );

#ifdef __cplusplus
}
#endif

#undef PRECISION_z
#endif /* SPARSE_Z_H */