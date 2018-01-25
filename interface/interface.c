#include <unistd.h>
#include <stdio.h>
#include <sparse.h>

void terminateFortranCharArray(
  char * name) 
{
  for (int i=0; i<1024; i++) {
    if ( name[i] == ' ' )
      name[i] = '\0'; 
    if ( name[i] == '\0' )
      break;
  }

}

int
read_z_coo_from_mtx_(
  data_storage_t * type,
  int *            n_row,
  int *            n_col,
  int *            nnz,
  dataType *      coo_val,
  int *           coo_row,
  int *           coo_col,
  char *           filename,
  int ll)
{
  int info = 0;
  printf("read_z_coo_from_mtx\n");
  char cwd[1024];
  //chdir("/path/to/change/directory/to");
  getcwd(cwd, sizeof(cwd));
  printf("Current working dir: %s\n", cwd);
  printf("file name = %s|\n", filename);
  terminateFortranCharArray( filename );
  //filename[25] = '\0';
  printf("file name = %s|\n", filename);
  //CHECK( read_z_coo_from_mtx(type, n_row, n_col, nnz, &coo_val, &coo_row, &coo_col, filename) );

  //LACE_CALLOC(coo_val, 147);
  for (int i=0; i<20; i++) {
    coo_val[i] = i;
    printf("%e\n", coo_val[i]);
  }

  //coo_val = &(coo_val[0]);
  return info;
}


int
data_PariLU_v0_3_(
  dataType **             A_val,
  int                     A_nnz,
  int                     A_nrows,
  int                     A_ncols,
  int **                  A_row,
  int **                  A_col,
  dataType **             L_val,
  int                     L_nnz,
  int                     L_nrows,
  int                     L_ncols,
  int **                  L_row,
  int **                  L_col,
  dataType **             U_val,
  int                     U_nnz,
  int                     U_nrows,
  int                     U_ncols,
  int **                  U_row,
  int **                  U_col,
  dataType                reduction)
{
  
  data_d_matrix ACOO = {Magma_COO};
  data_d_matrix LCOO = {Magma_COO};
  data_d_matrix UCOO = {Magma_COO};
  ACOO.nnz = A_nnz;
  ACOO.num_rows = A_nrows;
  ACOO.num_cols = A_ncols;
  ACOO.val = (*A_val);
  ACOO.row = (*A_row);
  ACOO.col = (*A_col);

  LCOO.nnz = L_nnz;
  LCOO.num_rows = L_nrows;
  LCOO.num_cols = L_ncols;
  LCOO.val = (*L_val);
  LCOO.row = (*L_row);
  LCOO.col = (*L_col);

  UCOO.nnz = U_nnz;
  UCOO.num_rows = U_nrows;
  UCOO.num_cols = U_ncols;
  UCOO.val = (*U_val);
  UCOO.row = (*U_row);
  UCOO.col = (*U_col);


  data_d_matrix A = {Magma_CSR};
  data_d_matrix L = {Magma_CSR};
  data_d_matrix U = {Magma_CSR};

  CHECK( data_zmconvert( ACOO, &A, Magma_COO, Magma_CSR) );
  CHECK( data_zmconvert( LCOO, &L, Magma_COO, Magma_CSR) );
  CHECK( data_zmconvert( UCOO, &U, Magma_COO, Magma_CSR) );

  struct data_z_preconditioner_log parilu_log;
  data_PariLU_v0_3(&A, &L, &U, reduction, &parilu_log);

  printf("PariLU_v0_3_sweeps = %d\n", parilu_log.sweeps);
  printf("PariLU_v0_3_tol = %e\n", parilu_log.tol);
  printf("PariLU_v0_3_A_Frobenius = %e\n", parilu_log.A_Frobenius);
  printf("PariLU_v0_3_generation_time = %e\n", parilu_log.precond_generation_time);
  printf("PariLU_v0_3_initial_residual = %e\n", parilu_log.initial_residual);
  printf("PariLU_v0_3_initial_nonlinear_residual = %e\n", parilu_log.initial_nonlinear_residual);
  printf("PariLU_v0_3_omp_num_threads = %d\n", parilu_log.omp_num_threads);

  return 0;
}

