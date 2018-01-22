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
  dataType **      coo_val,
  int **           coo_row,
  int **           coo_col,
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
  CHECK( read_z_coo_from_mtx(type, n_row, n_col, nnz, coo_val, coo_row, coo_col, filename) );
  return info;
}
