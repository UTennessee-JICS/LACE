/*
    -- DEV (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
       @author Hartwig Anzt
       @author Mark Gates
       @author Stephen Wood
*/

//  in this file, many routines are taken from
//  the IO functions provided by MatrixMarket

#include <math.h>
#include <algorithm>
#include <vector>
#include <utility>  // pair

#include "../include/mmio.h"
#include "../include/dense_types.h"


/**
    Purpose
    -------
    Returns true if first element of a is less than first element of b.
    Ignores second element. Used for sorting pairs,
    std::pair< int, dataType >, of column indices and values.
*/
static bool compare_first(
    const std::pair< int, dataType >& a,
    const std::pair< int, dataType >& b )
{
    return (a.first < b.first);
}


/**
    Purpose
    -------

    Reads in a matrix stored in coo format from a Matrix Market (.mtx)
    file and converts it into CSR format. It duplicates the off-diagonal
    entries in the symmetric case.

    Arguments
    ---------

    @param[out]
    type        data_storage_t*
                storage type of matrix

    @param[out]
    location    data_location_t*
                location of matrix

    @param[out]
    n_row       int*
                number of rows in matrix

    @param[out]
    n_col       int*
                number of columns in matrix

    @param[out]
    nnz         int*
                number of nonzeros in matrix

    @param[out]
    val         dataType**
                value array of CSR output

    @param[out]
    row         int**
                row pointer of CSR output

    @param[out]
    col         int**
                column indices of CSR output

    @param[in]
    filename    const char*
                filname of the mtx matrix

    @ingroup datasparse_zaux
    ********************************************************************/

extern "C"
int read_z_csr_from_mtx(
    data_storage_t *type,
    int* n_row,
    int* n_col,
    int* nnz,
    dataType **val,
    int **row,
    int **col,
    const char *filename )
{
    char buffer[ 1024 ];
    int info = 0;

    int *coo_col=NULL, *coo_row=NULL;
    dataType *coo_val=NULL;
    int *new_col=NULL, *new_row=NULL;
    dataType *new_val=NULL;
    //int hermitian = 0;

    std::vector< std::pair< int, dataType > > rowval;

    FILE *fid = NULL;
    MM_typecode matcode;
    fid = fopen(filename, "r");

    if (fid == NULL) {
        printf("%% Unable to open file %s\n", filename);
        info = DEV_ERR_NOT_FOUND;
        goto cleanup;
    }

    printf("%% Reading sparse matrix from file (%s):", filename);
    fflush(stdout);

    if (mm_read_banner(fid, &matcode) != 0) {
        printf("\n%% Could not process Matrix Market banner: %s.\n", matcode);
        info = DEV_ERR_NOT_SUPPORTED;
        goto cleanup;
    }

    if (!mm_is_valid(matcode)) {
        printf("\n%% Invalid Matrix Market file.\n");
        info = DEV_ERR_NOT_SUPPORTED;
        goto cleanup;
    }

    if ( ! ( ( mm_is_real(matcode)    ||
               mm_is_integer(matcode) ||
               mm_is_pattern(matcode) ||
               mm_is_complex(matcode) ) &&
             mm_is_coordinate(matcode)  &&
             mm_is_sparse(matcode) ) )
    {
        mm_snprintf_typecode( buffer, sizeof(buffer), matcode );
        printf("\n%% Sorry, DEV-sparse does not support Market Market type: [%s]\n", buffer );
        printf("%% Only real-valued or pattern coordinate matrices are supported.\n");
        info = DEV_ERR_NOT_SUPPORTED;
        goto cleanup;
    }

    int num_rows, num_cols, num_nonzeros;
    if (mm_read_mtx_crd_size(fid, &num_rows, &num_cols, &num_nonzeros) != 0) {
        info = DEV_ERR_UNKNOWN;
        goto cleanup;
    }

    *type     = Magma_CSR;
    *n_row    = num_rows;
    *n_col    = num_cols;
    *nnz      = num_nonzeros;

    //( data_index_malloc_cpu( &coo_col, *nnz ) );
    //( data_index_malloc_cpu( &coo_row, *nnz ) );
    //( data_zmalloc_cpu( &coo_val, *nnz ) );
    // coo_row = (int*) malloc( *nnz*sizeof(int) );
    // coo_col = (int*) malloc( *nnz*sizeof(int) );
    // coo_val = (dataType*) malloc( *nnz*sizeof(dataType) );
    LACE_CALLOC( coo_row, *nnz );
    LACE_CALLOC( coo_col, *nnz );
    LACE_CALLOC( coo_val, *nnz );

    if (mm_is_real(matcode) || mm_is_integer(matcode)) {
        for(int i = 0; i < *nnz; ++i) {
            int ROW, COL;
            dataType VAL;  // always read in a dataType and convert later if necessary

            fscanf(fid, " %d %d %lf \n", &ROW, &COL, &VAL);

            coo_row[ i ] = ROW - 1;
            coo_col[ i ] = COL - 1;
            coo_val[ i ] =  VAL;
        }
    } else if (mm_is_pattern(matcode) ) {
        for(int i = 0; i < *nnz; ++i) {
            int ROW, COL;

            fscanf(fid, " %d %d \n", &ROW, &COL );

            coo_row[ i ] = ROW - 1;
            coo_col[ i ] = COL - 1;
            coo_val[ i ] = 1.0;
        }
    } else if (mm_is_complex(matcode) ){
       for(int i = 0; i < *nnz; ++i) {
            int ROW, COL;
            dataType VAL, VALC;  // always read in a dataType and convert later if necessary

            fscanf(fid, " %d %d %lf %lf\n", &ROW, &COL, &VAL, &VALC);

            coo_row[ i ] = ROW - 1;
            coo_col[ i ] = COL - 1;
            coo_val[ i ] = VAL;//, VALC);
        }
        // printf(" ...successfully read complex matrix... ");
    } else {
        printf("\n%% Unrecognized data type\n");
        info = DEV_ERR_NOT_SUPPORTED;
        goto cleanup;
    }
    fclose(fid);
    fid = NULL;
    printf(" done. Converting to CSR:");
    fflush(stdout);


    if( mm_is_hermitian(matcode) ) {
        //hermitian = 1;
        printf("hermitian case!\n\n\n");
    }
    if ( mm_is_symmetric(matcode) || mm_is_hermitian(matcode) ) {
                                        // duplicate off diagonal entries
        printf("\n%% Detected symmetric case.");

        int off_diagonals = 0;
        for(int i = 0; i < *nnz; ++i) {
            if (coo_row[ i ] != coo_col[ i ])
                ++off_diagonals;
        }
        int true_nonzeros = 2*off_diagonals + (*nnz - off_diagonals);

        //printf("%% total number of nonzeros: %d\n%%", int(*nnz));

        //( data_index_malloc_cpu( &new_row, true_nonzeros ));
        //( data_index_malloc_cpu( &new_col, true_nonzeros ));
        //( data_zmalloc_cpu( &new_val, true_nonzeros ));
        // new_row = (int*) malloc( true_nonzeros*sizeof(int) );
        // new_col = (int*) malloc( true_nonzeros*sizeof(int) );
        // new_val = (dataType*) malloc( true_nonzeros*sizeof(dataType) );
        LACE_CALLOC( new_row, true_nonzeros );
        LACE_CALLOC( new_col, true_nonzeros );
        LACE_CALLOC( new_val, true_nonzeros );

        int ptr = 0;
        for(int i = 0; i < *nnz; ++i) {
            if (coo_row[ i ] != coo_col[ i ]) {
                new_row[ptr] = coo_row[ i ];
                new_col[ptr] = coo_col[ i ];
                new_val[ptr] = coo_val[ i ];
                ptr++;
                new_col[ptr] = coo_row[ i ];
                new_row[ptr] = coo_col[ i ];
                //new_val[ptr] = (hermitian == 0) ? coo_val[ i ] : conj(coo_val[ i ]);
                new_val[ptr] = coo_val[ i ];
                ptr++;
            } else {
                new_row[ptr] = coo_row[ i ];
                new_col[ptr] = coo_col[ i ];
                new_val[ptr] = coo_val[ i ];
                ptr++;
            }
        }

        free(coo_row);
        free(coo_col);
        free(coo_val);

        coo_row = new_row;
        coo_col = new_col;
        coo_val = new_val;

        *nnz = true_nonzeros;
    } // end symmetric case

    //( data_zmalloc_cpu( val, *nnz ) );
    //
    //( data_index_malloc_cpu( col, *nnz ) );
    //( data_index_malloc_cpu( row, (*n_row+1) ) );
    //( data_zmalloc_cpu( val, *nnz ) );
    // *row = (int*) malloc( (*n_row+1)*sizeof(int) );
    // *col = (int*) malloc( *nnz*sizeof(int) );
    // *val = (dataType*) malloc( *nnz*sizeof(dataType) );
    LACE_CALLOC( *row, (*n_row+1) );
    LACE_CALLOC( *col, *nnz );
    LACE_CALLOC( *val, *nnz );

    // original code from  Nathan Bell and Michael Garland
    for (int i = 0; i < num_rows; i++)
        (*row)[ i ] = 0;

    for (int i = 0; i < *nnz; i++)
        (*row)[coo_row[ i ]]++;

    // cumulative sum the nnz per row to get row[]
    int cumsum;
    cumsum = 0;
    for(int i = 0; i < num_rows; i++) {
        int temp = (*row)[ i ];
        (*row)[ i ] = cumsum;
        cumsum += temp;
    }
    (*row)[num_rows] = *nnz;

    // write Aj,Ax into Bj,Bx
    for(int i = 0; i < *nnz; i++) {
        int row_  = coo_row[ i ];
        int dest = (*row)[row_];
        (*col)[dest] = coo_col[ i ];
        (*val)[dest] = coo_val[ i ];
        (*row)[row_]++;
    }

    int last;
    last = 0;
    for(int i = 0; i <= num_rows; i++) {
        int temp  = (*row)[ i ];
        (*row)[ i ] = last;
        last = temp;
    }

    (*row)[*n_row] = *nnz;

    // sort column indices within each row
    // copy into vector of pairs (column index, value), sort by column index, then copy back
    for (int k=0; k < *n_row; ++k) {
        int kk  = (*row)[k];
        int len = (*row)[k+1] - (*row)[k];
        rowval.resize( len );
        for( int i=0; i < len; ++i ) {
            rowval[ i ] = std::make_pair( (*col)[kk+i], (*val)[kk+i] );
        }
        std::sort( rowval.begin(), rowval.end(), compare_first );
        for( int i=0; i < len; ++i ) {
            (*col)[kk+i] = rowval[ i ].first;
            (*val)[kk+i] = rowval[ i ].second;
        }
    }

    printf(" done.\n");
cleanup:
    if ( fid != NULL ) {
        fclose( fid );
        fid = NULL;
    }
    free(coo_row);
    free(coo_col);
    free(coo_val);
    return info;
}



extern "C"
int read_z_coo_from_mtx(
    data_storage_t *type,
    //data_location_t *location,
    int* n_row,
    int* n_col,
    int* nnz,
    dataType **coo_val,
    int **coo_row,
    int **coo_col,
    const char *filename )
{
    char buffer[ 1024 ];
    int info = 0;

    int *coo_colh=NULL, *coo_rowh=NULL;
    dataType *coo_valh=NULL;
    int *new_col=NULL, *new_row=NULL;
    dataType *new_val=NULL;
    //int hermitian = 0;

    FILE *fid = NULL;
    MM_typecode matcode;
    fid = fopen(filename, "r");

    if (fid == NULL) {
        printf("%% Unable to open file %s\n", filename);
        info = DEV_ERR_NOT_FOUND;
        goto cleanup;
    }

    printf("%% Reading sparse matrix from file (%s):", filename);
    fflush(stdout);

    if (mm_read_banner(fid, &matcode) != 0) {
        printf("\n%% Could not process Matrix Market banner: %s.\n", matcode);
        info = DEV_ERR_NOT_SUPPORTED;
        goto cleanup;
    }

    if (!mm_is_valid(matcode)) {
        printf("\n%% Invalid Matrix Market file.\n");
        info = DEV_ERR_NOT_SUPPORTED;
        goto cleanup;
    }

    if ( ! ( ( mm_is_real(matcode)    ||
               mm_is_integer(matcode) ||
               mm_is_pattern(matcode) ||
               mm_is_complex(matcode) ) &&
             mm_is_coordinate(matcode)  &&
             mm_is_sparse(matcode) ) )
    {
        mm_snprintf_typecode( buffer, sizeof(buffer), matcode );
        printf("\n%% Sorry, DEV-sparse does not support Market Market type: [%s]\n", buffer );
        printf("%% Only real-valued or pattern coordinate matrices are supported.\n");
        info = DEV_ERR_NOT_SUPPORTED;
        goto cleanup;
    }

    int num_rows, num_cols, num_nonzeros;
    if (mm_read_mtx_crd_size(fid, &num_rows, &num_cols, &num_nonzeros) != 0) {
        info = DEV_ERR_UNKNOWN;
        goto cleanup;
    }

    *type     = Magma_CSR;
    *n_row    = num_rows;
    *n_col    = num_cols;
    *nnz      = num_nonzeros;

    //( data_index_malloc_cpu( &coo_col, *nnz ) );
    //( data_index_malloc_cpu( &coo_row, *nnz ) );
    //( data_zmalloc_cpu( &coo_val, *nnz ) );
    // coo_rowh = (int*) malloc( *nnz*sizeof(int) );
    // coo_colh = (int*) malloc( *nnz*sizeof(int) );
    // coo_valh = (dataType*) malloc( *nnz*sizeof(dataType) );
    LACE_CALLOC( coo_rowh, *nnz );
    LACE_CALLOC( coo_colh, *nnz );
    LACE_CALLOC( coo_valh, *nnz );

    if (mm_is_real(matcode) || mm_is_integer(matcode)) {
        for(int i = 0; i < *nnz; ++i) {
            int ROW, COL;
            dataType VAL;  // always read in a dataType and convert later if necessary

            fscanf(fid, " %d %d %lf \n", &ROW, &COL, &VAL);

            coo_rowh[ i ] = ROW - 1;
            coo_colh[ i ] = COL - 1;
            coo_valh[ i ] = VAL;

        }
        // printf(" ...successfully read real matrix... ");
    } else if (mm_is_pattern(matcode) ) {
        for(int i = 0; i < *nnz; ++i) {
            int ROW, COL;

            fscanf(fid, " %d %d \n", &ROW, &COL );

            coo_rowh[ i ] = ROW - 1;
            coo_colh[ i ] = COL - 1;
            coo_valh[ i ] = 1.0;
        }
    } else if (mm_is_complex(matcode) ){
       for(int i = 0; i < *nnz; ++i) {
            int ROW, COL;
            dataType VAL, VALC;  // always read in a dataType and convert later if necessary

            fscanf(fid, " %d %d %lf %lf\n", &ROW, &COL, &VAL, &VALC);

            coo_rowh[ i ] = ROW - 1;
            coo_colh[ i ] = COL - 1;
            coo_valh[ i ] = VAL;//, VALC);
        }
        // printf(" ...successfully read complex matrix... ");
    } else {
        printf("\n%% Unrecognized data type\n");
        info = DEV_ERR_NOT_SUPPORTED;
        goto cleanup;
    }
    fclose(fid);
    fid = NULL;
    printf(" done. ");
    fflush(stdout);


    if( mm_is_hermitian(matcode) ) {
        //hermitian = 1;
        printf("hermitian case!\n\n\n");
    }
    if ( mm_is_symmetric(matcode) || mm_is_hermitian(matcode) ) {
                                        // duplicate off diagonal entries
        printf("\n%% Detected symmetric case.");

        int off_diagonals = 0;
        for(int i = 0; i < *nnz; ++i) {
            if (coo_rowh[ i ] != coo_colh[ i ])
                ++off_diagonals;
        }
        int true_nonzeros = 2*off_diagonals + (*nnz - off_diagonals);

        //printf("%% total number of nonzeros: %d\n%%", int(*nnz));

        //( data_index_malloc_cpu( &new_row, true_nonzeros ));
        //( data_index_malloc_cpu( &new_col, true_nonzeros ));
        //( data_zmalloc_cpu( &new_val, true_nonzeros ));
        // new_row = (int*) malloc( true_nonzeros*sizeof(int) );
        // new_col = (int*) malloc( true_nonzeros*sizeof(int) );
        // new_val = (dataType*) malloc( true_nonzeros*sizeof(dataType) );
        LACE_CALLOC( new_row, true_nonzeros );
        LACE_CALLOC( new_col, true_nonzeros );
        LACE_CALLOC( new_val, true_nonzeros );

        int ptr = 0;
        for(int i = 0; i < *nnz; ++i) {
            if (coo_rowh[ i ] != coo_colh[ i ]) {
                new_row[ptr] = coo_rowh[ i ];
                new_col[ptr] = coo_colh[ i ];
                new_val[ptr] = coo_valh[ i ];
                ptr++;
                new_col[ptr] = coo_rowh[ i ];
                new_row[ptr] = coo_colh[ i ];
                //new_val[ptr] = (hermitian == 0) ? coo_val[ i ] : conj(coo_val[ i ]);
                new_val[ptr] = coo_valh[ i ];
                ptr++;
            } else {
                new_row[ptr] = coo_rowh[ i ];
                new_col[ptr] = coo_colh[ i ];
                new_val[ptr] = coo_valh[ i ];
                ptr++;
            }
        }

        free(coo_rowh);
        free(coo_colh);
        free(coo_valh);
        *coo_row = new_row;
        *coo_col = new_col;
        *coo_val = new_val;
        free(new_row);
        free(new_col);
        free(new_val);
        *nnz = true_nonzeros;
    } // end symmetric case
    else {
        *coo_row = coo_rowh;
        *coo_col = coo_colh;
        *coo_val = coo_valh;
        //free(coo_rowh);
        //free(coo_colh);
        //free(coo_valh);
    }

    printf(" done.\n");
cleanup:
    if ( fid != NULL ) {
        fclose( fid );
        fid = NULL;
    }
//    free(coo_row);
//    free(coo_col);
//    free(coo_val);
    return info;
}

/**
    Purpose
    -------

    Writes a CSR matrix to a file using Matrix Market format.

    Arguments
    ---------

    @param[in]
    A           data_d_matrix
                matrix to write out

    @param[in]
    MajorType   int
                Row or Column sort
                default: 0 = RowMajor, 1 = ColMajor
                TODO: use named constants (e.g., MagmaRowMajor), not numbers.

    @param[in]
    filename    const char*
                output-filname of the mtx matrix

    @ingroup datasparse_zaux
    ********************************************************************/

extern "C"
int
data_zwrite_csr_mtx(
    data_d_matrix A,
    data_order_t MajorType,
    const char *filename )
{
    int info = 0;

    FILE *fp;
    data_d_matrix B = {Magma_CSR};

    if ( MajorType == MagmaColMajor && A.major != MagmaColMajor ) {
        // to obtain ColMajor output we transpose the matrix
        // and flip the row and col pointer in the output

        ( data_zmtranspose( A, &B ));

        // TODO avoid duplicating this code below.
        printf("%% Writing sparse matrix to file (%s):", filename);
        fflush(stdout);

        fp = fopen(filename, "w");
        if ( fp == NULL ){
            printf("\n%% error writing matrix: file exists or missing write permission\n");
            info = -1;
            goto cleanup;
        }

        //#define COMPLEX
        //
        //#ifdef COMPLEX
        //// complex case
        //fprintf( fp, "%%%%MatrixMarket matrix coordinate complex general\n" );
        //fprintf( fp, "%d %d %d\n", int(B.num_cols), int(B.num_rows), int(B.nnz));
        //
        //// TODO what's the difference between i (or i+1) and rowindex?
        //int i=0, j=0, rowindex=1;
        //
        //for(i=0; i < B.num_rows; i++) {
        //    int rowtemp1 = B.row[ i ];
        //    int rowtemp2 = B.row[i+1];
        //    for(j=0; j < rowtemp2 - rowtemp1; j++) {
        //        //fprintf( fp, "%d %d %.16g %.16g\n",
        //        fprintf( fp, "%d %d %.16g\n",
        //            ((B.col)[rowtemp1+j]+1), rowindex,
        //            //DEV_D_REAL((B.val)[rowtemp1+j]),
        //            //DEV_D_IMAG((B.val)[rowtemp1+j]) );
        //            ((B.val)[rowtemp1+j]) );
        //    }
        //    rowindex++;
        //}
        //#else
        // real case
        fprintf( fp, "%%%%MatrixMarket matrix coordinate real general\n" );
        fprintf( fp, "%d %d %d\n", int(B.num_cols), int(B.num_rows), int(B.nnz));

        // TODO what's the difference between i (or i+1) and rowindex?
        int i=0, j=0; //, rowindex=1;

        for(i=0; i < B.num_rows; i++) {
            int rowtemp1 = B.row[ i ];
            int rowtemp2 = B.row[i+1];
            for(j=0; j < rowtemp2 - rowtemp1; j++) {
                //fprintf( fp, "%d %d %.16g\n",
                //    ((B.col)[rowtemp1+j]+1), rowindex,
                //    ((B.val)[rowtemp1+j]) );
                fprintf( fp, "%d %d %.16g\n",
                    ((B.col)[rowtemp1+j]+1), (i+1),
                    ((B.val)[rowtemp1+j]) );
            }
            //rowindex++;
        }
        //#endif

        if (fclose(fp) != 0)
            printf("\n%% error: writing matrix failed\n");
        else
            printf(" done\n");

        fflush(stdout);
    }
    else {
        printf("%% Writing sparse matrix to file (%s):", filename);
        fflush(stdout);

        fp = fopen (filename, "w");
        if (  fp == NULL ){
            printf("\n%% error writing matrix: file exists or missing write permission\n");
            info = -1;
            goto cleanup;
        }


        //#define COMPLEX
        //
        //#ifdef COMPLEX
        //// complex case
        //fprintf( fp, "%%%%MatrixMarket matrix coordinate complex general\n" );
        //fprintf( fp, "%d %d %d\n", int(A.num_rows), int(A.num_cols), int(A.nnz));
        //
        //// TODO what's the difference between i (or i+1) and rowindex?
        //int i=0, j=0, rowindex=1;
        //
        //for(i=0; i < A.num_rows; i++) {
        //    int rowtemp1 = A.row[ i ];
        //    int rowtemp2 = A.row[i+1];
        //    for(j=0; j < rowtemp2 - rowtemp1; j++) {
        //        //fprintf( fp, "%d %d %.16g %.16g\n",
        //        fprintf( fp, "%d %d %.16g\n",
        //            rowindex, ((A.col)[rowtemp1+j]+1),
        //            //DEV_D_REAL((A.val)[rowtemp1+j]),
        //            //DEV_D_IMAG((A.val)[rowtemp1+j]) );
        //            ((A.val)[rowtemp1+j]) );
        //    }
        //    rowindex++;
        //}
        //#else
        // real case
        fprintf( fp, "%%%%MatrixMarket matrix coordinate real general\n" );
        fprintf( fp, "%d %d %d\n", int(A.num_rows), int(A.num_cols), int(A.nnz));

        // TODO what's the difference between i (or i+1) and rowindex?
        int i=0, j=0, rowindex=1;

        for(i=0; i < A.num_rows; i++) {
            int rowtemp1 = A.row[ i ];
            int rowtemp2 = A.row[i+1];
            for(j=0; j < rowtemp2 - rowtemp1; j++) {
                fprintf( fp, "%d %d %.16g\n",
                    rowindex, ((A.col)[rowtemp1+j]+1),
                    ((A.val)[rowtemp1+j]));
            }
            rowindex++;
        }
        //#endif

        if (fclose(fp) != 0)
            printf("\n%% error: writing matrix failed\n");
        else
            printf(" done\n");

        fflush(stdout);
    }
cleanup:
    return info;
}


/**
    Purpose
    -------

    Prints a CSR matrix in Matrix Market format.

    Arguments
    ---------

    @param[in]
    n_row       int*
                number of rows in matrix

    @param[in]
    n_col       int*
                number of columns in matrix

    @param[in]
    nnz         int*
                number of nonzeros in matrix

    @param[in]
    val         dataType**
                value array of CSR

    @param[in]
    row         int**
                row pointer of CSR

    @param[in]
    col         int**
                column indices of CSR

    @param[in]
    MajorType   int
                Row or Column sort
                default: 0 = RowMajor, 1 = ColMajor

    @ingroup datasparse_zaux
    ********************************************************************/

extern "C"
int
data_zprint_csr_mtx(
    int n_row,
    int n_col,
    int nnz,
    dataType **val,
    int **row,
    int **col,
    data_order_t MajorType )
{
    int info = 0;
    printf( "%%%%MatrixMarket matrix coordinate real general\n" );
    printf( "%d %d %d\n", int(n_col), int(n_row), int(nnz));

    int i=0, j=0;

    for(i=0; i < n_row; i++) {
        int rowtemp1 = (*row)[ i ];
        int rowtemp2 = (*row)[i+1];
        for(j=0; j < rowtemp2 - rowtemp1; j++) {
            printf( "%d %d %.6e\n",
                i+1, ((*col)[rowtemp1+j]+1),
                ((*val)[rowtemp1+j]) );
        }
    }

//cleanup:
    return info;
}

extern "C"
int
data_zprint_csr(
    data_d_matrix A )
{
    int info = 0;
    if (A.pad_rows > 0 && A.pad_cols > 0)
      data_zprint_csr_mtx( A.pad_rows, A.pad_cols, A.nnz, &A.val, &A.row, &A.col, A.major );
    else
      data_zprint_csr_mtx( A.num_rows, A.num_cols, A.nnz, &A.val, &A.row, &A.col, A.major );
    return info;
}

extern "C"
int
data_zwrite_csr(
    data_d_matrix* A ) {

    int info = 0;
    int rowlimit = A->num_rows+1;
    if (A->pad_rows > 0)
      rowlimit = A->pad_rows+1;
    for ( int i = 0; i < rowlimit; i++ ) {
     printf("%d\t", A->row[i]);
    }
    printf("\n");
    for ( int i = 0; i < A->nnz; i++ ) {
     printf("%d\t", A->col[i]);
    }
    printf("\n");for ( int i = 0; i < A->nnz; i++ ) {
     printf("%e\t", A->val[i]);
    }
    printf("\n");
    return info;
}

extern "C"
int
data_zprint_coo_mtx(
    int n_row,
    int n_col,
    int nnz,
    dataType **val,
    int **row,
    int **col )
{
    int info = 0;
    printf( "%%%%MatrixMarket matrix coordinate real general\n" );
    printf( "%d %d %d\n", int(n_col), int(n_row), int(nnz));

    for(int i=0; i < nnz; i++) {
      printf( "%d %d %.6e\n", ((*row)[ i ]+1), ((*col)[ i ]+1), ((*val)[ i ]));
    }
    //#endif

//cleanup:
    return info;
}

extern "C"
int
data_zprint_coo(
    data_d_matrix A )
{
    int info = 0;
    data_zprint_coo_mtx( A.num_rows, A.num_cols, A.nnz, &A.val, &A.row, &A.col );
    return info;
}

extern "C"
int
data_zprint_bcsr(
    data_d_matrix* A )
{
  int info = 0;

  printf("blocksize = %d\n", A->blocksize);
  printf("numblocks = %d\n", A->numblocks);
  printf("nnz = %d\n", A->nnz);
  printf("true_nnz = %d\n", A->true_nnz);
  printf("bsr_num_rows = %d\n", A->num_rows);
  for (int i=0; i<A->num_rows; i++ ) {
    printf("row %d:\n", i);
    for (int j=A->row[i]; j<A->row[i+1]; j++) {
      printf("block %d bcol %d\n", j, A->col[j]);
      for (int k=0; k<A->ldblock; k++ ) {
        printf("%e ", A->val[j*A->ldblock+k]);
        if ((k+1)%A->blocksize==0)
          printf("\n");
      }
      printf("\n");
    }
    printf("\n");
  }
  printf("bsrrows:\n");
  for (int i=0; i<A->num_rows+1; i++ ) {
    printf("%d, ", A->row[i]);
  }
  printf("\nbsrcols:\n");
  for (int i=0; i<A->numblocks; i++ ) {
    printf("%d, ", A->col[i]);
  }
  printf("\nabsr:\n");
  for (int i=0; i<A->numblocks*A->ldblock; i++ ) {
    printf("%e, ", A->val[i]);
  }
  printf("\n");
    return info;
}


/**
    Purpose
    -------

    Reads in a matrix stored in coo format from a Matrix Market (.mtx)
    file and converts it into CSR format. It duplicates the off-diagonal
    entries in the symmetric case.

    Arguments
    ---------

    @param[out]
    A           data_d_matrix*
                matrix in data sparse matrix format

    @param[in]
    filename    const char*
                filname of the mtx matrix

    @ingroup datasparse_zaux
    ********************************************************************/

extern "C"
int
data_z_csr_mtx(
    data_d_matrix *A,
    const char *filename )
{
  int info = 0;
  data_storage_t A_storage = Magma_CSR;
  read_z_csr_from_mtx( &A_storage, &A->num_rows, &A->num_cols, &A->nnz, &A->val,
    &A->row, &A->col, filename );
  A->true_nnz = A->nnz;
  A->major = MagmaRowMajor;
  return info;
}



extern "C"
int
data_z_coo_mtx(
    data_d_matrix *A,
    const char *filename )
{
  int info = 0;
  data_storage_t A_storage = Magma_COO;
  read_z_coo_from_mtx( &A_storage, &A->num_rows, &A->num_cols, &A->nnz, &A->val,
    &A->row, &A->col, filename );
  return info;
}



extern "C"
int
data_z_pad_csr(
    data_d_matrix *A,
    int tile_size )
{
  int info = 0;

  dataType * valtmp;
  int * coltmp;
  int * rowtmp;
  // valtmp = (dataType*) malloc( A->nnz*sizeof(dataType) );
  // coltmp = (int*) malloc( A->nnz*sizeof(int) );
  // rowtmp = (int*) malloc( (A->num_rows+1)*sizeof(int) );
  LACE_CALLOC( valtmp, A->nnz );
  LACE_CALLOC( coltmp, A->nnz );
  LACE_CALLOC( rowtmp, (A->num_rows+1) );
  for( int i = 0; i < A->nnz; ++i ) {
    valtmp[ i ] = A->val[ i ];
    coltmp[ i ] = A->col[ i ];
  }
  free( A->val );
  free( A->col );
  for( int i = 0; i < A->num_rows+1; ++i ) {
    rowtmp[ i ] = A->row[ i ];
  }
  free( A->row );

  // round up num_rows and num_cols to smallest size evenly divisible by tile_size
  A->pad_rows = ceil( float(A->num_rows)/tile_size)*tile_size;
  A->pad_cols = ceil( float(A->num_cols)/tile_size)*tile_size;

  printf("tile_size = %d num_rows = %d pad_rows = %d \n", tile_size, A->num_rows, A->pad_rows);
  printf(" %d additional unit diagonal terms added to pad sparse matrix in csr format\n",
    A->pad_rows - A->num_rows);

  A->true_nnz = A->nnz;
  A->nnz = A->nnz + (A->pad_rows - A->num_rows);
  // A->val = (dataType*) malloc( A->nnz*sizeof(dataType) );
  // A->col = (int*) malloc( A->nnz*sizeof(int) );
  // A->row = (int*) malloc( (A->pad_rows+1)*sizeof(int) );
  LACE_CALLOC( A->val, A->nnz );
  LACE_CALLOC( A->col, A->nnz );
  LACE_CALLOC( A->row, (A->pad_rows+1) );

  for ( int i = 0; i < A->true_nnz; i++ ) {
    A->val[ i ] = valtmp[ i ];
    A->col[ i ] = coltmp[ i ];
  }
  for ( int i = A->true_nnz; i < A->nnz; i++ ) {
    A->val[ i ] = 1.0;
    A->col[ i ] = A->num_cols + (i - A->true_nnz);
  }
  for( int i = 0; i < A->num_rows+1; ++i ) {
    A->row[ i ] = rowtmp[ i ];
  }
  for( int i = 0; i < (A->pad_rows - A->num_rows); ++i ) {
    A->row[ A->num_rows+1+i ] = (i + A->true_nnz +1 );
  }

  //for ( int i = 0; i < A->nnz; i++ ) {
  // printf("%e\t", A->val[i]);
  //}
  //printf("\n");
  //for ( int i = 0; i < A->nnz; i++ ) {
  // printf("%d\t", A->col[i]);
  //}
  //printf("\n");
  //for ( int i = 0; i < A->pad_rows+1; i++ ) {
  // printf("%d\t", A->row[i]);
  //}
  //printf("\n");

  A->ld = A->pad_rows;

  free( valtmp );
  free( rowtmp );
  free( coltmp );

  return info;
}



extern "C"
int read_z_dense_from_mtx(
    data_storage_t *type,
    int* n_row,
    int* n_col,
    int* nnz,
    data_order_t major,
    dataType **val,
    const char *filename )
{
    char buffer[ 1024 ];
    int info = 0;

    //int *coo_col=NULL, *coo_row=NULL;
    dataType *coo_val=NULL;
    //int *new_col=NULL, *new_row=NULL;
    //dataType *new_val=NULL;
    //int hermitian = 0;

    std::vector< std::pair< int, dataType > > rowval;

    FILE *fid = NULL;
    MM_typecode matcode;
    fid = fopen(filename, "r");

    if (fid == NULL) {
        printf("%% Unable to open file %s\n", filename);
        info = DEV_ERR_NOT_FOUND;
        goto cleanup;
    }

    printf("%% Reading dense matrix from file (%s):", filename);
    fflush(stdout);

    if (mm_read_banner(fid, &matcode) != 0) {
        printf("\n%% Could not process Matrix Market banner: %s.\n", matcode);
        info = DEV_ERR_NOT_SUPPORTED;
        goto cleanup;
    }

    if (!mm_is_valid(matcode)) {
        printf("\n%% Invalid Matrix Market file.\n");
        info = DEV_ERR_NOT_SUPPORTED;
        goto cleanup;
    }

    if ( ! ( ( mm_is_real(matcode)    ||
               mm_is_integer(matcode) ||
               mm_is_pattern(matcode) ||
               mm_is_complex(matcode) ) &&
             //mm_is_coordinate(matcode)  &&
             mm_is_array(matcode)  &&
             mm_is_dense(matcode) ) )
    {
        mm_snprintf_typecode( buffer, sizeof(buffer), matcode );
        printf("\n%% Sorry, read_z_dense_from_mtx() does not support Market Market type: [%s]\n", buffer );
        printf("%% Only real-valued or pattern coordinate matrices are supported.\n");
        info = DEV_ERR_NOT_SUPPORTED;
        goto cleanup;
    }

    int num_rows, num_cols; //, num_nonzeros;
    if (mm_read_mtx_array_size(fid, &num_rows, &num_cols) != 0) {
        info = DEV_ERR_UNKNOWN;
        goto cleanup;
    }

    *type     = Magma_DENSE;
    *n_row    = num_rows;
    *n_col    = num_cols;
    *nnz      = num_rows*num_cols;

    //( data_index_malloc_cpu( &coo_col, *nnz ) );
    //( data_index_malloc_cpu( &coo_row, *nnz ) );
    //( data_zmalloc_cpu( &coo_val, *nnz ) );
    //coo_row = (int*) malloc( *nnz*sizeof(int) );
    //coo_col = (int*) malloc( *nnz*sizeof(int) );
    // coo_val = (dataType*) malloc( *nnz*sizeof(dataType) );
    LACE_CALLOC( coo_val, *nnz );

    if (mm_is_real(matcode) || mm_is_integer(matcode)) {
        for(int i = 0; i < *nnz; ++i) {
            //int ROW, COL;
            dataType VAL;  // always read in a dataType and convert later if necessary

            //fscanf(fid, " %d %d %lf \n", &ROW, &COL, &VAL);
            fscanf(fid, " %lf \n",&VAL);

            //coo_row[ i ] = ROW - 1;
            //coo_col[ i ] = COL - 1;
            coo_val[ i ] = VAL;
        }
    } else if (mm_is_pattern(matcode) ) {
        for(int i = 0; i < *nnz; ++i) {
            int ROW;

            fscanf(fid, " %d \n",&ROW);

            //coo_row[ i ] = ROW - 1;
            //coo_col[ i ] = COL - 1;
            coo_val[ i ] = 1.0;
        }
    } else if (mm_is_complex(matcode) ){
       for(int i = 0; i < *nnz; ++i) {
            //int ROW, COL;
            dataType VAL, VALC;  // always read in a dataType and convert later if necessary

            //fscanf(fid, " %d %d %lf %lf\n", &ROW, &COL, &VAL, &VALC);
            fscanf(fid, " %lf %lf\n", &VAL, &VALC);

            //coo_row[ i ] = ROW - 1;
            //coo_col[ i ] = COL - 1;
            coo_val[ i ] = VAL;//, VALC);
        }
        // printf(" ...successfully read complex matrix... ");
    } else {
        printf("\n%% Unrecognized data type\n");
        info = DEV_ERR_NOT_SUPPORTED;
        goto cleanup;
    }
    fclose(fid);
    fid = NULL;
    printf(" done. ");
    fflush(stdout);

    *val = coo_val;

    printf(" done.\n");
cleanup:
    if ( fid != NULL ) {
        fclose( fid );
        fid = NULL;
    }
    //free(coo_row);
    //free(coo_col);
    //free(coo_val);
    return info;
}

extern "C"
int
data_zprint_dense_mtx(
    int n_row,
    int n_col,
    int nnz,
    data_order_t major,
    dataType **val )
{
    int info = 0;

    printf( "%%%%MatrixMarket matrix array real general\n" );
    if (major == MagmaRowMajor )
      printf( "%d %d\n", int(n_row), int(n_col));
    else
      printf( "%d %d\n", int(n_col), int(n_row));

    for(int i=0; i < n_col*n_row; i++) {
      printf( "%.6e\n", ((*val)[ i ]));
    }

//cleanup:
    return info;
}

extern "C"
int
data_zwrite_dense_mtx(
    int n_row,
    int n_col,
    int nnz,
    data_order_t major,
    dataType **val,
    const char* filename )
{
    int info = 0;

    printf("%% Writing dense matrix to file (%s):", filename);
    fflush(stdout);

    FILE *fp;
    fp = fopen(filename, "w");
    if ( fp == NULL ){
        printf("\n%% error writing matrix: file exists or missing write permission\n");
        info = -1;
        goto cleanup;
    }

    fprintf( fp, "%%%%MatrixMarket matrix array real general\n" );
    //if (major == MagmaRowMajor )
      fprintf( fp, "%d %d\n", int(n_row), int(n_col));
    //else
    //  fprintf( fp, "%d %d\n", int(n_col), int(n_row));

    for(int i=0; i < n_col*n_row; i++) {
      fprintf( fp, "%.16e\n", ((*val)[ i ]));
    }

    if (fclose(fp) != 0)
        printf("\n%% error: writing matrix failed\n");
    else
        printf(" done\n");

    fflush(stdout);

cleanup:
    return info;
}

extern "C"
int
data_z_dense_mtx(
    data_d_matrix *A,
    data_order_t major,
    const char *filename )
{
  int info = 0;
  data_storage_t A_storage = Magma_CSR;
  read_z_dense_from_mtx( &A_storage, &A->num_rows, &A->num_cols, &A->nnz, major,
    &A->val, filename );
  A->true_nnz = A->nnz;
  if (major == MagmaRowMajor)
    A->ld = A->num_cols;
  else if (major == MagmaColMajor)
    A->ld = A->num_rows;
  A->major = major;

  return info;
}

extern "C"
int
data_zprint_dense(
    data_d_matrix A )
{
    int info = 0;
    if ( A.storage_type == Magma_DENSED ) {
      printf("\nprinting a dense diagonal\n");
      data_zprint_dense_mtx( A.nnz, 1, A.nnz, A.major, &A.val );
    }
    else if (A.pad_rows > 0 && A.pad_cols > 0)
      data_zprint_dense_mtx( A.pad_rows, A.pad_cols, A.nnz, A.major, &A.val );

    else
      data_zprint_dense_mtx( A.num_rows, A.num_cols, A.nnz, A.major, &A.val );
    return info;
}

extern "C"
int
data_zwrite_dense(
    data_d_matrix A,
    const char* filename )
{
    int info = 0;
    if ( A.storage_type == Magma_DENSED ) {
      printf("\nprinting a dense diagonal\n");
      info = data_zwrite_dense_mtx( A.nnz, 1, A.nnz, A.major, &A.val, filename );
    }
    else if (A.pad_rows > 0 && A.pad_cols > 0)
      info = data_zwrite_dense_mtx( A.pad_rows, A.pad_cols, A.nnz, A.major, &A.val, filename );
    else
      info = data_zwrite_dense_mtx( A.num_rows, A.num_cols, A.nnz, A.major, &A.val, filename );
    return info;
}

extern "C"
int
data_zdisplay_dense(
    data_d_matrix* A )
{
    int info = 0;
    int row_limit = A->num_rows;
    int col_limit = A->num_cols;

    if (A->pad_rows > 0 && A->pad_cols > 0) {
      row_limit = A->pad_rows;
      col_limit = A->pad_cols;
    }

    printf( "%%%%MatrixMarket matrix array real general\n" );
    if (A->major == MagmaRowMajor ) {
      printf( "%d %d\n", int(row_limit), int(col_limit));
      for(int i=0; i < row_limit; i++) {
        for(int j=0; j < col_limit; j++) {
          printf( "%.6e ", (A->val[ i * A->ld + j ]));
        }
        printf("\n");
      }
    }
    else {
      printf( "%d %d\n", int(row_limit), int(col_limit));
      for(int i=0; i < row_limit; i++) {
        for(int j=0; j < col_limit; j++) {
          printf( "%.6e ", (A->val[ i + j * A->ld ]));
        }
        printf("\n");
      }
    }
    printf("\n");

//cleanup:
    return info;
}

extern "C"
int
data_z_pad_dense(
    data_d_matrix *A,
    int tile_size )
{
  int info = 0;
  int old_nnz = A->num_rows*A->num_cols;
  dataType * valtmp;
  // valtmp = (dataType*) malloc( old_nnz*sizeof(dataType) );
  LACE_CALLOC( valtmp, old_nnz );

  //#pragma omp parallel
  //{
  //  #pragma omp for nowait
    for( int i = 0; i < old_nnz; ++i ) {
      valtmp[ i ] = A->val[ i ];
    }
  //}

  free( A->val );

  // round up num_rows and num_cols to smallest size evenly divisible by tile_size
  A->pad_rows = ceil( float(A->num_rows)/tile_size )*tile_size;
  A->pad_cols = ceil( float(A->num_cols)/tile_size )*tile_size;
  A->nnz = A->pad_rows*A->pad_cols;

  printf("tile_size = %d num_rows = %d pad_rows = %d \n", tile_size, A->num_rows, A->pad_rows);

  A->val = (dataType*) calloc( A->nnz, sizeof(dataType) );

  if ( A->major == MagmaRowMajor ) {
    for ( int i = 0; i < A->num_rows; i++ ) {
      for ( int j = 0; j < A->num_cols; j++ ) {
        A->val[ i*A->pad_cols + j ] = valtmp[ i*A->num_cols + j ];
      }
    }
    for ( int i = A->num_rows; i < A->pad_rows; i++ ) {
      A->val[ i*A->pad_cols + i ] = 1.0;
    }
    A->ld = A->pad_cols;
  }
  else if ( A->major == MagmaColMajor ){
    for ( int j = 0; j < A->num_cols; j++ ) {
      for ( int i = 0; i < A->num_rows; i++ ) {
        A->val[ i + j*A->pad_rows ] = valtmp[ i + j*A->num_rows ];
      }
    }
    for ( int i = A->num_cols; i < A->pad_cols; i++ ) {
      A->val[ i + i*A->pad_rows ] = 1.0;
    }
    A->ld = A->pad_rows;
  }

  printf("pad done\n");
  fflush(stdout);

  free( valtmp );

  return info;
}
