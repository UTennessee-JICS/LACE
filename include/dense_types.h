#ifndef DEV_TYPES_H
#define DEV_TYPES_H

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#define DEV_ALIGN      (64)
#define DEV_STRIP      (1024)

#define DEV_D_ZERO     ( 0.0)
#define DEV_D_ONE      ( 1.0)
#define DEV_D_HALF     ( 0.5)
#define DEV_D_NEG_ONE  (-1.0)
#define DEV_D_NEG_HALF (-0.5)

#define DEV_D_MAKE(r, i)  (r)
#define DEV_D_REAL(x)     (x)
#define DEV_D_IMAG(x)     (0.0)
#define DEV_D_ADD(a, b)   ((a) + (b))
#define DEV_D_SUB(a, b)   ((a) - (b))
#define DEV_D_MUL(a, b)   ((a) * (b))
#define DEV_D_DIV(a, b)   ((a) / (b))
#define DEV_D_ABS(a)      ((a) > 0 ? (a) : -(a))
#define DEV_D_ABS1(a)     ((a) > 0 ? (a) : -(a))
#define DEV_D_CONJ(a)     (a)
#define DEV_D_EQUAL(a, b) ((a) == (b))
#define DEV_D_NEGATE(a)   (-a)

#define DEV_CHECKPT \
  { printf("Checkpoint: %s, line %d\n", __FILE__, __LINE__); \
    fflush(stdout); }

#ifdef DEBUG_P
  # define DEV_PRINTF(f_, ...) \
    { \
      printf((f_), __VA_ARGS__); \
      fflush(stdout); }
#else
  # define DEV_PRINTF(f_, ...) \
    { \
      (void) 0; }
#endif

#ifdef DEBUG_PARILU
  # define PARILUDBG(f_, ...) \
    { \
      printf((f_), __VA_ARGS__); \
      fflush(stdout); }
#else
  # define PARILUDBG(f_, ...) \
    { \
      (void) 0; }
#endif

#ifdef DEBUG_PARTRSV
  # define PARTRSVDBG(f_, ...) \
    { \
      printf((f_), __VA_ARGS__); \
      fflush(stdout); }
#else
  # define PARTRSVDBG(f_, ...) \
    { \
      (void) 0; }
#endif

#ifdef DEBUG_GMRES
  # define GMRESDBG(f_, ...) \
    { \
      printf((f_), __VA_ARGS__); \
      fflush(stdout); }
#else
  # define GMRESDBG(f_, ...) \
    { \
      (void) 0; }
#endif

#ifdef DEBUG_ORTHOG
  # define ORTHOGDBG(f_, ...) \
    { \
      printf((f_), __VA_ARGS__); \
      fflush(stdout); }
#else
  # define ORTHOGDBG(f_, ...) \
    { \
      (void) 0; }
#endif

#define idx(i, j, n) ((i) + (j) * (n))

#define MIN(a, b)    (((a) < (b)) ? (a) : (b))
#define MAX(a, b)    (((a) > (b)) ? (a) : (b))
#define SIGN(a, b)   ((b) < 0 ? -fabs(a) : fabs(a))

// ========================================
// To use int64_t, link with mkl_intel_ilp64 or similar (instead of mkl_intel_lp64).
// Similar to data_int_t we declare data_index_t used for row/column indices in sparse
#if defined(DEV_ILP64) || defined(MKL_ILP64)
  // typedef int64_t data_int_t;
  typedef long long int data_int_t; // MKL uses long long int, not int64_t
#else
  typedef int data_int_t;
#endif

typedef int data_index_t;
typedef int data_int_t;

// Define new type that the precision generator will not change (matches PLASMA)

#ifndef DATA_TYPE_H
  # define DATA_TYPE_H
  typedef double dataType;
#endif
typedef dataType           * dataType_ptr;
typedef dataType const * dataType_const_ptr;

typedef double real_Double_t;
typedef double dataDouble;

typedef void               * data_ptr;
typedef data_int_t         * dataInt_ptr;
typedef data_index_t       * dataIndex_ptr;
typedef float              * dataFloat_ptr;
typedef double             * dataDouble_ptr;

typedef void const * data_const_ptr;
typedef data_int_t const * dataInt_const_ptr;
typedef data_index_t const * dataIndex_const_ptr;
typedef float const * dataFloat_const_ptr;
typedef double const * dataDouble_const_ptr;

// ========================================
// opaque queue structure
struct data_queue;
typedef struct data_queue * data_queue_t;

// ----------------------------------------
// Return codes
// LAPACK argument errors are < 0 but > DEV_ERR.
// DEV errors are < DEV_ERR.
#define DEV_SUCCESS              0
#define DEV_ERR                  -100
#define DEV_ERR_NOT_INITIALIZED  -101
#define DEV_ERR_REINITIALIZED    -102 // unused
#define DEV_ERR_NOT_SUPPORTED    -103
#define DEV_ERR_ILLEGAL_VALUE    -104 // unused
#define DEV_ERR_NOT_FOUND        -105
#define DEV_ERR_ALLOCATION       -106 // unused
#define DEV_ERR_INTERNAL_LIMIT   -107 // unused
#define DEV_ERR_UNALLOCATED      -108 // unused
#define DEV_ERR_FILESYSTEM       -109 // unused
#define DEV_ERR_UNEXPECTED       -110 // unused
#define DEV_ERR_SEQUENCE_FLUSHED -111 // unused
#define DEV_ERR_HOST_ALLOC       -112
#define DEV_ERR_DEVICE_ALLOC     -113
#define DEV_ERR_CUDASTREAM       -114 // unused
#define DEV_ERR_INVALID_PTR      -115
#define DEV_ERR_UNKNOWN          -116
#define DEV_ERR_NOT_IMPLEMENTED  -117
#define DEV_ERR_NAN              -118

// some sparse errors
#define DEV_SLOW_CONVERGENCE -201
#define DEV_DIVERGENCE       -202
#define DEV_NONSPD           -203
#define DEV_ERR_BADPRECOND   -204
#define DEV_NOTCONVERGED     -205

// ----------------------------------------
// parameter constants
// numbering is consistent with CBLAS and PLASMA; see plasma/include/plasma.h
// also with lapack_cwrapper/include/lapack_enum.h
// see http://www.netlib.org/lapack/lapwrapc/
typedef enum {
  MagmaFalse = 0,
  MagmaTrue  = 1
} data_bool_t;

typedef enum {
  MagmaRowMajor = 101,
  MagmaColMajor = 102
} data_order_t;

// Magma_ConjTrans is an alias for those rare occasions (zlarfb, zun*, zher*k)
// where we want Magma_ConjTrans to convert to MagmaTrans in precision generation.
typedef enum {
  MagmaNoTrans    = 111,
  MagmaTrans      = 112,
  MagmaConjTrans  = 113,
  Magma_ConjTrans = MagmaConjTrans
} data_trans_t;

typedef enum {
  MagmaUpper      = 121,
  MagmaLower      = 122,
  MagmaFull       = 123, /* lascl, laset */
  MagmaHessenberg = 124  /* lascl */
} data_uplo_t;

typedef data_uplo_t data_type_t; /* lascl */

typedef enum {
  MagmaNonUnit = 131,
  MagmaUnit    = 132
} data_diag_t;

typedef enum {
  MagmaLeft      = 141,
  MagmaRight     = 142,
  MagmaBothSides = 143 /* trevc */
} data_side_t;

typedef enum {
  MagmaOneNorm       = 171, /* lange, lanhe */
  MagmaRealOneNorm   = 172,
  MagmaTwoNorm       = 173,
  MagmaFrobeniusNorm = 174,
  MagmaInfNorm       = 175,
  MagmaRealInfNorm   = 176,
  MagmaMaxNorm       = 177,
  MagmaRealMaxNorm   = 178
} data_norm_t;

typedef enum {
  MagmaDistUniform   = 201, /* latms */
  MagmaDistSymmetric = 202,
  MagmaDistNormal    = 203
} data_dist_t;

typedef enum {
  MagmaHermGeev   = 241, /* latms */
  MagmaHermPoev   = 242,
  MagmaNonsymPosv = 243,
  MagmaSymPosv    = 244
} data_sym_t;

typedef enum {
  MagmaNoPacking     = 291, /* latms */
  MagmaPackSubdiag   = 292,
  MagmaPackSupdiag   = 293,
  MagmaPackColumn    = 294,
  MagmaPackRow       = 295,
  MagmaPackLowerBand = 296,
  MagmaPackUpeprBand = 297,
  MagmaPackAll       = 298
} data_pack_t;

typedef enum {
  MagmaNoVec        = 301, /* geev, syev, gesvd */
  MagmaVec          = 302, /* geev, syev */
  MagmaIVec         = 303, /* stedc */
  MagmaAllVec       = 304, /* gesvd, trevc */
  MagmaSomeVec      = 305, /* gesvd, trevc */
  MagmaOverwriteVec = 306, /* gesvd */
  MagmaBacktransVec = 307  /* trevc */
} data_vec_t;

typedef enum {
  MagmaRangeAll = 311, /* syevx, etc. */
  MagmaRangeV   = 312,
  MagmaRangeI   = 313
} data_range_t;

typedef enum {
  MagmaQ = 322, /* unmbr, ungbr */
  MagmaP = 323
} data_vect_t;

typedef enum {
  MagmaForward  = 391, /* larfb */
  MagmaBackward = 392
} data_direct_t;

typedef enum {
  MagmaColumnwise = 401, /* larfb */
  MagmaRowwise    = 402
} data_storev_t;

// -----------------------------------------------------------------------------
// sparse
typedef enum {
  Magma_CSR          = 611,
  Magma_ELLPACKT     = 612,
  Magma_ELL          = 613,
  Magma_DENSE        = 614,
  Magma_BCSR         = 615,
  Magma_CSC          = 616,
  Magma_HYB          = 617,
  Magma_COO          = 618,
  Magma_ELLRT        = 619,
  Magma_SPMVFUNCTION = 620,
  Magma_SELLP        = 621,
  Magma_ELLD         = 622,
  Magma_CSRLIST      = 623,
  Magma_CSRD         = 624,
  Magma_CSRL         = 627,
  Magma_CSRU         = 628,
  Magma_CSRCOO       = 629,
  Magma_CUCSR        = 630,
  Magma_COOLIST      = 631,
  Magma_CSR5         = 632,
  Magma_CSCD         = 633,
  Magma_CSCL         = 634,
  Magma_CSCU         = 635,
  Magma_CSCCOO       = 636,
  Magma_DENSEL       = 637,
  Magma_DENSEU       = 638,
  Magma_DENSED       = 639,
  Magma_BCSRD        = 640,
  Magma_BCSRL        = 641,
  Magma_BCSRU        = 642,
  Magma_BCSC         = 643,
  Magma_BCSCD        = 644,
  Magma_BCSCL        = 645,
  Magma_BCSCU        = 646
} data_storage_t;

typedef enum {
  Magma_CG             = 431,
  Magma_CGMERGE        = 432,
  Magma_GMRES          = 433,
  Magma_BICGSTAB       = 434,
  Magma_BICGSTABMERGE  = 435,
  Magma_BICGSTABMERGE2 = 436,
  Magma_JACOBI         = 437,
  Magma_GS             = 438,
  Magma_ITERREF        = 439,
  Magma_BCSRLU         = 440,
  Magma_PCG            = 441,
  Magma_PGMRES         = 442,
  Magma_PBICGSTAB      = 443,
  Magma_PASTIX         = 444,
  Magma_ILU            = 445,
  Magma_ICC            = 446,
  Magma_PARILU         = 447,
  Magma_PARIC          = 448,
  Magma_BAITER         = 449,
  Magma_LOBPCG         = 450,
  Magma_NONE           = 451,
  Magma_FUNCTION       = 452,
  Magma_IDR            = 453,
  Magma_PIDR           = 454,
  Magma_CGS            = 455,
  Magma_PCGS           = 456,
  Magma_CGSMERGE       = 457,
  Magma_PCGSMERGE      = 458,
  Magma_TFQMR          = 459,
  Magma_PTFQMR         = 460,
  Magma_TFQMRMERGE     = 461,
  Magma_PTFQMRMERGE    = 462,
  Magma_QMR            = 463,
  Magma_PQMR           = 464,
  Magma_QMRMERGE       = 465,
  Magma_PQMRMERGE      = 466,
  Magma_BOMBARD        = 490,
  Magma_BOMBARDMERGE   = 491,
  Magma_PCGMERGE       = 492,
  Magma_BAITERO        = 493,
  Magma_IDRMERGE       = 494,
  Magma_PBICGSTABMERGE = 495,
  Magma_PARICT         = 496,
  Magma_CUSTOMIC       = 497,
  Magma_CUSTOMILU      = 498,
  Magma_PIDRMERGE      = 499,
  Magma_BICG           = 500,
  Magma_BICGMERGE      = 501,
  Magma_PBICG          = 502,
  Magma_PBICGMERGE     = 503,
  Magma_LSQR           = 504,
  Magma_PARILUT        = 505,
  Magma_ISAI           = 506,
  Magma_CUSOLVE        = 507,
  Magma_VBJACOBI       = 508,
  Magma_PARDISO        = 509,
  Magma_SPTRSV         = 510
} data_solver_type;

typedef enum {
  Magma_CGSO       = 561,
  Magma_FUSED_CGSO = 562,
  Magma_MGSO       = 563
} data_ortho_t;

typedef enum {
  Magma_GENERAL   = 581,
  Magma_SYMMETRIC = 582
} data_symmetry_t;

typedef enum {
  Magma_ORDERED   = 591,
  Magma_DIAGFIRST = 592,
  Magma_UNITY     = 593,
  Magma_VALUE     = 594,
  Magma_ZERO      = 595,
  Magma_NODIAG    = 596
} data_diagorder_t;

typedef enum {
  Magma_DCOMPLEX = 501,
  Magma_FCOMPLEX = 502,
  Magma_DOUBLE   = 503,
  Magma_FLOAT    = 504
} data_precision;

typedef enum {
  Magma_NOSCALE     = 511,
  Magma_UNITROW     = 512,
  Magma_UNITDIAG    = 513,
  Magma_UNITCOL     = 514,
  Magma_UNITROWCOL  = 515, // to be deprecated
  Magma_UNITDIAGCOL = 516, // to be deprecated
} data_scale_t;

// --------------------
// Convert development constants to CBLAS constants.
#include <mkl.h>
CBLAS_ORDER
cblas_order_const(data_order_t order);
CBLAS_TRANSPOSE
cblas_trans_const(data_trans_t trans);
CBLAS_UPLO
cblas_uplo_const(data_uplo_t uplo);
CBLAS_DIAG
cblas_diag_const(data_diag_t diag);
CBLAS_SIDE
cblas_side_const(data_side_t side);

// --------------------
//
#define CHECK(err)                          \
  do {                                      \
    data_int_t e_ = (err);                \
    if (e_ != 0) {                      \
      fprintf(stderr,                     \
        "CHECK ERROR : (%s, line %d)\n",  \
        __FILE__, __LINE__);              \
      exit(-1);                           \
    }                                     \
  } while (0)


#define LACE_CALLOC(ptr, nmemb) lace_calloc((void **) &(ptr), nmemb, sizeof(*(ptr)), #ptr, __FILE__, __LINE__)

static inline void
lace_calloc(void ** ptr, const size_t nmemb, const size_t size,
  const char * name, const char * file, const int line)
{
  size_t ns = (nmemb * size) / DEV_ALIGN;

  if (posix_memalign(ptr, DEV_ALIGN, (ns + 1) * DEV_ALIGN) != 0) {
    fprintf(stderr,
      "ERROR (memory): Unable to allocate memory [%lu bytes for %lu elements] for %s (%s, line %d) ns = %lu DEV_ALIGN = %d\n",
      (ns + 1) * DEV_ALIGN, (long int) (nmemb),
      name, file, line, (long int) (ns), DEV_ALIGN);
    #ifdef USING_MPI
      MPI_Abort(MPI_COMM_WORLD, -1);
    #else
      exit(-1);
    #endif
  }
  memset( (*ptr), '\0', nmemb * size);
}

/**
 * Macro checks the return code of a function;
 * if non-zero, sets info to err, then does goto cleanup.
 * err is evaluated only once.
 * Assumes variable info and label cleanup exist.
 * Usually, all paths (successful and error) exit through the cleanup code.
 * Example:
 *
 *    magma_int_t function()
 *    {
 *      data_int_t info = 0;
 *      double *A=NULL, *B=NULL;
 *      CHECK( data_malloc( &A, sizeA ));
 *      CHECK( data_malloc( &B, sizeB ));
 *      ...
 *    cleanup:
 *      data_free( A );
 *      data_free( B );
 *      return info;
 *    }
 * @ingroup data_error_internal
 ********************************************************************/

/*
 #define CHECK( err )                          \
 *  do {                                      \
 *      data_int_t e_ = (err);                \
 *      if ( e_ != 0 ) {                      \
 *        info = e_;                          \
 *        fprintf(stderr,                     \
 *          "CHECK ERROR : (%s, line %d)\n",  \
 *          file, line);                      \
 *        goto cleanup;                       \
 *      }                                     \
 *  } while(0)
 */
#endif //  #ifndef DEV_TYPES_H
