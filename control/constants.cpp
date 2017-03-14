#include <assert.h>
#include <stdio.h>

#include "../include/dense_types.h"


// ----------------------------------------
// Convert data constants to CBLAS constants.
// We assume that data constants are consistent with cblas constants,
// so verify that with asserts.

//#ifdef HAVE_CBLAS
//extern "C"
 CBLAS_ORDER     cblas_order_const  ( data_order_t data_const )
{
    assert( data_const >= MagmaRowMajor );
    assert( data_const <= MagmaColMajor );
    assert( (int)MagmaRowMajor == CblasRowMajor );
    return ( CBLAS_ORDER)     data_const;
}

//extern "C"
 CBLAS_TRANSPOSE cblas_trans_const  ( data_trans_t data_const )
{
    assert( data_const >= MagmaNoTrans   );
    assert( data_const <= MagmaConjTrans );
    assert( (int)MagmaNoTrans == CblasNoTrans );
    return ( CBLAS_TRANSPOSE) data_const;
}

//extern "C"
 CBLAS_UPLO      cblas_uplo_const   ( data_uplo_t data_const )
{
    assert( data_const >= MagmaUpper );
    assert( data_const <= MagmaLower );
    assert( (int)MagmaUpper == CblasUpper );
    return ( CBLAS_UPLO)      data_const;
}

//extern "C"
 CBLAS_DIAG      cblas_diag_const   ( data_diag_t data_const )
{
    assert( data_const >= MagmaNonUnit );
    assert( data_const <= MagmaUnit    );
    assert( (int)MagmaUnit == CblasUnit );
    return ( CBLAS_DIAG)      data_const;
}

//extern "C"
 CBLAS_SIDE      cblas_side_const   ( data_side_t data_const )
{
    assert( data_const >= MagmaLeft  );
    assert( data_const <= MagmaRight );
    assert( (int)MagmaLeft == CblasLeft );
    return ( CBLAS_SIDE)      data_const;
}
//#endif  // HAVE_CBLAS
