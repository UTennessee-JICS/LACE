/*
 *  -- LACE (version 0.0) --
 *     Univ. of Tennessee, Knoxville
 *
 *     @author Stephen Wood
 *
 */
#include "../include/sparse.h"
#include <mkl.h>
#include <float.h>
#include <stdlib.h>
#include <stdio.h>

/*******************************************************************************
*   Purpose
*   -------
*
*   Generate Givens rotation matrix.
*
*
*   Arguments
*   ---------
*
*   @param[in]
*   n           int
*               number of elements in X
*
*   @param[in,out]
*   X           dataType *
*               vector to be reflected
*
*   @param[in,out]
*   u           dataType *
*               reflected vector
*
*   @param[in]
*   nu          dataType *
*               norm of X
*
*   @ingroup datasparse_orthogonality
*******************************************************************************/


extern "C"
void
givens_rotation(dataType a,
  dataType               b,
  dataType               tol,
  dataType *             c,
  dataType *             s)
{
  if (fabs(b) <= tol) {
    GMRESDBG("fabs(b)=%e < tol=%e\n", fabs(b), tol);
    (*c) = 1.0;
    (*s) = 0.0;
  } else if (fabs(b) > fabs(a) ) {
    dataType temp = a / b;
    (*s) = 1.0 / sqrt(1.0 + temp * temp);
    (*c) = temp * (*s);
  } else {
    dataType temp = b / a;
    (*c) = 1.0 / sqrt(1.0 + temp * temp);
    (*s) = temp * (*c);
  }
}
