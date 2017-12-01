#ifndef DEV_CONTAINERTESTS_H
#define DEV_CONTAINERTESTS_H

#include <cmath>
#include <limits>
#include <iomanip>
#include <iostream>
#include <type_traits>
#include <algorithm>


#define EXPECT_EQ( ref, target, equivalent ) \
{ \
  equivalent = true; \
  if ( ref != target ) { \
    std::cout << "\033[31m[  FAILED  ]\033[0m " #ref " = " << ref \
      << " and " #target " = " << target \
      << " differ " << __FILE__ << " " << __LINE__ << std::endl; \
    equivalent = false; \
  } \
}

template<class T>
typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type
    floating_equal(T x, T y, int ulp)
{
      // the machine epsilon has to be scaled to the magnitude of the values used
      // and multiplied by the desired precision in ULPs (units in the last place)
      return std::fabs(x-y) < std::numeric_limits<T>::epsilon() * std::abs(x+y) * ulp
            // unless the result is subnormal
                   || std::fabs(x-y) < std::numeric_limits<T>::min();
}

#define EXPECT_DOUBLE_EQ( ref, target, equivalent ) \
{ \
  equivalent = true; \
  if ( floating_equal(ref, target, 2) != true ) { \
    std::ios::fmtflags f( std::cout.flags() ); \
    std::cout << std::scientific << std::setprecision(9) \
      << "\033[31m[  FAILED  ]\033[0m " #ref " = " << ref \
      << " and " #target " = " << target \
      << " differ " << __FILE__ << " " << __LINE__ << std::endl; \
    equivalent = false; \
    std::cout.flags( f ); \
  } \
}


#define EXPECT_ARRAY_INT_EQ( length, ref, target) \
{ \
  bool equivalent = true; \
  int i = 0; \
  for(i=0; i<length; i++) { \
    EXPECT_EQ(ref[i], target[i], equivalent ) \
    if (equivalent == false) \
      std::cout << "\tInteger arrays " #ref  " and " #target \
           " differ at index " << i << std::endl; \
    equivalent = true; \
  } \
}

#define EXPECT_ARRAY_DOUBLE_EQ( length, ref, target) \
{ \
  bool equivalent = true; \
  int i = 0; \
  for(i=0; i<length; i++) { \
    EXPECT_DOUBLE_EQ(ref[i], target[i], equivalent ) \
    if (equivalent == false) \
      std::cout << "\tDouble arrays " #ref  " and " #target \
           " differ at index " << i << std::endl; \
    equivalent = true; \
  } \
}


#endif
