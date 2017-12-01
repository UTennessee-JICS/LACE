#ifndef DEV_CONTAINERTESTS_H
#define DEV_CONTAINERTESTS_H

#include <cmath>
#include <limits>
#include <iomanip>
#include <iostream>
#include <type_traits>
#include <algorithm>


#define EXPECT_EQ(ref, target, equivalent) \
  { \
    equivalent = true; \
    std::ios::fmtflags f(std::cout.flags() ); \
    if (!std::isfinite(ref) || !std::isfinite(target) ) { \
      std::cout << std::scientific << std::setprecision(16) \
                << "\033[31m[  FAILED  ]\033[0m " #ref " = " << ref \
                << " " #target " = " << target << " " \
                << __FILE__ << " " << __LINE__ << std::endl; \
      equivalent = false; \
    } \
    else if ( (ref == target) != true) { \
      std::cout << std::scientific << std::setprecision(16) \
                << "\033[31m[  FAILED  ]\033[0m " #ref " = " << ref \
                << " and " #target " = " << target \
                << " differ " << __FILE__ << " " << __LINE__ << std::endl; \
      equivalent = false; \
    } \
    std::cout.flags(f); \
  }

#define EXPECT_NE(ref, target, equivalent) \
  { \
    equivalent = true; \
    std::ios::fmtflags f(std::cout.flags() ); \
    if (!std::isfinite(ref) || !std::isfinite(target) ) { \
      std::cout << std::scientific << std::setprecision(16) \
                << "\033[31m[  FAILED  ]\033[0m " #ref " = " << ref \
                << " " #target " = " << target << " " \
                << __FILE__ << " " << __LINE__ << std::endl; \
      equivalent = false; \
    } \
    else if ( (ref != target) != true) { \
      std::cout << std::scientific << std::setprecision(16) \
                << "\033[31m[  FAILED  ]\033[0m " #ref " = " << ref \
                << " and " #target " = " << target \
                << " are equivalent to machine precision " << __FILE__ << " " << __LINE__ << std::endl; \
      equivalent = false; \
    } \
    std::cout.flags(f); \
  }

#define EXPECT_LT(val1, val2, lessThan) \
  { \
    lessThan = true; \
    std::ios::fmtflags f(std::cout.flags() ); \
    if (!std::isfinite(val1) || !std::isfinite(val2) ) { \
      std::cout << std::scientific << std::setprecision(16) \
                << "\033[31m[  FAILED  ]\033[0m " #val1 " = " << val1 \
                << " " #val2 " = " << val2 << " " \
                << __FILE__ << " " << __LINE__ << std::endl; \
      lessThan = false; \
    } \
    else if ( (val1 < val2) != true) { \
      std::cout << std::scientific << std::setprecision(16) \
                << "\033[31m[  FAILED  ]\033[0m " #val1 " = " << val1 \
                << " is greater than or equal to, within machine precision, " #val2 " = " << val2 \
                << " " << __FILE__ << " " << __LINE__ << std::endl; \
      lessThan = false; \
    } \
    std::cout.flags(f); \
  }

#define EXPECT_LE(val1, val2, lessThan) \
  { \
    lessThan = true; \
    std::ios::fmtflags f(std::cout.flags() ); \
    if (!std::isfinite(val1) || !std::isfinite(val2) ) { \
      std::cout << std::scientific << std::setprecision(16) \
                << "\033[31m[  FAILED  ]\033[0m " #val1 " = " << val1 \
                << " " #val2 " = " << val2 << " " \
                << __FILE__ << " " << __LINE__ << std::endl; \
      lessThan = false; \
    } \
    else if ( (val1 <= val2) != true) { \
      std::cout << std::scientific << std::setprecision(16) \
                << "\033[31m[  FAILED  ]\033[0m " #val1 " = " << val1 \
                << " is greater than " #val2 " = " << val2 \
                << " " << __FILE__ << " " << __LINE__ << std::endl; \
      lessThan = false; \
    } \
    std::cout.flags(f); \
  }

#define EXPECT_GT(val1, val2, greaterThan) \
  { \
    greaterThan = true; \
    std::ios::fmtflags f(std::cout.flags() ); \
    if (!std::isfinite(val1) || !std::isfinite(val2) ) { \
      std::cout << std::scientific << std::setprecision(16) \
                << "\033[31m[  FAILED  ]\033[0m " #val1 " = " << val1 \
                << " " #val2 " = " << val2 << " " \
                << __FILE__ << " " << __LINE__ << std::endl; \
      greaterThan = false; \
    } \
    else if ( (val1 > val2) != true) { \
      std::cout << std::scientific << std::setprecision(16) \
                << "\033[31m[  FAILED  ]\033[0m " #val1 " = " << val1 \
                << " is less than or equal to, within machine precision, " #val2 " = " << val2 \
                << " " << __FILE__ << " " << __LINE__ << std::endl; \
      greaterThan = false; \
    } \
    std::cout.flags(f); \
  }

#define EXPECT_GE(val1, val2, greaterThan) \
  { \
    greaterThan = true; \
    std::ios::fmtflags f(std::cout.flags() ); \
    if (!std::isfinite(val1) || !std::isfinite(val2) ) { \
      std::cout << std::scientific << std::setprecision(16) \
                << "\033[31m[  FAILED  ]\033[0m " #val1 " = " << val1 \
                << " " #val2 " = " << val2 << " " \
                << __FILE__ << " " << __LINE__ << std::endl; \
      greaterThan = false; \
    } \
    else if ( (val1 >= val2) != true) { \
      std::cout << std::scientific << std::setprecision(16) \
                << "\033[31m[  FAILED  ]\033[0m " #val1 " = " << val1 \
                << " is less than " #val2 " = " << val2 \
                << " " << __FILE__ << " " << __LINE__ << std::endl; \
      greaterThan = false; \
    } \
    std::cout.flags(f); \
  }

template <class T>
typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type
floating_equal(T x, T y, int ulp)
{
  // the machine epsilon has to be scaled to the magnitude of the values used
  // and multiplied by the desired precision in ULPs (units in the last place)
  return std::abs(x - y) < std::numeric_limits<T>::epsilon() * std::abs(x + y) * ulp
 // unless the result is subnormal
         || std::abs(x - y) < std::numeric_limits<T>::min();
}

#define EXPECT_DOUBLE_EQ(ref, target, equivalent) \
  { \
    equivalent = true; \
    std::ios::fmtflags f(std::cout.flags() ); \
    if (!std::isfinite(ref) || !std::isfinite(target) ) { \
      std::cout << std::scientific << std::setprecision(16) \
                << "\033[31m[  FAILED  ]\033[0m " #ref " = " << ref \
                << " " #target " = " << target << " " \
                << __FILE__ << " " << __LINE__ << std::endl; \
      equivalent = false; \
    } \
    else if (floating_equal(ref, target, 4) != true) { \
      std::cout << std::scientific << std::setprecision(16) \
                << "\033[31m[  FAILED  ]\033[0m " #ref " = " << ref \
                << " and " #target " = " << target \
                << " differ " << __FILE__ << " " << __LINE__ << std::endl; \
      equivalent = false; \
    } \
    std::cout.flags(f); \
  }

#define EXPECT_DOUBLE_NE(ref, target, equivalent) \
  { \
    equivalent = true; \
    std::ios::fmtflags f(std::cout.flags() ); \
    if (!std::isfinite(ref) || !std::isfinite(target) ) { \
      std::cout << std::scientific << std::setprecision(16) \
                << "\033[31m[  FAILED  ]\033[0m " #ref " = " << ref \
                << " " #target " = " << target << " " \
                << __FILE__ << " " << __LINE__ << std::endl; \
      equivalent = false; \
    } \
    else if (floating_equal(ref, target, 4) == true) { \
      std::cout << std::scientific << std::setprecision(16) \
                << "\033[31m[  FAILED  ]\033[0m " #ref " = " << ref \
                << " and " #target " = " << target \
                << " are equivalent to machine precision " << __FILE__ << " " << __LINE__ << std::endl; \
      equivalent = false; \
    } \
    std::cout.flags(f); \
  }

#define EXPECT_ARRAY_INT_EQ(length, ref, target) \
  { \
    bool equivalent = true; \
    int i = 0; \
    for (i = 0; i < length; i++) { \
      EXPECT_EQ(ref[i], target[i], equivalent) \
      if (equivalent == false) \
        std::cout << "\tInteger arrays " #ref  " and " #target \
          " differ at index " << i << std::endl; \
      equivalent = true; \
    } \
  }

#define EXPECT_ARRAY_DOUBLE_EQ(length, ref, target) \
  { \
    bool equivalent = true; \
    int i = 0; \
    for (i = 0; i < length; i++) { \
      EXPECT_DOUBLE_EQ(ref[i], target[i], equivalent) \
      if (equivalent == false) \
        std::cout << "\tDouble arrays " #ref  " and " #target \
          " differ at index " << i << std::endl; \
      equivalent = true; \
    } \
  }


#endif // ifndef DEV_CONTAINERTESTS_H
