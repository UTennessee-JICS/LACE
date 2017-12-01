#include <cmath>
#include <limits>
#include <iomanip>
#include <iostream>
#include <type_traits>
#include <algorithm>

#include "gpu_tests.h"

int
main()
{
  int i1           = 1;
  int i2           = 2;
  int i3           = 1;
  int iarray1[]    = { 1, 1, 2, 3, 5 };
  int iarray2[]    = { 1, 3, 6, 10, 15 };
  int iarray3[]    = { 1, 1, 2, 3, 5 };
  double d1        = 0.2;
  double d2        = 1 / std::sqrt(5) / std::sqrt(5);
  double d3        = 0.2;
  double darray1[] = { 0.1, 0.4, 0.9, 1.6, 2.5 };
  double darray2[] = { 0.1, 0.3, 0.6, 1.0, 2.5 };
  double darray3[] = { 0.1, 0.4, 0.9, 1.6, 2.5 };
  bool comparison  = true;

  // double inf = std::numeric_limits<double>::infinity();

  EXPECT_EQ(i1, i2, comparison);
  EXPECT_DOUBLE_EQ(d1, d2, comparison);
  std::cout << "d1 = " << d1 << " d2 = " << d2 << std::endl;
  printf("d1 = %.16e d2 = %.16e d3 = %.16e\n", d1, d2, d3);

  if (d1 == d2)
    std::cout << "d1 == d2\n";
  else
    std::cout << "d1 != d2\n";

  if (floating_equal(d1, d2, 2))
    std::cout << "d1 almost equals d2\n";
  else
    std::cout << "d1 does not almost equal d2\n";

  EXPECT_EQ(i1, i2, comparison);
  EXPECT_EQ(i1, i3, comparison);
  EXPECT_DOUBLE_EQ(d1, d2, comparison);
  EXPECT_DOUBLE_EQ(d1, d3, comparison);

  EXPECT_ARRAY_INT_EQ(5, iarray1, iarray2);
  EXPECT_ARRAY_INT_EQ(5, iarray1, iarray3);
  EXPECT_ARRAY_DOUBLE_EQ(5, darray1, darray2);
  EXPECT_ARRAY_DOUBLE_EQ(5, darray1, darray3);

  std::cout << "std::numeric_limits<double>::epsilon() = "
            << std::numeric_limits<double>::epsilon() << std::endl;

  std::cout << "abs(+3.03) = " << std::abs(+3.03) << '\n'
            << "abs(-3.03) = " << std::abs(-3.03) << '\n';
  // special values
  std::cout << "abs(-0.0) = " << std::abs(-0.0) << '\n'
            << "abs(-Inf) = " << std::abs(-INFINITY) << '\n';

  EXPECT_NE(i1, i2, comparison);
  EXPECT_NE(i1, i3, comparison);
  EXPECT_DOUBLE_NE(d1, d2, comparison);
  EXPECT_DOUBLE_NE(d1, d3, comparison);

  EXPECT_LE(i1, i2, comparison);
  EXPECT_LE(i1, i3, comparison);
  EXPECT_LE(i2, i3, comparison);
  EXPECT_LE(d1, d2, comparison);
  EXPECT_LE(d1, d3, comparison);
  EXPECT_LE(d2, d3, comparison);
  EXPECT_GE(i1, i2, comparison);
  EXPECT_GE(i1, i3, comparison);
  EXPECT_GE(i2, i3, comparison);
  EXPECT_GE(d1, d2, comparison);
  EXPECT_GE(d1, d3, comparison);
  EXPECT_GE(d2, d3, comparison);

  EXPECT_LT(i1, i2, comparison);
  EXPECT_LT(i1, i3, comparison);
  EXPECT_LT(i2, i3, comparison);
  EXPECT_LT(d1, d2, comparison);
  EXPECT_LT(d1, d3, comparison);
  EXPECT_LT(d2, d3, comparison);
  EXPECT_GT(i1, i2, comparison);
  EXPECT_GT(i1, i3, comparison);
  EXPECT_GT(i2, i3, comparison);
  EXPECT_GT(d1, d2, comparison);
  EXPECT_GT(d1, d3, comparison);
  EXPECT_GT(d2, d3, comparison);

  EXPECT_EQ(NAN, i1, comparison);
  EXPECT_EQ(i2, -INFINITY, comparison);
  EXPECT_EQ(-NAN, d3, comparison);
  EXPECT_EQ(d2, INFINITY, comparison);

  EXPECT_LE(NAN, i1, comparison);
  EXPECT_LE(i2, -INFINITY, comparison);
  EXPECT_LE(-NAN, d3, comparison);
  EXPECT_LE(d2, INFINITY, comparison);
  EXPECT_GE(NAN, i1, comparison);
  EXPECT_GE(i2, -INFINITY, comparison);
  EXPECT_GE(-NAN, d3, comparison);
  EXPECT_GE(d2, INFINITY, comparison);

  EXPECT_LT(NAN, i1, comparison);
  EXPECT_LT(i2, -INFINITY, comparison);
  EXPECT_LT(-NAN, d3, comparison);
  EXPECT_LT(d2, INFINITY, comparison);
  EXPECT_GT(NAN, i1, comparison);
  EXPECT_GT(i2, -INFINITY, comparison);
  EXPECT_GT(-NAN, d3, comparison);
  EXPECT_GT(d2, INFINITY, comparison);

  return 0;
} // main
