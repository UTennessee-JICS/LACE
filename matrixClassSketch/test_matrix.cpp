#include <stdlib.h>
#include <stdio.h>
#include <gtest/gtest.h>
#include "matrix.hpp"

TEST(StorageLayout, defaultConstructor)
{
  const StorageLayout<double> sampleLayout;

  EXPECT_EQ(0, static_cast<int>(sampleLayout.bluePrint));
  EXPECT_EQ(0, sampleLayout.numRows);
  EXPECT_EQ(0, sampleLayout.numCols);
}

TEST(Matrix, defaultConstructor)
{
  const Matrix<double> sampleMatrix;

  EXPECT_EQ(0, sampleMatrix.entry.size());
}

int main(int argc, char* argv[])
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
