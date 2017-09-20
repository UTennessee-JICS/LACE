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

  EXPECT_EQ(1, sampleMatrix.numLayouts);
  EXPECT_EQ(0, sampleMatrix.matrixNumRows);
  EXPECT_EQ(0, sampleMatrix.matrixNumCols);
  EXPECT_EQ(1, sampleMatrix.entry.size() );
}

TEST(Matrix, calcMatrixDimensions)
{
  Matrix<double> sampleMatrix;

  EXPECT_EQ(0, sampleMatrix.calcNumRows() );
  EXPECT_EQ(0, sampleMatrix.calcNumCols() );

  sampleMatrix.entry[0].numRows = 10;
  sampleMatrix.entry[0].numCols = 5;
  sampleMatrix.calcDimensions();

  EXPECT_EQ(10, sampleMatrix.numRows() );
  EXPECT_EQ(5, sampleMatrix.numCols() );

}

TEST(Matrix0, defaultConstructor)
{
  Matrix0<double> sampleMatrix;

  EXPECT_EQ(1, sampleMatrix.numLayouts);
  EXPECT_EQ(0, sampleMatrix.matrixNumRows);
  EXPECT_EQ(0, sampleMatrix.matrixNumCols);
  EXPECT_EQ(NULL, sampleMatrix.entry );
}


TEST(CSMatrix, defaultConstructor)
{
  CSMatrix<double> sampleMatrix;

  EXPECT_EQ(1, sampleMatrix.numLayouts);
  EXPECT_EQ(0, sampleMatrix.matrixNumRows);
  EXPECT_EQ(0, sampleMatrix.matrixNumCols);
  EXPECT_EQ(NULL, sampleMatrix.row );
  EXPECT_EQ(NULL, sampleMatrix.col );
  EXPECT_EQ(NULL, sampleMatrix.entry );
}

TEST(CSMatrix, defaultConstructorMatrix0Double)
{
  CSMatrix< Matrix0<double> > sampleMatrix;

  EXPECT_EQ(1, sampleMatrix.numLayouts);
  EXPECT_EQ(0, sampleMatrix.matrixNumRows);
  EXPECT_EQ(0, sampleMatrix.matrixNumCols);
  EXPECT_EQ(NULL, sampleMatrix.row );
  EXPECT_EQ(NULL, sampleMatrix.col );
  EXPECT_EQ(NULL, sampleMatrix.entry );

  sampleMatrix.setup(3,4,3);
  sampleMatrix.entry[0].setup(2,3,3,1);
  sampleMatrix.entry[1].setup(4,3,3,1);
  sampleMatrix.entry[2].setup(5,5,5,1);

  EXPECT_EQ(2, sampleMatrix.entry[0].matrixNumRows);
  EXPECT_EQ(3, sampleMatrix.entry[0].matrixNumCols);
  EXPECT_EQ(4, sampleMatrix.entry[1].matrixNumRows);
  EXPECT_EQ(3, sampleMatrix.entry[1].matrixNumCols);
  EXPECT_EQ(5, sampleMatrix.entry[2].matrixNumRows);
  EXPECT_EQ(5, sampleMatrix.entry[2].matrixNumCols);

  for ( int b=0; b<sampleMatrix.matrixNNZ; ++b ) {
    std::cout << "block " << b << ":\n";
    sampleMatrix.entry[b].print();
  }

}

int main(int argc, char* argv[])
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
