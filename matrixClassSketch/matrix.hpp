#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef int   dim_t;

enum class storageBluePrint_t : int {
  dense,
  compressed,
  denseTriangular,
  compressedTraingular
};

template <class T>
class StorageLayout {
public:
  StorageLayout() : bluePrint(storageBluePrint_t::dense), numRows(0), numCols(0), nnz(0) {};
  ~StorageLayout() {};

  storageBluePrint_t bluePrint;
  dim_t numRows;
  dim_t numCols;
  int nnz;
  T*    val;
};

template <class T>
class Matrix {
public:
  Matrix() : entry(0) {};
  ~Matrix() {};

  std::vector< StorageLayout<T> > entry;

};
