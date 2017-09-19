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
  StorageLayout() : bluePrint(storageBluePrint_t::dense), numRows(0), numCols(0) {};
  //StorageLayout() {};
  ~StorageLayout() {};

  storageBluePrint_t bluePrint;
  dim_t numRows;
  dim_t numCols;
  T*    val;
};
