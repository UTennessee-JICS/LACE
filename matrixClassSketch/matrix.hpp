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
  StorageLayout() : bluePrint(storageBluePrint_t::dense),
                    numRows(0), numCols(0), nnz(0), val(NULL), subentry(0) {};
  ~StorageLayout() {};

  storageBluePrint_t bluePrint;
  dim_t numRows;
  dim_t numCols;
  dim_t nnz;
  T*    val;

  std::vector< StorageLayout<T> > subentry;
};

template <class T>
class Matrix {
public:
  Matrix() : numLayouts(1), entry(1), matrixNumRows(0), matrixNumCols(0) {};
  ~Matrix() {};

  dim_t numLayouts;
  std::vector< StorageLayout<T> > entry;


  dim_t matrixNumRows;
  dim_t matrixNumCols;

  dim_t calcNumRows() {
    matrixNumRows = 0;
    for(typename std::vector< StorageLayout<T> >::iterator it = entry.begin(); it != entry.end(); ++it) {
      matrixNumRows += it->numRows;
    }
    return matrixNumRows;
  }

  dim_t calcNumCols() {
    matrixNumCols = 0;
    for(typename std::vector< StorageLayout<T> >::iterator it = entry.begin(); it != entry.end(); ++it) {
      matrixNumCols += it->numCols;
    }
    return matrixNumCols;
  }

  void calcDimensions() {
    matrixNumRows = 0;
    matrixNumCols = 0;
    for(typename std::vector< StorageLayout<T> >::iterator it = entry.begin(); it != entry.end(); ++it) {
      matrixNumRows += it->numRows;
      matrixNumCols += it->numCols;
    }
  }

  dim_t numRows() {
    return matrixNumRows;
  }

  dim_t numCols() {
    return matrixNumCols;
  }

};


template <class T>
class Matrix0 {
public:
  Matrix0() : numLayouts(1), matrixNumRows(0), matrixNumCols(0), matrixNNZ(0), entry(NULL) {};
  ~Matrix0() {
    delete [] entry;
  };

  dim_t numLayouts;
  dim_t matrixNumRows;
  dim_t matrixNumCols;
  dim_t matrixNNZ;

  T* entry;

  void setup(int rows, int cols) {
    matrixNumRows = rows;
    matrixNumCols = cols;
    matrixNNZ = rows*cols;
    entry = new T[matrixNNZ];
  }

};


template <class T>
class CSRMatrix {
public:
  CSRMatrix() : numLayouts(1), matrixNumRows(0), matrixNumCols(0), row(NULL), col(NULL), entry(NULL) {};
  ~CSRMatrix() {
    delete [] entry;
  };

  dim_t numLayouts;
  dim_t matrixNumRows;
  dim_t matrixNumCols;
  dim_t matrixNNZ;

  dim_t* row;
  dim_t* col;

  T* entry;

  void setup(int rows, int cols, int nnz) {
    matrixNumRows = rows;
    matrixNumCols = cols;
    matrixNNZ = nnz;
    entry = new T[matrixNNZ];
  }

};
