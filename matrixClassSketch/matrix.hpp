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
                    numRows(0), numCols(0), nnz(0), val(nullptr), subentry(0) {};
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
  Matrix0() : numLayouts(1), matrixNumRows(0), matrixNumCols(0), matrixNNZ(0),
    rowStride(1), columnStride(1), entry(nullptr) {};
  ~Matrix0() {
    delete [] entry;
  };

  dim_t numLayouts;
  dim_t matrixNumRows;
  dim_t matrixNumCols;
  dim_t matrixNNZ;
  dim_t rowStride;
  dim_t columnStride;

  T* entry;

  void setup(dim_t rows, dim_t cols, dim_t rStride, dim_t cStride ) {
    matrixNumRows = rows;
    matrixNumCols = cols;
    matrixNNZ = rows*cols;
    rowStride = rStride;
    columnStride = cStride;
    entry = new T[matrixNNZ]();
  }

  T val( int i, int j ) {
    return entry[i*rowStride + j*columnStride];
  }

  void print() {
    for ( int i=0; i<matrixNumRows; ++i ) {
      for ( int j=0; j<matrixNumCols; ++j ) {
        std::cout << val(i,j) << " ";
      }
      std::cout << '\n';
    }

  }

};


template <class T>
class CSMatrix {
public:
  CSMatrix() : numLayouts(1), matrixNumRows(0), matrixNumCols(0),
    row(nullptr), col(nullptr), rowStride(1), columnStride(1), entry(nullptr),
    matrixNNZ(0) {};
  ~CSMatrix() {
    delete [] row;
    delete [] col;
    delete [] entry;
  };

  dim_t numLayouts;
  dim_t matrixNumRows;
  dim_t matrixNumCols;
  dim_t matrixNNZ;
  dim_t rowStride;
  dim_t columnStride;

  dim_t* row;
  dim_t* col;

  T* entry;

  void setup(dim_t rows, dim_t cols, dim_t nnz) {
    matrixNumRows = rows;
    matrixNumCols = cols;
    matrixNNZ = nnz;
    rowStride = cols;
    columnStride = dim_t(1); // row major example
    row = new dim_t[matrixNumRows+1]();
    col = new dim_t[matrixNNZ]();
    entry = new T[matrixNNZ]();
  }

};
