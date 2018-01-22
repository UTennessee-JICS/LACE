      program test

      integer ii, jj, kk
      integer storageType, nRow, nCol, nnz, cooRow, cooCol
      common/ijk/ ii, jj, kk
      real*8  ff, cooVal(20)
      character*32 cc
      character*1024 fileName

      ii = 2
      jj = 3
      kk = 4
      ff = 9.0567
      cc = 'Example of a character string'
      fileName = "matrices/Trefethen_20.mtx"

      write(6,10) ii, ff
10    format('ii= ',i2,' ff= ',f10.4)

      call read_z_coo_from_mtx(storageType, nRow, nCol, nnz,
     &  cooVal, cooRow, cooCol, fileName )

      stop
      end
