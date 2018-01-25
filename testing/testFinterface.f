      program test

      integer ii, jj, kk
      integer storageType, nRow, nCol, nnz, cooRow, cooCol
      integer LnRow, LnCol, Lnnz, LcooRow, LcooCol
      integer UnRow, UnCol, Unnz, UcooRow, UcooCol
      real*8  cooVal(147), LcooVal(147), UcooVal(147)
      character*1024 fileName

      fileName = "matrices/Trefethen_20.mtx"

      call read_z_coo_from_mtx(storageType, nRow, nCol, nnz,
     &  cooVal, cooRow, cooCol, fileName )

      write(*,*) 'nnz=', nnz
      write(*,*) 'nRow=', nRow

      do 10 ii = 1, 20
        write(*,*) cooVal(ii)
  10  continue
c      call data_PariLU_v0_3(

      stop
      end
