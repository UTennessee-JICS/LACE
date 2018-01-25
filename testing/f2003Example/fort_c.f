PROGRAM FORT_C

use, intrinsic :: iso_c_binding
IMPLICIT NONE

interface
    subroutine call_fc(pX,s) bind(C,name='call_fc_')
        use, intrinsic :: iso_c_binding
        import
        integer(c_int)              :: s
        type(c_ptr)     :: pX
    end subroutine call_fc

    subroutine c_func_deallocate(pX) bind(C,name='c_func_deallocate_')
      import
      type(c_ptr)    :: pX
    end subroutine c_func_deallocate
end interface

integer(c_int)                              :: i
integer(c_int), pointer                     :: X(:)
type(C_ptr)                    :: pX


call call_fc(pX,100)
call c_f_pointer(pX,X,[100])
do i=1,10
  write (*,*) X(i)
end do
call c_func_deallocate(pX)

END
