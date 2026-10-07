subroutine setprob
  use cparam_mod       
  implicit none
  integer :: iunit
  character(len=25) :: fname

  fname = 'setprob.data'
  iunit = 7

  call opendatafile(iunit, fname)

  ! Read and assign common variables (now handled with a module)
  read(iunit,*) tau
  read(iunit,*) kappa
  read(iunit,*) pnonlinear
  read(iunit,*) srcintegrator
  read(iunit,*) inclusion_idx
  read(iunit,*) inclusion_type

end subroutine setprob