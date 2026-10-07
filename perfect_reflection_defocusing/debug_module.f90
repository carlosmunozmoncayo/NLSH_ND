module debug_module
    contains
    subroutine debug_log(xc,yc,xp,yp,q,aux)
    implicit none
    real(kind=8), intent(in) :: xc,yc,xp,yp
    real(kind=8), intent(in) :: q(:)
    real(kind=8), intent(in) :: aux(:)

    character(len=140) :: filepath
    integer :: unit

    filepath ="/Users/munozmca/Desktop/Research/2024_hyperbolization/NLS_ND/"//&
                "diagonal_wall/debug_log.txt"

    ! Open file in append mode
    open(newunit=unit, file=trim(filepath), status='unknown', position='append', action='write')

    ! Write x, y, q, aux to file in a single line
    write(unit,*) xc,yc,xp,yp, q, aux

    close(unit)
end subroutine debug_log
end module