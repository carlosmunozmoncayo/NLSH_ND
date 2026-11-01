module map_module
  implicit none
  contains
    subroutine mapc2p(xc,yc,xp,yp)
        ! Maps computational coordinates (xc,yc) in [-1,1]x[-1,1]
        ! to physical coordinates (xp,yp) in a circular domain of radius rl
        implicit none
        double precision, intent(in) :: xc, yc
        double precision, intent(out) :: xp, yp
        double precision :: rl, d, D_cap, R, center
        rl = 10.d0  ! map [-1,1] x [-1,1] to circle of radius rl
        d = max(abs(xc), abs(yc))  ! value on diagonal of computation
        d = max(d, 1.d-10)  ! to avoid divide by zero at origin
        D_cap = rl * d / dsqrt(2.d0)  ! mapping d to D(d) from (3.2)
        R = rl * d  ! mapping d to R(d) from (3.2)
        ! R = rl * ones_like(d)  ! alternative for (3.3)
        center = D_cap - dsqrt(R**2 - D_cap**2)
        xp = D_cap / d * abs(xc)
        yp = D_cap / d * abs(yc)
        if (abs(yc) >= abs(xc)) then
            yp = center + dsqrt(R**2 - xp**2)
        else
            xp = center + dsqrt(R**2 - yp**2)
        end if
        xp = sign(1.d0, xc) * xp
        yp = sign(1.d0, yc) * yp
    end subroutine mapc2p

    subroutine vectorized_mapc2p(nx,ny,xc_array, yc_array, xp_array, yp_array)
    ! Vectorized version of mapc2p for arrays of points
    implicit none
    integer, intent(in)  :: nx, ny
    real(8), intent(in)  :: xc_array(nx,ny), yc_array(nx,ny)
    real(8), intent(out) :: xp_array(nx,ny), yp_array(nx,ny)
    integer :: i,j
    do i = 1, nx
        do j = 1, ny
            call mapc2p(xc_array(i,j), yc_array(i,j), xp_array(i,j), yp_array(i,j))
            end do
    end do
    end subroutine vectorized_mapc2p
end module map_module