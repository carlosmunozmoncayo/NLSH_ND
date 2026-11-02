
module map_module
  implicit none
  contains
    subroutine mapc2p(xc, yc, xp, yp)
        !Using the computational domain as the physical one , except for a circular
        !inclusion

        implicit none
        real(8), intent(in)  :: xc, yc
        real(8), intent(out) :: xp, yp

        ! Parameters defining quadrilateral
        real(8),  parameter :: x0 = 20.d0, y0 = 0.d0
        real(8),  parameter :: r1 = 1.5d0, r2 = 2.04d0
        logical :: insquare
        real(8) :: xc0, yc0, xc1, yc1, d, d1, R
        real(8) :: center

        insquare = .false.
        if (abs(xc-x0).le.r2 .and. abs(yc-y0).le.r2) then
            insquare = .true.
        end if

        xc0 = (xc - x0)/r2
        yc0 = (yc - y0)/r2
        xc1 = abs(xc0)
        yc1 = abs(yc0)
        d = max(xc1, yc1)
        d = max(d, 1.d-10)
        d = min(d, 0.99999d0)
        d1 = d*r2/sqrt(2.d0)

        ! Pick mapping
        !R = sqrt(2.d0) * d1
        R = r1
        R = r1**2 / (r2*d)
        ! Modify d1 and R outside circle to morph back to square:
        if (d.gt.(r1/r2)) then
            d1 = r1/sqrt(2.d0) + (d - r1/r2)*(r2 - r1/sqrt(2.d0))/(1.d0 - r1/r2)
            R = r1 * ((1.d0 - r1/r2) / (1.d0 - d))**(r2/r1 + 0.5d0)
        end if

        xp = d1/d * xc1
        yp = d1/d * yc1
        center = d1 - sqrt(R**2 - d1**2)
        if (xc1.ge.yc1) then
            xp = center + sqrt(R**2 - yp**2)
        end if
        if (yc1.ge.xc1) then
            yp = center + sqrt(R**2 - xp**2)
        end if
        xp = sign(1.d0, xc0) * xp
        yp = sign(1.d0, yc0) * yp
        xp = x0 + xp
        yp = y0 + yp

        if (.not. insquare) then
            xp = xc
            yp = yc
        end if

    end subroutine mapc2p

    ! subroutine mapc2p(xc,yc,xp,yp)
    !     ! Maps computational coordinates (xc,yc) to physical coordinates (xp,yp)
    !     implicit none
    !     real(8), intent(in)  :: xc, yc      ! computational coordinates
    !     real(8), intent(out) :: xp, yp      ! physical coordinates

    !     ! Rectangle corners for each section
    !     real(8) :: x_rectangle(2), y_rectangle(2)

    !     ! Trapezoid corners for each section
    !     real(8) :: x_trap(4), y_trap(4)

    !     ! Define rectangles and trapezoids for each section
    !     ! Left side of the domain
    !     x_rectangle = [0.d0, 1.d0]
    !     y_rectangle = [0.d0, 1.d0]
    !     ! Trapezoid corners (example values, can be modified)
    !     x_trap = [-10.d0, 35.d0, 35.d0, -10.d0]
    !     y_trap = [-3.d0,  -3.d0, 3.d0,  3.d0]

    !     call map_rect_to_trap(xc, yc, xp, yp, x_rectangle, y_rectangle, &
    !                           x_trap, y_trap)
    ! end subroutine mapc2p

    subroutine map_rect_to_trap(xc, yc, xp, yp, &
                                x_rectangle, y_rectangle, &
                                x_trap, y_trap)
        !------------------------------------------------------------
        ! Maps a rectangular computational domain [xc_min,xc_max]x[yc_min,yc_max]
        ! into a physical trapezoid with vertices:
        ! (x1,y1): lower left, (x2,y2): lower right,
        ! (x3,y3): upper right, (x4,y4): upper left
        !
        ! Example use:
        !   call map_rect_to_trap(0.5d0, 0.5d0, xp, yp)
        !------------------------------------------------------------
        implicit none
        real(8), intent(in)  :: xc, yc      ! computational coordinates
        real(8), intent(in)  :: x_rectangle(2), y_rectangle(2) ! rectangle corners
        real(8), intent(in)  :: x_trap(4), y_trap(4)         ! trapezoid corners
        real(8), intent(out) :: xp, yp      ! physical coordinates

        ! Parameters defining computational domain
        real(8) :: xc_min, xc_max
        real(8) :: yc_min, yc_max

        ! Trapezoid corner coordinates
        real(8) :: x1, y1 
        real(8) :: x2, y2 
        real(8) :: x3, y3
        real(8) :: x4, y4

        ! Local variables for normalized coords
        real(8) :: xi, eta

        ! Define computational domain corners
        xc_min = x_rectangle(1); xc_max = x_rectangle(2)
        yc_min = y_rectangle(1); yc_max = y_rectangle(2)
        ! Define trapezoid corners
        x1 = x_trap(1); y1 = y_trap(1)
        x2 = x_trap(2); y2 = y_trap(2)
        x3 = x_trap(3); y3 = y_trap(3)
        x4 = x_trap(4); y4 = y_trap(4)

        ! Normalize computational coordinates to [0,1]x[0,1]
        xi  = (xc - xc_min) / (xc_max - xc_min)
        eta = (yc - yc_min) / (yc_max - yc_min)

        ! Bilinear mapping (same as before, but now general)
        xp = (1.d0 - xi)*(1.d0 - eta)*x1 + xi*(1.d0 - eta)*x2 + xi*eta*x3 + (1.d0 - xi)*eta*x4
        yp = (1.d0 - xi)*(1.d0 - eta)*y1 + xi*(1.d0 - eta)*y2 + xi*eta*y3 + (1.d0 - xi)*eta*y4
    end subroutine map_rect_to_trap
    
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
