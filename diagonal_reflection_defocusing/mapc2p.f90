
module map_module
  implicit none
  contains
    subroutine mapc2p(xc, yc, xp, yp)
    ! Maps computational coords (xc,yc) in [0,1]x[0,1]
    ! into a quadrilateral physical domain, determined by
    ! (0,0)->(x1,y1), (1,0)->(x2,y2), (1,1)->(x3,y3), (0,1)->(x4,y4).

    implicit none
    real(8), intent(in)  :: xc, yc
    real(8), intent(out) :: xp, yp

    ! Define rectangle corners in computational space
    real(8), dimension(2) :: x_rectangle1, y_rectangle1
    real(8), dimension(2) :: x_rectangle2, y_rectangle2
    real(8), dimension(2) :: x_rectangle3, y_rectangle3

    ! Define trapezoid corners in physical space
    real(8), dimension(4) :: x_trap1, y_trap1
    real(8), dimension(4) :: x_trap2, y_trap2
    real(8), dimension(4) :: x_trap3, y_trap3


    !Left side of the domain, a trapezoid 
    ! Rectangle corners
    x_rectangle1 = [0.d0, 1.d0]! onethird]!
    y_rectangle1 = [0.d0, 1.d0]
    ! Trapezoid corners (example values, can be modified)
    x_trap1 = [-8.d0, 8.d0, 8.d0, -7.d0]
    y_trap1 = [-1.d0,  -1.d0, 1.d0,  1.d0]

        call map_rect_to_trap(xc, yc, xp, yp, x_rectangle1, y_rectangle1, &
                              x_trap1, y_trap1)
  end subroutine mapc2p

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
