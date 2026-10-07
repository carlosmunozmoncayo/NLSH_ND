module map_module
  implicit none

  !-------------------------------------------------------------------
  ! Multi-block mapping of the computational unit square [0,1]x[0,1]
  ! onto N_BLOCKS disjoint trapezoids whose union is EXACTLY the
  ! physical rectangle (X_LEFT,X_RIGHT) x (Y_BOT,Y_TOP).
  !
  ! Block k (k = 0 .. N_BLOCKS-1) occupies the computational strip
  !     xc in [k/N_BLOCKS, (k+1)/N_BLOCKS],  yc in [0,1]
  ! and is mapped onto the quadrilateral
  !     (X_BOT(k)  , Y_BOT)   lower left
  !     (X_BOT(k+1), Y_BOT)   lower right
  !     (X_TOP(k+1), Y_TOP)   upper right
  !     (X_TOP(k)  , Y_TOP)   upper left
  !
  ! X_BOT and X_TOP are two independent partitions of the physical
  ! x-interval; they share their endpoints so the outer boundary is the
  ! plain rectangle, but differ in the interior so every interior edge
  ! is slanted.  Adjacent blocks share the whole slanted edge, so the
  ! tiling is conforming: no gaps, no overlaps.
  !
  ! CONDITIONS (the only ones you need to check if you edit the data):
  !   (a) X_BOT and X_TOP are both strictly increasing,
  !   (b) X_BOT(0) = X_TOP(0) = X_LEFT and
  !       X_BOT(N) = X_TOP(N) = X_RIGHT.
  ! Then the bilinear map has Jacobian
  !       J = 2*d(xp)/d(xi) = 2*[(1-eta)*wbot_k + eta*wtop_k] > 0
  ! everywhere, so no cell can fold, and for every horizontal level
  ! y the block breakpoints (1-eta)*X_BOT + eta*X_TOP form a partition
  ! of (X_LEFT,X_RIGHT).  Nothing else can go wrong.
  !
  ! Useful consequences of the bilinear map here:
  !   * yp = Y_BOT + (Y_TOP-Y_BOT)*eta  is independent of xi, so
  !     horizontal grid lines stay horizontal and the top and bottom
  !     walls stay flat.
  !   * X_BOT(0)=X_TOP(0) and X_BOT(N)=X_TOP(N) make the left and right
  !     walls vertical, so all four physical boundaries remain
  !     axis-aligned and the wall BCs are unchanged.  Only INTERIOR
  !     edges are slanted -- which is exactly what you want if the point
  !     is to exercise the rotation to (n,t) coordinates while keeping
  !     the exact solution and the boundary treatment identical to a
  !     uniform-grid run.
  !   * The map is C^0 but not C^1 across block interfaces (grid lines
  !     kink), which is the intended stress.
  !
  ! ANISOTROPY.  With mx cells in xc, each block gets mx/N_BLOCKS cells,
  ! so a block's physical cell width is proportional to its local width
  ! (1-eta)*wbot_k + eta*wtop_k.  Unequal wbot/wtop values therefore
  ! give cells that vary in width both from block to block and from the
  ! bottom to the top of a single block.  Choose mx a multiple of
  ! N_BLOCKS so block interfaces land on cell edges.
  !-------------------------------------------------------------------

  ! NOTE ON SYNTAX: the breakpoint arrays below use the old (/ ... /)
  ! array-constructor form rather than [ ... ].  f2py's crackfortran
  ! parser cannot handle [ ... ] in a module-level PARAMETER: it reads
  ! the commas as further entity declarations and invents parameters
  ! named '1', '3', '5', which then emit invalid C.  (/ ... /) parses
  ! correctly and is identical Fortran.  Keep this form if the module
  ! is wrapped with f2py.  Inside procedure bodies [ ... ] is fine.

  integer, parameter :: N_BLOCKS = 5

  double precision, parameter :: X_LEFT  = -5.d0, X_RIGHT = 5.d0
  double precision, parameter :: Y_BOT   = -1.d0, Y_TOP   = 1.d0

  ! ---- moderate anisotropy (default) -------------------------------
  ! bottom widths 2.0 2.0 2.0 2.0 2.0 ; top widths 2.8 1.8 1.2 2.6 1.6
  ! offsets d_k = X_TOP-X_BOT : 0.0 0.8 0.6 -0.2 0.4 0.0  (all
  ! consecutive pairs differ, so no block is a parallelogram)
  ! max interface tilt atan(0.8/2) = 21.8 deg from vertical
  double precision, parameter :: X_BOT(0:N_BLOCKS) = &
        (/ -5.0d0, -3.0d0, -1.0d0,  1.0d0,  3.0d0, 5.0d0 /)
  double precision, parameter :: X_TOP(0:N_BLOCKS) = &
        (/ -5.0d0, -2.2d0, -0.4d0,  0.8d0,  3.4d0, 5.0d0 /)

  ! ---- aggressive variant (swap in to stress the solver harder) -----
  ! width ratio 6:1 across blocks, 5:1 within block 1, 45 deg tilt
  ! double precision, parameter :: X_BOT(0:N_BLOCKS) = &
  !       (/ -5.0d0, -4.0d0, -1.5d0,  0.0d0,  2.0d0, 5.0d0 /)
  ! double precision, parameter :: X_TOP(0:N_BLOCKS) = &
  !       (/ -5.0d0, -2.5d0, -2.0d0,  1.0d0,  4.0d0, 5.0d0 /)

  contains

    !---------------------------------------------------------------
    pure function block_index(xc) result(ib)
    ! Computational strip containing xc, clamped to [0,N_BLOCKS-1].
    ! Clamping is what handles ghost cells: points with xc<0 or xc>1
    ! are mapped by the smooth extension of the end block's bilinear
    ! map, which is the natural extrapolation outside the domain.
      implicit none
      double precision, intent(in) :: xc
      integer :: ib
      ib = floor(xc*dble(N_BLOCKS))
      ib = max(0, min(N_BLOCKS-1, ib))
    end function block_index

    !---------------------------------------------------------------
    subroutine mapc2p(xc, yc, xp, yp)
    ! Maps computational coords (xc,yc) in [0,1]x[0,1] into the
    ! multi-block trapezoidal tiling of the physical rectangle.
      implicit none
      real(8), intent(in)  :: xc, yc
      real(8), intent(out) :: xp, yp
      integer :: ib
      real(8) :: x_rectangle(2), y_rectangle(2)
      real(8) :: x_trap(4), y_trap(4)

      ib = block_index(xc)

      x_rectangle = [dble(ib)/dble(N_BLOCKS), dble(ib+1)/dble(N_BLOCKS)]
      y_rectangle = [0.d0, 1.d0]

      x_trap = [X_BOT(ib), X_BOT(ib+1), X_TOP(ib+1), X_TOP(ib)]
      y_trap = [Y_BOT,     Y_BOT,       Y_TOP,       Y_TOP    ]

      call map_rect_to_trap(xc, yc, xp, yp, x_rectangle, y_rectangle, &
                            x_trap, y_trap)
    end subroutine mapc2p

    !---------------------------------------------------------------
    subroutine map_rect_to_trap(xc, yc, xp, yp, &
                                x_rectangle, y_rectangle, &
                                x_trap, y_trap)
      !------------------------------------------------------------
      ! Bilinear map of a rectangular computational block
      ! [xc_min,xc_max]x[yc_min,yc_max] onto the quadrilateral with
      ! vertices (lower left, lower right, upper right, upper left).
      ! Unchanged from the original version.
      !------------------------------------------------------------
      implicit none
      real(8), intent(in)  :: xc, yc
      real(8), intent(in)  :: x_rectangle(2), y_rectangle(2)
      real(8), intent(in)  :: x_trap(4), y_trap(4)
      real(8), intent(out) :: xp, yp

      real(8) :: xc_min, xc_max, yc_min, yc_max
      real(8) :: x1, y1, x2, y2, x3, y3, x4, y4
      real(8) :: xi, eta

      xc_min = x_rectangle(1); xc_max = x_rectangle(2)
      yc_min = y_rectangle(1); yc_max = y_rectangle(2)

      x1 = x_trap(1); y1 = y_trap(1)
      x2 = x_trap(2); y2 = y_trap(2)
      x3 = x_trap(3); y3 = y_trap(3)
      x4 = x_trap(4); y4 = y_trap(4)

      xi  = (xc - xc_min) / (xc_max - xc_min)
      eta = (yc - yc_min) / (yc_max - yc_min)

      xp = (1.d0-xi)*(1.d0-eta)*x1 + xi*(1.d0-eta)*x2 + xi*eta*x3 + (1.d0-xi)*eta*x4
      yp = (1.d0-xi)*(1.d0-eta)*y1 + xi*(1.d0-eta)*y2 + xi*eta*y3 + (1.d0-xi)*eta*y4
    end subroutine map_rect_to_trap

    !---------------------------------------------------------------
    subroutine vectorized_mapc2p(nx,ny,xc_array, yc_array, xp_array, yp_array)
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

    !---------------------------------------------------------------
    subroutine check_blocks(ok)
    ! Verifies conditions (a) and (b) above and reports block geometry.
    ! Call once at startup; it is cheap and catches every way the
    ! partition data can be wrong.
      implicit none
      logical, intent(out) :: ok
      integer :: k
      double precision :: wb, wt, area, total

      ok = .true.
      total = 0.d0

      if (X_BOT(0) /= X_LEFT  .or. X_TOP(0) /= X_LEFT ) ok = .false.
      if (X_BOT(N_BLOCKS) /= X_RIGHT .or. X_TOP(N_BLOCKS) /= X_RIGHT) ok = .false.

      write(*,'(A)') " blk   x_bot span        x_top span      w_bot   w_top    area   tilt(deg)"
      do k = 0, N_BLOCKS-1
            wb = X_BOT(k+1) - X_BOT(k)
            wt = X_TOP(k+1) - X_TOP(k)
            if (wb <= 0.d0 .or. wt <= 0.d0) ok = .false.
            area = 0.5d0*(wb+wt)*(Y_TOP-Y_BOT)
            total = total + area
            write(*,'(I4,2(2X,F7.3,A,F7.3),3F8.3,F10.3)') k, &
                  X_BOT(k)," ->",X_BOT(k+1), X_TOP(k)," ->",X_TOP(k+1), &
                  wb, wt, area, &
                  datan((X_TOP(k)-X_BOT(k))/(Y_TOP-Y_BOT))*180.d0/dacos(-1.d0)
      end do
      write(*,'(A,F10.6,A,F10.6)') " total area =", total, &
            "   rectangle area =", (X_RIGHT-X_LEFT)*(Y_TOP-Y_BOT)
      if (dabs(total - (X_RIGHT-X_LEFT)*(Y_TOP-Y_BOT)) > 1.d-12) ok = .false.
    end subroutine check_blocks

end module map_module