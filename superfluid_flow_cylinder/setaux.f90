!     ============================================
subroutine setaux(mbc,mx,my,xlow,ylow,dxc,dyc, maux,aux)
!     ============================================
!
!     # set auxiliary arrays for advection on a curvilinear grid
!
!     # on input, (xc(i),yc(j)) gives uniformly spaced computational grid.
!     # on output, 

!     The normal vector for the face at the i'th Riemann problem
!     is stored in the aux array
!     in locations (1,2) if ixy=1 or (4,5) if ixy=2.  The ratio of the
!     length of the cell side to the length of the computational cell
!     is stored in aux(3) or aux(6), respectively.
!     aux(7,i,j) = kappa  is ratio of cell area to dxc*dyc
!
use amr_module, only: xlower, ylower
use map_module, only: mapc2p
use cparam_mod, only: inclusion_type,inclusion_idx
implicit none
double precision, intent(inout), dimension(maux, 1-mbc:mx+mbc,1-mbc:my+mbc) :: aux
! double precision :: xccorn(5),yccorn(5),xpcorn(5),ypcorn(5)
double precision :: area
integer :: i,j,ic
double precision :: dxc, dyc
double precision :: stream
double precision, intent(in) :: xlow, ylow
integer, intent(in) :: mbc, mx, my, maux


!Locals
integer :: ilo,jlo
double precision :: a_x, a_y, anorm, length_ratio_left
double precision :: b_x, b_y, bnorm, length_ratio_bottom
double precision :: area_min, area_max


!Making code more readable
!Computational coordinates of corners of cell:
double precision :: xcLL,xcLR,xcUL,xcUR
double precision :: ycLL,ycLR,ycUL,ycUR
!Physical coordinates of corners of cell:
double precision :: xpLL,xpLR,xpUL,xpUR
double precision :: ypLL,ypLR,ypUL,ypUR

!To determine if cell is inside the inclusion
double precision :: inside_inclusion, xc, yc,xp, yp
real(8),  parameter :: x0 = 20.d0, y0 = 0.d0
real(8),  parameter :: r1 = 1.5d0, r2 = 2.04d0



!Indices of lower left corner of computational domain
ilo = floor((xlow - xlower + .05*dxc)/dxc)
jlo = floor((ylow - ylower + .05*dyc)/dyc)

!Initialize area min and max
area_min = 1.e6
area_max = 0.d0

do j=1-mbc,my+mbc
    do i=1-mbc,mx+mbc
        !# computational points (xc,yc) are mapped to physical
        !# coordinates (xp,yp) by mapc2p:
        !# lower left corner:
        xcLL = xlower + (ilo + i-1)*dxc
        ycLL = ylower + (jlo + j-1)*dyc
        call mapc2p(xcLL,ycLL,xpLL,ypLL)

        !# upper left corner:
        xcUL = xcLL
        ycUL = ycLL + dyc
        call mapc2p(xcUL,ycUL,xpUL,ypUL)

        !# upper right corner:
        xcUR = xcUL + dxc
        ycUR = ycLL + dyc
        call mapc2p(xcUR,ycUR,xpUR,ypUR)

        !# lower right corner:
        xcLR = xcLL + dxc
        ycLR = ycLL
        call mapc2p(xcLR,ycLR,xpLR,ypLR)

        !Compute normal vectors and length ratios
        a_x =  ypUL - ypLL  !upper left and lower left
        a_y = -(xpUL - xpLL)
        anorm = sqrt(a_x**2 + a_y**2)
        if (anorm > 1d-15) then !Safeguard against zero length
            a_x = a_x/anorm
            a_y = a_y/anorm
        else
            print * ,'Warning: zero length left side in setaux: ', anorm
            a_x = 0.d0
            a_y = 0.d0
            anorm = 0.d0
        endif
        length_ratio_left = anorm/dyc

        b_x = -(ypLR - ypLL)  !lower right and lower left
        b_y =   xpLR - xpLL
        bnorm = sqrt(b_x**2 + b_y**2)
        if (bnorm > 1d-15) then !Safeguard against zero length
            b_x = b_x/bnorm
            b_y = b_y/bnorm
        else
            print * ,'Warning: zero length left side in setaux: ', bnorm
            b_x = 0.d0
            b_y = 0.d0
            bnorm = 0.d0
        end if
        length_ratio_bottom = bnorm/dxc

        !Compute area of physical cell from four corners:
        area =0.d0
        area = area + 0.5d0*(ypLL+ypUL)*(xpUL - xpLL)
        area = area + 0.5d0*(ypUL+ypUR)*(xpUR - xpUL)
        area = area + 0.5d0*(ypUR+ypLR)*(xpLR - xpUR)
        area = area + 0.5d0*(ypLR+ypLL)*(xpLL - xpLR)
        ! Limit area to avoid unphysical values
        area = min(area,area_min)
        area = max(area,area_max)


        ! !Cell center inside inclusion?
        xc = xlower + (ilo + i-0.5d0)*dxc
        yc = ylower + (jlo + j-0.5d0)*dyc
        call mapc2p(xc,yc,xp,yp)
        if ( ((xp - x0)**2 + (yp - y0)**2) <= r1**2 ) then
            inside_inclusion = 1.d0
        else 
            inside_inclusion = 0.d0
        end if

        !One of the corners inside inclusion?
        ! inside_inclusion = 0.d0
        ! if ( ((xpLL - x0)**2 + (ypLL - y0)**2) <= r1**2 ) inside_inclusion = 1.d0
        ! if ( ((xpLR - x0)**2 + (ypLR - y0)**2) <= r1**2 ) inside_inclusion = 1.d0
        ! if ( ((xpUL - x0)**2 + (ypUL - y0)**2) <= r1**2 ) inside_inclusion = 1.d0
        ! if ( ((xpUR - x0)**2 + (ypUR - y0)**2) <= r1**2 ) inside_inclusion = 1.d0

        aux(1,i,j) = a_x!1.d0 !
        aux(2,i,j) = a_y!0.d0 !
        aux(3,i,j) = length_ratio_left!1.d0 !
        aux(4,i,j) = b_x!0.d0 !
        aux(5,i,j) = b_y!1.d0 !
        aux(6,i,j) = length_ratio_bottom!1.d0 !
        aux(7,i,j) = area/(dxc*dyc) !Capacity function!1.d0 !
        if (inclusion_idx .gt. 0) then
            aux(inclusion_idx,i,j) = inside_inclusion !1.d0 !
        end if
    end do
end do

end subroutine setaux