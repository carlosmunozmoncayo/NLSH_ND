! :::::::::: bc2amr ::::::::::::::::::::::::::::::::::::::::::::::;
!> \callgraph
!! \callergraph
!!  Take a grid patch with mesh widths **hx**,**hy**, of dimensions **nrow** by
!!  **ncol**,  and set the values of any piece of
!!  of the patch which extends outside the physical domain 
!!  using the boundary conditions. 
!!
!!  ### Standard boundary condition choices for amr2ez in clawpack
!!
!!  At each boundary  k = 1 (left),  2 (right),  3 (bottom), 4 (top):
!!
!!  mthbc(k) =  
!!  * 0  for user-supplied BC's (must be inserted!)
!!  * 1  for zero-order extrapolation
!!  * 2  for periodic boundary conditions
!!  * 3  for solid walls, assuming this can be implemented
!!                   by reflecting the data about the boundary and then
!!                   negating the 2'nd (for k=1,2) or 3'rd (for k=3,4)
!!                   component of q.
!!  * 4  for sphere bcs (left half maps to right half of same side, and vice versa), as if domain folded in half
!!
!!  The corners of the grid patch are at 
!!     (xlo_patch,ylo_patch)  --  lower left corner
!!     (xhi_patch,yhi_patch) --  upper right corner
!!
!!  The physical domain itself is a rectangle bounded by
!!     (xlower,ylower)  -- lower left corner
!!     (xupper,yupper)  -- upper right corner
!!  
!   This figure below does not work with doxygen
!   the picture is the following: 
!  ____________________________________________________
! 
!                _____________________ (xupper,yupper)
!               |                     |  
!           ____|____ (xhi_patch,yhi_patch)   
!           |   |    |                |
!           |   |    |                |
!           |   |    |                |
!           |___|____|                |
!  (xlo_patch,ylo_patch) |            |
!               |                     |
!               |_____________________|   
!    (xlower,ylower)
!  ____________________________________________________
!!
!!
!>  Any cells that lie outside the physical domain are ghost cells whose
!!  values should be set in this routine.  This is tested for by comparing
!!  xlo_patch with xlower to see if values need to be set at the left
!   as in the figure above, 
!
!>  and similarly at the other boundaries.
!!  Patches are guaranteed to have at least 1 row of cells filled
!!  with interior values so it is possible to extrapolate. 
!!  Fix [trimbd()](@ref trimbd) if you want more than 1 row pre-set.
!!
!!  Make sure the order the boundaries are specified is correct
!!  so that diagonal corner cells are also properly taken care of.
!!
!!  Periodic boundaries are set before calling this routine, so if the
!!  domain is periodic in one direction only you
!!  can safely extrapolate in the other direction. 
!!
!!  Don't overwrite ghost cells in periodic directions!
!!
!! \param val data array for solution \f$q \f$ (cover the whole grid **msrc**)
!! \param aux data array for auxiliary variables 
!! \param nrow number of cells in *i* direction on this grid
!! \param ncol number of cells in *j* direction on this grid
!! \param meqn number of equations for the system
!! \param naux number of auxiliary variables
!! \param hx spacing (mesh size) in *i* direction
!! \param hy spacing (mesh size) in *j* direction
!! \param level AMR level of this grid
!! \param time setting ghost cell values at time **time**
!! \param xlo_patch left bound of the input grid
!! \param xhi_patch right bound of the input grid 
!! \param ylo_patch lower bound of the input grid 
!! \param yhi_patch upper bound of the input grid 
! ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::;

subroutine bc2amr(val,aux,nrow,ncol,meqn,naux, hx, hy, level, time,   &
                  xlo_patch, xhi_patch, ylo_patch, yhi_patch) 

    use amr_module, only: mthbc, xlower, ylower, xupper, yupper
    use amr_module, only: xperdom,yperdom,spheredom
    use map_module, only: mapc2p
    use cparam_mod, only: tau,kappa,pnonlinear
    ! use exact_sol_module, only: exact_solution
    use soliton_module, only: theta

    implicit none

    ! Input/Output
    integer, intent(in) :: nrow, ncol, meqn, naux, level
    real(kind=8), intent(in) :: hx, hy, time
    real(kind=8), intent(in) :: xlo_patch, xhi_patch
    real(kind=8), intent(in) :: ylo_patch, yhi_patch
    real(kind=8), intent(in out) :: val(meqn, nrow, ncol)
    real(kind=8), intent(in out) :: aux(naux, nrow, ncol)
    
    ! Local storage
    integer :: i, j, ibeg, jbeg, nxl, nxr, nyb, nyt
    real(kind=8) :: hxmarg, hymarg

    !For mapped grids
    real(kind=8) :: xc, yc, xp, yp
    real(kind=8) :: ct, st
    real(kind=8) :: u_normal(2), u_tangential(2)! 2 for real and imag parts
    real(kind=8) :: temp_boundary_value(meqn)

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !Dirichlet BCs for dark solitons
    real(kind=8) :: phase
    !Zero flux BCs variables
    real(kind=8) :: phase_interior(2), phase_ghost(2)
    real(kind=8) :: phase_x_ghost(2), phase_y_ghost(2)
    real(kind=8) :: phase_x_interior(2), phase_y_interior(2)
    real(kind=8) ::  xcint
    real(kind=8) ::  mu, sigma, soliton_speed

    if (kappa>0.d0) then
       mu = 1.d0
    else
       mu = -1.d0
    end if
    soliton_speed = 1.d0
    sigma  = dsqrt(mu/kappa)
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


    hxmarg = hx * .01d0
    hymarg = hy * .01d0

    ! Use periodic boundary condition specialized code only, if only one 
    ! boundary is periodic we still proceed below
    if (xperdom .and. (yperdom .or. spheredom)) then
        return
    end if

    ! Each check has an initial check to ensure that the boundary is a real
    ! boundary condition and otherwise skips the code.  Otherwise 
    !-------------------------------------------------------
    ! Left boundary:
    !-------------------------------------------------------
    if (xlo_patch < xlower-hxmarg) then
        ! number of grid cells from this patch lying outside physical domain:
        nxl = int((xlower + hxmarg - xlo_patch) / hx)

        select case(mthbc(1))
            case(0) ! User defined boundary condition
                do j = 1, ncol
                    yc = ylo_patch + (j - 0.5d0) * hy
                    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    do i=1, nxl
                        xc = xlo_patch + (i - 0.5d0) * hx
                        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        !Exact solution for homogeneous problem
                        call mapc2p(xc,yc,xp,yp)
                        ! call exact_solution(val(:, i, j),xp,yp,time)
                        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        !Dirichlet BCs for dark solitons
                        phase = theta(xp,time,soliton_speed,mu,0.d0)
                        val(:,i,j) = 0.d0
                        val(1,i,j) = -sigma*cos(phase) !Real part of q0_bg
                        val(4,i,j) = -sigma*sin(phase) !Real part of q1_bg
                        val(2,i,j) = -val(4,i,j)*0.5d0*soliton_speed 
                        val(5,i,j) = val(1,i,j)*0.5d0*soliton_speed !Imag part from phase gradient
                        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    end do
                end do
            case(1) ! Zero-order extrapolation
                do j = 1, ncol
                    do i=1, nxl
                        val(:, i, j) = val(:, nxl + 1, j)
                    end do
                end do

            case(2) ! Periodic boundary condition
                continue

            case(3) ! Wall boundary conditions
                do j = 1, ncol
                    do i=1, nxl
                        ! Mirror values about the boundary
                        val(:, i, j) = val(:, 2 * nxl + 1 - i, j)
                        ct = aux(1, 2 * nxl + 1 - i, j) ! cos angle between grid lines and x-axis
                        st = aux(2, 2 * nxl + 1 - i, j) ! sin angle between grid lines and x-axis
                        ! Rotate to normal/tangential components
                        u_normal(1) =  ct * val(2, i, j) + st * val(3, i, j) ! Real part
                        u_tangential(1) = -st * val(2, i, j) + ct * val(3, i, j)
                        u_normal(2) =  ct * val(5, i, j) + st * val(6, i, j) ! Imag part
                        u_tangential(2) = -st * val(5, i, j) + ct * val(6, i, j)
                        ! Reflect normal component
                        u_normal(1) = -u_normal(1)
                        u_normal(2) = -u_normal(2)
                        ! Rotate back to Cartesian components
                        val(2, i, j) =  ct * u_normal(1) - st * u_tangential(1) ! Real part
                        val(3, i, j) =  st * u_normal(1) + ct * u_tangential(1)
                        val(5, i, j) =  ct * u_normal(2) - st * u_tangential(2) ! Imag part
                        val(6, i, j) =  st * u_normal(2) + ct * u_tangential(2)
                    end do
                end do

            case(4) ! Spherical domain
                continue

            case default
                print *, "Invalid boundary condition requested."
                stop
        end select
    end if

    !-------------------------------------------------------
    ! Right boundary:
    !-------------------------------------------------------
    if (xhi_patch > xupper+hxmarg) then

        ! number of grid cells lying outside physical domain:
        nxr = int((xhi_patch - xupper + hxmarg) / hx)
        ibeg = max(nrow - nxr + 1, 1)

        select case(mthbc(2))
            case(0) ! User defined boundary condition
                ! Replace this code with a user defined boundary condition
                do i = ibeg, nrow
                    xc = xlo_patch + (i - 0.5d0) * hx
                    do j = 1, ncol
                        yc = ylo_patch + (j - 0.5d0) * hy
                        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        !Exact solution for homogeneous problem
                        call mapc2p(xc,yc,xp,yp)
                        ! call exact_solution(val(:, i, j),xp,yp,time)
                        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        !Dirichlet BCs for dark solitons
                        phase = theta(xp,time,soliton_speed,mu,0.d0)
                        val(:,i,j) = 0.d0
                        val(1,i,j) = sigma*cos(phase) !Real part of q0_bg
                        val(4,i,j) = sigma*sin(phase) !Real part of q1_bg
                        val(2,i,j) = -val(4,i,j)*0.5d0*soliton_speed 
                        val(5,i,j) = val(1,i,j)*0.5d0*soliton_speed !Imag part from phase gradient
                        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    end do
                end do

            case(1) ! Zero-order extrapolation
                do i = ibeg, nrow
                    do j = 1, ncol
                        val(:, i, j) = val(:, ibeg - 1, j)
                    end do
                end do

            case(2) ! Periodic boundary condition
                continue 

            case(3) ! Wall boundary conditions
                do i=ibeg, nrow
                    do j = 1, ncol
                        ! Mirror values about the boundary
                        val(:, i, j) = val(:, 2 * ibeg - 1 - i, j)
                        ct = aux(1, 2 * ibeg - i, j) ! cos angle between grid lines and x-axis
                        ! +1 index since the aux array is collocated at the left edge of the cell
                        st = aux(2, 2 * ibeg - i, j) ! sin angle between grid lines and x-axis
                        ! Rotate to normal/tangential components
                        u_normal(1) =  ct * val(2, i, j) + st * val(3, i, j) ! Real part
                        u_tangential(1) = -st * val(2, i, j) + ct * val(3, i, j)
                        u_normal(2) =  ct * val(5, i, j) + st * val(6, i, j) ! Imag part
                        u_tangential(2) = -st * val(5, i, j) + ct * val(6, i, j)
                        ! Reflect normal component
                        u_normal(1) = -u_normal(1)
                        u_normal(2) = -u_normal(2)
                        ! Rotate back to Cartesian components
                        val(2, i, j) =  ct * u_normal(1) - st * u_tangential(1) ! Real part
                        val(3, i, j) =  st * u_normal(1) + ct * u_tangential(1)
                        val(5, i, j) =  ct * u_normal(2) - st * u_tangential(2) ! Imag part
                        val(6, i, j) =  st * u_normal(2) + ct * u_tangential(2)
                    end do
                end do

            case(4) ! Spherical domain
                continue

            case default
                print *, "Invalid boundary condition requested."
                stop

        end select
    end if

    !-------------------------------------------------------
    ! Bottom boundary:
    !-------------------------------------------------------
    if (ylo_patch < ylower - hymarg) then

        ! number of grid cells lying outside physical domain:
        nyb = int((ylower + hymarg - ylo_patch) / hy)

        select case(mthbc(3))
            case(0) ! User defined boundary condition
                ! Replace this code with a user defined boundary condition
                do j = 1, nyb
                    yc = ylo_patch + (j - 0.5d0) * hy
                    do i = 1, nrow
                        xc = xlo_patch + (i - 0.5d0) * hx
                        call mapc2p(xc,yc,xp,yp)
                        ! call exact_solution(val(:, i, j),xp,yp,time)
                    end do
                end do
                
            
            case(1) ! Zero-order extrapolation
                do j = 1, nyb
                    do i = 1, nrow
                        val(:,i,j) = val(:, i, nyb + 1)
                    end do
                end do

            case(2) ! Periodic boundary condition
                continue

            case(3) ! Wall boundary conditions
                do j = 1, nyb
                    do i = 1, nrow
                        ! Mirror values about the boundary
                        val(:,i,j) = val(:, i, 2 * nyb + 1 - j)
                        ct = aux(4, i, 2 * nyb + 1 - j) ! cos angle between grid lines and x-axis
                        st = aux(5, i, 2 * nyb + 1 - j) ! sin angle between grid lines and x-axis
                        ! Rotate to normal/tangential components
                        u_normal(1) =  ct * val(2, i, j) + st * val(3, i, j) ! Real part
                        u_tangential(1) = -st * val(2, i, j) + ct * val(3, i, j)
                        u_normal(2) =  ct * val(5, i, j) + st * val(6, i, j) ! Imag part
                        u_tangential(2) = -st * val(5, i, j) + ct * val(6, i, j)
                        ! Reflect normal component
                        u_normal(1) = -u_normal(1)
                        u_normal(2) = -u_normal(2)
                        ! Rotate back to Cartesian components
                        val(2, i, j) =  ct * u_normal(1) - st * u_tangential(1) ! Real part
                        val(3, i, j) =  st * u_normal(1) + ct * u_tangential(1)
                        val(5, i, j) =  ct * u_normal(2) - st * u_tangential(2) ! Imag part
                        val(6, i, j) =  st * u_normal(2) + ct * u_tangential(2)
                    end do
                end do

            case(4) ! Spherical domain
                continue

            case default
                print *, "Invalid boundary condition requested."
                stop

        end select
    end if

    !-------------------------------------------------------
    ! Top boundary:
    !-------------------------------------------------------
    if (yhi_patch > yupper + hymarg) then

        ! number of grid cells lying outside physical domain:
        nyt = int((yhi_patch - yupper + hymarg) / hy)
        jbeg = max(ncol - nyt + 1, 1)

        select case(mthbc(4))
            case(0) ! User defined boundary condition
                ! Replace this code with a user defined boundary condition
                do j = jbeg, ncol
                    yc = ylo_patch + (j - 0.5d0) * hy
                    do i = 1, nrow
                        xc = xlo_patch + (i - 0.5d0) * hx
                        call mapc2p(xc,yc,xp,yp)
                        ! call exact_solution(val(:, i, j),xp,yp,time)
                    end do
                end do
                

            case(1) ! Zero-order extrapolation
                do j = jbeg, ncol
                    do i = 1, nrow
                        val(:, i, j) = val(:, i, jbeg - 1)
                    end do
                end do

            case(2) ! Periodic boundary condition
                continue

            case(3) ! Wall boundary conditions
                do j = jbeg, ncol 
                    do i = 1, nrow
                        ! Mirror values about the boundary
                        val(:, i, j) = val(:, i, 2 * jbeg - 1 - j)
                        ct = aux(4, i, 2 * jbeg - j) ! cos angle between grid lines and x-axis
                        !+1 index since the aux array is collocated at the bottom edge of the cell
                        st = aux(5, i, 2 * jbeg - j) ! sin angle between grid lines and x-axis
                        ! Rotate to normal/tangential components
                        u_normal(1) =  ct * val(2, i, j) + st * val(3, i, j)
                        u_tangential(1) = -st * val(2, i, j) + ct * val(3, i, j)
                        u_normal(2) =  ct * val(5, i, j) + st * val(6, i, j) 
                        u_tangential(2) = -st * val(5, i, j) + ct * val(6, i, j)
                        ! Reflect normal component
                        u_normal(1) = -u_normal(1)
                        u_normal(2) = -u_normal(2)
                        ! Rotate back to Cartesian components
                        val(2, i, j) =  ct * u_normal(1) - st * u_tangential(1) ! Real part
                        val(3, i, j) =  st * u_normal(1) + ct * u_tangential(1)
                        val(5, i, j) =  ct * u_normal(2) - st * u_tangential(2) ! Imag part
                        val(6, i, j) =  st * u_normal(2) + ct * u_tangential(2)
                    end do
                end do

            case(4) ! Spherical domain
                continue

            case default
                print *, "Invalid boundary condition requested."
                stop

        end select
    end if

end subroutine bc2amr