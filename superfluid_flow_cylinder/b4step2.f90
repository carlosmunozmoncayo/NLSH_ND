!> Called before each call to step2.
!! Use to set time-dependent aux arrays or perform other tasks.
!!
!! This version generates a wavetrain at the left boundary, and
!! absorbs it at the right boundary.
subroutine b4step2(mbc,mx,my,meqn,q,xlow,ylow,dxc,dyc,t,dt,maux,aux)

    use amr_module, only: xlower, ylower
    use map_module, only: mapc2p
    use soliton_module, only: theta, plane_wave_NLS, plane_wave_NLSH
    use cparam_mod, only: tau,kappa,pnonlinear,inclusion_type,inclusion_idx
    implicit none
    integer, intent(in) :: mbc,mx,my,meqn,maux
    real(kind=8), intent(in) :: xlow,ylow,dxc,dyc,t,dt
    real(kind=8), intent(inout) :: q(meqn,1-mbc:mx+mbc,1-mbc:my+mbc)
    real(kind=8), intent(inout) :: aux(maux,1-mbc:mx+mbc,1-mbc:my+mbc)

    ! Local variables
    integer :: i,j
    real(kind=8) :: xc, yc
    real(kind=8) :: xp, yp
    integer :: ilo,jlo

    real(kind=8) :: sigma, k_plane_wave, soliton_speed, mu
    real(kind=8) :: temp_boundary_value(4)
    real(kind=8) :: alpha
    real(kind=8) :: z0, w0

    real(kind=8) :: inside_inclusion


    if (kappa>0.d0) then
       mu = 1.d0
    else
       mu = -1.d0
    end if

    soliton_speed = 1.d0
    sigma  = dsqrt(mu/kappa)
    k_plane_wave = soliton_speed/2.d0


    ilo = floor((xlow - xlower + .05*dxc)/dxc)
    jlo = floor((ylow - ylower + .05*dyc)/dyc)

    do j=1-mbc,my+mbc
        do i=1-mbc,mx+mbc
            xc = xlower + (ilo + i-0.5d0)*dxc
            yc = ylower + (jlo + j-0.5d0)*dyc
            call mapc2p(xc,yc,xp,yp)
            !We generate the wavetrain with x in [-10,-5]
            if (xp .lt. -10.d0) then
                alpha = transition_function(xp,-30.d0,-10.d0,3) !Smoother close to -10 (1), steeper close to -30 (0)
                temp_boundary_value(1:4) = plane_wave_NLSH(xp,t,kappa,sigma,k_plane_wave,tau,1.d0)
                !Set q as convex combination
                q(1,i,j) = alpha*q(1,i,j) + (1.d0-alpha)*temp_boundary_value(1)
                q(4,i,j) = alpha*q(4,i,j) + (1.d0-alpha)*temp_boundary_value(2)
                q(2,i,j) = alpha*q(2,i,j) + (1.d0-alpha)*temp_boundary_value(3)
                q(5,i,j) = alpha*q(5,i,j) + (1.d0-alpha)*temp_boundary_value(4)
            end if
            !Absorb outgoing waves for x>40
            if (xp .gt. 50.d0) then
                ! Relaxation (not working well):
                alpha = transition_function(xp,50.d0,70.d0,3) !Smoother close to 50 (0), steeper close to 70 (1)
                temp_boundary_value(1:4) = plane_wave_NLSH(xp,t,kappa,sigma,k_plane_wave,tau,1.d0)
                q(1,i,j) = (1.d0-alpha)*q(1,i,j)+alpha*temp_boundary_value(1)
                q(4,i,j) = (1.d0-alpha)*q(4,i,j)+alpha*temp_boundary_value(2)
                q(2,i,j) = (1.d0-alpha)*q(2,i,j)+alpha*temp_boundary_value(3)
                q(5,i,j) = (1.d0-alpha)*q(5,i,j)+alpha*temp_boundary_value(4)
            end if
            !Absorb outgoing waves for y>30
            if (yp .gt. 30.d0) then
                alpha = transition_function(yp,30.d0,50.d0,3) 
                temp_boundary_value(1:4) = plane_wave_NLSH(xp,t,kappa,sigma,k_plane_wave,tau,1.d0)
                q(1,i,j) = (1.d0-alpha)*q(1,i,j)+alpha*temp_boundary_value(1)
                q(4,i,j) = (1.d0-alpha)*q(4,i,j)+alpha*temp_boundary_value(2)
                q(2,i,j) = (1.d0-alpha)*q(2,i,j)+alpha*temp_boundary_value(3)
                q(5,i,j) = (1.d0-alpha)*q(5,i,j)+alpha*temp_boundary_value(4)
            end if
            !Absorb outgoing waves for y<-30
            if (yp .lt. -30.d0) then
                alpha = transition_function(yp,-50.d0,-30.d0,3)
                temp_boundary_value(1:4) = plane_wave_NLSH(xp,t,kappa,sigma,k_plane_wave,tau,1.d0)
                q(1,i,j) = alpha*q(1,i,j)+(1.d0-alpha)*temp_boundary_value(1)
                q(4,i,j) = alpha*q(4,i,j)+(1.d0-alpha)*temp_boundary_value(2)
                q(2,i,j) = alpha*q(2,i,j)+(1.d0-alpha)*temp_boundary_value(3)
                q(5,i,j) = alpha*q(5,i,j)+(1.d0-alpha)*temp_boundary_value(4)
            end if
            !Set inclusion if any
            if ((inclusion_idx .gt. 0) .and. ((trim(inclusion_type)=='dirichlet').or.(trim(inclusion_type)=='neumann'))) then
                if (aux(inclusion_idx,i,j)  .gt. 0.5d0) then
                    !aux(inclusion_idx,i,j) = 1.d0 inside inclusion, 0.d0 outside
                    !We should just set q=0 for Dirichlet, but maybe some fluctuations
                    !are updating the boundary cells of the inclusion (wouldn't hurt 
                    !either way: For Neumann we ignore those values in the normal Riemann solves)
                    q(:,i,j) = 0.d0
                end if
            end if

        end do
    end do

    contains
    function transition_function(s,s0,s1,optional_transition_type_fun,optional_shape) result(fs)
        !Returns f(s), where f(s0)=0 and f(s1)=1
        implicit none
        double precision, intent(in) :: s, s0, s1
        integer, optional, intent(in) :: optional_transition_type_fun
        double precision, optional, intent(in) :: optional_shape
        !Output
        double precision :: fs
        !Locals
        double precision :: shape = 1.d0 !Different effects depending on transition_type
        integer :: transition_type_fun = 1
        double precision :: sn !Normalized s (0<=sn<=1)
        !Edge cases
        if (s < s0) then
            fs = 0.d0
            return
        else if (s >= s1) then
            fs = 1.d0
            return
        end if
        !Normalize and get linear transition
        sn = (s-s0)/(s1-s0)
        if (present(optional_shape)) shape = optional_shape
        if (present(optional_transition_type_fun)) transition_type_fun = optional_transition_type_fun
        !Transition function
        select case(transition_type_fun)
            case(1) !Linear (can be made a Heaviside if s0==s1)
                fs = sn
            case(2) !Hyperbolic tangent
                !Larger shape -> sharper transition, better match at 0 and 1
                !Smaller shape -> smoother transition, does not match 0 and 1
                fs = 0.5*(tanh(10.d0*shape*(sn-0.5))+1)
            case(3) !Engsig-Karup (2007)
                !Matches 0 and 1, smooth in the middle, sharper at the ends
                fs = -2.d0*sn**3+3.d0*sn**2
            case(4) !Mayer et al. (1998)
                !Very smooth at 0, does not match 1
                !Smaller shape -> smoother transition at 0, better match at 1
                !and faster decrease close to 1
                fs = (0.01d0*shape*sn)**3+(1.d0-0.01d0*shape)*sn**6
            case(5) !Modified Mayer et al. (1998)
                !Very smooth at 1, does not match 0
                !Smaller shape -> smoother transition at 1, better match at 0
                !and faster decrease close to 0
                fs = 1.d0-(0.01d0*shape*(1.d0-sn))**3+(1.d0-0.01d0*shape)*(1.d0-sn)**6
            case default
                fs = sn !Linear
        end select
    end function transition_function


end subroutine b4step2
