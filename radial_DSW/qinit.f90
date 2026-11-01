!     =====================================================
subroutine qinit(meqn,mbc,mx,my,xlower,ylower,&
                  dx,dy,q,maux,aux)
!     =====================================================
!
!     # Set initial conditions for q.
!     # All the variables are 0 except for the real part of z0, which is a radial smoothed front (Busto et al. 2021).

    use map_module, only: mapc2p
    !
    implicit none 
    ! intrinsic :: sqrt, tanh
    double precision :: q(meqn, 1-mbc:mx+mbc, 1-mbc:my+mbc)
    double precision :: aux(maux, 1-mbc:mx+mbc, 1-mbc:my+mbc)

    integer :: meqn, mbc, mx, my, maux
    double precision :: xlower, ylower, dx, dy
    double precision :: xi, yj
    integer :: i, j

    !     #Local variables
    double precision :: r, rhoL, rhoR, rCap, sigma
    double precision :: xp, yp


    rhoL = 2.d0
    rhoR = 1.d0
    rCap = 3.d0
    sigma = 0.d0!0.01d0
      
!     # set concentration profile
!     ---------------------------
    if (maux<2) then
        !Cartesian coordinates
        do j=1,my
            yj = ylower + (j-0.5d0)*dy
            do i=1,mx
                q(:,i,j)=0.d0
                xi = xlower + (i-0.5d0)*dx
                r = sqrt((xi)**2 + (yj)**2)
                q(1,i,j) = 0.5*(rhoR+rhoL)+0.5d0*(rhoR-rhoL)*tanh((r-rCap)/sigma)
            end do
        end do
    else
        !Mapped coordinates
        do j=1,my
            yj = ylower + (j-0.5d0)*dy
            do i=1,mx
                q(:,i,j)=0.d0
                xi = xlower + (i-0.5d0)*dx
                call mapc2p(xi,yj,xp,yp)
                r = sqrt((xp)**2 + (yp)**2)
                q(1,i,j) = sqrt(0.5*(rhoR+rhoL)+0.5d0*(rhoR-rhoL)*tanh((r-rCap)/sigma))
            end do
        end do
    end if      
            
end subroutine qinit



      