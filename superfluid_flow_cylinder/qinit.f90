!     =====================================================
subroutine qinit(meqn,mbc,mx,my,xlower,ylower,&
                  dx,dy,q,maux,aux)
!     =====================================================
!
!     # Set initial conditions for q.
!     # All the variables are 0 except for the real part of z0, which is a radial smoothed front (Busto et al. 2021).
!
      use map_module, only: mapc2p
      use cparam_mod, only: tau,kappa,pnonlinear,inclusion_type,inclusion_idx
      use soliton_module, only: soliton_NLSH_1D,  plane_wave_NLS, plane_wave_NLSH
      implicit none 
      ! intrinsic :: sqrt, tanh
      double precision :: q(meqn, 1-mbc:mx+mbc, 1-mbc:my+mbc)
      double precision :: aux(maux, 1-mbc:mx+mbc, 1-mbc:my+mbc)

      integer :: meqn, mbc, mx, my, maux
      double precision :: xlower, ylower, dx, dy
      double precision :: xi, yj
      integer :: i, j

!     #Local variables
      double precision :: K0,K1, sigma, mu, phase, sim_var, soliton_speed
      double precision :: temp_array(4)
      double precision :: xp, yp
      double precision :: k_plane_wave


      if (kappa>0.d0) then
         mu = 1.d0
      else
         mu = -1.d0
      end if
      K0 = 0.d0; K1 = 0.d0
      soliton_speed = 1.d0
      sigma  = dsqrt(mu/kappa)
      k_plane_wave = soliton_speed/2.d0

      

!     # set concentration profile
!     ---------------------------
      do j=1,my
         yj = ylower + (j-0.5d0)*dy
         do i=1,mx
            q(:,i,j)=0.d0
            xi = xlower + (i-0.5d0)*dx
            call mapc2p(xi,yj,xp,yp)
            sim_var = xp
            temp_array(1:4) = plane_wave_NLSH(xp,0.d0,kappa,sigma,k_plane_wave,tau,1.d0)            ! q(1,i,j) = temp_array(1) !Real part of q0
            q(1,i,j) = temp_array(1) !Real part of q0
            q(4,i,j) = temp_array(2) !Imaginary part of q0
            q(2,i,j) = temp_array(3) !Real part of q1
            q(5,i,j) = temp_array(4) !Imaginary part of q1
            if ( ( (xp-20.d0)**2 + (yp)**2 ) .lt. 1.5d0**2 ) then
                q(:,i,j) = 0.d0
            end if
         end do
      end do
end subroutine qinit

      