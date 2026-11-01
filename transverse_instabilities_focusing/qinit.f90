!     =====================================================
subroutine qinit(meqn,mbc,mx,my,xlower,ylower,&
                  dx,dy,q,maux,aux)
!     =====================================================
!
!     # Set initial conditions for q.
!     # All the variables are 0 except for the real part of z0, which is a radial smoothed front (Busto et al. 2021).
!
      use soliton_module, only: soliton_NLSH_1D
      use cparam_mod, only: tau,kappa,pnonlinear
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

      double precision :: random_noise(2)


      if (kappa>0.d0) then
         mu = 1.d0
      else
         mu = -1.d0
      end if
      K0 = 0.d0; K1 = 0.d0
      soliton_speed = 0.d0
      sigma  = dsqrt(mu/kappa)

!     # set concentration profile
!     ---------------------------
      do j=1,my
         yj = ylower + (j-0.5d0)*dy
         !Introduce small random noise in y-direction only for transverse instability
         call random_number(random_noise)
         random_noise = 2.e-2*(random_noise - 0.5d0) ! Scale to [-1e-2, 1e-2]
         do i=1,mx
            q(:,i,j)=0.d0
            xi = xlower + (i-0.5d0)*dx
            sim_var = xi
            temp_array = soliton_NLSH_1D(sim_var,0.d0,kappa,mu,sigma,K0,K1,soliton_speed,0.d0,0.d0)
            q(1,i,j) = temp_array(1)*(1.d0+random_noise(1)) !Real part of q0
            q(2,i,j) = temp_array(2) !Imaginary part of q0
            q(4,i,j) = temp_array(3)*(1.d0+random_noise(2)) !Real part of q1
            q(5,i,j) = temp_array(4) !Imaginary part of q1
         end do
         

      end do

end subroutine qinit

      