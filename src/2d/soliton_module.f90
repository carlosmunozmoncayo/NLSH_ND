module soliton_module
    implicit none
    contains
    pure function soliton_NLSH_1D(sim_var,tau,kappa,mu,sigma,K0,K1,soliton_speed,t,zeta)
      !Only a real soliton if tau=0.d0
            implicit none
            double precision, intent(in) :: sim_var, tau,kappa, mu, sigma
            double precision, intent(in) :: K0, K1, soliton_speed
            double precision, intent(in) :: t, zeta
            double precision :: soliton_NLSH_1D(4)
            double precision :: phase, temp_array(2)
            soliton_NLSH_1D = 0.d0
            temp_array = ground_state_NLSH_1D(sim_var, tau, kappa, mu, sigma, K0, K1)
            phase = theta(sim_var,t,soliton_speed,mu,zeta)
            soliton_NLSH_1D(1) = cos(phase)*temp_array(1) !Real part of q0
            soliton_NLSH_1D(2) = cos(phase)*temp_array(2) !Real part of q1
            soliton_NLSH_1D(3) = sin(phase)*temp_array(1) !Imaginary part of q0
            soliton_NLSH_1D(4) = sin(phase)*temp_array(2) !Imaginary part of q1
      end function soliton_NLSH_1D

      pure function ground_state_NLSH_1D(sim_var,tau,kappa,mu,sigma,K0,K1)
      !To get an NLS ground state, one just has to pass tau=0.d0
            implicit none
            double precision, intent(in) :: sim_var, tau,kappa, mu, sigma, K0, K1
            double precision :: ground_state_NLSH_1D(2)
            double precision :: temp
            ground_state_NLSH_1D = 0.d0
                  if (kappa>=0.d0)then
                        !Focusing case
                        ground_state_NLSH_1D(1) = dsqrt(2.d0)*sigma*sech(dsqrt(mu*(1-mu*tau))*sim_var+K1)
                        ground_state_NLSH_1D(2) = -(dsqrt(2.d0*mu)*sigma/dsqrt(1-mu*tau))*&
                              sech(dsqrt(mu*(1-mu*tau))*sim_var+K1) &
                              *tanh(dsqrt(mu*(1-mu*tau))*sim_var+K1)
                  else 
                        !Defocusing case
                        temp = dsqrt(kappa*(mu*tau-1)/2.d0)
                        ground_state_NLSH_1D(1) = sigma*tanh(sigma*temp*sim_var+K0)
                        ground_state_NLSH_1D(2) = (sigma*sigma*temp/(1-mu*tau))*&
                                    sech(sigma*temp*sim_var+K0)**2.d0
                  end if
      end function ground_state_NLSH_1D
      pure function theta(x,t,c,mu,zeta)
            implicit none
            double precision, intent(in) :: x, t,c,mu,zeta
            double precision :: theta
            theta = zeta+0.5d0*c*x-((c**2.d0)/4.d0-mu)*t
      end function theta
      pure function sech(x)
            implicit none
            double precision, intent(in) :: x
            double precision :: sech
            sech = 1.d0 / cosh(x)
      end function sech

      pure function plane_wave_NLS(x,t,kappa,sigma,k)
      !Returns real and imaginary parts of u and u_x,
      !where u = sigma*exp(i*(k*x - omega*t)) is a plane wave solution to NLS
            implicit none
            double precision, intent(in) :: x, t, kappa, sigma, k
            double precision :: plane_wave_NLS(4) !First two components are real and imag parts of u, last two are real and imag parts of u_x
            double precision :: omega
            omega = k**2.d0 - kappa*sigma**2.d0
            plane_wave_NLS(1) = sigma * cos(k*x - omega*t) !Real part of u
            plane_wave_NLS(2) = sigma * sin(k*x - omega*t) !Imag part of u
            plane_wave_NLS(3) = -k * sigma * sin(k*x - omega*t) !Real part of u_x
            plane_wave_NLS(4) = k * sigma * cos(k*x - omega*t) !Imag part of u_x
      end function plane_wave_NLS

      pure function plane_wave_NLSH(x,t,kappa,sigma,k,tau,sgn)
      !Returns real and imaginary parts of q0(approximates u) and q1 (approximates u_x),
      !where q0 = sigma*exp(i*(k*x - omega*t)) and
      !q1 = q0_x*i*k/(tau*omega+1) are plane wave solutions to NLSH
      !The stable branch is sgn=1
            implicit none
            double precision, intent(in) :: x, t, kappa, sigma, k, tau, sgn
            double precision :: plane_wave_NLSH(4) !First two components are real and imag parts of u, last two are real and imag parts of u_x
            double precision :: kappas, omega, coeffq1
            kappas = kappa*sigma**2.d0
            omega = -(kappas*tau+1.d0)
            omega = omega + sgn*sqrt((kappas*tau-1.d0)**2.d0+4.d0*k**2.d0*tau)
            omega = omega / (2.d0*tau)
            ! if (tau<1.e-2.and. sgn>0) then
            !       !Using asymptotic expansion for small tau to avoid numerical issues
            !       omega = (k**2.d0 - kappas)*(1.d0-k**2.d0*tau) !+O(tau^2)
            ! end if
            coeffq1 = k/(tau*omega+1.d0)
            plane_wave_NLSH(1) = sigma * cos(k*x - omega*t) !Real part of q0
            plane_wave_NLSH(2) = sigma * sin(k*x - omega*t) !Imag part of q0
            plane_wave_NLSH(3) = -coeffq1 * plane_wave_NLSH(2) !Real part of q1 
            plane_wave_NLSH(4) = coeffq1 * plane_wave_NLSH(1) !Imag part of q1
      end function plane_wave_NLSH

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      !NLS grey soliton 
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      pure function grey_soliton_sound_speed(kappa,sigma)
      !Bogoliubov sound speed c_s = sqrt(-2*kappa)*sigma.  NaN if kappa >= 0.
            implicit none
            double precision, intent(in) :: kappa, sigma
            double precision :: grey_soliton_sound_speed
            grey_soliton_sound_speed = dsqrt(-2.d0*kappa)*sigma
      end function grey_soliton_sound_speed
 
      pure function grey_soliton_admissible(kappa,sigma,soliton_speed)
      !.true. iff a grey soliton with this (speed, amplitude) pair exists:
      !defocusing, positive background, and strictly subsonic.
      !Written without sqrt so it is safe to call for any input.
            implicit none
            double precision, intent(in) :: kappa, sigma, soliton_speed
            logical :: grey_soliton_admissible
            grey_soliton_admissible = (kappa < 0.d0) .and. (sigma > 0.d0) .and. &
                  (soliton_speed*soliton_speed < -2.d0*kappa*sigma*sigma)
      end function grey_soliton_admissible
 
      pure function grey_soliton_NLSH_1D(x,t,tau,kappa,sigma,soliton_speed, &
                                         x0,zeta,well_prepared)
      !4 components of the 1D NLSH state for a grey soliton on a still
      !background, ordered [Re q0, Re q1, Im q0, Im q1] to match
      !soliton_NLSH_1D above.
      !  x0   : initial soliton centre
      !  zeta : global background phase offset
      !  well_prepared : .true.  -> q1 = (1+tau*mu)*f' + i*tau*v*f''  (O(tau^2))
      !                  .false. -> q1 = d_x q0                       (O(tau))
      !Note mu is NOT an argument: mu = kappa*sigma^2 is fixed by the
      !background and an inconsistent value would not solve NLS.
            implicit none
            double precision, intent(in) :: x, t, tau, kappa, sigma
            double precision, intent(in) :: soliton_speed, x0, zeta
            logical,          intent(in) :: well_prepared
            double precision :: grey_soliton_NLSH_1D(4)
            double precision :: cs, mach, B, C, beta, mu, xi, Tnh, S2
            double precision :: phase, cth, sth, P, Q, fp, fpp
 
            grey_soliton_NLSH_1D = 0.d0
 
            cs   = dsqrt(-2.d0*kappa)*sigma      !NaN if kappa >= 0
            mach = soliton_speed/cs
            C    = mach                          !greyness
            B    = dsqrt(1.d0 - mach*mach)       !NaN if |v| >= c_s
            mu   = kappa*sigma*sigma             !background frequency
            beta = B*sigma*dsqrt(-0.5d0*kappa)   !inverse width
 
            xi  = x - soliton_speed*t - x0
            Tnh = dtanh(beta*xi)
            S2  = sech(beta*xi)**2.d0
 
            phase = mu*t + zeta
            cth   = dcos(phase)
            sth   = dsin(phase)
 
            !f(xi) = sigma*(B*tanh(beta*xi) + i*C);  f', f'' are real
            fp  =  sigma*B*beta*S2
            fpp = -2.d0*sigma*B*beta*beta*Tnh*S2
 
            !q1 = exp(i*phase)*(P + i*Q)
            if (well_prepared) then
                  P = (1.d0 + tau*mu)*fp
                  Q = tau*soliton_speed*fpp
            else
                  P = fp
                  Q = 0.d0
            end if
 
            grey_soliton_NLSH_1D(1) = sigma*(B*Tnh*cth - C*sth) !Real part of q0
            grey_soliton_NLSH_1D(2) = P*cth - Q*sth             !Real part of q1
            grey_soliton_NLSH_1D(3) = sigma*(B*Tnh*sth + C*cth) !Imaginary part of q0
            grey_soliton_NLSH_1D(4) = P*sth + Q*cth             !Imaginary part of q1
      end function grey_soliton_NLSH_1D

end module soliton_module