subroutine src2(meqn,mbc,mx,my,xlower,ylower,dx,dy,q,maux,aux,t,dt)

!> Called to update q by solving source term equation 
!! $q_t = S(q)$ over time dt starting at time t.
!!
!! Since we can show that mass is conserved for this substep, the source
!! is linear and we can solve exactly by
!! q(dt)= exp(dt S) q(0)
    use cparam_mod, only: tau,kappa,pnonlinear,srcintegrator
    implicit none
    integer, intent(in) :: mbc,mx,my,meqn,maux
    real(kind=8), intent(in) :: xlower,ylower,dx,dy,t,dt
    real(kind=8), intent(in) ::  aux(maux,1-mbc:mx+mbc,1-mbc:my+mbc)
    real(kind=8), intent(inout) ::  q(meqn,1-mbc:mx+mbc,1-mbc:my+mbc)


    ! Conserved quantities:
    !       1   z0 (first component of z)
    !       2   first component of z1
    !       3   second component of z1
    !       4   w0 (first component of w)
    !       5   first component of w1
    !       6   second component of w1
    ! local
    integer :: i,j
    real(kind=8) :: tauinv

    tauinv = 1.d0/tau

    !Checking integrator type
    ! print *, 'srcintegrator =', srcintegrator    
    if (trim(srcintegrator)=='exact') then
        ! print *, 'Using exact integrator'
        ! loop over all grid points
        do i = 1, mx
            do j = 1, my
                    q(:,i,j) = exact_integrator(tauinv,kappa,q(:,i,j),dt,pnonlinear) 
            end do
        end do

    else if (trim(srcintegrator)=='dirk22') then   
        ! print *, 'Using DIRK22 integrator'
        ! loop over all grid points
        do i = 1, mx
            do j = 1, my
                    q(:,i,j) = DIRK22(tauinv,kappa,q(:,i,j),dt,pnonlinear) 
            end do
        end do

    else if (trim(srcintegrator)=='be') then
        ! print *, 'Using Backward Euler integrator'
        ! loop over all grid points
        do i = 1, mx
            do j = 1, my
                    q(:,i,j) = BE(tauinv,kappa,q(:,i,j),dt,pnonlinear) 
            end do
        end do

    else
        print *, 'Error: srcintegrator not recognized. Use exact, dirk22 or BE.'
        stop
    end if

contains
    pure function applyS(tauinv,kQ,q)
        implicit none
        real(kind=8), intent(in) :: q(6), kQ
        real(kind=8), intent(in) :: tauinv
        real(kind=8) :: applyS(6)

        applyS(1) = -kQ*q(4)
        applyS(2) = -tauinv*q(5)
        applyS(3) = -tauinv*q(6)
        applyS(4) = kQ*q(1)
        applyS(5) = tauinv*q(2)
        applyS(6) = tauinv*q(3)
    end function applyS

    pure function exact_integrator(tauinv,kappa,local_q,dt,pnonlinear)
        implicit none
        real(kind=8), intent(in) :: kappa, tauinv, local_q(6),dt, pnonlinear
        real(kind=8) :: a, temp(6,1)
        real(kind=8) :: c(2), s(2)
        real(kind=8) :: expdtS(6,6)
        real(kind=8) :: exact_integrator(6)
        
        ! compute exponential of source term (linear since 
        ! mass is conserved)
        
        if (pnonlinear>3.d0) then
            a = kappa*(local_q(1)**2.d0+local_q(4)**2.d0)**((pnonlinear-1.d0)/2.d0)
        else
            a = kappa*(local_q(1)**2.d0+local_q(4)**2.d0)
        end if

        c(1) = cos(a*dt); s(1) = sin(a*dt)
        c(2) = cos(tauinv*dt); s(2) = sin(tauinv*dt)

        expdtS(:,:) = 0.d0

        expdtS(1,1) = c(1);  expdtS(1,4) = -s(1)
        expdtS(2,2) = c(2);  expdtS(2,5) = -s(2)
        expdtS(3,3) = c(2);  expdtS(3,6) = -s(2)
        expdtS(4,1) = s(1);  expdtS(4,4) = c(1)
        expdtS(5,2) = s(2);  expdtS(5,5) = c(2)
        expdtS(6,3) = s(2);  expdtS(6,6) = c(2)
        
        temp(:,1) = local_q(:)
        ! update local conserved quantities
        temp = matmul(expdtS,temp)
        exact_integrator(:) = temp(:,1)  
    end function exact_integrator


    pure function DIRK22(tauinv,kappa,local_q,dt,pnonlinear)
        implicit none
        real(kind=8), intent(in) :: tauinv,kappa,local_q(6),dt,pnonlinear
        real(kind=8) :: kq,tau
        real(kind=8) :: b1,b2,a11,a12,a21,a22
        real(kind=8) :: gammaorder
        real(kind=8) :: q1(6),q2(6) !RK stages
        real(kind=8) :: DIRK22(6)
        tau = 1.d0/tauinv
        gammaorder = 1.d0-1.d0/dsqrt(2.d0)
        if (pnonlinear>3.d0) then
            kq = kappa*(local_q(1)**2 + local_q(4)**2)**((pnonlinear-1.d0)/2.d0)
        else
            kq = kappa*(local_q(1)**2 + local_q(4)**2)
        end if
        kq = kappa*(local_q(1)**2 + local_q(4)**2)

        b1 = 1.d0-gammaorder;   b2 = gammaorder
        a11 = gammaorder;       a12 = 0.d0
        a21 = 1.d0-gammaorder;  a22 = gammaorder

        q1 = auxImRK(local_q, kq, a11,tau,dt) 
        q2 = auxImRK(local_q+dt*a21*applyS(tauinv,kq,q1),kq,a22,tau,dt)
        DIRK22(:) = local_q + dt*(b1*applyS(tauinv,kq,q1) + b2*applyS(tauinv,kq,q2))
    end function DIRK22

    pure function BE(tauinv,kappa,local_q,dt,pnonlinear)
        implicit none
        real(kind=8), intent(in) :: tauinv,kappa,local_q(6), dt,pnonlinear
        real(kind=8) :: kq,tau
        real(kind=8) :: BE(6)

        tau = 1.d0/tauinv
        if (pnonlinear>3.d0) then
            kq = kappa*(local_q(1)**2 + local_q(4)**2)**((pnonlinear-1.d0)/2.d0)
        else
            kq = kappa*(local_q(1)**2 + local_q(4)**2)
        end if
        BE(:) = auxImRK(local_q, kq, 1.d0, tau,dt)
    end function BE

    pure function auxImRK(q,kQ,aii,tau,dt)
        ! Computes (I-dt*aii*S)^{-1}q
        implicit none
        real(kind=8), intent(in) :: q(6), tau
        real(kind=8), intent(in) :: aii, kQ, dt
        real(kind=8) :: g,b,c,y,p
        real(kind=8) :: auxImRK(6)

        g = 1+(aii**2)*(dt**2)*(kQ**2)
        b = (aii**2)*(dt**2)+tau**2
        c = tau**2
        y = aii*dt*tau
        p = aii*dt*kQ
        
        auxImRK(1) = (q(1)      -q(4)*p)/g
        auxImRK(2) = (c*q(2)    -q(5)*y)/b
        auxImRK(3) = (c*q(3)    -q(6)*y)/b
        auxImRK(4) = (p*q(1)    +q(4))/g
        auxImRK(5) = (y*q(2)    +c*q(5))/b
        auxImRK(6) = (y*q(3)    +c*q(6))/b
    end function auxImRK

end subroutine src2