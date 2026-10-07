! =====================================================
subroutine rpt2(ixy,imp,maxm,meqn,mwaves,maux,mbc,mx,ql,qr,aux1,aux2,aux3,asdq,bmasdq,bpasdq)
! =====================================================

! # Riemann solver in the transverse direction for NLSH in 2D
! # Split asdq into down-going flux bmasdq and up-going flux bpasdq.
! # imp=1  means  asdq=amdq,    imp=2 means asdq=apdq
! # Not sure if this is well coded yet. We'll have to try.  
    implicit none
    !Input
    integer, intent(in)  :: ixy,imp,maxm,meqn,mwaves,maux,mbc,mx
    double precision, dimension(meqn,1-mbc:maxm+mbc), intent(in) :: ql,qr
    double precision, dimension(maux,1-mbc:maxm+mbc), intent(in) :: aux1,aux2,aux3
    double precision, dimension(meqn,1-mbc:maxm+mbc), intent(in) :: asdq
    !Output
    double precision, dimension(meqn,1-mbc:maxm+mbc), intent(out) :: bmasdq,bpasdq


    !Locals
    integer :: i, mw
    real(kind=8) :: delta(4,1), R(6,4)
    real(kind=8) :: beta(4,1)
    real(kind=8) :: sqtau, sqtauinv

    !New approach
    real(kind=8) :: Rtilde(4,4), Rtildeinv(4,4)

    !For mapped grid
    integer :: ixtran, iytran, ilenrat, i1
    double precision :: st, ct
    double precision :: temp(6)

    !Common bloack
    real (kind=8) :: tau, kappa

    common /cparam/ tau, kappa

    if (ixy.eq.1) then
        ixtran = 4
        iytran = 5
        ilenrat = 6
    else
        ixtran = 1
        iytran = 2
        ilenrat = 3
    endif

    !Common quantities:
    sqtau = sqrt(tau)
    sqtauinv = 1.d0/sqtau
    
    Rtilde(:,:) = 0.d0
    Rtilde(1,1) = -sqtau;    Rtilde(1,3) = sqtau
    Rtilde(2,2) = sqtauinv;  Rtilde(2,4) = -sqtauinv
    Rtilde(3,2) = 1.d0;      Rtilde(3,4) = 1.d0
    Rtilde(4,1) = 1.d0;      Rtilde(4,3) = 1.d0

    Rtildeinv(:,:) = 0.d0
    Rtildeinv(1,1) = -0.5d0*sqtauinv;    Rtildeinv(1,4) = 0.5d0
    Rtildeinv(2,2) = 0.5d0*sqtau;         Rtildeinv(2,3) = 0.5d0
    Rtildeinv(3,1) = 0.5d0*sqtauinv;     Rtildeinv(3,4) = 0.5d0
    Rtildeinv(4,2) = -0.5*sqtau;      Rtildeinv(4,3) = 0.5d0


    bmasdq(:,:)  = 0.d0
    bpasdq(:,:)  = 0.d0

    do 20 i = 2-mbc, mx+mbc
        i1 = i + imp - 2

        !Get rotation angle from aux array
        !up-going:
        ct = aux3(ixtran,i1)
        st = aux3(iytran,i1)
        !Rotate forward asdq
        delta(:,1) = 0.d0
        delta(1,1) = asdq(1,i)
        delta(2,1) = ct*asdq(2,i)+st*asdq(3,i)
        delta(3,1) = asdq(4,i)
        delta(4,1) = ct*asdq(5,i)+st*asdq(6,i)
        !Get coefficients for the 1d problem in x direction
        beta = matmul(Rtildeinv, delta)
        !Split fluctuations
        !Rotate  back 3rd eigenvectors
        temp = [Rtilde(1,3), ct*Rtilde(2,3), st*Rtilde(2,3), Rtilde(3,3), ct*Rtilde(4,3), st*Rtilde(4,3)]
        !Add contribution from 3rd wave
        bpasdq(:,i) = bpasdq(:,i) + sqtauinv*beta(3,1)*temp
        !Rotate back 4th eigenvector
        temp = [Rtilde(1,4), ct*Rtilde(2,4), st*Rtilde(2,4), Rtilde(3,4), ct*Rtilde(4,4), st*Rtilde(4,4)]
        !Add contribution from 4th wave
        bpasdq(:,i) = bpasdq(:,i) + sqtauinv*beta(4,1)*temp
        !Scale by length ratio
        bpasdq(:,i) = bpasdq(:,i)*aux3(ilenrat,i1)

        !down-going:
        ct = aux2(ixtran,i1)
        st = aux2(iytran,i1)
        !Rotate forward asdq
        delta(:,1) = 0.d0
        delta(1,1) = asdq(1,i)
        delta(2,1) = ct*asdq(2,i)+st*asdq(3,i)
        delta(3,1) = asdq(4,i)
        delta(4,1) = ct*asdq(5,i)+st*asdq(6,i)
        !Get coefficients for the 1d problem in x direction
        beta = matmul(Rtildeinv, delta)
        !Split fluctuations
        !Rotate back 1st eigenvector
        temp = [Rtilde(1,1), ct*Rtilde(2,1), st*Rtilde(2,1), Rtilde(3,1), ct*Rtilde(4,1), st*Rtilde(4,1)]
        !Add contribution from 1st wave
        bmasdq(:,i) = bmasdq(:,i) - sqtauinv*beta(1,1)*temp
        !Rotate back 2nd eigenvector   
        temp = [Rtilde(1,2), ct*Rtilde(2,2), st*Rtilde(2,2), Rtilde(3,2), ct*Rtilde(4,2), st*Rtilde(4,2)]
        !Add contribution from 2nd wave
        bmasdq(:,i) = bmasdq(:,i) - sqtauinv*beta(2,1)*temp
        !Scale by length ratio
        bmasdq(:,i) = bmasdq(:,i)*aux2(ilenrat,i1)       
    20 end do
end subroutine rpt2
