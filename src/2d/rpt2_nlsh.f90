! =====================================================
subroutine rpt2(ixy,imp,maxm,meqn,mwaves,maux,mbc,mx,ql,qr,aux1,aux2,aux3,asdq,bmasdq,bpasdq)
! =====================================================

! # Riemann solver in the transverse direction for NLSH in 2D
! # Split asdq into down-going flux bmasdq and up-going flux bpasdq.
! # imp=1  means  asdq=amdq,    imp=2 means asdq=apdq
! # Not sure if this is well coded yet. We'll have to try.  
    use cparam_mod, only: tau,kappa
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
    real(kind=8) :: alpha(4,1)
    real(kind=8) :: sqtau, sqtauinv

    !New approach
    real(kind=8) :: Rtilde(4,4), Rtildeinv(4,4)
    integer :: nuz, nuw

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

    if (ixy == 1) then
        !Riemann problem in the x-direction, will grab the vertical eigenstructure
        !The opposite of the normal Riemann solver
        nuz = 3
        nuw = 6
    else
        !Riemann problem in the y-direction, will grab the horizontal eigenstructure
        nuz = 2
        nuw = 5
    endif

    R(:,:) = 0.d0
    R(1,:) = [-sqtau, 0.d0, sqtau, 0.d0]
    R(nuz,:) = [0.d0, sqtauinv, 0.d0, -sqtauinv]
    R(4,:) = [0.d0,1.d0,0.d0, 1.d0]
    R(nuw,:) = [1.d0, 0.d0, 1.d0, 0.d0]

    bmasdq(:,:)  = 0.d0
    bpasdq(:,:)  = 0.d0

    do 20 i = 2-mbc, mx+mbc
        !delta = P_j^T asdq(:,i)
        delta(:,1) = 0.d0
        delta(1,1) = asdq(1,i)
        delta(2,1) = asdq(nuz,i)
        delta(3,1) = asdq(4,i)
        delta(4,1) = asdq(nuw,i)

        alpha = matmul(Rtildeinv, delta)

        bmasdq(:,i) = bmasdq(:,i) - sqtauinv*alpha(1,1)*R(:,1)
        bmasdq(:,i) = bmasdq(:,i) - sqtauinv*alpha(2,1)*R(:,2)

        bpasdq(:,i) = bpasdq(:,i) + sqtauinv*alpha(3,1)*R(:,3)
        bpasdq(:,i) = bpasdq(:,i) + sqtauinv*alpha(4,1)*R(:,4)
    20 end do



end subroutine rpt2
