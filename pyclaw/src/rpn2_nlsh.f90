! =====================================================
subroutine rpn2(ixy,maxm,meqn,mwaves,maux,mbc,mx,ql,qr,auxl,auxr,wave,s,amdq,apdq)
! =====================================================
! Riemann solver for NLSH in 2d in real and complex form.

! waves: 4 (really 6, but only 4 have non-zero speeds)
! equations: 6

!$q = z+iw$
!where the scalar q0 approximates the solution to the NLSH equation
!and the 2D vector q1 approximates $\nabla q0$.

! Conserved quantities:
!       1   z0 (first component of z)
!       2   first component of z1
!       3   second component of z1
!       4   w0 (first component of w)
!       5   first component of w1
!       6   second component of w1

! solve Riemann problems along one slice of data.

! On input, ql contains the state vector at the left edge of each cell
!           qr contains the state vector at the right edge of each cell

! This data is along a slice in the x-direction if ixy=1
!                            or the y-direction if ixy=2.
! On output, wave contains the waves,
!            s the speeds,
!            amdq the  left-going flux difference  A^- \Delta q
!            apdq the right-going flux difference  A^+ \Delta q


! Note that the i'th Riemann problem has left state qr(i-1,:)
!                                    and right state ql(i,:)
! From the basic clawpack routines, this routine is called with ql = qr
    
    implicit none

    integer, intent(in) :: ixy, maxm, meqn, mwaves, maux, mbc, mx
    real(kind=8), intent(in) :: ql(meqn,1-mbc:maxm+mbc)
    real(kind=8), intent(in) :: qr(meqn,1-mbc:maxm+mbc)
    real(kind=8), intent(in) :: auxl(maux,1-mbc:maxm+mbc)
    real(kind=8), intent(in) :: auxr(maux,1-mbc:maxm+mbc)

    real(kind=8), intent(out) :: s(mwaves,1-mbc:maxm+mbc)
    real(kind=8), intent(out) :: wave(meqn, mwaves,1-mbc:maxm+mbc)
    real(kind=8), intent(out) :: amdq(meqn,1-mbc:maxm+mbc)
    real(kind=8), intent(out) :: apdq(meqn,1-mbc:maxm+mbc)

    !Locals
    integer :: i, mw, nuz,nuw
    real(kind=8) :: delta(4,1), R(4,4), Rinv(4,4)
    real(kind=8) :: alpha(4,1)
    real(kind=8) :: sqtau, sqtauinv

    !Common bloack
    real (kind=8) :: tau, kappa

    common /cparam/ tau, kappa


    if (ixy == 1) then
        !Riemann problem in the x-direction
        nuz = 2
        nuw = 5
    else
        !Riemann problem in the y-direction
        nuz = 3
        nuw = 6
    endif

    !Common quantities:
    sqtau = sqrt(tau)
    sqtauinv = 1.d0/sqtau

    R(:,:) = 0.d0
    R(1,1) = -sqtau;    R(1,3) = sqtau
    R(2,2) = sqtauinv;  R(2,4) = -sqtauinv
    R(3,2) = 1.d0;      R(3,4) = 1.d0
    R(4,1) = 1.d0;      R(4,3) = 1.d0

    Rinv(:,:) = 0.d0
    Rinv(1,1) = -0.5d0*sqtauinv;    Rinv(1,4) = 0.5d0
    Rinv(2,2) = 0.5d0*sqtau;         Rinv(2,3) = 0.5d0
    Rinv(3,1) = 0.5d0*sqtauinv;     Rinv(3,4) = 0.5d0
    Rinv(4,2) = -0.5*sqtau;      Rinv(4,3) = 0.5d0

    amdq(:,:)  = 0.d0
    apdq(:,:)  = 0.d0
    wave(:,:,:) = 0.d0
    ! s(:,:) = 0.d0
    s(1:2,:) = -sqtauinv
    s(3:4,:) = sqtauinv


    do 20 i = 2-mbc, mx+mbc
        delta(1,1) = ql(1,i) - qr(1,i-1)
        delta(2,1) = ql(nuz,i) - qr(nuz,i-1)
        delta(3,1) = ql(4,i) - qr(4,i-1)
        delta(4,1) = ql(nuw,i) - qr(nuw,i-1)

        alpha = matmul(Rinv, delta)
        
        do mw = 1, mwaves
            wave(1,mw,i)    = alpha(mw,1)*R(1,mw)
            wave(nuz,mw,i)  = alpha(mw,1)*R(2,mw)
            wave(4,mw,i)    = alpha(mw,1)*R(3,mw)
            wave(nuw,mw,i)  = alpha(mw,1)*R(4,mw)
        end do

        amdq(:,i) = amdq(:,i) + s(1,i)*wave(:,1,i)
        amdq(:,i) = amdq(:,i) + s(2,i)*wave(:,2,i)

        apdq(:,i) = apdq(:,i) + s(3,i)*wave(:,3,i)
        apdq(:,i) = apdq(:,i) + s(4,i)*wave(:,4,i)
    20 END DO

    end subroutine rpn2