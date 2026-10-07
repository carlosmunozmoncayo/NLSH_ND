
! =====================================================
subroutine rp1(maxm,meqn,mwaves,maux,mbc,mx,ql,qr,auxl,auxr,wave,s,amdq,apdq)
! =====================================================
! Riemann solver for NLSH in 1d in real form.
 
! waves: 4
! equations: 4
 
! Conserved quantities:
!       1   z0 = Re(q0)
!       2   z1 = Re(q1)
!       3   w0 = Im(q0)
!       4   w1 = Im(q1)
 
! Note that the i'th Riemann problem has left state qr(:,i-1)
!                                    and right state ql(:,i)
 
    implicit none
 
    integer, intent(in) :: maxm, meqn, mwaves, maux, mbc, mx
    real(kind=8), intent(in) :: ql(meqn,1-mbc:maxm+mbc)
    real(kind=8), intent(in) :: qr(meqn,1-mbc:maxm+mbc)
    real(kind=8), intent(in) :: auxl(maux,1-mbc:maxm+mbc)
    real(kind=8), intent(in) :: auxr(maux,1-mbc:maxm+mbc)
 
    real(kind=8), intent(out) :: s(mwaves,1-mbc:maxm+mbc)
    real(kind=8), intent(out) :: wave(meqn,mwaves,1-mbc:maxm+mbc)
    real(kind=8), intent(out) :: amdq(meqn,1-mbc:maxm+mbc)
    real(kind=8), intent(out) :: apdq(meqn,1-mbc:maxm+mbc)
 
    !Locals
    integer :: i, mw
    real(kind=8) :: delta(4), alpha(4), R(4,4), Rinv(4,4)
    real(kind=8) :: sqtau, sqtauinv
 
    !Common block (filled from state.problem_data by PyClaw)
    real(kind=8) :: tau, kappa
    common /cparam/ tau, kappa
 
    sqtau = sqrt(tau)
    sqtauinv = 1.d0/sqtau
 
    R(:,:) = 0.d0
    R(1,1) = -sqtau;    R(1,3) = sqtau
    R(2,2) = sqtauinv;  R(2,4) = -sqtauinv
    R(3,2) = 1.d0;      R(3,4) = 1.d0
    R(4,1) = 1.d0;      R(4,3) = 1.d0
 
    Rinv(:,:) = 0.d0
    Rinv(1,1) = -0.5d0*sqtauinv;    Rinv(1,4) = 0.5d0
    Rinv(2,2) = 0.5d0*sqtau;        Rinv(2,3) = 0.5d0
    Rinv(3,1) = 0.5d0*sqtauinv;     Rinv(3,4) = 0.5d0
    Rinv(4,2) = -0.5d0*sqtau;       Rinv(4,3) = 0.5d0
 
    amdq(:,:)   = 0.d0
    apdq(:,:)   = 0.d0
    wave(:,:,:) = 0.d0
    s(1:2,:) = -sqtauinv
    s(3:4,:) =  sqtauinv
 
    do i = 2-mbc, mx+mbc
        delta = ql(:,i) - qr(:,i-1)
        alpha = matmul(Rinv, delta)
        do mw = 1, mwaves
            wave(:,mw,i) = alpha(mw)*R(:,mw)
        end do
        amdq(:,i) = s(1,i)*wave(:,1,i) + s(2,i)*wave(:,2,i)
        apdq(:,i) = s(3,i)*wave(:,3,i) + s(4,i)*wave(:,4,i)
    end do
 
end subroutine rp1