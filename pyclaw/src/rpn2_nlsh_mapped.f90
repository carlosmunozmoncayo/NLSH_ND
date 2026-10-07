! =====================================================
subroutine rpn2(ixy,maxm,meqn,mwaves,maux,mbc,mx,ql,qr,auxl,auxr,wave,s,amdq,apdq)
! =====================================================
! Riemann solver for NLSH in 2d in real and complex form.
! Modified to handle mapped grids and inclusions with Neumann BCs.

! waves: 4 (really 6, but only 4 have non-zero speeds)
! equations: 6

!$q = z+iw$
!where the scalar q0 approximates the solution to the NLSH equation
!and the 2D complex vector q1 approximates $\nabla q0$.

! Conserved quantities:
!       1   z0 (first component of z)
!       2   first component of z1
!       3   second component of z1
!       4   w0 (first component of w)
!       5   first component of w1
!       6   second component of w1

! Auxiliary quantities:
!       1 a_x
!       2 a_y   where (a_x,a_y) is unit normal to left face
!       3 length_ratio_left ratio of length of left face to dyc
!       4 b_x
!       5 b_y   where (b_x,b_y) is unit normal to bottom face
!       6 length_ratio_bottom   ratio of length of bottom face to dxc
!       7 cell_area   ratio of cell area to dxc*dyc
!         (approximately Jacobian of mapping function)
!       inclusion_idx (if any) index for inclusion (0 if none)

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
    integer :: i, mw
    real(kind=8) :: delta(4,1), Rtilde(4,4), Rtildeinv(4,4)
    real(kind=8) :: alpha(4,1)
    real(kind=8) :: sqtau, sqtauinv

    double precision :: inside_inclusionL, inside_inclusionR

    !For mapped grid
    integer :: inx, iny, ilenrat
    double precision :: st, ct ! sin and cos of angle between grid lines and x-axis
    double precision :: waves4(4,4)

    !Common bloack
    double precision :: tau, kappa, pnonlinear
    integer :: inclusion_idx
    character(len=25) :: srcintegrator, inclusion_type

    common /cparam/ tau, kappa, pnonlinear, srcintegrator, inclusion_idx, inclusion_type

    if (ixy == 1) then
        !Riemann problem in the x-direction
        inx = 1 
        iny = 2 
        ilenrat = 3
    else
        !Riemann problem in the y-direction
        inx = 4 
        iny = 5 
        ilenrat = 6
    endif

    ! Determine rotation matrix:
    !               [ cos(\theta) sin(\theta) ]
    !               [-sin(\theta) cos(\theta) ]
    ! Note that this reduces to the identity on a standard cartesian grid.

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
    Rtildeinv(2,2) = 0.5d0*sqtau;        Rtildeinv(2,3) = 0.5d0
    Rtildeinv(3,1) = 0.5d0*sqtauinv;     Rtildeinv(3,4) = 0.5d0
    Rtildeinv(4,2) = -0.5*sqtau;         Rtildeinv(4,3) = 0.5d0

    amdq(:,:)  = 0.d0
    apdq(:,:)  = 0.d0
    wave(:,:,:) = 0.d0

    !Scale wave speeds by the length ratio
    s(1,:) = -sqtauinv*auxl(ilenrat,:)
    s(2,:) = -sqtauinv*auxl(ilenrat,:)
    s(3,:) = sqtauinv*auxl(ilenrat,:)
    s(4,:) = sqtauinv*auxl(ilenrat,:)


    if ((trim(inclusion_type) == "neumann") .and. (inclusion_idx .gt. 0) ) then
        !Impose Neumann boundary conditions on the inclusion boundary
        !For Dirichlet, set the target value in b4step2.f90, and proceed as usual here.
        do i = 2-mbc, mx+mbc
            !Get normal vector to the interface from aux array
            ct  = auxl(inx,i)
            st  = auxl(iny,i)

            inside_inclusionR = auxl(inclusion_idx,i)
            inside_inclusionL = auxr(inclusion_idx,i-1)

            !Initialize delta
            delta(:,1) = 0.d0

            if ( (inside_inclusionL .gt. 0.5d0) .and. (inside_inclusionR .gt. 0.5d0) ) then
                !Both cell centers inside inclusion: Do nothing at this interface
                cycle
            else if ( (inside_inclusionL .lt. 0.5d0) .and. (inside_inclusionR .lt. 0.5d0) ) then
                !Both cell centers outside inclusion: Proceed as usual
                !Rotate QR-QL difference normal to the interface.
                !We only care about 4 components of the difference,
                !since we are going to solve the Riemann problem in the x-direction,
                !where the coefficient matrix has rank 4.
                delta(1,1) = ql(1,i) - qr(1,i-1)
                delta(2,1) = ct*(ql(2,i) - qr(2,i-1)) + st*(ql(3,i) - qr(3,i-1))
                delta(3,1) = ql(4,i) - qr(4,i-1)
                delta(4,1) = ct*(ql(5,i) - qr(5,i-1)) + st*(ql(6,i) - qr(6,i-1))
            else if ( (inside_inclusionL .gt. 0.5d0) .and. (inside_inclusionR .lt. 0.5d0) ) then
                !Left cell center inside inlcusion, right cell center outside inclusion
                !constant extrapolation and reflect normal derivatives
                delta(1,1) = 0.d0 !Constant extrapolation of z0
                delta(2,1) = 2.d0*(ct*ql(2,i) + st*ql(3,i)) !Reflect normal rotation of z1
                delta(3,1) = 0.d0 !Constant extrapolation of w0
                delta(4,1) = 2.d0*(ct*ql(5,i) + st*ql(6,i)) !Reflect normal rotation of w1
            else 
                !Left cell center outside inlcusion, right cell center inside inclusion
                !constant extrapolation and reflect normal derivatives
                delta(1,1) = 0.d0 !Constant extrapolation of z0
                delta(2,1) = -2.d0*(ct*qr(2,i-1) + st*qr(3,i-1)) !Reflect normal rotation of z1
                delta(3,1) = 0.d0 !Constant extrapolation of w0
                delta(4,1) = -2.d0*(ct*qr(5,i-1) + st*qr(6,i-1)) !Reflect normal rotation of w1
            end if

            alpha = matmul(Rtildeinv, delta)

            do mw = 1, mwaves
                !For mapped grid, the waves must be rotated back to the original coordinates.
                waves4(:,mw) = alpha(mw,1)*Rtilde(:,mw)
            end do
            
            do mw = 1, mwaves
                wave(1,mw,i) = waves4(1,mw)
                wave(2,mw,i) = ct*waves4(2,mw)
                wave(3,mw,i) = st*waves4(2,mw)
                wave(4,mw,i) = waves4(3,mw)
                wave(5,mw,i) = ct*waves4(4,mw)
                wave(6,mw,i) = st*waves4(4,mw)
            end do

            amdq(:,i) = amdq(:,i) + s(1,i)*wave(:,1,i)
            amdq(:,i) = amdq(:,i) + s(2,i)*wave(:,2,i)

            apdq(:,i) = apdq(:,i) + s(3,i)*wave(:,3,i)
            apdq(:,i) = apdq(:,i) + s(4,i)*wave(:,4,i)
        end do
    else
        do i = 2-mbc, mx+mbc
            !Get normal vector to the interface from aux array
            ct  = auxl(inx,i)
            st  = auxl(iny,i)

            !Rotate QR-QL difference normal to the interface.
            !We only care about 4 components of the difference,
            !since we are going to solve the Riemann problem in the x-direction,
            !where the coefficient matrix has rank 4.

            delta(1,1) = ql(1,i) - qr(1,i-1)
            delta(2,1) = ct*(ql(2,i) - qr(2,i-1)) + st*(ql(3,i) - qr(3,i-1))
            delta(3,1) = ql(4,i) - qr(4,i-1)
            delta(4,1) = ct*(ql(5,i) - qr(5,i-1)) + st*(ql(6,i) - qr(6,i-1))

            alpha = matmul(Rtildeinv, delta)

            do mw = 1, mwaves
                !For mapped grid, the waves must be rotated back to the original coordinates.
                waves4(:,mw) = alpha(mw,1)*Rtilde(:,mw)
            end do
            
            do mw = 1, mwaves
                wave(1,mw,i) = waves4(1,mw)
                wave(2,mw,i) = ct*waves4(2,mw)
                wave(3,mw,i) = st*waves4(2,mw)
                wave(4,mw,i) = waves4(3,mw)
                wave(5,mw,i) = ct*waves4(4,mw)
                wave(6,mw,i) = st*waves4(4,mw)
            end do

            amdq(:,i) = amdq(:,i) + s(1,i)*wave(:,1,i)
            amdq(:,i) = amdq(:,i) + s(2,i)*wave(:,2,i)

            apdq(:,i) = apdq(:,i) + s(3,i)*wave(:,3,i)
            apdq(:,i) = apdq(:,i) + s(4,i)*wave(:,4,i)
        END DO
    end if

    end subroutine rpn2