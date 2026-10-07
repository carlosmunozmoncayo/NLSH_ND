module cparam_mod
    !Module to hold common parameters that will be written in setprob
    !The idea is to drop common blocks
    implicit none
    double precision :: tau, kappa, pnonlinear
    integer :: inclusion_idx
    character(len=25) :: srcintegrator, inclusion_type

end module cparam_mod