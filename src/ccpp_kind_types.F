module ccpp_kind_types
#ifdef DP
    integer, parameter :: kind_phys = selected_real_kind(12) ! 8-byte real (USE_DOUBLE)
#else
    integer, parameter :: kind_phys = selected_real_kind(6)  ! 4-byte real
#endif
   contains
end module ccpp_kind_types
