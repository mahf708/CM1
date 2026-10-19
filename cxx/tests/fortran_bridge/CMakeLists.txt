# Component tests that call the original CM1 Fortran (library cm1f) through
# bind(C) shims and compare against the C++ ports. Add a shim file
# (cm1f_bridge_<component>.F90) and a test file (<component>_tests.cpp).
file(GLOB CM1F_BRIDGE_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cm1f_bridge_*.F90)
# cm1f_bridge_turbpriv.F90 defines cm1f_turbcopy_module through a #define,
# which CMake's Fortran dependency scan cannot see; build it as its own
# library first (with cm1f_bridge_turb.F90, whose module it uses) so the
# shims that use the module never race it.
set(CM1F_TURBCOPY ${CMAKE_CURRENT_SOURCE_DIR}/cm1f_bridge_turb.F90 ${CMAKE_CURRENT_SOURCE_DIR}/cm1f_bridge_turbpriv.F90)
list(REMOVE_ITEM CM1F_BRIDGE_SOURCES ${CM1F_TURBCOPY})
set(CM1F_BRIDGE_FFLAGS -fdefault-real-8 -fdefault-double-8 -ffp-contract=off -ffree-line-length-none)
add_library(cm1f_bridge_turbcopy STATIC ${CM1F_TURBCOPY})
target_link_libraries(cm1f_bridge_turbcopy PUBLIC cm1f)
target_compile_options(cm1f_bridge_turbcopy PRIVATE ${CM1F_BRIDGE_FFLAGS})
# cm1f_bridge_morrisonpriv.F90 likewise compiles src/morrison.F again as
# cm1f_graupelcopy_module (all public) and holds every Morrison shim; it is
# built with cm1f's own flags so the copy computes exactly what the real
# module does.
set(CM1F_GRAUPELCOPY ${CMAKE_CURRENT_SOURCE_DIR}/cm1f_bridge_morrisonpriv.F90)
list(REMOVE_ITEM CM1F_BRIDGE_SOURCES ${CM1F_GRAUPELCOPY})
add_library(cm1f_bridge_graupelcopy STATIC ${CM1F_GRAUPELCOPY})
target_link_libraries(cm1f_bridge_graupelcopy PUBLIC cm1f)
target_compile_options(cm1f_bridge_graupelcopy PRIVATE ${CM1F_BRIDGE_FFLAGS} -O2 -finline-functions -fallow-argument-mismatch -w)
add_library(cm1f_bridge STATIC ${CM1F_BRIDGE_SOURCES})
target_link_libraries(cm1f_bridge PUBLIC cm1f_bridge_turbcopy cm1f_bridge_graupelcopy cm1f)
target_compile_options(cm1f_bridge PRIVATE ${CM1F_BRIDGE_FFLAGS})
# The surface-layer shims copy loops of solve1.F/sfcphys.F that call alog;
# gfortran would vectorize those through glibc's libmvec (_ZGVbN2v_log, not
# correctly rounded), which the CM1 sources themselves do not do.
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/cm1f_bridge_sfc.F90 PROPERTIES COMPILE_OPTIONS -fno-tree-vectorize)
# The MYNN shims only pack and copy, but keep them scalar too.
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/cm1f_bridge_mynn_sfc.F90 ${CMAKE_CURRENT_SOURCE_DIR}/cm1f_bridge_mynn_pbl.F90 PROPERTIES COMPILE_OPTIONS -fno-tree-vectorize)
# The MYJ shim only packs and calls, keep it scalar too.
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/cm1f_bridge_myj.F90 PROPERTIES COMPILE_OPTIONS -fno-tree-vectorize)

file(GLOB CM1F_BRIDGE_TESTS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*_tests.cpp)
cm1k_add_test(fortran_bridge_tests SOURCES ${CM1F_BRIDGE_TESTS} LIBS cm1f_bridge gfortran)
