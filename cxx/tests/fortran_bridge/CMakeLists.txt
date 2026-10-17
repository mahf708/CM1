# Component tests that call the original CM1 Fortran (library cm1f) through
# bind(C) shims and compare against the C++ ports. Add a shim file
# (cm1f_bridge_<component>.F90) and a test file (<component>_tests.cpp).
file(GLOB CM1F_BRIDGE_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cm1f_bridge_*.F90)
add_library(cm1f_bridge STATIC ${CM1F_BRIDGE_SOURCES})
target_link_libraries(cm1f_bridge PUBLIC cm1f)
target_compile_options(cm1f_bridge PRIVATE -fdefault-real-8 -fdefault-double-8 -ffp-contract=off -ffree-line-length-none)
# The surface-layer shims copy loops of solve1.F/sfcphys.F that call alog;
# gfortran would vectorize those through glibc's libmvec (_ZGVbN2v_log, not
# correctly rounded), which the CM1 sources themselves do not do.
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/cm1f_bridge_sfc.F90 PROPERTIES COMPILE_OPTIONS -fno-tree-vectorize)

file(GLOB CM1F_BRIDGE_TESTS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*_tests.cpp)
cm1k_add_test(fortran_bridge_tests SOURCES ${CM1F_BRIDGE_TESTS} LIBS cm1f_bridge gfortran)
