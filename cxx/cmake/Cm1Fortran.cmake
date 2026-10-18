# Builds the CM1 Fortran sources in ../src (everything except the main
# program cm1.F) into the static library `cm1f`, in double precision and
# without MPI/netCDF, mirroring `make FC=gfortran USE_DOUBLE=true`.
#
# Component tests link `cm1f` plus small bind(C) shims (tests/fortran_bridge)
# and compare individual routines against their C++ ports.

set(CM1_FORTRAN_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../src)
file(GLOB CM1F_SOURCES ${CM1_FORTRAN_SRC_DIR}/*.F)
list(REMOVE_ITEM CM1F_SOURCES ${CM1_FORTRAN_SRC_DIR}/cm1.F)

add_library(cm1f STATIC ${CM1F_SOURCES})
set_target_properties(cm1f PROPERTIES
  Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/cm1f_mod
  Fortran_PREPROCESS ON
  Fortran_FORMAT FREE)
target_compile_definitions(cm1f PRIVATE DP)
if (CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
  target_compile_options(cm1f PRIVATE
    -ffree-line-length-none -fallow-argument-mismatch
    -fdefault-real-8 -fdefault-double-8 -ffp-contract=off -O2 -finline-functions
    -w)
endif()
target_include_directories(cm1f PUBLIC ${CMAKE_BINARY_DIR}/cm1f_mod)
