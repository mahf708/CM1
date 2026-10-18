cmake_minimum_required(VERSION 3.18)

project(CM1K VERSION 0.1.0 LANGUAGES C CXX)

#------------------------------------------------------------------------------
# cm1k: a C++/Kokkos port of CM1 (in the style of E3SM's EAMxx, built on EKAT)
#------------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(CM1K_ENABLE_TESTS          "Build cm1k tests" ON)
option(CM1K_ENABLE_FORTRAN_BRIDGE "Build the CM1 Fortran sources as a library for component tests" ON)
option(CM1K_ENABLE_PYTHON         "Build the pybind11 Python module" OFF)
set(CM1K_REF_DIR "" CACHE PATH "Directory with Fortran reference dumps (see cases/README.md)")

# Value-safe floating point: we compare against the Fortran to round-off,
# so do not let the compiler contract a*b+c into FMAs or reassociate.
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)
endif()

# Never pull in the deprecated MPI C++ bindings.
add_compile_definitions(OMPI_SKIP_MPICXX=1 MPICH_SKIP_MPICXX=1)

#------------------------------------------------------------------------------
# EKAT (and Kokkos, through EKAT's submodule)
#------------------------------------------------------------------------------
if (NOT DEFINED Kokkos_ENABLE_SERIAL)
  set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "")
endif()
set(EKAT_ENABLE_FORTRAN OFF CACHE BOOL "")
set(EKAT_ENABLE_TESTS   OFF CACHE BOOL "")
set(EKAT_ENABLE_KOKKOS  ON  CACHE BOOL "")
set(EKAT_ENABLE_PACK    ON  CACHE BOOL "")
set(EKAT_ENABLE_ALGORITHM ON CACHE BOOL "")
set(EKAT_DISABLE_TPL_WARNINGS ON CACHE BOOL "")
add_subdirectory(externals/ekat ${CMAKE_BINARY_DIR}/externals/ekat)
list(APPEND CMAKE_MODULE_PATH ${EKAT_CMAKE_PATH})

#------------------------------------------------------------------------------
# Fortran CM1 as a library (component-level reference tests)
#------------------------------------------------------------------------------
if (CM1K_ENABLE_FORTRAN_BRIDGE)
  enable_language(Fortran)
  include(cmake/Cm1Fortran.cmake)
endif()

add_subdirectory(src)

if (CM1K_ENABLE_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if (CM1K_ENABLE_PYTHON)
  add_subdirectory(python)
endif()
