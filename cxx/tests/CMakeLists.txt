include(EkatCreateUnitTest)

# cm1k_add_test(<name> SOURCES ... [LIBS ...] [THREADS ...] [MPI_RANKS ...])
#   Catch2 unit test linked against cm1k. Tests that need Fortran reference
#   dumps use $CM1K_REF_DIR, defaulting to the CMake cache variable of the
#   same name, and are skipped (with a warning) when neither is set.
#   MPI_RANKS (EKAT's convention: N, or START END [INCREMENT]) runs the test
#   under mpiexec with those rank counts (tests <name>_np<N>).
function(cm1k_add_test name)
  cmake_parse_arguments(T "" "" "SOURCES;LIBS;THREADS;LABELS;MPI_RANKS" ${ARGN})
  set(threads 1)
  if (T_THREADS)
    set(threads ${T_THREADS})
  endif()
  set(ranks 1)
  if (T_MPI_RANKS)
    set(ranks ${T_MPI_RANKS})
  endif()
  EkatCreateUnitTest(${name}
    SOURCES ${T_SOURCES}
    LIBS cm1k cm1k_testing ${T_LIBS}
    THREADS ${threads}
    MPI_RANKS ${ranks}
    LABELS cm1k ${T_LABELS}
    COMPILER_CXX_DEFS CM1K_DEFAULT_REF_DIR="${CM1K_REF_DIR}" CM1K_DEFAULT_CASES_DIR="${CMAKE_SOURCE_DIR}/cases")
endfunction()

add_library(cm1k_testing INTERFACE)
target_include_directories(cm1k_testing INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/share)

# Every subdirectory with a CMakeLists.txt is a test group.
file(GLOB children RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
list(SORT children)
foreach(child ${children})
  if (IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${child} AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${child}/CMakeLists.txt)
    if (child STREQUAL "fortran_bridge" AND NOT CM1K_ENABLE_FORTRAN_BRIDGE)
      continue()
    endif()
    if (child STREQUAL "mpi" AND NOT CM1K_ENABLE_MPI)
      continue()
    endif()
    add_subdirectory(${child})
  endif()
endforeach()
