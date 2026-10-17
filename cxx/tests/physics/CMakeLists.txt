# Physics tests that do not need the Fortran (conservation, option handling).
# Fortran-parity tests for the same code live in ../fortran_bridge.
file(GLOB PHYSICS_TESTS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*_tests.cpp)
cm1k_add_test(physics_tests SOURCES ${PHYSICS_TESTS})
