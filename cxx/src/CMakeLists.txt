# cm1k library. Sources are globbed per directory so that parallel work on
# different components does not need to touch this file.
file(GLOB_RECURSE CM1K_SOURCES CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/cm1k/*.cpp)
list(FILTER CM1K_SOURCES EXCLUDE REGEX ".*/cm1k_main\\.cpp$")

add_library(cm1k ${CM1K_SOURCES})
target_include_directories(cm1k PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cm1k ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cm1k PUBLIC ekat::KokkosUtils ekat::Pack ekat::Algorithm)
# glibc's libmvec: gfortran vectorizes some table-building loops of the
# Fortran reference through it (e.g. sfclayinit), so the host-side table
# builders call the same entry points to stay bitwise identical.
find_library(CM1K_LIBMVEC NAMES mvec)
if(CM1K_LIBMVEC AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  target_link_libraries(cm1k PUBLIC ${CM1K_LIBMVEC})
  target_compile_definitions(cm1k PRIVATE CM1K_HAVE_LIBMVEC)
endif()

add_executable(cm1k_exe cm1k/driver/cm1k_main.cpp)
set_target_properties(cm1k_exe PROPERTIES OUTPUT_NAME cm1k)
target_link_libraries(cm1k_exe PRIVATE cm1k)
