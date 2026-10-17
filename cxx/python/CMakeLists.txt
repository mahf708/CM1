# pybind11 module `cm1k` (in the spirit of E3SM's pyeamxx).
#
# The build tree is directly importable:
#   cmake -S cxx -B build -DCM1K_ENABLE_PYTHON=ON -DCM1K_ENABLE_TESTS=OFF
#   cmake --build build --target _cm1k -j
#   PYTHONPATH=build/python python -c "import cm1k; print(cm1k.execution_space())"
# `pip install ./cxx` (scikit-build-core, see cxx/pyproject.toml) installs the
# `python` component only.

include(GNUInstallDirs)
find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)

# pip-installed pybind11 ships its CMake config inside the package.
if (NOT pybind11_DIR)
  execute_process(COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
                  OUTPUT_VARIABLE _pybind11_dir OUTPUT_STRIP_TRAILING_WHITESPACE
                  RESULT_VARIABLE _pybind11_rc ERROR_QUIET)
  if (_pybind11_rc EQUAL 0)
    set(pybind11_DIR ${_pybind11_dir} CACHE PATH "pybind11 CMake config directory")
  endif()
endif()
find_package(pybind11 CONFIG REQUIRED)

set(CM1K_PY_PKG_DIR ${CMAKE_CURRENT_BINARY_DIR}/cm1k)

pybind11_add_module(_cm1k MODULE src/cm1k_pybind.cpp)
target_link_libraries(_cm1k PRIVATE cm1k)
set_target_properties(_cm1k PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CM1K_PY_PKG_DIR})

# Mirror the pure-Python sources next to the extension so that the build
# directory works with PYTHONPATH=<build>/python.
file(GLOB CM1K_PY_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cm1k/*.py)
set(_copied)
foreach(src ${CM1K_PY_SOURCES})
  get_filename_component(fname ${src} NAME)
  add_custom_command(OUTPUT ${CM1K_PY_PKG_DIR}/${fname}
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different ${src} ${CM1K_PY_PKG_DIR}/${fname}
                     DEPENDS ${src})
  list(APPEND _copied ${CM1K_PY_PKG_DIR}/${fname})
endforeach()
add_custom_target(cm1k_py_sources ALL DEPENDS ${_copied})
add_dependencies(_cm1k cm1k_py_sources)

# Installation (wheel layout: <platlib>/cm1k/{__init__.py,...,_cm1k.so}).
if (DEFINED SKBUILD_PLATLIB_DIR)
  set(CM1K_PY_INSTALL_DIR cm1k)
else()
  set(CM1K_PY_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR}/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages/cm1k
      CACHE STRING "Install directory for the cm1k Python package (relative to CMAKE_INSTALL_PREFIX)")
endif()
install(TARGETS _cm1k LIBRARY DESTINATION ${CM1K_PY_INSTALL_DIR} COMPONENT python)
install(FILES ${CM1K_PY_SOURCES} DESTINATION ${CM1K_PY_INSTALL_DIR} COMPONENT python)
