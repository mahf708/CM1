#!/usr/bin/env bash
# Build the cm1k Python module and run its pytest suite.
#
#   cxx/scripts/ci_python.sh [BUILD_DIR] [-- extra pytest args]
#
# Environment:
#   CM1K_REF_DIR        reference dumps (cxx/scripts/make_refs.sh); ref tests skip if unset
#   CM1K_PY_MODE        "cmake" (default): CMake build tree + PYTHONPATH (needs numpy,
#                       pybind11, pytest); "pip": pip install ./cxx into a venv in BUILD_DIR
#                       (also needs scikit-build-core, fetched by pip if missing)
#   CM1K_PY_BACKEND     Kokkos backend: openmp (default) or serial
#   CM1K_JOBS           parallel build jobs (default: nproc)
#   PYTHON              python interpreter (default: python3)
#
# On ubuntu-latest: apt install gfortran libopenmpi-dev libnetcdff-dev;
# pip install numpy pybind11 pytest.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
cxx=$(cd "$here/.." && pwd)
root=$(cd "$cxx/.." && pwd)

build=${1:-$root/build/cm1k-python}
shift || true
[ "${1:-}" = "--" ] && shift
mkdir -p "$build"
build=$(cd "$build" && pwd)

py=${PYTHON:-python3}
jobs=${CM1K_JOBS:-$(nproc)}
mode=${CM1K_PY_MODE:-cmake}
backend=${CM1K_PY_BACKEND:-openmp}
export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export OMP_PROC_BIND=${OMP_PROC_BIND:-false} OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}

case "$backend" in
  openmp) kokkos_args=(-DKokkos_ENABLE_OPENMP=ON) ;;
  serial) kokkos_args=(-DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_OPENMP=OFF) ;;
  *) echo "ci_python.sh: unknown CM1K_PY_BACKEND '$backend'" >&2; exit 2 ;;
esac

# EKAT and its Kokkos (the Kokkos submodule URL is ssh; use https).
if [ ! -f "$cxx/externals/ekat/extern/kokkos/CMakeLists.txt" ]; then
  git -C "$root" -c url."https://github.com/".insteadOf=git@github.com: \
    submodule update --init --recursive
fi

echo "== cm1k python: mode=$mode backend=$backend build=$build"
if [ "$mode" = "pip" ]; then
  "$py" -m venv --system-site-packages "$build/venv"
  vpy="$build/venv/bin/python"
  defs=()
  for a in "${kokkos_args[@]}"; do defs+=("-Ccmake.define.${a#-D}"); done
  CMAKE_BUILD_PARALLEL_LEVEL=$jobs "$vpy" -m pip install -v "$cxx" "${defs[@]}" \
    -Cbuild-dir="$build/skbuild" >"$build/pip_install.log" 2>&1 \
    || { tail -60 "$build/pip_install.log"; exit 1; }
  run_py=("$vpy")
  unset PYTHONPATH
else
  cmake -S "$cxx" -B "$build" -DCM1K_ENABLE_PYTHON=ON -DCM1K_ENABLE_TESTS=OFF \
    -DCM1K_ENABLE_FORTRAN_BRIDGE=OFF -DPython_EXECUTABLE="$(command -v "$py")" \
    "${kokkos_args[@]}" >"$build/configure.log" 2>&1 \
    || { tail -60 "$build/configure.log"; exit 1; }
  cmake --build "$build" --target _cm1k -j"$jobs"
  export PYTHONPATH="$build/python${PYTHONPATH:+:$PYTHONPATH}"
  run_py=("$py")
fi

"${run_py[@]}" -c "import cm1k; print('cm1k', cm1k.__file__, 'execution space:', cm1k.execution_space())"
if [ -n "${CM1K_REF_DIR:-}" ]; then
  echo "== reference dumps: $CM1K_REF_DIR"
else
  echo "== CM1K_REF_DIR not set: reference-based tests will be skipped"
fi
# Run from the build dir so the source package (cxx/python/cm1k, no extension)
# does not shadow the built one.
cd "$build"
"${run_py[@]}" -m pytest -p no:cacheprovider -c "$cxx/python/pytest.ini" --rootdir "$cxx/python" \
  "$cxx/python/tests" "$@"
