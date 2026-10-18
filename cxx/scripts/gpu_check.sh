#!/usr/bin/env bash
# Run a GPU (or any) build of cm1k against the Fortran reference dumps.
#
#   cxx/scripts/gpu_check.sh BUILD_DIR REF_DIR OUT_DIR case [case ...]
#
# BUILD_DIR is a cm1k CMake build tree configured with CM1K_ENABLE_PYTHON=ON
# (and, for the ctest part, CM1K_ENABLE_TESTS=ON with CM1K_REF_DIR=REF_DIR);
# REF_DIR holds cxx/scripts/make_refs.sh output for the cases.  Writes
# OUT_DIR/report.md and, per case, OUT_DIR/<case>.{log,json}.
#
# Gate: for every case, ports/crosscheck/crosscheck.py runs cm1k (native
# init, every step) against the dumps with the round-off tolerance schedule
# (a device build is not bitwise: its math library and FMA contraction
# differ from glibc/gfortran).  ctest also runs, but only for information:
# its step tests ask for bitwise agreement, which holds on the CPU only.
#
# Environment: PYTHON (default python3), CM1K_GPU_STEPS (cap the steps per
# case; default: the last dumped step).
set -uo pipefail

[ $# -ge 4 ] || { echo "usage: gpu_check.sh BUILD_DIR REF_DIR OUT_DIR case [case ...]" >&2; exit 2; }
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
build=$(cd "$1" && pwd)
refs=$(cd "$2" && pwd)
mkdir -p "$3"
out=$(cd "$3" && pwd)
shift 3
py=${PYTHON:-python3}
export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export PYTHONPATH="$build/python${PYTHONPATH:+:$PYTHONPATH}"

report="$out/report.md"
: >"$report"
{
  echo "### cm1k on $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo 'no NVIDIA GPU')"
  echo
  "$py" -c "import cm1k; print('Kokkos execution space:', cm1k.execution_space(), '| memory space:', cm1k.memory_space(), '| host-accessible fields:', cm1k.host_accessible())"
  echo
} | tee -a "$report"

status=0
for c in "$@"; do
  last=$(ls -d "$refs/$c"/step_* 2>/dev/null | sed 's/.*step_0*//' | sort -n | tail -1)
  if [ -z "$last" ]; then
    echo "gpu_check.sh: no dumps for $c in $refs" | tee -a "$report"
    status=1
    continue
  fi
  steps=${CM1K_GPU_STEPS:-$last}
  [ "$steps" -gt "$last" ] && steps=$last
  echo "== $c: $steps steps"
  "$py" "$root/ports/crosscheck/crosscheck.py" --case "$c" --steps "$steps" \
    --impl cm1k --impl fortran --ref-dir "$refs/$c" \
    --md "$report" --json "$out/$c.json" >"$out/$c.log" 2>&1
  rc=$?
  tail -3 "$out/$c.log"
  [ $rc -ne 0 ] && status=1
done

if [ -f "$build/CTestTestfile.cmake" ]; then
  echo "== ctest (information only: step tests ask for bitwise agreement)"
  (cd "$build" && ctest --output-on-failure --timeout 1800 >"$out/ctest.log" 2>&1)
  {
    echo "#### ctest (information only)"
    echo
    echo '```'
    grep -E "Test +#|tests passed|tests failed" "$out/ctest.log"
    echo '```'
  } | tee -a "$report"
fi
exit $status
