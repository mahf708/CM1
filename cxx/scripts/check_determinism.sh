#!/usr/bin/env bash
# Check that the Fortran reference dumps are reproducible.
#
#   cxx/scripts/check_determinism.sh WORK_DIR [case ...]
#
# 1. Serial build (as in make_refs.sh), run twice: dumps must be bitwise
#    identical (fails otherwise).
# 2. OpenMP build, run with OMP_NUM_THREADS=1 and twice with 2: the two
#    2-thread runs must be bitwise identical (fails otherwise); 1 vs 2 threads
#    and serial-build vs OpenMP-build are reported, and fail only with
#    STRICT_THREADS=1.
#
# Each run's dumps are deleted after use to bound disk usage. Uses
# make_refs.sh (so CM1_BUILD_DIR must not be set) and compare_dumps.py.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
[ $# -ge 1 ] || { echo "usage: check_determinism.sh WORK_DIR [case ...]" >&2; exit 1; }
work=$1
shift
[ -z "${CM1_BUILD_DIR:-}" ] || { echo "check_determinism.sh: unset CM1_BUILD_DIR" >&2; exit 1; }
mkdir -p "$work"
work=$(cd "$work" && pwd)
cmp_py=(python3 "$here/compare_dumps.py")
fail=0
summary=()

run() {  # run NAME USE_OMP THREADS
  echo "######## run $1 (openmp=$2, threads=$3)"
  CM1_USE_OPENMP=$2 CM1_THREADS=$3 "$here/make_refs.sh" "$work/$1" "${CASES[@]}"
}
compare() {  # compare LABEL A B REQUIRED
  echo "######## compare: $1"
  local rc=0
  "${cmp_py[@]}" "$work/$2" "$work/$3" || rc=$?
  if [ $rc -eq 0 ]; then
    summary+=("BITWISE    $1")
  else
    summary+=("DIFFERENT  $1")
    if [ "$4" = required ]; then fail=1; fi
  fi
}

CASES=("$@")

run serial_a false 1
run serial_b false 1
compare "serial build, run twice" serial_a serial_b required
rm -rf "$work/serial_b"

run omp1 true 1
compare "serial build vs OpenMP build (1 thread)" serial_a omp1 \
  "$([ "${STRICT_THREADS:-0}" = 1 ] && echo required || echo info)"
rm -rf "$work/serial_a"

run omp2_a true 2
run omp2_b true 2
compare "OpenMP build, 2 threads, run twice" omp2_a omp2_b required
rm -rf "$work/omp2_b"
compare "OpenMP build, 1 vs 2 threads" omp1 omp2_a \
  "$([ "${STRICT_THREADS:-0}" = 1 ] && echo required || echo info)"
rm -rf "$work/omp1" "$work/omp2_a"

echo
echo "######## determinism summary"
printf '%s\n' "${summary[@]}"
exit $fail
