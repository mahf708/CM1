#!/usr/bin/env bash
# Build the Fortran CM1 in double precision and write reference dumps for
# the cases in cxx/cases.
#
#   cxx/scripts/make_refs.sh OUT_DIR [case ...]
#
# Each case directory holds a namelist.input (and optionally an
# `every` file with the dump interval in steps; default 1). Dumps go to
# OUT_DIR/<case>/step_NNNNNN (see src/cm1dump.F).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
out=${1:?usage: make_refs.sh OUT_DIR [case ...]}
shift || true
mkdir -p "$out"
out=$(cd "$out" && pwd)

netcdf_opts=()
if command -v nf-config >/dev/null 2>&1; then
  netcdf_opts=(USE_NETCDF=true NETCDFBASE="$(nf-config --prefix)")
fi
make -C "$root/src" -j"$(nproc)" FC=gfortran USE_DOUBLE=true "${netcdf_opts[@]}" >"$out/build_fortran.log" 2>&1 \
  || { tail -50 "$out/build_fortran.log"; exit 1; }

cases=("$@")
if [ ${#cases[@]} -eq 0 ]; then
  for d in "$here"/../cases/*/; do [ -f "$d/namelist.input" ] && cases+=("$(basename "$d")"); done
fi

for c in "${cases[@]}"; do
  src="$here/../cases/$c"
  run="$out/$c"
  rm -rf "$run" && mkdir -p "$run"
  cp "$src/namelist.input" "$run/"
  # extra input files some cases need (e.g. input_sounding)
  for f in "$src"/*; do [ "$(basename "$f")" != namelist.input ] && cp -r "$f" "$run/"; done
  every=1
  [ -f "$src/every" ] && every=$(cat "$src/every")
  echo "== $c (dump every $every steps)"
  (cd "$run" && CM1_DUMP_DIR="$run" CM1_DUMP_EVERY="$every" OMP_NUM_THREADS=1 "$root/run/cm1.exe" >cm1.log 2>&1) \
    || { tail -30 "$run/cm1.log"; exit 1; }
  grep -q "Program terminated normally" "$run/cm1.log" || { tail -30 "$run/cm1.log"; exit 1; }
  rm -f "$run"/cm1out_* 
  ls -d "$run"/step_* | wc -l | xargs echo "   steps dumped:"
done
