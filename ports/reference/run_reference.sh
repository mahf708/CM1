#!/usr/bin/env bash
# Build Fortran CM1 with netCDF output and run one reference case.
# Usage: run_reference.sh <case-dir> <outdir> [r4|r8] [extra namelist overrides as key=value ...]
#   r4 = CM1's default single precision, r8 = -fdefault-real-8 double precision.
# Requires gfortran and netCDF-Fortran (nf-config on PATH).
set -euo pipefail
case_dir=$(cd "${1:?case dir}" && pwd); out=${2:?outdir}; prec=${3:-r4}; shift 3 || true
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
mkdir -p "$out"; out=$(cd "$out" && pwd)
build=${CM1_BUILD_DIR:-$out/build_$prec}
if [[ ! -x "$build/cm1.exe" ]]; then
  rm -rf "$build"; mkdir -p "$build/run"; cp -r "$root/src" "$build/src"
  cd "$build/src"
  extra=""
  if [[ "$prec" == r8 ]]; then
    extra="-fdefault-real-8 -fdefault-double-8"
    sed -i 's/selected_real_kind(6)/kind(1.0)/' ccpp_kind_types.F module_ra_rrtmg_lw.F module_ra_rrtmg_sw.F
    sed -i 's/\balog(/log(/g' sf_sfclayrev.F
  fi
  fflags=$(nf-config --fflags); flibs=$(nf-config --flibs)
  # CM1's Makefile puts -lnetcdf before the objects, which fails with --as-needed
  # linkers, so let make compile and then link explicitly.
  make -j"${CM1X_MAKE_JOBS:-$(nproc)}" FC=gfortran USE_NETCDF=true NETCDFBASE="$(nf-config --prefix)" \
    "OPTS=-ffree-form -ffree-line-length-none -O2 -fallow-argument-mismatch $extra $fflags" \
    >"$build/build.log" 2>&1 || true
  objs=$(ls *.o)
  gfortran -O2 $objs $flibs -o "$build/cm1.exe" >>"$build/build.log" 2>&1 || { tail -40 "$build/build.log"; exit 1; }
fi
cd "$out"
cp "$case_dir/namelist.input" namelist.input
for kv in "$@"; do
  k=${kv%%=*}; v=${kv#*=}
  sed -i -E "s/^(\s*$k\s*=\s*)[^,]*,/\1$v,/" namelist.input
done
cp "$root/run/LANDUSE.TBL" . 2>/dev/null || true
# RRTMG k-distribution tables (radopt=2)
cp "$root/run/RRTMG_LW_DATA" "$root/run/RRTMG_SW_DATA" . 2>/dev/null || true
"$build/cm1.exe" > cm1.log 2>&1
tail -1 cm1.log
