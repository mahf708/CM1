#!/usr/bin/env bash
# Compile CM1's Fortran sources into object files that per-routine harness
# drivers link against.  Usage: build_objs.sh <outdir> [r4|r8]
#   r4: CM1's default single precision;  r8: -fdefault-real-8 double precision
set -euo pipefail
out=${1:?outdir}; prec=${2:-r4}
here=$(cd "$(dirname "$0")" && pwd)
src=$(cd "$here/../../../src" && pwd)
mkdir -p "$out"
if [[ -f "$out/.done" ]]; then exit 0; fi
rm -rf "$out/src"; cp -r "$src" "$out/src"; rm -f "$out"/src/*.o "$out"/src/*.mod
extra=""
[[ "$prec" == r8 ]] && extra="-fdefault-real-8 -fdefault-double-8"
cd "$out/src"
if [[ "$prec" == r8 ]]; then
  # make CCPP/MMM physics follow the promoted default real kind
  sed -i 's/selected_real_kind(6)/kind(1.0)/' ccpp_kind_types.F module_ra_rrtmg_lw.F module_ra_rrtmg_sw.F
  sed -i 's/\balog(/log(/g' sf_sfclayrev.F
fi
mkdir -p ../run
make -j"$(nproc)" FC=gfortran \
  "OPTS=-ffree-form -ffree-line-length-none -O2 -fallow-argument-mismatch $extra" >"$out/build.log" 2>&1 || true
test -f kessler.o -a -f sound.o -a -f solve2.o || { tail -30 "$out/build.log"; exit 1; }
touch "$out/.done"
