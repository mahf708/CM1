#!/usr/bin/env bash
# Build the Fortran CM1 in double precision and write reference dumps for
# the cases in cxx/cases.  See cxx/cases/README.md for the output layout.
#
#   cxx/scripts/make_refs.sh OUT_DIR [case ...]
#
# Each case directory holds a namelist.input, optionally an `every` file with
# the dump interval in steps (default 1), and any extra input files the case
# needs (copied next to the namelist).  Dumps go to OUT_DIR/<case>/step_NNNNNN
# (see src/cm1dump.F); OUT_DIR/MANIFEST.txt records how they were made.
#
# The Fortran is built out of tree (the source tree is never written to), in
# a build directory chosen by the configuration, and rebuilt incrementally.
#
# Environment:
#   CM1_BUILD_DIR    build directory     (default: <repo>/build/cm1-fortran-<config>)
#   CM1_USE_OPENMP   true|false          (default: false)
#   CM1_THREADS      OMP_NUM_THREADS for the runs (default: 1)
#   CM1_DEBUG        true|false          (default: false; -g -fbacktrace)
#   JOBS             parallel make jobs  (default: nproc)
#   FC               Fortran compiler    (default: gfortran)
set -euo pipefail

die() { echo "make_refs.sh: error: $*" >&2; exit 1; }

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
cases_dir="$root/cxx/cases"
[ $# -ge 1 ] || die "usage: make_refs.sh OUT_DIR [case ...]"
out=$1
shift
mkdir -p "$out"
out=$(cd "$out" && pwd)

fc=${FC:-gfortran}
use_omp=${CM1_USE_OPENMP:-false}
threads=${CM1_THREADS:-1}
debug=${CM1_DEBUG:-false}
jobs=${JOBS:-$(nproc 2>/dev/null || echo 2)}
config="$fc-dp"
[ "$use_omp" = true ] && config="$config-omp"
[ "$debug" = true ] && config="$config-debug"
build=${CM1_BUILD_DIR:-$root/build/cm1-fortran-$config}

command -v "$fc" >/dev/null 2>&1 || die "Fortran compiler '$fc' not found"

# ---------------------------------------------------------------- case list
cases=("$@")
if [ ${#cases[@]} -eq 0 ]; then
  for d in "$cases_dir"/*/; do
    [ -f "$d/namelist.input" ] && cases+=("$(basename "$d")")
  done
fi
[ ${#cases[@]} -gt 0 ] || die "no cases found in $cases_dir"
for c in "${cases[@]}"; do
  [ -f "$cases_dir/$c/namelist.input" ] || die "unknown case '$c' (no $cases_dir/$c/namelist.input)"
done

# ---------------------------------------------------------------- build
# Copy src/ (preserving mtimes, so make stays incremental) and build there;
# the Makefile writes ../run/cm1.exe relative to src.
echo "== building Fortran CM1 ($config) in $build"
mkdir -p "$build/src" "$build/run"
cp -a "$root/src/." "$build/src/"
make_opts=(FC="$fc" USE_DOUBLE=true USE_OPENMP="$use_omp" DEBUG="$debug")
if command -v nf-config >/dev/null 2>&1; then
  make_opts+=(USE_NETCDF=true NETCDFBASE="$(nf-config --prefix)")
fi
if ! make -C "$build/src" -j"$jobs" "${make_opts[@]}" >"$build/build.log" 2>&1; then
  tail -50 "$build/build.log" >&2
  die "Fortran build failed; full log: $build/build.log"
fi
exe="$build/run/cm1.exe"
[ -x "$exe" ] || die "build finished but $exe is missing"

# ---------------------------------------------------------------- manifest
# OUT/<case>/MANIFEST.txt describes one case; OUT/MANIFEST.txt is rebuilt
# from all case manifests present (so partial reruns keep other cases).
build_info=$(
  echo "format_version 1"
  echo "git_commit $(git -C "$root" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "git_dirty $(git -C "$root" status --porcelain -- src cxx/cases 2>/dev/null | grep -q . && echo yes || echo no)"
  echo "compiler $($fc --version | head -1)"
  echo "build_config $config"
  echo "make_options ${make_opts[*]}"
  echo "omp_num_threads $threads"
  echo "created_utc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
)

# Every .bin must be 8*prod(ub-lb+1) bytes as index.txt says.
check_dump() {
  local dir=$1 n=0 name kind rest bytes want
  [ -s "$dir/index.txt" ] || die "$dir/index.txt is missing or empty"
  while read -r name kind _ rest; do
    [ "${name:0:1}" = "#" ] && continue
    [ "$kind" = 8 ] || die "$dir: $name has kind $kind, expected 8 (double precision build?)"
    want=$(echo "$rest" | awk -v k="$kind" '{p=k; for(i=1;i<=NF;i+=2) p*=($(i+1)-$i+1); print p}')
    [ -f "$dir/$name.bin" ] || die "$dir/$name.bin listed in index.txt but missing"
    bytes=$(stat -c %s "$dir/$name.bin")
    [ "$bytes" = "$want" ] || die "$dir/$name.bin has $bytes bytes, index.txt implies $want"
    n=$((n + 1))
  done <"$dir/index.txt"
  [ $n -gt 0 ] || die "$dir/index.txt lists no arrays"
}

# ---------------------------------------------------------------- run cases
for c in "${cases[@]}"; do
  src="$cases_dir/$c"
  run="$out/$c"
  rm -rf "$run" && mkdir -p "$run"
  cp "$src/namelist.input" "$run/"
  # extra input files some cases need (e.g. input_sounding)
  for f in "$src"/*; do
    case "$(basename "$f")" in namelist.input|every|README*) ;; *) cp -r "$f" "$run/" ;; esac
  done
  every=1
  [ -f "$src/every" ] && every=$(tr -d '[:space:]' <"$src/every")
  echo "== $c (dump every $every steps, OMP_NUM_THREADS=$threads)"
  if ! (cd "$run" && CM1_DUMP_DIR="$run" CM1_DUMP_EVERY="$every" OMP_NUM_THREADS="$threads" "$exe" >cm1.log 2>&1); then
    tail -30 "$run/cm1.log" >&2
    die "case $c: cm1.exe failed; log: $run/cm1.log"
  fi
  if ! grep -q "Program terminated normally" "$run/cm1.log"; then
    tail -30 "$run/cm1.log" >&2
    die "case $c: cm1.exe did not terminate normally; log: $run/cm1.log"
  fi
  # model output files are not part of the references
  rm -f "$run"/cm1out_* "$run"/*.dat "$run"/*.ctl "$run"/*.nc
  shopt -s nullglob
  steps=("$run"/step_*)
  shopt -u nullglob
  [ ${#steps[@]} -gt 0 ] || die "case $c: no step_* dumps written (is src/cm1dump.F hooked in?)"
  for s in "${steps[@]}"; do check_dump "$s"; done
  last=$(basename "${steps[-1]}")
  { echo "$build_info"; echo "case $c every $every steps ${#steps[@]} last $last"; } >"$run/MANIFEST.txt"
  echo "   ${#steps[@]} dumps, last $last, $(du -sh "$run" | cut -f1)"
done

manifest="$out/MANIFEST.txt"
{
  echo "$build_info" | grep -v '^created_utc'
  for m in "$out"/*/MANIFEST.txt; do grep '^case ' "$m"; done
} >"$manifest"
echo "== reference dumps written to $out (manifest: $manifest)"
