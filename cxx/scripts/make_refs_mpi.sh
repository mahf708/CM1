#!/usr/bin/env bash
# Build the Fortran CM1 with MPI (double precision) and write per-rank
# reference dumps for the cases in cxx/cases that have a `ranks` file.
# See cxx/cases/README.md ("MPI references") for the layout.
#
#   cxx/scripts/make_refs_mpi.sh OUT_DIR [case ...]
#
# A case's `ranks` file lists the rank counts to run (e.g. "2 4"). For each
# count N the dumps go to OUT_DIR/<case>/np<N>/rank_RRRR/step_NNNNNN (one
# directory per rank, src/cm1dump.F; index.txt also records the rank's
# subdomain in a '#decomp' line). Only the np<N> directories are written, so
# OUT_DIR can be the same directory as make_refs.sh's (the serial run of the
# case, OUT_DIR/<case>/step_*). OUT_DIR/MANIFEST_MPI.txt records how the
# dumps were made.
#
# Environment:
#   CM1_BUILD_DIR    build directory     (default: <repo>/build/cm1-fortran-<fc>-dp-mpi)
#   CM1_THREADS      OMP_NUM_THREADS for the runs (default: 1)
#   MPIRUN           MPI launcher        (default: mpirun)
#   MPIRUN_ARGS      extra launcher arguments (default: --oversubscribe)
#   JOBS             parallel make jobs  (default: nproc)
#   FC               Fortran compiler    (default: gfortran; built through mpifort)
#
# The MPI build needs netCDF (cases with MPI references write netCDF output,
# output_format = 2: CM1's GrADS output does not work with MPI in double
# precision).
set -euo pipefail

die() { echo "make_refs_mpi.sh: error: $*" >&2; exit 1; }

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
cases_dir="$root/cxx/cases"
[ $# -ge 1 ] || die "usage: make_refs_mpi.sh OUT_DIR [case ...]"
out=$1
shift
mkdir -p "$out"
out=$(cd "$out" && pwd)

fc=${FC:-gfortran}
threads=${CM1_THREADS:-1}
jobs=${JOBS:-$(nproc 2>/dev/null || echo 2)}
mpirun=${MPIRUN:-mpirun}
mpirun_args=${MPIRUN_ARGS:---oversubscribe}
config="$fc-dp-mpi"
build=${CM1_BUILD_DIR:-$root/build/cm1-fortran-$config}

command -v mpifort >/dev/null 2>&1 || die "mpifort not found (install an MPI, e.g. libopenmpi-dev)"
command -v "$mpirun" >/dev/null 2>&1 || die "MPI launcher '$mpirun' not found"
command -v nf-config >/dev/null 2>&1 || die "nf-config not found (netCDF-Fortran is required)"

# ---------------------------------------------------------------- case list
cases=("$@")
if [ ${#cases[@]} -eq 0 ]; then
  for d in "$cases_dir"/*/; do
    [ -f "$d/namelist.input" ] && [ -f "$d/ranks" ] && cases+=("$(basename "$d")")
  done
fi
[ ${#cases[@]} -gt 0 ] || die "no cases with a ranks file found in $cases_dir"
for c in "${cases[@]}"; do
  [ -f "$cases_dir/$c/namelist.input" ] || die "unknown case '$c' (no $cases_dir/$c/namelist.input)"
  [ -f "$cases_dir/$c/ranks" ] || die "case '$c' has no ranks file"
done

# ---------------------------------------------------------------- build
echo "== building Fortran CM1 ($config) in $build"
mkdir -p "$build/src" "$build/run"
cp -a "$root/src/." "$build/src/"
make_opts=(FC="$fc" USE_DOUBLE=true USE_MPI=true USE_NETCDF=true NETCDFBASE="$(nf-config --prefix)")
if ! make -C "$build/src" -j"$jobs" "${make_opts[@]}" >"$build/build.log" 2>&1; then
  tail -50 "$build/build.log" >&2
  die "Fortran build failed; full log: $build/build.log"
fi
exe="$build/run/cm1.exe"
[ -x "$exe" ] || die "build finished but $exe is missing"

build_info=$(
  echo "format_version 1"
  echo "git_commit $(git -C "$root" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "git_dirty $(git -C "$root" status --porcelain -- src cxx/cases 2>/dev/null | grep -q . && echo yes || echo no)"
  echo "compiler $($fc --version | head -1)"
  echo "mpi $($mpirun --version 2>&1 | head -1)"
  echo "build_config $config"
  echo "make_options ${make_opts[*]}"
  echo "omp_num_threads $threads"
  echo "created_utc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
)

# Every .bin must be 8*prod(ub-lb+1) bytes as index.txt says.
check_dump() {
  local dir=$1 n=0 name kind rest bytes want
  [ -s "$dir/index.txt" ] || die "$dir/index.txt is missing or empty"
  grep -q '^#decomp' "$dir/index.txt" || die "$dir/index.txt has no #decomp line"
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
  every=1
  [ -f "$src/every" ] && every=$(tr -d '[:space:]' <"$src/every")
  for np in $(cat "$src/ranks"); do
    [ "$np" -ge 2 ] || die "case $c: rank counts in $src/ranks must be >= 2"
    run="$out/$c/np$np"
    rm -rf "$run" && mkdir -p "$run"
    cp "$src/namelist.input" "$run/"
    for f in "$src"/*; do
      case "$(basename "$f")" in namelist.input|every|ranks|README*) ;; *) cp -r "$f" "$run/" ;; esac
    done
    echo "== $c on $np ranks (dump every $every steps)"
    # shellcheck disable=SC2086
    if ! (cd "$run" && CM1_DUMP_DIR="$run" CM1_DUMP_EVERY="$every" OMP_NUM_THREADS="$threads" \
          "$mpirun" $mpirun_args -np "$np" "$exe" >cm1.log 2>&1); then
      tail -30 "$run/cm1.log" >&2
      die "case $c ($np ranks): cm1.exe failed; log: $run/cm1.log"
    fi
    if ! grep -q "Program terminated normally" "$run/cm1.log"; then
      tail -30 "$run/cm1.log" >&2
      die "case $c ($np ranks): cm1.exe did not terminate normally; log: $run/cm1.log"
    fi
    rm -f "$run"/cm1out_* "$run"/*.dat "$run"/*.ctl "$run"/*.nc
    nsteps=-1
    for ((r = 0; r < np; r++)); do
      rdir=$(printf '%s/rank_%04d' "$run" "$r")
      shopt -s nullglob
      steps=("$rdir"/step_*)
      shopt -u nullglob
      [ ${#steps[@]} -gt 0 ] || die "case $c ($np ranks): no dumps in $rdir"
      [ $nsteps -lt 0 ] || [ ${#steps[@]} -eq $nsteps ] || die "case $c ($np ranks): ranks wrote different numbers of dumps"
      nsteps=${#steps[@]}
      for s in "${steps[@]}"; do check_dump "$s"; done
    done
    last=$(basename "${steps[-1]}")
    { echo "$build_info"; echo "case $c ranks $np every $every steps $nsteps last $last"; } >"$run/MANIFEST.txt"
    echo "   $nsteps dumps per rank, last $last, $(du -sh "$run" | cut -f1)"
  done
done

manifest="$out/MANIFEST_MPI.txt"
{
  echo "$build_info" | grep -v '^created_utc'
  for m in "$out"/*/np*/MANIFEST.txt; do grep '^case ' "$m"; done
} >"$manifest"
echo "== MPI reference dumps written to $out (manifest: $manifest)"
