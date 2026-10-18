# cm1drive: one driver for every CM1 implementation

`cm1drive.py` runs a CM1 case on any implementation in this repository through
one interface and compares the results:

| backend | what |
|---|---|
| `fortran` | CM1's own double-precision state dumps (`cxx/scripts/make_refs.sh`), read step by step |
| `cm1k` | the C++/Kokkos port (`cxx/`) through its Python module, on whatever Kokkos backend it was built for (it reports itself as `cm1k:Cuda` on a GPU build) |
| `cm1x:numpy`, `cm1x:jax`, `cm1x:torch` | the array-API port (`ports/array`, on the array-port branch) |
| `cm1x:jax:jit`, `cm1x:torch:jit` | the same, compiled (jax: the whole step; torch: per routine, slow first compile) |

It is a thin layer over the adapters in `ports/crosscheck/crosscheck.py`: each
port initializes the case from its namelist, and a run keeps float64 numpy
snapshots of the prognostic fields (interior points, `[k, j, i]`).

```python
import cm1drive

ref = cm1drive.run(
    "supercell_kessler",
    "fortran",
    steps=300,
    every=30,
    ref_dir="refs/supercell_kessler",
)
k = cm1drive.run("supercell_kessler", "cm1k", steps=300, every=30)
t = cm1drive.run("supercell_kessler", "cm1x:torch", steps=300, every=30)

k["wa"]  # last snapshot of w, [k, j, i]
k[150, "qa_qr"]  # rain water at step 150
cm1drive.compare(t, ref)  # {field: {step: relative difference}}
print(cm1drive.table([k, t], ref))
```

or from the shell (all backends that import, plus Fortran when given dumps):

```bash
python ports/driver/cm1drive.py --case supercell_kessler --steps 30 --every 30 \
    --ref-dir refs/supercell_kessler
```

Cases are looked up in `cxx/cases`, then `ports/reference`, or given as a
namelist path. `run(..., overrides={"dtl": 3.0})` changes namelist values
(not with `fortran`).

## Notebook

`supercell_backends.ipynb` runs the bundled `supercell_kessler` case (40×40×40
at 1 km, Kessler, TKE turbulence, open radiative boundaries, moving domain) for
30 minutes on Fortran, cm1k, and cm1x on numpy, jax (eager and jit) and torch,
plots the storm, the difference of every backend from Fortran over time,
time per step and where the differences sit, and fails its last cell when a
backend leaves its expected agreement:

| backend | vs Fortran over 30 min | why |
|---|---|---|
| `cm1k` (CPU) | bitwise (1e-12) | same operations in the same order |
| `cm1x:numpy` | bitwise with numpy's AVX-512 math off, else 1e-5 | glibc's exp/log/pow, like CM1 |
| `cm1x:jax`, `cm1x:torch` (and `:jit`) | 1e-5 | their own vectorized math; the open boundaries amplify last-bit differences |

Settings come from the environment: `CM1_REF_DIR` (default
`refs/supercell_kessler`), `CM1_STEPS` (300), `CM1_EVERY` (30, the case's dump
interval) and `CM1_BACKENDS` (comma-separated).

## Setup

```bash
# Fortran dumps (double precision CM1; needs gfortran and netCDF-Fortran)
cxx/scripts/make_refs.sh refs supercell_kessler

# cm1k Python module (cxx/python/README.md)
git submodule update --init --recursive cxx/externals/ekat
cmake -S cxx -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCM1K_ENABLE_PYTHON=ON \
      -DCM1K_ENABLE_TESTS=OFF -DCM1K_ENABLE_FORTRAN_BRIDGE=OFF -DKokkos_ENABLE_OPENMP=ON \
      -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build build --target _cm1k -j
export PYTHONPATH=$PWD/build/python

# cm1x from the array-port branch, with the array libraries you want
git worktree add ../cm1x origin/claude/project-thread-p9tdpk
pip install -e ../cm1x/ports/array jax torch matplotlib jupyter

# bitwise numpy and one libm code path, as in CI
export GLIBC_TUNABLES=glibc.cpu.hwcaps=-AVX512F,-AVX2,-FMA,-FMA4
export NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL AVX512_SPR X86_V4"
jupyter lab ports/driver/supercell_backends.ipynb
```

`python ports/driver/run_notebook.py ports/driver/supercell_backends.ipynb out`
executes it headless and writes `out/supercell_backends.{ipynb,html}`.

## CI

`.github/workflows/driver.yml` builds the Fortran dumps (cached), the cm1k
module (OpenMP) and cm1x at `CM1X_REF_SHA` with numpy, jax and CPU torch, and
executes the notebook on `ubuntu-latest`. The executed notebook and its html are
the `driver-notebook` artifact; the comparison table is in the job summary.
