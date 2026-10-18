# Cross-checks between the CM1 ports

`crosscheck.py` runs one namelist in several implementations, step by step,
and compares every prognostic array after each step:

| implementation | what |
|---|---|
| `cm1k` | the C++/Kokkos port (`cxx/`) through its Python module, on whatever Kokkos backend it was built for (`cm1k:Cuda` on a GPU build) |
| `cm1x:<backend>[:jit]` | the array-API port (`ports/array`, on the array-port branch) on numpy, jax or torch; `:jit` compiles the whole step |
| `fortran` | CM1's double-precision state dumps (`cxx/scripts/make_refs.sh`), with `--ref-dir` |

Each implementation initializes the case itself (no shared initial dump), so
initialization is checked as well as stepping.

```bash
# cm1k module on PYTHONPATH (cxx/python/README.md), cm1x installed (pip install -e ports/array)
python ports/crosscheck/crosscheck.py --case kessler_tke --steps 150 \
    --impl cm1k --impl cm1x:numpy --impl fortran --ref-dir refs/kessler_tke \
    --md report.md --json kessler_tke.json
```

`--set name=value` overrides a namelist value in every implementation (useful
to isolate a difference: `--set irdamp=0`). Exit status: 0 agree, 1 a pair is
out of tolerance, 2 an implementation cannot run the case.

**Tolerances.** The relative difference of a field is
`max|a-b| / max(max|a|, max|b|)` over the interior points. Pairs of exact
implementations (cm1k on the CPU, Fortran) must agree to 1e-12 (in practice
they are bitwise identical). Pairs involving cm1x or a GPU build must agree to
1e-10 up to step 4 and 1e-6 up to step 100 (the schedule the cm1x tests
use), then 3e-5: over the 30-minute supercell the differences grow to ~1e-5
in the pressure perturbation under the rain shaft.

**Conventions it accounts for.** CM1 and cm1k keep the eddy viscosities
(`kmh kmv khh khv`) of the previous step in the state between steps, while
cm1x updates them to the new state at the end of the step; the check lags
cm1x's by one step (they then match bitwise).

## CI

`.github/workflows/crosscheck.yml`:

- `ports`: cm1k (OpenMP) vs cm1x (numpy, and jax eager and jit on
  `kessler_periodic`) vs Fortran on `kessler_periodic`, `kessler_bubble`,
  `kessler_openbc`, `kessler_tke`, the bundled `supercell_kessler` (300 steps)
  and `kessler_periodic_long` (300 steps). cm1x comes from the array-port
  branch at `CM1X_REF_SHA` (a `workflow_dispatch` input can pick another ref).
  The report is in the job summary.
- `cuda-gpu`: builds cm1k for CUDA with its tests and Python module, then runs
  it on a GPU through [Modal](https://modal.com) (`cxx/scripts/modal_gpu.py`,
  which runs `cxx/scripts/gpu_check.sh` there): cm1k on the device against the
  Fortran dumps for eight cases covering the dynamics, Kessler, TKE, open
  boundaries, the surface layer and the simple, YSU and MYNN PBL schemes, plus
  ctest for information (its step tests ask for bitwise agreement, which a GPU
  build does not give).

To enable the GPU run, create a Modal account, make a token
(`modal token new`, or Settings > API tokens on modal.com) and add it to the
repository as the Actions secrets `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`.
Without them the CUDA build still runs and the GPU step is skipped with a
notice. The default GPU is a T4 (`workflow_dispatch` input `gpu`: T4, L4,
A10G, A100, H100).

On any machine with an NVIDIA GPU you can run the same check directly:
configure `cxx/` with `-DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_<ARCH>=ON
-DCM1K_ENABLE_PYTHON=ON -DCM1K_REF_DIR=$PWD/refs`, build, then
`cxx/scripts/gpu_check.sh build refs gpu_out kessler_periodic kessler_tke`.
