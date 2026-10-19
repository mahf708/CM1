"""One Python interface over every CM1 implementation in this repository.

    import cm1drive

    cm1drive.available()                    # backends that can run here
    ref = cm1drive.run("supercell_kessler", "fortran", steps=300, every=30,
                       ref_dir="refs/supercell_kessler")
    k = cm1drive.run("supercell_kessler", "cm1k", steps=300, every=30)
    t = cm1drive.run("supercell_kessler", "cm1x:torch", steps=300, every=30)
    cm1drive.compare(k, ref)                # {field: {step: rel. difference}}

Backends (the names of ports/crosscheck/crosscheck.py):

  fortran           CM1's own double-precision state dumps
                    (cxx/scripts/make_refs.sh); needs ``ref_dir``
  cm1k              the C++/Kokkos port through its Python module (cxx/python),
                    on whatever Kokkos backend it was built for
  cm1x:<bk>[:jit]   the array-API port (ports/array, installed from the
                    array-port branch) on numpy, jax or torch; ``:jit`` compiles
                    the step (jax: the whole step, torch: per routine)

Each run starts from the implementation's own initialization of the namelist,
so the comparison covers initialization as well as time stepping.  A run keeps
snapshots of the prognostic fields (interior points, C order ``[k, j, i]``,
float64 numpy) every ``every`` steps, plus the wall time spent stepping.
"""

from __future__ import annotations

import importlib.util
import math
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ports" / "crosscheck"))

import crosscheck  # noqa: E402  (the implementation adapters live there)

INTERIOR = crosscheck.INTERIOR

BACKENDS = (
    "fortran",
    "cm1k",
    "cm1x:numpy",
    "cm1x:jax",
    "cm1x:jax:jit",
    "cm1x:torch",
    "cm1x:torch:jit",
)


def _has(module):
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def available(ref_dir=None):
    """The backends of BACKENDS whose packages import here.

    ``fortran`` is listed when ``ref_dir`` holds dumps.
    """
    out = []
    for b in BACKENDS:
        kind, *rest = b.split(":")
        if kind == "fortran":
            ok = ref_dir is not None and (Path(ref_dir) / "step_000000").is_dir()
        elif kind == "cm1k":
            ok = _has("cm1k._cm1k") if _has("cm1k") else False
        else:
            ok = _has("cm1x") and _has(rest[0])
        if ok:
            out.append(b)
    return out


def find_namelist(case):
    """Path of a case's namelist.input (cxx/cases, ports/reference, or a path)."""
    return crosscheck.find_namelist(case)


def read_namelist(path):
    """The scalar values of a namelist.input as a flat {name: value} dict."""
    out = {}
    for line in Path(path).read_text().splitlines():
        line = line.split("!", 1)[0]
        for name, val in re.findall(r"(\w+)\s*=\s*([^,=]+?)\s*(?:,|$)", line):
            v = val.strip().strip("'\"")
            for conv in (int, float):
                try:
                    v = conv(v)
                    break
                except ValueError:
                    pass
            if v in (".true.", ".false."):
                v = v == ".true."
            out[name.lower()] = v
    return out


@dataclass
class Grid:
    """Cell-centre coordinates (km) of the interior points of a uniform grid."""

    nx: int
    ny: int
    nz: int
    dx: float
    dy: float
    dz: float
    dtl: float

    @classmethod
    def from_namelist(cls, path):
        nl = read_namelist(path)
        return cls(
            *(int(nl[k]) for k in ("nx", "ny", "nz")),
            *(float(nl[k]) for k in ("dx", "dy", "dz", "dtl")),
        )

    @property
    def x(self):
        return (np.arange(self.nx) + 0.5) * self.dx / 1000.0

    @property
    def y(self):
        return (np.arange(self.ny) + 0.5) * self.dy / 1000.0

    @property
    def z(self):
        return (np.arange(self.nz) + 0.5) * self.dz / 1000.0


@dataclass
class Run:
    """Snapshots of one case on one backend."""

    case: str
    backend: str
    name: str  # as the implementation reports itself (e.g. cm1k:Cuda)
    grid: Grid
    fields: list
    snapshots: dict = field(default_factory=dict)  # step -> {field: [k,j,i]}
    seconds: float = 0.0  # wall time spent stepping (not initializing)
    init_seconds: float = 0.0
    first_seconds: float = 0.0  # the first step (includes any jit compile)

    @property
    def steps(self):
        return sorted(self.snapshots)

    def __getitem__(self, key):
        """``run[step, field]`` or ``run[field]`` (last snapshot)."""
        if isinstance(key, tuple):
            step, name = key
        else:
            step, name = self.steps[-1], key
        return self.snapshots[step][name]

    @property
    def seconds_per_step(self):
        """Wall time per step after the first (which includes compiling)."""
        n = self.steps[-1] if self.snapshots else 0
        if n > 1:
            return (self.seconds - self.first_seconds) / (n - 1)
        return self.seconds if n else math.nan


def _fields(impl, extra=()):
    names = list(crosscheck.BASE_FIELDS) + list(impl.qnames)
    if impl.les:
        names += list(crosscheck.TURB_FIELDS)
    return names + [f for f in extra if f not in names]


def run(
    case,
    backend,
    steps,
    every=1,
    ref_dir=None,
    fields=None,
    overrides=None,
    progress=None,
):
    """Run ``case`` for ``steps`` steps on ``backend``; keep every ``every``-th.

    ``fields`` defaults to the prognostic set crosscheck.py compares (winds,
    theta, Exner and pressure perturbations, density, moisture, and the TKE
    fields with ``sgsmodel = 1``).  ``overrides`` changes namelist values
    (not with ``fortran``).  ``progress(step)`` is called after each step.
    """
    nml = find_namelist(case)
    name = Path(case).name if not Path(case).is_file() else nml.parent.name
    t0 = time.perf_counter()
    impl = crosscheck.make_impl(backend, nml, ref_dir, overrides)
    init = time.perf_counter() - t0
    if isinstance(impl, crosscheck.Fortran):
        missing = [s for s in range(0, steps + 1, every) if s not in impl.steps]
        if missing:
            have = sorted(impl.steps)
            raise ValueError(
                f"{ref_dir} has no dump for steps {missing[:5]}...; it has "
                f"{have[:3]}...{have[-1:]} (choose `every` to match the case's)"
            )
    names = fields or _fields(impl)
    r = Run(name, backend, impl.name, Grid.from_namelist(nml), names)
    r.init_seconds = init

    def snap(n):
        snap = {}
        for f in names:
            a = impl.get(f)
            if a is not None:
                snap[f] = np.array(a[INTERIOR], dtype=np.float64)
        r.snapshots[n] = snap

    snap(0)
    for n in range(1, steps + 1):
        t0 = time.perf_counter()
        impl.step()
        dt = time.perf_counter() - t0
        r.seconds += dt
        if n == 1:
            r.first_seconds = dt
        assert impl.nstep == n, (backend, impl.nstep, n)
        if n % every == 0 or n == steps:
            snap(n)
        if progress:
            progress(n)
    return r


def rel_diff(a, b):
    """max|a - b| / max(max|a|, max|b|) (absolute where both are zero)."""
    d = float(np.max(np.abs(a - b))) if a.size else 0.0
    scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))))
    return d / scale if scale > 0 else d


def compare(a, b, fields=None):
    """{field: {step: relative difference}} over the snapshots both runs have."""
    steps = sorted(set(a.snapshots) & set(b.snapshots))
    names = fields or [f for f in a.fields if f in b.fields]
    out = {}
    for f in names:
        row = {}
        for s in steps:
            if f in a.snapshots[s] and f in b.snapshots[s]:
                row[s] = rel_diff(a.snapshots[s][f], b.snapshots[s][f])
        if row:
            out[f] = row
    return out


def worst(diffs):
    """{step: (largest relative difference, its field)} of a compare() result."""
    out = {}
    for f, row in diffs.items():
        for s, v in row.items():
            if s not in out or not v <= out[s][0]:
                out[s] = (v, f)
    return dict(sorted(out.items()))


BITWISE = 1e-12
# Implementations with their own math library (jax/XLA, torch, a GPU) differ
# from CM1 in the last bit of exp/log/pow; most cases keep that below ~1e-9
# (crosscheck.py's tolerance), but open radiative boundaries amplify it: on
# supercell_kessler jax reaches 2e-6 (ppi) after 30 minutes.
LOOSE = 1e-5


def numpy_uses_libm():
    """True when numpy's AVX-512 exp/log/pow are off (NPY_DISABLE_CPU_FEATURES
    or a CPU without them): numpy then calls glibc like CM1 and cm1k."""
    try:
        from numpy._core._multiarray_umath import __cpu_features__
    except ImportError:  # numpy < 2
        from numpy.core._multiarray_umath import __cpu_features__
    return not __cpu_features__.get("AVX512_SKX", False)


def tolerance(r):
    """Largest relative difference expected between run ``r`` and Fortran.

    cm1k on the CPU is bitwise identical to CM1 (same operations in the same
    order), and so is cm1x on numpy when numpy uses glibc's libm (the
    environment of .github/workflows/crosscheck.yml).  Everything else (jax,
    torch, a GPU build of cm1k) gets LOOSE.
    """
    if r.backend == "fortran":
        return 0.0
    if r.backend == "cm1k":
        return BITWISE if r.name == "cm1k" else LOOSE
    if r.backend == "cm1x:numpy" and numpy_uses_libm():
        return BITWISE
    return LOOSE


def check(r, ref):
    """None if ``r`` agrees with ``ref`` (Fortran) within tolerance(r), else
    a sentence saying where it does not."""
    tol = tolerance(r)
    for s, (v, f) in worst(compare(r, ref)).items():
        if not v <= tol:
            return f"{r.name}: {v:.2e} in {f} at step {s} (tolerance {tol:.0e})"
    return None


def table(runs, ref):
    """Markdown table: each run against ``ref`` (largest difference, timing)."""
    lines = [
        f"| backend | largest rel. diff vs {ref.backend} (field, step) | tolerance | s/step |",
        "|---|---|---|---|",
    ]
    for r in runs:
        if r is ref:
            continue
        w = worst(compare(r, ref))
        if not w:
            continue
        step, (v, f) = max(w.items(), key=lambda kv: kv[1][0])
        tol = tolerance(r)
        big = "identical (bitwise)" if v == 0 else f"{v:.2e} ({f}, {step})"
        ok = "✅" if v <= tol else "❌"
        lines.append(
            f"| {r.name} | {big} | {tol:.0e} {ok} | {r.seconds_per_step:.3f} |"
        )
    return "\n".join(lines)


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="Run a CM1 case on several backends.")
    ap.add_argument("--case", default="supercell_kessler")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--backend", action="append", help="default: all available")
    ap.add_argument("--ref-dir", help="Fortran dumps of the case")
    args = ap.parse_args(argv)
    backends = args.backend or available(args.ref_dir)
    runs = [
        run(args.case, b, args.steps, args.every, ref_dir=args.ref_dir)
        for b in backends
    ]
    ref = next((r for r in runs if r.backend == "fortran"), runs[0])
    print(table(runs, ref))
    bad = [m for m in (check(r, ref) for r in runs if r is not ref) if m]
    print("\n".join(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
