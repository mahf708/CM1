#!/usr/bin/env python3
"""Cross-check the CM1 ports against each other and against CM1 itself.

Runs the same case, step by step, in several implementations and compares
every prognostic array (interior points) after each step:

  cm1k          C++/Kokkos port (cxx/), through its Python module; whatever
                Kokkos backend it was built for (Serial, OpenMP, Cuda, ...)
  cm1x:<bk>     array-API port (ports/array) on backend <bk> (numpy, jax,
                torch); ``cm1x:jax:jit`` jits the whole step
  fortran       CM1's own float64 state dumps (cxx/scripts/make_refs.sh),
                given with --ref-dir

Every implementation starts from its own initialization of the namelist
(no shared initial dump), so the check covers init as well as stepping.

    python ports/crosscheck/crosscheck.py --case kessler_periodic --steps 10 \\
        --impl cm1k --impl cm1x:numpy --ref-dir refs/kessler_periodic \\
        --md report.md --json report.json

The relative difference of a field is max|a - b| / max(max|a|, max|b|) over
the interior (a field that is zero in both is compared absolutely).  A pair fails when its worst field exceeds the
tolerance for that step (``--tol``; see TOLERANCES).  Exit status 1 on any
failure, 2 if an implementation cannot run the case.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# Interior of a CM1 array stored [k, j, i] with its halos (1 in z, 3 in x/y):
# Fortran k = 1..nk(+1), j = 1..nj(+1), i = 1..ni(+1).
INTERIOR = (slice(1, -1), slice(3, -3), slice(3, -3))

BASE_FIELDS = ("ua", "va", "wa", "tha", "ppi", "prs", "rho")
TURB_FIELDS = ("tkea", "kmh", "kmv", "khh", "khv")
LAGGED_K = ("kmh", "kmv", "khh", "khv")

# Expected agreement (relative), as (last step, tolerance) pieces.  cm1k is
# bitwise-identical to the double-precision Fortran on the CPU (cxx tests);
# cm1x agrees to round-off for the first steps, and from step ~5 the
# differences of a few ulps in the moist physics grow to ~1e-8 (two Fortran
# builds with different flags differ by as much; ports/array tests use the
# same schedule); past step 100 in deep moist convection (supercell_kessler,
# 30 min) they reach ~1e-5 in the pressure perturbation under the rain shaft
# and in the eddy viscosities.  A GPU build of cm1k is not bitwise either (device math
# library), so it gets the cm1x schedule.
TOLERANCES = {
    "bitwise": [(math.inf, 1e-12)],
    "roundoff": [(4, 1e-10), (100, 1e-6), (math.inf, 3e-5)],
}


def tol_at(schedule, step):
    for last, tol in schedule:
        if step <= last:
            return tol
    return schedule[-1][1]


def describe(schedule):
    parts = []
    for n, (last, tol) in enumerate(schedule):
        if math.isinf(last):
            parts.append(f"{tol:.0e}" + (" after" if n else ""))
        else:
            parts.append(f"{tol:.0e} to step {last}")
    return ", ".join(parts)


# ----------------------------------------------------------------------------
# implementations
# ----------------------------------------------------------------------------
class Cm1k:
    """cm1k.Model from the namelist (native init, cxx/src/cm1k/init)."""

    exact = True  # bitwise vs Fortran on host backends

    def __init__(self, namelist, overrides=None):
        import cm1k

        self.m = cm1k.Model(str(namelist), **(overrides or {}))
        self.m.init()
        space = cm1k.execution_space()
        self.name = (
            "cm1k" if space in ("Serial", "OpenMP", "Threads") else f"cm1k:{space}"
        )
        self.exact = space in ("Serial", "OpenMP", "Threads")
        self.qnames = [f"qa_{q}" for q in self.m.qname]
        self.les = int(self.m.namelist.get("sgsmodel", 0)) == 1

    @property
    def nstep(self):
        return self.m.nstep

    def step(self):
        self.m.step(1)

    def get(self, name):
        if not self.m.has_field(name):
            return None
        # Fortran-ordered a(i,j,k) -> C-ordered [k,j,i]
        return np.ascontiguousarray(np.asarray(self.m.get_field(name, copy=True)).T)


class Cm1x:
    """cm1x.model.Model with -DDP thresholds (fortran_dp=True)."""

    exact = False

    def __init__(self, namelist, backend="numpy", jit=False, overrides=None):
        from cm1x.model import Model

        self.m = Model(
            namelist, backend=backend, fortran_dp=True, jit=jit, overrides=overrides
        )
        self.name = f"cm1x:{backend}" + (":jit" if jit else "")
        cfg = self.m.cfg
        self.qnames = [
            f"qa_{q}"
            for q, idx in (("qv", cfg.nqv), ("qc", cfg.nqc), ("qr", cfg.nqr))
            if idx
        ]
        self._qidx = {
            f"qa_{q}": idx - 1
            for q, idx in (("qv", cfg.nqv), ("qc", cfg.nqc), ("qr", cfg.nqr))
            if idx
        }
        self.les = self.m.les

        # CM1 (and cm1k) keep the eddy viscosities of the previous step in the
        # state between steps (dump N holds K computed from state N-1, zero at
        # N = 0); cm1x's state holds them already updated to the new state.
        self._lagged = {n: None for n in LAGGED_K} if self.les else {}

    @property
    def nstep(self):
        return self.m.nstep

    def step(self):
        for n in self._lagged:
            self._lagged[n] = self._raw(n)
        self.m.step()

    def get(self, name):
        if name in self._lagged:
            a = self._lagged[name]
            return np.zeros_like(self._raw(name)) if a is None else a
        return self._raw(name)

    def _raw(self, name):
        s = self.m.state
        if name in self._qidx:
            a = s["qa"][self._qidx[name]]
        elif name in s:
            a = s[name]
        else:
            return None
        return np.asarray(self.m.bk.to_numpy(a), dtype=np.float64)


class Fortran:
    """CM1's float64 dumps (one step_NNNNNN directory per dumped step)."""

    exact = True

    def __init__(self, ref_dir):
        try:
            from cm1k import dump  # pure numpy, works without the extension
        except ImportError:
            sys.path.insert(0, str(ROOT / "cxx" / "python"))
            from cm1k import dump

        self.dump = dump
        self.root = Path(ref_dir)
        self.name = "fortran"
        self.steps = set(dump.list_steps(self.root))
        self._cur = 0
        self._d = dump.read(self.root, 0)
        names = self._d.field_names()
        self.qnames = [n for n in names if n.startswith("qa_")]
        self.les = "tkea" in names

    @property
    def nstep(self):
        return self._cur

    def step(self):
        self._cur += 1
        self._d = (
            self.dump.read(self.root, self._cur) if self._cur in self.steps else None
        )

    def get(self, name):
        if self._d is None or name not in self._d.field_names():
            return None
        return np.ascontiguousarray(np.asarray(self._d[name].data).T)


def make_impl(spec, namelist, ref_dir, overrides=None):
    parts = spec.split(":")
    if parts[0] == "cm1k":
        return Cm1k(namelist, overrides)
    if parts[0] == "cm1x":
        backend = parts[1] if len(parts) > 1 else "numpy"
        return Cm1x(
            namelist, backend=backend, jit="jit" in parts[2:], overrides=overrides
        )
    if parts[0] == "fortran":
        if not ref_dir:
            raise SystemExit("--impl fortran needs --ref-dir")
        if overrides:
            raise SystemExit("--set cannot be used with --impl fortran")
        return Fortran(ref_dir)
    raise SystemExit(f"unknown implementation {spec!r}")


# ----------------------------------------------------------------------------
def rel_diff(a, b):
    a, b = a[INTERIOR], b[INTERIOR]
    if a.shape != b.shape:
        return math.inf, None
    d = np.abs(a - b)
    d[~np.isfinite(d)] = np.inf
    i = np.unravel_index(int(np.argmax(d)), d.shape)
    scale = max(float(np.abs(a).max()), float(np.abs(b).max()))
    dmax = float(d[i])
    # report Fortran (i, j, k) of the interior point
    where = (int(i[2]) + 1, int(i[1]) + 1, int(i[0]) + 1)
    return (dmax / scale if scale > 0 else dmax), where


def find_namelist(case):
    p = Path(case)
    if p.is_file():
        return p
    if (p / "namelist.input").is_file():
        return p / "namelist.input"
    for d in (ROOT / "cxx" / "cases", ROOT / "ports" / "reference"):
        if (d / case / "namelist.input").is_file():
            return d / case / "namelist.input"
    raise SystemExit(f"no namelist for case {case!r}")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--case",
        required=True,
        help="case name (cxx/cases, ports/reference) or namelist path",
    )
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument(
        "--impl",
        action="append",
        default=[],
        help="cm1k, cm1x:<backend>[:jit], fortran",
    )
    ap.add_argument(
        "--ref-dir", help="Fortran dump directory of this case (for --impl fortran)"
    )
    ap.add_argument(
        "--tol",
        action="append",
        default=[],
        metavar="A~B=SCHEDULE",
        help="tolerance schedule for a pair (bitwise, roundoff or a number); default from the pair",
    )
    ap.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="override a namelist value in every implementation (not with fortran)",
    )
    ap.add_argument("--md", help="append a markdown report here")
    ap.add_argument("--json", help="write the per-step results here")
    args = ap.parse_args(argv)

    namelist = find_namelist(args.case)
    case = (
        Path(args.case).name if not Path(args.case).is_file() else namelist.parent.name
    )
    nml_overrides = {}
    for kv in args.set:
        k, v = kv.split("=", 1)
        nml_overrides[k] = json.loads(v)
    impls = []
    for spec in args.impl or ["cm1k", "cm1x:numpy"]:
        try:
            impls.append(make_impl(spec, namelist, args.ref_dir, nml_overrides))
        except NotImplementedError as e:
            print(f"{case}: {spec} cannot run this case: {e}")
            return 2
    if len(impls) < 2:
        raise SystemExit("need at least two implementations")

    fields = list(BASE_FIELDS)
    qsets = [set(im.qnames) for im in impls]
    if len({frozenset(q) for q in qsets}) != 1:
        print(
            f"{case}: implementations disagree on the moisture variables: {[im.qnames for im in impls]}"
        )
        return 1
    fields += impls[0].qnames
    if all(im.les for im in impls):
        fields += list(TURB_FIELDS)

    overrides = {}
    for t in args.tol:
        pair, sched = t.split("=")
        overrides[pair] = (
            TOLERANCES[sched] if sched in TOLERANCES else [(math.inf, float(sched))]
        )

    pairs = []
    for i in range(len(impls)):
        for j in range(i + 1, len(impls)):
            a, b = impls[i], impls[j]
            key = f"{a.name}~{b.name}"
            sched = (
                overrides.get(key)
                or TOLERANCES["bitwise" if a.exact and b.exact else "roundoff"]
            )
            pairs.append((a, b, key, sched))

    results = {key: [] for _, _, key, _ in pairs}
    timing = {im.name: 0.0 for im in impls}
    failed = False

    def compare(step):
        nonlocal failed
        for a, b, key, sched in pairs:
            worst, wf, ww, nf = -1.0, None, None, 0
            per = {}
            for f in fields:
                xa, xb = a.get(f), b.get(f)
                if xa is None or xb is None:
                    continue
                r, where = rel_diff(xa, xb)
                per[f] = r
                nf += 1
                if not r <= worst:  # also catches nan
                    worst, wf, ww = r, f, where
            if nf == 0:
                continue
            worst = max(worst, 0.0)
            tol = tol_at(sched, step)
            ok = worst <= tol
            failed |= not ok
            results[key].append(
                {
                    "step": step,
                    "worst": worst,
                    "field": wf,
                    "where": ww,
                    "tol": tol,
                    "ok": ok,
                    "fields": per,
                }
            )
            print(
                f"{case} step {step:4d} {key:28s} worst {worst:9.2e} ({wf} at {ww})"
                f"  tol {tol:.0e} {'ok' if ok else 'FAIL'}"
            )

    compare(0)
    for n in range(1, args.steps + 1):
        for im in impls:
            t0 = time.perf_counter()
            try:
                im.step()
            except NotImplementedError as e:
                print(f"{case}: {im.name} cannot run this case: {e}")
                return 2
            timing[im.name] += time.perf_counter() - t0
            assert im.nstep == n, (im.name, im.nstep, n)
        compare(n)

    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "case": case,
                    "steps": args.steps,
                    "results": results,
                    "seconds": timing,
                },
                indent=1,
            )
        )
    if args.md:
        lines = [
            f"#### {case} ({args.steps} steps)",
            "",
            "| pair | largest rel. diff (field, step) | at last step | tolerance | |",
            "|---|---|---|---|---|",
        ]
        for _, _, key, sched in pairs:
            rs = results[key]
            if not rs:
                lines.append(f"| {key} | n/a | | | |")
                continue
            w = max(rs, key=lambda r: r["worst"])
            last = rs[-1]
            ok = all(r["ok"] for r in rs)
            big = (
                "identical (bitwise)"
                if w["worst"] == 0
                else f"{w['worst']:.2e} ({w['field']}, {w['step']})"
            )
            fin = (
                "identical"
                if last["worst"] == 0
                else f"{last['worst']:.2e} ({last['field']})"
            )
            lines.append(
                f"| {key} | {big} | {fin} | {describe(sched)} "
                f"| {'✅' if ok else '❌'} |"
            )
        lines.append("")
        lines.append(
            "Time stepping: "
            + ", ".join(
                f"{k} {v / max(args.steps, 1):.3f} s/step"
                for k, v in timing.items()
                if k != "fortran"
            )
        )
        lines.append("")
        with open(args.md, "a") as fh:
            fh.write("\n".join(lines) + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    os.environ.setdefault("OMP_PROC_BIND", "false")
    sys.exit(main())
