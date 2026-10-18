#!/usr/bin/env python3
"""Compare two trees of CM1 reference dumps (see cxx/cases/README.md).

    compare_dumps.py A B [--rtol R] [--atol T] [--quiet]

A and B are either reference roots (holding <case>/step_NNNNNN) or single
cases (holding step_NNNNNN). Every array in every step of A must exist in B
with the same bounds. With the default tolerances (0) the comparison is
bit-for-bit. Prints, per case, whether it is bitwise identical and otherwise
the largest absolute and relative difference and where it first appears.
Exit status: 0 if all within tolerance, 1 otherwise, 2 on a layout mismatch.
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def read_index(step_dir):
    arrays = {}
    for line in (step_dir / "index.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        f = line.split()
        name, kind, nd = f[0], int(f[1]), int(f[2])
        bounds = [(int(f[3 + 2 * d]), int(f[4 + 2 * d])) for d in range(nd)]
        arrays[name] = (kind, bounds)
    return arrays


def read_array(step_dir, name, kind, bounds):
    shape = [ub - lb + 1 for lb, ub in bounds]
    dtype = {8: "<f8", 4: "<f4"}[kind]
    a = np.fromfile(step_dir / f"{name}.bin", dtype=dtype)
    return a.reshape(shape, order="F")


def cases_of(root):
    if any(root.glob("step_*")):
        return {root.name: root}
    return {
        d.name: d
        for d in sorted(root.iterdir())
        if d.is_dir() and any(d.glob("step_*"))
    }


def compare_case(a_dir, b_dir, rtol, atol, quiet):
    ok, layout_ok, bitwise = True, True, True
    worst = (0.0, 0.0, None)  # (abs, rel, where)
    first = None
    for sa in sorted(a_dir.glob("step_*")):
        sb = b_dir / sa.name
        if not (sb / "index.txt").exists():
            print(f"  missing {sb}")
            return False, False
        ia, ib = read_index(sa), read_index(sb)
        if ia != ib:
            print(f"  {sa.name}: index.txt differs")
            return False, False
        for name, (kind, bounds) in ia.items():
            x = read_array(sa, name, kind, bounds)
            y = read_array(sb, name, kind, bounds)
            if x.tobytes() == y.tobytes():
                continue
            bitwise = False
            d = np.abs(x - y)
            scale = np.maximum(np.abs(x), np.abs(y))
            amax = float(np.nanmax(d))
            with np.errstate(divide="ignore", invalid="ignore"):
                rel = np.where(scale > 0, d / scale, 0.0)
            rmax = float(np.nanmax(rel))
            if first is None:
                idx = np.unravel_index(int(np.argmax(d)), d.shape)
                fidx = tuple(int(i) + lb for i, (lb, _) in zip(idx, bounds))
                first = f"{sa.name}/{name}{fidx}"
            if amax > worst[0]:
                worst = (amax, rmax, f"{sa.name}/{name}")
            if not np.all(d <= atol + rtol * scale):
                ok = False
    if bitwise:
        status = "bitwise identical"
    else:
        status = (
            f"differs: first at {first}, max abs {worst[0]:.3e} "
            f"(rel {worst[1]:.3e}) in {worst[2]}"
        )
        if ok:
            status += " [within tolerance]"
    if not quiet or not ok:
        print(f"  {a_dir.name}: {status}")
    return ok, layout_ok


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("a", type=Path)
    p.add_argument("b", type=Path)
    p.add_argument("--rtol", type=float, default=0.0)
    p.add_argument("--atol", type=float, default=0.0)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    ca, cb = cases_of(args.a), cases_of(args.b)
    if not ca:
        print(f"no dumps under {args.a}")
        return 2
    if len(ca) == 1 and len(cb) == 1:
        pairs = [(next(iter(ca.values())), next(iter(cb.values())))]
    else:
        missing = sorted(set(ca) - set(cb))
        if missing:
            print(f"cases missing in {args.b}: {missing}")
            return 2
        pairs = [(ca[c], cb[c]) for c in ca]

    all_ok, layout = True, True
    for a_dir, b_dir in pairs:
        ok, lay = compare_case(a_dir, b_dir, args.rtol, args.atol, args.quiet)
        all_ok &= ok
        layout &= lay
    if not layout:
        return 2
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
