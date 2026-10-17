"""Reader for the float64 reference dumps written by CM1's cm1dump module
(CM1_DUMP_DIR; see cxx/README.md on the Kokkos branch).

Each ``step_NNNNNN`` directory holds ``index.txt`` (name, bytes per value,
rank, then lower/upper bound per dimension) and raw ``<name>.bin`` files in
Fortran order.  Fortran order of ``a(i,j,k)`` is C order of ``a[k,j,i]``, so
arrays come back in cm1x's layout, halos included.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_index(step_dir):
    """Return {name: [(lo, hi), ...]} in Fortran dimension order, and the step time."""
    out, time = {}, None
    for line in (Path(step_dir) / "index.txt").read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0].startswith("#"):
            if parts[0] == "#step_time":
                time = float(parts[-1].replace("D", "E"))
            continue
        name, nbytes, rank = parts[0], int(parts[1]), int(parts[2])
        b = [int(x) for x in parts[3 : 3 + 2 * rank]]
        assert nbytes == 8, (name, nbytes)
        out[name] = [(b[2 * d], b[2 * d + 1]) for d in range(rank)]
    return out, time


def read_dump(step_dir, names=None):
    """Read a dump step as {name: float64 array} in [k, j, i] order (halos kept)."""
    idx, time = read_index(step_dir)
    res = {}
    for name, bounds in idx.items():
        if names is not None and name not in names:
            continue
        shape = tuple(hi - lo + 1 for lo, hi in reversed(bounds))
        res[name] = np.fromfile(Path(step_dir) / f"{name}.bin", dtype="<f8").reshape(
            shape
        )
    return res, time
