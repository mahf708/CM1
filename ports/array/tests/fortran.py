"""Per-routine Fortran harness: compile a driver against CM1's own objects and
run it on arrays supplied from Python.

Driver convention (see fortran_harness/README in ports/array/README.md):
  * ``fortran_harness/drv_<name>.F90`` is a program that ``use harness_io`` and
    reads every input with ``call rd('x', x, size(x))`` and writes outputs with
    ``call wr('y', y, size(y))``. Integer/scalar parameters are passed in the
    same way as 1-element real arrays (or a ``params`` array).
  * Arrays cross the boundary in Fortran order; ``run_driver`` handles the
    transpose so Python sees ``[k, j, i]`` arrays.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
HARNESS = HERE.parent / "fortran_harness"
CACHE = Path(os.environ.get("CM1X_FORTRAN_CACHE", HERE.parent / ".fortran_cache"))

DTYPES = {"r4": np.float32, "r8": np.float64}


def have_gfortran() -> bool:
    return shutil.which("gfortran") is not None


requires_gfortran = pytest.mark.skipif(not have_gfortran(), reason="gfortran not found")


def objdir(prec: str) -> Path:
    d = CACHE / f"cm1_{prec}"
    subprocess.run([str(HARNESS / "build_objs.sh"), str(d), prec], check=True)
    return d / "src"


def build_driver(name: str, prec: str = "r8") -> Path:
    obj = objdir(prec)
    drv = HARNESS / f"drv_{name}.F90"
    exe = CACHE / f"drv_{name}_{prec}.exe"
    deps = [drv, HARNESS / "harness_io.F90"]
    if exe.exists() and all(exe.stat().st_mtime > p.stat().st_mtime for p in deps):
        return exe
    bdir = CACHE / f"build_{name}_{prec}"
    bdir.mkdir(parents=True, exist_ok=True)
    extra = ["-fdefault-real-8", "-fdefault-double-8"] if prec == "r8" else []
    objs = [str(p) for p in sorted(obj.glob("*.o")) if p.name != "cm1.o"]
    cmd = [
        "gfortran",
        "-O1",
        "-ffree-line-length-none",
        "-fallow-argument-mismatch",
        *extra,
        f"-I{obj}",
        f"-J{bdir}",
        "-cpp",
        str(HARNESS / "harness_io.F90"),
        str(drv),
        *objs,
        "-o",
        str(exe),
    ]
    nc = subprocess.run(["nf-config", "--flibs"], capture_output=True, text=True)
    if nc.returncode == 0:
        cmd += nc.stdout.split()
    subprocess.run(cmd, check=True, cwd=bdir)
    return exe


def run_driver(name: str, inputs: dict, outputs: dict, prec: str = "r8") -> dict:
    """inputs: name -> numpy array (Python [k,j,i] order); outputs: name -> shape."""
    exe = build_driver(name, prec)
    dt = DTYPES[prec]
    with tempfile.TemporaryDirectory() as wd:
        for k, v in inputs.items():
            # C order of a [k, j, i] array == Fortran order of a(i, j, k)
            np.ascontiguousarray(v, dtype=dt).tofile(os.path.join(wd, k + ".bin"))
        subprocess.run([str(exe)], check=True, cwd=wd, capture_output=True)
        res = {}
        for k, shape in outputs.items():
            a = np.fromfile(os.path.join(wd, k + ".bin"), dtype=dt)
            res[k] = a.reshape(tuple(shape))
        return res
