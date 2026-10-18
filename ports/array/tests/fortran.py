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
import re
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


def public_module(text: str) -> str:
    """Source edit for ``build_driver(..., public=...)``: make every entity of
    a Fortran module public (drop the default ``private`` statement, the
    ``PRIVATE ::`` lists and the ``, PRIVATE`` attributes), so that a
    harness driver can call a module's private routines and read its
    constants.  The generated code of the routines is unchanged."""
    out = []
    for line in text.splitlines():
        s = line.strip().lower()
        if s == "private" or (s.startswith("private") and "::" in s):
            continue
        out.append(re.sub(r",\s*private\b", "", line, flags=re.IGNORECASE))
    return "\n".join(out) + "\n"


def variant_obj(stem: str, prec: str, edit=public_module) -> tuple[Path, Path]:
    """Compile ``edit`` applied to CM1's preprocessed ``<stem>.f90`` with the
    harness object flags (build_objs.sh), in its own directory so that its
    ``.mod`` files do not replace CM1's.  Returns (object file, module dir).
    The object has the same symbols as ``<stem>.o`` (plus the ones made
    public) and replaces it when linking a driver."""
    obj = objdir(prec)
    vdir = CACHE / f"variant_{stem}_{prec}"
    o = vdir / f"{stem}.o"
    srcf = obj / f"{stem}.f90"
    if o.exists() and o.stat().st_mtime > srcf.stat().st_mtime:
        return o, vdir
    vdir.mkdir(parents=True, exist_ok=True)
    (vdir / f"{stem}.f90").write_text(edit(srcf.read_text()))
    extra = ["-fdefault-real-8", "-fdefault-double-8"] if prec == "r8" else []
    cmd = [
        "gfortran",
        "-ffree-form",
        "-ffree-line-length-none",
        "-O2",
        "-fallow-argument-mismatch",
        *extra,
        f"-I{obj}",
        "-c",
        f"{stem}.f90",
    ]
    subprocess.run(cmd, check=True, cwd=vdir)
    return o, vdir


def build_driver(name: str, prec: str = "r8", public: tuple = ()) -> Path:
    """Compile and link ``drv_<name>.F90`` against CM1's objects.  ``public``
    names CM1 source stems (e.g. ``("morrison",)``) whose module is linked
    in its ``public_module`` variant (see ``variant_obj``)."""
    obj = objdir(prec)
    drv = HARNESS / f"drv_{name}.F90"
    exe = CACHE / f"drv_{name}_{prec}.exe"
    deps = [drv, HARNESS / "harness_io.F90"]
    variants = {stem: variant_obj(stem, prec) for stem in public}
    deps += [o for o, _ in variants.values()]
    if exe.exists() and all(exe.stat().st_mtime > p.stat().st_mtime for p in deps):
        return exe
    bdir = CACHE / f"build_{name}_{prec}"
    bdir.mkdir(parents=True, exist_ok=True)
    extra = ["-fdefault-real-8", "-fdefault-double-8"] if prec == "r8" else []
    objs = [
        str(variants[p.stem][0]) if p.stem in variants else str(p)
        for p in sorted(obj.glob("*.o"))
        if p.name != "cm1.o"
    ]
    cmd = [
        "gfortran",
        "-O1",
        "-ffree-line-length-none",
        "-fallow-argument-mismatch",
        *extra,
        *[f"-I{d}" for _, d in variants.values()],
        f"-I{obj}",
        f"-J{bdir}",
        "-cpp",
        str(HARNESS / "harness_io.F90"),
        str(drv),
        *objs,
        "-o",
        str(exe),
    ]
    if shutil.which("nf-config"):
        nc = subprocess.run(["nf-config", "--flibs"], capture_output=True, text=True)
        if nc.returncode == 0:
            cmd += nc.stdout.split()
    subprocess.run(cmd, check=True, cwd=bdir)
    return exe


def run_driver(
    name: str, inputs: dict, outputs: dict, prec: str = "r8", public: tuple = ()
) -> dict:
    """inputs: name -> numpy array (Python [k,j,i] order); outputs: name -> shape.
    ``public``: see ``build_driver``."""
    exe = build_driver(name, prec, public)
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
