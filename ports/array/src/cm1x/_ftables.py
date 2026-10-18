"""Constant tables of CM1's Fortran sources, read at init time with numpy.

RRTMG keeps its cloud-optics, reference-atmosphere and Planck tables as
array-constructor assignments (``absliq1(:, 1) = (/ ... /)``) and ``data``
statements inside initialisation subroutines, and its k-distributions in
the big-endian unformatted files run/RRTMG_LW_DATA and run/RRTMG_SW_DATA.
This module reads both straight from the CM1 tree (``src/`` and ``run/``
next to ``ports/``; ``CM1X_CM1_ROOT`` overrides), so the Python port uses
exactly the numbers the Fortran compiles in.  Host-side setup only: the
results are numpy arrays that the radiation modules convert with
``bk.asarray``.
"""

from __future__ import annotations

import functools
import os
import re
from pathlib import Path

import numpy as np


def cm1_root() -> Path:
    env = os.environ.get("CM1X_CM1_ROOT")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[4]


@functools.lru_cache(maxsize=None)
def _source_lines(fname: str) -> tuple:
    """Fortran statements of src/<fname>: comments stripped, continuation
    lines joined."""
    out, buf = [], ""
    for raw in (cm1_root() / "src" / fname).read_text(errors="replace").split("\n"):
        s = raw.split("!")[0].rstrip()
        if buf:
            s = s.lstrip()
            if s.startswith("&"):
                s = s[1:]
        if s.endswith("&"):
            buf += s[:-1] + " "
            continue
        out.append(buf + s)
        buf = ""
    return tuple(out)


def _num(tok: str) -> float:
    t = tok.strip().lower()
    t = re.sub(r"_(rb|im|r8|r4|8|4)$", "", t)
    return float(t.replace("d", "e"))


def _values(text: str) -> list:
    vals = []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "*" in tok:  # data repeat count  n*value
            n, v = tok.split("*")
            vals += [_num(v)] * int(n)
        else:
            vals.append(_num(tok))
    return vals


def _index(sub: str, lo: int, hi: int):
    sub = sub.strip()
    if sub == ":":
        return slice(None)
    if ":" in sub:
        a, b = sub.split(":")
        a = int(a) if a.strip() else lo
        b = int(b) if b.strip() else hi
        return slice(a - lo, b - lo + 1)
    return int(sub) - lo


def subroutine_body(fname: str, name: str) -> list:
    """Statements of ``subroutine name`` (first match) in src/<fname>."""
    lines = _source_lines(fname)
    start = re.compile(r"^\s*subroutine\s+%s\b" % re.escape(name), re.I)
    end = re.compile(r"^\s*end\s+subroutine\b", re.I)
    for i, s in enumerate(lines):
        if start.match(s):
            body = []
            for t in lines[i + 1 :]:
                if end.match(t):
                    return body
                body.append(t)
    raise KeyError(f"subroutine {name} not found in {fname}")


def assignments(fname: str, sub: str, dims: dict) -> dict:
    """Constant tables assigned in subroutine ``sub`` of src/<fname>.

    ``dims``: {name: ((lo, hi), ...)} Fortran bounds of every table wanted
    (``()`` for a scalar).  Handles ``x = v``, ``x(subs) = (/ ... /)`` and
    ``data x / ... /``; values fill the target section in Fortran order."""
    out = {
        k: np.full(tuple(h - lo + 1 for lo, h in d), np.nan) if d else np.nan
        for k, d in dims.items()
    }
    pat = re.compile(r"^\s*(\w+)\s*(\(([^=]*)\))?\s*=\s*(.*)$")
    dat = re.compile(r"^\s*data\s+(\w+)\s*/(.*)/\s*$", re.I)
    for s in subroutine_body(fname, sub):
        m = dat.match(s)
        if m:
            name, rhs, subs = m.group(1).lower(), m.group(2), None
        else:
            m = pat.match(s)
            if not m:
                continue
            name, subs, rhs = m.group(1).lower(), m.group(3), m.group(4).strip()
            if rhs.startswith("(/"):
                rhs = rhs[2 : rhs.rindex("/)")]
        if name not in dims:
            continue
        vals = _values(rhs)
        d = dims[name]
        if not d:
            out[name] = vals[0]
            continue
        a = out[name]
        if subs is None:
            idx = tuple(slice(None) for _ in d)
        else:
            idx = tuple(_index(x, lo, hi) for x, (lo, hi) in zip(subs.split(","), d))
        tgt = a[idx]
        if np.ndim(tgt) == 0:
            a[idx] = vals[0]
        else:
            a[idx] = np.array(vals).reshape(tgt.shape, order="F")
    for k, v in out.items():
        if np.any(np.isnan(v)):
            raise ValueError(f"{fname}:{sub}: table {k} not fully assigned")
    return out


def data_statement(fname: str, name: str, scope: str | None = None) -> np.ndarray:
    """Values of ``DATA name / ... /`` in src/<fname> (optionally only inside
    subroutine ``scope``)."""
    lines = subroutine_body(fname, scope) if scope else _source_lines(fname)
    pat = re.compile(r"^\s*data\s+%s\s*/(.*)/\s*$" % re.escape(name), re.I)
    for s in lines:
        m = pat.match(s)
        if m:
            return np.array(_values(m.group(1)))
    raise KeyError(f"DATA {name} not found in {fname}")


def read_unformatted(path) -> list:
    """Records (bytes) of a big-endian Fortran sequential unformatted file."""
    raw = Path(path).read_bytes()
    recs, off = [], 0
    while off < len(raw):
        n = int.from_bytes(raw[off : off + 4], "big")
        recs.append(raw[off + 4 : off + 4 + n])
        off += n + 8
    return recs


def unpack_record(rec: bytes, items) -> dict:
    """Split one record into named arrays.  ``items``: [(name, kind, shape)]
    with kind 'f4' (big-endian real*4, returned as float64) or 'i4'; shapes
    are Fortran shapes (column-major)."""
    out, off = {}, 0
    for name, kind, shape in items:
        n = int(np.prod(shape)) if shape else 1
        a = np.frombuffer(rec, dtype=">" + kind, count=n, offset=off)
        off += 4 * n
        a = a.astype(np.float64 if kind == "f4" else np.int64)
        out[name] = a.reshape(shape, order="F") if shape else a[0]
    if off != len(rec):
        raise ValueError(f"record length {len(rec)} != schema {off}")
    return out
