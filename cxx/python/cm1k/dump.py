"""Read and write CM1 reference dumps with numpy only (no C++ needed).

Format (written by ``src/cm1dump.F`` when ``CM1_DUMP_DIR`` is set, and by
``cm1k.Model.write_dump``)::

    <root>/step_NNNNNN/index.txt     "#step_time <nstep> <mtime>", then one line
                                     per array: name kind ndims lb1 ub1 [lb2 ub2 ...]
    <root>/step_NNNNNN/<name>.bin    raw values (kind 8: float64, kind 4: float32),
                                     Fortran order, halos included

Example::

    from cm1k import dump
    ref = dump.Dump(dump.step_dir("refs/dry_bubble", 10))
    tha = ref["tha"]                       # FieldView (array + Fortran bounds)
    stats = dump.compare("out/step_000010", ref, names=["ua", "va", "wa", "tha"])
    for name, s in stats.items():
        print(name, s.max_abs, s.rel, s.where)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field as _dc_field
from typing import Dict, Iterable, Mapping, Optional, Tuple

import numpy as np

from .field import FieldView

__all__ = [
    "DumpEntry",
    "Dump",
    "DiffStats",
    "step_dir",
    "list_steps",
    "read_index",
    "read_field",
    "read",
    "write",
    "compare",
    "diff_fields",
]

_DTYPES = {8: np.dtype("<f8"), 4: np.dtype("<f4")}


def step_dir(root, nstep: int) -> str:
    """``root/step_NNNNNN``"""
    return os.path.join(os.fspath(root), f"step_{int(nstep):06d}")


def list_steps(root) -> list:
    """Sorted step numbers that have a dump (``step_NNNNNN/index.txt``) under ``root``."""
    steps = []
    for d in os.listdir(root):
        m = re.fullmatch(r"step_(\d+)", d)
        if m and os.path.isfile(os.path.join(root, d, "index.txt")):
            steps.append(int(m.group(1)))
    return sorted(steps)


@dataclass(frozen=True)
class DumpEntry:
    name: str
    kind: int
    lbounds: Tuple[int, ...]
    ubounds: Tuple[int, ...]

    @property
    def shape(self):
        return tuple(ub - lb + 1 for lb, ub in zip(self.lbounds, self.ubounds))

    @property
    def ndim(self):
        return len(self.lbounds)

    @property
    def dtype(self):
        if self.kind not in _DTYPES:
            raise ValueError(f"dump '{self.name}': unsupported kind {self.kind}")
        return _DTYPES[self.kind]


def read_index(path) -> Tuple[int, float, Dict[str, DumpEntry]]:
    """Parse ``path/index.txt``; returns ``(nstep, mtime, {name: DumpEntry})``."""
    nstep, mtime = 0, 0.0
    entries: Dict[str, DumpEntry] = {}
    with open(os.path.join(os.fspath(path), "index.txt")) as f:
        for line in f:
            tok = line.split()
            if not tok:
                continue
            if tok[0] == "#step_time":
                nstep, mtime = (
                    int(tok[1]),
                    float(tok[2].replace("D", "E").replace("d", "e")),
                )
                continue
            name, kind, nd = tok[0], int(tok[1]), int(tok[2])
            b = [int(x) for x in tok[3 : 3 + 2 * nd]]
            if len(b) != 2 * nd:
                raise ValueError(f"bad dump index line: {line!r}")
            entries[name] = DumpEntry(name, kind, tuple(b[0::2]), tuple(b[1::2]))
    return nstep, mtime, entries


class Dump:
    """One dump directory (``step_NNNNNN``); fields are read lazily.

    ``dump[name]`` / ``dump.field(name)`` return a :class:`~cm1k.field.FieldView`.
    Has the same ``field_names()`` / ``field(name)`` interface as
    :class:`cm1k.Model`, so either can be passed to :func:`compare`.
    """

    def __init__(self, path, nstep: Optional[int] = None):
        path = os.fspath(path)
        if nstep is not None:
            path = step_dir(path, nstep)
        self.path = path
        self.nstep, self.mtime, self.entries = read_index(path)
        self._cache: Dict[str, FieldView] = {}

    def field_names(self):
        return list(self.entries)

    names = field_names

    def __contains__(self, name):
        return name in self.entries

    def __iter__(self):
        return iter(self.entries)

    def __len__(self):
        return len(self.entries)

    def field(self, name: str, copy: bool = False) -> FieldView:
        if name not in self.entries:
            raise KeyError(f"'{name}' not in dump {self.path}")
        if name not in self._cache:
            e = self.entries[name]
            fn = os.path.join(self.path, name + ".bin")
            data = np.fromfile(fn, dtype=e.dtype)
            n = int(np.prod(e.shape))
            if data.size != n:
                raise ValueError(f"{fn}: expected {n} values, found {data.size}")
            arr = data.astype(np.float64, copy=False).reshape(e.shape, order="F")
            self._cache[name] = FieldView(name, arr, e.lbounds)
        fv = self._cache[name]
        return fv.copy() if copy else fv

    __getitem__ = field

    def to_dict(self) -> Dict[str, FieldView]:
        return {n: self.field(n) for n in self.entries}

    def __repr__(self):
        return f"Dump({self.path!r}, nstep={self.nstep}, mtime={self.mtime}, {len(self.entries)} fields)"


def read(path, nstep: Optional[int] = None) -> Dump:
    """Open a dump directory (or ``root`` and ``nstep``)."""
    return Dump(path, nstep)


def read_field(path, name: str) -> FieldView:
    return Dump(path).field(name)


def _as_fieldview(name, value) -> FieldView:
    if isinstance(value, FieldView):
        return value
    if isinstance(value, tuple) and len(value) == 2:
        return FieldView(name, value[0], value[1])
    return FieldView(name, value)  # lower bounds default to 1


def write(path, fields: Mapping, nstep: int = 0, mtime: float = 0.0) -> str:
    """Write a dump directory ``path`` (created if needed) in the reference format.

    ``fields`` maps names to :class:`FieldView`, ``(array, lbounds)`` tuples or
    plain arrays (lower bounds 1). Values are written as float64.
    Returns ``path``.
    """
    path = os.fspath(path)
    os.makedirs(path, exist_ok=True)
    lines = [f"#step_time {int(nstep):10d} {float(mtime):24.16E}\n"]
    for name, value in fields.items():
        fv = _as_fieldview(name, value)
        a = np.asarray(fv.data, dtype="<f8")
        if a.ndim < 1 or a.ndim > 9:
            raise ValueError(f"{name}: cannot dump a rank-{a.ndim} array")
        a.ravel(order="F").tofile(os.path.join(path, name + ".bin"))
        b = "".join(f" {lb:6d} {ub:6d}" for lb, ub in zip(fv.lbounds, fv.ubounds))
        lines.append(f"{name}  8 {a.ndim}{b}\n")
    with open(os.path.join(path, "index.txt"), "w") as f:
        f.writelines(lines)
    return path


@dataclass
class DiffStats:
    """Same definition as ``cm1k::diff`` in C++."""

    max_abs: float = 0.0
    ref_max: float = 0.0
    rel: float = 0.0
    where: Tuple[int, ...] = ()  # Fortran indices of the max abs difference
    bounds_match: bool = True
    extra: dict = _dc_field(default_factory=dict)


def _source(x):
    if isinstance(x, (str, os.PathLike)):
        return Dump(x)
    return x


def diff_fields(a: FieldView, ref: FieldView, region=None) -> DiffStats:
    """Max abs / relative difference of ``a`` against ``ref``.

    ``region`` optionally restricts the comparison to Fortran bounds
    ``((lo1, hi1), (lo2, hi2), ...)`` (inclusive); by default the overlap of
    the two arrays' bounds is compared.
    """
    a, ref = _as_fieldview("a", a), _as_fieldview("ref", ref)
    if a.ndim != ref.ndim:
        raise ValueError(f"rank mismatch: {a.ndim} vs {ref.ndim}")
    match = a.lbounds == ref.lbounds and a.ubounds == ref.ubounds
    if region is None:
        region = [
            (max(l1, l2), min(u1, u2))
            for (l1, u1), (l2, u2) in zip(a.bounds, ref.bounds)
        ]
    sa = tuple(slice(lo - lb, hi - lb + 1) for (lo, hi), lb in zip(region, a.lbounds))
    sr = tuple(slice(lo - lb, hi - lb + 1) for (lo, hi), lb in zip(region, ref.lbounds))
    x, r = np.asarray(a.data)[sa], np.asarray(ref.data)[sr]
    s = DiffStats(bounds_match=match)
    if x.size == 0:
        return s
    d = np.abs(x - r)
    d = np.where(np.isnan(d), np.inf, d)
    idx = np.unravel_index(int(np.argmax(d)), d.shape)
    s.max_abs = float(d[idx])
    s.ref_max = float(np.max(np.abs(r)))
    s.rel = s.max_abs / s.ref_max if s.ref_max > 0 else s.max_abs
    s.where = tuple(int(i) + lo for i, (lo, _) in zip(idx, region))
    return s


def compare(a, ref, names: Optional[Iterable[str]] = None) -> Dict[str, DiffStats]:
    """Compare two runs field by field.

    ``a`` and ``ref`` may be dump directories, :class:`Dump` objects or
    :class:`cm1k.Model` instances (anything with ``field_names()`` and
    ``field(name)``). By default every field present in both is compared.
    """
    a, ref = _source(a), _source(ref)
    if names is None:
        rn = set(ref.field_names())
        names = [n for n in a.field_names() if n in rn]
    return {n: diff_fields(a.field(n), ref.field(n)) for n in names}
