"""cm1k: Python bindings for the C++/Kokkos port of CM1.

Quick start::

    import cm1k
    nl = cm1k.Namelist.from_file("namelist.input")
    nl["dtl"] = 3.0                          # override a namelist value
    m = cm1k.Model(nl)                       # initializes Kokkos on demand
    m.init_from_dump("refs/dry_bubble/step_000000")
    ua = m.field("ua")                       # FieldView: numpy array + Fortran bounds
    ua.f[1, 1, 1] += 1.0                     # zero-copy on host backends
    m.step(10)
    m.write_dump("out")                      # out/step_000010

``cm1k.dump`` (pure numpy) reads and writes the reference dump format and
works without the compiled extension.
"""

from __future__ import annotations

import atexit as _atexit
import os as _os

from . import dump
from .field import FieldView

__all__ = [
    "Model",
    "Namelist",
    "FieldView",
    "dump",
    "initialize",
    "finalize",
    "is_initialized",
    "is_finalized",
    "execution_space",
    "memory_space",
    "host_accessible",
    "concurrency",
    "kokkos_configuration",
    "step_dir",
]

try:
    from . import _cm1k
except ImportError as _e:   # allow `cm1k.dump` / `cm1k.field` without the C++ extension
    _cm1k = None
    _import_error = _e

    def __getattr__(name):
        if name in __all__:
            raise ImportError(
                f"cm1k.{name} needs the compiled extension cm1k._cm1k, which failed to import: {_import_error}"
            )
        raise AttributeError(name)

else:
    from ._cm1k import (  # noqa: F401
        concurrency,
        execution_space,
        finalize,
        host_accessible,
        is_finalized,
        is_initialized,
        kokkos_configuration,
        memory_space,
        step_dir,
    )

    def initialize(num_threads: int = -1, device_id: int = -1, args=()):
        """Initialize Kokkos.

        Idempotent: returns True if this call initialized Kokkos, False if it
        was already initialized (settings are then ignored). Models initialize
        Kokkos on demand, so calling this is only needed to choose settings,
        e.g. ``cm1k.initialize(num_threads=4)`` or
        ``cm1k.initialize(args=["--kokkos-device-id=1"])``. Kokkos is
        finalized automatically at interpreter exit.
        """
        return _cm1k.initialize(num_threads=num_threads, device_id=device_id, args=list(args))

    # ------------------------------------------------------------ Namelist
    Namelist = _cm1k.Namelist

    _LOGICAL = {".true.": True, ".false.": False, "t": True, "f": False,
                ".t.": True, ".f.": False, "true": True, "false": False}

    def _parse_value(v: str):
        s = v.strip()
        if len(s) >= 2 and s[0] in "'\"" and s[-1] == s[0]:
            return s[1:-1]
        low = s.lower()
        if low in _LOGICAL:
            return _LOGICAL[low]
        try:
            return int(s)
        except ValueError:
            pass
        try:
            return float(low.replace("d", "e"))
        except ValueError:
            return s

    def _expand(raw):
        out = []
        for v in raw:
            m = None
            if "*" in v and not (v[:1] in "'\""):
                n, _, val = v.partition("*")
                if n.strip().isdigit():
                    m = [val] * int(n)
            out.extend(m if m is not None else [v])
        return out

    def _format_value(v) -> str:
        if isinstance(v, bool):
            return ".true." if v else ".false."
        if isinstance(v, int):
            return str(v)
        if isinstance(v, float):
            return repr(v)
        if isinstance(v, str):
            return "'" + v.replace("'", "''") + "'"
        if hasattr(v, "item"):   # numpy scalar
            return _format_value(v.item())
        raise TypeError(f"cannot convert {type(v).__name__} to a Fortran namelist value")

    _MISSING = object()

    def _nl_get(self, key, default=None):
        """Value of ``key`` converted to Python (int, float, bool, str, or a list of them).

        Returns ``default`` if the key is not set.
        """
        if not self.has(key):
            return default
        vals = [_parse_value(v) for v in _expand(self.getv(key))]
        return vals[0] if len(vals) == 1 else vals

    def _nl_getitem(self, key):
        v = _nl_get(self, key, _MISSING)
        if v is _MISSING:
            raise KeyError(key)
        return v

    def _nl_set(self, key, value):
        """Set ``key`` from a Python value (bool -> .true., str -> quoted, list -> comma list).

        Use ``set_raw(key, "fortran spelling")`` to pass the Fortran text as is.
        """
        if isinstance(value, (list, tuple)):
            text = ", ".join(_format_value(v) for v in value)
        else:
            text = _format_value(value)
        self.set_raw(key, text)

    def _nl_to_dict(self):
        return {k: _nl_get(self, k) for k in self.keys()}

    def _nl_repr(self):
        return f"Namelist({len(self.keys())} keys)"

    Namelist.get = _nl_get
    Namelist.set = _nl_set
    Namelist.__getitem__ = _nl_getitem
    Namelist.__setitem__ = _nl_set
    Namelist.to_dict = _nl_to_dict
    Namelist.__repr__ = _nl_repr

    # --------------------------------------------------------------- Model
    class Model(_cm1k.Model):
        """One CM1 domain (cm1k::Model).

        Parameters
        ----------
        namelist : Namelist or path-like
            The namelist (copied) or a path to a ``namelist.input``.
        **overrides
            Namelist values to override, e.g. ``Model(nl, dtl=3.0, nx=16)``.
        """

        def __init__(self, namelist, **overrides):
            if isinstance(namelist, (str, _os.PathLike)):
                namelist = Namelist.from_file(_os.fspath(namelist))
            elif not isinstance(namelist, Namelist):
                raise TypeError("Model(namelist): expected a cm1k.Namelist or a path")
            if overrides:
                namelist = namelist.copy()
                for k, v in overrides.items():
                    namelist[k] = v
            super().__init__(namelist)

        def init_from_dump(self, path, nstep=None):
            """Load grid, base state and prognostic fields from a dump directory.

            ``path`` is a ``step_NNNNNN`` directory, or a dump root when
            ``nstep`` is given.
            """
            path = _os.fspath(path)
            if nstep is not None:
                path = step_dir(path, int(nstep))
            super().init_from_dump(path)
            return self

        def field(self, name: str, copy: bool = False) -> FieldView:
            """Field ``name`` as a :class:`FieldView` (array + Fortran bounds).

            With ``copy=False`` on a host-accessible backend the array is a
            zero-copy view of the model's memory; otherwise it is a copy.
            """
            arr = self.get_field(name, copy=copy)
            lb, _ = self.field_bounds(name)
            return FieldView(name, arr, lb, is_view=(not copy and host_accessible()))

        def fields(self, copy: bool = False):
            """``{name: FieldView}`` for every field."""
            return {n: self.field(n, copy=copy) for n in self.field_names()}

        def __getitem__(self, name):
            return self.get_field(name)

        def __setitem__(self, name, value):
            self.set_field(name, value.data if isinstance(value, FieldView) else value)

        def __contains__(self, name):
            return self.has_field(name)

        def __repr__(self):
            if self.released:
                return "<cm1k.Model (released)>"
            b = self.bounds
            return (f"<cm1k.Model {b['nx']}x{b['ny']}x{b['nz']} nstep={self.nstep} "
                    f"mtime={self.mtime} dt={self.dt} on {execution_space()}>")

    _atexit.register(finalize)
