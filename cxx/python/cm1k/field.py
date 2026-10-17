"""FieldView: a numpy array plus its Fortran index bounds.

CM1 arrays carry Fortran bounds, e.g. ``ua(ib:ie+1, jb:je, kb:ke)`` with
``ib = 1 - ngxy`` (halo points). A :class:`FieldView` keeps the plain
(zero-based, Fortran-ordered) numpy array in ``.data`` and the Fortran lower
bounds in ``.lbounds`` so that indices can be translated either way::

    fv = model.field("ua")
    fv.lbounds, fv.ubounds        # ((-2, -2, 0), (36, 35, 21)) for a 32x32x20 grid with 3 halo points
    fv.at(1, 1, 1)                # ua(1,1,1), Fortran indices
    fv.f[1:33, 1, :]              # Fortran-origin indexing (Python slice semantics: stop exclusive)
    fv[0, 0, 0]                   # plain numpy (zero-based) indexing
    np.asarray(fv)                # the array itself

This module is pure numpy: it is used by :mod:`cm1k.dump` without the C++
extension.
"""

from __future__ import annotations

import numpy as np

__all__ = ["FieldView"]


class _FortranIndexer:
    __slots__ = ("_fv",)

    def __init__(self, fv):
        self._fv = fv

    def _convert(self, key):
        fv = self._fv
        if not isinstance(key, tuple):
            key = (key,)
        if any(k is Ellipsis for k in key):
            raise IndexError("FieldView.f does not support Ellipsis")
        if len(key) > fv.ndim:
            raise IndexError(f"too many indices for a rank-{fv.ndim} field")
        out = []
        for d, k in enumerate(key):
            lb, ub = fv.lbounds[d], fv.ubounds[d]
            if isinstance(k, slice):
                start = lb if k.start is None else k.start
                stop = ub + 1 if k.stop is None else k.stop
                if not (lb <= start <= ub + 1 and lb <= stop <= ub + 1):
                    raise IndexError(f"slice {k} out of bounds {lb}:{ub} in dimension {d + 1}")
                out.append(slice(start - lb, stop - lb, k.step))
            else:
                k = int(k)
                if not lb <= k <= ub:
                    raise IndexError(f"index {k} out of bounds {lb}:{ub} in dimension {d + 1}")
                out.append(k - lb)
        return tuple(out)

    def __getitem__(self, key):
        return self._fv.data[self._convert(key)]

    def __setitem__(self, key, value):
        self._fv.data[self._convert(key)] = value


class FieldView:
    """A numpy array with Fortran lower bounds.

    Attributes
    ----------
    name : str
    data : numpy.ndarray
        Fortran-ordered array (zero-based numpy indexing). For a field obtained
        from a :class:`cm1k.Model` with ``copy=False`` on a host-accessible
        backend this is a zero-copy view of the model's memory: writing to it
        changes the model state.
    lbounds, ubounds : tuple of int
        Fortran (inclusive) bounds of each dimension.
    is_view : bool
        True when ``data`` aliases the model's memory.
    """

    __slots__ = ("name", "data", "lbounds", "is_view")

    def __init__(self, name, data, lbounds=None, is_view=False):
        data = np.asarray(data)
        if lbounds is None:
            lbounds = (1,) * data.ndim
        lbounds = tuple(int(x) for x in lbounds)
        if len(lbounds) != data.ndim:
            raise ValueError(f"{name}: {len(lbounds)} lower bounds for a rank-{data.ndim} array")
        self.name = name
        self.data = data
        self.lbounds = lbounds
        self.is_view = bool(is_view)

    @property
    def ubounds(self):
        return tuple(lb + n - 1 for lb, n in zip(self.lbounds, self.data.shape))

    @property
    def bounds(self):
        """((lb1, ub1), (lb2, ub2), ...)"""
        return tuple(zip(self.lbounds, self.ubounds))

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def f(self):
        """Indexer using Fortran index origin (slices keep Python stop-exclusive semantics)."""
        return _FortranIndexer(self)

    def index(self, *fidx):
        """Zero-based numpy index tuple for Fortran indices ``fidx``."""
        return _FortranIndexer(self)._convert(tuple(fidx))

    def at(self, *fidx):
        """Element at Fortran indices, e.g. ``fv.at(i, j, k)``."""
        return self.data[self.index(*fidx)]

    def copy(self):
        return FieldView(self.name, self.data.copy(order="F"), self.lbounds, is_view=False)

    def __array__(self, dtype=None, copy=None):
        if copy:
            return np.array(self.data, dtype=dtype, copy=True)
        return np.asarray(self.data, dtype=dtype)

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        b = ", ".join(f"{lb}:{ub}" for lb, ub in self.bounds)
        kind = "view" if self.is_view else "copy"
        return f"FieldView({self.name}({b}), {self.dtype}, {kind})"
