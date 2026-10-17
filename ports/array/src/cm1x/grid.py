"""Grid layout (uniform, no terrain), mirroring CM1's index conventions.

Fortran CM1 dimensions every 3D scalar ``a(ib:ie, jb:je, kb:ke)`` with
``ib = 1-ngxy``, ``ie = ni+ngxy``, ``kb = 1-ngz``, ``ke = nk+ngz`` (ngxy=3, ngz=1).
Staggered fields add one point: u in i, v in j, w in k.

cm1x stores the same data C-ordered as ``a[k, j, i]`` and keeps the halos, so a
Fortran index maps to a Python index by a fixed shift:

    python_i = fortran_i + ngxy - 1      (same for j)
    python_k = fortran_k + ngz  - 1

Interior scalar points are ``a[K, J, I]`` with the slices exported below.
"""

from __future__ import annotations

from dataclasses import dataclass, field

NGXY = 3
NGZ = 1


def fi(i: int) -> int:
    """Fortran i/j index -> Python index."""
    return i + NGXY - 1


def fk(k: int) -> int:
    """Fortran k index -> Python index."""
    return k + NGZ - 1


@dataclass(frozen=True)
class Grid:
    ni: int
    nj: int
    nk: int
    dx: float
    dy: float
    dz: float
    ngxy: int = NGXY
    ngz: int = NGZ

    # interior slices (scalar points)
    @property
    def I(self):  # noqa: E743
        return slice(self.ngxy, self.ngxy + self.ni)

    @property
    def J(self):
        return slice(self.ngxy, self.ngxy + self.nj)

    @property
    def K(self):
        return slice(self.ngz, self.ngz + self.nk)

    # interior slices including the extra staggered point
    @property
    def IF(self):
        return slice(self.ngxy, self.ngxy + self.ni + 1)

    @property
    def JF(self):
        return slice(self.ngxy, self.ngxy + self.nj + 1)

    @property
    def KF(self):
        return slice(self.ngz, self.ngz + self.nk + 1)

    # array shapes
    @property
    def s_shape(self):
        return (self.nk + 2 * self.ngz, self.nj + 2 * self.ngxy, self.ni + 2 * self.ngxy)

    @property
    def u_shape(self):
        k, j, i = self.s_shape
        return (k, j, i + 1)

    @property
    def v_shape(self):
        k, j, i = self.s_shape
        return (k, j + 1, i)

    @property
    def w_shape(self):
        k, j, i = self.s_shape
        return (k + 1, j, i)

    @property
    def rdx(self):
        return 1.0 / self.dx

    @property
    def rdy(self):
        return 1.0 / self.dy

    @property
    def rdz(self):
        return 1.0 / self.dz

    # 1-D coordinates (python-index aligned, include halos), as plain floats
    def zh(self):
        return [(k - self.ngz + 0.5) * self.dz for k in range(self.s_shape[0])]

    def zf(self):
        return [(k - self.ngz) * self.dz for k in range(self.w_shape[0])]

    def xh(self):
        return [(i - self.ngxy + 0.5) * self.dx for i in range(self.s_shape[2])]

    def yh(self):
        return [(j - self.ngxy + 0.5) * self.dy for j in range(self.s_shape[1])]
