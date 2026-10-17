"""Lateral boundary conditions (periodic only for now; ported from bc.F)."""

from __future__ import annotations


def periodic_xy(bk, a, g, stag_x: bool = False, stag_y: bool = False):
    """Fill the ngxy-wide horizontal halos of ``a`` periodically.

    For a u-staggered array (stag_x) the interior is i = 1..ni+1 in Fortran, and
    CM1 sets u(ni+1) = u(1) periodically; same for v in y.
    """
    xp = bk.xp
    n = g.ngxy
    # x direction (last axis)
    ni = g.ni
    core = a[..., n : n + ni]
    left = core[..., ni - n :]
    right_len = a.shape[-1] - (n + ni)
    right = core[..., :right_len]
    a = xp.concat([left, core, right], axis=-1)
    # y direction
    nj = g.nj
    core = a[..., n : n + nj, :]
    lo = core[..., nj - n :, :]
    hi_len = a.shape[-2] - (n + nj)
    hi = core[..., :hi_len, :]
    a = xp.concat([lo, core, hi], axis=-2)
    return a
