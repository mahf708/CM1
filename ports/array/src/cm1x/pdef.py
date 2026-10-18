"""Positive-definite moisture fixers.

``pdefq`` is ported from ``src/misclibs.F`` (it is called from mp_driver.F and
cm1.F); it lives here with the other positive-definite routines of pdef.F.

Index conventions follow cm1x.grid: arrays are ``a[k, j, i]`` with halos.
"""

from __future__ import annotations

_DEFAULTS = dict(pdscheme=1)  # param.F: "cm1r19: hard-wire pdscheme = 1"


def _opt(cfg, name):
    if cfg is None:
        return _DEFAULTS[name]
    # dict.get / Config.get: an unset option falls back to this module's
    # default rather than Config's zero for unknown names
    get = getattr(cfg, "get", None)
    return get(name, _DEFAULTS[name]) if get else getattr(cfg, name, _DEFAULTS[name])


def _hbcast(a, axis):
    """1-D horizontal array (ruh over i, rvh over j) -> broadcastable [k,j,i]."""
    if a.ndim != 1:
        return a
    return a[None, None, :] if axis == "i" else a[None, :, None]


def pdefq(bk, g, cfg=None, *, rmax, asq, ruh, rvh, rmh, rho, q3d):
    """misclibs.F ``pdefq``: zero values below ``rmax`` and (pdscheme=1)
    rescale each column to conserve its mass.

    ``ruh``/``rvh`` are 1-D over the scalar i/j points (with halos); ``rmh``,
    ``rho``, ``q3d`` are ``g.s_shape``.  Only the interior of ``q3d`` changes.

    Returns ``{"q3d": ..., "asq": asq + dmass}`` (asq a 0-d array).
    """
    xp = bk.xp
    K, J, I = g.K, g.J, g.I
    pdscheme = int(_opt(cfg, "pdscheme"))
    tem = g.dx * g.dy * g.dz

    q = q3d[K, J, I]
    r = rho[K, J, I]
    w = _hbcast(ruh, "i")[..., I] * _hbcast(rvh, "j")[..., J, :] * rmh[K, J, I]
    rmax = float(rmax)

    if pdscheme == 1:
        rq = r * q
        t1 = xp.sum(rq, axis=0)
        a1 = xp.sum(rq * w, axis=0)
        qz = xp.where(q < rmax, xp.zeros_like(q), q)
        t2 = xp.sum(r * qz, axis=0)
        t3 = (t1 + 1.0e-20) / (t2 + 1.0e-20)
        t3 = xp.where(t3 < 0.0, xp.ones_like(t3), t3)
        qn = t3[None, :, :] * qz
        a2 = xp.sum(r * qn * w, axis=0)
        dmass = xp.sum(a2 - a1)
    else:
        a1 = r * q * w
        qn = xp.where(q < rmax, xp.zeros_like(q), q)
        a2 = r * qn * w
        dmass = xp.sum(a2 - a1)

    out = bk.set(bk.copy(q3d), (K, J, I), qn)
    return {"q3d": out, "asq": asq + dmass * tem}
