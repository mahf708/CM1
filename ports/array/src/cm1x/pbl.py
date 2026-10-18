"""The boundary-layer ("mesoscale", cm1setup=2) path of CM1 and the registry
of PBL schemes and surface layers, ported from src/turb.F, src/solve1.F and
src/param.F.

turb.F
  the cm1setup=2 branch of ``sfc_and_turb`` (``sfc_and_turb_pbl``: dissten=0,
  the surface layer, the surface boundary conditions, calcnm, calcdef,
  ``turbparam_horiz`` (horizturb=1), ``turbparam_vert`` (ipbl=2), gettau, the
  PBL scheme, then bcs(upten), bcs(vpten)), ``get_vars``-style inputs of the
  WRF physics (``wrf_inputs``), and the ``dobr09`` block (``br09``, ipbl=2:
  Bryan and Rotunno 2009's simple PBL);
solve1.F
  the ``use_pbl`` blocks (``add_theta_tendency``, ``add_tendencies``);
param.F / init_surface.F
  the option adjustments of the PBL path (``options``).

Registry
--------
``PBL_SCHEMES`` maps ``ipbl`` and ``SURFACE_LAYERS`` maps ``sfcmodel`` to the
module (dotted path) that implements it; adding a scheme is one line in the
dict.  A scheme module exposes

``pbl(ctx) -> dict``             (PBL scheme)   or
``surface_layer(ctx) -> dict``   (surface layer)

and optionally

``options(cfg)``        param.F adjustments, called once by the Model before
                        anything else (e.g. ``cfg.idiss = 0``);
``check(cfg)``          raise NotImplementedError for unported options;
``init_state(ctx) -> dict``   arrays the scheme carries across steps (added to
                        the model state at construction, after init_surface;
                        ``ctx.nstep`` is None there);
``WRF_INPUTS = True``   build ``ctx.w`` (the WRF physics' column inputs);
``DUMP_KEYS``           extra state names the whole-model dump test compares.

``ctx`` is a ``Context``: ``bk, g, cfg, dt``, ``nstep`` (0 in the startup pass,
so ``update_sfc = nstep > 0``), ``rtime``, ``s`` (the model state dict: the
surface arrays, K arrays, PBL tendencies, everything a scheme carries;
updates returned by the earlier stages of this call are already in it), ``b``
(base state), the new time level as CM1 passes it (``u3d, v3d, w3d, pp3d,
th3d, q3d``) and ``rho, rr, rf, prs``, and ``w`` (lazy; see ``wrf_inputs``).
The returned dict is merged into ``ctx.s``.  Shapes: 3-D arrays
``g.s_shape``/``g.w_shape`` ``[k, j, i]`` with halos; surface arrays 2-D
``[j, i]`` with halos (``g.s_shape[1:]``).  A PBL scheme returns the
rho-weighted tendencies ``thpten, qvpten, qcpten, qipten, upten, vpten``
(``g.s_shape``, interior 1..ni x 1..nj x 1..nk filled, upten/vpten at
scalar points; sfc_and_turb_pbl applies bcs to upten/vpten), and may update
``dissten``, ``kmv``/``khv``/``kmh``/``khh``, ``hpbl`` and its own arrays.
A surface layer fills the interior of the surface arrays (``ust, znt, cd,
ch, cq, hfx, qfx, thflux, qvflux, u10, v10, s10, qsfc, wspd, ...``) after the
shared prelude (``psfc, u1, v1, s1, t1``, and ``hpbl`` from gethpbl2 for
ipbl=2) has run; sfc_and_turb_pbl then fills the halos.
"""

from __future__ import annotations

import importlib

from .turbtend import _cat

PBL_SCHEMES = {
    2: "cm1x.pbl",  # CM1 simple scheme (Bryan and Rotunno 2009)
}

SURFACE_LAYERS = {
    1: "cm1x.sfcphys",  # CM1 simple scheme (getcecd, sfcflux, sfcdiags)
    2: "cm1x.sfclay",  # MM5 similarity (sf_sfclay) + SLAB
    3: "cm1x.sfclay",  # revised MM5 (sf_sfclayrev) + SLAB
}


def _opt(cfg, name, default):
    get = getattr(cfg, "get", None)
    return get(name, default) if get else getattr(cfg, name, default)


def scheme(cfg, kind):
    """The registered module for ``kind`` "pbl" (ipbl) or "sfc" (sfcmodel),
    or None when the option is 0."""
    key = "ipbl" if kind == "pbl" else "sfcmodel"
    reg = PBL_SCHEMES if kind == "pbl" else SURFACE_LAYERS
    v = int(_opt(cfg, key, 0))
    if v == 0:
        return None
    if v not in reg:
        raise NotImplementedError(f"{key}={v} is not ported")
    return importlib.import_module(reg[v])


class Context:
    """What a PBL scheme / surface layer gets (see the module docstring)."""

    def __init__(self, bk, g, cfg, *, dt, nstep, rtime, s, b, **fields):
        self.bk, self.g, self.cfg = bk, g, cfg
        self.dt, self.nstep, self.rtime = dt, nstep, rtime
        self.update_sfc = nstep is not None and nstep > 0
        self.s, self.b = s, b
        self._w = None
        for k, v in fields.items():
            setattr(self, k, v)

    @property
    def w(self):
        if self._w is None:
            self._w = wrf_inputs(self)
        return self._w


def options(cfg):
    """param.F / init_surface.F adjustments of the surface and PBL options
    (applied once, in place): bbc=3 with a surface layer or PBL scheme,
    isfcflx=1 for sfcmodel 2..7, tconfig=2 for cm1setup=2, l_inf=0 for
    ipbl 1,3,4,5,6, dohturb/dovturb, use_pbl, dosfcflx; then the scheme
    modules' own ``options(cfg)``."""
    cm1setup = int(_opt(cfg, "cm1setup", 1))
    ipbl = int(_opt(cfg, "ipbl", 0))
    sfcmodel = int(_opt(cfg, "sfcmodel", 0))
    if cm1setup not in (2, 4):
        ipbl = 0
        cfg.ipbl = 0
    if sfcmodel >= 1 or ipbl >= 1 or int(_opt(cfg, "oceanmodel", 1)) == 2:
        cfg.bbc = 3
    if 2 <= sfcmodel <= 7:
        cfg.isfcflx = 1
    isfcflx = int(_opt(cfg, "isfcflx", 0))
    if cm1setup == 2:
        cfg.tconfig = 2
    if cm1setup in (2, 4) and ipbl in (1, 3, 4, 5, 6):
        cfg.l_inf = 0.0
    idoles = cm1setup in (1, 4)
    cfg.dohturb = (
        idoles
        or cm1setup == 3
        or (cm1setup == 2 and int(_opt(cfg, "horizturb", 0)) == 1)
    )
    cfg.dovturb = idoles or cm1setup == 3
    cfg.use_pbl = cm1setup in (2, 4) and ipbl >= 1
    dosfcflx = False
    if cm1setup >= 1 or ipbl == 2:
        if isfcflx == 1 or sfcmodel in (2, 3, 4, 6, 7):
            dosfcflx = True
    if ipbl in (1, 3, 4, 5):
        dosfcflx = False
    if cm1setup == 4 and isfcflx == 1:
        dosfcflx = True
    cfg.dosfcflx = dosfcflx
    for kind in ("sfc", "pbl"):
        mod = scheme(cfg, kind)
        if mod is not None and hasattr(mod, "options"):
            mod.options(cfg)


def _ugr_vgr(cfg, b, ua, va):
    """turb.F ground-relative winds (imove=0): ua (- u0 with pertflx=1)."""
    if int(_opt(cfg, "imove", 0)) == 1:
        raise NotImplementedError("PBL path: imove=1 is not ported")
    if int(_opt(cfg, "pertflx", 0)) == 1:
        return ua - b["u0"], va - b["v0"]
    return ua, va


def wrf_inputs(ctx):
    """turb.F: the column inputs of the WRF physics on the new state (the
    dum1..dum7 / divx / ppten / dum8 arrays of the surface block and of
    get_vars, which use the same formulas): dict of ``g.s_shape`` arrays,
    valid on the interior i=1..ni, j=1..nj, k=1..nk (p3di on k=1..nk+1):

    u, v      ground-relative wind at scalar points (dum1, dum2)
    th        potential temperature th0 + tha (dum3)
    pi        Exner function pi0 + ppi (dum7)
    t         temperature th * pi (dum4)
    dz8w      layer depth dz * rmh (dum5)
    p3di      pressure on w levels (dum6): psfc at k=1, c1/c2 interpolation,
              cgt extrapolation at nk+1 (``g.w_shape``)
    qv, qc, qi   qa(nqv), qa(nqc), qa(nqi), or 0 (divx, ppten, dum8)
    rthraten  0 (CM1 passes zero radiative heating to the PBL schemes)
    """
    bk, g, cfg, b = ctx.bk, ctx.g, ctx.cfg, ctx.b
    xp = bk.xp
    ni, nj, nk, n, z = g.ni, g.nj, g.nk, g.ngxy, g.ngz
    if nk < 3:
        raise NotImplementedError("wrf_inputs: needs nk >= 3")
    K, J, I = slice(z, z + nk), slice(n, n + nj), slice(n, n + ni)
    Jp, Ip = slice(n + 1, n + nj + 1), slice(n + 1, n + ni + 1)
    ugr, vgr = _ugr_vgr(cfg, b, ctx.u3d, ctx.v3d)
    S = g.s_shape
    zs = bk.zeros(S)

    def put(v):
        return bk.set(zs, (K, J, I), v)

    th = b["th0"][K, J, I] + ctx.th3d[K, J, I]
    pi = b["pi0"][K, J, I] + ctx.pp3d[K, J, I]
    prs = ctx.prs
    c1, c2 = b["c1"], b["c2"]
    Kw = slice(z + 1, z + nk)  # k = 2..nk
    Km = slice(z, z + nk - 1)
    p = (c1[Kw, J, I] * prs[Km, J, I]) + (c2[Kw, J, I] * prs[Kw, J, I])
    ptop = (
        b["cgt1"] * prs[z + nk - 1, J, I]
        + b["cgt2"] * prs[z + nk - 2, J, I]
        + b["cgt3"] * prs[z + nk - 3, J, I]
    )
    p3di = _cat(xp, ctx.s["psfc"][J, I][None], p, ptop[None])
    out = {
        "u": put(0.5 * (ugr[K, J, I] + ugr[K, J, Ip])),
        "v": put(0.5 * (vgr[K, J, I] + vgr[K, Jp, I])),
        "th": put(th),
        "pi": put(pi),
        "t": put(th * pi),
        "dz8w": put(float(cfg.dz) * b["rmh"][K, J, I]),
        "p3di": bk.set(bk.zeros(g.w_shape), (slice(z, z + nk + 1), J, I), p3di),
        "rthraten": zs,
    }
    q = ctx.q3d
    imoist = int(_opt(cfg, "imoist", 0)) == 1
    for name, idx in (("qv", "nqv"), ("qc", "nqc"), ("qi", "nqi")):
        m = int(_opt(cfg, idx, 0)) if imoist else 0
        out[name] = put(q[m - 1][K, J, I]) if m >= 1 else zs
    return out
