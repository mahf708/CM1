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
    3: "cm1x.pbl_gfs",  # GFS-EDMF (as configured in HWRF)
    6: "cm1x.pbl_myj",  # Mellor-Yamada-Janjic
}

SURFACE_LAYERS = {
    1: "cm1x.sfcphys",  # CM1 simple scheme (getcecd, sfcflux, sfcdiags)
    2: "cm1x.sfclay",  # MM5 similarity (sf_sfclay) + SLAB
    3: "cm1x.sfclay",  # revised MM5 (sf_sfclayrev) + SLAB
    4: "cm1x.sfc_gfdl",  # GFDL surface layer (over water)
    7: "cm1x.sfc_myj",  # Mellor-Yamada-Janjic (MYJSFC) + SLAB
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


def apply_options(cfg):
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
        return bk.set(bk.zeros(S), (K, J, I), v)

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


# ---------------------------------------------------------------------------
# surface arrays


# cm1.F: every surface array is 0 at allocation, then these
SFC_DEFAULTS = {
    "znt": 1.0e-6,
    "zntmp": 1.0e-6,
    "ust": 1.0e-6,
    "tsk": 300.0,
    "psfc": 100000.0,
    "qsfc": 0.00001,
    "hpbl": 100.0,
}
SFC_KEYS = (
    "tsk", "znt", "zntmp", "xland", "mavail", "ust", "u1", "v1", "s1", "t1",
    "cd", "ch", "cq", "thflux", "qvflux", "psfc", "u10", "v10", "s10", "hfx",
    "qfx", "qsfc", "wspd", "hpbl", "dsxy", "br", "cda", "gz1oz0", "psim",
    "psih", "fm", "fh", "zol", "mol", "rmol", "th2", "t2", "q2", "cpmm",
)  # fmt: skip
# the 3-D arrays of the PBL path (cm1.F, use_pbl)
PBL_TENDENCIES = ("thpten", "qvpten", "qcpten", "qipten", "upten", "vpten")


def interior2(g, a):
    """Interior [nj, ni] of a 2-D [j, i] array with halos."""
    n = g.ngxy
    return a[n : n + g.nj, n : n + g.ni]


def put2(bk, g, a, v):
    """``a`` with its interior (i=1..ni, j=1..nj) set to ``v``."""
    n = g.ngxy
    return bk.set(a, (slice(n, n + g.nj), slice(n, n + g.ni)), v)


def init_surface_state(bk, g, cfg, base):
    """The surface arrays of the model state (sfcmodel >= 1): cm1.F's
    allocation values (SFC_DEFAULTS, else 0), then init_surface.F (initsfc=1:
    tsk, znt (set_znt: cnst_znt), mavail (1 over water), xland, dsxy).
    Returns a dict of 2-D [j, i] arrays with halos."""
    from .init_surface import getlanduse

    S = g.s_shape[1:]
    z = bk.zeros(S)
    out = {k: z + SFC_DEFAULTS.get(k, 0.0) for k in SFC_KEYS}
    lu = getlanduse(int(cfg.season), int(cfg.lu0))
    out["tsk"] = z + float(cfg.tsk0)
    znt = lu["znt"]
    if int(_opt(cfg, "set_znt", 0)) == 1:
        znt = max(1.0e-12, float(_opt(cfg, "cnst_znt", 0.16)))
    out["znt"] = z + znt
    out["xland"] = z + lu["xland"]
    out["mavail"] = z + (1.0 if lu["xland"] > 1.5 else lu["mavail"])
    dx, dy = float(cfg.dx), float(cfg.dy)
    xp = bk.xp
    ruh = base["ruh"][None, :]
    rvh = base["rvh"][:, None]
    if int(_opt(cfg, "axisymm", 0)) == 1 or int(cfg.ny) == 1:
        dsxy = z + dx * ruh
    elif int(cfg.nx) == 1:
        dsxy = z + dy * rvh
    else:
        dsxy = xp.maximum(z + dx * ruh, z + dy * rvh)
    out["dsxy"] = dsxy
    return out


def sfc_prelude(ctx):
    """turb.F sfc_and_turb, the surface block every surface layer shares:
    psfc (cgs extrapolation), u1, v1, s1, t1 at k=1 (ground-relative winds)
    and, for ipbl=2, gethpbl2 (qsfc, hpbl).  Returns state updates."""
    from .sfcphys import gethpbl2

    bk, g, cfg, b, s = ctx.bk, ctx.g, ctx.cfg, ctx.b, ctx.s
    ni, nj, nk, n, z = g.ni, g.nj, g.nk, g.ngxy, g.ngz
    J, I = slice(n, n + nj), slice(n, n + ni)
    Jp, Ip = slice(n + 1, n + nj + 1), slice(n + 1, n + ni + 1)
    prs = ctx.prs
    psfc = (
        b["cgs1"] * prs[z, J, I]
        + b["cgs2"] * prs[z + 1, J, I]
        + b["cgs3"] * prs[z + 2, J, I]
    )
    ugr, vgr = _ugr_vgr(cfg, b, ctx.u3d, ctx.v3d)
    u1 = 0.5 * (ugr[z, J, I] + ugr[z, J, Ip])
    v1 = 0.5 * (vgr[z, J, I] + vgr[z, Jp, I])
    s1 = bk.xp.sqrt(u1 * u1 + v1 * v1)
    t1 = b["th0"][z, J, I] + ctx.th3d[z, J, I]
    out = {}
    for k, v in (("psfc", psfc), ("u1", u1), ("v1", v1), ("s1", s1), ("t1", t1)):
        out[k] = put2(bk, g, s[k], v)
    if int(_opt(cfg, "testcase", 0)) != 0:
        raise NotImplementedError("sfc_prelude: testcase /= 0 is not ported")
    if int(_opt(cfg, "ipbl", 0)) == 2:
        K = slice(z, z + nk)
        nqv = int(_opt(cfg, "nqv", 0))
        th = b["th0"][K, J, I] + ctx.th3d[K, J, I]
        if int(_opt(cfg, "imoist", 0)) == 1 and nqv >= 1:
            qv = ctx.q3d[nqv - 1][K, J, I]
        else:
            qv = bk.xp.zeros_like(th)
        r = gethpbl2(
            bk,
            cfg,
            psfc=psfc,
            tsk=interior2(g, s["tsk"]),
            thflux=interior2(g, s["thflux"]),
            qvflux=interior2(g, s["qvflux"]),
            ust=interior2(g, s["ust"]),
            hpbl=interior2(g, s["hpbl"]),
            zh=b["zh"][K, J, I],
            th=th,
            qv=qv,
            ugr=ugr[K, J, slice(n, n + ni + 1)],
            vgr=vgr[K, slice(n, n + nj + 1), I],
        )
        out["qsfc"] = put2(bk, g, s["qsfc"], r["qsfc"])
        # hpbl = -1.0 on the whole array (halos included)
        out["hpbl"] = put2(bk, g, bk.zeros(g.s_shape[1:]) - 1.0, r["hpbl"])
    return out


def sfc_bcs(ctx):
    """turb.F "bc/comms" (bbc=3): bc2d of ust, u1, v1, s1, znt, wspd (and
    mznt for sfcmodel=4), then zntmp = znt (mznt for sfcmodel=4) on
    0..ni+1, 0..nj+1."""
    from .turb import bc2d

    bk, g, cfg, s = ctx.bk, ctx.g, ctx.cfg, ctx.s
    keys = ["ust", "u1", "v1", "s1", "znt", "wspd"]
    gfdl = int(_opt(cfg, "sfcmodel", 0)) == 4 and "mznt" in s
    if gfdl:
        keys.append("mznt")
    out = {k: bc2d(bk, g, cfg, s[k]) for k in keys}
    n = g.ngxy
    R = (slice(n - 1, n + g.nj + 1), slice(n - 1, n + g.ni + 1))
    out["zntmp"] = bk.set(s["zntmp"], R, out["mznt" if gfdl else "znt"][R])
    return out


def surface(ctx):
    """The surface block of sfc_and_turb (bbc=3): ``sfc_prelude``, the
    registered surface layer, ``sfc_bcs``.  Updates ``ctx.s`` in place."""
    mod = scheme(ctx.cfg, "sfc")
    if mod is None:
        return
    ctx.s.update(sfc_prelude(ctx))
    ctx.s.update(mod.surface_layer(ctx))
    ctx.s.update(sfc_bcs(ctx))


def wrf_sfc_epilogue(ctx, out, *, slab=True):
    """turb.F, after a WRF surface layer (sfcmodel 2, 3, 6, 7; ``out`` holds
    its interior results merged over ``ctx.s``): SLAB when update_sfc (else
    qsfc = qv + qfx/flqc and chklowq = mavail), ch = chs2, cq = cqs2, s10 =
    |(u10, v10)|, and thflux = hfx/(cpmm rf1), qvflux = qfx/rf1 with
    dosfcflx or output_sfcflx.  ``out`` must carry the 2-D interior arrays
    hfx, qfx, flhc, flqc, chs2, cqs2, cpmm, u10, v10, qsfc (and lh, tsk for
    SLAB).  Returns the state updates (2-D arrays with halos)."""
    bk, g, cfg, s = ctx.bk, ctx.g, ctx.cfg, ctx.s
    xp = bk.xp
    out = dict(out)
    qv1 = ctx.w["qv"][g.ngz, g.ngxy : g.ngxy + g.nj, g.ngxy : g.ngxy + g.ni]
    if slab and ctx.update_sfc and int(_opt(cfg, "testcase", 0)) != 11:
        from .slab import slab_step

        out.update(slab_step(ctx, out))
    elif int(_opt(cfg, "sfcmodel", 0)) != 7:
        flqc = out["flqc"]
        safe = xp.where(flqc != 0.0, flqc, xp.ones_like(flqc))
        out["qsfc"] = xp.where(flqc != 0.0, qv1 + out["qfx"] / safe, qv1)
        out["chklowq"] = interior2(g, s["mavail"])
    sfcmodel = int(_opt(cfg, "sfcmodel", 0))
    if sfcmodel in (2, 3, 6, 7):
        if "chs2" in out:
            out["ch"] = out["chs2"]
        if "cqs2" in out:
            out["cq"] = out["cqs2"]
        u10, v10 = out["u10"], out["v10"]
        out["s10"] = xp.sqrt(u10 * u10 + v10 * v10)
    if _opt(cfg, "dosfcflx", False) or int(_opt(cfg, "output_sfcflx", 0)) == 1:
        rf1 = interior2(g, ctx.rf[g.ngz])
        out["thflux"] = out["hfx"] / (out["cpmm"] * rf1)
        out["qvflux"] = out["qfx"] / rf1
    res = {}
    for k, v in out.items():
        if k in s and getattr(s[k], "ndim", 0) == 2:
            res[k] = put2(bk, g, s[k], v)
        else:
            res[k] = put2(bk, g, bk.zeros(g.s_shape[1:]), v) if v.ndim == 2 else v
    return res


# ---------------------------------------------------------------------------
# turb.F: turbparam_horiz, turbparam_vert

PRINV = 1.0  # 1/prandtl
DMIN = 1.0e-10


def _atleast(xp, a, b):
    """Fortran max(a, b) with a Python-float b (torch's maximum needs two
    tensors)."""
    return xp.where(a > b, a, b)


def turbparam_horiz(bk, g, cfg, *, defh, kmh, khh, dissten, xland, psfc):
    """turb.F ``turbparam_horiz`` (cm1setup=2, horizturb=1): kmh = tlh^2
    sqrt(max(defh, dmin)) on k=2..nk (bcw; k=1 extrapolated), khh = kmh/Pr,
    dissten += kmh^3/tlh^4 (idiss=1).  tlh = l_h, or over water a function
    of psfc between lhref1 and lhref2.  ``xland``, ``psfc``: 2-D with halos.
    Returns kmh, khh, dissten."""
    from .turb import bcw
    from .turbtend import Reg

    xp = bk.xp
    l_h = float(_opt(cfg, "l_h", 100.0))
    lhref1 = float(_opt(cfg, "lhref1", 100.0))
    lhref2 = float(_opt(cfg, "lhref2", 1000.0))
    out = {"kmh": kmh, "khh": khh, "dissten": dissten}
    if not (l_h > 1.0e-12 or lhref1 > 1.0e-12 or lhref2 > 1.0e-12):
        return out
    if _opt(cfg, "terrain_flag", False):
        raise NotImplementedError("turbparam_horiz: terrain is not ported")
    ni, nj, nk = g.ni, g.nj, g.nk
    xl, ps = interior2(g, xland), interior2(g, psfc)
    tlh = xp.where(
        xl > 1.5,
        lhref2 + (lhref1 - lhref2) * (ps - 90000.0) / (101500.0 - 90000.0),
        xp.zeros_like(ps) + l_h,
    )[None]
    R = Reg(g, (2, nk), (1, nj), (1, ni))
    kmh = bk.set(
        bk.copy(kmh), R.idx, (tlh * tlh) * xp.sqrt(_atleast(xp, R(defh), DMIN))
    )
    kmh = bcw(bk, g, cfg, kmh, 1)
    R1 = Reg(g, (1, 1), (0, nj + 1), (0, ni + 1))
    kmh = bk.set(
        kmh, R1.idx, xp.maximum(0.0 * R1(kmh), 2.0 * R1(kmh, dk=1) - R1(kmh, dk=2))
    )
    RA = Reg(g, (1, nk + 1), (0, nj + 1), (0, ni + 1))
    khh = bk.set(bk.copy(khh), RA.idx, RA(kmh) * PRINV)
    if int(_opt(cfg, "idiss", 0)) == 1 or int(_opt(cfg, "output_dissten", 0)) == 1:
        km = R(kmh)
        dissten = bk.set(
            bk.copy(dissten),
            R.idx,
            R(dissten) + ((km * km) * km) / ((tlh * tlh) * (tlh * tlh)),
        )
    return {"kmh": kmh, "khh": khh, "dissten": dissten}


def turbparam_vert(bk, g, cfg, *, zf, nm, defv, kmv, khv, dissten, zntmp, ust):
    """turb.F ``turbparam_vert`` (ipbl=2): l_v from 1/l_v^2 = 1/l_inf^2 +
    1/(karman (z + z0))^2, kmv = l_v^2 sqrt(max(S_v^2 - N^2/Pr, dmin)) on
    k=2..nk (bcw), khv = kmv/Pr, dissten += kmv^3/l_v^4 (idiss=1), and with
    bbc=3 kmv = khv = karman z0 u* at k=1.  Returns kmv, khv, dissten."""
    from .constants import karman
    from .turb import bc2d_level, bcw
    from .turbtend import Reg

    xp = bk.xp
    l_inf = float(_opt(cfg, "l_inf", 75.0))
    if not l_inf > 1.0e-12:
        return {"kmv": kmv, "khv": khv, "dissten": dissten}
    if _opt(cfg, "terrain_flag", False):
        raise NotImplementedError("turbparam_vert: terrain is not ported")
    ni, nj, nk = g.ni, g.nj, g.nk
    a = 1.0e-6 + l_inf
    rlinf = 1.0 / (a * a)
    R = Reg(g, (2, nk), (1, nj), (1, ni))
    z0 = interior2(g, zntmp)[None]
    x = karman * ((R(zf) - _zf1(g, zf)) + z0)
    lvz = xp.sqrt(1.0 / (rlinf + 1.0 / (x * x)))
    kv = (lvz * lvz) * xp.sqrt(_atleast(xp, R(defv) - R(nm) * PRINV, DMIN))
    kmv = bcw(bk, g, cfg, bk.set(bk.copy(kmv), R.idx, kv), 1)
    RA = Reg(g, (1, nk + 1), (0, nj + 1), (0, ni + 1))
    khv = bk.set(bk.copy(khv), RA.idx, RA(kmv) * PRINV)
    if int(_opt(cfg, "idiss", 0)) == 1 or int(_opt(cfg, "output_dissten", 0)) == 1:
        km = R(kmv)
        dissten = bk.set(
            bk.copy(dissten),
            R.idx,
            R(dissten) + ((km * km) * km) / ((lvz * lvz) * (lvz * lvz)),
        )
    if int(_opt(cfg, "bbc", 1)) == 3:
        R1 = Reg(g, (1, 1), (1, nj), (1, ni))
        kmv = bk.set(kmv, R1.idx, (karman * z0 * interior2(g, ust)[None]))
        kmv = bc2d_level(bk, g, cfg, kmv, 1)
        R0 = Reg(g, (1, 1), (0, nj + 1), (0, ni + 1))
        khv = bk.set(khv, R0.idx, R0(kmv) * PRINV)
    return {"kmv": kmv, "khv": khv, "dissten": dissten}


def _zf1(g, zf):
    """zf(i, j, 1) on the interior, as [1, nj, ni]."""
    n, z = g.ngxy, g.ngz
    return zf[z : z + 1, n : n + g.nj, n : n + g.ni]


# ---------------------------------------------------------------------------
# ipbl=2: the dobr09 block of sfc_and_turb


def check(cfg):
    if int(_opt(cfg, "sfcmodel", 0)) not in (1, 2, 3):
        raise NotImplementedError("ipbl=2 needs sfcmodel=1, 2 or 3")


def pbl(ctx):
    """turb.F ``dobr09`` (ipbl=2, Bryan and Rotunno 2009): the vimpl
    coefficients (doimpl=1), sadv = th0 - th0r + tha, then the implicit
    vertical diffusion tendencies (turbsz: thpten with the surface flux
    thflux, qvpten with qvflux, qcpten, qipten; turbuz/turbvz: upten,
    vpten with gettau's surface stress t13/t23).  Rho-weighted, as CM1 has
    them."""
    from .turb import th0r, vimpl_coefs
    from .turbtend import Reg, _uvz, turbsz

    bk, g, cfg, b, s = ctx.bk, ctx.g, ctx.cfg, ctx.b, ctx.s
    ni, nj, nk = g.ni, g.nj, g.nk
    dt = ctx.dt
    dum7 = dum8 = None
    if int(_opt(cfg, "doimpl", 1)) == 1:
        v = vimpl_coefs(
            bk, g, khv=s["khv"], mf=b["mf"], rf=ctx.rf, mh=b["mh"], rr=ctx.rr
        )
        dum7, dum8 = v["dum7"], v["dum8"]
    RS = Reg(g, (1, nk), (0, nj + 1), (0, ni + 1))
    sadv = bk.set(bk.zeros(g.s_shape), RS.idx, (RS(b["th0"]) - th0r) + RS(ctx.th3d))
    common = dict(
        dt=dt, mh=b["mh"], mf=b["mf"], rho=ctx.rho, rr=ctx.rr, rf=ctx.rf,
        khv=s["khv"], dum7=dum7, dum8=dum8,
    )  # fmt: skip
    R = Reg(g, (1, nk), (1, nj), (1, ni))
    out = {}

    def put(name, v):
        out[name] = bk.set(bk.copy(s[name]), R.idx, v)

    put("thpten", turbsz(bk, g, cfg, iflux=1, s=sadv, sflux=s["thflux"], **common))
    imoist = int(_opt(cfg, "imoist", 0)) == 1
    nqv, nqc, nqi = (int(_opt(cfg, k, 0)) for k in ("nqv", "nqc", "nqi"))
    q = ctx.q3d
    if imoist:
        put(
            "qvpten",
            turbsz(bk, g, cfg, iflux=1, s=q[nqv - 1], sflux=s["qvflux"], **common),
        )
    if imoist and nqc >= 1:
        put("qcpten", turbsz(bk, g, cfg, iflux=0, s=q[nqc - 1], **common))
    if imoist and nqi >= 1:
        put("qipten", turbsz(bk, g, cfg, iflux=0, s=q[nqi - 1], **common))
    uv = dict(
        dt=dt, rho=ctx.rho, rf=ctx.rf, mh=b["mh"], mf=b["mf"], kmv=s["kmv"], w=ctx.w3d
    )
    tu = _uvz(bk, g, cfg, ax="i", u=ctx.u3d, t13=s["t13"], uf_or_vf=b["uf"], **uv)
    tv = _uvz(bk, g, cfg, ax="j", u=ctx.v3d, t13=s["t23"], uf_or_vf=b["vf"], **uv)
    # turbuz/turbvz fill i=1..ni+1 (j=1..nj+1); bcs(upten/vpten) then
    # overwrites the halo, so only the interior matters
    put("upten", tu[:, :, :ni])
    put("vpten", tv[:, :nj, :])
    return out


# ---------------------------------------------------------------------------
# turb.F sfc_and_turb, cm1setup=2


def sfc_and_turb_pbl(ctx):
    """turb.F ``sfc_and_turb`` for cm1setup=2 (see the module docstring).
    Updates ``ctx.s`` in place."""
    from .bc import bcs
    from .turb import calcdef, calcnm, gettau
    from .turbtend import Reg

    bk, g, cfg, b, s = ctx.bk, ctx.g, ctx.cfg, ctx.b, ctx.s
    ni, nj, nk = g.ni, g.nj, g.nk
    if _opt(cfg, "terrain_flag", False):
        raise NotImplementedError("cm1setup=2: terrain is not ported")
    if int(_opt(cfg, "axisymm", 0)) != 0:
        raise NotImplementedError("cm1setup=2: axisymm=1 is not ported")
    s["dissten"] = bk.set(
        bk.copy(s["dissten"]), Reg(g, (1, nk + 1), (1, nj), (1, ni)).idx, 0.0
    )
    surface(ctx)
    s["nm"] = calcnm(
        bk, g, cfg, c1=b["c1"], c2=b["c2"], mf=b["mf"], pi0=b["pi0"], th0=b["th0"],
        nm=s["nm"], prs=ctx.prs, pp=ctx.pp3d, th=ctx.th3d, qa=ctx.q3d,
    )["nm"]  # fmt: skip
    d = calcdef(
        bk, g, cfg, uh=b["uh"], uf=b["uf"], vh=b["vh"], vf=b["vf"], mh=b["mh"],
        c1=b["c1"], c2=b["c2"], mf=b["mf"], rho=ctx.rho, rf=ctx.rf, ua=ctx.u3d,
        va=ctx.v3d, wa=ctx.w3d, s11=s["t11"], s12=s["t12"], s13=s["t13"],
        s22=s["t22"], s23=s["t23"], s33=s["t33"], defv=s["defv"], defh=s["defh"],
    )  # fmt: skip
    s["defv"], s["defh"] = d["defv"], d["defh"]
    for k in ("11", "12", "13", "22", "23", "33"):
        s["t" + k] = d["s" + k]
    horizturb = int(_opt(cfg, "horizturb", 0)) == 1
    ipbl = int(_opt(cfg, "ipbl", 0))
    if horizturb:
        s.update(
            turbparam_horiz(
                bk,
                g,
                cfg,
                defh=s["defh"],
                kmh=s["kmh"],
                khh=s["khh"],
                dissten=s["dissten"],
                xland=s["xland"],
                psfc=s["psfc"],
            )  # fmt: skip
        )
    if ipbl == 2:
        s.update(
            turbparam_vert(
                bk,
                g,
                cfg,
                zf=b["zf"],
                nm=s["nm"],
                defv=s["defv"],
                kmv=s["kmv"],
                khv=s["khv"],
                dissten=s["dissten"],
                zntmp=s["zntmp"],
                ust=s["ust"],
            )  # fmt: skip
        )
    if ipbl == 2 or horizturb:
        s.update(
            gettau(
                bk,
                g,
                cfg,
                kmh=s["kmh"],
                kmv=s["kmv"],
                t11=s["t11"],
                t12=s["t12"],
                t13=s["t13"],
                t22=s["t22"],
                t23=s["t23"],
                t33=s["t33"],
                sfc=s,
                rf=ctx.rf,
            )  # fmt: skip
        )
    mod = scheme(cfg, "pbl")
    if mod is not None:
        s.update(mod.pbl(ctx))
        if _opt(cfg, "use_pbl", False):
            s["upten"] = bcs(bk, g, s["upten"], cfg)
            s["vpten"] = bcs(bk, g, s["vpten"], cfg)


# ---------------------------------------------------------------------------
# solve1.F, use_pbl


def add_theta_tendency(bk, g, *, thten1, thpten):
    """solve1.F: thten1(1:ni,1:nj,1:nk) += thpten (before doturb)."""
    from .turbtend import Reg

    R = Reg(g, (1, g.nk), (1, g.nj), (1, g.ni))
    return bk.set(bk.copy(thten1), R.idx, R(thten1) + R(thpten))


def add_tendencies(bk, g, cfg, *, uten1, vten1, qten, s):
    """solve1.F, after doturb: uten1(1:ni+1) += 0.5 (upten(i-1) + upten(i)),
    vten1(1:nj+1) likewise, and (imoist=1) qten(nqv) += qvpten, qten(nqc)
    += qcpten, qten(nqi) += qipten.  ``s`` holds the PBL tendencies.
    Returns uten1, vten1, qten (a list)."""
    from .turbtend import Reg

    nk, nj, ni = g.nk, g.nj, g.ni
    up, vp = s["upten"], s["vpten"]
    Ru = Reg(g, (1, nk), (1, nj), (1, ni + 1))
    uten1 = bk.set(bk.copy(uten1), Ru.idx, Ru(uten1) + 0.5 * (Ru(up, di=-1) + Ru(up)))
    Rv = Reg(g, (1, nk), (1, nj + 1), (1, ni))
    vten1 = bk.set(bk.copy(vten1), Rv.idx, Rv(vten1) + 0.5 * (Rv(vp, dj=-1) + Rv(vp)))
    if qten is not None and int(_opt(cfg, "imoist", 0)) == 1:
        qten = list(qten)
        R = Reg(g, (1, nk), (1, nj), (1, ni))
        for idx, name in (("nqv", "qvpten"), ("nqc", "qcpten"), ("nqi", "qipten")):
            m = int(_opt(cfg, idx, 0))
            if m >= 1:
                qten[m - 1] = bk.set(
                    bk.copy(qten[m - 1]), R.idx, R(qten[m - 1]) + R(s[name])
                )
    return {"uten1": uten1, "vten1": vten1, "qten": qten}
