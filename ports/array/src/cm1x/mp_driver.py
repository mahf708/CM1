"""Microphysics driver, ported from src/mp_driver.F.

``mp_driver`` dispatches on ``ptype`` to the branch of the scheme
(``MP_BRANCHES``): ptype=1 (Kessler) lives here, the ice schemes in their
own modules (ptype=3 Thompson: cm1x.thompson, ptype=5 Morrison:
cm1x.morrison), imported on first use.  Every branch takes the same keyword
arguments (the Fortran dummy names; unused ones are ignored) and returns a
dict with at least q3d, th3d, pp3d, prs, rho, qbudget, asq, rain, prate.

Kessler path: ptype=1, efall=0, axisymm=0, no budget diagnostics
(dotbud/doqbud with td_mp/qd_mp) and no reflectivity diagnostic (for ptype=1
CM1 only zeroes qdiag(qd_dbz) here).  Other options raise NotImplementedError.
"""

from __future__ import annotations

import importlib

from . import constants as C
from .kessler import fallout, geterain, k_fallout, kessler, satadj
from .pdef import pdefq

_DEFAULTS = dict(ptype=1, efall=0, axisymm=0, qsmall=1.0e-14, nfall_max=0)

# ptype -> branch: a function here, or "module:function" imported lazily
MP_BRANCHES = {
    1: "mp_driver_kessler",
    3: "cm1x.thompson:mp_driver_ptype3",
    5: "cm1x.morrison:mp_driver_ptype5",
}


def _opt(cfg, name):
    if cfg is None:
        return _DEFAULTS[name]
    # dict.get / Config.get: an unset option falls back to this module's
    # default rather than Config's zero for unknown names
    get = getattr(cfg, "get", None)
    return get(name, _DEFAULTS[name]) if get else getattr(cfg, name, _DEFAULTS[name])


def mp_driver(bk, g, cfg=None, **kw):
    """mp_driver.F ``mp_driver``: the branch of ``MP_BRANCHES`` for
    ``cfg.ptype`` (see the module docstring)."""
    ptype = int(_opt(cfg, "ptype"))
    target = MP_BRANCHES.get(ptype)
    fn = None
    if target is not None and ":" in target:
        mod, name = target.split(":")
        try:
            fn = getattr(importlib.import_module(mod), name, None)
        except ModuleNotFoundError as e:
            if e.name != mod:  # a real import error inside the scheme
                raise
    elif target is not None:
        fn = globals()[target]
    if fn is None:
        raise NotImplementedError(f"mp_driver: ptype={ptype} is not ported")
    return fn(bk, g, cfg, **kw)


def mp_driver_kessler(
    bk,
    g,
    cfg=None,
    *,
    dt,
    qbudget,
    asq,
    ruh,
    rvh,
    mh,
    rmh,
    zh,
    mf,
    rain,
    prate,
    pi0,
    th0,
    rho,
    prs,
    pp3d,
    th3d,
    q3d,
    qten=None,
    thten=None,
    dum1=None,
    dum3=None,
    nfall_max=None,
    satadj_maxiter=None,
    **unused,
):
    """mp_driver.F ``mp_driver`` for ptype=1 (Kessler).

    Shapes: scalars ``g.s_shape``; ``mf`` ``g.w_shape``; ``q3d``/``qten``
    ``[3, k, j, i]`` (qv, qc, qr); ``rain`` ``[nrain, j, i]``; ``prate``
    ``[j, i]``; ``qbudget`` 1-D length >= 7 (nbudget=10); ``asq`` 1-D length
    >= 3.  ``ruh``/``rvh`` 1-D over i/j.  Other Fortran dummy arguments may be
    passed and are ignored.  ``nfall_max``/``satadj_maxiter``: see
    ``kessler.fallout``/``kessler.satadj`` (needed static under jax.jit).

    Sequence (as in CM1): T -> dum1 and thten, qv -> qten(nqv), rho -> dum3;
    pdefq(qv,qc,qr); k_fallout; geterain; fallout(qr); kessler; satadj(nrk=4).

    Returns dict with q3d, th3d, pp3d, prs, rho, qbudget, asq, rain, prate,
    qten, thten, dum1, dum3.
    """
    if int(_opt(cfg, "efall")) != 0:
        raise NotImplementedError("mp_driver: efall>=1 not ported")
    if int(_opt(cfg, "axisymm")) != 0:
        raise NotImplementedError("mp_driver: axisymm not ported")
    xp = bk.xp
    S = (g.K, g.J, g.I)
    qsmall = float(_opt(cfg, "qsmall"))
    if nfall_max is None and int(_opt(cfg, "nfall_max") or 0) > 0:
        # static sedimentation sub-step bound (e.g. Model(...,
        # overrides={"nfall_max": 4})): fixed-trip loop, reverse-mode AD
        # works under jax.jit
        nfall_max = int(_opt(cfg, "nfall_max"))

    def z(a, shape):
        return bk.zeros(shape) if a is None else bk.copy(a)

    qshape = (3,) + tuple(g.s_shape)
    qten = z(qten, qshape)
    thten = z(thten, g.s_shape)
    dum1 = z(dum1, g.s_shape)
    dum3 = z(dum3, g.s_shape)

    t = (th0[S] + th3d[S]) * (pi0[S] + pp3d[S])
    dum1 = bk.set(dum1, S, t)
    thten = bk.set(thten, S, t)
    qten = bk.set(qten, (0,) + S, q3d[(0,) + S])
    dum3 = bk.set(dum3, S, rho[S])

    # pdefq on qv, qc, qr
    qv, qc, qr = q3d[0], q3d[1], q3d[2]
    asq_l = [asq[n] for n in range(asq.shape[0])]
    qs = []
    for n, (rmax, qq) in enumerate(((0.0, qv), (qsmall, qc), (qsmall, qr))):
        o = pdefq(
            bk,
            g,
            cfg,
            rmax=rmax,
            asq=asq_l[n],
            ruh=ruh,
            rvh=rvh,
            rmh=rmh,
            rho=rho,
            q3d=qq,
        )
        qs.append(o["q3d"])
        asq_l[n] = o["asq"]
    qv, qc, qr = qs

    qb = [qbudget[n] for n in range(qbudget.shape[0])]
    vr = k_fallout(bk, g, cfg, rho=rho, qr3d=qr, vr=qten[2])["vr"]
    qb[6] = geterain(
        bk,
        g,
        cfg,
        dt=dt,
        cpx=C.cpl,
        lx1=C.lv1,
        erain=qb[6],
        ruh=ruh,
        rvh=rvh,
        t=dum1,
        rr=dum3,
        q3d=qr,
        vr=vr,
    )["erain"]
    o = fallout(
        bk,
        g,
        cfg,
        dt=dt,
        train=qb[5],
        ruh=ruh,
        rvh=rvh,
        zh=zh,
        mh=mh,
        mf=mf,
        rain=rain,
        prate=prate,
        rr=dum3,
        rho=rho,
        q3d=qr,
        vq=vr,
        nfall_max=nfall_max,
    )
    rain, prate, qr, vr, qb[5] = o["rain"], o["prate"], o["q3d"], o["vq"], o["train"]

    # a separately compiled routine with torch jit=True (Backend.kernel)
    o = bk.kernel(kessler, traced=("dt",))(
        bk,
        g,
        cfg,
        dt=dt,
        tauto=qb[2],
        taccr=qb[3],
        tevar=qb[4],
        ruh=ruh,
        rvh=rvh,
        rmh=rmh,
        pi0=pi0,
        th0=th0,
        tmp=dum1,
        rho=rho,
        rr=dum3,
        pp3d=pp3d,
        th3d=th3d,
        prs=prs,
        qv3d=qv,
        qc3d=qc,
        qr3d=qr,
    )
    dum1, rho, pp3d, th3d, prs = o["tmp"], o["rho"], o["pp3d"], o["th3d"], o["prs"]
    qv, qc, qr = o["qv3d"], o["qc3d"], o["qr3d"]
    qb[2], qb[3], qb[4] = o["tauto"], o["taccr"], o["tevar"]

    q3d = xp.stack([qv, qc, qr], axis=0)
    o = satadj(
        bk,
        g,
        cfg,
        nrk=4,
        dt=dt,
        tcond=qb[0],
        tevac=qb[1],
        ruh=ruh,
        rvh=rvh,
        rmh=rmh,
        pi0=pi0,
        th0=th0,
        rho=rho,
        rr=dum3,
        pp3d=pp3d,
        prs=prs,
        th3d=th3d,
        q3d=q3d,
        maxiter=satadj_maxiter,
    )
    rho, pp3d, prs, th3d, q3d = o["rho"], o["pp3d"], o["prs"], o["th3d"], o["q3d"]
    qb[0], qb[1] = o["tcond"], o["tevac"]

    qten = bk.set(qten, 2, vr)
    dt_ = qbudget.dtype
    return {
        "q3d": q3d,
        "th3d": th3d,
        "pp3d": pp3d,
        "prs": prs,
        "rho": rho,
        "qbudget": xp.stack([xp.asarray(b, dtype=dt_) for b in qb]),
        "asq": xp.stack([xp.asarray(a, dtype=dt_) for a in asq_l]),
        "rain": rain,
        "prate": prate,
        "qten": qten,
        "thten": thten,
        "dum1": dum1,
        "dum3": dum3,
    }
