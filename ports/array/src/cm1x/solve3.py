"""Post-RK work, ported from src/solve3.F (active path of the reference case).

After microphysics (mp_driver, called between solve2 and solve3 by cm1.F):
  * lateral BCs of pp3d, th3d, q3d;
  * ppx = pp3d + ppx (psolver=3) and its BCs; BCs of rho;
  * diabatic tendencies for the next step (eqtset=2, ptype>=1);
  * "equate" the time levels: ua=u3d, va=v3d, wa=w3d, ppi=pp3d, tha=th3d,
    qa=q3d and, with prognostic TKE (sgsmodel=1), tkea=tke3d, over solve3.F's
    index ranges (the outer halo points of the "a" arrays keep their values);
  * rr = 1/rho, rf (rho on w levels).
Lateral BCs use cm1x.bc with ``cfg`` (periodic, open or walls).  Surface
swath diagnostics (sws, svs, ...) are not ported (output only); imove only
enters those and the base state, so it is accepted.  iprcl=1 (parcels, which
would need bcu(rru)/bcv(rrv)/bcw(rrw,1) here) is not ported.
"""

from __future__ import annotations

from . import bc as _bc
from .misc import rhoextra
from .turb import tke_copy


def _opt(cfg, name, default):
    if cfg is None:
        return default
    # dict.get / Config.get: an unset option falls back to ``default``
    # rather than Config's zero for unknown names
    get = getattr(cfg, "get", None)
    return get(name, default) if get else getattr(cfg, name, default)


def solve3(
    bk,
    g,
    cfg=None,
    *,
    dt,
    c1,
    c2,
    u3d,
    v3d,
    w3d,
    pp3d,
    th3d,
    q3d,
    rho,
    ppx,
    qpten,
    qtten,
    qvten,
    qcten,
    ua=None,
    va=None,
    wa=None,
    ppi=None,
    tha=None,
    qa=None,
    tke3d=None,
    tkea=None,
    **unused,
):
    """solve3.F ``solve3`` (active path).  ``q3d`` is a sequence of numq
    arrays.  ``ua .. qa`` (and ``tkea``) are the old time level, overwritten
    on solve3.F's equate ranges (omitted: full copies of the new level).
    Returns pp3d, th3d, q3d, ppx, rho, rr, rf, qpten, qtten, qvten, qcten,
    the equated arrays ua, va, wa, ppi, tha, qa and, when tke3d is given,
    tkea."""
    if int(_opt(cfg, "iprcl", 0)) != 0:
        raise NotImplementedError("solve3: iprcl not ported")
    nz = g.ngz
    nk = g.nk
    nqv = int(_opt(cfg, "nqv", 1)) - 1
    nqc = int(_opt(cfg, "nqc", 2)) - 1
    ptype = int(_opt(cfg, "ptype", 1))
    SI = (slice(nz, nz + nk), g.J, g.I)

    pp3d = _bc.bcp(bk, g, bk.copy(pp3d), cfg)
    th3d = _bc.bcs(bk, g, bk.copy(th3d), cfg)
    q3d = [_bc.bcs(bk, g, bk.copy(q), cfg) for q in q3d]

    ppx = bk.copy(ppx)
    ppx = bk.set(ppx, SI, pp3d[SI] + ppx[SI])
    ppx = _bc.bcp(bk, g, ppx, cfg)
    rho = _bc.bcp(bk, g, bk.copy(rho), cfg)

    qpten, qtten, qvten, qcten = (bk.copy(a) for a in (qpten, qtten, qvten, qcten))
    if ptype >= 1:
        rdt = 1.0 / dt
        qpten = bk.set(qpten, SI, (pp3d[SI] - qpten[SI]) * rdt)
        qtten = bk.set(qtten, SI, (th3d[SI] - qtten[SI]) * rdt)
        qvten = bk.set(qvten, SI, (q3d[nqv][SI] - qvten[SI]) * rdt)
        qcten = bk.set(qcten, SI, (q3d[nqc][SI] - qcten[SI]) * rdt)

    # equate the two time levels: u, v, w on k=1..nk, j,i=0..n+2; scalars
    # on k=1..nk, j,i=0..n+1 (solve3.F)
    n = g.ngxy
    K = slice(nz, nz + nk)
    J2, I2 = slice(n - 1, n + g.nj + 2), slice(n - 1, n + g.ni + 2)
    J1, I1 = slice(n - 1, n + g.nj + 1), slice(n - 1, n + g.ni + 1)

    def eq(old, new, J, I):
        if old is None:
            return bk.copy(new)
        return bk.set(bk.copy(old), (K, J, I), new[K, J, I])

    ua, va, wa = eq(ua, u3d, J2, I2), eq(va, v3d, J2, I2), eq(wa, w3d, J2, I2)
    ppi, tha = eq(ppi, pp3d, J1, I1), eq(tha, th3d, J1, I1)
    if qa is None:
        qa = [None] * len(q3d)
    qa = [eq(o, q, J1, I1) for o, q in zip(qa, q3d)]
    out_tke = {}
    if tke3d is not None:
        if tkea is None:
            tkea = bk.zeros(tke3d.shape)
        out_tke["tkea"] = tke_copy(bk, g, cfg, tke3d=tke3d, tkea=tkea)["tkea"]

    rx = rhoextra(bk, g, rho=rho, c1=c1, c2=c2)
    return {
        "pp3d": pp3d,
        "th3d": th3d,
        "q3d": q3d,
        "ppx": ppx,
        "rho": rho,
        "rr": rx["rr"],
        "rf": rx["rf"],
        "qpten": qpten,
        "qtten": qtten,
        "qvten": qvten,
        "qcten": qcten,
        "ua": ua,
        "va": va,
        "wa": wa,
        "ppi": ppi,
        "tha": tha,
        "qa": qa,
        **out_tke,
    }
