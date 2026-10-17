"""Post-RK work, ported from src/solve3.F (active path of the reference case).

After microphysics (mp_driver, called between solve2 and solve3 by cm1.F):
  * lateral BCs of pp3d, th3d, q3d;
  * ppx = pp3d + ppx (psolver=3) and its BCs; BCs of rho;
  * diabatic tendencies for the next step (eqtset=2, ptype>=1);
  * "equate" the time levels: ua=u3d, va=v3d, wa=w3d, ppi=pp3d, tha=th3d, qa=q3d;
  * rr = 1/rho, rf (rho on w levels).
Surface swath diagnostics (sws, svs, ...) are not ported (output only).
"""

from __future__ import annotations

from .misc import rhoextra
from .solve2 import bcp, bcs


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
    **unused,
):
    """solve3.F ``solve3`` (active path).  ``q3d`` is a sequence of numq
    arrays.  Returns pp3d, th3d, q3d, ppx, rho, rr, rf, qpten, qtten, qvten,
    qcten and the equated arrays ua, va, wa, ppi, tha, qa."""
    if int(_opt(cfg, "imove", 0)) != 0 or int(_opt(cfg, "iprcl", 0)) != 0:
        raise NotImplementedError("solve3: imove/iprcl not ported")
    nz = g.ngz
    nk = g.nk
    nqv = int(_opt(cfg, "nqv", 1)) - 1
    nqc = int(_opt(cfg, "nqc", 2)) - 1
    ptype = int(_opt(cfg, "ptype", 1))
    SI = (slice(nz, nz + nk), g.J, g.I)

    pp3d = bcp(bk, g, bk.copy(pp3d))
    th3d = bcs(bk, g, bk.copy(th3d))
    q3d = [bcs(bk, g, bk.copy(q)) for q in q3d]

    ppx = bk.copy(ppx)
    ppx = bk.set(ppx, SI, pp3d[SI] + ppx[SI])
    ppx = bcp(bk, g, ppx)
    rho = bcp(bk, g, bk.copy(rho))

    qpten, qtten, qvten, qcten = (bk.copy(a) for a in (qpten, qtten, qvten, qcten))
    if ptype >= 1:
        rdt = 1.0 / dt
        qpten = bk.set(qpten, SI, (pp3d[SI] - qpten[SI]) * rdt)
        qtten = bk.set(qtten, SI, (th3d[SI] - qtten[SI]) * rdt)
        qvten = bk.set(qvten, SI, (q3d[nqv][SI] - qvten[SI]) * rdt)
        qcten = bk.set(qcten, SI, (q3d[nqc][SI] - qcten[SI]) * rdt)

    # equate the two time levels (CM1 copies only 1-2 halo rows of the "a"
    # arrays; the 3d arrays' halos are periodic copies, so copying all is
    # equivalent for every point that is read later)
    ua, va, wa = bk.copy(u3d), bk.copy(v3d), bk.copy(w3d)
    ppi, tha = bk.copy(pp3d), bk.copy(th3d)
    qa = [bk.copy(q) for q in q3d]

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
    }
