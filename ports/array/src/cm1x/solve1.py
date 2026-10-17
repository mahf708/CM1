"""Pre-RK tendencies, ported from src/solve1.F.

Only the code path active for the reference case (irdamp=1, hrdamp=0,
sgsmodel=0, idiff=0, lspgrad=0, testcase=0, no radiation/PBL/cumulus/
nudging/recycling, psolver=3, eqtset=2, imoist=1, ptype>=1) is ported:

  * Rayleigh damping (irdamp=1) into uten1, vten1, wten1, thten1;
  * qten = 0;
  * the mass-conserving pressure tendency ppten (eqtset=2, moist) plus the
    diabatic tendencies of the previous step (qpten, qtten, qvten, qcten).

Terms that are identically zero for these options are omitted: the subgrid
turbulence tendencies (cm1setup=1 makes CM1 call turbu/turbv/turbw/turbs, but
with sgsmodel=0 the eddy viscosities and stresses are zero) and dissipative
heating (idiss=1, but the dissipation rate is zero without an SGS model).
"""

from __future__ import annotations

from . import constants as c
from .misc import QSMALL, TSMALL


def _opt(cfg, name, default):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _check(cfg):
    bad = []
    for name, ok in (
        ("irdamp", (0, 1)),
        ("hrdamp", (0,)),
        ("lspgrad", (0,)),
        ("idiff", (0,)),
        ("sgsmodel", (0,)),
        ("ipbl", (0,)),
        ("radopt", (0,)),
        ("cuparam", (0,)),
        ("testcase", (0,)),
        ("rterm", (0,)),
        ("eqtset", (2,)),
        ("psolver", (3,)),
        ("imoist", (1,)),
        ("isfcflx", (0,)),
        ("axisymm", (0,)),
    ):
        v = _opt(cfg, name, ok[0])
        if v not in ok:
            bad.append(f"{name}={v}")
    for name in (
        "do_lsnudge",
        "terrain_flag",
        "do_recycle_w",
        "do_recycle_e",
        "do_recycle_s",
        "do_recycle_n",
    ):
        if _opt(cfg, name, False):
            bad.append(f"{name}=T")
    if bad:
        raise NotImplementedError("solve1: options not ported: " + ", ".join(bad))


def solve1(
    bk,
    g,
    cfg=None,
    *,
    dt,
    tauh,
    taus,
    tauf,
    rho,
    pi0,
    th0,
    u0,
    ua,
    v0,
    va,
    wa,
    ppi,
    tha,
    qa,
    qpten,
    qtten,
    qvten,
    qcten,
    uten1=None,
    vten1=None,
    wten1=None,
    thten1=None,
    ppten=None,
    qten=None,
    **unused,
):
    """solve1.F ``solve1`` (active path only).

    ``qa`` is a sequence of the numq moisture arrays (Fortran qa(:,:,:,n),
    n = nqv, nqc, nqr for Kessler).  Returns uten1, vten1, wten1, thten1,
    ppten, qten (list of numq arrays).
    """
    _check(cfg)
    xp = bk.xp
    ni, nj, nk = g.ni, g.nj, g.nk
    n, nz = g.ngxy, g.ngz
    numq = len(qa)
    nqv = int(_opt(cfg, "nqv", 1)) - 1
    nqc = int(_opt(cfg, "nqc", 2)) - 1
    irdamp = int(_opt(cfg, "irdamp", 1))
    rdalpha = float(_opt(cfg, "rdalpha", 3.3333333333e-3))
    ptype = int(_opt(cfg, "ptype", 1))

    uten1 = bk.zeros(g.u_shape) if uten1 is None else bk.copy(uten1)
    vten1 = bk.zeros(g.v_shape) if vten1 is None else bk.copy(vten1)
    wten1 = bk.zeros(g.w_shape) if wten1 is None else bk.copy(wten1)
    thten1 = bk.zeros(g.s_shape) if thten1 is None else bk.copy(thten1)
    ppten = bk.zeros(g.s_shape) if ppten is None else bk.copy(ppten)
    if qten is None:
        qten = [bk.zeros(g.s_shape) for _ in range(numq)]
    else:
        qten = [bk.copy(q) for q in qten]

    # Fortran k=1..nk, j=1..nj+1, i=1..ni+1
    K = slice(nz, nz + nk)
    Jp = slice(n, n + nj + 1)
    Ip = slice(n, n + ni + 1)
    Jm = slice(n - 1, n + nj)  # j-1
    Im = slice(n - 1, n + ni)  # i-1
    R = (K, Jp, Ip)

    # ---- Rayleigh damping -------------------------------------------------
    if irdamp == 1:
        uten1 = bk.set(
            uten1, R, -rdalpha * 0.5 * (tauh[K, Jp, Im] + tauh[R]) * (ua[R] - u0[R])
        )
        vten1 = bk.set(
            vten1, R, -rdalpha * 0.5 * (tauh[K, Jm, Ip] + tauh[R]) * (va[R] - v0[R])
        )
        wten1 = bk.set(wten1, R, -rdalpha * tauf[R] * wa[R])
        thten1 = bk.set(thten1, R, -rdalpha * taus[R] * tha[R])
    else:
        z = bk.zeros(ua[R].shape)
        uten1 = bk.set(uten1, R, z)
        vten1 = bk.set(vten1, R, z)
        wten1 = bk.set(wten1, R, z)
        thten1 = bk.set(thten1, R, z)

    # ---- moisture tendencies ----------------------------------------------
    S = (K, slice(n, n + nj), slice(n, n + ni))
    for m in range(numq):
        qten[m] = bk.set(qten[m], S, bk.zeros(qa[m][S].shape))

    # ---- pressure tendency for mass conservation (eqtset=2, moist) --------
    rdt = 1.0 / dt
    tem = 0.0001 * TSMALL
    th1 = thten1[S]
    qv1 = qten[nqv][S]
    thnew = tha[S] + dt * th1
    qvnew = qa[nqv][S] + dt * qv1
    pinew = (
        rho[S] * (th0[S] + thnew) * (c.rd + c.rv * qvnew) * c.rp00
    ) ** c.rddcv - pi0[S]
    cond = (xp.abs(dt * th1) > tem) | (xp.abs(dt * qv1) > QSMALL)
    pt = xp.where(cond, (pinew - ppi[S]) * rdt, xp.zeros_like(pinew))
    ppten = bk.set(ppten, S, pt)

    if ptype >= 1:
        # diabatic tendencies from last timestep as a good estimate
        ppten = bk.set(ppten, S, ppten[S] + qpten[S])
        thten1 = bk.set(thten1, S, thten1[S] + qtten[S])
        qten[nqv] = bk.set(qten[nqv], S, qten[nqv][S] + qvten[S])
        qten[nqc] = bk.set(qten[nqc], S, qten[nqc][S] + qcten[S])

    return {
        "uten1": uten1,
        "vten1": vten1,
        "wten1": wten1,
        "thten1": thten1,
        "ppten": ppten,
        "qten": qten,
    }
