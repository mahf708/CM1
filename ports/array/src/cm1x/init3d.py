"""Initial conditions (port of src/init3d.F: ``init3d`` and ``getset``).

Scope: ``iinit = 0`` (no perturbation) or ``1`` (warm bubble), ``irandp = 0``,
``ibalance = 0`` or ``1``, ``psolver <= 3``, no terrain, periodic lateral
boundaries, Kessler moisture (qv, qc, qr) or dry.

Moisture arrays ``qa``/``q3d`` are 4-D like CM1's ``qa(i,j,k,n)``, stored
C-ordered as ``qa[n, k, j, i]`` with ``n = nqv-1, nqc-1, ...`` (0-based).
"""

from __future__ import annotations

import warnings

from . import constants as C
from .bc import periodic_xy

__all__ = ["init3d", "getset"]


def init3d(bk, cfg, *, xh, yh, zh, pi0, prs0, th0, thv0, qv0, u0, v0, centerx, centery):
    """Port of ``init3d`` for the options listed in the module docstring.

    Returns: ua, u3d, va, v3d, wa, w3d (=0), tha (theta perturbation), qa,
    ppi (pi perturbation).
    """
    xp = bk.xp
    S = bk.asarray
    shp_s = zh.shape
    nk = shp_s[0] - 2
    ng = 3
    nj, ni = shp_s[1] - 2 * ng, shp_s[2] - 2 * ng
    K, J, I = slice(1, nk + 1), slice(ng, ng + nj), slice(ng, ng + ni)
    shp_w = (shp_s[0] + 1, shp_s[1], shp_s[2])

    if cfg.irandp == 1:
        raise NotImplementedError("irandp=1 (random perturbations) not ported")
    if cfg.terrain_flag:
        raise NotImplementedError("terrain not ported")

    ua = bk.copy(u0)
    va = bk.copy(v0)
    wa = bk.zeros(shp_w)

    # qv set to base-state value (incl. halos); qc, qr = 0
    numq = cfg.numq if cfg.imoist == 1 else 0
    if numq:
        qa = bk.zeros((numq,) + shp_s)
        qa = bk.set(qa, cfg.nqv - 1, qv0)
    else:
        qa = bk.zeros((0,) + shp_s)

    # ---------------- iinit options
    tha = bk.zeros(shp_s)
    if cfg.iinit == 1:
        ric, rjc = S(centerx), S(centery)
        zc, bhrad, bvrad, bptpert = S(1400.0), S(10000.0), S(1400.0), S(1.0)
        xi = xh[I][None, None, :]
        yj = yh[J][None, :, None]
        zk = zh[K, J, I]
        beta = xp.sqrt(
            ((xi - ric) / bhrad) ** 2
            + ((yj - rjc) / bhrad) ** 2
            + ((zk - zc) / bvrad) ** 2
        )
        pert = xp.where(
            beta < S(1.0), bptpert * (xp.cos(S(0.5 * C.pi) * beta) ** 2), S(0.0)
        )
        tha = bk.set(tha, (K, J, I), pert)
    elif cfg.iinit not in (0, None):
        raise NotImplementedError(f"iinit={cfg.iinit} not ported")

    # ---------------- 3-D pressure
    ppi = bk.zeros(shp_s)
    if cfg.ibalance == 1:
        g, cp, rd, rv = S(C.g), S(C.cp), S(C.rd), S(C.rv)
        reps = rv / rd
        one, half = S(1.0), S(0.5)
        if cfg.imoist == 1:
            q = qa[:, K, J, I]
            qt = q[cfg.nql1 - 1]
            for n in range(cfg.nql1, cfg.nql2):
                qt = qt + q[n]
            qvn = q[cfg.nqv - 1]
            thvnew = (
                (th0[K, J, I] + tha[K, J, I]) * (one + reps * qvn) / (one + qvn + qt)
            )
        else:
            thvnew = th0[K, J, I] + tha[K, J, I]
        zz = zh[K, J, I]
        pis = [None] * nk
        pis[nk - 1] = pi0[nk, J, I]
        for k in range(nk - 2, -1, -1):
            pis[k] = pis[k + 1] + g * (zz[k + 1] - zz[k]) / (
                cp * half * (thvnew[k + 1] + thvnew[k])
            )
        pinew = xp.stack(pis)
        pp = pinew - pi0[K, J, I]
        pp = xp.where(xp.abs(pp) < S(1.0e-6), S(0.0), pp)
        ppi = bk.set(ppi, (K, J, I), pp)
    elif cfg.ibalance not in (0, None):
        raise NotImplementedError(f"ibalance={cfg.ibalance} not ported")

    return dict(
        ua=ua,
        u3d=bk.copy(ua),
        va=va,
        v3d=bk.copy(va),
        wa=wa,
        w3d=bk.copy(wa),
        tha=tha,
        qa=qa,
        ppi=ppi,
    )


class _G:
    ngxy = 3

    def __init__(self, ni, nj):
        self.ni, self.nj = ni, nj


def getset(
    bk,
    cfg,
    *,
    ua,
    va,
    wa,
    ppi,
    tha,
    qa,
    pi0,
    th0,
    rho0,
    rf0,
    prs0,
    c1,
    c2,
    rmh,
    ruh,
    rvh,
    cg: dict,
) -> dict:
    """Port of the non-restart path of ``getset`` (psolver <= 3, no terrain/LES).

    Fills lateral halos (periodic), copies the a-arrays into the 3d-arrays and
    diagnoses prs, rho, rr, rf, ppx and mass1. Returns ua, va, wa, ppi, tha, qa,
    u3d, v3d, w3d, pp3d, th3d, q3d, rho, prs, rr, rf, ppx, rth0, mass1.
    """
    xp = bk.xp
    S = bk.asarray
    shp_s = th0.shape
    nk = shp_s[0] - 2
    ng = 3
    nj, ni = shp_s[1] - 2 * ng, shp_s[2] - 2 * ng
    K, J, I = slice(1, nk + 1), slice(ng, ng + nj), slice(ng, ng + ni)
    gg = _G(ni, nj)
    if cfg.psolver > 3:
        raise NotImplementedError("getset: only psolver <= 3 ported")
    if not (cfg.wbc == cfg.ebc == cfg.sbc == cfg.nbc == 1):
        raise NotImplementedError("getset: only periodic lateral boundaries ported")

    cp, rd, rv, p00 = S(C.cp), S(C.rd), S(C.rv), S(C.p00)
    cpdrd = cp / rd

    # u/v base-state fields are horizontally homogeneous: bcu/bcv leave them unchanged.
    ppi = periodic_xy(bk, ppi, gg)
    tha = periodic_xy(bk, tha, gg)
    if qa.shape[0]:
        qa = xp.stack([periodic_xy(bk, qa[n], gg) for n in range(qa.shape[0])])
    # bcw on wa (zeros) is a no-op
    rth0 = S(1.0) / th0

    u3d, v3d, w3d = bk.copy(ua), bk.copy(va), bk.copy(wa)
    pp3d, th3d, q3d = bk.copy(ppi), bk.copy(tha), bk.copy(qa)

    # prs, rho (interior)
    pit = pi0[K, J, I] + pp3d[K, J, I]
    prs_i = p00 * (pit**cpdrd)
    if cfg.imoist == 1:
        qv = q3d[cfg.nqv - 1, K, J, I]
        rho_i = prs_i / (
            (th0[K, J, I] + th3d[K, J, I]) * pit * (rd + xp.maximum(S(0.0), qv) * rv)
        )
    else:
        rho_i = prs_i / (rd * (th0[K, J, I] + th3d[K, J, I]) * pit)
    prs = bk.set(bk.zeros(shp_s), (K, J, I), prs_i)
    rho = bk.set(bk.zeros(shp_s), (K, J, I), rho_i)
    rho = periodic_xy(bk, rho, gg)  # bcs: k = 1..nk only (k=0, nk+1 stay 0)

    rr = bk.set(bk.zeros(shp_s), (K, J, I), S(1.0) / rho[K, J, I])
    rr = periodic_xy(bk, rr, gg)
    rf_i = c1[K, J, I] * rho[0:nk, J, I] + c2[K, J, I] * rho[K, J, I]
    rf = bk.set(bk.zeros(shp_s), (K, J, I), rf_i)
    rf = periodic_xy(bk, rf, gg)
    # Fortran j,i = 0..nj+1 / 0..ni+1  -> python ng-1 .. ng+n
    J1, I1 = slice(ng - 1, ng + nj + 1), slice(ng - 1, ng + ni + 1)
    cgs1, cgs2, cgs3 = (S(cg[k]) for k in ("cgs1", "cgs2", "cgs3"))
    cgt1, cgt2, cgt3 = (S(cg[k]) for k in ("cgt1", "cgt2", "cgt3"))
    rf = bk.set(
        rf,
        (1, J1, I1),
        cgs1 * rho[1, J1, I1] + cgs2 * rho[2, J1, I1] + cgs3 * rho[3, J1, I1],
    )
    rf = bk.set(
        rf,
        (nk + 1, J1, I1),
        cgt1 * rho[nk, J1, I1]
        + cgt2 * rho[nk - 1, J1, I1]
        + cgt3 * rho[nk - 2, J1, I1],
    )

    # mass1 (double / quad precision in CM1)
    w = rho[K, J, I] * ruh[I][None, None, :] * rvh[J][None, :, None] * rmh[K, J, I]
    with warnings.catch_warnings():  # jax without x64 keeps float32
        warnings.simplefilter("ignore")
        w = xp.astype(w, xp.float64)
    mass1 = xp.sum(xp.sum(w, axis=(1, 2)))

    ppx = bk.set(bk.zeros(shp_s), (K, J, I), ppi[K, J, I])
    ppx = periodic_xy(bk, ppx, gg)

    return dict(
        ua=ua,
        va=va,
        wa=wa,
        ppi=ppi,
        tha=tha,
        qa=qa,
        u3d=u3d,
        v3d=v3d,
        w3d=w3d,
        pp3d=pp3d,
        th3d=th3d,
        q3d=q3d,
        rho=rho,
        prs=prs,
        rr=rr,
        rf=rf,
        ppx=ppx,
        rth0=rth0,
        mass1=mass1,
    )
