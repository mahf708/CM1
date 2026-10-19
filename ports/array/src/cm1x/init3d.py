"""Initial conditions (port of src/init3d.F: ``init3d`` and ``getset``).

Scope: ``iinit = 0`` (no perturbation) or ``1`` (warm bubble), ``irandp = 0`` or
``1`` (the random perturbations of each test case, cm1x.testcase),
``ibalance = 0`` or ``1``, ``psolver <= 3``, with or without terrain, any lateral boundary
type of cm1x.bc (periodic / open / walls), Kessler moisture (qv, qc, qr) or
dry, and the prognostic TKE of sgsmodel=1 (cm1setup=1; zero initial TKE).

Moisture arrays ``qa``/``q3d`` are 4-D like CM1's ``qa(i,j,k,n)``, stored
C-ordered as ``qa[n, k, j, i]`` with ``n = nqv-1, nqc-1, ...`` (0-based).
"""

from __future__ import annotations

import warnings

from . import constants as C
from . import bc as _bc
from .grid import Grid
from .misc import level_sums, qp_sum

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

    # ---------------- random perturbations (irandp=1), by test case
    if cfg.irandp == 1:
        import numpy as np

        from .testcase import random_perturbations

        npy = bk.to_numpy
        qn = npy(qa) if numq else np.zeros((0,) + tuple(shp_s))
        ua_n, va_n, tha_n, qa_n = random_perturbations(
            cfg,
            zh=npy(zh),
            u0=npy(u0),
            v0=npy(v0),
            ua=npy(ua),
            va=npy(va),
            tha=npy(tha),
            qa=qn,
        )
        ua, va, tha = S(ua_n), S(va_n), S(tha_n)
        if numq:
            qa = S(qa_n)
    elif cfg.irandp not in (0, None):
        raise NotImplementedError(f"irandp={cfg.irandp} not ported")

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
    g=None,
    gz=None,
    dzdx=None,
    dzdy=None,
    zh=None,
    zf=None,
) -> dict:
    """Port of the non-restart path of ``getset`` (psolver <= 3, no terrain/LES).

    Fills lateral halos (cm1x.bc with ``cfg``'s wbc/ebc/sbc/nbc), copies the
    a-arrays into the 3d-arrays and diagnoses prs, rho, rr, rf, ppx and mass1.
    For idoles with prognostic TKE (cm1setup=1, sgsmodel=1) also sets up
    tkea/tke3d (zero: init3d.F gives TKE an initial value only for testcases
    3/4/5/9/11) with bcw and tkea(k=1)=tkea(k=2).  Returns ua, va, wa, ppi,
    tha, qa, u3d, v3d, w3d, pp3d, th3d, q3d, rho, prs, rr, rf, ppx, rth0,
    mass1 (+ tkea, tke3d when TKE is prognostic).  ``g`` is the cm1x Grid
    (built from the array shapes when omitted).  With ``terrain_flag`` the
    terrain metrics ``gz``, ``dzdx``, ``dzdy`` give w(k=1) via ``bcwsfc``.
    """
    xp = bk.xp
    S = bk.asarray
    shp_s = th0.shape
    nk = shp_s[0] - 2
    ng = 3
    nj, ni = shp_s[1] - 2 * ng, shp_s[2] - 2 * ng
    K, J, I = slice(1, nk + 1), slice(ng, ng + nj), slice(ng, ng + ni)
    if g is None:
        g = Grid(ni, nj, nk, float(cfg.dx), float(cfg.dy), float(cfg.dz))
    if cfg.psolver > 3:
        raise NotImplementedError("getset: only psolver <= 3 ported")

    cp, rd, rv, p00 = S(C.cp), S(C.rd), S(C.rv), S(C.p00)
    cpdrd = cp / rd

    ua = _bc.bcu(bk, g, ua, cfg)
    va = _bc.bcv(bk, g, va, cfg)
    wa = _bc.bcw(bk, g, wa, 1, cfg)
    ppi = _bc.bcs(bk, g, ppi, cfg)
    tha = _bc.bcs(bk, g, tha, cfg)
    if qa.shape[0]:
        qa = xp.stack([_bc.bcs(bk, g, qa[n], cfg) for n in range(qa.shape[0])])
    # bcu(u0)/bcv(v0)/bcs(th0): base-state fields are horizontally homogeneous
    # and already filled on every halo point, so these leave them unchanged.
    if cfg.get("terrain_flag", False):
        wa = _bc.bcwsfc(bk, g, gz=gz, dzdx=dzdx, dzdy=dzdy, u=ua, v=va, w=wa)
        wa = bk.set(wa, (1,), _bc.bc2d(bk, g, wa[1], cfg))
    tke = None
    if int(cfg.get("cm1setup", 0)) in (1, 4) and int(cfg.get("sgsmodel", 0)) == 1:
        # tkea = 0 (init3d.F; the initial TKE of testcase 3, 4, 5, 9, 11:
        # cm1x.testcase.initial_tke, with bcw); bcw(tkea,1);
        # tkea(k=1)=tkea(k=2); bc2d(tkea(k=1))
        from .testcase import initial_tke

        tke0 = None
        if zh is not None and zf is not None:
            tke0 = initial_tke(
                cfg, zh=bk.to_numpy(zh), zf=bk.to_numpy(zf), shape=tuple(g.w_shape)
            )
        elif int(cfg.get("testcase", 0)) in (3, 4, 5, 9, 11):
            raise ValueError("getset: the testcase initial TKE needs zh and zf")
        if tke0 is None:
            tke = bk.zeros(g.w_shape)
        else:
            tke = _bc.bcw(bk, g, S(tke0), 1, cfg)  # init3d.F bcw(tkea,1)
        tke = _bc.bcw(bk, g, tke, 1, cfg)
        tke = bk.set(tke, (1, J, I), tke[2, J, I])
        tke = bk.set(tke, (1,), _bc.bc2d(bk, g, tke[1], cfg))
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
    rho = _bc.bcs(bk, g, rho, cfg)  # k = 1..nk only (k=0, nk+1 stay 0)

    rr = bk.set(bk.zeros(shp_s), (K, J, I), S(1.0) / rho[K, J, I])
    rr = _bc.bcs(bk, g, rr, cfg)
    rf_i = c1[K, J, I] * rho[0:nk, J, I] + c2[K, J, I] * rho[K, J, I]
    rf = bk.set(bk.zeros(shp_s), (K, J, I), rf_i)
    rf = _bc.bcs(bk, g, rf, cfg)
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
    mass1 = qp_sum(bk, level_sums(bk, w))  # CM1's summation order

    ppx = bk.set(bk.zeros(shp_s), (K, J, I), ppi[K, J, I])
    ppx = _bc.bcs(bk, g, ppx, cfg)

    out = dict(
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
    if tke is not None:
        out["tkea"] = tke
        out["tke3d"] = bk.copy(tke)
    return out
