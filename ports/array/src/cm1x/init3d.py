"""Initial conditions (port of src/init3d.F: ``init3d`` and ``getset``).

Scope: ``iinit = 0`` (no perturbation) or ``1`` (warm bubble), ``irandp = 0`` or ``1``
(testcase=0: +-0.25 K theta noise from gfortran's random_number with CM1's seed),
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

__all__ = ["init3d", "getset", "GfortranRandom", "randp_theta"]

_M64 = (1 << 64) - 1


def _rotl64(x, k):
    return ((x << k) | (x >> (64 - k))) & _M64


class GfortranRandom:
    """gfortran ``random_number`` for real(8) (xoshiro256**, top 53 bits
    scaled by 2**-64), seeded like init3d.F's ``random_seed(put=...)``."""

    _XOR = (
        0xBD0C5B6E50C2DF49,
        0xD46061CD46E1DF38,
        0xBB4F4D4ED6103544,
        0x114A583D0756AD39,
    )

    def __init__(self, put=None):
        if put is None:
            k = 8
            put = [
                int(round(2.0e9 * (2.0 * (n - 1) / (k - 1) - 1.0)))
                for n in range(1, k + 1)
            ]
        rev = [int(v) & 0xFFFFFFFF for v in reversed(put)]
        seed = [rev[2 * i] | (rev[2 * i + 1] << 32) for i in range(4)]
        self.s = [seed[i] ^ self._XOR[i] for i in range(4)]

    def next(self):
        s = self.s
        result = (_rotl64((s[1] * 5) & _M64, 7) * 9) & _M64
        t = (s[1] << 17) & _M64
        s[2] ^= s[0]
        s[3] ^= s[1]
        s[1] ^= s[2]
        s[0] ^= s[3]
        s[2] ^= t
        s[3] = _rotl64(s[3], 45)
        return float(result & (_M64 << 11) & _M64) * 2.0**-64


def randp_theta(nk, nj, ni, ng):
    """init3d.F irandp=1 (testcase=0) theta noise as a numpy [k,j,i] array
    with halos: one leading draw, then (rand, rand2) per point of
    (0:nx+2, 0:ny+2, 1:nk), i fastest; amplitude*(2*rand-1)."""
    import numpy as np

    rng = GfortranRandom()
    rng.next()
    out = np.zeros((nk + 2, nj + 2 * ng, ni + 2 * ng))
    amp = 0.25
    for k in range(1, nk + 1):
        for jj in range(0, nj + 3):
            for ii in range(0, ni + 3):
                r = rng.next()
                rng.next()
                out[k, jj + ng - 1, ii + ng - 1] = amp * (2.0 * r - 1.0)
    return out


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

    if int(cfg.get("irandp", 0)) not in (0, 1):
        raise NotImplementedError(f"irandp={cfg.irandp} not ported")
    if int(cfg.get("testcase", 0)) != 0 and int(cfg.get("irandp", 0)) == 1:
        raise NotImplementedError("irandp=1 needs testcase=0")

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

    # ---------------- random perturbations (irandp=1, testcase=0)
    if int(cfg.get("irandp", 0)) == 1:
        tha = tha + S(randp_theta(nk, nj, ni, ng))

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
    if int(cfg.get("testcase", 0)) in (3, 4, 5, 9, 11):
        raise NotImplementedError("getset: TKE initial profiles of testcases")

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
        # tkea = 0 (init3d.F); bcw(tkea,1); tkea(k=1)=tkea(k=2); bc2d(tkea(k=1))
        tke = _bc.bcw(bk, g, bk.zeros(g.w_shape), 1, cfg)
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
