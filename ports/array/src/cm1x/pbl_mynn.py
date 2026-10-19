"""MYNN-EDMF boundary layer as CM1 runs it (ipbl=4: level 2.5, ipbl=5:
level 3), ported from src/module_bl_mynn.F, module_bl_mynn_common.F and
module_bl_mynn_wrapper.F (mynnedmf_wrapper_run), with the CM1 glue of
src/turb.F (the get_vars packing and the domynn block).

Every column is processed at once: a column variable is a Python list
indexed by the Fortran level k (``a[k]``, k = 1..nk(+1); ``a[0]`` unused)
whose elements are 1-D arrays over the columns, so the k loops are Python
loops in the Fortran order and every operation is vectorised over columns.
The mass-flux plumes of DMP_mf carry an extra leading plume axis
(``[NUP, ncol]``).  Data-dependent loop exits (GET_PBLH, boulac_length,
mym_length, DMP_mf, the condensation iteration, ...) are masks; the search
loops of boulac_length run over the offset izz-iz for all starting levels at
once.  The order of operations follows the Fortran (and the bit-for-bit C++
translation, cm1k/physics/pbl_mynn).

Configuration (param.F fixes these whenever ipbl=4/5): bl_mynn_cloudpdf=2,
icloud_bl=1, spp_pbl=0, bl_mynn_tkebudget=1, bl_mynn_mixlength=1,
bl_mynn_edmf=1, bl_mynn_edmf_mom=1, bl_mynn_edmf_tke=0, bl_mynn_cloudmix=1,
bl_mynn_mixscalars=0, bl_mynn_mixqt=0, bl_mynn_output=1,
bl_mynn_tkeadvect=.true.; the module fixes bl_mynn_topdown=0,
bl_mynn_edmf_dd=0, dheat_opt=1, env_subs=.false., bl_mynn_stfunc=1,
scaleaware=1, CKmod=1; turb.F passes FLAG_QC=FLAG_QI=.true., the qni it
passes is 0.  bl_mynn_edmf and bl_mynn_edmf_mom may also be 0 here
(``cfg`` keys ``bl_mynn_edmf`` / ``bl_mynn_edmf_mom``, default 1).

Not ported (unreachable from CM1): DDMF, topdown_cloudrad, mynn_mix_chem,
mym_condensation cases 0/1, mym_length cases 0/2, scalar mixing, mixqt=1,
SPP, restart/cycling, the env_subs branch of DMP_mf (sub_*/det_* stay 0),
cm1setup=4.
"""

from __future__ import annotations

import math

# module_model_constants / module_bl_mynn_common / module_bl_mynn
r_d, cp, karman, g, p1000mb, r_v = 287.04, 1005.7, 0.4, 9.81, 100000.0, 461.5
rcp = r_d / cp
xls, xlv = 2.834e6, 2.501e6
xlf = xls - xlv
svpt0 = 273.15
rvovrd = r_v / r_d
cpv, cliq, cice = 4.0 * r_v, 4190.0, 2106.0
p608 = rvovrd - 1.0
onethird, twothirds = 1.0 / 3.0, 2.0 / 3.0
tref, tice, grav, t0c = 300.0, 240.0, g, svpt0
gtr = grav / tref
tv0 = p608 * tref
xlscp, xlvcp = (xlv + xlf) / cp, xlv / cp

pr, g1, b1, b2, c2, c3, c5 = 0.74, 0.235, 24.0, 15.0, 0.729, 0.340, 0.2
a1 = b1 * (1.0 - 3.0 * g1) / 6.0
c1 = g1 - 1.0 / (3.0 * a1 * 2.88449914061481660)
a2 = a1 * (g1 - c1) / (g1 * pr)
g2 = b2 / b1 * (1.0 - c3) + 2.0 * a1 / b1 * (3.0 - 2.0 * c2)
cc2, cc3 = 1.0 - c2, 1.0 - c3
e1c = 3.0 * a2 * b2 * cc3
e2c = 9.0 * a1 * a2 * cc2
e3c = 9.0 * a2 * a2 * cc2 * (1.0 - c5)
e4c = 12.0 * a1 * a2 * cc2
e5c = 6.0 * a1 * a1
qmin, zmax, Sqfac = 0.0, 1.0, 3.0
qkemin = 1.0e-12
tliq = 269.0
cphm_unst = cphh_unst = 16.0
_J = (
    0.611583699e03,
    0.444606896e02,
    0.143177157e01,
    0.264224321e-1,
    0.299291081e-3,
    0.203154182e-5,
    0.702620698e-8,
    0.379534310e-11,
    -0.321582393e-13,
)
_K = (
    0.609868993e03,
    0.499320233e02,
    0.184672631e01,
    0.402737184e-1,
    0.565392987e-3,
    0.521693933e-5,
    0.307839583e-7,
    0.105785160e-9,
    0.161444444e-12,
)
EXP_M1 = 0.36787944117144233  # EXP(-1.), folded by gfortran
NUP = 10  # DMP_mf plumes
am_st, bm_st = 6.1, 2.5
rbm_st = 1.0 / bm_st
ah_st, bh_st = 5.3, 1.1
rbh_st = 1.0 / bh_st
am_unst, ah_unst = 10.0, 34.0


def _opt(cfg, name, default):
    get = getattr(cfg, "get", None)
    return get(name, default) if get else getattr(cfg, name, default)


def _pow(xp, x, p):
    """x**p through pow() as the Fortran calls it: an array exponent keeps
    numpy off its scalar fast paths (x**0.5 -> sqrt)."""
    return x ** (xp.zeros_like(x) + p)


def _tanh(xp, x):
    """glibc tanh on numpy (numpy's own tanh differs from it in the last
    bit); the backend's tanh elsewhere."""
    if xp.__name__ in ("numpy", "array_api_compat.numpy"):
        import numpy as np

        a = np.asarray(x)
        return np.frompyfunc(math.tanh, 1, 1)(a).astype(a.dtype)
    return xp.tanh(x)


def _mx(xp, a, b):
    return xp.where(a > b, a, b)


def _mn(xp, a, b):
    return xp.where(a < b, a, b)


def check(cfg):
    ipbl = int(_opt(cfg, "ipbl", 0))
    if ipbl not in (4, 5):
        return
    if int(_opt(cfg, "cm1setup", 2)) == 4:
        raise NotImplementedError(
            "pbl_mynn: cm1setup=4 (qke zeroed in the LES subdomain) not ported"
        )
    for k in ("bl_mynn_edmf", "bl_mynn_edmf_mom"):
        if int(_opt(cfg, k, 1)) not in (0, 1):
            raise NotImplementedError(f"pbl_mynn: {k} must be 0 or 1")


# ---------------------------------------------------------------------------
# thermodynamics


def _poly(XC, c):
    r = c[8]
    for a in c[7::-1]:
        r = a + XC * r
    return r


def qsat_blend(xp, t, P):
    XC = _mx(xp, -80.0, t - t0c)
    ESL = _poly(XC, _J)
    ESI = _poly(XC, _K)
    RSLF = 0.622 * ESL / _mx(xp, P - ESL, 1e-5)
    RSIF = 0.622 * ESI / _mx(xp, P - ESI, 1e-5)
    chi = (t0c - t) / (t0c - tice)
    blend = (1.0 - chi) * RSLF + chi * RSIF
    return xp.where(t >= t0c, RSLF, xp.where(t <= tice, RSIF, blend))


def xl_blend(xp, t):
    xlvt = xlv + (cpv - cliq) * (t - t0c)
    xlst = xls + (cpv - cice) * (t - t0c)
    chi = (t0c - t) / (t0c - tice)
    blend = (1.0 - chi) * xlvt + chi * xlst
    return xp.where(t >= t0c, xlvt, xp.where(t <= tice, xlst, blend))


# ---------------------------------------------------------------------------
# similarity functions (bl_mynn_stfunc = 1)


def _phi_unstable(xp, zet, c_unst, a_unst, m):
    """The zet < 0 branch of phim (m=True) / phih."""
    if m:
        d0 = (1.0 - c_unst * zet) ** 0.25
        phi = 1.0 / d0
        psi = (
            2.0 * xp.log(0.5 * (1.0 + d0))
            + xp.log(0.5 * (1.0 + d0 * d0))
            - 2.0 * xp.atan(d0)
            + 1.570796
        )
    else:
        d0 = _pow(xp, 1.0 - c_unst * zet, 0.5)
        phi = 1.0 / d0
        psi = 2.0 * xp.log(0.5 * (1.0 + d0))
    d0 = 1.0 - a_unst * zet
    d1 = d0**0.333333
    d11 = -0.33333 * a_unst * d0**-0.6666667
    d2 = 0.33333 * (d1 * d1 + d1 + 1.0)
    d22 = 0.3333 * d11 * (2.0 * d1 + 1.0)
    d3 = 0.57735 * (2.0 * d1 + 1.0)
    d33 = 1.1547 * d11
    d4 = 1.5 * xp.log(d2) - 1.73205 * xp.atan(d3) + 1.813799364
    d44 = (1.5 / d2) * d22 - 1.73205 * d33 / (1.0 + d3 * d3)
    d0 = zet * zet
    d1 = 1.0 / (1.0 + d0)
    d11 = 2.0 * zet
    d2 = ((1 - phi) / zet + d11 * d4 + d0 * d44) * d1
    d22 = -d11 * (psi + d0 * d4) * (d1 * d1)
    return 1.0 - zet * (d2 + d22)


def _phi_stable(xp, zet, a, b, rb):
    d0 = 1 + zet**b
    d1 = zet + d0**rb
    d11 = 1 + d0 ** (rb - 1) * zet ** (b - 1)
    d2 = (-a / d1) * d11
    return 1 - zet * d2


def phim(xp, zet):
    st = zet >= 0.0
    s = _phi_stable(xp, xp.where(st, zet, 0.0), am_st, bm_st, rbm_st)
    u = _phi_unstable(xp, xp.where(st, -1.0, zet), cphm_unst, am_unst, True)
    return xp.where(st, s, u)


def phih(xp, zet):
    st = zet >= 0.0
    s = _phi_stable(xp, xp.where(st, zet, 0.0), ah_st, bh_st, rbh_st)
    u = _phi_unstable(xp, xp.where(st, -1.0, zet), cphh_unst, ah_unst, False)
    return xp.where(st, s, u)


# ---------------------------------------------------------------------------


def tridiag2(n, a, b, c, d):
    """Thomas algorithm, k = 1..n; returns x (list indexed by k)."""
    cp_ = [None] * (n + 1)
    dp = [None] * (n + 1)
    cp_[1] = c[1] / b[1]
    dp[1] = d[1] / b[1]
    for i in range(2, n + 1):
        m = b[i] - cp_[i - 1] * a[i]
        cp_[i] = c[i] / m
        dp[i] = (d[i] - dp[i - 1] * a[i]) / m
    x = [None] * (n + 2)
    x[n] = dp[n]
    for i in range(n - 1, 0, -1):
        x[i] = dp[i] - cp_[i] * x[i + 1]
    return x


def get_pblh(xp, nk, thetav, qke, zw, dz, landsea):
    """GET_PBLH: returns (zi, kzi) (kzi as a float array)."""
    sbl_lim, sbl_damp = 200.0, 400.0
    one = xp.ones_like(thetav[1])
    minthv = 9.0e9 * one
    for k in range(2, nk + 1):  # (the Fortran may read k = nk+1 if zw(nk+1) <= 200 m)
        m = zw[k] <= 200.0
        minthv = xp.where(m & (minthv > thetav[k]), thetav[k], minthv)
    delt_thv = xp.where((landsea - 1.5) >= 0, xp.ones_like(landsea), 1.25)
    zi = 0.0 * one
    done = one < 0
    for k in range(2, nk):
        th = minthv + delt_thv
        c = thetav[k] >= th
        den = _mx(xp, thetav[k] - thetav[k - 1], 1e-6)
        new = xp.where(c, zw[k] - dz[k - 1] * _mn(xp, (thetav[k] - th) / den, 1.0), zi)
        if k == nk - 1:
            new = zw[2] + 0.0 * one
        zi = xp.where(done, zi, new)
        done = done | (zi != 0.0)
    maxqke = _mx(xp, qke[1], 0.0)
    TKEeps = maxqke / 40.0
    TKEeps = _mx(xp, TKEeps, 0.02)
    ptke = 0.0 * one
    done = one < 0
    for k in range(2, nk):
        qtke = _mx(xp, qke[k] / 2.0, 0.0)
        qtkem1 = _mx(xp, qke[k - 1] / 2.0, 0.0)
        c = qtke <= TKEeps
        v = zw[k] - dz[k - 1] * _mn(
            xp, (TKEeps - qtke) / _mx(xp, qtkem1 - qtke, 1e-6), 1.0
        )
        v = _mx(xp, v, zw[2])
        new = xp.where(c, v, ptke)
        if k == nk - 1:
            new = zw[2] + 0.0 * one
        ptke = xp.where(done, ptke, new)
        done = done | (ptke != 0.0)
    ptke = _mn(xp, ptke, zi + 350.0)
    ptke = _mx(xp, ptke, _mx(xp, zi - 350.0, 10.0))
    wt = 0.5 * _tanh(xp, (zi - sbl_lim) / sbl_damp) + 0.5
    zi = xp.where(maxqke <= 0.05, zi, ptke * (1.0 - wt) + zi * wt)
    kzi = 2.0 * one
    done = one < 0
    for k in range(2, nk):
        c = (zw[k] >= zi) & ~done
        kzi = xp.where(c, float(k - 1), kzi)
        done = done | c
    return zi, kzi


def scale_aware(xp, dx, PBL1):
    dxdh = max(2.5 * dx, 10.0) / _mn(xp, PBL1, 3000.0)
    d6 = dxdh**0.667
    Psig_bl = ((dxdh * dxdh) + 0.106 * d6) / ((dxdh * dxdh) + 0.066 * d6 + 0.071)
    dxdh = max(2.5 * dx, 10.0) / _mn(xp, PBL1 + 500.0, 3500.0)
    d6 = dxdh**0.667
    Psig_shcu = ((dxdh * dxdh) + 0.145 * d6) / ((dxdh * dxdh) + 0.172 * d6 + 0.170)
    Psig_bl = _mx(xp, _mn(xp, Psig_bl, 1.0), 0.0)
    Psig_shcu = _mx(xp, _mn(xp, Psig_shcu, 1.0), 0.0)
    return Psig_bl, Psig_shcu


def mym_level2(xp, nk, dz, u, v, thl, qw, vt, vq, sm, sh):
    """Returns dtl, dqw, dtv, gm, gh (k=2..nk) and sm, sh (k=2..nk
    replaced; k=1 kept)."""
    dtl, dqw, dtv, gm, gh = ([None] * (nk + 1) for _ in range(5))
    sm, sh = list(sm), list(sh)
    rfc = g1 / (g1 + g2)
    for k in range(2, nk + 1):
        dzk = 0.5 * (dz[k] + dz[k - 1])
        afk = dz[k] / (dz[k] + dz[k - 1])
        abk = 1.0 - afk
        du = u[k] - u[k - 1]
        dv = v[k] - v[k - 1]
        duz = du * du + dv * dv
        duz = duz / (dzk * dzk)
        dtz = (thl[k] - thl[k - 1]) / dzk
        dqz = (qw[k] - qw[k - 1]) / dzk
        vtt = 1.0 + vt[k] * abk + vt[k - 1] * afk
        vqq = tv0 + vq[k] * abk + vq[k - 1] * afk
        dtq = vtt * dtz + vqq * dqz
        dtl[k], dqw[k], dtv[k], gm[k] = dtz, dqz, dtq, duz
        gh[k] = -dtq * gtr
        ri = -gh[k] / _mx(xp, duz, 1.0e-10)
        a2fac = 1.0 / (1.0 + _mx(xp, ri, 0.0))
        f1 = (
            b1 * (g1 - c1)
            + 3.0 * a2 * a2fac * (1.0 - c2) * (1.0 - c5)
            + 2.0 * a1 * (3.0 - 2.0 * c2)
        )
        f2 = b1 * (g1 + g2) - 3.0 * a1 * (1.0 - c2)
        rf1 = b1 * (g1 - c1) / f1
        rf2 = b1 * g1 / f2
        smc = a1 / (a2 * a2fac) * f1 / f2
        shc = 3.0 * (a2 * a2fac) * (g1 + g2)
        ri1 = 0.5 / smc
        ri2 = rf1 * smc
        ri3 = 4.0 * rf2 * smc - 2.0 * ri2
        ri4 = ri2 * ri2
        rf = _mn(xp, ri1 * (ri + ri2 - xp.sqrt(ri * ri - ri3 * ri + ri4)), rfc)
        sh[k] = shc * (rfc - rf) / (1.0 - rf)
        sm[k] = smc * (rf1 - rf) / (rf2 - rf) * sh[k]
    return dtl, dqw, dtv, gm, gh, sm, sh


def boulac_length(xp, nk, zw, dz, qtke, theta):
    """boulac_length's lb2 (elBLavg), k = 1..nk.  The up- and downward
    searches run over the offset |izz - iz| for every starting level iz at
    once (rows of [nk, ncol] stacks)."""
    Lmax = 2000.0
    beta = gtr
    Th = xp.stack([theta[k] for k in range(1, nk + 1)])
    Dz = xp.stack([dz[k] for k in range(1, nk + 1)])
    Qt = xp.stack([qtke[k] for k in range(1, nk + 1)])
    Zw = xp.stack([zw[k] for k in range(1, nk + 2)])
    rows = xp.reshape(xp.astype(xp.arange(nk), Th.dtype), (nk,) + (1,) * (Th.ndim - 1))
    # padded copies: theta(izz), dz(izz) for izz up to 2nk / down to -nk
    Thp = xp.concat([Th] + [Th[-1:]] * (nk + 1), axis=0)
    Dzp = xp.concat([Dz] + [Dz[-1:]] * (nk + 1), axis=0)
    P = nk + 1
    Thf = xp.concat([Th[:1]] * P + [Th], axis=0)
    Dzf = xp.concat([Dz[:1]] * P + [Dz], axis=0)
    zero = 0.0 * Th
    # upward
    zup = zero
    dlu = Zw[nk : nk + 1] - Zw[:nk] - Dz * 0.5
    zzz = zero
    zup_inf = zero
    found = (rows + zero) >= nk - 1  # iz = nk: no search
    for m in range(nk - 1):
        valid = rows <= nk - 2 - m  # izz = iz + m <= nk - 1
        act = valid & ~found
        dzt = Dzp[m : m + nk]
        th_z = Thp[m : m + nk]
        th_z1 = Thp[m + 1 : m + 1 + nk]
        zu = zup - beta * Th * dzt
        zu = zu + beta * (th_z1 + th_z) * dzt * 0.5
        zz = zzz + dzt
        c = (Qt < zu) & (Qt >= zup_inf)
        bbb = (th_z1 - th_z) / dzt
        nz = bbb != 0.0
        bs = xp.where(nz, bbb, 1.0)
        x = beta * (th_z - Th)
        tl1 = (
            (
                -beta * (th_z - Th)
                + xp.sqrt(_mx(xp, 0.0, x * x + 2.0 * bs * beta * (Qt - zup_inf)))
            )
            / bs
            / beta
        )
        ne = th_z != Th
        den = xp.where(ne, beta * (th_z - Th), 1.0)
        tl2 = xp.where(ne, (Qt - zup_inf) / den, 0.0)
        tl = xp.where(nz, tl1, tl2)
        hit = act & c
        dlu = xp.where(hit, zz - dzt + tl, dlu)
        found = found | hit
        zup = xp.where(act, zu, zup)
        zzz = xp.where(act, zz, zzz)
        zup_inf = xp.where(act, zu, zup_inf)
    # downward
    zdo = zero
    zdo_sup = zero
    dld = Zw[:nk] + zero
    zzz = zero
    found = (rows + zero) < 1  # iz = 1: no search
    for m in range(nk - 1):
        valid = rows >= m + 1  # izz = iz - m >= 2
        act = valid & ~found
        dzt = Dzf[P - m - 1 : P - m - 1 + nk]  # dz(izz-1)
        th_z = Thf[P - m : P - m + nk]  # theta(izz)
        th_zm = Thf[P - m - 1 : P - m - 1 + nk]  # theta(izz-1)
        zd = zdo + beta * Th * dzt
        zd = zd - beta * (th_zm + th_z) * dzt * 0.5
        zz = zzz + dzt
        c = (Qt < zd) & (Qt >= zdo_sup)
        bbb = (th_z - th_zm) / dzt
        nz = bbb != 0.0
        bs = xp.where(nz, bbb, 1.0)
        x = beta * (th_z - Th)
        tl1 = (
            (
                beta * (th_z - Th)
                + xp.sqrt(_mx(xp, 0.0, x * x + 2.0 * bs * beta * (Qt - zdo_sup)))
            )
            / bs
            / beta
        )
        ne = th_z != Th
        den = xp.where(ne, beta * (th_z - Th), 1.0)
        tl2 = xp.where(ne, (Qt - zdo_sup) / den, 0.0)
        tl = xp.where(nz, tl1, tl2)
        hit = act & c
        dld = xp.where(hit, zz - dzt + tl, dld)
        found = found | hit
        zdo = xp.where(act, zd, zdo)
        zzz = xp.where(act, zz, zzz)
        zdo_sup = xp.where(act, zd, zdo_sup)
    dld = _mn(xp, dld, Zw[1 : nk + 1])
    dlu = _mx(xp, 0.1, _mn(xp, dlu, 1000.0))
    dld = _mx(xp, 0.1, _mn(xp, dld, 1000.0))
    lb = xp.sqrt(dlu * dld)
    lb = lb / (1.0 + (lb / Lmax))
    lb2 = [None] + [lb[k - 1] for k in range(1, nk + 1)]
    lb2[nk] = lb2[nk - 1]
    return lb2


def mym_length(
    xp, nk, dz, zw, rmo, flt, u1, v1, qke, dtv, zi, theta, Psig_bl, edmf_w1, edmf_a1
):
    """bl_mynn_mixlength = 1.  Returns (el, qkw)."""
    ugrid = xp.sqrt(u1[1] * u1[1] + v1[1] * v1[1])
    uonset = 15.0
    wt_u = 1.0 - _mn(xp, _mx(xp, ugrid - uonset, 0.0) / 30.0, 0.5)
    cns, alp1, alp2, alp4, alp5, alp6 = 3.5, 0.22, 0.25, 5.0, 0.3, 50.0
    alp3 = 2.0 * wt_u
    zi2 = _mx(xp, zi, 200.0)
    h1 = _mx(xp, 0.3 * zi2, 200.0)
    h1 = _mn(xp, h1, 500.0)
    h2 = h1 / 2.0
    qtke = [None] * (nk + 1)
    thetaw = [None] * (nk + 1)
    qkw = [None] * (nk + 1)
    qtke[1] = _mx(xp, 0.5 * qke[1], 0.01)
    thetaw[1] = theta[1]
    qkw[1] = xp.sqrt(_mx(xp, qke[1], 1.0e-10))
    for k in range(2, nk + 1):
        afk = dz[k] / (dz[k] + dz[k - 1])
        abk = 1.0 - afk
        qkw[k] = xp.sqrt(_mx(xp, qke[k] * abk + qke[k - 1] * afk, 1.0e-3))
        qtke[k] = 0.5 * (qkw[k] * qkw[k])
        thetaw[k] = theta[k] * abk + theta[k - 1] * afk
    elt = 1.0e-5 + 0.0 * zi
    vsc = 1.0e-5 + 0.0 * zi
    for k in range(2, nk + 1):
        m = zw[k] <= zi2 + h1
        dzk = 0.5 * (dz[k] + dz[k - 1])
        qdz = _mn(xp, _mx(xp, qkw[k] - qmin, 0.03), 30.0) * dzk
        elt = xp.where(m, elt + qdz * zw[k], elt)
        vsc = xp.where(m, vsc + qdz, vsc)
    elt = _mn(xp, _mx(xp, alp1 * elt / vsc, 10.0), 400.0)
    vflx = flt
    vsc = (gtr * elt * _mx(xp, vflx, 0.0)) ** onethird
    el = [None] * (nk + 1)
    el[1] = 0.0 * zi
    elBLavg = boulac_length(xp, nk, zw, dz, qtke, thetaw)
    stab_r = rmo > 0.0
    rmo_u = xp.where(stab_r, 0.0, rmo)
    for k in range(2, nk + 1):
        zwk = zw[k]
        st = dtv[k] > 0.0
        bv = _mx(xp, xp.sqrt(gtr * xp.where(st, dtv[k], 1.0)), 0.001)
        mfw = alp6 * edmf_a1[k - 1] * edmf_w1[k - 1]
        elb = (
            _mx(xp, alp2 * qkw[k], mfw) / bv * (1.0 + alp3 * xp.sqrt(vsc / (bv * elt)))
        )
        elb = _mn(xp, elb, zwk)
        elf = 0.65 * qkw[k] / bv
        elbl = xp.where(st, _mx(xp, elBLavg[k], mfw / bv), elBLavg[k])
        elb = xp.where(st, elb, 1.0e10)
        elf = xp.where(st, elf, 1.0e10)
        els_s = karman * zwk / (1.0 + cns * _mn(xp, zwk * rmo, zmax))
        els_u = karman * zwk * (1.0 - alp4 * zwk * rmo_u) ** 0.2
        els = xp.where(stab_r, els_s, els_u)
        wt = 0.5 * _tanh(xp, (zwk - (zi2 + h1)) / h2) + 0.5
        e = xp.sqrt(
            (els * els)
            / (1.0 + ((els * els) / (elt * elt)) + ((els * els) / (elb * elb)))
        )
        e = _mn(xp, e, elf)
        e = e * (1.0 - wt) + alp5 * elbl * wt
        el[k] = e * Psig_bl
    return el, qkw


def mym_turbulence(
    xp,
    nk,
    closure,
    dz,
    zw,
    u,
    v,
    thl,
    qw,
    qke,
    tsq,
    qsq,
    cov,
    vt,
    vq,
    rmo,
    flt,
    zi,
    theta,
    sh,
    sm,
    Psig_bl,
    cldfra_bl1D,
    edmf_w1,
    edmf_a1,
):
    """closure 2.5 / 3.0 (spp_pbl = 0, TKEprodTD = 0).  Returns a dict with
    sh, sm, el, dfm, dfh, dfq, tcd, qcd, pdk, pdt, pdq, pdc, qSHEAR, qBUOY."""
    Prlimit = 5.0
    dtl, dqw, dtv, gm, gh, sm, sh = mym_level2(
        xp, nk, dz, u, v, thl, qw, vt, vq, sm, sh
    )
    el, qkw = mym_length(
        xp, nk, dz, zw, rmo, flt, u, v, qke, dtv, zi, theta, Psig_bl, edmf_w1, edmf_a1
    )
    L = nk + 1
    dfm, dfh, dfq, tcd, qcd, pdk, pdt, pdq, pdc, qS, qB = (
        [None] * L for _ in range(11)
    )
    for k in range(2, nk + 1):
        dzk = 0.5 * (dz[k] + dz[k - 1])
        afk = dz[k] / (dz[k] + dz[k - 1])
        abk = 1.0 - afk
        elsq = el[k] * el[k]
        q3sq = qkw[k] * qkw[k]
        q2sq = b1 * elsq * (sm[k] * gm[k] + sh[k] * gh[k])
        shk = _mx(xp, sh[k], 1e-5)
        smk = sm[k]
        du = u[k] - u[k - 1]
        dv = v[k] - v[k - 1]
        duz = du * du + dv * dv
        duz = duz / (dzk * dzk)
        ri = -gh[k] / _mx(xp, duz, 1.0e-10)
        a2fac = 1.0 / (1.0 + _mx(xp, ri, 0.0))
        gmel = gm[k] * elsq
        ghel = gh[k] * elsq
        dlsq = elsq
        q3sq = xp.where(q3sq / dlsq < -gh[k], -dlsq * gh[k], q3sq)
        low = q3sq < q2sq
        # q3sq < q2sq
        qd = xp.sqrt(q3sq / xp.where(low, q2sq, 1.0))
        qd2 = qd * qd
        e1a = q3sq - e1c * ghel * a2fac * qd2
        e2a = q3sq - e2c * ghel * a2fac * qd2
        e3a = e1a + e3c * ghel * (a2fac * a2fac) * qd2
        e4a = e1a - e4c * ghel * a2fac * qd2
        edena = _mx(xp, e2a * e4a + e3a * e5c * gmel * qd2, 1.0e-20)
        # else
        e1b = q3sq - e1c * ghel * a2fac
        e2b = q3sq - e2c * ghel * a2fac
        e3b = e1b + e3c * ghel * (a2fac * a2fac)
        e4b = e1b - e4c * ghel * a2fac
        edenb = _mx(xp, e2b * e4b + e3b * e5c * gmel, 1.0e-20)
        smb = q3sq * a1 * (e3b - 3.0 * c1 * e4b) / edenb
        shb = q3sq * (a2 * a2fac) * (e2b + 3.0 * c1 * e5c * gmel) / edenb
        qdiv = xp.where(low, qd, 1.0)
        shk = xp.where(low, shk * qd, shb)
        smk = xp.where(low, smk * qd, smb)
        e1 = xp.where(low, e1a, e1b)
        e2 = xp.where(low, e2a, e2b)
        eden = xp.where(low, edena, edenb)
        shk = xp.where(shk > 4.0, 4.0, shk)
        shk = xp.where(shk < 0.0, 0.0, shk)
        smk = _mn(xp, smk, Prlimit * shk)
        if closure >= 3.0:
            t2sq = qdiv * b2 * elsq * shk * (dtl[k] * dtl[k])
            r2sq = qdiv * b2 * elsq * shk * (dqw[k] * dqw[k])
            c2sq = qdiv * b2 * elsq * shk * dtl[k] * dqw[k]
            t3sq = _mx(xp, tsq[k] * abk + tsq[k - 1] * afk, 0.0)
            r3sq = _mx(xp, qsq[k] * abk + qsq[k - 1] * afk, 0.0)
            c3sq = cov[k] * abk + cov[k - 1] * afk
            c3sq = xp.copysign(_mn(xp, xp.abs(c3sq), xp.sqrt(t3sq * r3sq)), c3sq)
            vtt = 1.0 + vt[k] * abk + vt[k - 1] * afk
            vqq = tv0 + vq[k] * abk + vq[k - 1] * afk
            t2sq = vtt * t2sq + vqq * c2sq
            r2sq = vtt * c2sq + vqq * r2sq
            c2sq = _mx(xp, vtt * t2sq + vqq * r2sq, 0.0)
            t3sq = vtt * t3sq + vqq * c3sq
            r3sq = vtt * c3sq + vqq * r3sq
            c3sq = _mx(xp, vtt * t3sq + vqq * r3sq, 0.0)
            qq = qdiv * qdiv
            cw25 = e1 * (e2 + 3.0 * c1 * e5c * gmel * qq) / (3.0 * eden)
            dlsq = elsq
            q3sq = xp.where(q3sq / dlsq < -gh[k], -dlsq * gh[k], q3sq)
            e2 = q3sq - e2c * ghel * a2fac * qq
            e3 = q3sq + e3c * ghel * (a2fac * a2fac) * qq
            e4 = q3sq - e4c * ghel * a2fac * qq
            eden = e2 * e4 + e3 * e5c * gmel * qq
            cc3gtr2 = cc3 * (gtr * gtr)
            wden = (
                cc3gtr2
                * (dlsq * dlsq)
                / elsq
                * qq
                * (e2 * e4c * a2fac - e3c * e5c * gmel * (a2fac * a2fac) * qq)
            )
            nz = wden != 0.0
            ws = xp.where(nz, wden, 1.0)
            clow = q3sq * (0.12 - cw25) * eden / ws
            cupp = q3sq * (0.76 - cw25) * eden / ws
            cpos = _mn(xp, _mx(xp, c3sq, c2sq + clow), c2sq + cupp)
            cneg = _mx(xp, _mn(xp, c3sq, c2sq + clow), c2sq + cupp)
            c3sq = xp.where(nz, xp.where(wden > 0.0, cpos, cneg), c3sq)
            e1 = e2 + e5c * gmel * qq
            eden = _mx(xp, eden, 1.0e-20)
            e6c = 3.0 * (a2 * a2fac) * cc3 * gtr * dlsq / elsq
            en = qdiv * e6c * (t3sq - t2sq)
            enumr = xp.where(t2sq >= 0.0, _mx(xp, en, 0.0), _mn(xp, en, 0.0))
            gamt = -e1 * enumr / eden
            en = qdiv * e6c * (r3sq - r2sq)
            enumr = xp.where(r2sq >= 0.0, _mx(xp, en, 0.0), _mn(xp, en, 0.0))
            gamq = -e1 * enumr / eden
            enumr = _mx(xp, qdiv * e6c * (c3sq - c2sq), 0.0)
            smd = (
                dlsq
                * enumr
                * gtr
                / eden
                * qq
                * (e3c * (a2fac * a2fac) + e4c * a2fac)
                * a1
                / (a2 * a2fac)
            )
            gamv = e1 * enumr * gtr / eden
            smk = smk + smd
            qdiv = 1.0 + 0.0 * qdiv
        else:
            gamt = gamq = gamv = 0.0
        cldavg = 0.5 * (cldfra_bl1D[k - 1] + cldfra_bl1D[k])
        mf = (edmf_a1[k] > 0.001) | (cldavg > 0.02)
        aw = 0.03 * _mn(xp, 10.0 * edmf_a1[k] * edmf_w1[k], 1.0)
        cf = 0.05 * _mn(xp, cldavg, 1.0)
        smk = xp.where(mf, _mx(xp, _mx(xp, smk, aw), cf), smk)
        shk = xp.where(mf, _mx(xp, _mx(xp, shk, aw), cf), shk)
        sm[k], sh[k] = smk, shk
        elq = el[k] * qkw[k]
        elh = elq * qdiv
        pdk[k] = elq * (smk * gm[k] + shk * gh[k] + gamv)
        pdt[k] = elh * (shk * dtl[k] + gamt) * dtl[k]
        pdq[k] = elh * (shk * dqw[k] + gamq) * dqw[k]
        pdc[k] = (
            elh * (shk * dtl[k] + gamt) * dqw[k] * 0.5
            + elh * (shk * dqw[k] + gamq) * dtl[k] * 0.5
        )
        tcd[k] = elq * gamt
        qcd[k] = elq * gamq
        dfm[k] = elq * smk / dzk
        dfh[k] = elq * shk / dzk
        dfq[k] = dfm[k]
        qS[k] = elq * smk * gm[k]
        qB[k] = elq * (shk * gh[k] + gamv)
    z = 0.0 * zi
    dfm[1] = dfh[1] = dfq[1] = tcd[1] = qcd[1] = z
    tcd[nk] = qcd[nk] = z
    tcd = [None] + [t + z for t in tcd[1:]]
    qcd = [None] + [t + z for t in qcd[1:]]
    for k in range(1, nk):
        tcd[k] = (tcd[k + 1] - tcd[k]) / dz[k]
        qcd[k] = (qcd[k + 1] - qcd[k]) / dz[k]
    return {
        "sh": sh,
        "sm": sm,
        "el": el,
        "dfm": dfm,
        "dfh": dfh,
        "dfq": dfq,
        "tcd": tcd,
        "qcd": qcd,
        "pdk": pdk,
        "pdt": pdt,
        "pdq": pdq,
        "pdc": pdc,
        "qSHEAR": qS,
        "qBUOY": qB,
    }


def mym_predict(
    xp,
    nk,
    closure,
    delt,
    dz,
    ust,
    pmz,
    el,
    dfq,
    rho,
    pdk,
    pdt,
    pdq,
    pdc,
    qke,
    tsq,
    qsq,
    cov,
    s_aw,
):
    """bl_mynn_edmf_tke = 0 (onoff = 0), tke_budget = 1.  Returns qke,
    tsq, qsq, cov, qWT, qDISS (lists)."""
    L = nk + 2
    pdk, pdt, pdq, pdc = list(pdk), list(pdt), list(pdq), list(pdc)
    qkw, df3q, dtz, rhoz, rhoinv, kqdz, kmdz = ([None] * L for _ in range(7))
    vkz = karman * 0.5 * dz[1]
    for k in range(1, nk + 1):
        qkw[k] = xp.sqrt(_mx(xp, qke[k], 0.0))
        df3q[k] = Sqfac * dfq[k]
        dtz[k] = delt / dz[k]
    rhoz[1] = rho[1]
    rhoinv[1] = 1.0 / rho[1]
    kqdz[1] = rhoz[1] * df3q[1]
    kmdz[1] = rhoz[1] * dfq[1]
    for k in range(2, nk + 1):
        rz = (rho[k] * dz[k - 1] + rho[k - 1] * dz[k]) / (dz[k - 1] + dz[k])
        rhoz[k] = _mx(xp, rz, 1e-4)
        rhoinv[k] = 1.0 / _mx(xp, rho[k], 1e-4)
        kqdz[k] = rhoz[k] * df3q[k]
        kmdz[k] = rhoz[k] * dfq[k]
    rhoz[nk + 1] = rhoz[nk]
    kqdz[nk + 1] = rhoz[nk + 1] * df3q[nk]
    kmdz[nk + 1] = rhoz[nk + 1] * dfq[nk]
    for k in range(2, nk):
        kqdz[k] = _mx(xp, kqdz[k], 0.5 * s_aw[k])
        kqdz[k] = _mx(xp, kqdz[k], -0.5 * (s_aw[k] - s_aw[k + 1]))
        kmdz[k] = _mx(xp, kmdz[k], 0.5 * s_aw[k])
        kmdz[k] = _mx(xp, kmdz[k], -0.5 * (s_aw[k] - s_aw[k + 1]))
    pdk1 = 2.0 * ((ust * ust) * ust) * pmz / vkz
    pdk[1] = pdk1 - pdk[2]
    pdt[1] = pdt[2]
    pdq[1] = pdq[2]
    pdc[1] = pdc[2]
    bp, rp = [None] * L, [None] * L
    a, b, c, d = ([None] * L for _ in range(4))
    for k in range(1, nk):
        b1l = b1 * 0.5 * (el[k + 1] + el[k])
        bp[k] = 2.0 * qkw[k] / b1l
        rp[k] = pdk[k + 1] + pdk[k]
    for k in range(1, nk):
        a[k] = -dtz[k] * kqdz[k] * rhoinv[k]
        b[k] = 1.0 + dtz[k] * (kqdz[k] + kqdz[k + 1]) * rhoinv[k] + bp[k] * delt
        c[k] = -dtz[k] * kqdz[k + 1] * rhoinv[k]
        d[k] = rp[k] * delt + qke[k]
    z = 0.0 * ust
    a[nk], b[nk], c[nk], d[nk] = z, z + 1.0, z, qke[nk]
    x = tridiag2(nk, a, b, c, d)
    qke = [None] * L
    for k in range(1, nk + 1):
        qke[k] = _mn(xp, _mx(xp, x[k], 1.0e-4), 150.0)
    # tke_budget = 1
    tke_up = [None] + [0.5 * qke[k] for k in range(1, nk + 1)]
    qWT = [None] * L
    qWT[1] = (1.0 / dz[1]) * (kqdz[2] * (tke_up[2] - tke_up[1]) - kqdz[1] * tke_up[1])
    for k in range(2, nk):
        qWT[k] = (1.0 / dz[k]) * (
            kqdz[k + 1] * (tke_up[k + 1] - tke_up[k])
            - kqdz[k] * (tke_up[k] - tke_up[k - 1])
        )
    qWT[nk] = (1.0 / dz[nk]) * (-kqdz[nk] * (tke_up[nk] - tke_up[nk - 1]))
    qDISS = [None] * L
    for k in range(1, nk):
        qDISS[k] = bp[k] * tke_up[k]
    qDISS[nk] = z
    tsq, qsq, cov = list(tsq), list(qsq), list(cov)

    def solve(pd, s, bdiv):
        for k in range(1, nk):
            b2l = b2 * 0.5 * (el[k + 1] + el[k])
            bp[k] = 2.0 * qkw[k] / b2l
            rp[k] = pd[k + 1] + pd[k]
        for k in range(1, nk):
            a[k] = -dtz[k] * kmdz[k] * rhoinv[k]
            b[k] = 1.0 + dtz[k] * (kmdz[k] + kmdz[k + 1]) * rhoinv[k] + bp[k] * delt
            c[k] = -dtz[k] * kmdz[k + 1] * rhoinv[k]
            d[k] = rp[k] * delt + s[k]
        a[nk], b[nk], c[nk], d[nk] = z - 1.0, z + 1.0, z, z
        return tridiag2(nk, a, b, c, d)

    def diag(pd):
        out = [None] * L
        for k in range(1, nk):
            pos = qkw[k] > 0.0
            b2l = xp.where(
                pos, b2 * 0.25 * (el[k + 1] + el[k]) / xp.where(pos, qkw[k], 1.0), 0.0
            )
            out[k] = b2l * (pd[k + 1] + pd[k])
        out[nk] = out[nk - 1]
        return out

    if closure > 2.5:
        x = solve(pdq, qsq, None)
        for k in range(1, nk + 1):
            qsq[k] = _mx(xp, x[k], 1e-17)
    else:
        qsq = diag(pdq)
    if closure >= 3.0:
        x = solve(pdt, tsq, None)
        tsq = [None] + [x[k] for k in range(1, nk + 1)]
        x = solve(pdc, cov, None)
        cov = [None] + [x[k] for k in range(1, nk + 1)]
    else:
        tsq = diag(pdt)
        cov = diag(pdc)
    return qke, tsq, qsq, cov, qWT, qDISS


def mym_condensation(xp, nk, dz, qw, qc, p, exner, qsq, th):
    """bl_mynn_cloudpdf = 2.  Returns qc_bl, qi_bl, cldfra_bl, vt, vq."""
    L = nk + 1
    # k_tropo: first k (from nk-3 down to 1) with the tropopause criterion
    one = xp.ones_like(th[1])
    kt = 0.0 * one
    done = one < 0
    for k in range(nk - 3, 0, -1):
        ht1 = 44307.692 * (1.0 - (p[k] / 101325.0) ** 0.190)
        ht2 = 44307.692 * (1.0 - (p[k + 2] / 101325.0) ** 0.190)
        cnd = (
            (((th[k + 2] - th[k]) / (ht2 - ht1)) < 10.0 / 1500.0)
            & (ht1 < 19000.0)
            & (ht1 > 4000.0)
        )
        hit = cnd & ~done
        kt = xp.where(hit, float(k), kt)
        done = done | hit
    k_tropo = _mx(xp, kt + 2.0, 3.0)
    qc_bl, qi_bl, cld, vt, vq = ([None] * L for _ in range(5))
    zagl = 0.0 * one
    for k in range(1, nk):
        zagl = zagl + dz[k]
        t = th[k] * exner[k]
        xl = xl_blend(xp, t)
        qsat_tk = qsat_blend(xp, t, p[k])
        rh = _mx(xp, _mn(xp, 1.0, qw[k] / _mx(xp, 1.0e-8, qsat_tk)), 0.001)
        rsl = xl * qsat_tk / (r_v * (t * t))
        cpm = cp + qw[k] * cpv
        a = 1.0 / (1.0 + xl * rsl / cpm)
        b = a * rsl
        qmq = qw[k] - qsat_tk
        r3sq = _mx(xp, qsq[k], 0.0)
        sgm = xp.sqrt(r3sq)
        sgm = _mn(xp, sgm, qsat_tk * 0.666)
        sgm = _mx(xp, sgm, qsat_tk * 0.035)
        q1 = qmq / sgm
        q1k = q1
        cf = _mx(xp, 0.0, _mn(xp, 1.0, 0.5 + 0.36 * xp.atan(1.8 * (q1 + 0.2))))
        e_neg = sgm * xp.exp(1.2 * _mn(xp, q1k, 0.0) - 1)
        e_mid = sgm * (EXP_M1 + 0.66 * q1k + 0.086 * (q1k * q1k))
        e_big = sgm * q1k
        ql = xp.where(q1k < 0.0, e_neg, xp.where(q1k > 2.0, e_big, e_mid))
        ql_water = xp.where(qc[k] > 1.0e-6, 0.5 * (ql + qc[k]), ql)
        ql_ice = ql
        small = cf < 0.01
        ql_ice = xp.where(small, 0.0, ql_ice)
        ql_water = xp.where(small, 0.0, ql_water)
        cf = xp.where(small, 0.0, cf)
        liq_frac = _mn(xp, 1.0, _mx(xp, 0.0, (t - tice) / (tliq - tice)))
        qcb = liq_frac * ql_water
        qib = (1.0 - liq_frac) * ql_ice
        trop = float(k) >= k_tropo - 1.0
        cf = xp.where(trop, 0.0, cf)
        qc_bl[k] = xp.where(trop, 0.0, qcb)
        qi_bl[k] = xp.where(trop, 0.0, qib)
        q1k = _mx(xp, q1, -2.0)
        fng = xp.where(
            q1k >= 1.0,
            1.0,
            xp.where(
                q1k >= -1.7,
                xp.exp(-0.4 * (q1k - 1.0)),
                xp.where(
                    q1k >= -2.5,
                    3.0 + xp.exp(-3.8 * (q1k + 1.7)),
                    _mn(xp, 23.9 + xp.exp(-1.6 * (q1k + 2.5)), 60.0),
                ),
            ),
        )
        cfmax = _mn(xp, cf, 0.5)
        bb = b * t / th[k]
        qww = 1.0 + 0.61 * qw[k]
        alpha = 0.61 * th[k]
        beta = (th[k] / t) * (xl / cp) - 1.61 * th[k]
        vt[k] = qww - cfmax * beta * bb * fng - 1.0
        vq[k] = alpha + cfmax * beta * a * fng - tv0
        fac_damp = _mn(xp, zagl * 0.0025, 1.0)
        rhx = _mx(xp, 0.0, (rh - 0.92)) / 0.145
        cld_factor = 1.0 + fac_damp * _mn(xp, rhx * rhx, 0.4)
        cld[k] = _mn(xp, 1.0, cld_factor * cf)
    vt[nk] = vt[nk - 1]
    vq[nk] = vq[nk - 1]
    z = 0.0 * one
    qc_bl[nk] = qi_bl[nk] = cld[nk] = z
    return qc_bl, qi_bl, cld, vt, vq


def condensation_edmf(bk, QT, THL, P, zagl, QC):
    """Returns (THV, QC); the 50-pass iteration is masked after convergence."""
    xp = bk.xp
    niter, diff = 50, 1.0e-6
    EXN = (P / p1000mb) ** rcp

    def body(i, c):
        qc, act = c
        T = EXN * THL + xlvcp * qc
        QS = qsat_blend(xp, T, P)
        new = 0.5 * qc + 0.5 * _mx(xp, QT - QS, 0.0)
        qcn = xp.where(act, new, qc)
        act = act & ~(xp.abs(new - qc) < diff)
        return qcn, act

    QC, _ = bk.fori_loop(niter, body, (QC, xp.ones_like(QC) > 0))
    T = EXN * THL + xlvcp * QC
    QS = qsat_blend(xp, T, P)
    QC = _mx(xp, QT - QS, 0.0)
    QC = xp.where(zagl < 100.0, 0.0, QC)
    THV = (THL + xlvcp * QC) * (1.0 + QT * (rvovrd - 1.0) - rvovrd * QC)
    return THV, QC


def dmp_mf(
    bk, nk, zw, dz, p, rho, momentum_opt, u, v, w, th, thl, thv, tk, qt, qc, exner, vt, vq,
    flt, fltv, flq, pblh, dx, landsea, ts, qc_bl1d, cldfra_bl1d, Psig_shcu,
):  # fmt: skip
    """DMP_mf (env_subs = .false., tke_opt = 0, scalar_opt = 0, no chem,
    spp_pbl = 0).  Returns a dict with edmf_a, edmf_w, edmf_qt, edmf_thl,
    edmf_ent, edmf_qc, s_aw, s_awthl, s_awqt, s_awqv, s_awqc, s_awu, s_awv,
    qc_bl1d, cldfra_bl1d, vt, vq, nup2, ktop, maxmf."""
    xp = bk.xp
    Atot, dl, dcut, cf_thresh, fluxportion = 0.10, 100.0, 1.2, 0.5, 0.75
    L = nk + 2
    one = xp.ones_like(pblh)
    zero = 0.0 * one
    qc_bl1d, cldfra_bl1d, vt, vq = list(qc_bl1d), list(cldfra_bl1d), list(vt), list(vq)
    Ii = xp.reshape(
        xp.astype(xp.arange(1, NUP + 1), pblh.dtype), (NUP,) + (1,) * pblh.ndim
    )
    zp = 0.0 * Ii * one  # [NUP, ncol] zeros
    UPW, UPTHL, UPQT, UPQC, UPA, UPU, UPV = ([zp] * L for _ in range(7))
    ENT = [zp + 0.001] * L
    edmf_a, edmf_w, edmf_qt, edmf_thl, edmf_ent, edmf_qc = (
        [zero] * L for _ in range(6)
    )
    s_aw, s_awthl, s_awqt, s_awqv, s_awqc, s_awu, s_awv = ([zero] * L for _ in range(7))
    maxw = zero
    cloud_base = zero + 9000.0
    k50 = one
    for k in range(1, nk):
        inr = ~(zw[k] > pblh + 500.0)  # zw increases: the Fortran exit
        wpbl = xp.where(w[k] < 0.0, 2.0 * w[k], w[k])
        maxw = xp.where(inr, _mx(xp, maxw, xp.abs(wpbl)), maxw)
        k50 = xp.where(inr & (zw[k] <= 50.0), float(k), k50)
        qc_sgs = _mx(xp, qc[k], qc_bl1d[k] * cldfra_bl1d[k])
        cb = inr & (qc_sgs > 1e-5) & (cloud_base == 9000.0)
        cloud_base = xp.where(cb, 0.5 * (zw[k] + zw[k + 1]), cloud_base)
    maxw = _mx(xp, 0.0, maxw - 1.0)
    Psig_w = _mx(xp, 0.0, 1.0 - maxw)
    Psig_w = _mn(xp, Psig_w, Psig_shcu)
    fltv2 = xp.where((Psig_w == 0.0) & (fltv > 0.0), -1.0 * fltv, fltv)
    hux = xp.where((landsea - 1.5) >= 0, xp.zeros_like(landsea) - 0.001, -0.005)
    kmax = _mx(xp, k50 - 1.0, 1.0)
    superad = one < 0
    act = one > 0
    for k in range(1, nk):
        inr = float(k) <= kmax
        if k == 1:
            cnd = (th[1] - ts) / (0.5 * dz[1]) < hux
        else:
            cnd = (th[k] - th[k - 1]) / (0.5 * (dz[k] + dz[k - 1])) < hux
        superad = xp.where(act & inr, cnd, superad)
        act = act & inr & cnd
    n0 = int(dx * dcut / dl)
    nup2 = max(min(n0, NUP), 1)
    maxwidth = 1.1 * pblh
    maxwidth = _mn(xp, maxwidth, 0.5 * cloud_base)
    wspd_pbl = xp.sqrt(_mx(xp, u[1] * u[1] + v[1] * v[1], 0.01))
    wfl = xp.where(
        (landsea - 1.5) < 0,
        1000.0 * (0.6 * _tanh(xp, (flt - 0.040) / 0.03) + 0.5),
        1000.0 * (0.6 * _tanh(xp, (flt - 0.003) / 0.01) + 0.5),
    )
    width_flx = _mx(xp, _mn(xp, wfl, 1000.0), 0.0)
    maxwidth = _mn(xp, maxwidth, width_flx)
    n2 = xp.trunc((maxwidth - xp.remainder(maxwidth, 100.0)) / 100)
    n2 = xp.where(n2 < 0.0, 0.0, n2)
    nup2 = xp.where(n2 < nup2, n2, float(nup2) + zero)
    fire = (fltv2 > 0.002) & (nup2 >= 1.0) & superad
    nup2 = xp.where(fire, nup2, 0.0)
    ktop = zero
    maxmf = zero
    # plume initialisation
    dd = -1.9
    cn = zero
    for i in range(1, NUP + 1):
        l_ = dl * i
        cn = xp.where(float(i) <= nup2, cn + l_**dd * (l_ * l_) / (dx * dx) * dl, cn)
    Cc = Atot / xp.where(cn > 0.0, cn, 1.0)
    acfac = xp.where(
        (landsea - 1.5) < 0,
        0.5 * _tanh(xp, (fltv2 - 0.02) / 0.05) + 0.5,
        0.5 * _tanh(xp, (fltv2 - 0.01) / 0.03) + 0.5,
    )
    ac_wsp = 1.0 - _mn(xp, _mx(xp, wspd_pbl - 20.0, 0.0), 10.0) / 10.0
    ac_cld = _mn(xp, cloud_base / 1200.0, 1.0)
    acfac = acfac * _mn(xp, ac_wsp, ac_cld)
    lI = dl * Ii
    on0 = Ii <= nup2  # [NUP, ncol]
    ldd = xp.reshape(
        bk.asarray([math.pow(dl * i, dd) for i in range(1, NUP + 1)]), lI.shape
    )
    upa1 = Cc * ldd * lI * lI / (dx * dx) * dl
    UPA[1] = xp.where(on0, upa1 * acfac, 0.0)
    z0, pwmin, pwmax = 50.0, 0.1, 0.4
    fp = xp.where(fire, fltv2, 1.0)
    wstar = _mx(xp, 1.0e-2, (gtr * fp * pblh) ** onethird)
    qstar = _mx(xp, flq, 1.0e-5) / wstar
    thstar = flt / wstar
    csigma = 1.34
    exc_fac = xp.where(
        (landsea - 1.5) >= 0, 0.58 * 4.0 * _mn(xp, cloud_base / 1000.0, 1.0), 0.58
    )
    zr = (z0 / pblh) ** onethird
    sigmaW = csigma * wstar * zr * (1 - 0.8 * z0 / pblh)
    sigmaQT = csigma * qstar * zr
    sigmaTH = csigma * thstar * zr
    wmin = _mn(xp, sigmaW * pwmin, 0.1)
    wmax = _mn(xp, sigmaW * pwmax, 0.5)
    upw1 = wmin + Ii / float(NUP) * (wmax - wmin)
    UPW[1] = xp.where(on0, upw1, 0.0)
    UPU[1] = xp.where(on0, (u[1] * dz[2] + u[2] * dz[1]) / (dz[1] + dz[2]) + zp, 0.0)
    UPV[1] = xp.where(on0, (v[1] * dz[2] + v[2] * dz[1]) / (dz[1] + dz[2]) + zp, 0.0)
    exc_heat = exc_fac * upw1 * sigmaTH / sigmaW
    UPTHL[1] = xp.where(
        on0, (thl[1] * dz[2] + thl[2] * dz[1]) / (dz[1] + dz[2]) + exc_heat, 0.0
    )
    exc_moist = exc_fac * upw1 * sigmaQT / sigmaW
    UPQT[1] = xp.where(
        on0, (qt[1] * dz[2] + qt[2] * dz[1]) / (dz[1] + dz[2]) + exc_moist, 0.0
    )
    # plume ascent, all plumes at once
    act = on0 & fire
    QCn = zp
    any_ev = one < 0
    for k in range(2, nk):
        wminl = 0.3 + lI * 0.0005
        ent = 0.33 / (_mn(xp, _mx(xp, UPW[k - 1], wminl), 0.9) * lI)
        ent = _mx(xp, ent, 0.0003)
        zt = _mn(xp, pblh + 1500.0, 4000.0)
        ent = xp.where(zw[k] >= zt, ent + (zw[k] - zt) * 5.0e-6, ent)
        ent = _mn(xp, ent, 0.9 / (zw[k + 1] - zw[k]))
        ENT[k] = xp.where(act, ent, ENT[k])
        EntExp = ent * (zw[k + 1] - zw[k])
        EntExm = EntExp * 0.3333
        QTn = UPQT[k - 1] * (1.0 - EntExp) + qt[k] * EntExp
        THLn = UPTHL[k - 1] * (1.0 - EntExp) + thl[k] * EntExp
        Un = UPU[k - 1] * (1.0 - EntExm) + u[k] * EntExm
        Vn = UPV[k - 1] * (1.0 - EntExm) + v[k] * EntExm
        Pk = (p[k] * dz[k + 1] + p[k + 1] * dz[k]) / (dz[k + 1] + dz[k])
        THVn, qcn = condensation_edmf(bk, QTn, THLn, Pk, zw[k + 1], QCn)
        QCn = xp.where(act, qcn, QCn)
        THVk = (thv[k] * dz[k + 1] + thv[k + 1] * dz[k]) / (dz[k + 1] + dz[k])
        B = grav * (THVn / THVk - 1.0)
        BCOEFF = xp.where(B > 0.0, xp.zeros_like(B) + 0.15, 0.2)
        upw = UPW[k - 1]
        den = xp.where(upw < 0.2, _mx(xp, upw, 0.2), upw)
        Wn = upw + (-2.0 * ent * upw + BCOEFF * B / den) * _mn(
            xp, zw[k] - zw[k - 1], 250.0
        )
        lim = _mn(xp, 1.25 * (zw[k] - zw[k - 1]) / 200.0, 2.0)
        Wn = xp.where(Wn > upw + lim, upw + lim, Wn)
        Wn = xp.where(Wn < upw - lim, upw - lim, Wn)
        Wn = _mn(xp, _mx(xp, Wn, 0.0), 3.0)
        if k == 2:
            # a plume that stops at k=2 sets nup2 = 0: the plumes after it
            # are not computed
            ev = act & (Wn == 0.0)
            prior = []
            seen = one < 0
            for i in range(NUP):
                prior.append(seen)
                seen = seen | ev[i]
            act = act & ~xp.stack(prior) & ~ev
            any_ev = seen
        Wn = Wn * xp.exp(
            -_mx(xp, zw[k + 1] - _mn(xp, pblh + 2000.0, 3500.0), 0.0) / 1000.0
        )
        com = act & (Wn > 0.0)
        UPW[k] = xp.where(com, Wn, UPW[k])
        UPTHL[k] = xp.where(com, THLn, UPTHL[k])
        UPQT[k] = xp.where(com, QTn, UPQT[k])
        UPQC[k] = xp.where(com, QCn, UPQC[k])
        UPU[k] = xp.where(com, Un, UPU[k])
        UPV[k] = xp.where(com, Vn, UPV[k])
        UPA[k] = xp.where(com, UPA[k - 1], UPA[k])
        anyc = xp.any(com, axis=0)
        ktop = xp.where(anyc, _mx(xp, ktop, float(k)), ktop)
        act = com
    nup2 = xp.where(any_ev, 0.0, nup2)
    ktop = _mn(xp, ktop, float(nk - 1))
    # fluxes (nup2 > 0)
    for i in range(1, NUP + 1):
        for k in range(1, nk):
            m = (float(i) <= nup2) & (float(k) <= ktop)
            rho_int = (rho[k] * dz[k + 1] + rho[k + 1] * dz[k]) / (dz[k + 1] + dz[k])
            aw = rho_int * UPA[k][i - 1] * UPW[k][i - 1]
            s_aw[k + 1] = xp.where(m, s_aw[k + 1] + aw * Psig_w, s_aw[k + 1])
            s_awthl[k + 1] = xp.where(
                m, s_awthl[k + 1] + aw * UPTHL[k][i - 1] * Psig_w, s_awthl[k + 1]
            )
            s_awqt[k + 1] = xp.where(
                m, s_awqt[k + 1] + aw * UPQT[k][i - 1] * Psig_w, s_awqt[k + 1]
            )
            qcp = xp.where((qc[k] > 1e-12) | (qc[k + 1] > 1e-12), UPQC[k][i - 1], 0.0)
            s_awqc[k + 1] = xp.where(
                m, s_awqc[k + 1] + aw * qcp * Psig_w, s_awqc[k + 1]
            )
            if momentum_opt > 0:
                s_awu[k + 1] = xp.where(
                    m, s_awu[k + 1] + aw * UPU[k][i - 1] * Psig_w, s_awu[k + 1]
                )
                s_awv[k + 1] = xp.where(
                    m, s_awv[k + 1] + aw * UPV[k][i - 1] * Psig_w, s_awv[k + 1]
                )
            s_awqv[k + 1] = xp.where(m, s_awqt[k + 1] - s_awqc[k + 1], s_awqv[k + 1])
    has = nup2 > 0.0
    dzi1 = 0.5 * (dz[1] + dz[2])
    flx1 = xp.where(
        s_aw[2] != 0.0, _mx(xp, s_aw[2] * (th[1] - th[2]) / dzi1, 1.0e-5), 0.0
    )
    adj = has & (flx1 > fluxportion * flt / dz[1]) & (flx1 > 0.0)
    adjustment = fluxportion * flt / dz[1] / xp.where(adj, flx1, 1.0)

    def sc(a):
        return xp.where(adj, a * adjustment, a)

    s_aw = [None] + [sc(a) for a in s_aw[1:]]
    s_awthl = [None] + [sc(a) for a in s_awthl[1:]]
    s_awqt = [None] + [sc(a) for a in s_awqt[1:]]
    s_awqc = [None] + [sc(a) for a in s_awqc[1:]]
    s_awqv = [None] + [sc(a) for a in s_awqv[1:]]
    if momentum_opt > 0:
        s_awu = [None] + [sc(a) for a in s_awu[1:]]
        s_awv = [None] + [sc(a) for a in s_awv[1:]]
    UPA = [None] + [sc(a) for a in UPA[1:]]
    for k in range(1, nk):
        mk = has & (float(k) <= ktop)
        rho_int = (rho[k] * dz[k + 1] + rho[k + 1] * dz[k]) / (dz[k + 1] + dz[k])
        ea, ew, eq, et, ee, ec = (
            edmf_a[k],
            edmf_w[k],
            edmf_qt[k],
            edmf_thl[k],
            edmf_ent[k],
            edmf_qc[k],
        )
        for i in range(1, NUP + 1):
            m = mk & (float(i) <= nup2)
            ua = UPA[k][i - 1]
            ea = xp.where(m, ea + ua, ea)
            ew = xp.where(m, ew + rho_int * ua * UPW[k][i - 1], ew)
            eq = xp.where(m, eq + rho_int * ua * UPQT[k][i - 1], eq)
            et = xp.where(m, et + rho_int * ua * UPTHL[k][i - 1], et)
            ee = xp.where(m, ee + rho_int * ua * ENT[k][i - 1], ee)
            ec = xp.where(m, ec + rho_int * ua * UPQC[k][i - 1], ec)
        pos = mk & (ea > 0.0)
        eas = xp.where(pos, ea, 1.0)
        ew = xp.where(pos, ew / eas, ew)
        eq = xp.where(pos, eq / eas, eq)
        et = xp.where(pos, et / eas, et)
        ee = xp.where(pos, ee / eas, ee)
        ec = xp.where(pos, ec / eas, ec)
        ea = xp.where(pos, ea * Psig_w, ea)
        maxmf = xp.where(pos & (ea * ew > maxmf), ea * ew, maxmf)
        edmf_a[k], edmf_w[k], edmf_qt[k], edmf_thl[k], edmf_ent[k], edmf_qc[k] = (
            ea,
            ew,
            eq,
            et,
            ee,
            ec,
        )
    dzi = [None] * L
    for k in range(1, nk):
        dzi[k] = 0.5 * (dz[k] + dz[k + 1])
    for k in range(2, nk - 1):
        mk = has & (float(k) <= ktop) & (0.5 * (edmf_qc[k] + edmf_qc[k - 1]) > 0.0)
        Aup = (edmf_a[k] * dzi[k - 1] + edmf_a[k - 1] * dzi[k]) / (dzi[k - 1] + dzi[k])
        QTp = (edmf_qt[k] * dzi[k - 1] + edmf_qt[k - 1] * dzi[k]) / (
            dzi[k - 1] + dzi[k]
        )
        QCp = xp.where(
            (edmf_qc[k] > 0.0) & (edmf_qc[k - 1] > 0.0),
            (edmf_qc[k] * dzi[k - 1] + edmf_qc[k - 1] * dzi[k]) / (dzi[k - 1] + dzi[k]),
            _mx(xp, edmf_qc[k], edmf_qc[k - 1]),
        )
        xl = xl_blend(xp, tk[k])
        qsat_tk = qsat_blend(xp, tk[k], p[k])
        rsl = xl * qsat_tk / (r_v * (tk[k] * tk[k]))
        cpm = cp + qt[k] * cpv
        a = 1.0 / (1.0 + xl * rsl / cpm)
        b9 = a * rsl
        q2p = xlvcp / exner[k]
        pt = thl[k] + q2p * QCp * Aup
        bb = b9 * tk[k] / pt
        qww = 1.0 + 0.61 * qt[k]
        alpha = 0.61 * pt
        beta = pt * xl / (tk[k] * cp) - 1.61 * pt
        sigq = 10.0 * Aup * (QTp - qt[k])
        sigq = _mx(xp, sigq, qsat_tk * 0.02)
        sigq = _mn(xp, sigq, qsat_tk * 0.25)
        qmq = a * (qt[k] - qsat_tk)
        Q1 = qmq / sigq
        mf_cf = _mn(xp, _mx(xp, 0.5 + 0.36 * xp.atan(1.55 * Q1), 0.01), 0.6)
        water = (landsea - 1.5) >= 0
        mf_cf = _mx(xp, mf_cf, xp.where(water, 1.2 * Aup, 1.75 * Aup))
        mf_cf = _mn(xp, mf_cf, 5.0 * Aup)
        qa = QCp * Aup
        qcb = xp.where(qa > 5e-5, 1.86 * qa - 2.2e-5, 1.18 * qa)
        qcb = xp.where(mf_cf >= Aup, qcb / xp.where(mf_cf != 0.0, mf_cf, 1.0), qcb)
        upd = mk & (~water | (cldfra_bl1d[k] < cf_thresh))
        qc_bl1d[k] = xp.where(upd, qcb, qc_bl1d[k])
        cldfra_bl1d[k] = xp.where(upd, mf_cf, cldfra_bl1d[k])
        m2 = mk & (cldfra_bl1d[k] < cf_thresh)
        Q1 = _mx(xp, Q1, -2.25)
        fng = xp.where(
            Q1 >= 1.0,
            1.0,
            xp.where(
                Q1 >= -1.7,
                xp.exp(-0.4 * (Q1 - 1.0)),
                xp.where(
                    Q1 >= -2.5,
                    3.0 + xp.exp(-3.8 * (Q1 + 1.7)),
                    _mn(xp, 23.9 + xp.exp(-1.6 * (Q1 + 2.5)), 60.0),
                ),
            ),
        )
        vt[k] = xp.where(m2, qww - (1.5 * Aup) * beta * bb * fng - 1.0, vt[k])
        vq[k] = xp.where(m2, alpha + (1.5 * Aup) * beta * a * fng - tv0, vq[k])
    maxqc = zero - 1.0e300
    for k in range(1, nk):
        maxqc = xp.where((float(k) <= ktop) & (edmf_qc[k] > maxqc), edmf_qc[k], maxqc)
    maxmf = xp.where((ktop > 0.0) & (maxqc < 1.0e-8), -1.0 * maxmf, maxmf)
    return {
        "edmf_a": edmf_a,
        "edmf_w": edmf_w,
        "edmf_qt": edmf_qt,
        "edmf_thl": edmf_thl,
        "edmf_ent": edmf_ent,
        "edmf_qc": edmf_qc,
        "s_aw": s_aw,
        "s_awthl": s_awthl,
        "s_awqt": s_awqt,
        "s_awqv": s_awqv,
        "s_awqc": s_awqc,
        "s_awu": s_awu,
        "s_awv": s_awv,
        "qc_bl1d": qc_bl1d,
        "cldfra_bl1d": cldfra_bl1d,
        "vt": vt,
        "vq": vq,
        "nup2": nup2,
        "ktop": ktop,
        "maxmf": maxmf,
    }


def moisture_check(xp, nk, delt, dp, exner, qv, qc, qi, th, dqv, dqc, dqi):
    """Returns qv, qc, qi, th, dqv, dqc, dqi (lists)."""
    qvmin, qcmin, qimin = 1e-20, 0.0, 0.0
    qv, qc, qi, th, dqv, dqc, dqi = (list(a) for a in (qv, qc, qi, th, dqv, dqc, dqi))
    dqv2 = None
    for k in range(nk, 0, -1):
        dqc2 = _mx(xp, 0.0, qcmin - qc[k])
        dqi2 = _mx(xp, 0.0, qimin - qi[k])
        dqc[k] = dqc[k] + dqc2 / delt
        dqi[k] = dqi[k] + dqi2 / delt
        dqv[k] = dqv[k] - (dqc2 + dqi2) / delt
        qc[k] = qc[k] + dqc2
        qi[k] = qi[k] + dqi2
        qv[k] = qv[k] - dqc2 - dqi2
        th[k] = th[k] + xlvcp / exner[k] * dqc2 + xlscp / exner[k] * dqi2
        dqv2 = _mx(xp, 0.0, qvmin - qv[k])
        dqv[k] = dqv[k] + dqv2 / delt
        qv[k] = qv[k] + dqv2
        if k != 1:
            qv[k - 1] = qv[k - 1] - dqv2 * dp[k] / dp[k - 1]
            dqv[k - 1] = dqv[k - 1] - dqv2 * dp[k] / dp[k - 1] / delt
        qv[k] = _mx(xp, qv[k], qvmin)
        qc[k] = _mx(xp, qc[k], qcmin)
        qi[k] = _mx(xp, qi[k], qimin)
    big = dqv2 > 1.0e-20
    s = 0.0 * dqv2
    for k in range(1, nk + 1):
        s = xp.where(qv[k] > 2.0 * qvmin, s + qv[k] * dp[k], s)
    aa = dqv2 * dp[1] / _mx(xp, 1.0e-20, s)
    go = big & (aa < 0.5)
    for k in range(1, nk + 1):
        m = go & (qv[k] > 2.0 * qvmin)
        dum = aa * qv[k]
        qv[k] = xp.where(m, qv[k] - dum, qv[k])
        dqv[k] = xp.where(m, dqv[k] - dum / delt, dqv[k])
    return qv, qc, qi, th, dqv, dqc, dqi


def mynn_tendencies(xp, nk, delt, psfc, ust, flt, flqv, flqc, wspd, edmf_mom, t):
    """bl_mynn_mixqt = 0, bl_mynn_cloudmix = 1, FLAG_QC = FLAG_QI, no
    scalars, uoce = voce = 0, sd_* = sub_* = det_* = 0.  ``t``: dict of the
    column inputs.  Returns du, dv, dth, dqv, dqc, dqi (lists)."""
    L = nk + 2
    dz, rho, p, dfh, dfm, s_aw = (
        t["dz"],
        t["rho"],
        t["p"],
        t["dfh"],
        t["dfm"],
        t["s_aw"],
    )
    onoff = 0.0 if edmf_mom == 0 else 1.0
    rhosfc = psfc / (r_d * (t["tk"][1] + p608 * t["qv"][1]))
    dtz, rhoz, rhoinv, khdz, kmdz, delp = ([None] * L for _ in range(6))
    dtz[1] = delt / dz[1]
    rhoz[1] = rho[1]
    rhoinv[1] = 1.0 / rho[1]
    khdz[1] = rhoz[1] * dfh[1]
    kmdz[1] = rhoz[1] * dfm[1]
    delp[1] = psfc - (p[2] * dz[1] + p[1] * dz[2]) / (dz[1] + dz[2])
    for k in range(2, nk + 1):
        dtz[k] = delt / dz[k]
        rz = (rho[k] * dz[k - 1] + rho[k - 1] * dz[k]) / (dz[k - 1] + dz[k])
        rhoz[k] = _mx(xp, rz, 1e-4)
        rhoinv[k] = 1.0 / _mx(xp, rho[k], 1e-4)
        khdz[k] = rhoz[k] * dfh[k]
        kmdz[k] = rhoz[k] * dfm[k]
    for k in range(2, nk):
        delp[k] = (p[k] * dz[k - 1] + p[k - 1] * dz[k]) / (dz[k] + dz[k - 1]) - (
            p[k + 1] * dz[k] + p[k] * dz[k + 1]
        ) / (dz[k] + dz[k + 1])
    delp[nk] = delp[nk - 1]
    rhoz[nk + 1] = rhoz[nk]
    khdz[nk + 1] = rhoz[nk + 1] * dfh[nk]
    kmdz[nk + 1] = rhoz[nk + 1] * dfm[nk]
    for k in range(2, nk):
        khdz[k] = _mx(xp, khdz[k], 0.5 * s_aw[k])
        khdz[k] = _mx(xp, khdz[k], -0.5 * (s_aw[k] - s_aw[k + 1]))
        kmdz[k] = _mx(xp, kmdz[k], 0.5 * s_aw[k])
        kmdz[k] = _mx(xp, kmdz[k], -0.5 * (s_aw[k] - s_aw[k + 1]))
    ust2 = ust * ust
    z = 0.0 * ust
    a, b, c, d = ([None] * L for _ in range(4))

    def mom(f, s_f):
        k = 1
        a[k] = -dtz[k] * kmdz[k] * rhoinv[k]
        b[k] = (
            1.0
            + dtz[k] * (kmdz[k + 1] + rhosfc * ust2 / wspd) * rhoinv[k]
            - 0.5 * dtz[k] * s_aw[k + 1] * onoff
        )
        c[k] = -dtz[k] * kmdz[k + 1] * rhoinv[k] - 0.5 * dtz[k] * s_aw[k + 1] * onoff
        d[k] = f[k] + dtz[k] * 0.0 * ust2 / wspd - dtz[k] * s_f[k + 1] * onoff
        for k in range(2, nk):
            a[k] = (
                -dtz[k] * kmdz[k] * rhoinv[k]
                + 0.5 * dtz[k] * rhoinv[k] * s_aw[k] * onoff
            )
            b[k] = (
                1.0
                + dtz[k] * (kmdz[k] + kmdz[k + 1]) * rhoinv[k]
                + 0.5 * dtz[k] * rhoinv[k] * (s_aw[k] - s_aw[k + 1]) * onoff
            )
            c[k] = (
                -dtz[k] * kmdz[k + 1] * rhoinv[k]
                - 0.5 * dtz[k] * rhoinv[k] * s_aw[k + 1] * onoff
            )
            d[k] = f[k] + dtz[k] * rhoinv[k] * (s_f[k] - s_f[k + 1]) * onoff
        a[nk], b[nk], c[nk], d[nk] = z, z + 1.0, z, f[nk]
        x = tridiag2(nk, a, b, c, d)
        return [None] + [(x[k] - f[k]) / delt for k in range(1, nk + 1)]

    du = mom(t["u"], t["s_awu"])
    dv = mom(t["v"], t["s_awv"])

    def sca(f, s_f, sfc_flux, src, mf=True):
        k = 1
        if mf:
            a[k] = -dtz[k] * khdz[k] * rhoinv[k]
            b[k] = (
                1.0
                + dtz[k] * (khdz[k + 1] + khdz[k]) * rhoinv[k]
                - 0.5 * dtz[k] * rhoinv[k] * s_aw[k + 1]
            )
            c[k] = (
                -dtz[k] * khdz[k + 1] * rhoinv[k]
                - 0.5 * dtz[k] * rhoinv[k] * s_aw[k + 1]
            )
            d[k] = (
                f[k]
                + dtz[k] * rhosfc * sfc_flux * rhoinv[k]
                + src[0][k]
                - dtz[k] * rhoinv[k] * s_f[k + 1]
            )
            if src[1] is not None:
                d[k] = d[k] + src[1][k]
            for k in range(2, nk):
                a[k] = (
                    -dtz[k] * khdz[k] * rhoinv[k] + 0.5 * dtz[k] * rhoinv[k] * s_aw[k]
                )
                b[k] = (
                    1.0
                    + dtz[k] * (khdz[k] + khdz[k + 1]) * rhoinv[k]
                    + 0.5 * dtz[k] * rhoinv[k] * (s_aw[k] - s_aw[k + 1])
                )
                c[k] = (
                    -dtz[k] * khdz[k + 1] * rhoinv[k]
                    - 0.5 * dtz[k] * rhoinv[k] * s_aw[k + 1]
                )
                d[k] = f[k] + src[0][k] + dtz[k] * rhoinv[k] * (s_f[k] - s_f[k + 1])
                if src[1] is not None:
                    d[k] = d[k] + src[1][k]
        else:
            a[k] = -dtz[k] * khdz[k] * rhoinv[k]
            b[k] = 1.0 + dtz[k] * (khdz[k + 1] + khdz[k]) * rhoinv[k]
            c[k] = -dtz[k] * khdz[k + 1] * rhoinv[k]
            d[k] = f[k]
            for k in range(2, nk):
                a[k] = -dtz[k] * khdz[k] * rhoinv[k]
                b[k] = 1.0 + dtz[k] * (khdz[k] + khdz[k + 1]) * rhoinv[k]
                c[k] = -dtz[k] * khdz[k + 1] * rhoinv[k]
                d[k] = f[k]
        a[nk], b[nk], c[nk], d[nk] = z, z + 1.0, z, f[nk]
        return tridiag2(nk, a, b, c, d)

    tcdd = [None] + [t["tcd"][k] * delt for k in range(1, nk + 1)]
    qcdd = [None] + [t["qcd"][k] * delt for k in range(1, nk + 1)]
    dhd = [None] + [t["diss_heat"][k] * delt for k in range(1, nk + 1)]
    thl = sca(t["thl"], t["s_awthl"], flt, (tcdd, dhd))
    sqc2 = sca(t["sqc"], t["s_awqc"], flqc, (qcdd, None))
    qvflux = xp.where(
        flqv < 0.0,
        _mx(xp, flqv, (_mn(xp, 0.9 * t["sqv"][1] - 1e-8, 0.0) / dtz[1])),
        flqv,
    )
    sqv2 = sca(t["sqv"], t["s_awqv"], qvflux, (qcdd, None))
    sqi2 = sca(t["sqi"], None, None, None, mf=False)
    dqv = [None] + [
        (sqv2[k] / (1.0 - sqv2[k]) - t["qv"][k]) / delt for k in range(1, nk + 1)
    ]
    dqc = [None] + [
        (sqc2[k] / (1.0 - sqv2[k]) - t["qc"][k]) / delt for k in range(1, nk + 1)
    ]
    dqi = [None] + [
        (sqi2[k] / (1.0 - sqv2[k]) - t["qi"][k]) / delt for k in range(1, nk + 1)
    ]
    sqv2, sqc2, sqi2, thl, dqv, dqc, dqi = moisture_check(
        xp, nk, delt, delp, t["exner"], sqv2, sqc2, sqi2, thl, dqv, dqc, dqi
    )
    ex = t["exner"]
    dth = [None] + [
        (thl[k] + xlvcp / ex[k] * sqc2[k] + xlscp / ex[k] * sqi2[k] - t["th"][k]) / delt
        for k in range(1, nk + 1)
    ]
    return du, dv, dth, dqv, dqc, dqi


def mym_initialize(
    xp,
    nk,
    dz,
    zw,
    u,
    v,
    thl,
    qw,
    zi,
    theta,
    sh,
    sm,
    ust,
    rmo,
    Psig_bl,
    edmf_w1,
    edmf_a1,
):
    """bl_mynn_mixlength = 1, INITIALIZE_QKE.  Returns el, qke, tsq, qsq,
    cov, sh, sm."""
    L = nk + 1
    pmz = phh = 1.0
    flt = flq = 0.0
    z = 0.0 * ust
    vt = [None] + [z] * nk
    vq = [None] + [z] * nk
    dtl, dqw, dtv, gm, gh, sm, sh = mym_level2(
        xp, nk, dz, u, v, thl, qw, vt, vq, sm, sh
    )
    el = [None] * L
    qke = [None] * L
    tsq, qsq, cov = ([None] * L for _ in range(3))
    el[1] = z
    qke[1] = 1.5 * (ust * ust) * (b1 * pmz) ** (2.0 / 3.0)
    for k in range(2, nk + 1):
        qke[k] = qke[1] * _mx(
            xp, (ust * 700.0 - zw[k]) / (_mx(xp, ust, 0.01) * 700.0), 0.01
        )
    phm = phh * b2 / (b1 * pmz) ** (1.0 / 3.0)
    tsq[1] = phm * ((flt / ust) * (flt / ust))
    qsq[1] = phm * ((flq / ust) * (flq / ust))
    cov[1] = phm * (flt / ust) * (flq / ust)
    for k in range(2, nk + 1):
        vkz = karman * zw[k]
        el[k] = vkz / (1.0 + vkz / 100.0)
        tsq[k] = qsq[k] = cov[k] = z
    pdk, pdt, pdq, pdc = ([None] * (nk + 2) for _ in range(4))
    for _ in range(5):
        el, qkw = mym_length(
            xp,
            nk,
            dz,
            zw,
            rmo,
            flt + z,
            u,
            v,
            qke,
            dtv,
            zi,
            theta,
            Psig_bl,
            edmf_w1,
            edmf_a1,
        )
        for k in range(2, nk + 1):
            elq = el[k] * qkw[k]
            pdk[k] = elq * (sm[k] * gm[k] + sh[k] * gh[k])
            pdt[k] = elq * sh[k] * (dtl[k] * dtl[k])
            pdq[k] = elq * sh[k] * (dqw[k] * dqw[k])
            pdc[k] = elq * sh[k] * dtl[k] * dqw[k]
        vkz = karman * 0.5 * dz[1]
        elv = 0.5 * (el[2] + el[1]) / vkz
        ustm = _mx(xp, ust, 0.02)
        qke[1] = 1.0 * (ustm * ustm) * (b1 * pmz * elv) ** (2.0 / 3.0)
        phm = phh * b2 / (b1 * pmz / (elv * elv)) ** (1.0 / 3.0)
        tsq[1] = phm * ((flt / ust) * (flt / ust))
        qsq[1] = phm * ((flq / ust) * (flq / ust))
        cov[1] = phm * (flt / ust) * (flq / ust)
        for k in range(2, nk):
            b1l = b1 * 0.25 * (el[k + 1] + el[k])
            tmpq = _mn(xp, _mx(xp, b1l * (pdk[k + 1] + pdk[k]), qkemin), 125.0)
            qke[k] = tmpq**twothirds
            pos = qke[k] > 0.0
            b2l = xp.where(
                pos, b2 * (b1l / b1) / xp.sqrt(xp.where(pos, qke[k], 1.0)), 0.0
            )
            tsq[k] = b2l * (pdt[k + 1] + pdt[k])
            qsq[k] = b2l * (pdq[k + 1] + pdq[k])
            cov[k] = b2l * (pdc[k + 1] + pdc[k])
    qke[1] = 0.5 * (qke[1] + qke[2])
    qke[nk] = qke[nk - 1]
    tsq[nk] = tsq[nk - 1]
    qsq[nk] = qsq[nk - 1]
    cov[nk] = cov[nk - 1]
    return el, qke, tsq, qsq, cov, sh, sm


# ---------------------------------------------------------------------------
# mynnedmf_wrapper_run + mynn_bl_driver on every column

STATE_NAMES = [
    "qke",
    "qke_adv",
    "sh3d",
    "sm3d",
    "tsq",
    "qsq",
    "cov",
    "el_pbl",
    "qc_bl",
    "qi_bl",
    "cldfra_bl",
    "edmf_a",
    "edmf_w",
    "edmf_qt",
    "edmf_thl",
    "edmf_ent",
    "edmf_qc",
    "sub_thl3d",
    "sub_sqv3d",
    "det_thl3d",
    "det_sqv3d",
    "dqke",
    "qwt",
    "qshear",
    "qbuoy",
    "qdiss",
]
TEND_NAMES = (
    "upten",
    "vpten",
    "thpten",
    "qvpten",
    "qcpten",
    "qipten",
    "xkzh",
    "xkzm",
    "dheat",
)
INPUT_NAMES = ("dz", "u", "v", "w", "th", "qv", "qc", "qi", "p", "exner", "rho", "t3d")


def mynnedmf_wrapper_run(bk, cfg, *, initflag, delt, dxc, inp, sfc, st):
    """mynnedmf_wrapper_run + mynn_bl_driver as turb.F calls it, every
    column at once.

    inp: dict INPUT_NAMES -> [nk, ...] arrays (turb.F's packing: dz =
    dz*rmh, u/v at scalar points, w = 0.5*(wa(k)+wa(k+1)), th, qv/qc/qi
    mixing ratios, p, exner, rho, t3d = th*exner).  sfc: dict of [...]
    arrays znt, xland, tsk, qsfc, psfc, ust, ch, hfx, qfx, wspd, rmol, hpbl
    (inout rmol, hpbl).  st: dict STATE_NAMES -> [nk, ...] (qke_adv is the
    advected qke turb.F passes; the rest the carried MYNN arrays).
    ``initflag``: Python bool (nstep == 0).  Returns (sfc_out, st_out,
    tend): sfc_out adds kpbl, nupdraft, maxmf, ktop_plume (float arrays);
    tend: TEND_NAMES -> [nk, ...]."""
    check(cfg)
    xp = bk.xp
    closure = 3.0 if int(_opt(cfg, "ipbl", 4)) == 5 else 2.5
    edmf = int(_opt(cfg, "bl_mynn_edmf", 1))
    edmf_mom = int(_opt(cfg, "bl_mynn_edmf_mom", 1))
    nk = inp["th"].shape[0]
    if nk < 4:
        raise ValueError("pbl_mynn: needs nk >= 4")
    cshape = inp["th"].shape[1:]

    def cols(a):
        return [None] + [xp.reshape(a[k - 1, ...], (-1,)) for k in range(1, nk + 1)]

    def flat(a):
        return xp.reshape(a, (-1,))

    I = {k: cols(inp[k]) for k in INPUT_NAMES}
    S = {k: cols(st[k]) for k in STATE_NAMES}
    F = {k: flat(v) for k, v in sfc.items()}
    ust, xland = F["ust"], F["xland"]
    z = 0.0 * ust
    L = nk + 1
    rng = range(1, nk + 1)
    # wrapper: specific humidities
    sqv = [None] + [I["qv"][k] / (1.0 + I["qv"][k]) for k in rng]
    sqc = [None] + [I["qc"][k] / (1.0 + I["qv"][k]) for k in rng]
    sqi = [None] + [I["qi"][k] / (1.0 + I["qv"][k]) for k in rng]
    zw = [None] * (nk + 2)
    zw[1] = z
    for k in range(2, nk + 2):
        zw[k] = zw[k - 1] + I["dz"][k - 1]
    dz1, u1, v1, w1, th1 = I["dz"], I["u"], I["v"], I["w"], I["th"]
    tk1, p1, ex1, rho1 = I["t3d"], I["p"], I["exner"], I["rho"]
    zcol = [None] + [z] * nk
    if initflag:
        thetav = [None] + [th1[k] * (1.0 + 0.608 * sqv[k]) for k in rng]
        sqw = [None] + [sqv[k] + sqc[k] + sqi[k] for k in rng]
        thl = [None] + [
            th1[k] - xlvcp / ex1[k] * sqc[k] - xlscp / ex1[k] * sqi[k] for k in rng
        ]
        qke1 = [None] + [
            5.0
            * ust
            * _mx(xp, (ust * 700.0 - zw[k]) / (_mx(xp, ust, 0.01) * 700.0), 0.01)
            for k in rng
        ]
        pblh, _ = get_pblh(xp, nk, thetav, qke1, zw, dz1, xland)
        Psig_bl, _ = scale_aware(xp, dxc, pblh)
        el, qke1, tsq1, qsq1, cov1, sh, sm = mym_initialize(
            xp,
            nk,
            dz1,
            zw,
            u1,
            v1,
            thl,
            sqv,
            pblh,
            th1,
            zcol,
            zcol,
            ust,
            F["rmol"],
            Psig_bl,
            zcol,
            zcol,
        )
        S["el_pbl"], S["sh3d"], S["sm3d"] = el, sh, sm
        S["qke"], S["tsq"], S["qsq"], S["cov"] = qke1, tsq1, qsq1, cov1
        S["qke_adv"] = qke1
        S["cldfra_bl"] = S["qc_bl"] = zcol
        el[1] = el[1] + z
    # bl_mynn_tkeadvect: qke = qke_adv
    S["qke"] = S["qke_adv"]
    dqke0 = S["qke"]
    cldfra, qcbl, qibl = S["cldfra_bl"], S["qc_bl"], S["qi_bl"]
    qv1 = [None] + [sqv[k] / (1.0 - sqv[k]) for k in rng]
    qc1 = [None] + [sqc[k] / (1.0 - sqv[k]) for k in rng]
    qi1 = [None] + [sqi[k] / (1.0 - sqv[k]) for k in rng]
    sqw = [None] + [sqv[k] + sqc[k] + sqi[k] for k in rng]
    thl = [None] + [
        th1[k] - xlvcp / ex1[k] * sqc[k] - xlscp / ex1[k] * sqi[k] for k in rng
    ]
    thetav = [None] + [th1[k] * (1.0 + 0.608 * sqv[k]) for k in rng]
    qke1, sh, sm = S["qke"], S["sh3d"], S["sm3d"]
    tsq1, qsq1, cov1 = S["tsq"], S["qsq"], S["cov"]
    pblh, kpbl = get_pblh(xp, nk, thetav, qke1, zw, dz1, xland)
    Psig_bl, Psig_shcu = scale_aware(xp, dxc, pblh)
    cpm = cp * (1.0 + 0.84 * qv1[1])
    flqv = F["qfx"] / rho1[1]
    flqc = 0.0
    th_sfc = F["tsk"] / ex1[1]
    flq = flqv + flqc
    flt = F["hfx"] / (rho1[1] * cpm) - xlvcp * flqc / ex1[1]
    fltv = flt + flqv * p608 * th_sfc
    rmol = -karman * gtr * fltv / _mx(xp, (ust * ust) * ust, 1.0e-6)
    zet = 0.5 * dz1[1] * rmol
    zet = _mx(xp, zet, -20.0)
    zet = _mn(xp, zet, 20.0)
    phi_m = phim(xp, zet)
    pmz = phi_m - zet
    qcbl, qibl, cldfra, vt, vq = mym_condensation(
        xp, nk, dz1, sqw, sqc, p1, ex1, qsq1, th1
    )
    zc2 = [None] + [z] * (nk + 1)
    mf = {
        k: zc2
        for k in ("edmf_a", "edmf_w", "edmf_qt", "edmf_thl", "edmf_ent", "edmf_qc")
    }
    mf.update(
        {
            k: zc2
            for k in ("s_aw", "s_awthl", "s_awqt", "s_awqv", "s_awqc", "s_awu", "s_awv")
        }
    )
    nupdraft = ktop_plume = maxmf = z
    if edmf > 0:
        r = dmp_mf(
            bk, nk, zw, dz1, p1, rho1, edmf_mom, u1, v1, w1, th1, thl, thetav, tk1, sqw, sqc, ex1, vt, vq,
            flt, fltv, flq, pblh, dxc, xland, th_sfc, qcbl, cldfra, Psig_shcu,
        )  # fmt: skip
        mf.update({k: r[k] for k in mf})
        qcbl, cldfra, vt, vq = r["qc_bl1d"], r["cldfra_bl1d"], r["vt"], r["vq"]
        nupdraft, ktop_plume, maxmf = r["nup2"], r["ktop"], r["maxmf"]
    tr = mym_turbulence(
        xp, nk, closure, dz1, zw, u1, v1, thl, sqw, qke1, tsq1, qsq1, cov1, vt, vq, rmol, flt, pblh, th1,
        sh, sm, Psig_bl, cldfra, mf["edmf_w"], mf["edmf_a"],
    )  # fmt: skip
    el = tr["el"]
    qke1, tsq1, qsq1, cov1, qWT1, qDISS1 = mym_predict(
        xp, nk, closure, delt, dz1, ust, pmz, el, tr["dfq"], rho1, tr["pdk"], tr["pdt"], tr["pdq"], tr["pdc"],
        qke1, tsq1, qsq1, cov1, mf["s_aw"],
    )  # fmt: skip
    diss = [None] * (nk + 1)
    for k in range(1, nk):
        dh = _mn(
            xp,
            _mx(
                xp,
                1.0
                * (qke1[k] ** 1.5)
                / (b1 * _mx(xp, 0.5 * (el[k] + el[k + 1]), 1.0))
                / cp,
                0.0,
            ),
            0.002,
        )
        diss[k] = dh * xp.exp(-10000.0 / _mx(xp, p1[k], 1.0))
    diss[nk] = z
    tin = {
        "dz": dz1, "rho": rho1, "u": u1, "v": v1, "th": th1, "tk": tk1, "qv": qv1, "qc": qc1, "qi": qi1,
        "p": p1, "exner": ex1, "thl": thl, "sqv": sqv, "sqc": sqc, "sqi": sqi, "tcd": tr["tcd"],
        "qcd": tr["qcd"], "dfm": tr["dfm"], "dfh": tr["dfh"], "diss_heat": diss, "s_aw": mf["s_aw"],
        "s_awthl": mf["s_awthl"], "s_awqv": mf["s_awqv"], "s_awqc": mf["s_awqc"], "s_awu": mf["s_awu"],
        "s_awv": mf["s_awv"],
    }  # fmt: skip
    du, dv, dth, dqv, dqc, dqi = mynn_tendencies(
        xp, nk, delt, F["psfc"], ust, flt, flqv, flqc, F["wspd"], edmf_mom, tin
    )
    xkzm, xkzh = [None] * L, [None] * L
    xkzm[1] = xkzh[1] = z
    for k in range(2, nk + 1):
        dzk = 0.5 * (dz1[k] + dz1[k - 1])
        xkzm[k] = tr["dfm"][k] * dzk
        xkzh[k] = tr["dfh"][k] * dzk
    u3 = (ust * ust) * ust
    qS, qB = list(tr["qSHEAR"]), list(tr["qBUOY"])
    qS[1] = 4.0 * (u3 * phi_m / (karman * dz1[1])) - qS[2]
    qB[1] = 4.0 * (-u3 * zet / (karman * dz1[1])) - qB[2]
    qshear = [None] + [0.5 * (qS[k] + qS[k + 1]) for k in range(1, nk)] + [z]
    qbuoy = [None] + [0.5 * (qB[k] + qB[k + 1]) for k in range(1, nk)] + [z]
    qwt = [None] + [qWT1[k] for k in range(1, nk)] + [z]
    qdiss = [None] + [qDISS1[k] for k in range(1, nk)] + [z]
    dqke = [None] + [(qke1[k] - dqke0[k]) * 0.5 / delt for k in range(1, nk)] + [z]

    def stack(c):
        return xp.reshape(xp.stack([c[k] + z for k in rng]), (nk,) + cshape)

    def unflat(a):
        return xp.reshape(a + z, cshape)

    sto = dict(st)
    sto.update(
        {
            "qke": stack(qke1), "qke_adv": stack(qke1), "tsq": stack(tsq1), "qsq": stack(qsq1),
            "cov": stack(cov1), "sh3d": stack(tr["sh"]), "sm3d": stack(tr["sm"]), "el_pbl": stack(el),
            "qc_bl": stack(qcbl), "qi_bl": stack(qibl), "cldfra_bl": stack(cldfra), "dqke": stack(dqke),
            "qwt": stack(qwt), "qshear": stack(qshear), "qbuoy": stack(qbuoy), "qdiss": stack(qdiss),
        }
    )  # fmt: skip
    zz = stack(zcol)
    for k in ("edmf_a", "edmf_w", "edmf_qt", "edmf_thl", "edmf_ent", "edmf_qc"):
        sto[k] = stack(mf[k]) if edmf > 0 else zz
    for k in ("sub_thl3d", "sub_sqv3d", "det_thl3d", "det_sqv3d"):
        sto[k] = zz
    w1_ = [None] + [1.0 - sqv[k] for k in rng]
    tend = {
        "upten": stack(du), "vpten": stack(dv), "thpten": stack(dth),
        "qvpten": stack([None] + [dqv[k] / w1_[k] for k in rng]),
        "qcpten": stack([None] + [dqc[k] / w1_[k] for k in rng]),
        "qipten": stack([None] + [dqi[k] / w1_[k] for k in rng]),
        "xkzh": stack(xkzh), "xkzm": stack(xkzm), "dheat": stack(diss),
    }  # fmt: skip
    sfo = dict(sfc)
    sfo.update(
        {
            "rmol": unflat(rmol), "hpbl": unflat(pblh), "kpbl": unflat(kpbl), "nupdraft": unflat(nupdraft),
            "ktop_plume": unflat(ktop_plume), "maxmf": unflat(maxmf),
        }
    )  # fmt: skip
    return sfo, sto, tend


# ---------------------------------------------------------------------------
# the model glue (cm1x.pbl registry: PBL_SCHEMES[4], PBL_SCHEMES[5])

# 3-D arrays the scheme carries (cm1.F allocations, 0) besides STATE_NAMES'
# qke_adv (CM1's work array: the wrapper's in/out, refilled by solve2)
_CARRIED_3D = tuple(k for k in STATE_NAMES if k != "qke_adv") + (
    "qke3d",
    "xkzh",
    "xkzm",
)
_CARRIED_2D = ("kpbl", "nupdraft", "maxmf", "ktop_plume")
_SFC_IN = (
    "znt",
    "xland",
    "tsk",
    "qsfc",
    "psfc",
    "ust",
    "ch",
    "hfx",
    "qfx",
    "wspd",
    "rmol",
    "hpbl",
)
DUMP_KEYS = (
    "hpbl", "xkzh", "xkzm", "qke", "qke3d", "tsq", "qsq", "cov", "sh3d", "sm3d",
    "el_pbl", "qc_bl", "qi_bl", "cldfra_bl", "dqke",
)  # fmt: skip
WRF_INPUTS = True


def options(cfg):
    """param.F for ipbl=4, 5: idiss = 0 (cm1setup /= 4) and the fixed
    bl_mynn_* settings (edmf = edmf_mom = 1; tkeadvect, cloudpdf = 2,
    mixlength = 1, cloudmix = 1, mixqt = 0 are what the port implements)."""
    if int(_opt(cfg, "ipbl", 0)) not in (4, 5):
        return
    if int(_opt(cfg, "idiss", 0)) >= 1 and int(_opt(cfg, "cm1setup", 1)) != 4:
        cfg.idiss = 0
    cfg.bl_mynn_edmf = 1
    cfg.bl_mynn_edmf_mom = 1


def init_state(ctx):
    """The MYNN arrays (cm1.F allocates them as 0): STATE_NAMES (but
    qke_adv), qke3d, xkzh, xkzm on ``g.s_shape``; kpbl, nupdraft, maxmf,
    ktop_plume 2-D."""
    bk, g = ctx.bk, ctx.g
    # distinct arrays (bk.set works in place on numpy/torch)
    out = {k: bk.zeros(g.s_shape) for k in _CARRIED_3D}
    out.update({k: bk.zeros(g.s_shape[1:]) for k in _CARRIED_2D})
    return out


def pbl(ctx):
    """turb.F get_vars + the domynn block: mynnedmf_wrapper_run on every
    column (initflag = nstep == 0, the startup pass), with qke_adv = qke3d
    (solve2's advected qke).  Returns the PBL tendencies (thpten, qvpten,
    qcpten, qipten, upten, vpten: not rho-weighted, as MYNN gives them and
    CM1 stores them), xkzh, xkzm, hpbl, rmol and the carried arrays."""
    from .pbl import interior2, put2

    bk, g, cfg, s = ctx.bk, ctx.g, ctx.cfg, ctx.s
    n, z, nk = g.ngxy, g.ngz, g.nk
    J, I, K = slice(n, n + g.nj), slice(n, n + g.ni), slice(z, z + nk)
    Kp = slice(z + 1, z + nk + 1)
    R = (K, J, I)
    w = ctx.w
    inp = {
        "dz": w["dz8w"][R],
        "u": w["u"][R],
        "v": w["v"][R],
        "w": 0.5 * (ctx.w3d[R] + ctx.w3d[Kp, J, I]),
        "th": w["th"][R],
        "qv": w["qv"][R],
        "qc": w["qc"][R],
        "qi": w["qi"][R],
        "p": ctx.prs[R],
        "exner": w["pi"][R],
        "rho": ctx.rho[R],
        "t3d": w["t"][R],
    }
    st = {k: s["qke3d" if k == "qke_adv" else k][R] for k in STATE_NAMES}
    sfc = {k: interior2(g, s[k]) for k in _SFC_IN}
    r_sfc, r_st, r_te = mynnedmf_wrapper_run(
        bk,
        cfg,
        initflag=ctx.nstep == 0,
        delt=float(ctx.dt),
        dxc=float(cfg.dx),
        inp=inp,
        sfc=sfc,
        st=st,
    )
    out = {}
    for k in STATE_NAMES:
        if k != "qke_adv":
            out[k] = bk.set(bk.copy(s[k]), R, r_st[k])
    for k in ("thpten", "qvpten", "qcpten", "qipten", "upten", "vpten", "xkzh", "xkzm"):
        out[k] = bk.set(bk.copy(s[k]), R, r_te[k])
    for k in ("hpbl", "rmol") + _CARRIED_2D:
        out[k] = put2(bk, g, bk.copy(s[k]), r_sfc[k])
    return out
