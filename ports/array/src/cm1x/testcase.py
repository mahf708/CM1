"""CM1's pre-configured test cases (namelist ``testcase`` = 1..15) and the
large-scale pressure gradients ``lspgrad`` = 2, 3, 4.

Ported from

* src/param.F: ``imoist`` forced by the test case (namelist.py ``_derive``),
  ``dolsw`` (testcase 3..7), ``hurr_angle``;
* src/base.F: the forcing profiles wprof, ufrc, vfrc, thfrc, qvfrc, ug, vg,
  dvdr, ulspg, vlspg (``profiles``), u0 = 18 m/s of testcase 13 (``base_wind``);
* src/solve1.F: lspgrad = 2, 3, 4 (``lspgrad_tend``) and the testcase 10
  theta/qv nudging (``tqnudge``);
* src/testcase_simple_phys.F: the simple radiation of testcase 4, 5
  (``simplerad``), the lspgrad = 3 mesoscale tendencies, the testcase 14 SGP
  forcing and the prescribed tendencies (``simple_phys``), the domain
  averages ``get_avg_uvt(q)``;
* src/adv_routines.F ``wsub`` (dolsw: cm1x.adv);
* src/init3d.F: the testcase random perturbations (``random_perturbations``,
  with gfortran's ``random_number``: ``GfortranRandom``) and initial TKE
  (``initial_tke``);
* src/constants.F: testcase 4, 5 (cm1x.constants.set_constants);
* src/radiation_driver.F: testcase 8 (cm1x.radiation_driver.solar).

Not ported (``check`` raises NotImplementedError): the test cases whose
surface-layer branches cm1x lacks (testcase 1, 2, 7, 9, 11, 13, 14 with
sfcmodel >= 1: ``use_avg_sfc``, sfcflux, the testcase 7 stress, the
testcase 13 roughness), testcase 4, 5 without liquid water or with a PBL
scheme, testcase 3..7 (dolsw) with terrain or axisymm.  Diagnostics
(domaindiag, budgets) are not ported.

Index conventions as everywhere in cm1x: arrays ``[k, j, i]`` with Python
``k`` = Fortran ``k`` (ngz = 1) and Python ``i`` = Fortran ``i + 2``
(ngxy = 3); the 1-D profiles have length nk + 2 (Fortran kb:ke = 0:nk+1).
"""

from __future__ import annotations

import math

import numpy as np

from . import constants as c
from .misc import level_sums

# testcase_simple_phys.F parameters
BIGD = 3.75e-6
QCRIT = 0.008
F0 = 70.0
F1 = 22.0
KAPPA = 85.0
ALPHAZ = 1.0
RHOI = 1.12

PROFILES = (
    "wprof",
    "ufrc",
    "vfrc",
    "thfrc",
    "qvfrc",
    "ug",
    "vg",
    "dvdr",
    "ulspg",
    "vlspg",
)


def _opt(cfg, name, default):
    get = getattr(cfg, "get", None)
    return get(name, default) if get else getattr(cfg, name, default)


def testcase(cfg):
    return int(_opt(cfg, "testcase", 0) or 0)


def active(cfg):
    return testcase(cfg) >= 1 or int(_opt(cfg, "lspgrad", 0) or 0) >= 2


def dolsw(cfg):
    """param.F: large-scale vertical velocity for testcase 3..7."""
    return 3 <= testcase(cfg) <= 7


def simple_rad(cfg):
    return testcase(cfg) in (4, 5)


def hurr_angle(cfg):
    """param.F: hurr_angle = dble(hurr_rotate)*(pi_dp/180.0d0).  With
    -fdefault-real-8 input.F's kind ``dp`` is selected_real_kind(30), i.e.
    binary128: the product is formed in quad precision and rounded to
    double (for hurr_rotate = 30, sin(-hurr_angle) is then exactly -0.5).
    Emulated exactly with fractions (pi to 40 digits, rounded to binary128
    first, as 4*atan(1) in quad precision is)."""
    from fractions import Fraction

    rot = float(_opt(cfg, "hurr_rotate", 0.0) or 0.0)
    pi_qp = _round_binary128(Fraction("3.141592653589793238462643383279502884197"))
    q180 = _round_binary128(pi_qp / 180)
    prod = _round_binary128(Fraction(rot) * q180)
    return float(prod)  # float(Fraction) rounds to nearest double


def _round_binary128(x):
    """Round a positive/negative Fraction to the nearest binary128 value
    (113-bit significand, round half to even)."""
    from fractions import Fraction

    if x == 0:
        return Fraction(0)
    s = -1 if x < 0 else 1
    a = abs(x)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    # a in [2^e, 2^(e+1)): scale so that the significand has 113 bits
    scale = Fraction(2) ** (112 - e)
    m = a * scale
    n = m.numerator // m.denominator
    r = m - n
    if r > Fraction(1, 2) or (r == Fraction(1, 2) and n % 2 == 1):
        n += 1
    return s * Fraction(n) / scale


def check(cfg):
    """Raise NotImplementedError for test-case options cm1x does not port."""
    tc = testcase(cfg)
    lsp = int(_opt(cfg, "lspgrad", 0) or 0)
    bad = []
    if tc < 0 or tc > 15:
        bad.append(f"testcase={tc} (no such test case)")
    if lsp < 0 or lsp > 4:
        bad.append(f"lspgrad={lsp}")
    if tc == 0 and lsp <= 1:
        if bad:
            raise NotImplementedError("testcase: " + ", ".join(bad))
        return
    sfcmodel = int(_opt(cfg, "sfcmodel", 0) or 0)
    if simple_rad(cfg) and int(_opt(cfg, "radopt", 0) or 0) >= 1:
        bad.append("testcase=4,5 with radopt>=1 (param.F stops)")
    if dolsw(cfg) and (
        bool(_opt(cfg, "terrain_flag", False)) or int(_opt(cfg, "axisymm", 0) or 0)
    ):
        bad.append("dolsw (testcase 3-7) with terrain or axisymm (param.F stops)")
    if sfcmodel >= 1 and tc in (1, 2, 7, 9, 11, 13, 14):
        bad.append(
            f"testcase={tc} with sfcmodel={sfcmodel} (its surface-layer branches"
            " are not ported)"
        )
    if simple_rad(cfg) and int(_opt(cfg, "cm1setup", 1) or 0) in (2, 4):
        bad.append("testcase=4,5 with a PBL scheme")
    if simple_rad(cfg) and (
        int(_opt(cfg, "imoist", 0) or 0) != 1 or int(_opt(cfg, "ptype", 0) or 0) < 1
    ):
        bad.append("testcase=4,5 without liquid-water microphysics")
    if (lsp == 3 or tc in (6, 10, 15)) and not float(_opt(cfg, "hurr_rad", 0.0) or 0.0):
        bad.append("lspgrad=3 / testcase=6,10,15 without hurr_rad")
    if int(_opt(cfg, "imove", 0) or 0) == 1 and active(cfg):
        bad.append("imove=1 with a test case or lspgrad>=2")
    if bad:
        raise NotImplementedError("testcase: options not ported: " + ", ".join(bad))


# ---------------------------------------------------------------------------
# base.F
def base_wind(cfg, u0, v0, bk):
    """base.F testcase 13 (isnd /= 7): u0 = 18, v0 = 0 on the whole arrays."""
    if testcase(cfg) == 13 and int(_opt(cfg, "isnd", 0) or 0) != 7:
        u0 = bk.full(u0.shape, 18.0)
        v0 = bk.zeros(v0.shape)
    return u0, v0


def profiles(cfg, bk, *, zh, zf, u0, v0):
    """base.F: the 1-D forcing profiles (numpy float64 on the host, then
    ``bk`` arrays), Fortran kb:ke as Python 0:nk+1."""
    check(cfg)
    nk = int(cfg.nk)
    tc = testcase(cfg)
    z_h = bk.to_numpy(zh)[:, 3, 3]  # zh(1,1,k)
    z_f = bk.to_numpy(zf)[:, 3, 3]
    u_0 = bk.to_numpy(u0)[:, 3, 3]
    v_0 = bk.to_numpy(v0)[:, 3, 3]
    p = {n: np.zeros(nk + 2) for n in PROFILES}
    wprof, thfrc, qvfrc, ug, vg, dvdr = (
        p["wprof"],
        p["thfrc"],
        p["qvfrc"],
        p["ug"],
        p["vg"],
        p["dvdr"],
    )
    umove = float(_opt(cfg, "umove", 0.0) or 0.0)
    vmove = float(_opt(cfg, "vmove", 0.0) or 0.0)
    rng = range(1, nk + 1)

    if int(_opt(cfg, "iwnd", 0) or 0) == 8:
        for k in rng:
            ug[k] = 0.0
            vg[k] = float(cfg.hurr_vg)

    if tc == 3:
        wmin, z1, z2 = -0.0065, 1500.0, 2100.0
        for k in rng:
            if z_f[k] <= z1:
                wprof[k] = 0.0 + (wmin - 0.0) * (z_f[k] - 0.0) / (z1 - 0.0)
            elif z_f[k] <= z2:
                wprof[k] = wmin + (0.0 - wmin) * (z_f[k] - z1) / (z2 - z1)
            else:
                wprof[k] = 0.0
        radsfc = -2.0 / (3600.0 * 24.0)
        for k in rng:
            if z_h[k] <= z1:
                thfrc[k] = radsfc
            elif z_h[k] <= z2:
                thfrc[k] = radsfc + (0.0 - radsfc) * (z_h[k] - z1) / (z2 - z1)
            else:
                thfrc[k] = 0.0
        qvsfc, z1, z2 = -1.2e-8, 300.0, 500.0
        for k in rng:
            if z_h[k] <= z1:
                qvfrc[k] = -1.2e-8
            elif z_h[k] <= z2:
                qvfrc[k] = qvsfc + (0.0 - qvsfc) * (z_h[k] - z1) / (z2 - z1)
            else:
                qvfrc[k] = 0.0
        for k in rng:
            ug[k] = -10.0 + 0.0018 * z_h[k]
            vg[k] = 0.0

    if tc in (4, 5):
        for k in range(2, nk + 1):
            wprof[k] = -3.75e-6 * z_f[k]

    if tc in (6, 10, 15):
        wmin, z1, z2 = -0.01, 1000.0, 4000.0
        vgpl, rad = float(cfg.hurr_vgpl), float(cfg.hurr_rad)
        for k in rng:
            dvdr[k] = vgpl * (vg[k] / rad)
            if z_f[k] <= z1:
                wprof[k] = 0.0 + (wmin - 0.0) * (z_f[k] - 0.0) / (z1 - 0.0)
            elif z_f[k] <= z2:
                wprof[k] = wmin + (0.0 - wmin) * (z_f[k] - z1) / (z2 - z1)
            else:
                wprof[k] = 0.0

    if tc == 7:
        wmin, z1 = -0.005, 2260.0
        for k in rng:
            if z_f[k] <= z1:
                wprof[k] = 0.0 + (wmin - 0.0) * (z_f[k] - 0.0) / (z1 - 0.0)
            else:
                wprof[k] = wmin
        thsfc = -2.5 / (3600.0 * 24.0)
        for k in rng:
            thfrc[k] = thsfc
        qv1 = -1.0 / 1000.0 / (3600.0 * 24.0)
        qv2 = 0.3456 / 1000.0 / (3600.0 * 24.0)
        z1 = 2980.0
        for k in rng:
            if z_h[k] <= z1:
                qvfrc[k] = qv1 + (qv2 - qv1) * (z_h[k] - 0.0) / (z1 - 0.0)
            else:
                qvfrc[k] = qv2
        for k in rng:
            ug[k] = u_0[k]
            vg[k] = v_0[k]

    if tc == 11:
        ug[:] = u_0[1] + umove
        vg[:] = v_0[1] + vmove
        for name in ("ufrc", "vfrc", "thfrc", "qvfrc", "wprof"):
            p[name][:] = 0.0

    if tc == 12:
        p["ulspg"][:] = 36.0
        p["vlspg"][:] = 0.0

    if tc == 13:
        if int(_opt(cfg, "isnd", 0) or 0) != 7:
            ug[:] = 18.0
            vg[:] = 0.0
        p["ulspg"][:] = 0.000545
        p["vlspg"][:] = 0.0

    ug -= umove
    vg -= vmove
    wprof[1] = 0.0
    wprof[nk + 1] = 0.0
    return {n: bk.asarray(a) for n, a in p.items()}


# ---------------------------------------------------------------------------
# solve1.F
def _rows(bk, g):
    n, nz = g.ngxy, g.ngz
    return slice(nz, nz + g.nk), slice(n, n + g.nj + 1), slice(n, n + g.ni + 1)


def lspgrad_tend(bk, g, cfg, prof, uten1, vten1):
    """solve1.F lspgrad = 2, 3, 4: ulspg, vlspg (2, 3 from ug, vg) added to
    uten1, vten1 on i = 1..ni+1, j = 1..nj+1.  Returns (uten1, vten1, ulspg,
    vlspg)."""
    lsp = int(_opt(cfg, "lspgrad", 0) or 0)
    K, Jp, Ip = _rows(bk, g)
    ulspg, vlspg = prof["ulspg"], prof["vlspg"]
    ug, vg = prof["ug"][K], prof["vg"][K]
    fcor = float(_opt(cfg, "fcor", 0.0)) if int(_opt(cfg, "icor", 0) or 0) == 1 else 0.0
    if lsp == 2:
        umove = float(_opt(cfg, "umove", 0.0) or 0.0)
        vmove = float(_opt(cfg, "vmove", 0.0) or 0.0)
        ulspg = bk.set(ulspg, K, -fcor * (vg + vmove))
        vlspg = bk.set(vlspg, K, fcor * (ug + umove))
    elif lsp == 3:
        ang = hurr_angle(cfg)
        fr = -fcor * vg - vg * vg / float(cfg.hurr_rad)
        ulspg = bk.set(ulspg, K, fr * math.cos(-ang))
        vlspg = bk.set(vlspg, K, fr * math.sin(-ang))
    R = (K, Jp, Ip)
    uten1 = bk.set(uten1, R, uten1[R] + ulspg[K][:, None, None])
    vten1 = bk.set(vten1, R, vten1[R] + vlspg[K][:, None, None])
    return uten1, vten1, ulspg, vlspg


def get_avg(bk, g, cfg, *, th0, ua, va, tha, ruh, ruf, rvh, rvf, xf, yf, qa=None):
    """testcase_simple_phys.F get_avg_uvt / get_avg_uvtq: area-weighted
    domain averages on k = 1..nk, summed in the Fortran loop order (i
    fastest).  Returns arrays of length nk (Fortran k = 1..nk)."""
    n, nz = g.ngxy, g.ngz
    ni, nj, nk = g.ni, g.nj, g.nk
    K, J, I = slice(nz, nz + nk), slice(n, n + nj), slice(n, n + ni)
    tem = float(cfg.dx) * float(cfg.dy)
    rufi, ruhi = ruf[I][None, None, :], ruh[I][None, None, :]
    rvhj, rvfj = rvh[J][None, :, None], rvf[J][None, :, None]
    xfn, yfn = bk.to_numpy(xf), bk.to_numpy(yf)
    dxx = float(xfn[n + ni] - xfn[n])  # xf(nx+1) - xf(1) (serial)
    dyy = float(yfn[n + nj] - yfn[n])
    temd = 1.0 / (dxx * dyy)
    out = {
        "uavg": level_sums(bk, ua[K, J, I] * tem * rufi * rvhj) * temd,
        "vavg": level_sums(bk, va[K, J, I] * tem * ruhi * rvfj) * temd,
        "thavg": level_sums(bk, (th0[K, J, I] + tha[K, J, I]) * tem * ruhi * rvhj)
        * temd,
    }
    if qa is not None:
        out["qavg"] = [
            level_sums(bk, q[K, J, I] * tem * ruhi * rvhj) * temd for q in qa
        ]
    return out


def tqnudge(bk, g, cfg, *, mtime, avg, th0, qv0, thten1, qten):
    """solve1.F (testcase 10): nudge the domain-mean theta and qv towards the
    base state, ramped up over 900 s, tau = 300 s; on i = 1..ni+1,
    j = 1..nj+1.  ``mtime``: the model time at the start of the step."""
    xp = bk.xp
    K, Jp, Ip = _rows(bk, g)
    n = g.ngxy
    nudgefac = (mtime - 0.0) / (900.0 - 0.0)
    nudgefac = xp.minimum(xp.maximum(nudgefac * 1.0, 0.0), 1.0)
    on = xp.abs(nudgefac) > 0.001
    tem = nudgefac / 300.0
    tn = -(avg["thavg"] - th0[K, n, n]) * tem
    tn = xp.where(on, tn, xp.zeros_like(tn))
    R = (K, Jp, Ip)
    thten1 = bk.set(thten1, R, thten1[R] + tn[:, None, None])
    if int(_opt(cfg, "imoist", 0) or 0) == 1:
        nqv = int(cfg.nqv) - 1
        qn = -(avg["qavg"][nqv] - qv0[K, n, n]) * tem
        qn = xp.where(on, qn, xp.zeros_like(qn))
        qten = list(qten)
        qten[nqv] = bk.set(qten[nqv], R, qten[nqv][R] + qn[:, None, None])
    return thten1, qten


# ---------------------------------------------------------------------------
# testcase_simple_phys.F
def _cumsum0(bk, x, reverse=False):
    """Sequential running sum along axis 0 (the Fortran k loop order)."""
    xp = bk.xp
    if reverse:
        x = xp.flip(x, axis=0)
    if bk.name == "jax":
        import jax

        def f(a, v):
            a = a + v
            return a, a

        _, ys = jax.lax.scan(f, xp.zeros_like(x[0]), x)
        out = ys
    else:
        out = xp.cumsum(x, axis=0)
    if reverse:
        out = xp.flip(out, axis=0)
    return out


def simplerad(bk, g, cfg, *, rho0, zh, zf, qa):
    """``simplerad`` (Stevens et al 2005) on i = 1..ni, j = 1..nj.  Returns
    frad (Fortran k = 1..nk+1, as [nk+1, nj, ni]) and zir ([nj, ni]).  The
    Fortran's scratch ql, qt at k = nk+1 (never set) are taken as zero; the
    cloud top (last level with qt > qcrit) must lie in 1..nk-1, else CM1
    reads unset values: checked when the values are concrete."""
    xp = bk.xp
    n, nz = g.ngxy, g.ngz
    ni, nj, nk = g.ni, g.nj, g.nk
    J, I = slice(n, n + nj), slice(n, n + ni)
    K = slice(nz, nz + nk)
    nql1, nql2 = int(cfg.nql1) - 1, int(cfg.nql2) - 1
    nqv = int(cfg.nqv) - 1
    ql = xp.zeros_like(qa[nql1][K, J, I])
    for m in range(nql1, nql2 + 1):
        ql = ql + qa[m][K, J, I]
    qt = qa[nqv][K, J, I] + ql
    r0 = rho0[K, n, n][:, None, None]
    dzf = zf[nz + 1 : nz + nk + 1, J, I] - zf[K, J, I]
    t = KAPPA * r0 * ql * dzf  # k = 1..nk
    zero = xp.zeros_like(t[:1])
    # dum2(1) = 0, dum2(k+1) = dum2(k) + t(k): k = 1..nk+1
    dum2 = xp.concat([zero, _cumsum0(bk, t)], axis=0)
    # dum1(nk+1) = 0 + t(nk+1) (= 0), dum1(k) = dum1(k+1) + t(k)
    dum1 = xp.concat([_cumsum0(bk, xp.concat([t, zero], axis=0), reverse=True)], axis=0)
    # kzi: the last k with qt > qcrit (0 if none)
    kk = xp.arange(1, nk + 1)[:, None, None]
    above = qt > QCRIT
    kzi = xp.max(xp.where(above, kk, xp.zeros_like(kk)), axis=0)
    try:
        bad = bool(xp.any((kzi < 1) | (kzi > nk - 1)))
    except Exception:  # traced (jax.jit): no concrete values
        bad = False
    if bad:
        raise NotImplementedError(
            "simplerad: the qt = 0.008 crossing (cloud top) must lie between"
            " k = 1 and nk (CM1 reads unset values)"
        )
    kc = xp.clip(kzi, 1, nk - 1)

    def at(a, d):  # a(i,j,kzi+d), a on k = 1..nk
        sel = kk == (kc + d)[None]
        return xp.sum(xp.where(sel, a, xp.zeros_like(a)), axis=0)

    zhk = zh[K, J, I]
    zir = at(zhk, 0) + (at(zhk, 1) - at(zhk, 0)) * (QCRIT - at(qt, 0)) / (
        at(qt, 1) - at(qt, 0)
    )
    zfk = zf[nz : nz + nk + 1, J, I]  # k = 1..nk+1
    fr1 = F0 * xp.exp(-dum1)
    fr2 = F1 * xp.exp(-dum2)
    dz = xp.maximum(zfk - zir[None], xp.zeros_like(zfk))
    fr3 = (
        RHOI
        * c.cp
        * BIGD
        * ALPHAZ
        * (0.25 * (dz**1.333333) + zir[None] * (dz**0.333333))
    )
    fr3 = xp.where(zfk < zir[None], xp.zeros_like(fr3), fr3)
    frad = fr1 + fr2 + fr3
    return frad, zir


def _sgp(mtime, xp):
    """testcase 14: (tt1, tt2, at1, rt1, qt1, at2, rt2, qt2) at ``mtime``."""
    h = 3600.0
    table = (
        (3.0 * h, 0.0 * h, 3.0 * h, 0.000, -0.125, 0.080, 0.000, 0.000, 0.020),
        (6.0 * h, 3.0 * h, 6.0 * h, 0.000, 0.000, 0.020, 0.000, 0.000, -0.040),
        (9.0 * h, 6.0 * h, 9.0 * h, 0.000, 0.000, -0.040, -0.080, 0.000, -0.100),
        (12.0 * h, 9.0 * h, 12.0 * h, -0.080, 0.000, -0.100, -0.160, 0.000, -0.160),
        (14.5 * h, 12.0 * h, 14.5 * h, -0.160, 0.000, -0.160, -0.160, -0.100, -0.300),
        (99.0 * h, 14.5 * h, 99.0 * h, -0.160, -0.100, -0.300, -0.160, -0.100, -0.300),
    )
    if isinstance(mtime, (int, float)):  # a Python number: no array select
        for row in table:
            if mtime <= row[0]:
                return list(row[1:])
        return list(table[-1][1:])
    # an array (traced) mtime: select with where, on arrays of mtime's dtype
    # (torch.where on two Python floats would give its default float32)
    vals = [xp.asarray(v, dtype=mtime.dtype) for v in table[-1][1:]]
    for row in reversed(table[:-1]):
        cond = mtime <= row[0]
        vals = [
            xp.where(cond, xp.asarray(a, dtype=mtime.dtype), b)
            for a, b in zip(row[1:], vals)
        ]
    return vals


def simple_phys(
    bk,
    g,
    cfg,
    *,
    mtime,
    prof,
    rho0,
    zh,
    zf,
    mh,
    th0,
    ua,
    va,
    tha,
    qa,
    ruh,
    ruf,
    rvh,
    rvf,
    xf,
    yf,
    uten1,
    vten1,
    thten1,
    qten,
):
    """``testcase_simple_phys`` (testcase >= 1).  ``mtime``: the model time
    at the start of the step.  Returns a dict with uten1, vten1, thten1,
    qten and the updated profiles ufrc, vfrc, thfrc, qvfrc (plus frad,
    thraten, zir for testcase 4, 5)."""
    xp = bk.xp
    n, nz = g.ngxy, g.ngz
    ni, nj, nk = g.ni, g.nj, g.nk
    K, J, I = slice(nz, nz + nk), slice(n, n + nj), slice(n, n + ni)
    tc = testcase(cfg)
    out = {}
    ufrc, vfrc, thfrc, qvfrc = prof["ufrc"], prof["vfrc"], prof["thfrc"], prof["qvfrc"]
    qten = list(qten)

    if simple_rad(cfg):
        frad, zir = simplerad(bk, g, cfg, rho0=rho0, zh=zh, zf=zf, qa=qa)
        thraten = -(frad[1:] - frad[:-1]) * g.rdz * mh[K, J, I] / (c.cp * RHOI)
        thten1 = bk.set(thten1, (K, J, I), thten1[K, J, I] + thraten)
        out.update(frad=frad, zir=zir, thraten=thraten)

    if int(_opt(cfg, "lspgrad", 0) or 0) == 3:
        avg = get_avg(
            bk,
            g,
            cfg,
            th0=th0,
            ua=ua,
            va=va,
            tha=tha,
            ruh=ruh,
            ruf=ruf,
            rvh=rvh,
            rvf=rvf,
            xf=xf,
            yf=yf,
        )
        tem = 1.0 / float(cfg.hurr_rad)
        ang = hurr_angle(cfg)
        s, co = math.sin(-ang), math.cos(-ang)
        uu = avg["vavg"] * s + avg["uavg"] * co
        vv = avg["vavg"] * co - avg["uavg"] * s
        vg, dvdr = prof["vg"][K], prof["dvdr"][K]
        fr = +uu * uu * tem + vv * vg * tem
        ft = -uu * dvdr - uu * vg * tem
        ufrc = bk.set(ufrc, K, fr * co - ft * s)
        vfrc = bk.set(vfrc, K, fr * s + ft * co)

    if tc == 14:
        tt1, tt2, at1, rt1, qt1, at2, rt2, qt2 = _sgp(mtime, xp)
        at = at1 + (at2 - at1) * (mtime - tt1) / (tt2 - tt1)
        rt = rt1 + (rt2 - rt1) * (mtime - tt1) / (tt2 - tt1)
        qt = qt1 + (qt2 - qt1) * (mtime - tt1) / (tt2 - tt1)
        z = zh[K, n, n]
        fac = 1.0 + (0.0 - 1.0) * (z - 1000.0) / (2000.0 - 1000.0)
        fac = xp.minimum(xp.maximum(fac, 0.0 * fac), 0.0 * fac + 1.0)
        thfrc = bk.set(thfrc, K, fac * (at + rt) / 3600.0)
        qvfrc = bk.set(qvfrc, K, fac * qt * 0.001 / 3600.0)

    Ip, Jp = slice(n, n + ni + 1), slice(n, n + nj + 1)
    uten1 = bk.set(uten1, (K, J, Ip), uten1[K, J, Ip] + ufrc[K][:, None, None])
    vten1 = bk.set(vten1, (K, Jp, I), vten1[K, Jp, I] + vfrc[K][:, None, None])
    thten1 = bk.set(thten1, (K, J, I), thten1[K, J, I] + thfrc[K][:, None, None])
    if int(_opt(cfg, "imoist", 0) or 0) == 1:
        nqv = int(cfg.nqv) - 1
        qten[nqv] = bk.set(
            qten[nqv], (K, J, I), qten[nqv][K, J, I] + qvfrc[K][:, None, None]
        )
    out.update(
        uten1=uten1,
        vten1=vten1,
        thten1=thten1,
        qten=qten,
        ufrc=ufrc,
        vfrc=vfrc,
        thfrc=thfrc,
        qvfrc=qvfrc,
    )
    return out


# ---------------------------------------------------------------------------
# init3d.F
class GfortranRandom:
    """gfortran's ``random_number`` (libgfortran xoshiro256**, rnd_8) after
    ``random_seed(put=...)`` with CM1's seed (init3d.F)."""

    _M = (1 << 64) - 1
    _KEYS = (
        0xBD0C5B6E50C2DF49,
        0xD46061CD46E1DF38,
        0xBB4F4D4ED6103544,
        0x114A583D0756AD39,
    )

    def __init__(self, put=None):
        if put is None:
            k = 8
            put = [
                int(_round_half_away(2.0e9 * (2.0 * float(m) / float(k - 1) - 1.0)))
                for m in range(k)
            ]
        rev = [put[7 - i] & 0xFFFFFFFF for i in range(8)]
        seed = [rev[2 * i] | (rev[2 * i + 1] << 32) for i in range(4)]
        self.s = [seed[i] ^ self._KEYS[i] for i in range(4)]

    def next(self):
        M = self._M
        s = self.s
        x = (s[1] * 5) & M
        result = (((x << 7) | (x >> 57)) & M) * 9 & M
        t = (s[1] << 17) & M
        s[2] ^= s[0]
        s[3] ^= s[1]
        s[1] ^= s[2]
        s[0] ^= s[3]
        s[2] ^= t
        s[3] = ((s[3] << 45) | (s[3] >> 19)) & M
        return float(result & (M ^ ((1 << 11) - 1))) * 2.0**-64

    def draws(self, count):
        return np.array([self.next() for _ in range(count)])


def _round_half_away(x):
    return math.floor(x + 0.5) if x >= 0 else -math.floor(-x + 0.5)


def random_perturbations(cfg, *, zh, u0, v0, ua, va, tha, qa):
    """init3d.F ``irandp = 1``: the random perturbations of each test case
    (numpy float64 arrays in cm1x layout; serial: the whole domain, i
    fastest).  Returns (ua, va, tha, qa) (copies when changed)."""
    tc = testcase(cfg)
    nx, ny, nk = int(cfg.nx), int(cfg.ny), int(cfg.nk)
    z = zh[:, 3, 3]  # zh(1,1,k)
    rng = GfortranRandom()
    rng.next()  # init3d.F: "rand-1"
    tha = tha.copy()

    def P(ii):  # Fortran i (or j) -> Python index
        return ii + 2

    def theta(levels, jmax, imax, ndraw):
        # for each level: draws over jj = 0..jmax, ii = 0..imax (i fastest),
        # ndraw per point; the first one perturbs theta
        for k, amp in levels:
            d = rng.draws((jmax + 1) * (imax + 1) * ndraw).reshape(
                jmax + 1, imax + 1, ndraw
            )
            tha[k, P(0) : P(jmax) + 1, P(0) : P(imax) + 1] += amp * (
                2.0 * d[:, :, 0] - 1.0
            )

    if tc == 0:
        theta([(k, 0.25) for k in range(1, nk + 1)], ny + 2, nx + 2, 2)
    elif tc in (1, 2, 6, 10, 11, 15):
        theta([(k, 0.10) for k in range(1, nk + 1) if z[k] <= 100.0], ny + 1, nx + 1, 2)
    elif tc == 3:
        if int(cfg.imoist) != 1:
            raise NotImplementedError("testcase=3 needs imoist=1")
        qa = qa.copy()
        nqv = int(cfg.nqv) - 1
        for k in range(1, nk + 1):
            if not z[k] <= 1600.0:
                continue
            d = rng.draws((ny + 2) * (nx + 2) * 2).reshape(ny + 2, nx + 2, 2)
            tha[k, P(0) : P(ny + 1) + 1, P(0) : P(nx + 1) + 1] += 0.1 * (
                2.0 * d[:, :, 0] - 1.0
            )
            qa[nqv, k, P(0) : P(ny + 1) + 1, P(0) : P(nx + 1) + 1] += (
                0.001 * 0.025 * (2.0 * d[:, :, 1] - 1.0)
            )
    elif tc in (4, 5, 7):
        theta(
            [(k, 0.10) for k in range(1, nk + 1) if z[k] <= 1000.0], ny + 1, nx + 1, 1
        )
    elif tc == 8:
        theta([(k, 0.10 - (k - 1) * 0.02) for k in range(1, 6)], ny + 1, nx + 1, 1)
    elif tc == 9:
        theta([(k, 0.10) for k in range(1, nk + 1) if z[k] < 50.0], ny + 1, nx + 1, 1)
    elif tc in (12, 13):
        ua, va = ua.copy(), va.copy()
        amp = 0.5
        nj, ni = ny, nx
        for k in range(1, nk + 1):
            d1 = np.zeros(tha.shape[1:])
            d2 = np.zeros(tha.shape[1:])
            d1[P(0) : P(nj + 1) + 1, P(0) : P(ni + 1) + 1] = u0[k, 3, 3]
            d2[P(0) : P(nj + 1) + 1, P(0) : P(ni + 1) + 1] = v0[k, 3, 3]
            nii = (nx + 1 - 5) - 5 + 1
            d = rng.draws((ny + 3) * nii * 2).reshape(ny + 3, nii, 2)
            sl = (slice(P(0), P(ny + 2) + 1), slice(P(5), P(nx - 4) + 1))
            d1[sl] = d1[sl] + amp * (2.0 * d[:, :, 0] - 1.0)
            d2[sl] = d2[sl] + amp * (2.0 * d[:, :, 1] - 1.0)
            J1, I1 = slice(P(1), P(nj + 1) + 1), slice(P(1), P(ni + 1) + 1)
            ua[k, J1, I1] = 0.5 * (d1[J1, P(0) : P(ni) + 1] + d1[J1, I1])
            va[k, J1, I1] = 0.5 * (d2[P(0) : P(nj) + 1, I1] + d2[J1, I1])
    else:
        theta([(k, 0.10) for k in range(1, nk + 1)], ny + 1, nx + 1, 1)
    return ua, va, tha, qa


def initial_tke(cfg, *, zh, zf, shape):
    """init3d.F: initial TKE of testcase 3, 4, 5, 9, 11 (idoles with
    prognostic TKE), on k = 1..nk+1 (numpy, w-point layout ``shape``), or
    None for the other cases."""
    tc = testcase(cfg)
    if tc not in (3, 4, 5, 9, 11):
        return None
    nk = int(cfg.nk)
    tke = np.zeros(shape)
    K = slice(1, nk + 2)
    if tc == 3:
        tke[K] = np.maximum(0.0, 1.0 - zf[K] / 3000.0)
    elif tc in (9, 11):
        r = 1.0 - zf[K] / 255.0
        tke[K] = np.where(zh[K] < 255.0, 0.4 * ((r * r) * r), 0.0)
    else:
        tke[:] = 1.0
    return tke
