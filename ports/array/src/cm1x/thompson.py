"""Thompson microphysics (ptype=3), ported from src/thompson.F and the
ptype=3 branch of src/mp_driver.F.

Routines: ``thompson_init`` (constants and lookup tables, built in numpy by
``cm1x.thompson_tables`` and handed to the backend), ``mp_thompson`` (the
column scheme, vectorised over every column at once), ``mp_gt_driver`` (the
3-D wrapper with CM1's precipitation/budget bookkeeping), ``rslf``/``rsif``,
``calc_refl10cm`` (rain/snow/graupel reflectivity without the melting-layer
branch) and ``mp_driver_ptype3`` (mp_driver.F).

Vectorisation: ``mp_thompson`` works on arrays ``[k, ...]`` (level first,
any column shape after it; ``mp_gt_driver`` passes the ``[nk, nj, ni]``
interior).  Every ``if`` of the Fortran becomes a mask (``xp.where``), the
per-column ``ANY(L_qx)`` tests and the early ``no_micro`` return are
per-column masks, the fall-speed fills below the precipitation top are
loops over k, and the sedimentation sub-steps (``nstep`` differs per
column) run ``max(nstep)`` masked sub-steps through ``Backend.fori_loop``
(a Python loop eagerly, ``lax.while_loop`` under ``jax.jit``; pass a static
``nstep_max`` for reverse-mode AD).  Table lookups are integer gathers
(``xp.take`` on the flattened table).

CM1 calls ``thompson_init`` without aerosols, so ``is_aerosol_aware`` is
false: the aerosol tendencies, ``Eff_aero``, ``activ_ncloud``,
``iceDeMott``/``iceKoop`` and the CCN table are not on the path and not
ported (``nc`` is the namelist ``nt_c``).  ``calc_effectRad`` is only
called with ``radopt=2`` (radiation is not ported in cm1x).  There are no
random numbers in the scheme.

``mp_gt_driver``'s fix-up of negative qv (``qv(i,j,k) = max(1e-7, ...)``)
can never trigger, because ``mp_thompson`` returns ``qv1d >= 1e-10``.
"""

from __future__ import annotations

from . import constants as C
from .kessler import _hw
from . import thompson_tables as T

_DEFAULTS = dict(
    eqtset=2,
    nt_c=100.0,
    tsmall=1.0e-4,
    qsmall=1.0e-14,
    nrain=1,
    radopt=0,
    efall=0,
    axisymm=0,
    nstep_max=0,
)


def _opt(cfg, name):
    if cfg is None:
        return _DEFAULTS[name]
    get = getattr(cfg, "get", None)
    return get(name, _DEFAULTS[name]) if get else getattr(cfg, name, _DEFAULTS[name])


# ---------------------------------------------------------------------------
# tables on the backend


class ThompsonTables:
    """Constants (Python floats / numpy) and gather tables (backend arrays)
    of ``thompson_init``."""

    def __init__(self, bk, raw: dict, nt_c_in: float):
        import numpy as np

        self.raw = raw
        self.nt_c_in = nt_c_in
        c = {k: v for k, v in raw.items() if np.ndim(v) == 0}
        for k, v in c.items():
            setattr(self, k, float(v))
        for k in ("nic2", "nii2", "nii3", "nir2", "nir3", "nis2", "nig2", "nig3"):
            setattr(self, k, int(raw[k]))
        self.niIN2 = int(raw["niIN2"])
        self.nic1 = int(raw["nic1"])
        for k in ("cce", "ccg", "ocg1", "ocg2", "cie", "cig", "cre", "crg", "cse",
                  "csg", "cge", "cgg"):  # fmt: skip
            setattr(self, k, np.asarray(raw[k]))
        for k in ("Dc", "Dr", "Ds", "t_Nc"):
            setattr(self, k, np.asarray(raw[k]))

        def A(x):
            return bk.asarray(np.ascontiguousarray(x))

        # per-nu_c (1..15) vectors gathered by the cloud-droplet code
        self.g_ccg = A(raw["ccg"])  # [5, 15]
        self.g_ocg1 = A(raw["ocg1"])
        self.g_ocg2 = A(raw["ocg2"])
        self.g_cce = A(raw["cce"])
        # powers of ten as Fortran's 10.**n (libgcc __powidf2), n = -60..60
        p10 = np.array([T.powi(10.0, n) for n in range(-60, 61)])
        self.p10 = A(p10)
        self.t_Efrw = A(raw["t_Efrw"])
        self.t_Efsw = A(raw["t_Efsw"])
        self.tps_iaus = A(raw["tps_iaus"])
        self.tni_iaus = A(raw["tni_iaus"])
        self.tpi_ide = A(raw["tpi_ide"])
        # sums of tables exactly as mp_thompson adds them (same order, so
        # bitwise the Fortran sums of the gathered values)
        r = {k: np.asarray(raw[k]) for k in raw if k.startswith("t")}
        self.racs_prr_cold = A(
            -(r["tmr_racs2"] + r["tcr_sacr2"] + r["tmr_racs1"] + r["tcr_sacr1"])
        )
        self.racs_prs_cold = A(
            r["tmr_racs2"] + r["tcr_sacr2"] - r["tcs_racs1"] - r["tms_sacr1"]
        )
        self.racs_prg_cold = A(
            r["tmr_racs1"] + r["tcr_sacr1"] + r["tcs_racs1"] + r["tms_sacr1"]
        )
        self.racs_pnr_cold = A(
            r["tnr_racs1"] + r["tnr_racs2"] + r["tnr_sacr1"] + r["tnr_sacr2"]
        )
        self.racs_prs_warm = A(
            -r["tcs_racs1"] - r["tms_sacr1"] + r["tmr_racs2"] + r["tcr_sacr2"]
        )
        self.racg_prg_cold = A(r["tmr_racg"] + r["tcr_gacr"])
        self.racg_pnr_cold = A(r["tnr_racg"] + r["tnr_gacr"])
        self.racg_prr_warm = A(r["tcg_racg"])
        self.racg_pnr_warm = A(r["tnr_gacr"])
        # freezing tables at the (constant, aerosol-unaware) IN index:
        # xni = 1000 per m**3 -> idx_IN
        xni = 1.0 * 1000.0
        niin = T.nint(np.log10(xni))
        n = niin + 1
        for nn in range(niin - 1, niin + 2):
            n = nn
            if 1.0 <= xni / T.powi(10.0, nn) < 10.0:
                break
        idx_in = int(xni / T.powi(10.0, n)) + 10 * (n - self.niIN2) - (n - self.niIN2)
        self.idx_IN = max(1, min(idx_in, T.ntb_IN))
        m = self.idx_IN - 1
        for k in ("tpi_qrfz", "tni_qrfz", "tpg_qrfz", "tnr_qrfz", "tpi_qcfz",
                  "tni_qcfz"):  # fmt: skip
            setattr(self, k, A(r[k][..., m]))


_INIT_CACHE: dict = {}


def thompson_init(bk, cfg=None, *, nt_c_in=None, cache=True):
    """thompson.F ``thompson_init`` (non-aerosol): constants and lookup
    tables for ``bk``.  ``nt_c_in`` defaults to the namelist ``nt_c``
    (droplets per cc).  Tables come from ``thompson_tables.tables`` (disk
    cached); the backend copy is memoised per backend/device."""
    if nt_c_in is None:
        nt_c_in = float(_opt(cfg, "nt_c"))
    key = (bk.name, str(bk.device), str(bk.dtype), float(nt_c_in))
    tb = _INIT_CACHE.get(key)
    if tb is None:
        tb = ThompsonTables(bk, T.tables(nt_c_in, cache=cache), float(nt_c_in))
        _INIT_CACHE[key] = tb
    return tb


# ---------------------------------------------------------------------------
# helpers


class _Ops:
    """Scalar-friendly elementwise helpers on ``bk.xp``."""

    def __init__(self, bk, like):
        self.bk = bk
        self.xp = xp = bk.xp
        self.dt = like.dtype
        self.dev = getattr(like, "device", None) if bk.name != "jax" else None
        self.i64 = xp.int64

    def A(self, v, like=None):
        if isinstance(v, (int, float)):
            return self.xp.asarray(float(v), dtype=self.dt, device=self.bk.device)
        return v

    def W(self, c, a, b):
        return self.xp.where(c, self.A(a), self.A(b))

    def mx(self, a, b):
        return self.xp.maximum(self.A(a), self.A(b))

    def mn(self, a, b):
        return self.xp.minimum(self.A(a), self.A(b))

    def nint(self, x):
        """Fortran NINT -> int64."""
        xp = self.xp
        a = xp.abs(x)
        f = xp.floor(a)
        r = f + self.W(a - f >= 0.5, 1.0, 0.0)
        return xp.astype(xp.where(x < 0, -r, r), self.i64)

    def int_(self, x):
        """Fortran INT (truncation) -> int64."""
        return self.xp.astype(self.xp.trunc(x), self.i64)

    def ii(self, v):
        return self.xp.asarray(v, dtype=self.i64, device=self.bk.device)

    def clip_i(self, i, lo, hi):
        xp = self.xp
        return xp.minimum(xp.maximum(i, self.ii(lo)), self.ii(hi))

    def take(self, table, *idx):
        """``table[idx0, idx1, ...]`` for 0-based int arrays (broadcast)."""
        xp = self.xp
        shp = table.shape
        flat = idx[0]
        for d, ix in zip(shp[1:], idx[1:]):
            flat = flat * d + ix
        out_shape = flat.shape
        v = xp.take(xp.reshape(table, (-1,)), xp.reshape(flat, (-1,)), axis=0)
        return xp.reshape(v, out_shape)

    def safe(self, mask, x, v):
        """``x`` where ``mask`` else a harmless value ``v`` (keeps masked-out
        lanes finite before logs/divisions/int conversions)."""
        return self.W(mask, x, v)


def rslf(xp, P, T_):
    """thompson.F ``RSLF``: liquid saturation mixing ratio."""
    c0, c1, c2 = 0.611583699e03, 0.444606896e02, 0.143177157e01
    c3, c4, c5 = 0.264224321e-1, 0.299291081e-3, 0.203154182e-5
    c6, c7, c8 = 0.702620698e-8, 0.379534310e-11, -0.321582393e-13
    X = xp.maximum(T_ - 273.16, xp.full_like(T_, -80.0))
    esl = c0 + X * (c1 + X * (c2 + X * (c3 + X * (c4 + X * (c5 + X * (c6 + X * (c7 + X * c8)))))))  # fmt: skip
    esl = xp.minimum(esl, P * 0.15)
    return 0.622 * esl / (P - esl)


def rsif(xp, P, T_):
    """thompson.F ``RSIF``: ice saturation mixing ratio."""
    c0, c1, c2 = 0.609868993e03, 0.499320233e02, 0.184672631e01
    c3, c4, c5 = 0.402737184e-1, 0.565392987e-3, 0.521693933e-5
    c6, c7, c8 = 0.307839583e-7, 0.105785160e-9, 0.161444444e-12
    X = xp.maximum(T_ - 273.16, xp.full_like(T_, -80.0))
    esi = c0 + X * (c1 + X * (c2 + X * (c3 + X * (c4 + X * (c5 + X * (c6 + X * (c7 + X * c8)))))))  # fmt: skip
    esi = xp.minimum(esi, P * 0.15)
    return 0.622 * esi / (P - esi)


def _snow_mom(tc0, cx):
    """a_, b_ of the general Field et al. moment relation (Python-float
    moment ``cx``; ``tc0`` an array), in thompson.F's term order."""
    sa, sb = T.sa, T.sb
    loga_ = (sa[0] + sa[1] * tc0 + sa[2] * cx + sa[3] * tc0 * cx + sa[4] * tc0 * tc0
             + sa[5] * cx * cx + sa[6] * tc0 * tc0 * cx + sa[7] * tc0 * cx * cx
             + sa[8] * tc0 * tc0 * tc0 + sa[9] * cx * cx * cx)  # fmt: skip
    b_ = (sb[0] + sb[1] * tc0 + sb[2] * cx + sb[3] * tc0 * cx + sb[4] * tc0 * tc0
          + sb[5] * cx * cx + sb[6] * tc0 * tc0 * cx + sb[7] * tc0 * cx * cx
          + sb[8] * tc0 * tc0 * tc0 + sb[9] * cx * cx * cx)  # fmt: skip
    return 10.0**loga_, b_


def _decade_index(o, tb, x, thr, n2, ntb, default):
    """The ``nic/nii/nir/...`` lookup-table index of thompson.F: the decade
    n in NINT(log10 x) +- 1 with 1 <= x/10**n < 10, then
    INT(x/10**n) + 9*(n-n2), clamped to [1, ntb]; ``default`` where
    x <= thr.  Returns a 0-based int64 index."""
    xp = o.xp
    ok = x > thr
    xs = o.safe(ok, x, 1.0)
    nI = o.nint(xp.log10(xs))
    nI = o.clip_i(nI, -59, 59)

    def ratio(n):
        return xs / o.take(tb.p10, n + 60)

    n = nI + 1
    for m in (1, 0, -1):  # first match wins: test from the last candidate
        q = ratio(nI + m)
        n = xp.where((q >= 1.0) & (q < 10.0), nI + m, n)
    idx = o.int_(ratio(n)) + 10 * (n - n2) - (n - n2)
    idx = o.clip_i(idx, 1, ntb)
    return xp.where(ok, idx, o.ii(default)) - 1


def _nu_c(o, nc):
    """nu_c = MIN(15, NINT(1000.E6/nc) + 2) (1-based), as int64."""
    return o.clip_i(o.nint(1000.0e6 / nc) + 2, -(10**9), 15)


# ---------------------------------------------------------------------------


def mp_thompson(
    bk,
    tb,
    cfg=None,
    *,
    qv1d,
    qc1d,
    qi1d,
    qr1d,
    qs1d,
    qg1d,
    ni1d,
    nr1d,
    nc1d,
    t1d,
    p1d,
    w1d,
    dzq,
    dt,
    nstep_max=None,
):
    """thompson.F ``mp_thompson`` (non-aerosol) on columns: every input is
    ``[nk, ...]`` (level first).  ``tb`` from ``thompson_init``.  ``dt`` a
    float or 0-d array.  Returns dict with the updated qv1d, qc1d, qi1d,
    qr1d, qs1d, qg1d, ni1d, nr1d, nc1d, t1d, the column precipitation
    pptrain, pptsnow, pptgraul, pptice (``[...]``), prv_rev, prw_vcd
    (``[nk, ...]``) and the fall speeds vtik, vtrk, vtsk, vtgk, vtck
    (``[nk+1, ...]``)."""
    xp = bk.xp
    o = _Ops(bk, t1d)
    W, mx, mn = o.W, o.mx, o.mn
    eqtset = int(_opt(cfg, "eqtset"))
    nk = t1d.shape[0]
    Z = xp.zeros_like(t1d)
    F = Z > 1.0  # all-false mask
    R1, R2, eps = T.R1, T.R2, T.eps
    Nt_c, Nt_c_max = tb.Nt_c, T.Nt_c_max
    am_r, am_i, am_g = T.am_r, T.am_i, T.am_g
    bm_r, bm_i = T.bm_r, T.bm_i
    mu_r, mu_i = T.mu_r, T.mu_i
    crg, cre, cgg, cge, cig, cie, csg, cse = (
        tb.crg, tb.cre, tb.cgg, tb.cge, tb.cig, tb.cie, tb.csg, tb.cse
    )  # fmt: skip

    def nuc_g(table_row, nu):
        """ccg(row, nu_c) etc. for a 1-based int nu_c array."""
        return o.take(table_row, nu - 1)

    ccg = [tb.g_ccg[r] for r in range(5)]
    cce = [tb.g_cce[r] for r in range(5)]

    dtsave = dt
    odt = 1.0 / dt
    odts = 1.0 / dtsave

    # --- column into local arrays
    temp = t1d
    qv = mx(1.0e-10, qv1d)
    pres = p1d
    rho = 0.622 * pres / (T.R * temp * (qv + 0.622))

    L_qc = qc1d > R1
    rc = W(L_qc, qc1d * rho, R1)
    nc = W(L_qc, Nt_c, 2.0)  # (non-aerosol: nc = Nt_c wherever qc > R1)
    qc1d = W(L_qc, qc1d, 0.0)
    nc1d = W(L_qc, nc1d, 0.0)

    L_qi = qi1d > R1
    ri = W(L_qi, qi1d * rho, R1)
    ni0 = mx(R2, ni1d * rho)
    ri_s = o.safe(L_qi, ri, 1.0)
    lami = cie[1] / 5.0e-6
    ni_lo = mn(9999.0e3, cig[0] * tb.oig2 * ri_s / am_i * lami**bm_i)
    ni0 = W(ni0 <= R2, ni_lo, ni0)
    lami = (am_i * cig[1] * tb.oig1 * ni0 / ri_s) ** tb.obmi
    xDi = (bm_i + mu_i + 1.0) * (1.0 / lami)
    ni_hi = cig[0] * tb.oig2 * ri_s / am_i * (cie[1] / 300.0e-6) ** bm_i
    ni0 = W(xDi < 5.0e-6, ni_lo, W(xDi > 300.0e-6, ni_hi, ni0))
    ni = W(L_qi, ni0, R2)
    qi1d = W(L_qi, qi1d, 0.0)
    ni1d = W(L_qi, ni1d, 0.0)

    L_qr = qr1d > R1
    rr = W(L_qr, qr1d * rho, R1)
    rr_s = o.safe(L_qr, rr, 1.0)
    nr0 = mx(R2, nr1d * rho)
    c672 = 3.0 + mu_r + 0.672

    def nr_of_mvd(mvd, rrx):
        lam = c672 / mvd
        return crg[1] * tb.org3 * rrx * lam**bm_r / am_r

    nr0 = W(nr0 <= R2, nr_of_mvd(1.0e-3, rr_s), nr0)
    lamr = (am_r * crg[2] * tb.org2 * nr0 / rr_s) ** tb.obmr
    mvd = c672 / lamr
    nr0 = W(
        mvd > 2.5e-3,
        nr_of_mvd(2.5e-3, rr_s),
        W(mvd < T.D0r * 0.75, nr_of_mvd(T.D0r * 0.75, rr_s), nr0),
    )
    nr = W(L_qr, nr0, R2)
    qr1d = W(L_qr, qr1d, 0.0)
    nr1d = W(L_qr, nr1d, 0.0)

    L_qs = qs1d > R1
    rs = W(L_qs, qs1d * rho, R1)
    qs1d = W(L_qs, qs1d, 0.0)
    L_qg = qg1d > R1
    rg = W(L_qg, qg1d * rho, R1)
    qg1d = W(L_qg, qg1d, 0.0)

    anyk = lambda m: xp.any(m, axis=0)  # noqa: E731
    micro = anyk(L_qc | L_qi | L_qr | L_qs | L_qg)

    # --- thermodynamics
    tempc = temp - 273.15
    rhof = xp.sqrt(T.rho_not / rho)
    rhof2 = xp.sqrt(rhof)
    qvs = rslf(xp, pres, temp)
    delQvs = mx(0.0, rslf(xp, pres, Z + 273.15) - qv)
    qvsi = W(tempc <= 0.0, rsif(xp, pres, temp), qvs)
    satw = qv / qvs
    sati = qv / qvsi
    ssatw = satw - 1.0
    ssati = sati - 1.0
    ssatw = W(xp.abs(ssatw) < eps, 0.0, ssatw)
    ssati = W(xp.abs(ssati) < eps, 0.0, ssati)
    micro = micro | anyk(ssati > 0.0)

    def diffu_of(tp, pr):
        return 2.11e-5 * (tp / 273.15) ** 1.94 * (101325.0 / pr)

    def visco_of(tc):
        return W(
            tc >= 0.0,
            (1.718 + 0.0049 * tc) * 1.0e-5,
            (1.718 + 0.0049 * tc - 1.2e-5 * tc * tc) * 1.0e-5,
        )

    diffu = diffu_of(temp, pres)
    visco = visco_of(tempc)
    ocp = 1.0 / (T.Cp * (1.0 + 0.887 * qv))
    vsc2 = xp.sqrt(rho / visco)
    lvap = T.lvap0 + (2106.0 - 4218.0) * tempc
    tcond = (5.69 + 0.0168 * tempc) * 1.0e-5 * 418.936
    if eqtset == 2:
        rcvm = 1.0 / (
            T.cv + T.cvv * qv1d + T.cpl * (qc1d + qr1d) + T.cpi * (qi1d + qs1d + qg1d)
        )
    else:
        rcvm = Z

    # --- snow moments
    def snow_moments(rsx, Lq, tempx, names):
        tc0 = mn(-0.1, tempx - 273.15)
        smob = W(Lq, rsx * tb.oams, 0.0)
        smo2 = smob  # bm_s == 2
        smo2s = o.safe(Lq, smo2, 1.0)
        out = {"smob": smob, "smo2": smo2}
        for name, cx in names:
            if name == "smo0":
                a_, b_ = _snow_mom0(tc0)
            elif name == "smo1":
                a_, b_ = _snow_mom1(tc0)
            else:
                a_, b_ = _snow_mom(tc0, cx)
            out[name] = W(Lq, a_ * smo2s**b_, 0.0)
        return out

    sm = snow_moments(
        rs,
        L_qs,
        temp,
        [("smo0", None), ("smo1", None), ("smoc", cse[0]), ("smoe", cse[12]),
         ("smof", cse[15])],
    )  # fmt: skip
    smob, smo0, smo1, smoc, smoe, smof = (
        sm["smob"], sm["smo0"], sm["smo1"], sm["smoc"], sm["smoe"], sm["smof"]
    )  # fmt: skip

    def graupel_slopes(rgx):
        ygra1 = xp.log10(mx(1.0e-9, rgx))
        zans1 = 3.0 + 2.0 / 7.0 * (ygra1 + 8.0)
        zans1 = mx(2.0, mn(zans1, 6.0))
        N0_exp = 10.0**zans1
        lam_exp = (N0_exp * am_g * cgg[0] / rgx) ** tb.oge1
        lamg = lam_exp * (cgg[2] * tb.ogg2 * tb.ogg1) ** tb.obmg
        return 1.0 / lamg, N0_exp / (cgg[1] * lam_exp) * lamg ** cge[1]

    def rain_slopes(nrx, rrx):
        lam = (am_r * crg[2] * tb.org2 * nrx / rrx) ** tb.obmr
        return 1.0 / lam, c672 / lam, nrx * tb.org2 * lam ** cre[1]

    ilamg, N0_g = graupel_slopes(rg)
    ilamr, mvd_r, N0_r = rain_slopes(nr, rr)

    # --- warm-rain processes
    zero = Z
    Lr_big = L_qr & (mvd_r > T.D0r)
    pnr_rcr = W(
        Lr_big, (1.0 - xp.exp(2300.0 * (mvd_r - 1950.0e-6))) * 2.0 * nr * rr, 0.0
    )

    nc_s = o.safe(L_qc, nc, Nt_c)
    rc_s = o.safe(L_qc, rc, 1.0)
    nu_c = _nu_c(o, nc_s)
    nuf = xp.astype(nu_c, o.dt)
    xDc = mx(T.D0c * 1.0e6, ((rc_s / (am_r * nc_s)) ** tb.obmr) * 1.0e6)
    lamc = (
        nc_s * am_r * nuc_g(ccg[1], nu_c) * nuc_g(tb.g_ocg1, nu_c) / rc_s
    ) ** tb.obmr
    mvd_c = W(L_qc, (3.0 + nuf + 0.672) / lamc, T.D0c)

    au = rc > 0.01e-3
    Dc_g = ((nuc_g(ccg[2], nu_c) * nuc_g(tb.g_ocg2, nu_c)) ** tb.obmr / lamc) * 1.0e6
    Dc_b3 = xDc * xDc * xDc * Dc_g * Dc_g * Dc_g - xDc * xDc * xDc * xDc * xDc * xDc
    Dc_b = o.safe(au, Dc_b3, 1.0) ** (1.0 / 6.0)  # masked lanes: no NaN warning
    zt = 6.25e-6 * xDc * Dc_b * Dc_b * Dc_b - 0.4
    zeta1 = 0.5 * (zt + xp.abs(zt))
    zeta = 0.027 * rc * zeta1
    tt = 0.5 * Dc_b - 7.5
    taud = 0.5 * (tt + xp.abs(tt)) + R1
    tau = 3.72 / (rc * taud)
    prr_wau = W(au, mn(rc * odts, zeta / tau), 0.0)
    pnr_wau = W(au, prr_wau / (am_r * nuf * 200.0 * T.D0r * T.D0r * T.D0r), 0.0)
    pnc_wau = W(au, mn(nc * odts, prr_wau / (am_r * mvd_c * mvd_c * mvd_c)), 0.0)

    lamr = 1.0 / ilamr
    m_rcw = Lr_big & (mvd_c > T.D0c)
    mvr_s = o.safe(m_rcw, mvd_r, T.D0r * 2.0)
    Dr = tb.Dr
    idx = 1 + o.int_(T.nbr * xp.log(mvr_s / Dr[0]) / float(_log(Dr[-1] / Dr[0])))
    idx = o.clip_i(idx, 1, T.nbr)
    idxc = o.clip_i(o.int_(o.safe(m_rcw, mvd_c, 2.0e-6) * 1.0e6), 1, T.nbc)
    Ef_rw = o.take(tb.t_Efrw, idx - 1, idxc - 1)
    fr9 = (lamr + T.fv_r) ** (-cre[8])
    prr_rcw = W(m_rcw, mn(rc * odts, rhof * tb.t1_qr_qc * Ef_rw * rc * N0_r * fr9), 0.0)
    pnc_rcw = W(m_rcw, mn(nc * odts, rhof * tb.t1_qr_qc * Ef_rw * nc * N0_r * fr9), 0.0)

    # --- ice-phase processes
    xDs = W(L_qs, smoc / o.safe(L_qs, smob, 1.0), 0.0)
    idx_tc = o.clip_i(o.nint(-tempc), 1, 45)
    idx_t = o.int_((tempc - 2.5) / 5.0) - 1
    idx_t = o.clip_i(-idx_t, 1, T.ntb_t)
    idx_c = _decade_index(o, tb, rc, T.r_c[0], tb.nic2, T.ntb_c, 1)
    idx_n = o.nint(1.0 + float(T.nbc) * xp.log(nc / tb.t_Nc[0]) / float(tb.nic1))
    idx_n = o.clip_i(idx_n, 1, T.nbc) - 1
    idx_i = _decade_index(o, tb, ri, T.r_i[0], tb.nii2, T.ntb_i, 1)
    idx_i1 = _decade_index(o, tb, ni, T.Nt_i[0], tb.nii3, T.ntb_i1, 1)
    idx_r = _decade_index(o, tb, rr, T.r_r[0], tb.nir2, T.ntb_r, 1)
    lam_exp = lamr * (crg[2] * tb.org2 * tb.org1) ** bm_r
    N0_exp = tb.org1 * rr / am_r * lam_exp ** cre[0]
    idx_r1 = _decade_index(o, tb, W(rr > T.r_r[0], N0_exp, 0.0), 0.0, tb.nir3,
                           T.ntb_r1, T.ntb_r1)  # fmt: skip
    idx_s = _decade_index(o, tb, rs, T.r_s[0], tb.nis2, T.ntb_s, 1)
    idx_g = _decade_index(o, tb, rg, T.r_g[0], tb.nig2, T.ntb_g, 1)
    lamg = 1.0 / ilamg
    lam_exp = lamg * (cgg[2] * tb.ogg2 * tb.ogg1) ** T.bm_g
    N0_exp = tb.ogg1 * rg / am_g * lam_exp ** cge[0]
    idx_g1 = _decade_index(o, tb, W(rg > T.r_g[0], N0_exp, 0.0), 0.0, tb.nig3,
                           T.ntb_g1, T.ntb_g1)  # fmt: skip

    # deposition/sublimation prefactor
    otemp = 1.0 / temp
    rvs = rho * qvsi
    Xs = T.lsub * otemp * T.oRv - 1.0
    rvs_p = rvs * otemp * Xs
    rvs_pp = rvs * (otemp * Xs * otemp * Xs + (-2.0 * T.lsub * otemp * otemp * otemp * T.oRv) + otemp * otemp)  # fmt: skip
    gamsc = T.lsub * diffu / tcond * rvs_p
    g1 = gamsc / (1.0 + gamsc)
    alphsc = mx(1.0e-9, 0.5 * g1 * g1 * rvs_pp / rvs_p * rvs / rvs_p)
    xsat = W(xp.abs(ssati) < 1.0e-9, 0.0, ssati)
    t1_subl = (4.0 * T.PI * (1.0 - alphsc * xsat + 2.0 * alphsc * alphsc * xsat * xsat
               - 5.0 * alphsc * alphsc * alphsc * xsat * xsat * xsat) / (1.0 + gamsc))  # fmt: skip

    # snow / graupel collecting cloud water
    m_qc = L_qc & (mvd_c > T.D0c)
    m_scw = m_qc & (xDs > T.D0s)
    Ds = tb.Ds
    xDs_s = o.safe(m_scw, xDs, T.D0s * 2.0)
    idx = 1 + o.int_(T.nbs * xp.log(xDs_s / Ds[0]) / float(_log(Ds[-1] / Ds[0])))
    idx = o.clip_i(idx, 1, T.nbs)
    Ef_sw = o.take(tb.t_Efsw, idx - 1, idxc_any(o, m_qc, mvd_c) - 1)
    prs_scw = W(m_scw, mn(rc * odts, rhof * tb.t1_qs_qc * Ef_sw * rc * smoe), 0.0)
    pnc_scw = W(m_scw, mn(nc * odts, rhof * tb.t1_qs_qc * Ef_sw * nc * smoe), 0.0)

    m_gw = m_qc & (rg >= T.r_g[0])
    xDg = (T.bm_g + T.mu_g + 1.0) * ilamg
    vtg = rhof * T.av_g * cgg[5] * tb.ogg3 * ilamg**T.bv_g
    stoke_g = mvd_c * mvd_c * vtg * T.rho_w / (9.0 * visco * xDg)
    m_gw = m_gw & (xDg > T.D0g)
    st_s = o.safe(m_gw, stoke_g, 1.0)
    Ef_gw = W(
        (stoke_g >= 0.4) & (stoke_g <= 10.0),
        0.55 * xp.log10(2.51 * st_s),
        W(stoke_g < 0.4, 0.0, W(stoke_g > 10.0, 0.77, 0.0)),
    )
    ilg9 = ilamg ** cge[8]
    prg_gcw = W(m_gw, rhof * tb.t1_qg_qc * Ef_gw * rc * N0_g * ilg9, 0.0)
    pnc_gcw = W(m_gw, mn(nc * odts, rhof * tb.t1_qg_qc * Ef_gw * nc * N0_g * ilg9), 0.0)

    # rain collecting snow / graupel (lookup tables)
    cold = temp < T.T_0
    m_rs = (rr >= T.r_r[0]) & (rs >= T.r_s[0])
    ix_s = (idx_s, idx_t - 1, idx_r1, idx_r)
    prr_rcs = W(m_rs & cold, mx(-rr * odts, o.take(tb.racs_prr_cold, *ix_s)), 0.0)
    prs_cold = mx(-rs * odts, o.take(tb.racs_prs_cold, *ix_s))
    prs_warm = mx(-rs * odts, o.take(tb.racs_prs_warm, *ix_s))
    prs_rcs = W(m_rs, W(cold, prs_cold, prs_warm), 0.0)
    prr_rcs = W(m_rs & ~cold, -prs_warm, prr_rcs)
    prg_rcs = W(m_rs & cold, mn((rr + rs) * odts, o.take(tb.racs_prg_cold, *ix_s)), 0.0)
    pnr_rcs = W(m_rs & cold, mn(nr * odts, o.take(tb.racs_pnr_cold, *ix_s)), 0.0)

    m_rg = (rr >= T.r_r[0]) & (rg >= T.r_g[0])
    ix_g = (idx_g1, idx_g, idx_r1, idx_r)
    prg_c = mn(rr * odts, o.take(tb.racg_prg_cold, *ix_g))
    prr_w = mn(rg * odts, o.take(tb.racg_prr_warm, *ix_g))
    prg_rcg = W(m_rg, W(cold, prg_c, -prr_w), 0.0)
    prr_rcg = W(m_rg, W(cold, -prg_c, prr_w), 0.0)
    pnr_rcg = W(
        m_rg,
        W(cold, mn(nr * odts, o.take(tb.racg_pnr_cold, *ix_g)),
          -5.0 * o.take(tb.racg_pnr_warm, *ix_g)),
        0.0,
    )  # fmt: skip

    # --- below 0C
    rate_max = (qv - qvsi) * rho * odts * 0.999
    ixf = (idx_r, idx_r1, idx_tc - 1)
    m1 = cold & (rr > T.r_r[0])
    m2 = cold & ~(rr > T.r_r[0]) & (rr > R1) & (temp < T.HGFR)
    prg_rfz = W(m1, o.take(tb.tpg_qrfz, *ixf) * odts, 0.0)
    pri_rfz = W(m1, o.take(tb.tpi_qrfz, *ixf) * odts, W(m2, rr * odts, 0.0))
    pni_rfz = W(m1, o.take(tb.tni_qrfz, *ixf) * odts, W(m2, nr * odts, 0.0))
    pnr_rfz = W(m1, mn(nr * odts, o.take(tb.tnr_qrfz, *ixf) * odts), 0.0)

    ixc = (idx_c, idx_n, idx_tc - 1)
    m1 = cold & (rc > T.r_c[0])
    m2 = cold & ~(rc > T.r_c[0]) & (rc > R1) & (temp < T.HGFR)
    pwf = mn(rc * odts, o.take(tb.tpi_qcfz, *ixc) * odts)
    pri_wfz = W(m1, pwf, W(m2, rc * odts, 0.0))
    pnw = mn(mn(nc * odts, pwf / (2.0 * T.xm0i)), o.take(tb.tni_qcfz, *ixc) * odts)
    pni_wfz = W(m1, pnw, W(m2, nc * odts, 0.0))

    m_nu = cold & ((ssati >= 0.25) | ((ssatw > eps) & (temp < 253.15)))
    xnc = mn(250.0e3, T.TNO * xp.exp(T.ATO * (T.T_0 - temp)))
    xni = ni + (pni_rfz + pni_wfz) * dtsave
    pni_inu = 0.5 * (xnc - xni + xp.abs(xnc - xni)) * odts
    pri_inu = W(m_nu, mn(rate_max, T.xm0i * pni_inu), 0.0)
    pni_inu = W(m_nu, pri_inu / T.xm0i, 0.0)

    # cloud ice deposition/sublimation and conversion to snow
    m_i = cold & L_qi
    ri_s = o.safe(L_qi, ri, 1.0)
    lami = (am_i * cig[1] * tb.oig1 * ni / ri_s) ** tb.obmi
    ilami = 1.0 / lami
    xDi = mx(tb.D0i, (bm_i + mu_i + 1.0) * ilami)
    xmi = am_i * xDi**bm_i
    oxmi = 1.0 / xmi
    pide = T.C_cube * t1_subl * diffu * ssati * rvs * tb.oig1 * cig[4] * ni * ilami
    neg = pide < 0.0
    pide_n = mx(mx(-ri * odts, pide), rate_max)
    pide_p = mn(pide, rate_max)
    tpi = o.take(tb.tpi_ide, idx_i, idx_i1)
    pri_ide = W(m_i, W(neg, pide_n, tpi * pide_p), 0.0)
    pni_ide = W(m_i & neg, mx(-ni * odts, pide_n * oxmi), 0.0)
    prs_ide = W(m_i & ~neg, (1.0 - tpi) * pide_p, 0.0)

    big_i = (idx_i == T.ntb_i - 1) | (xDi > 5.0 * T.D0s)
    small_i = xDi < 0.1 * T.D0s
    p_tab = mn(ri * 0.99 * odts, o.take(tb.tps_iaus, idx_i, idx_i1) * odts)
    n_tab = mn(ni * 0.95 * odts, o.take(tb.tni_iaus, idx_i, idx_i1) * odts)
    prs_iau = W(m_i, W(big_i, ri * 0.99 * odts, W(small_i, 0.0, p_tab)), 0.0)
    pni_iau = W(m_i, W(big_i, ni * 0.95 * odts, W(small_i, 0.0, n_tab)), 0.0)

    # snow / graupel deposition
    m_s = cold & L_qs
    C_snow = T.C_sqrd + (tempc + 1.5) * (T.C_cube - T.C_sqrd) / (-30.0 + 1.5)
    C_snow = mx(T.C_sqrd, mn(C_snow, T.C_cube))
    sde_fac = tb.t1_qs_sd * smo1 + tb.t2_qs_sd * rhof2 * vsc2 * smof
    psde = C_snow * t1_subl * diffu * ssati * rvs * sde_fac
    psde = W(psde < 0.0, mx(mx(-rs * odts, psde), rate_max), mn(psde, rate_max))
    prs_sde = W(m_s, psde, 0.0)

    m_g = cold & L_qg & (ssati < -eps)
    ilg10 = ilamg ** cge[9]
    ilg11 = ilamg ** cge[10]
    gde_fac = tb.t1_qg_sd * ilg10 + tb.t2_qg_sd * vsc2 * rhof2 * ilg11
    pgde = T.C_cube * t1_subl * diffu * ssati * rvs * N0_g * gde_fac
    pgde = W(pgde < 0.0, mx(mx(-rg * odts, pgde), rate_max), mn(pgde, rate_max))
    prg_gde = W(m_g, pgde, 0.0)

    # snow collecting ice, rain collecting ice
    m_si = m_i & (rs >= T.r_s[0])
    prs_sci = W(m_si, tb.t1_qs_qi * rhof * T.Ef_si * ri * smoe, 0.0)
    pni_sci = W(m_si, prs_sci * oxmi, 0.0)
    m_ri = m_i & (rr >= T.r_r[0]) & (mvd_r > 4.0 * xDi)
    pri_rci = W(m_ri, rhof * tb.t1_qr_qi * T.Ef_ri * ri * N0_r * fr9, 0.0)
    pnr_rci = W(m_ri, rhof * tb.t1_qr_qi * T.Ef_ri * ni * N0_r * fr9, 0.0)
    pni_rci = W(m_ri, pri_rci * oxmi, 0.0)
    fr8 = (lamr + T.fv_r) ** (-cre[7])
    prr_rci = W(
        m_ri, mn(rr * odts, rhof * tb.t2_qr_qi * T.Ef_ri * ni * N0_r * fr8), 0.0
    )
    prg_rci = W(m_ri, pri_rci + prr_rci, 0.0)

    # Hallett-Mossop
    m_hm = cold & (prg_gcw > eps) & (tempc > -8.0)
    tf = W(
        (tempc >= -5.0) & (tempc < -3.0),
        0.5 * (-3.0 - tempc),
        W((tempc > -8.0) & (tempc < -5.0), 0.33333333 * (8.0 + tempc), 0.0),
    )
    pni_ihm = W(m_hm, 3.5e8 * tf * prg_gcw, 0.0)
    pri_ihm = W(m_hm, T.xm0i * pni_ihm, 0.0)
    sden = o.safe(m_hm, prs_scw + prg_gcw, 1.0)
    prs_ihm = W(m_hm, prs_scw / sden * pri_ihm, 0.0)
    prg_ihm = W(m_hm, prg_gcw / sden * pri_ihm, 0.0)

    # rimed snow -> graupel
    m_rim = cold & (prs_scw > 2.0 * prs_sde) & (prs_sde > eps)
    r_frac = mn(30.0, prs_scw / o.safe(m_rim, prs_sde, 1.0))
    g_frac = mn(0.95, 0.15 + (r_frac - 2.0) * 0.028)
    vts_boost = W(m_rim, mn(1.5, 1.1 + (r_frac - 2.0) * 0.014), 1.0)
    prg_scw = W(m_rim, g_frac * prs_scw, 0.0)
    prs_scw = W(m_rim, (1.0 - g_frac) * prs_scw, prs_scw)

    # --- above 0C: melting (and sublimation when subsaturated)
    warm = ~cold
    m_s = warm & L_qs
    rs_s = o.safe(L_qs, rs, 1.0)
    me_fac = tempc * tcond - T.lvap0 * diffu * delQvs
    psml = me_fac * (tb.t1_qs_me * smo1 + tb.t2_qs_me * rhof2 * vsc2 * smof)
    psml = W(psml > 0.0, psml + 4218.0 * T.olfus * tempc * (prr_rcs + prs_scw), psml)
    psml = mn(rs * odts, mx(0.0, psml))
    prr_sml = W(m_s, psml, 0.0)
    pnr_sml = W(m_s, mn(smo0 * odts, smo0 / rs_s * psml * 10.0 ** (-0.25 * tempc)), 0.0)
    m_s2 = m_s & (ssati < 0.0)
    psde_w = T.C_cube * t1_subl * diffu * ssati * rvs * sde_fac
    prs_sde = W(m_s2, mx(-rs * odts, psde_w), prs_sde)

    m_g = warm & L_qg
    rg_s = o.safe(L_qg, rg, 1.0)
    pgml = me_fac * N0_g * (tb.t1_qg_me * ilg10 + tb.t2_qg_me * rhof2 * vsc2 * ilg11)
    pgml = mn(rg * odts, mx(0.0, pgml))
    prr_gml = W(m_g, pgml, 0.0)
    pnr_gml = W(
        m_g, N0_g * cgg[1] * ilamg ** cge[1] / rg_s * pgml * 10.0 ** (-0.5 * tempc), 0.0
    )
    m_g2 = m_g & (ssati < 0.0)
    prg_gde = W(m_g2, mx(-rg * odts, T.C_cube * t1_subl * diffu * ssati * rvs * N0_g * gde_fac), prg_gde)  # fmt: skip

    long_dt = warm & (o.A(dt) > 120.0)
    prr_rcw = W(long_dt, prr_rcw + prs_scw + prg_gcw, prr_rcw)
    prs_scw = W(long_dt, 0.0, prs_scw)
    prg_gcw = W(long_dt, 0.0, prg_gcw)

    # --- conservation
    pri_iha = zero
    sump = pri_inu + pri_ide + prs_ide + prs_sde + prg_gde + pri_iha
    rate_max = (qv - qvsi) * rho * odts * 0.999
    m = ((sump > eps) & (sump > rate_max)) | ((sump < -eps) & (sump < rate_max))
    ratio = rate_max / o.safe(m, sump, 1.0)
    pri_inu = W(m, pri_inu * ratio, pri_inu)
    pri_ide = W(m, pri_ide * ratio, pri_ide)
    pni_ide = W(m, pni_ide * ratio, pni_ide)
    prs_ide = W(m, prs_ide * ratio, prs_ide)
    prs_sde = W(m, prs_sde * ratio, prs_sde)
    prg_gde = W(m, prg_gde * ratio, prg_gde)

    sump = -prr_wau - pri_wfz - prr_rcw - prs_scw - prg_scw - prg_gcw
    rate_max = -rc * odts
    m = (sump < rate_max) & L_qc
    ratio = rate_max / o.safe(m, sump, 1.0)
    prr_wau, pri_wfz, prr_rcw, prs_scw, prg_scw, prg_gcw = (
        W(m, p * ratio, p)
        for p in (prr_wau, pri_wfz, prr_rcw, prs_scw, prg_scw, prg_gcw)
    )

    sump = pri_ide - prs_iau - prs_sci - pri_rci
    rate_max = -ri * odts
    m = (sump < rate_max) & L_qi
    ratio = rate_max / o.safe(m, sump, 1.0)
    pri_ide, prs_iau, prs_sci, pri_rci = (
        W(m, p * ratio, p) for p in (pri_ide, prs_iau, prs_sci, pri_rci)
    )

    sump = -prg_rfz - pri_rfz - prr_rci + prr_rcs + prr_rcg
    rate_max = -rr * odts
    m = (sump < rate_max) & L_qr
    ratio = rate_max / o.safe(m, sump, 1.0)
    prg_rfz, pri_rfz, prr_rci, prr_rcs, prr_rcg = (
        W(m, p * ratio, p) for p in (prg_rfz, pri_rfz, prr_rci, prr_rcs, prr_rcg)
    )

    sump = prs_sde - prs_ihm - prr_sml + prs_rcs
    rate_max = -rs * odts
    m = (sump < rate_max) & L_qs
    ratio = rate_max / o.safe(m, sump, 1.0)
    prs_sde, prs_ihm, prr_sml, prs_rcs = (
        W(m, p * ratio, p) for p in (prs_sde, prs_ihm, prr_sml, prs_rcs)
    )

    sump = prg_gde - prg_ihm - prr_gml + prg_rcg
    rate_max = -rg * odts
    m = (sump < rate_max) & L_qg
    ratio = rate_max / o.safe(m, sump, 1.0)
    prg_gde, prg_ihm, prr_gml, prg_rcg = (
        W(m, p * ratio, p) for p in (prg_gde, prg_ihm, prr_gml, prg_rcg)
    )

    pri_ihm = prs_ihm + prg_ihm
    one = o.A(1.0)
    ratio = mn(xp.abs(prr_rcg), xp.abs(prg_rcg))
    prr_rcg = ratio * xp.copysign(one, prr_rcg)
    prg_rcg = -prr_rcg
    hot = temp > T.T_0
    ratio = mn(xp.abs(prr_rcs), xp.abs(prs_rcs))
    prr_rcs = W(hot, ratio * xp.copysign(one, prr_rcs), prr_rcs)
    prs_rcs = W(hot, -prr_rcs, prs_rcs)

    # --- tendencies
    orho = 1.0 / rho
    lfus2 = T.lsub - lvap
    qvten = (-pri_inu - pri_iha - pri_ide - prs_ide - prs_sde - prg_gde) * orho
    qcten = (-prr_wau - pri_wfz - prr_rcw - prs_scw - prg_scw - prg_gcw) * orho
    ncten = (-pnc_wau - pnc_rcw - pni_wfz - pnc_scw - pnc_gcw) * orho

    xrc = mx(R1, (qc1d + qcten * dtsave) * rho)
    xnc = mx(2.0, (nc1d + ncten * dtsave) * rho)
    nu2 = _nu_c(o, xnc)
    lamc = (xnc * am_r * nuc_g(ccg[1], nu2) * nuc_g(tb.g_ocg1, nu2) / rc) ** tb.obmr
    xDc = (bm_r + xp.astype(nu2, o.dt) + 1.0) / lamc
    c2 = nuc_g(cce[1], nu2)
    lam_lo = c2 / T.D0c
    lam_hi = c2 / (T.D0r * 2.0)
    cg12 = nuc_g(ccg[0], nu2) * nuc_g(tb.g_ocg2, nu2)
    nct_lo = (cg12 * xrc / am_r * lam_lo**bm_r - nc1d * rho) * odts * orho
    nct_hi = (cg12 * xrc / am_r * lam_hi**bm_r - nc1d * rho) * odts * orho
    ncten = W(
        xrc > R1,
        W(xDc < T.D0c, nct_lo, W(xDc > T.D0r * 2.0, nct_hi, ncten)),
        -nc1d * odts,
    )
    xnc = mx(0.0, (nc1d + ncten * dtsave) * rho)
    ncten = W(xnc > Nt_c_max, (Nt_c_max - nc1d * rho) * odts * orho, ncten)

    qiten = (pri_inu + pri_iha + pri_ihm + pri_wfz + pri_rfz + pri_ide - prs_iau
             - prs_sci - pri_rci) * orho  # fmt: skip
    niten = (pni_inu + 0.0 + pni_ihm + pni_wfz + pni_rfz + pni_ide - pni_iau
             - pni_sci - pni_rci) * orho  # fmt: skip
    xri = mx(R1, (qi1d + qiten * dtsave) * rho)
    xni = mx(R2, (ni1d + niten * dtsave) * rho)
    lami = (am_i * cig[1] * tb.oig1 * xni / xri) ** tb.obmi
    xDi = (bm_i + mu_i + 1.0) * (1.0 / lami)
    l_lo = cie[1] / 5.0e-6
    l_hi = cie[1] / 300.0e-6
    nit_lo = (mn(9999.0e3, cig[0] * tb.oig2 * xri / am_i * l_lo**bm_i) - ni1d * rho) * odts * orho  # fmt: skip
    nit_hi = (cig[0] * tb.oig2 * xri / am_i * l_hi**bm_i - ni1d * rho) * odts * orho
    niten = W(
        xri > R1,
        W(xDi < 5.0e-6, nit_lo, W(xDi > 300.0e-6, nit_hi, niten)),
        -ni1d * odts,
    )
    xni = mx(0.0, (ni1d + niten * dtsave) * rho)
    niten = W(xni > 9999.0e3, (9999.0e3 - ni1d * rho) * odts * orho, niten)

    qrten = (prr_wau + prr_rcw + prr_sml + prr_gml + prr_rcs + prr_rcg - prg_rfz
             - pri_rfz - prr_rci) * orho  # fmt: skip
    nrten = (pnr_wau + pnr_sml + pnr_gml - (pnr_rfz + pnr_rcr + pnr_rcg + pnr_rcs
             + pnr_rci + pni_rfz)) * orho  # fmt: skip
    xrr = mx(R1, (qr1d + qrten * dtsave) * rho)
    xnr = mx(R2, (nr1d + nrten * dtsave) * rho)
    lamr = (am_r * crg[2] * tb.org2 * xnr / xrr) ** tb.obmr
    mvd = c672 / lamr
    nrt_hi = (nr_of_mvd(2.5e-3, xrr) - nr1d * rho) * odts * orho
    nrt_lo = (nr_of_mvd(T.D0r * 0.75, xrr) - nr1d * rho) * odts * orho
    mr = xrr > R1
    nrten = W(
        mr, W(mvd > 2.5e-3, nrt_hi, W(mvd < T.D0r * 0.75, nrt_lo, nrten)), -nr1d * odts
    )
    qrten = W(mr, qrten, -qr1d * odts)

    qsten = (prs_iau + prs_sde + prs_sci + prs_scw + prs_rcs + prs_ide - prs_ihm
             - prr_sml) * orho  # fmt: skip
    qgten = (prg_scw + prg_rfz + prg_gde + prg_rcg + prg_gcw + prg_rci + prg_rcs
             - prg_ihm - prr_gml) * orho  # fmt: skip

    s_dep = pri_inu + pri_ide + prs_ide + prs_sde + prg_gde + pri_iha
    s_frz = (pri_wfz + pri_rfz + prg_rfz + prs_scw + prg_scw + prg_gcw + prg_rcs
             + prs_rcs + prr_rci + prg_rcg)  # fmt: skip
    s_mlt = -prr_sml - prr_gml - prr_rcg - prr_rcs
    s_sub = prs_sde + prg_gde
    if eqtset == 1:
        tt_c = (T.lsub * ocp * s_dep + lfus2 * ocp * s_frz) * orho * 1.0
        tt_w = (T.lfus * ocp * s_mlt + T.lsub * ocp * s_sub) * orho * 1.0
    else:
        edep = (T.lsub - T.Rv * temp) * rcvm
        tt_c = (edep * s_dep + lfus2 * rcvm * s_frz) * orho * 1.0
        tt_w = (T.lfus * rcvm * s_mlt + edep * s_sub) * orho * 1.0
    tten = W(cold, tt_c, tt_w)

    # --- update to TAU+1 before condensation and sedimentation
    temp = t1d + dt * tten
    otemp = 1.0 / temp
    tempc = temp - 273.15
    qv = mx(1.0e-10, qv1d + dt * qvten)
    if eqtset == 1:
        rho = 0.622 * pres / (T.R * temp * (qv + 0.622))
    else:
        pres = rho * (T.R + T.Rv * qv) * temp
    rhof = xp.sqrt(T.rho_not / rho)
    rhof2 = xp.sqrt(rhof)
    qvs = rslf(xp, pres, temp)
    ssatw = qv / qvs - 1.0
    ssatw = W(xp.abs(ssatw) < eps, 0.0, ssatw)
    diffu = diffu_of(temp, pres)
    visco = visco_of(tempc)
    vsc2 = xp.sqrt(rho / visco)
    lvap = T.lvap0 + (2106.0 - 4218.0) * tempc
    tcond = (5.69 + 0.0168 * tempc) * 1.0e-5 * 418.936
    ocp = 1.0 / (T.Cp * (1.0 + 0.887 * qv))
    lvt2 = lvap * lvap * ocp * T.oRv * otemp * otemp

    qcn = qc1d + qcten * dt
    L_qc = qcn > R1
    rc = W(L_qc, qcn * rho, R1)
    nc = W(L_qc, Nt_c, 2.0)
    qin = qi1d + qiten * dt
    L_qi = qin > R1
    ri = W(L_qi, qin * rho, R1)
    ni = W(L_qi, mx(R2, (ni1d + niten * dt) * rho), R2)
    qrn = qr1d + qrten * dt
    L_qr = qrn > R1
    rr = W(L_qr, qrn * rho, R1)
    rr_s = o.safe(L_qr, rr, 1.0)
    nr0 = mx(R2, (nr1d + nrten * dt) * rho)
    lamr = (am_r * crg[2] * tb.org2 * nr0 / rr_s) ** tb.obmr
    mvd = c672 / lamr
    nr0 = W(
        mvd > 2.5e-3,
        nr_of_mvd(2.5e-3, rr_s),
        W(mvd < T.D0r * 0.75, nr_of_mvd(T.D0r * 0.75, rr_s), nr0),
    )
    nr = W(L_qr, nr0, R2)
    qsn = qs1d + qsten * dt
    L_qs = qsn > R1
    rs = W(L_qs, qsn * rho, R1)
    qgn = qg1d + qgten * dt
    L_qg = qgn > R1
    rg = W(L_qg, qgn * rho, R1)

    sm = snow_moments(rs, L_qs, temp, [("smoc", cse[0]), ("smod", cse[13])])
    smob, smoc = sm["smob"], sm["smoc"]
    ilamg, N0_g = graupel_slopes(rg)
    ilamr, mvd_r, N0_r = rain_slopes(nr, rr)

    # --- condensation / evaporation of cloud water
    orho = 1.0 / rho
    m_cd = (ssatw > eps) | ((ssatw < -eps) & L_qc)
    clap = (qv - qvs) / (1.0 + lvt2 * qvs)
    for _ in range(3):
        e = xp.exp(lvt2 * clap)
        fcd = qvs * e - qv + clap
        dfcd = qvs * lvt2 * e + 1.0
        clap = clap - fcd / dfcd
    xrc = rc + clap * rho
    pos = xrc > R1
    prw_vcd = W(m_cd, W(pos, clap * odt, -rc * orho * odt), 0.0)
    pnc_wcd = W(
        m_cd,
        W(pos, W(clap > eps, 0.5 * (Nt_c - nc + xp.abs(Nt_c - nc)) * odts * orho, 0.0),
          -nc * orho * odt),
        0.0,
    )  # fmt: skip
    qvten = qvten - prw_vcd
    qcten = qcten + prw_vcd
    ncten = ncten + pnc_wcd
    if eqtset == 1:
        tten = tten + W(m_cd, lvap * ocp * prw_vcd * 1.0, 0.0)
    else:
        tten = tten + W(m_cd, (lvap - T.Rv * temp) * rcvm * prw_vcd * 1.0, 0.0)
    rc_n = mx(R1, (qc1d + dt * qcten) * rho)
    rc = W(m_cd, rc_n, rc)
    L_qc = L_qc & ~(m_cd & (rc_n == R1))
    nc = W(m_cd, Nt_c, nc)
    qv = W(m_cd, mx(1.0e-10, qv1d + dt * qvten), qv)
    temp = W(m_cd, t1d + dt * tten, temp)
    if eqtset == 1:
        rho = W(m_cd, 0.622 * pres / (T.R * temp * (qv + 0.622)), rho)
    else:
        pres = W(m_cd, rho * (T.R + T.Rv * qv) * temp, pres)
    qvs = W(m_cd, rslf(xp, pres, temp), qvs)
    ssatw = W(m_cd, qv / qvs - 1.0, ssatw)

    # --- rain evaporation
    m_ev = (ssatw < -eps) & L_qr & ~(prw_vcd > 0.0)
    tempc = temp - 273.15
    otemp = 1.0 / temp
    orho = 1.0 / rho
    rhof = W(m_ev, xp.sqrt(T.rho_not * orho), rhof)
    rhof2 = W(m_ev, xp.sqrt(rhof), rhof2)
    diffu = W(m_ev, diffu_of(temp, pres), diffu)
    visco = W(m_ev, visco_of(tempc), visco)
    vsc2 = W(m_ev, xp.sqrt(rho / visco), vsc2)
    lvap = W(m_ev, T.lvap0 + (2106.0 - 4218.0) * tempc, lvap)
    tcond = W(m_ev, (5.69 + 0.0168 * tempc) * 1.0e-5 * 418.936, tcond)
    ocp = W(m_ev, 1.0 / (T.Cp * (1.0 + 0.887 * qv)), ocp)
    rvs = rho * qvs
    Xs = lvap * otemp * T.oRv - 1.0
    rvs_p = rvs * otemp * Xs
    rvs_pp = rvs * (otemp * Xs * otemp * Xs + (-2.0 * lvap * otemp * otemp * otemp * T.oRv) + otemp * otemp)  # fmt: skip
    gamsc = lvap * diffu / tcond * rvs_p
    g1 = gamsc / (1.0 + gamsc)
    alphsc = mx(1.0e-9, 0.5 * g1 * g1 * rvs_pp / rvs_p * rvs / rvs_p)
    xsat = mn(-1.0e-9, ssatw)
    t1_evap = (2.0 * T.PI * (1.0 - alphsc * xsat + 2.0 * alphsc * alphsc * xsat * xsat
               - 5.0 * alphsc * alphsc * alphsc * xsat * xsat * xsat) / (1.0 + gamsc))  # fmt: skip
    lamr = 1.0 / ilamr
    quick = (qv / qvs < 0.95) & (rr * orho <= 1.0e-8)
    pev = (
        t1_evap
        * diffu
        * (-ssatw)
        * N0_r
        * rvs
        * (
            tb.t1_qr_ev * ilamr ** cre[9]
            + tb.t2_qr_ev * vsc2 * rhof2 * ((lamr + 0.5 * T.fv_r) ** (-cre[10]))
        )
    )
    rate_max = mn(rr * orho * odts, (qvs - qv) * odts)
    pev = mn(rate_max, pev * orho)
    eva = mn(1.0, 0.01 + (0.99 - 0.01) * (tempc / 20.0))
    pev = W(prr_gml > 0.0, pev * eva, pev)
    prv_rev = W(m_ev, W(quick, rr * orho * odts, pev), 0.0)
    pnr_rev = W(m_ev, mn(nr * 0.99 * orho * odts, prv_rev * nr / rr), 0.0)
    qrten = qrten - prv_rev
    qvten = qvten + prv_rev
    nrten = nrten - pnr_rev
    if eqtset == 1:
        tten = tten - W(m_ev, lvap * ocp * prv_rev * 1.0, 0.0)
    else:
        tten = tten - W(m_ev, (lvap - T.Rv * temp) * rcvm * prv_rev * 1.0, 0.0)
    rr = W(m_ev, mx(R1, (qr1d + dt * qrten) * rho), rr)
    qv = W(m_ev, mx(1.0e-10, qv1d + dt * qvten), qv)
    nr = W(m_ev, mx(R2, (nr1d + dt * nrten) * rho), nr)
    temp = W(m_ev, t1d + dt * tten, temp)
    if eqtset == 1:
        rho = W(m_ev, 0.622 * pres / (T.R * temp * (qv + 0.622)), rho)
    else:
        pres = W(m_ev, rho * (T.R + T.Rv * qv) * temp, pres)

    # --- fall speeds, sub-step counts
    kidx = xp.reshape(
        xp.arange(nk, dtype=o.i64, device=bk.device), (nk,) + (1,) * (t1d.ndim - 1)
    )
    any_qr, any_qc, any_qi, any_qs, any_qg = (
        anyk(L_qr), anyk(L_qc), anyk(L_qi), anyk(L_qs), anyk(L_qg)
    )  # fmt: skip

    def fill_down(vals, have):
        """vt(k) = vals(k) where have(k) else vt(k+1), from the top (vt(nk+1) = 0)."""
        out = [None] * nk
        prev = Z[0]
        for k in range(nk - 1, -1, -1):
            prev = W(have[k], vals[k], prev)
            out[k] = prev
        return xp.stack(out, axis=0)

    def steps(vt):
        """ksed1 (0-based level), nstep of one species."""
        fast = vt > 1.0e-3
        ks = xp.max(W(fast, xp.astype(kidx, o.dt), 0.0), axis=0)
        ks = xp.astype(ks, o.i64)
        ks = xp.where(ks == nk - 1, o.ii(nk - 2), ks)
        dtp = dzq / o.safe(fast, vt, 1.0)
        ns = W(fast, xp.trunc(dt / dtp + 1.0), 0.0)
        ns = xp.max(ns, axis=0)
        return ks, ns

    def onstep_of(any_x, ns):
        return W(any_x & (ns > 0.0), 1.0 / o.safe(ns > 0.0, ns, 1.0), 1.0)

    rhof = W(any_qr[None], xp.sqrt(T.rho_not / rho), rhof)
    rr_s = o.safe(rr > R1, rr, 1.0)
    lamr = (am_r * crg[2] * tb.org2 * nr / rr_s) ** tb.obmr
    fl = lamr + T.fv_r
    vtr = rhof * T.av_r * crg[5] * tb.org3 * lamr ** cre[2] * (fl ** (-cre[5]))
    vtnr = rhof * T.av_r * crg[6] / crg[11] * lamr ** cre[11] * (fl ** (-cre[6]))
    have = rr > R1
    vtrk = W(any_qr[None], fill_down(vtr, have), 0.0)
    vtnrk = W(any_qr[None], fill_down(vtnr, have), 0.0)
    ks_r, ns_r = steps(mx(vtrk, vtnrk))
    on_r = onstep_of(any_qr, ns_r)

    # cloud: ksed1(5) = last k with rc > R2 up to the level where the
    # height above ground exceeds 500 m
    ks_c = xp.zeros(t1d.shape[1:], dtype=o.i64, device=bk.device)
    stop = F[0]
    hgt_agl = Z[0]
    for k in range(nk - 1):
        ks_c = xp.where(~stop & (rc[k] > R2), o.ii(k), ks_c)
        hgt_agl = hgt_agl + dzq[k]
        stop = stop | (hgt_agl > 500.0)
    m_c = (kidx <= ks_c[None]) & (rc > R1) & (w1d < 1.0e-1) & any_qc[None]
    nc_s = o.safe(m_c, nc, Nt_c)
    nu3 = _nu_c(o, nc_s)
    lamc = (nc_s * am_r * nuc_g(ccg[1], nu3) * nuc_g(tb.g_ocg1, nu3) / o.safe(m_c, rc, 1.0)) ** tb.obmr  # fmt: skip
    ilamc = 1.0 / lamc
    vtck = W(m_c, rhof * T.av_c * nuc_g(ccg[4], nu3) * nuc_g(tb.g_ocg2, nu3) * (ilamc * ilamc), 0.0)  # fmt: skip
    vtnck = W(m_c, rhof * T.av_c * nuc_g(ccg[3], nu3) * nuc_g(tb.g_ocg1, nu3) * (ilamc * ilamc), 0.0)  # fmt: skip

    have = ri > R1
    lami = (am_i * cig[1] * tb.oig1 * ni / o.safe(have, ri, 1.0)) ** tb.obmi
    ilami = 1.0 / lami
    vtik = W(
        any_qi[None], fill_down(rhof * T.av_i * cig[2] * tb.oig2 * ilami, have), 0.0
    )
    vtnik = W(
        any_qi[None], fill_down(rhof * T.av_i * cig[5] / cig[6] * ilami, have), 0.0
    )
    ks_i, ns_i = steps(vtik)
    on_i = onstep_of(any_qi, ns_i)

    have = rs > R1
    xDs = smoc / o.safe(have, smob, 1.0)
    Mrat = 1.0 / o.safe(have, xDs, 1.0)
    ils1 = 1.0 / (Mrat * T.Lam0 + T.fv_s)
    ils2 = 1.0 / (Mrat * T.Lam1 + T.fv_s)
    t1v = T.Kap0 * csg[3] * ils1 ** cse[3]
    t2v = T.Kap1 * Mrat**T.mu_s * csg[9] * ils2 ** cse[9]
    ils1 = 1.0 / (Mrat * T.Lam0)
    ils2 = 1.0 / (Mrat * T.Lam1)
    t3v = T.Kap0 * csg[0] * ils1 ** cse[0]
    t4v = T.Kap1 * Mrat**T.mu_s * csg[6] * ils2 ** cse[6]
    vts = rhof * T.av_s * (t1v + t2v) / (t3v + t4v)
    SR = rs / (rs + rr)
    vts = W(temp > T.T_0 + 0.1, vts * SR + (1.0 - SR) * vtrk, vts * vts_boost)
    vtsk = W(any_qs[None], fill_down(vts, have), 0.0)
    ks_s, ns_s = steps(vtsk)
    on_s = onstep_of(any_qs, ns_s)

    have = rg > R1
    vtg = rhof * T.av_g * cgg[5] * tb.ogg3 * ilamg**T.bv_g
    vtg = W(temp > T.T_0, mx(vtg, vtrk), vtg)
    vtgk = W(any_qg[None], fill_down(vtg, have), 0.0)
    ks_g, ns_g = steps(vtgk)
    on_g = onstep_of(any_qg, ns_g)

    # --- sedimentation
    odzq = 1.0 / dzq
    orho = 1.0 / rho
    top = kidx == nk - 1
    ksub = kidx < nk - 1
    zk = Z[:1]

    def up(a):  # a(k+1), zero above the top
        return xp.concat([a[1:], zk], axis=0)

    def nsteps(any_x, on):
        return W(any_x, xp.round(1.0 / on), 0.0)

    def sediment(any_x, on, ks, vt_m, vt_n, q, n, qten_, nten_, nmax):
        """The ``nstep`` sub-steps of one species (``n``/``vt_n`` None for
        snow and graupel), masked per column; returns q, n, qten, nten, ppt."""
        nst = nsteps(any_x, on)
        lev = ksub & (kidx <= ks[None])
        upd_k = top | lev
        has_n = n is not None

        def flux_div(s):
            return W(top, -s, W(lev, up(s) - s, 0.0))

        def body(it, c):
            q, n, qt, nt, ppt = c
            act = nst > it
            upd = act[None] & upd_k
            sq = vt_m * q
            dq = flux_div(sq)
            qt = W(upd, qt + dq * odzq * on * orho, qt)
            q = W(upd, mx(R1, q + dq * odzq * dt * on), q)
            if has_n:
                sn = vt_n * n
                dn = flux_div(sn)
                nt = W(upd, nt + dn * odzq * on * orho, nt)
                n = W(upd, mx(R2, n + dn * odzq * dt * on), n)
            ppt = W(act & (q[0] > R1 * 1000.0), ppt + sq[0] * dt * on, ppt)
            return q, n, qt, nt, ppt

        if nmax is None:
            nmax = xp.max(nst) if nst.ndim else nst
        c = (q, n if has_n else Z, qten_, nten_ if has_n else Z, Z[0])
        q, n_, qt, nt, ppt = bk.fori_loop(nmax, body, c, max_n=nstep_max or None)
        return q, (n_ if has_n else None), qt, (nt if has_n else None), ppt

    rr, nr, qrten, nrten, pptrain = sediment(
        any_qr, on_r, ks_r, vtrk, vtnrk, rr, nr, qrten, nrten, None
    )
    # cloud water: one step over k <= ksed1(5)
    sc = vtck * rc
    lev_c = (kidx <= ks_c[None]) & any_qc[None]
    qcten = W(lev_c, qcten + (up(sc) - sc) * odzq * orho, qcten)
    sn = vtnck * nc
    ncten = W(lev_c, ncten + (up(sn) - sn) * odzq * orho, ncten)
    ri, ni, qiten, niten, pptice = sediment(
        any_qi, on_i, ks_i, vtik, vtnik, ri, ni, qiten, niten, None
    )
    rs, _, qsten, _, pptsnow = sediment(
        any_qs, on_s, ks_s, vtsk, None, rs, None, qsten, None, None
    )
    rg, _, qgten, _, pptgraul = sediment(
        any_qg, on_g, ks_g, vtgk, None, rg, None, qgten, None, None
    )

    # --- instant melting of ice above 0C, freezing of cloud water below HGFR
    xri = mx(0.0, qi1d + qiten * dt)
    m = (temp > T.T_0) & (xri > 0.0)
    qcten = W(m, qcten + xri * odt, qcten)
    ncten = W(m, ncten + ni1d * odt, ncten)
    qiten = W(m, qiten - xri * odt, qiten)
    niten = W(m, -ni1d * odt, niten)
    if eqtset == 1:
        tten = W(m, tten - T.lfus * ocp * xri * odt * 1.0, tten)
    else:
        tten = W(m, tten - T.lfus * rcvm * xri * odt * 1.0, tten)
    xrc = mx(0.0, qc1d + qcten * dt)
    m = (temp < T.HGFR) & (xrc > 0.0)
    xnc = nc1d + ncten * dt
    qiten = W(m, qiten + xrc * odt, qiten)
    niten = W(m, niten + xnc * odt, niten)
    qcten = W(m, qcten - xrc * odt, qcten)
    ncten = W(m, ncten - xnc * odt, ncten)
    if eqtset == 1:
        tten = W(m, tten + (T.lsub - lvap) * ocp * xrc * odt * 1.0, tten)
    else:
        tten = W(m, tten + T.lfus * rcvm * xrc * odt * 1.0, tten)

    # --- final values
    t_o = t1d + tten * dt
    qv_o = mx(1.0e-10, qv1d + qvten * dt)
    qc_o = qc1d + qcten * dt
    nc_o = mx(2.0 / rho, mn(nc1d + ncten * dt, Nt_c_max))
    okc = qc_o > R1
    qcs = o.safe(okc, qc_o, 1.0)
    ncs = o.safe(okc, nc_o, 1.0)
    nu4 = _nu_c(o, ncs * rho)
    lamc = (am_r * nuc_g(ccg[1], nu4) * nuc_g(tb.g_ocg1, nu4) * ncs / qcs) ** tb.obmr
    xDc = (bm_r + xp.astype(nu4, o.dt) + 1.0) / lamc
    c2 = nuc_g(cce[1], nu4)
    lamc = W(xDc < T.D0c, c2 / T.D0c, W(xDc > T.D0r * 2.0, c2 / (T.D0r * 2.0), lamc))
    ncf = mn(nuc_g(ccg[0], nu4) * nuc_g(tb.g_ocg2, nu4) * qcs / am_r * lamc**bm_r, Nt_c_max / rho)  # fmt: skip
    nc_o = W(okc, ncf, 0.0)
    qc_o = W(okc, qc_o, 0.0)

    qi_o = qi1d + qiten * dt
    ni_o = mx(R2 / rho, ni1d + niten * dt)
    oki = qi_o > R1
    qis = o.safe(oki, qi_o, 1.0)
    lami = (am_i * cig[1] * tb.oig1 * ni_o / qis) ** tb.obmi
    xDi = (bm_i + mu_i + 1.0) * (1.0 / lami)
    lami = W(xDi < 5.0e-6, cie[1] / 5.0e-6, W(xDi > 300.0e-6, cie[1] / 300.0e-6, lami))
    nif = mn(cig[0] * tb.oig2 * qis / am_i * lami**bm_i, 9999.0e3 / rho)
    ni_o = W(oki, nif, 0.0)
    qi_o = W(oki, qi_o, 0.0)

    qr_o = qr1d + qrten * dt
    nr_o = mx(R2 / rho, nr1d + nrten * dt)
    okr = qr_o > R1
    qrs = o.safe(okr, qr_o, 1.0)
    lamr = (am_r * crg[2] * tb.org2 * nr_o / qrs) ** tb.obmr
    mvd = c672 / lamr
    mvd = W(mvd > 2.5e-3, 2.5e-3, W(mvd < T.D0r * 0.75, T.D0r * 0.75, mvd))
    lamr = c672 / mvd
    nrf = crg[1] * tb.org3 * qrs * lamr**bm_r / am_r
    nr_o = W(okr, nrf, 0.0)
    qr_o = W(okr, qr_o, 0.0)
    qs_o = qs1d + qsten * dt
    qs_o = W(qs_o <= R1, 0.0, qs_o)
    qg_o = qg1d + qgten * dt
    qg_o = W(qg_o <= R1, 0.0, qg_o)

    # columns with nothing to do return right after the initial clean-up
    mc = micro[None]
    zt1 = xp.zeros_like(vtrk[:1])

    def vt_out(v):
        return xp.concat([W(mc, v, 0.0), zt1], axis=0)

    return {
        "qv1d": W(mc, qv_o, qv1d),
        "qc1d": W(mc, qc_o, qc1d),
        "qi1d": W(mc, qi_o, qi1d),
        "qr1d": W(mc, qr_o, qr1d),
        "qs1d": W(mc, qs_o, qs1d),
        "qg1d": W(mc, qg_o, qg1d),
        "ni1d": W(mc, ni_o, ni1d),
        "nr1d": W(mc, nr_o, nr1d),
        "nc1d": W(mc, nc_o, nc1d),
        "t1d": W(mc, t_o, t1d),
        "pptrain": W(micro, pptrain, 0.0),
        "pptsnow": W(micro, pptsnow, 0.0),
        "pptgraul": W(micro, pptgraul, 0.0),
        "pptice": W(micro, pptice, 0.0),
        "prv_rev": W(mc, prv_rev, 0.0),
        "prw_vcd": W(mc, prw_vcd, 0.0),
        "vtik": vt_out(vtik),
        "vtrk": vt_out(vtrk),
        "vtsk": vt_out(vtsk),
        "vtgk": vt_out(vtgk),
        "vtck": vt_out(vtck),
    }


def idxc_any(o, m, mvd_c):
    """INT(mvd_c*1.E6) (1-based) for the Ef_sw lookup, clamped."""
    return o.clip_i(o.int_(o.safe(m, mvd_c, 2.0e-6) * 1.0e6), 1, T.nbc)


def _log(x):
    import math

    return math.log(x)


def _snow_mom0(tc0):
    sa, sb = T.sa, T.sb
    loga_ = sa[0] + sa[1] * tc0 + sa[4] * tc0 * tc0 + sa[8] * tc0 * tc0 * tc0
    b_ = sb[0] + sb[1] * tc0 + sb[4] * tc0 * tc0 + sb[8] * tc0 * tc0 * tc0
    return 10.0**loga_, b_


def _snow_mom1(tc0):
    sa, sb = T.sa, T.sb
    loga_ = (sa[0] + sa[1] * tc0 + sa[2] + sa[3] * tc0 + sa[4] * tc0 * tc0 + sa[5]
             + sa[6] * tc0 * tc0 + sa[7] * tc0 + sa[8] * tc0 * tc0 * tc0 + sa[9])  # fmt: skip
    b_ = (sb[0] + sb[1] * tc0 + sb[2] + sb[3] * tc0 + sb[4] * tc0 * tc0 + sb[5]
          + sb[6] * tc0 * tc0 + sb[7] * tc0 + sb[8] * tc0 * tc0 * tc0 + sb[9])  # fmt: skip
    return 10.0**loga_, b_


# ---------------------------------------------------------------------------
# 3-D wrapper and the mp_driver branch


def mp_gt_driver(
    bk,
    g,
    cfg=None,
    tb=None,
    *,
    qv,
    qc,
    qr,
    qi,
    qs,
    qg,
    ni,
    nr,
    t3d,
    p,
    w,
    dz,
    dt,
    rain,
    prate,
    tcond,
    tevac,
    tevar,
    train,
    ruh,
    rvh,
    rmh,
    rr,
    getvt=False,
    vt=None,
    nstep_max=None,
):
    """thompson.F ``mp_gt_driver`` as CM1 calls it (non-aerosol, no
    reflectivity or effective radii): ``mp_thompson`` on every interior
    column at once, then CM1's rain/prate/budget bookkeeping.

    3-D arrays ``g.s_shape`` (``w`` may be ``g.w_shape``), ``rain``
    ``[nrain, j, i]``, ``prate`` ``[j, i]``, budgets scalars.  ``vt``: the
    (vtc, vtr, vti, vts, vtg) arrays filled on the interior when ``getvt``.
    Budget sums use ``xp.sum`` (not CM1's i-then-j order).  Returns dict
    with the updated qv..nr, t3d, rain, prate, tcond, tevac, tevar, train
    and vt."""
    xp = bk.xp
    if tb is None:
        tb = thompson_init(bk, cfg)
    S = (g.K, g.J, g.I)
    pt = p[S]
    tt = t3d[S]
    qvt = qv[S]
    rho = 0.622 * pt / (T.R * tt * (qvt + 0.622))
    r = mp_thompson(
        bk,
        tb,
        cfg,
        qv1d=qvt,
        qc1d=qc[S],
        qi1d=qi[S],
        qr1d=qr[S],
        qs1d=qs[S],
        qg1d=qg[S],
        ni1d=ni[S],
        nr1d=nr[S],
        nc1d=tb.Nt_c / rho,
        t1d=tt,
        p1d=pt,
        w1d=w[S],
        dzq=dz[S],
        dt=dt,
        nstep_max=nstep_max,
    )
    ppt = r["pptrain"] + r["pptsnow"] + r["pptgraul"] + r["pptice"]
    rain_i = rain[:, g.J, g.I] + (0.1 * ppt)[None]
    rdt = 1.0 / dt
    ru, rv = _hw(g, ruh, rvh)
    ru2, rv2 = ru[0], rv[0]
    bud1 = xp.sum(ppt * ru2 * rv2 * g.dx * g.dy)
    tem1 = g.dx * g.dy * g.dz
    tem2 = dt * rr[S] * ru * rv * rmh[S] * tem1
    z = xp.zeros_like(tem2)
    prw = r["prw_vcd"]
    out = {
        "rain": bk.set(bk.copy(rain), (slice(None), g.J, g.I), rain_i),
        "prate": bk.set(bk.copy(prate), (g.J, g.I), rdt * ppt),
        "train": train + bud1,
        "tevar": tevar + xp.sum(tem2 * r["prv_rev"]),
        "tcond": tcond + xp.sum(tem2 * xp.maximum(prw, z)),
        "tevac": tevac - xp.sum(tem2 * xp.minimum(prw, z)),
    }
    for name, key in (("qv", "qv1d"), ("qc", "qc1d"), ("qr", "qr1d"), ("qi", "qi1d"),
                      ("qs", "qs1d"), ("qg", "qg1d"), ("ni", "ni1d"), ("nr", "nr1d"),
                      ("t3d", "t1d")):  # fmt: skip
        src = {"qv": qv, "qc": qc, "qr": qr, "qi": qi, "qs": qs, "qg": qg,
               "ni": ni, "nr": nr, "t3d": t3d}[name]  # fmt: skip
        out[name] = bk.set(bk.copy(src), S, r[key])
    if getvt and vt is not None:
        nk = g.nk
        out["vt"] = tuple(
            bk.set(bk.copy(a), S, r[key][:nk])
            for a, key in zip(vt, ("vtck", "vtrk", "vtik", "vtsk", "vtgk"))
        )
    else:
        out["vt"] = vt
    return out


def mp_driver_ptype3(
    bk,
    g,
    cfg=None,
    *,
    dt,
    qbudget,
    asq,
    ruh,
    rvh,
    rmh,
    rain,
    prate,
    pi0,
    th0,
    rho,
    prs,
    pp3d,
    th3d,
    q3d,
    w3d,
    qten=None,
    thten=None,
    dum1=None,
    dum3=None,
    getvt=False,
    nstep_max=None,
    **unused,
):
    """mp_driver.F ``mp_driver`` for ptype=3 (Thompson).

    ``q3d``/``qten`` ``[8, k, j, i]`` (qv qc qr qi qs qg nci ncr), other
    shapes as ``mp_driver_kessler``; ``w3d`` ``g.w_shape``.  Sequence: T ->
    dum1 and thten, qv -> qten(nqv), rho -> dum3; pdefq(q1..q6);
    mp_gt_driver; then prs/pp3d/th3d (eqtset=2) or th3d/rho (eqtset=1)
    where T or qv changed.  ``getvt`` stores the fall speeds in
    qten(nqc..nqg).  The reflectivity diagnostic (getdbz) and radopt=2
    effective radii are not ported.  Returns dict with q3d, th3d, pp3d,
    prs, rho, qbudget, asq, rain, prate, qten, thten, dum1, dum3."""
    from .pdef import pdefq

    if int(_opt(cfg, "efall")) != 0 or int(_opt(cfg, "axisymm")) != 0:
        raise NotImplementedError("mp_driver ptype=3: efall/axisymm not ported")
    if int(_opt(cfg, "radopt")) == 2:
        raise NotImplementedError("mp_driver ptype=3: radopt=2 (calc_effectRad)")
    xp = bk.xp
    S = (g.K, g.J, g.I)
    qsmall = float(_opt(cfg, "qsmall"))
    tsmall = float(_opt(cfg, "tsmall"))
    eqtset = int(_opt(cfg, "eqtset"))
    if nstep_max is None and int(_opt(cfg, "nstep_max") or 0) > 0:
        nstep_max = int(_opt(cfg, "nstep_max"))

    def z(a, shape):
        return bk.zeros(shape) if a is None else bk.copy(a)

    nq = q3d.shape[0]
    qten = z(qten, (nq,) + tuple(g.s_shape))
    thten = z(thten, g.s_shape)
    dum1 = z(dum1, g.s_shape)
    dum3 = z(dum3, g.s_shape)
    t = (th0[S] + th3d[S]) * (pi0[S] + pp3d[S])
    dum1 = bk.set(dum1, S, t)
    thten = bk.set(thten, S, t)
    qten = bk.set(qten, (0,) + S, q3d[(0,) + S])
    dum3 = bk.set(dum3, S, rho[S])

    q = [q3d[n] for n in range(nq)]
    asq_l = [asq[n] for n in range(asq.shape[0])]
    for n in range(6):
        o = pdefq(bk, g, cfg, rmax=0.0 if n == 0 else qsmall, asq=asq_l[n],
                  ruh=ruh, rvh=rvh, rmh=rmh, rho=rho, q3d=q[n])  # fmt: skip
        q[n], asq_l[n] = o["q3d"], o["asq"]

    dum4 = g.dz * rmh
    qb = [qbudget[n] for n in range(qbudget.shape[0])]
    vt = tuple(qten[n] for n in (1, 2, 3, 4, 5))  # qc qr qi qs qg slots
    o = mp_gt_driver(
        bk, g, cfg, qv=q[0], qc=q[1], qr=q[2], qi=q[3], qs=q[4], qg=q[5],
        ni=q[6], nr=q[7], t3d=dum1, p=prs, w=w3d, dz=dum4, dt=dt, rain=rain,
        prate=prate, tcond=qb[0], tevac=qb[1], tevar=qb[4], train=qb[5],
        ruh=ruh, rvh=rvh, rmh=rmh, rr=dum3, getvt=getvt,
        vt=(vt[0], vt[1], vt[2], vt[3], vt[4]), nstep_max=nstep_max,
    )  # fmt: skip
    q = [o[k] for k in ("qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr")]
    dum1 = o["t3d"]
    rain, prate = o["rain"], o["prate"]
    qb[0], qb[1], qb[4], qb[5] = o["tcond"], o["tevac"], o["tevar"], o["train"]
    if getvt:
        # vtc, vtr, vti, vts, vtg -> qten(nqc, nqr, nqi, nqs, nqg)
        for slot, a in zip((1, 2, 3, 4, 5), o["vt"]):
            qten = bk.set(qten, slot, a)

    tn = dum1[S]
    qvn = q[0][S]
    chg = (xp.abs(tn - thten[S]) >= tsmall) | (xp.abs(qvn - qten[(0,) + S]) >= qsmall)
    p0, t0 = pi0[S], th0[S]
    if eqtset == 2:
        prs_n = rho[S] * (C.rd + C.rv * qvn) * tn
        pp_n = (prs_n * C.rp00) ** C.rovcp - p0
        th_n = tn / (p0 + pp_n) - t0
        prs = bk.set(bk.copy(prs), S, xp.where(chg, prs_n, prs[S]))
        pp3d = bk.set(bk.copy(pp3d), S, xp.where(chg, pp_n, pp3d[S]))
        th3d = bk.set(bk.copy(th3d), S, xp.where(chg, th_n, th3d[S]))
    else:
        th_n = tn / (p0 + pp3d[S]) - t0
        rho_n = prs[S] / (C.rd * tn * (1.0 + qvn * C.reps))
        th3d = bk.set(bk.copy(th3d), S, xp.where(chg, th_n, th3d[S]))
        rho = bk.set(bk.copy(rho), S, xp.where(chg, rho_n, rho[S]))

    dt_ = qbudget.dtype
    return {
        "q3d": xp.stack(q, axis=0),
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
