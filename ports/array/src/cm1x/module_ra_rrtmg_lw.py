"""RRTMG longwave radiation (src/module_ra_rrtmg_lw.F) as CM1 uses it
(radopt=2: inflglw=2, iceflglw=3, liqflglw=1, McICA maximum-random overlap
with the KISS generator, no aerosols).

Layout: the Fortran works one column at a time (``ncol = 1``); here every
routine works on all columns at once.  Layer arrays are ``[L, C]`` (layer
first, ``L = nlayers``, ``C`` columns), level arrays ``[L+1, C]`` (Fortran
0..nlayers or 1..nlayers+1), g-point arrays ``[L, C, G]`` (G = 140 g-points,
or the band's points inside ``taugbN``) and band arrays ``[..., 16]``.
Fortran ``x(ig, lay)`` (McICA layout) is ``x[lay, :, ig]`` here.

Tables (k-distributions from run/RRTMG_LW_DATA, the g-point reduction of
``rrtmg_lw_ini``/``cmbgbN``, cloud optics, reference atmosphere, Planck
tables) are built once with numpy (``rrtmg_lwinit``) and cached per backend.
Table look-ups are gathers (``xp.take``) with clamped indices; every branch
of an ``if`` on array values is evaluated and the result chosen with
``xp.where`` (both sides finite), so the routines trace under ``jax.jit``.
Sums over g-points and bands run in the Fortran order.
"""

from __future__ import annotations

import functools

import numpy as np

from . import _ftables as ft

SRC = "module_ra_rrtmg_lw.F"

nbndlw = 16
ngptlw = 140
mg = 16
deltap = 4.0  # buffer-layer pressure interval (mb)

# rrlw_kgNN contents of RRTMG_LW_DATA, in READ order: name:Fortran shape
_SCHEMA = {
    1: "fracrefao:16 fracrefbo:16 kao:5,13,16 kbo:5,47,16 kao_mn2:19,16 kbo_mn2:19,16 selfrefo:10,16 forrefo:4,16",
    2: "fracrefao:16 fracrefbo:16 kao:5,13,16 kbo:5,47,16 selfrefo:10,16 forrefo:4,16",
    3: "fracrefao:16,9 fracrefbo:16,5 kao:9,5,13,16 kbo:5,5,47,16 kao_mn2o:9,19,16 kbo_mn2o:5,19,16 selfrefo:10,16 forrefo:4,16",
    4: "fracrefao:16,9 fracrefbo:16,5 kao:9,5,13,16 kbo:5,5,47,16 selfrefo:10,16 forrefo:4,16",
    5: "fracrefao:16,9 fracrefbo:16,5 kao:9,5,13,16 kbo:5,5,47,16 kao_mo3:9,19,16 ccl4o:16 selfrefo:10,16 forrefo:4,16",
    6: "fracrefao:16 kao:5,13,16 kao_mco2:19,16 cfc11adjo:16 cfc12o:16 selfrefo:10,16 forrefo:4,16",
    7: "fracrefao:16,9 fracrefbo:16 kao:9,5,13,16 kbo:5,47,16 kao_mco2:9,19,16 kbo_mco2:19,16 selfrefo:10,16 forrefo:4,16",
    8: "fracrefao:16 fracrefbo:16 kao:5,13,16 kbo:5,47,16 kao_mco2:19,16 kbo_mco2:19,16 kao_mn2o:19,16 kbo_mn2o:19,16 kao_mo3:19,16 cfc12o:16 cfc22adjo:16 selfrefo:10,16 forrefo:4,16",
    9: "fracrefao:16,9 fracrefbo:16 kao:9,5,13,16 kbo:5,47,16 kao_mn2o:9,19,16 kbo_mn2o:19,16 selfrefo:10,16 forrefo:4,16",
    10: "fracrefao:16 fracrefbo:16 kao:5,13,16 kbo:5,47,16 selfrefo:10,16 forrefo:4,16",
    11: "fracrefao:16 fracrefbo:16 kao:5,13,16 kbo:5,47,16 kao_mo2:19,16 kbo_mo2:19,16 selfrefo:10,16 forrefo:4,16",
    12: "fracrefao:16,9 kao:9,5,13,16 selfrefo:10,16 forrefo:4,16",
    13: "fracrefao:16,9 fracrefbo:16 kao:9,5,13,16 kao_mco2:9,19,16 kao_mco:9,19,16 kbo_mo3:19,16 selfrefo:10,16 forrefo:4,16",
    14: "fracrefao:16 fracrefbo:16 kao:5,13,16 kbo:5,47,16 selfrefo:10,16 forrefo:4,16",
    15: "fracrefao:16,9 kao:9,5,13,16 kao_mn2:9,19,16 selfrefo:10,16 forrefo:4,16",
    16: "fracrefao:16,9 fracrefbo:16 kao:9,5,13,16 kbo:5,47,16 selfrefo:10,16 forrefo:4,16",
}


def parse_schema(spec):
    items = []
    for tok in spec.split():
        name, shp = tok.split(":")
        kind = "f4"
        if shp.endswith("i"):
            kind, shp = "i4", shp[:-1]
        shape = tuple(int(x) for x in shp.split(",")) if shp else ()
        items.append((name, kind, shape))
    return items


def reduced_name(name):
    """kao -> ka, kao_mn2 -> ka_mn2, fracrefao -> fracrefa, ccl4o -> ccl4."""
    i = name.find("o_")
    return name[:i] + name[i + 1 :] if i >= 0 else name[:-1]


def cmbgb(raw, items, ngc, ngn, rwgt, g_first_prefixes):
    """Generic ``cmbgbN`` of one band: combine the 16 original g-points of
    every table into ``ngc`` points (``ngn`` original points each), weighted
    by ``rwgt`` (the band's 16 weights) except the Planck / solar-source
    fractions (plain sums).  Sequential sums as in the Fortran.  Returns
    {reduced name: array with the g axis last}; non-g items pass through."""
    out = {}
    for name, _kind, shape in items:
        a = raw[name]
        if not shape or name in ("rayl", "strrat", "strrat1", "givfac", "scalekur"):
            out[name] = a
            continue
        gfirst = len(shape) > 1 and name.startswith(g_first_prefixes)
        a = np.moveaxis(a, 0, -1) if gfirst else a
        weighted = not name.startswith(("fracref", "sfluxref"))
        red = np.zeros(a.shape[:-1] + (ngc,))
        iprsm = 0
        for igc in range(ngc):
            s = np.zeros(a.shape[:-1])
            for _ipr in range(ngn[igc]):
                s = s + (a[..., iprsm] * rwgt[iprsm] if weighted else a[..., iprsm])
                iprsm += 1
            red[..., igc] = s
        out[reduced_name(name)] = red
    return out


@functools.cache
def _wvn():
    """lwdatinit / lwcmbdat constants (rrlw_wvn)."""
    d = {}
    d["delwave"] = np.array(
        [340.0, 150.0, 130.0, 70.0, 120.0, 160.0, 100.0, 100.0]
        + [210.0, 90.0, 320.0, 280.0, 170.0, 130.0, 220.0, 650.0]
    )
    d["nspa"] = [1, 1, 9, 9, 9, 1, 9, 1, 9, 1, 1, 9, 9, 1, 9, 9]
    d["nspb"] = [1, 1, 5, 5, 5, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0]
    d["ngc"] = [10, 12, 16, 14, 16, 8, 12, 8, 12, 6, 8, 8, 4, 2, 2, 2]
    d["ngs"] = list(np.cumsum(d["ngc"]))
    body = ft.subroutine_body(SRC, "lwcmbdat")
    txt = " ".join(body)

    def arr(name):
        i = txt.index(name + "(:) =")
        j = txt.index("/)", i)
        return ft._values(txt[txt.index("(/", i) + 2 : j])

    d["ngm"] = [int(x) for x in arr("ngm")]
    d["ngn"] = [int(x) for x in arr("ngn")]
    d["ngb"] = [int(x) for x in arr("ngb")]
    d["wt"] = np.array(arr("wt"))
    return d


@functools.cache
def rrtmg_lwinit(cpdair=1005.7, grav=9.81):
    """``rrtmg_lwinit`` -> ``rrtmg_lwlookuptable`` + ``rrtmg_lw_ini``:
    numpy tables (float64) of the reduced 140-g-point k-distribution and
    every constant table the LW routines use."""
    w = _wvn()
    recs = ft.read_unformatted(ft.cm1_root() / "run" / "RRTMG_LW_DATA")
    if len(recs) != 16:
        raise ValueError("RRTMG_LW_DATA: expected 16 records")
    # rrtmg_lw_ini: g-point weights
    ngc, ngn, ngm, wt = w["ngc"], w["ngn"], w["ngm"], w["wt"]
    rwgt = np.zeros(nbndlw * mg)
    igcsm = 0
    for ibnd in range(nbndlw):
        wtsm = np.zeros(mg)
        iprsm = 0
        if ngc[ibnd] < mg:
            for igc in range(ngc[ibnd]):
                wtsum = 0.0
                for _ipr in range(ngn[igcsm]):
                    wtsum = wtsum + wt[iprsm]
                    iprsm += 1
                wtsm[igc] = wtsum
                igcsm += 1
            for ig in range(mg):
                ind = ibnd * mg + ig
                rwgt[ind] = wt[ig] / wtsm[ngm[ind] - 1]
        else:
            for ig in range(mg):
                rwgt[ibnd * mg + ig] = 1.0
                igcsm += 1
    bands = {}
    off = 0
    for b in range(1, 17):
        items = parse_schema(_SCHEMA[b])
        raw = ft.unpack_record(recs[b - 1], items)
        red = cmbgb(
            raw, items, ngc[b - 1], ngn[off : off + ngc[b - 1]],
            rwgt[(b - 1) * mg : b * mg], ("fracref",),
        )  # fmt: skip
        off += ngc[b - 1]
        if "ka" in red:  # equivalence absa(65|585, ng) <-> ka(.., 13, ng)
            red["absa"] = red["ka"].reshape((-1, ngc[b - 1]), order="F")
        if "kb" in red:
            red["absb"] = red["kb"].reshape((-1, ngc[b - 1]), order="F")
        bands[b] = red
    T = {"bands": bands, "rwgt": rwgt}
    T.update({k: w[k] for k in ("delwave", "nspa", "nspb", "ngc", "ngs", "ngb")})
    # lwdatinit constants
    T["avogad"] = 6.02214199e23
    T["grav"] = grav
    T["secdy"] = 8.64e4
    T["oneminus"] = 1.0 - 1.0e-6
    T["fluxfac"] = 2.0 * np.arcsin(1.0) * 2.0e4
    T["heatfac"] = grav * T["secdy"] / (cpdair * 1.0e2)
    # rrlw_tbl
    ntbl, pade, expeps = 10000, 0.278, 1.0e-20
    bpade = 1.0 / pade
    tau = np.zeros(ntbl + 1)
    ex = np.zeros(ntbl + 1)
    tfn = np.zeros(ntbl + 1)
    tau[ntbl], ex[0], ex[ntbl], tfn[ntbl] = 1.0e10, 1.0, expeps, 1.0
    for itr in range(1, ntbl):
        f = float(itr) / float(ntbl)
        tau[itr] = bpade * f / (1.0 - f)
        ex[itr] = np.exp(-tau[itr])
        ex[itr] = max(expeps, ex[itr])
        if tau[itr] < 0.06:
            tfn[itr] = tau[itr] / 6.0
        else:
            tfn[itr] = 1.0 - 2.0 * ((1.0 / tau[itr]) - (ex[itr] / (1.0 - ex[itr])))
    T.update(tau_tbl=tau, exp_tbl=ex, tfn_tbl=tfn, bpade=bpade, tblint=10000.0)
    T.update(lwcldpr())
    T.update(lwatmref())
    T.update(lwavplank())
    return T


def lwcldpr():
    t = ft.assignments(
        SRC, "lwcldpr", {"absice3": ((1, 46), (1, 16)), "absliq1": ((1, 58), (1, 16))}
    )
    return t


def lwatmref():
    return ft.assignments(
        SRC,
        "lwatmref",
        {
            "pref": ((1, 59),),
            "preflog": ((1, 59),),
            "tref": ((1, 59),),
            "chi_mls": ((1, 7), (1, 59)),
        },
    )


def lwavplank():
    return ft.assignments(
        SRC, "lwavplank", {"totplnk": ((1, 181), (1, 16)), "totplk16": ((1, 181),)}
    )


# ---------------------------------------------------------------------------
# backend helpers
# ---------------------------------------------------------------------------
_BK_CACHE = {}


def tables(bk, cpdair=1005.7, grav=9.81):
    """rrtmg_lwinit tables as backend arrays (cached per backend/dtype)."""
    key = (bk.name, str(bk.dtype), str(bk.device), cpdair, grav)
    if key not in _BK_CACHE:
        T = rrtmg_lwinit(cpdair, grav)

        def conv(v):
            if isinstance(v, np.ndarray) and v.dtype.kind == "f":
                return bk.asarray(v)
            if isinstance(v, dict):
                return {k: conv(x) for k, x in v.items()}
            return v

        _BK_CACHE[key] = conv(T)
    return _BK_CACHE[key]


def ity(bk):
    xp = bk.xp
    if bk.name == "jax":
        import jax

        if not jax.config.jax_enable_x64:
            raise NotImplementedError("RRTMG needs 64-bit integers (jax x64)")
    return xp.int64


def fint(bk, x):
    """Fortran ``int()`` (truncation toward zero) of a real array."""
    xp = bk.xp
    return xp.astype(xp.trunc(x), ity(bk))


def ffloat(bk, i):
    return bk.xp.astype(i, bk.dtype)


def clip(bk, i, lo, hi):
    """min(max(i, lo), hi) with Python-number bounds."""
    return bk.xp.clip(i, lo, hi)


def take(bk, tab, idx):
    """Rows ``tab[idx]`` (0-based ``idx`` of any shape, clamped to the table)
    of a table with leading table axis: result ``idx.shape + tab.shape[1:]``."""
    xp = bk.xp
    n = tab.shape[0]
    i = clip(bk, idx, 0, n - 1)
    flat = xp.take(tab, xp.reshape(i, (-1,)), axis=0)
    return xp.reshape(flat, tuple(idx.shape) + tuple(tab.shape[1:]))


def take2(bk, tab3, i, j):
    """``tab3(i, j, :)`` of a 3-d Fortran table (1-based i, j)."""
    xp = bk.xp
    n1, n2 = tab3.shape[0], tab3.shape[1]
    flat = xp.reshape(xp.permute_dims(tab3, (1, 0, 2)), (n1 * n2, tab3.shape[2]))
    i = clip(bk, i, 1, n1)
    j = clip(bk, j, 1, n2)
    return take(bk, flat, (j - 1) * n1 + (i - 1))


def _e(x):
    return x[..., None]


# ---------------------------------------------------------------------------
# rrtmg_lw_rad
# ---------------------------------------------------------------------------
def inatm(
    bk,
    *,
    play,
    plev,
    tlay,
    tlev,
    tsfc,
    h2ovmr,
    o3vmr,
    co2vmr,
    ch4vmr,
    n2ovmr,
    o2vmr,
    cfc11vmr,
    cfc12vmr,
    cfc22vmr,
    ccl4vmr,
    emis,
    cldfmcl,
    taucmcl,
    ciwpmcl,
    clwpmcl,
    cswpmcl,
    reicmcl,
    relqmcl,
    resnmcl,
    T=None,
):
    """``inatm`` (iaer=10 with zero aerosol, icld>=1).  Inputs ``[L, C]``
    layers, ``[L+1, C]`` levels, ``tsfc[C]``, ``emis[C, 16]``, McICA arrays
    ``[L, C, 140]``.  Returns pavel, tavel, pz, tz, tbound, semiss, coldry,
    wkl (dict of the 7 molecules, index 1..7), wbrodl, wx (1..4), pwvcm and
    the McICA cloud arrays."""
    xp = bk.xp
    T = tables(bk) if T is None else T
    amd, amw = 28.966, 18.016
    avogad, grav = T["avogad"], T["grav"]
    pz, tz = plev, tlev
    w1 = h2ovmr
    amm = (1.0 - w1) * amd + w1 * amw
    coldry = (pz[:-1] - pz[1:]) * 1.0e3 * avogad / (1.0e2 * grav * amm * (1.0 + w1))
    vmr = {1: h2ovmr, 2: co2vmr, 3: o3vmr, 4: n2ovmr, 5: None, 6: ch4vmr, 7: o2vmr}
    summol = co2vmr + o3vmr + n2ovmr + 0.0 + ch4vmr + o2vmr
    wbrodl = coldry * (1.0 - summol)
    wkl = {
        m: (coldry * v if v is not None else xp.zeros_like(coldry))
        for m, v in vmr.items()
    }
    wx = {
        m: coldry * v * 1.0e-20
        for m, v in ((1, ccl4vmr), (2, cfc11vmr), (3, cfc12vmr), (4, cfc22vmr))
    }

    def acc(c, x):  # amttl = amttl + coldry(l)+wkl(1,l); wvttl = wvttl + wkl(1,l)
        return (c[0] + x[0] + x[1], c[1] + x[1]), None

    z = xp.zeros_like(coldry[0])
    (amttl, wvttl), _ = bk.scan(acc, (z, z), (coldry, wkl[1]))
    wvsh = (amw * wvttl) / (amd * amttl)
    pwvcm = wvsh * (1.0e3 * pz[0]) / (1.0e2 * grav)
    return {
        "pavel": play,
        "tavel": tlay,
        "pz": pz,
        "tz": tz,
        "tbound": tsfc,
        "semiss": emis,
        "coldry": coldry,
        "wkl": wkl,
        "wbrodl": wbrodl,
        "wx": wx,
        "pwvcm": pwvcm,
        "cldfmc": cldfmcl,
        "taucmc": taucmcl,
        "ciwpmc": ciwpmcl,
        "clwpmc": clwpmcl,
        "cswpmc": cswpmcl,
        "reicmc": reicmcl,
        "relqmc": relqmcl,
        "resnmc": resnmcl,
    }


# ---------------------------------------------------------------------------
# rrtmg_lw_cldprmc
# ---------------------------------------------------------------------------
def cldprmc(bk, *, cldfmc, ciwpmc, clwpmc, cswpmc, reicmc, relqmc, taucmc, T=None):
    """``cldprmc`` for inflag=2, iceflag=3, liqflag=1 (no snow: iceflag<5).
    ``[L, C, 140]`` arrays, effective sizes ``[L, C]``.  Returns taucmc.
    (The Fortran stops when a radius leaves the table range; here the table
    index is clamped instead.)"""
    xp = bk.xp
    T = tables(bk) if T is None else T
    ngb = xp.asarray(np.array(T["ngb"]) - 1)
    absice3 = xp.take(T["absice3"], ngb, axis=1)  # [46, 140]
    absliq1 = xp.take(T["absliq1"], ngb, axis=1)  # [58, 140]
    cldmin = 1.0e-20
    cwp = ciwpmc + clwpmc + cswpmc
    active = (cldfmc >= cldmin) & ((cwp >= cldmin) | (taucmc >= cldmin))
    g = xp.arange(ngptlw)
    # ice (iceflag 3), per layer -> same for every g-point
    radice = _e(reicmc)
    factor = (radice - 2.0) / 3.0
    index = fint(bk, factor)
    index = xp.where(index == 46, 45, index)
    fi = factor - ffloat(bk, index)
    index = clip(bk, index, 1, 45)
    flat = xp.reshape(absice3, (-1,))
    i0 = (index - 1) * ngptlw + g
    a0 = xp.take(flat, xp.reshape(i0, (-1,)))
    a1 = xp.take(flat, xp.reshape(i0 + ngptlw, (-1,)))
    sh = cldfmc.shape
    a0, a1 = xp.reshape(a0, sh), xp.reshape(a1, sh)
    abscoice = a0 + fi * (a1 - (a0))
    abscoice = xp.where((ciwpmc + cswpmc) == 0.0, 0.0, abscoice)
    abscosno = xp.zeros_like(abscoice)
    # liquid (liqflag 1)
    radliq = _e(relqmc)
    index = fint(bk, radliq - 1.5)
    index = xp.where(index == 0, 1, index)
    index = xp.where(index == 58, 57, index)
    fl = radliq - 1.5 - ffloat(bk, index)
    index = clip(bk, index, 1, 57)
    flat = xp.reshape(absliq1, (-1,))
    i0 = (index - 1) * ngptlw + g
    b0 = xp.reshape(xp.take(flat, xp.reshape(i0, (-1,))), sh)
    b1 = xp.reshape(xp.take(flat, xp.reshape(i0 + ngptlw, (-1,))), sh)
    abscoliq = b0 + fl * (b1 - (b0))
    abscoliq = xp.where(clwpmc == 0.0, 0.0, abscoliq)
    tau = ciwpmc * abscoice + clwpmc * abscoliq + cswpmc * abscosno
    return {"taucmc": xp.where(active, tau, taucmc)}


# ---------------------------------------------------------------------------
# rrtmg_lw_setcoef
# ---------------------------------------------------------------------------
def _tindex(bk, t):
    """``ind = t - 159`` (Fortran int assignment), clamped to 1..180, and
    the fraction ``t - 159 - float(ind)``."""
    ind = clip(bk, fint(bk, t - 159.0), 1, 180)
    return ind, t - 159.0 - ffloat(bk, ind)


def setcoef(bk, *, pavel, tavel, tz, tbound, semiss, coldry, wkl, wbroad, T=None):
    """``setcoef`` (istart=1).  Layer inputs ``[L, C]``, ``tz[L+1, C]``,
    ``tbound[C]``, ``semiss[C, 16]``, ``wkl`` {1..7: [L, C]}.  Returns laytrop
    ([C], count of layers with ln p > 4.56), jp, jt, jt1 (1-based integer
    arrays), planklay[L, C, 16], planklev[L+1, C, 16], plankbnd[C, 16], the
    column amounts, fac00..fac11, the rat_* ratios, selffac, selffrac,
    indself, forfac, forfrac, indfor, minorfrac, scaleminor, scaleminorn2,
    indminor.  Quantities the Fortran sets only in the lower (or upper)
    atmosphere are computed for every layer from that branch's formula;
    ``taumol`` uses them only where the Fortran does."""
    xp = bk.xp
    T = tables(bk) if T is None else T
    totplnk = T["totplnk"]  # [181, 16]
    preflog, tref, chi = T["preflog"], T["tref"], T["chi_mls"]  # chi [7, 59]
    stpfac = 296.0 / 1013.0

    def plank(ind, frac):
        b0 = take(bk, totplnk, ind - 1)
        b1 = take(bk, totplnk, ind)
        return b0 + _e(frac) * (b1 - b0)

    indbound, tbndfrac = _tindex(bk, tbound)
    indlev0, t0frac = _tindex(bk, tz[0])
    indlay, tlayfrac = _tindex(bk, tavel)
    indlev, tlevfrac = _tindex(bk, tz[1:])
    plankbnd = semiss * plank(indbound, tbndfrac)
    planklay = plank(indlay, tlayfrac)
    planklev = xp.concat(
        [plank(indlev0, t0frac)[None], plank(indlev, tlevfrac)], axis=0
    )

    plog = xp.log(pavel)
    jp = clip(bk, fint(bk, 36.0 - 5 * (plog + 0.04)), 1, 58)
    jp1 = jp + 1
    fp = 5.0 * (take(bk, preflog, jp - 1) - plog)
    trj = take(bk, tref, jp - 1)
    trj1 = take(bk, tref, jp1 - 1)
    jt = clip(bk, fint(bk, 3.0 + (tavel - trj) / 15.0), 1, 4)
    ft = ((tavel - trj) / 15.0) - ffloat(bk, jt - 3)
    jt1 = clip(bk, fint(bk, 3.0 + (tavel - trj1) / 15.0), 1, 4)
    ft1 = ((tavel - trj1) / 15.0) - ffloat(bk, jt1 - 3)
    water = wkl[1] / coldry
    scalefac = pavel * stpfac / tavel
    lower = plog > 4.56
    laytrop = xp.sum(xp.astype(lower, ity(bk)), axis=0)

    forfac = scalefac / (1.0 + water)
    factor = (332.0 - tavel) / 36.0
    indfor_lo = clip(bk, fint(bk, factor), 1, 2)
    forfrac_lo = factor - ffloat(bk, indfor_lo)
    factor_up = (tavel - 188.0) / 36.0
    indfor = xp.where(lower, indfor_lo, 3)
    forfrac = xp.where(lower, forfrac_lo, factor_up - 1.0)
    selffac = water * forfac
    factor = (tavel - 188.0) / 7.2
    indself = clip(bk, fint(bk, factor) - 7, 1, 9)
    selffrac = factor - ffloat(bk, indself + 7)
    scaleminor = pavel / tavel
    scaleminorn2 = (pavel / tavel) * (wbroad / (coldry + wkl[1]))
    factor = (tavel - 180.8) / 7.2
    indminor = clip(bk, fint(bk, factor), 1, 18)
    minorfrac = factor - ffloat(bk, indminor)

    def chim(m, j):
        return take(bk, chi[m - 1], j - 1)

    rat = {}
    for name, (a, b) in {
        "h2oco2": (1, 2),
        "h2oo3": (1, 3),
        "h2on2o": (1, 4),
        "h2och4": (1, 6),
        "n2oco2": (4, 2),
        "o3co2": (3, 2),
    }.items():
        rat["rat_" + name] = chim(a, jp) / chim(b, jp)
        rat["rat_" + name + "_1"] = chim(a, jp1) / chim(b, jp1)

    col = {}
    for m, name in ((1, "colh2o"), (2, "colco2"), (3, "colo3"), (4, "coln2o")):
        col[name] = 1.0e-20 * wkl[m]
    col["colco"] = 1.0e-20 * wkl[5]
    col["colch4"] = 1.0e-20 * wkl[6]
    col["colo2"] = 1.0e-20 * wkl[7]
    for name in ("colco2", "colo3", "coln2o", "colco", "colch4"):
        col[name] = xp.where(col[name] == 0.0, 1.0e-32 * coldry, col[name])
    col["colbrd"] = 1.0e-20 * wbroad

    compfp = 1.0 - fp
    out = {
        "laytrop": laytrop,
        "jp": jp,
        "jt": jt,
        "jt1": jt1,
        "planklay": planklay,
        "planklev": planklev,
        "plankbnd": plankbnd,
        "fac10": compfp * ft,
        "fac00": compfp * (1.0 - ft),
        "fac11": fp * ft1,
        "fac01": fp * (1.0 - ft1),
        "selffac": col["colh2o"] * selffac,
        "selffrac": selffrac,
        "indself": indself,
        "forfac": col["colh2o"] * forfac,
        "forfrac": forfrac,
        "indfor": indfor,
        "minorfrac": minorfrac,
        "scaleminor": scaleminor,
        "scaleminorn2": scaleminorn2,
        "indminor": indminor,
    }
    out.update(rat)
    out.update(col)
    return out


# ---------------------------------------------------------------------------
# rrtmg_lw_taumol
# ---------------------------------------------------------------------------
class _Tau:
    """Shared look-up arithmetic of the ``taugbN`` bands (one instance per
    ``taumol`` call): ``d`` = setcoef output plus wx, coldry, pavel."""

    def __init__(self, bk, T, d):
        self.bk, self.xp, self.T, self.d = bk, bk.xp, T, d
        L = d["jp"].shape[0]
        lay = self.xp.reshape(self.xp.arange(1, L + 1, dtype=ity(bk)), (L, 1))
        self.lower = lay <= d["laytrop"][None, :]
        self.oneminus = T["oneminus"]

    def chi(self, m, j):
        """chi_mls(m, j) for a Python or array index j (1-based)."""
        c = self.T["chi_mls"]
        if isinstance(j, int):
            return c[m - 1, j - 1]
        return take(self.bk, c[m - 1], j - 1)

    def pick(self, lo, up):
        """Layer-wise choice: lower-atmosphere value where lay <= laytrop."""
        m = self.lower
        return self.xp.where(m[..., None] if lo.ndim == 3 else m, lo, up)

    def interp(self, tab, i, f):
        """tab(i) + f*(tab(i+1) - tab(i)) along the leading axis (1-based i)."""
        a = take(self.bk, tab, i - 1)
        b = take(self.bk, tab, i)
        return a + _e(f) * (b - a)

    def interp2(self, tab3, j, f, indm, mf):
        """3-d minor table k(j, indm, ig): interpolate in j (fraction f),
        then in indm (fraction mf)."""
        bk = self.bk
        m1 = take2(bk, tab3, j, indm)
        m1 = m1 + _e(f) * (take2(bk, tab3, j + 1, indm) - m1)
        m2 = take2(bk, tab3, j, indm + 1)
        m2 = m2 + _e(f) * (take2(bk, tab3, j + 1, indm + 1) - m2)
        return m1 + _e(mf) * (m2 - m1)

    def tself(self, B):
        d = self.d
        return _e(d["selffac"]) * self.interp(B["selfref"], d["indself"], d["selffrac"])

    def tfor(self, B):
        d = self.d
        return _e(d["forfac"]) * self.interp(B["forref"], d["indfor"], d["forfrac"])

    def spec(self, cola, ratio, colb, mult):
        """speccomb, specparm (capped at oneminus), js = 1 + int(specmult),
        fs = mod(specmult, 1)."""
        bk, xp = self.bk, self.xp
        speccomb = cola + ratio * colb
        specparm = cola / speccomb
        specparm = xp.where(specparm >= self.oneminus, self.oneminus, specparm)
        specmult = mult * specparm
        ji = fint(bk, specmult)
        return speccomb, specparm, 1 + ji, specmult - ffloat(bk, ji)

    def ind_lo(self, nsp, js, js1):
        d = self.d
        ind0 = ((d["jp"] - 1) * 5 + (d["jt"] - 1)) * nsp + js
        ind1 = (d["jp"] * 5 + (d["jt1"] - 1)) * nsp + js1
        return ind0, ind1

    def ind_up(self, nsp, js, js1):
        d = self.d
        ind0 = ((d["jp"] - 13) * 5 + (d["jt"] - 1)) * nsp + js
        ind1 = ((d["jp"] - 12) * 5 + (d["jt1"] - 1)) * nsp + js1
        return ind0, ind1

    def simple4(self, tab, ind0, ind1):
        d, bk = self.d, self.bk
        return (
            _e(d["fac00"]) * take(bk, tab, ind0 - 1)
            + _e(d["fac10"]) * take(bk, tab, ind0)
            + _e(d["fac01"]) * take(bk, tab, ind1 - 1)
            + _e(d["fac11"]) * take(bk, tab, ind1)
        )

    def major9(self, tab, ind, specparm, fs, fa, fb, speccomb):
        """Lower-atmosphere major-species term with the 3-case specparm
        interpolation (nspa = 9): fa = fac00|fac01, fb = fac10|fac11."""
        bk, xp = self.bk, self.xp
        lt = specparm < 0.125
        gt = specparm > 0.875
        p = xp.where(lt, fs - 1, -fs)
        p2 = p * p
        p4 = p2 * p2
        fk0 = p4
        fk1 = 1 - p - 2.0 * p4
        fk2 = p + p4
        w0 = xp.where(lt, fk0, xp.where(gt, fk2, 1.0 - fs))
        w1 = xp.where(lt, fk1, xp.where(gt, fk1, fs))
        w2 = xp.where(lt, fk2, xp.where(gt, fk0, 0.0))
        base = xp.where(gt, ind - 1, ind) - 1  # 0-based
        t = _e(w0 * fa) * take(bk, tab, base)
        t = t + _e(w1 * fa) * take(bk, tab, base + 1)
        t = t + _e(w2 * fa) * take(bk, tab, base + 2)
        t = t + _e(w0 * fb) * take(bk, tab, base + 9)
        t = t + _e(w1 * fb) * take(bk, tab, base + 10)
        t = t + _e(w2 * fb) * take(bk, tab, base + 11)
        return _e(speccomb) * t

    def major_lo(self, B, nsp, sp0, sp1):
        """tau_major + tau_major1 of the lower atmosphere."""
        d = self.d
        speccomb, specparm, js, fs = sp0
        speccomb1, specparm1, js1, fs1 = sp1
        ind0, ind1 = self.ind_lo(nsp, js, js1)
        a = B["absa"]
        tm = self.major9(a, ind0, specparm, fs, d["fac00"], d["fac10"], speccomb)
        tm1 = self.major9(a, ind1, specparm1, fs1, d["fac01"], d["fac11"], speccomb1)
        return tm + tm1

    def major_up(self, B, nsp, sp0, sp1):
        """Upper-atmosphere two-species term (nspb = 5 offsets)."""
        d, bk = self.d, self.bk
        speccomb, _, js, fs = sp0
        speccomb1, _, js1, fs1 = sp1
        ind0, ind1 = self.ind_up(nsp, js, js1)
        b = B["absb"]
        f000, f010 = (1.0 - fs) * d["fac00"], (1.0 - fs) * d["fac10"]
        f100, f110 = fs * d["fac00"], fs * d["fac10"]
        f001, f011 = (1.0 - fs1) * d["fac01"], (1.0 - fs1) * d["fac11"]
        f101, f111 = fs1 * d["fac01"], fs1 * d["fac11"]
        t0 = (
            _e(f000) * take(bk, b, ind0 - 1)
            + _e(f100) * take(bk, b, ind0)
            + _e(f010) * take(bk, b, ind0 + 4)
            + _e(f110) * take(bk, b, ind0 + 5)
        )
        t1 = (
            _e(f001) * take(bk, b, ind1 - 1)
            + _e(f101) * take(bk, b, ind1)
            + _e(f011) * take(bk, b, ind1 + 4)
            + _e(f111) * take(bk, b, ind1 + 5)
        )
        return _e(speccomb) * t0 + _e(speccomb1) * t1

    def planck(self, frac, cola, ratio, colb, mult):
        """fracs = fracref(ig, jpl) + fpl*(fracref(ig, jpl+1) - fracref(ig, jpl))."""
        _, _, jpl, fpl = self.spec(cola, ratio, colb, mult)
        return self.interp(frac, jpl, fpl)

    def adjcol(self, col, m, thresh, a, pw, chiref=None):
        """Minor-gas column adjusted for a mixing ratio far above the MLS
        reference (``ratn2o``/``ratco2`` blocks)."""
        xp, d = self.xp, self.d
        ref = self.chi(m, d["jp"] + 1) if chiref is None else chiref
        ratio = 1.0e20 * (col / d["coldry"]) / ref
        big = ratio > thresh
        adjfac = a + xp.where(big, ratio - a, 1.0) ** pw
        return xp.where(big, adjfac * ref * d["coldry"] * 1.0e-20, col)

    def minor(self, tab, frac_name="minorfrac"):
        d = self.d
        return self.interp(tab, d["indminor"], d[frac_name])

    def bcast(self, v):
        """Band constant [ng] -> [L, C, ng]."""
        L, C = self.d["jp"].shape
        return self.xp.broadcast_to(v, (L, C, v.shape[-1]))


def _scale(bk, x, start, factors):
    """taug(lay, ngs+start+i) *= factors[i] (1-based start within the band)."""
    ng = x.shape[-1]
    f = np.ones(ng)
    f[start - 1 : start - 1 + len(factors)] = factors
    return x * bk.asarray(f)


def taugb1(bk, tm):
    B, d, xp = tm.T["bands"][1], tm.d, tm.xp
    pp = d["pavel"]
    ind0, ind1 = tm.ind_lo(1, 1, 1)
    corr = xp.where(pp < 250.0, 1.0 - 0.15 * (250.0 - pp) / 154.4, 1.0)
    scalen2 = _e(d["colbrd"] * d["scaleminorn2"])
    lo = _e(corr) * (
        _e(d["colh2o"]) * tm.simple4(B["absa"], ind0, ind1)
        + tm.tself(B)
        + tm.tfor(B)
        + scalen2 * tm.minor(B["ka_mn2"])
    )
    ind0, ind1 = tm.ind_up(1, 1, 1)
    corr = 1.0 - 0.15 * (pp / 95.6)
    up = _e(corr) * (
        _e(d["colh2o"]) * tm.simple4(B["absb"], ind0, ind1)
        + tm.tfor(B)
        + scalen2 * tm.minor(B["kb_mn2"])
    )
    return tm.pick(lo, up), tm.pick(tm.bcast(B["fracrefa"]), tm.bcast(B["fracrefb"]))


def taugb2(bk, tm):
    B, d = tm.T["bands"][2], tm.d
    pp = d["pavel"]
    ind0, ind1 = tm.ind_lo(1, 1, 1)
    corr = 1.0 - 0.05 * (pp - 100.0) / 900.0
    lo = _e(corr) * (
        _e(d["colh2o"]) * tm.simple4(B["absa"], ind0, ind1) + tm.tself(B) + tm.tfor(B)
    )
    ind0, ind1 = tm.ind_up(1, 1, 1)
    up = _e(d["colh2o"]) * tm.simple4(B["absb"], ind0, ind1) + tm.tfor(B)
    return tm.pick(lo, up), tm.pick(tm.bcast(B["fracrefa"]), tm.bcast(B["fracrefb"]))


def taugb3(bk, tm):
    B, d, c = tm.T["bands"][3], tm.d, tm.chi
    h2o, co2, n2o = d["colh2o"], d["colco2"], d["coln2o"]
    rp_a, rp_b = c(1, 9) / c(2, 9), c(1, 13) / c(2, 13)
    rm_a, rm_b = c(1, 3) / c(2, 3), c(1, 13) / c(2, 13)
    adjn2o = _e(tm.adjcol(n2o, 4, 1.5, 0.5, 0.65))
    # lower
    sp0 = tm.spec(h2o, d["rat_h2oco2"], co2, 8.0)
    sp1 = tm.spec(h2o, d["rat_h2oco2_1"], co2, 8.0)
    _, _, jm, fm = tm.spec(h2o, rm_a, co2, 8.0)
    absn2o = tm.interp2(B["ka_mn2o"], jm, fm, d["indminor"], d["minorfrac"])
    lo = tm.major_lo(B, 9, sp0, sp1) + tm.tself(B) + tm.tfor(B) + adjn2o * absn2o
    flo = tm.planck(B["fracrefa"], h2o, rp_a, co2, 8.0)
    # upper
    sp0 = tm.spec(h2o, d["rat_h2oco2"], co2, 4.0)
    sp1 = tm.spec(h2o, d["rat_h2oco2_1"], co2, 4.0)
    _, _, jm, fm = tm.spec(h2o, rm_b, co2, 4.0)
    absn2o = tm.interp2(B["kb_mn2o"], jm, fm, d["indminor"], d["minorfrac"])
    up = tm.major_up(B, 5, sp0, sp1) + tm.tfor(B) + adjn2o * absn2o
    fup = tm.planck(B["fracrefb"], h2o, rp_b, co2, 4.0)
    return tm.pick(lo, up), tm.pick(flo, fup)


def taugb4(bk, tm):
    B, d, c = tm.T["bands"][4], tm.d, tm.chi
    h2o, co2, o3 = d["colh2o"], d["colco2"], d["colo3"]
    rp_a, rp_b = c(1, 11) / c(2, 11), c(3, 13) / c(2, 13)
    sp0 = tm.spec(h2o, d["rat_h2oco2"], co2, 8.0)
    sp1 = tm.spec(h2o, d["rat_h2oco2_1"], co2, 8.0)
    lo = tm.major_lo(B, 9, sp0, sp1) + tm.tself(B) + tm.tfor(B)
    flo = tm.planck(B["fracrefa"], h2o, rp_a, co2, 8.0)
    sp0 = tm.spec(o3, d["rat_o3co2"], co2, 4.0)
    sp1 = tm.spec(o3, d["rat_o3co2_1"], co2, 4.0)
    up = tm.major_up(B, 5, sp0, sp1)
    up = _scale(bk, up, 8, [0.92, 0.88, 1.07, 1.1, 0.99, 0.88, 0.943])
    fup = tm.planck(B["fracrefb"], o3, rp_b, co2, 4.0)
    return tm.pick(lo, up), tm.pick(flo, fup)


def taugb5(bk, tm):
    B, d, c = tm.T["bands"][5], tm.d, tm.chi
    h2o, co2, o3 = d["colh2o"], d["colco2"], d["colo3"]
    rp_a, rp_b = c(1, 5) / c(2, 5), c(3, 43) / c(2, 43)
    rm_a = c(1, 7) / c(2, 7)
    ccl4 = _e(d["wx"][1]) * B["ccl4"]
    sp0 = tm.spec(h2o, d["rat_h2oco2"], co2, 8.0)
    sp1 = tm.spec(h2o, d["rat_h2oco2_1"], co2, 8.0)
    _, _, jm, fm = tm.spec(h2o, rm_a, co2, 8.0)
    abso3 = tm.interp2(B["ka_mo3"], jm, fm, d["indminor"], d["minorfrac"])
    lo = tm.major_lo(B, 9, sp0, sp1) + tm.tself(B) + tm.tfor(B) + abso3 * _e(o3) + ccl4
    flo = tm.planck(B["fracrefa"], h2o, rp_a, co2, 8.0)
    sp0 = tm.spec(o3, d["rat_o3co2"], co2, 4.0)
    sp1 = tm.spec(o3, d["rat_o3co2_1"], co2, 4.0)
    up = tm.major_up(B, 5, sp0, sp1) + ccl4
    fup = tm.planck(B["fracrefb"], o3, rp_b, co2, 4.0)
    return tm.pick(lo, up), tm.pick(flo, fup)


def taugb6(bk, tm):
    B, d = tm.T["bands"][6], tm.d
    adjco2 = _e(tm.adjcol(d["colco2"], 2, 3.0, 2.0, 0.77))
    cfc = _e(d["wx"][2]) * B["cfc11adj"] + _e(d["wx"][3]) * B["cfc12"]
    ind0, ind1 = tm.ind_lo(1, 1, 1)
    lo = (
        _e(d["colh2o"]) * tm.simple4(B["absa"], ind0, ind1)
        + tm.tself(B)
        + tm.tfor(B)
        + adjco2 * tm.minor(B["ka_mco2"])
        + _e(d["wx"][2]) * B["cfc11adj"]
        + _e(d["wx"][3]) * B["cfc12"]
    )
    up = 0.0 + cfc
    f = tm.bcast(B["fracrefa"])
    return tm.pick(lo, up), f


def taugb7(bk, tm):
    B, d, c = tm.T["bands"][7], tm.d, tm.chi
    h2o, co2, o3 = d["colh2o"], d["colco2"], d["colo3"]
    rp_a = c(1, 3) / c(3, 3)
    rm_a = c(1, 3) / c(3, 3)
    sp0 = tm.spec(h2o, d["rat_h2oo3"], o3, 8.0)
    sp1 = tm.spec(h2o, d["rat_h2oo3_1"], o3, 8.0)
    _, _, jm, fm = tm.spec(h2o, rm_a, o3, 8.0)
    adj = _e(tm.adjcol(co2, 2, 3.0, 3.0, 0.79))
    absco2 = tm.interp2(B["ka_mco2"], jm, fm, d["indminor"], d["minorfrac"])
    lo = tm.major_lo(B, 9, sp0, sp1) + tm.tself(B) + tm.tfor(B) + adj * absco2
    flo = tm.planck(B["fracrefa"], h2o, rp_a, o3, 8.0)
    adj = _e(tm.adjcol(co2, 2, 3.0, 2.0, 0.79))
    ind0, ind1 = tm.ind_up(1, 1, 1)
    up = _e(o3) * tm.simple4(B["absb"], ind0, ind1) + adj * tm.minor(B["kb_mco2"])
    up = _scale(bk, up, 6, [0.92, 0.88, 1.07, 1.1, 0.99, 0.855])
    return tm.pick(lo, up), tm.pick(flo, tm.bcast(B["fracrefb"]))


def taugb8(bk, tm):
    B, d = tm.T["bands"][8], tm.d
    adj = _e(tm.adjcol(d["colco2"], 2, 3.0, 2.0, 0.65))
    cfc = _e(d["wx"][3]) * B["cfc12"]
    cfc22 = _e(d["wx"][4]) * B["cfc22adj"]
    ind0, ind1 = tm.ind_lo(1, 1, 1)
    lo = (
        _e(d["colh2o"]) * tm.simple4(B["absa"], ind0, ind1)
        + tm.tself(B)
        + tm.tfor(B)
        + adj * tm.minor(B["ka_mco2"])
        + _e(d["colo3"]) * tm.minor(B["ka_mo3"])
        + _e(d["coln2o"]) * tm.minor(B["ka_mn2o"])
        + cfc
        + cfc22
    )
    ind0, ind1 = tm.ind_up(1, 1, 1)
    up = (
        _e(d["colo3"]) * tm.simple4(B["absb"], ind0, ind1)
        + adj * tm.minor(B["kb_mco2"])
        + _e(d["coln2o"]) * tm.minor(B["kb_mn2o"])
        + cfc
        + cfc22
    )
    return tm.pick(lo, up), tm.pick(tm.bcast(B["fracrefa"]), tm.bcast(B["fracrefb"]))


def taugb9(bk, tm):
    B, d, c = tm.T["bands"][9], tm.d, tm.chi
    h2o, ch4 = d["colh2o"], d["colch4"]
    rp_a, rm_a = c(1, 9) / c(6, 9), c(1, 3) / c(6, 3)
    adj = _e(tm.adjcol(d["coln2o"], 4, 1.5, 0.5, 0.65))
    sp0 = tm.spec(h2o, d["rat_h2och4"], ch4, 8.0)
    sp1 = tm.spec(h2o, d["rat_h2och4_1"], ch4, 8.0)
    _, _, jm, fm = tm.spec(h2o, rm_a, ch4, 8.0)
    absn2o = tm.interp2(B["ka_mn2o"], jm, fm, d["indminor"], d["minorfrac"])
    lo = tm.major_lo(B, 9, sp0, sp1) + tm.tself(B) + tm.tfor(B) + adj * absn2o
    flo = tm.planck(B["fracrefa"], h2o, rp_a, ch4, 8.0)
    ind0, ind1 = tm.ind_up(1, 1, 1)
    up = _e(ch4) * tm.simple4(B["absb"], ind0, ind1) + adj * tm.minor(B["kb_mn2o"])
    return tm.pick(lo, up), tm.pick(flo, tm.bcast(B["fracrefb"]))


def taugb10(bk, tm):
    B, d = tm.T["bands"][10], tm.d
    ind0, ind1 = tm.ind_lo(1, 1, 1)
    lo = _e(d["colh2o"]) * tm.simple4(B["absa"], ind0, ind1) + tm.tself(B) + tm.tfor(B)
    ind0, ind1 = tm.ind_up(1, 1, 1)
    up = _e(d["colh2o"]) * tm.simple4(B["absb"], ind0, ind1) + tm.tfor(B)
    return tm.pick(lo, up), tm.pick(tm.bcast(B["fracrefa"]), tm.bcast(B["fracrefb"]))


def taugb11(bk, tm):
    B, d = tm.T["bands"][11], tm.d
    scaleo2 = _e(d["colo2"] * d["scaleminor"])
    ind0, ind1 = tm.ind_lo(1, 1, 1)
    lo = (
        _e(d["colh2o"]) * tm.simple4(B["absa"], ind0, ind1)
        + tm.tself(B)
        + tm.tfor(B)
        + scaleo2 * tm.minor(B["ka_mo2"])
    )
    ind0, ind1 = tm.ind_up(1, 1, 1)
    up = (
        _e(d["colh2o"]) * tm.simple4(B["absb"], ind0, ind1)
        + tm.tfor(B)
        + scaleo2 * tm.minor(B["kb_mo2"])
    )
    return tm.pick(lo, up), tm.pick(tm.bcast(B["fracrefa"]), tm.bcast(B["fracrefb"]))


def taugb12(bk, tm):
    B, d, c = tm.T["bands"][12], tm.d, tm.chi
    h2o, co2 = d["colh2o"], d["colco2"]
    rp_a = c(1, 10) / c(2, 10)
    sp0 = tm.spec(h2o, d["rat_h2oco2"], co2, 8.0)
    sp1 = tm.spec(h2o, d["rat_h2oco2_1"], co2, 8.0)
    lo = tm.major_lo(B, 9, sp0, sp1) + tm.tself(B) + tm.tfor(B)
    flo = tm.planck(B["fracrefa"], h2o, rp_a, co2, 8.0)
    z = tm.xp.zeros_like(lo)
    return tm.pick(lo, z), tm.pick(flo, z)


def taugb13(bk, tm):
    B, d, c = tm.T["bands"][13], tm.d, tm.chi
    h2o, n2o = d["colh2o"], d["coln2o"]
    rp_a = c(1, 5) / c(4, 5)
    rm_a, rm_a3 = c(1, 1) / c(4, 1), c(1, 3) / c(4, 3)
    sp0 = tm.spec(h2o, d["rat_h2on2o"], n2o, 8.0)
    sp1 = tm.spec(h2o, d["rat_h2on2o_1"], n2o, 8.0)
    _, _, jm, fm = tm.spec(h2o, rm_a, n2o, 8.0)
    adj = _e(tm.adjcol(d["colco2"], 2, 3.0, 2.0, 0.68, chiref=3.55e-4))
    _, _, jc, fc = tm.spec(h2o, rm_a3, n2o, 8.0)
    absco2 = tm.interp2(B["ka_mco2"], jm, fm, d["indminor"], d["minorfrac"])
    absco = tm.interp2(B["ka_mco"], jc, fc, d["indminor"], d["minorfrac"])
    lo = (
        tm.major_lo(B, 9, sp0, sp1)
        + tm.tself(B)
        + tm.tfor(B)
        + adj * absco2
        + _e(d["colco"]) * absco
    )
    flo = tm.planck(B["fracrefa"], h2o, rp_a, n2o, 8.0)
    up = _e(d["colo3"]) * tm.minor(B["kb_mo3"])
    return tm.pick(lo, up), tm.pick(flo, tm.bcast(B["fracrefb"]))


def taugb14(bk, tm):
    B, d = tm.T["bands"][14], tm.d
    ind0, ind1 = tm.ind_lo(1, 1, 1)
    lo = _e(d["colco2"]) * tm.simple4(B["absa"], ind0, ind1) + tm.tself(B) + tm.tfor(B)
    ind0, ind1 = tm.ind_up(1, 1, 1)
    up = _e(d["colco2"]) * tm.simple4(B["absb"], ind0, ind1)
    return tm.pick(lo, up), tm.pick(tm.bcast(B["fracrefa"]), tm.bcast(B["fracrefb"]))


def taugb15(bk, tm):
    B, d, c = tm.T["bands"][15], tm.d, tm.chi
    n2o, co2 = d["coln2o"], d["colco2"]
    rp_a = c(4, 1) / c(2, 1)
    rm_a = c(4, 1) / c(2, 1)
    sp0 = tm.spec(n2o, d["rat_n2oco2"], co2, 8.0)
    sp1 = tm.spec(n2o, d["rat_n2oco2_1"], co2, 8.0)
    _, _, jm, fm = tm.spec(n2o, rm_a, co2, 8.0)
    scalen2 = _e(d["colbrd"] * d["scaleminor"])
    taun2 = scalen2 * tm.interp2(B["ka_mn2"], jm, fm, d["indminor"], d["minorfrac"])
    lo = tm.major_lo(B, 9, sp0, sp1) + tm.tself(B) + tm.tfor(B) + taun2
    flo = tm.planck(B["fracrefa"], n2o, rp_a, co2, 8.0)
    z = tm.xp.zeros_like(lo)
    return tm.pick(lo, z), tm.pick(flo, z)


def taugb16(bk, tm):
    B, d, c = tm.T["bands"][16], tm.d, tm.chi
    h2o, ch4 = d["colh2o"], d["colch4"]
    rp_a = c(1, 6) / c(6, 6)
    sp0 = tm.spec(h2o, d["rat_h2och4"], ch4, 8.0)
    sp1 = tm.spec(h2o, d["rat_h2och4_1"], ch4, 8.0)
    lo = tm.major_lo(B, 9, sp0, sp1) + tm.tself(B) + tm.tfor(B)
    flo = tm.planck(B["fracrefa"], h2o, rp_a, ch4, 8.0)
    # nspb(16) = 0: the Fortran's upper-atmosphere index is always 1
    ind0, ind1 = tm.ind_up(0, 1, 1)
    up = _e(ch4) * tm.simple4(B["absb"], ind0, ind1)
    return tm.pick(lo, up), tm.pick(flo, tm.bcast(B["fracrefb"]))


_TAUGB = (
    taugb1, taugb2, taugb3, taugb4, taugb5, taugb6, taugb7, taugb8,
    taugb9, taugb10, taugb11, taugb12, taugb13, taugb14, taugb15, taugb16,
)  # fmt: skip


def taumol(bk, *, pavel, wx, coldry, coef, T=None):
    """``taumol``: gaseous optical depths ``taug`` and Planck fractions
    ``fracs`` ``[L, C, 140]`` from the setcoef output ``coef``."""
    xp = bk.xp
    T = tables(bk) if T is None else T
    d = dict(coef)
    d.update(pavel=pavel, wx=wx, coldry=coldry)
    tm = _Tau(bk, T, d)
    taug, fracs = [], []
    for f in _TAUGB:
        t, fr = f(bk, tm)
        taug.append(t)
        fracs.append(fr)
    return {"taug": xp.concat(taug, axis=-1), "fracs": xp.concat(fracs, axis=-1)}


# ---------------------------------------------------------------------------
# rrtmg_lw_rtrnmc
# ---------------------------------------------------------------------------
_A0 = [1.66, 1.55, 1.58, 1.66, 1.54, 1.454, 1.89, 1.33]
_A0 += [1.668, 1.66, 1.66, 1.66, 1.66, 1.66, 1.66, 1.66]
_A1 = [0.00, 0.25, 0.22, 0.00, 0.13, 0.446, -0.10, 0.40]
_A1 += [-0.006, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
_A2 = [0.00, -12.0, -11.7, 0.00, -0.72, -0.243, 0.19, -0.062]
_A2 += [0.414, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]


def rtrnmc(
    bk,
    *,
    pz,
    semiss,
    cldfmc,
    taucmc,
    planklay,
    planklev,
    plankbnd,
    pwvcm,
    fracs,
    taut,
    T=None,
):
    """``rtrnmc`` (istart=1, iend=16, iout=0): McICA longwave transfer.
    Returns totuflux, totdflux, fnet, totuclfl, totdclfl, fnetc ``[L+1, C]``
    (levels 0..nlayers) and htr, htrc ``[L+1, C]`` (htr[nlayers] = 0)."""
    xp = bk.xp
    T = tables(bk) if T is None else T
    rec_6, wtdiff = 0.166667, 0.5
    ngb = xp.asarray(np.array(T["ngb"]) - 1)
    tblint, bpade = T["tblint"], T["bpade"]
    tau_tbl, exp_tbl, tfn_tbl = T["tau_tbl"], T["exp_tbl"], T["tfn_tbl"]

    # diffusivity angle per band
    a0, a1, a2 = (bk.asarray(np.array(a)) for a in (_A0, _A1, _A2))
    sd = a0 + a1 * xp.exp(a2 * _e(pwvcm))
    sd = xp.where(sd > 1.80, 1.80, sd)
    sd = xp.where(sd < 1.50, 1.50, sd)
    fixed = np.zeros(nbndlw, dtype=bool)
    fixed[[0, 3]] = True
    fixed[9:] = True
    secdiff = xp.where(xp.asarray(fixed), 1.66, sd)  # [C, 16]
    secg = xp.take(secdiff, ngb, axis=-1)  # [C, 140]

    cloudy = cldfmc == 1.0
    odcld = xp.where(cloudy, secg[None] * taucmc, 0.0)
    abscld = xp.where(cloudy, 1.0 - xp.exp(-odcld), 0.0)
    efclfrac = xp.where(cloudy, abscld * cldfmc, 0.0)
    icldlyr = xp.any(cloudy, axis=-1)  # [L, C]

    def lookup(tab, od):
        it = fint(bk, tblint * (od / (bpade + od)) + 0.5)
        return take(bk, tab, it)

    plfrac = fracs
    blay = xp.take(planklay, ngb, axis=-1)
    plev = xp.take(planklev, ngb, axis=-1)
    dplankup = plev[1:] - blay
    dplankdn = plev[:-1] - blay
    odepth = secg[None] * taut
    odepth = xp.where(odepth < 0.0, 0.0, odepth)

    # clear layer
    small = odepth <= 0.06
    at_s = odepth - 0.5 * odepth * odepth
    od_r = rec_6 * odepth
    tausfac = lookup(tfn_tbl, odepth)
    at_clr = xp.where(small, at_s, 1.0 - lookup(exp_tbl, odepth))
    bbd_clr = xp.where(
        small, plfrac * (blay + dplankdn * od_r), plfrac * (blay + tausfac * dplankdn)
    )
    bbu_clr = xp.where(
        small, plfrac * (blay + dplankup * od_r), plfrac * (blay + tausfac * dplankup)
    )
    # cloudy layer (any g-point of the layer cloudy): three cases
    odtot = odepth + odcld
    c1 = odtot < 0.06
    c2 = (~c1) & small
    gassrc12 = plfrac * (blay + dplankdn * od_r) * at_s
    odtot_r = rec_6 * odtot
    tfactot = lookup(tfn_tbl, odtot)
    atot12 = xp.where(c1, odtot - 0.5 * odtot * odtot, 1.0 - lookup(exp_tbl, odtot))
    bbdtot12 = xp.where(
        c1, plfrac * (blay + dplankdn * odtot_r), plfrac * (blay + tfactot * dplankdn)
    )
    bbutot12 = xp.where(
        c1, plfrac * (blay + dplankup * odtot_r), plfrac * (blay + tfactot * dplankup)
    )
    bbd12 = plfrac * (blay + dplankdn * od_r)
    bbugas12 = plfrac * (blay + dplankup * od_r)
    # case 3: odepth > 0.06
    tfacgas = tausfac
    at3 = 1.0 - lookup(exp_tbl, odepth)
    od3 = lookup(tau_tbl, odepth)
    gassrc3 = at3 * plfrac * (blay + tfacgas * dplankdn)
    odtot3 = od3 + odcld
    tfactot3 = lookup(tfn_tbl, odtot3)
    bbdtot3 = plfrac * (blay + tfactot3 * dplankdn)
    bbd3 = plfrac * (blay + tfacgas * dplankdn)
    atot3 = 1.0 - lookup(exp_tbl, odtot3)
    bbugas3 = plfrac * (blay + tfacgas * dplankup)
    bbutot3 = plfrac * (blay + tfactot3 * dplankup)
    c12 = c1 | c2
    at_cld = xp.where(c12, at_s, at3)
    gassrc = xp.where(c12, gassrc12, gassrc3)
    atot = xp.where(c12, atot12, atot3)
    bbdtot = xp.where(c12, bbdtot12, bbdtot3)
    bbutot = xp.where(c12, bbutot12, bbutot3)
    bbd_cld = xp.where(c12, bbd12, bbd3)
    bbu_cld = xp.where(c12, bbugas12, bbugas3)

    icl = icldlyr[..., None]
    atrans = xp.where(icl, at_cld, at_clr)
    bbd = xp.where(icl, bbd_cld, bbd_clr)
    bbugas = xp.where(icl, bbu_cld, bbu_clr)

    def down(c, x):
        radld, radclrd, iclddn = c
        icl_l, at, efc, gs, cf, bt, ato, bd = x
        cld = radld - radld * (at + efc * (1.0 - at)) + gs + cf * (bt * ato - gs)
        clr = radld + (bd - radld) * at
        radld = xp.where(icl_l[..., None], cld, clr)
        iclddn = iclddn | icl_l
        radclrd = xp.where(iclddn[..., None], radclrd + (bd - radclrd) * at, radld)
        return (radld, radclrd, iclddn), (radld, radclrd)

    C = pz.shape[1]
    z = xp.zeros_like(taut[0])
    f0 = xp.zeros((C,), dtype=xp.bool)
    (radld_s, radclrd_s, iclddn), (drad, clrdrad) = bk.scan(
        down,
        (z, z, f0),
        (icldlyr, atrans, efclfrac, gassrc, cldfmc, bbdtot, atot, bbd),
        reverse=True,
    )
    plb = xp.take(plankbnd, ngb, axis=-1)
    rad0 = fracs[0] * plb
    reflect = 1.0 - xp.take(semiss, ngb, axis=-1)
    radlu0 = rad0 + reflect * radld_s
    radclru0 = rad0 + reflect * radclrd_s
    icldn = iclddn[..., None]

    def up(c, x):
        radlu, radclru = c
        icl_l, at, efc, bu, cf, bt, ato = x
        gs = bu * at
        cld = radlu - radlu * (at + efc * (1.0 - at)) + gs + cf * (bt * ato - gs)
        clr = radlu + (bu - radlu) * at
        radlu = xp.where(icl_l[..., None], cld, clr)
        radclru = xp.where(icldn, radclru + (bu - radclru) * at, radlu)
        return (radlu, radclru), (radlu, radclru)

    _, (urad, clrurad) = bk.scan(
        up,
        (radlu0, radclru0),
        (icldlyr, atrans, efclfrac, bbugas, cldfmc, bbutot, atot),
    )
    zl = xp.zeros_like(drad[:1])
    urad = xp.concat([radlu0[None], urad], axis=0)
    clrurad = xp.concat([radclru0[None], clrurad], axis=0)
    drad = xp.concat([drad, zl], axis=0)
    clrdrad = xp.concat([clrdrad, zl], axis=0)

    out = {}
    for name, rad in (
        ("totuflux", urad),
        ("totdflux", drad),
        ("totuclfl", clrurad),
        ("totdclfl", clrdrad),
    ):
        tot = None
        g0 = 0
        for ib in range(nbndlw):
            ng = T["ngc"][ib]
            s = rad[..., g0]
            for g in range(g0 + 1, g0 + ng):
                s = s + rad[..., g]
            g0 += ng
            term = (s * wtdiff) * float(_wvn()["delwave"][ib])
            tot = term if tot is None else tot + term
        out[name] = tot * T["fluxfac"]
    heatfac = T["heatfac"]
    zt = xp.zeros_like(pz[:1])
    for f, u, dn, h in (
        ("fnet", "totuflux", "totdflux", "htr"),
        ("fnetc", "totuclfl", "totdclfl", "htrc"),
    ):
        fn = out[u] - out[dn]
        out[f] = fn
        hr = heatfac * (fn[:-1] - fn[1:]) / (pz[:-1] - pz[1:])
        out[h] = xp.concat([hr, zt], axis=0)
    return out


# ---------------------------------------------------------------------------
# mcica_subcol_gen_lw
# ---------------------------------------------------------------------------
_M32 = 0xFFFFFFFF


def _wrap32(bk, x):
    """Two's-complement int32 value of the low 32 bits of int64 ``x``."""
    return ((x + 2**31) & _M32) - 2**31


def _lshift(bk, x, n):
    """Fortran ishft(k, n) on int32 bit patterns (logical shift), result
    as a signed int32 value held in int64."""
    u = x & _M32
    if n >= 0:
        return _wrap32(bk, (u << n) & _M32)
    return _wrap32(bk, u >> (-n))


def kissvec(bk, seed1, seed2, seed3, seed4):
    """``kissvec``: one KISS draw per column (int32 arithmetic with
    wrap-around, emulated in int64).  Returns the new seeds and ran_arr."""

    def m(k, n):
        return _wrap32(bk, k ^ _lshift(bk, k, n))

    seed1 = _wrap32(bk, 69069 * seed1 + 1327217885)
    seed2 = m(m(m(seed2, 13), -17), 5)
    seed3 = _wrap32(bk, 18000 * (seed3 & 65535) + _lshift(bk, seed3, -16))
    seed4 = _wrap32(bk, 30903 * (seed4 & 65535) + _lshift(bk, seed4, -16))
    kiss = _wrap32(bk, seed1 + seed2 + _lshift(bk, seed3, 16) + seed4)
    ran = ffloat(bk, kiss) * 2.328306e-10 + 0.5
    return seed1, seed2, seed3, seed4, ran


def generate_stochastic_clouds(
    bk, *, pmid, cld, clwp, ciwp, cswp, tauc, changeseed, nsubcol, ngb
):
    """``generate_stochastic_clouds`` for icld=2 (maximum-random overlap),
    irng=0 (KISS seeded from the bottom four layer pressures).  ``pmid``,
    ``cld``, water paths ``[L, C]``, ``tauc[L, C, nbnd]``; returns
    cld_stoch, clwp_stoch, ciwp_stoch, cswp_stoch, tauc_stoch ``[L, C, nsub]``."""
    xp = bk.xp
    it = ity(bk)
    L, C = cld.shape
    cldf = xp.where(cld < 1.0e-20, 0.0, cld)

    def seed(p):
        return xp.astype(xp.trunc((p - ffloat(bk, fint(bk, p))) * 1000000000), it)

    seeds = tuple(seed(pmid[n]) for n in range(4))

    def draw(c, _):
        *s, r = kissvec(bk, *c)
        return tuple(s), r

    seeds, _ = bk.scan(draw, seeds, xp.zeros((changeseed,)))
    # CDF(isubcol, :, ilev): draws in subcolumn-major, level-minor order
    seeds, ran = bk.scan(draw, seeds, xp.zeros((nsubcol * L,)))
    ran = xp.reshape(ran, (nsubcol, L, C))
    cdf0 = xp.permute_dims(ran, (1, 2, 0))  # [L, C, nsub]

    def overlap(prev, x):
        cdf_l, cf_prev = x
        thr = 1.0 - cf_prev[..., None]
        cdf = xp.where(prev > thr, prev, cdf_l * thr)
        return cdf, cdf

    first = cdf0[0]
    _, rest = bk.scan(overlap, first, (cdf0[1:], cldf[:-1]))
    cdf = xp.concat([first[None], rest], axis=0)
    cloudy = cdf >= 1.0 - cldf[..., None]
    zero = xp.zeros_like(cdf)
    taug = xp.take(tauc, xp.asarray(np.array(ngb) - 1), axis=-1)
    return {
        "cld_stoch": xp.where(cloudy, 1.0, 0.0),
        "clwp_stoch": xp.where(cloudy, clwp[..., None], zero),
        "ciwp_stoch": xp.where(cloudy, ciwp[..., None], zero),
        "cswp_stoch": xp.where(cloudy, cswp[..., None], zero),
        "tauc_stoch": xp.where(cloudy, taug, zero),
    }


def mcica_subcol_lw(
    bk, *, permuteseed, play, cldfrac, ciwp, clwp, cswp, rei, rel, res, tauc, T=None
):
    """``mcica_subcol_lw`` (icld=2, irng=0).  ``[L, C]`` inputs (play in mb),
    ``tauc[L, C, 16]``.  Returns cldfmcl, ciwpmcl, clwpmcl, cswpmcl, taucmcl
    ``[L, C, 140]`` and reicmcl, relqmcl, resnmcl ``[L, C]``."""
    T = tables(bk) if T is None else T
    pmid = play * 1.0e2
    r = generate_stochastic_clouds(
        bk,
        pmid=pmid,
        cld=cldfrac,
        clwp=clwp,
        ciwp=ciwp,
        cswp=cswp,
        tauc=tauc,
        changeseed=permuteseed,
        nsubcol=ngptlw,
        ngb=T["ngb"],
    )
    return {
        "cldfmcl": r["cld_stoch"],
        "clwpmcl": r["clwp_stoch"],
        "ciwpmcl": r["ciwp_stoch"],
        "cswpmcl": r["cswp_stoch"],
        "taucmcl": r["tauc_stoch"],
        "reicmcl": rei,
        "relqmcl": rel,
        "resnmcl": res,
    }


# ---------------------------------------------------------------------------
# rrtmg_lw_rad: rrtmg_lw
# ---------------------------------------------------------------------------
def rrtmg_lw(
    bk,
    *,
    play,
    plev,
    tlay,
    tlev,
    tsfc,
    h2ovmr,
    o3vmr,
    co2vmr,
    ch4vmr,
    n2ovmr,
    o2vmr,
    cfc11vmr,
    cfc12vmr,
    cfc22vmr,
    ccl4vmr,
    emis,
    cldfmcl,
    taucmcl,
    ciwpmcl,
    clwpmcl,
    cswpmcl,
    reicmcl,
    relqmcl,
    resnmcl,
    T=None,
    trace=None,
):
    """``rrtmg_lw`` (iaer=10 with zero aerosol optical depth): inatm ->
    cldprmc -> setcoef -> taumol -> rtrnmc for all columns.  Returns uflx,
    dflx, uflxc, dflxc ``[L+1, C]`` (levels 1..nlayers+1) and hr, hrc
    ``[L, C]``.  ``trace`` (a dict) collects the intermediate results."""
    T = tables(bk) if T is None else T
    a = inatm(
        bk,
        play=play,
        plev=plev,
        tlay=tlay,
        tlev=tlev,
        tsfc=tsfc,
        h2ovmr=h2ovmr,
        o3vmr=o3vmr,
        co2vmr=co2vmr,
        ch4vmr=ch4vmr,
        n2ovmr=n2ovmr,
        o2vmr=o2vmr,
        cfc11vmr=cfc11vmr,
        cfc12vmr=cfc12vmr,
        cfc22vmr=cfc22vmr,
        ccl4vmr=ccl4vmr,
        emis=emis,
        cldfmcl=cldfmcl,
        taucmcl=taucmcl,
        ciwpmcl=ciwpmcl,
        clwpmcl=clwpmcl,
        cswpmcl=cswpmcl,
        reicmcl=reicmcl,
        relqmcl=relqmcl,
        resnmcl=resnmcl,
        T=T,
    )
    taucmc = cldprmc(
        bk,
        cldfmc=a["cldfmc"],
        ciwpmc=a["ciwpmc"],
        clwpmc=a["clwpmc"],
        cswpmc=a["cswpmc"],
        reicmc=a["reicmc"],
        relqmc=a["relqmc"],
        taucmc=a["taucmc"],
        T=T,
    )["taucmc"]
    coef = setcoef(
        bk,
        pavel=a["pavel"],
        tavel=a["tavel"],
        tz=a["tz"],
        tbound=a["tbound"],
        semiss=a["semiss"],
        coldry=a["coldry"],
        wkl=a["wkl"],
        wbroad=a["wbrodl"],
        T=T,
    )
    tau = taumol(bk, pavel=a["pavel"], wx=a["wx"], coldry=a["coldry"], coef=coef, T=T)
    r = rtrnmc(
        bk,
        pz=a["pz"],
        semiss=a["semiss"],
        cldfmc=a["cldfmc"],
        taucmc=taucmc,
        planklay=coef["planklay"],
        planklev=coef["planklev"],
        plankbnd=coef["plankbnd"],
        pwvcm=a["pwvcm"],
        fracs=tau["fracs"],
        taut=tau["taug"],
        T=T,
    )
    if trace is not None:
        trace.update(inatm=a, taucmc=taucmc, setcoef=coef, taumol=tau, rtrnmc=r)
    return {
        "uflx": r["totuflux"],
        "dflx": r["totdflux"],
        "uflxc": r["totuclfl"],
        "dflxc": r["totdclfl"],
        "hr": r["htr"][:-1],
        "hrc": r["htrc"][:-1],
    }


# ---------------------------------------------------------------------------
# module_ra_rrtmg_lw: RRTMG_LWRAD and helpers
# ---------------------------------------------------------------------------
@functools.cache
def _profiles():
    """retab (module data), PPROF/TPROF (RRTMG_LWRAD) and the O3DATA
    annual-mean ozone profile on its half levels (ppwrkh, o3wrk)."""
    retab = ft.data_statement(SRC, "retab")
    pprof = ft.data_statement(SRC, "PPROF", "RRTMG_LWRAD")
    tprof = ft.data_statement(SRC, "TPROF", "RRTMG_LWRAD")
    o3sum = ft.data_statement(SRC, "O3SUM", "O3DATA")
    ppsum = ft.data_statement(SRC, "PPSUM", "O3DATA")
    o3win = ft.data_statement(SRC, "O3WIN", "O3DATA")
    ppwin = ft.data_statement(SRC, "PPWIN", "O3DATA")
    o3ann = np.zeros(31)
    o3ann[0] = 0.5 * (o3sum[0] + o3win[0])
    for k in range(1, 31):
        o3ann[k] = o3win[k - 1] + (o3win[k] - o3win[k - 1]) / (
            ppwin[k] - ppwin[k - 1]
        ) * (ppsum[k] - ppwin[k - 1])
    for k in range(1, 31):
        o3ann[k] = 0.5 * (o3ann[k] + o3sum[k])
    ppwrkh = np.zeros(32)
    ppwrkh[0] = 1100.0
    for k in range(1, 31):
        ppwrkh[k] = (ppsum[k] + ppsum[k - 1]) / 2.0
    ppwrkh[31] = 0.0
    return {
        "retab": retab,
        "pprof": pprof,
        "tprof": tprof,
        "o3wrk": o3ann,
        "ppwrkh": ppwrkh,
    }


def o3data(bk, *, plev):
    """``O3DATA`` (usecode=2): ozone mass mixing ratio at each ``plev`` (mb),
    linear in pressure on the annual-mean profile."""
    xp = bk.xp
    P = _profiles()
    ph = bk.asarray(P["ppwrkh"])
    o3 = bk.asarray(P["o3wrk"])
    # interior: kk = first index with ppwrkh(kk) <= plev (searched from 1)
    kk = 1 + xp.sum(xp.astype(ph[None, None, :31] > plev[..., None], ity(bk)), axis=-1)
    kk = clip(bk, kk, 2, 31)
    ph0, ph1 = take(bk, ph, kk - 2), take(bk, ph, kk - 1)
    o0, o1 = take(bk, o3, kk - 2), take(bk, o3, kk - 1)
    frac = (plev - ph0) / (ph1 - ph0)
    mid = o0 + (o1 - o0) * frac
    frac = (plev - ph[0]) / (ph[1] - ph[0])
    bot = o3[0] + (o3[1] - o3[0]) * frac
    frac = (plev - ph[30]) / (ph[30] - ph[29])
    top = o3[30] + (o3[30] - o3[29]) * frac
    return xp.where(plev > ph[0], bot, xp.where(plev < ph[30], top, mid))


def inirad(bk, *, plev):
    """``INIRAD``: o3prof(k) for every level k of ``plev``."""
    return o3data(bk, plev=plev)


def relcalc(bk, *, t, landfrac, landm, icefrac, snowh):
    """``relcalc``: liquid effective radius (microns)."""
    xp = bk.xp
    tmelt, rliqocean, rliqice, rliqland = 273.16, 14.0, 14.0, 8.0

    def c01(x):
        return xp.clip(x, 0.0, 1.0)

    rel = rliqland + (rliqocean - rliqland) * c01((tmelt - t) * 0.05)
    rel = rel + (rliqocean - rel) * c01(snowh * 10.0)
    rel = rel + (rliqocean - rel) * c01(1.0 - landm)
    rel = rel + (rliqice - rel) * c01(icefrac)
    return rel


def reicalc(bk, *, t):
    """``reicalc``: ice effective radius (microns) from the retab table."""
    retab = bk.asarray(_profiles()["retab"])
    index = clip(bk, fint(bk, t - 179.0), 1, 94)
    corr = t - ffloat(bk, fint(bk, t))
    return take(bk, retab, index - 1) * (1.0 - corr) + take(bk, retab, index) * corr


def nint(x):
    """Fortran nint of a Python float."""
    import math

    return math.floor(abs(x) + 0.5) * (1 if x >= 0 else -1)


def nlayers_lw(nk, p_top):
    """``rrtmg_lwinit``: nlayers = kme + nint(p_top*0.01/deltap) - 1 with
    kme = nk + 1 (model layers plus buffer layers up to 0 mb)."""
    return (nk + 1) + nint(p_top * 0.01 / deltap) - 1


def RRTMG_LWRAD(
    bk,
    *,
    nlayers,
    emiss,
    p8w,
    p3d,
    tsk,
    t3d,
    t8w,
    cldfra3d,
    xland,
    xice,
    snow,
    qv3d,
    qc3d,
    qr3d,
    yr,
    g=9.81,
    T=None,
    trace=None,
):
    """``RRTMG_LWRAD`` as CM1's radopt=2 calls it for warm-rain (ptype=1)
    microphysics: f_qv, f_qc, f_qr only (no ice species; the "MP option 3"
    rule moves cloud/rain below 273.15 K to ice/snow), icloud=1,
    cldovrlp=2, has_reqc=has_reqi=has_reqs=0, o3input=0, no aerosols.

    Column layout: ``[nk, C]`` layer arrays (k = 1..nk), ``[nk+1, C]`` level
    arrays (p8w, t8w: k = 1..nk+1), ``[C]`` surface arrays.  Returns
    rthratenlw, lwtenc, o33d ``[nk, C]``, glw, olr, lwcf, lwupt, lwuptc,
    lwdnt, lwdntc, lwupb, lwupbc, lwdnb, lwdnbc ``[C]`` and co2out."""
    xp = bk.xp
    T = tables(bk) if T is None else T
    P = _profiles()
    nk = p3d.shape[0]
    nbuf = nlayers - nk
    co2 = (280.0 + 90.0 * np.exp(0.02 * (yr - 2000))) * 1.0e-6
    ch4, n2o, cfc11, cfc12 = 1774.0e-9, 319.0e-9, 0.251e-9, 0.538e-9
    cfc22, ccl4, o2 = 0.169e-9, 0.093e-9, 0.209488
    amdw, amdo = 1.607793, 0.603461

    pw1d = p8w * 0.01
    tw1d = t8w
    qv1d = xp.clip(qv3d, 0.0, None)
    t1d = t3d
    p1d = p3d * 0.01
    qc1d = xp.clip(qc3d, 0.0, None)
    qr1d = xp.clip(qr3d, 0.0, None)
    cold = t1d < 273.15
    qi1d = xp.where(cold, qc1d, 0.0)
    qs1d = xp.where(cold, qr1d, 0.0)
    qc1d = xp.where(cold, 0.0, qc1d)
    qr1d = xp.where(cold, 0.0, qr1d)
    qv1d = xp.clip(qv1d, 1.0e-12, None)

    # model layers, then buffer layers every deltap mb up to 0 mb
    plev = [pw1d[k] for k in range(nk + 1)]
    for _ in range(nbuf):
        plev.append(plev[-1] - deltap)
    plev[nlayers] = xp.zeros_like(plev[0])
    play = [p1d[k] for k in range(nk)]
    for L in range(nk, nlayers):
        play.append(0.5 * (plev[L] + plev[L + 1]))
    plev = xp.stack(plev, axis=0)
    play = xp.stack(play, axis=0)
    pdel = plev[:nk] - plev[1 : nk + 1]
    # temperatures above the model top from the reference profile, shifted
    pprof, tprof = bk.asarray(P["pprof"]), bk.asarray(P["tprof"])
    npl = 60
    n = xp.sum(xp.astype(pprof[None, None, 1:] >= plev[..., None], ity(bk)), axis=-1)
    klev = 1 + n
    inside = (pprof[npl - 1] < plev) & (klev != npl)
    klev = clip(bk, klev, 1, npl)
    vark = take(bk, tprof, klev - 1)
    vark1 = take(bk, tprof, clip(bk, klev, 1, npl - 1))
    pk = take(bk, pprof, klev - 1)
    pk1 = take(bk, pprof, clip(bk, klev, 1, npl - 1))
    wght = xp.where(inside, (plev - pk) / (pk1 - pk), 0.0)
    vark1 = xp.where(inside, vark1, vark)
    varint = wght * (vark1 - vark) + vark
    tlev_top = varint[nk:] + _e0(tw1d[nk - 1] - varint[nk - 1])
    tlev = xp.concat([tw1d[:nk], tlev_top], axis=0)  # levels 1..nlayers+1
    tlay = xp.concat([t1d[: nk - 1], 0.5 * (tlev[nk:] + tlev[nk - 1 : -1])], axis=0)

    def ext(x):  # constant above the model top
        return xp.concat([x, xp.broadcast_to(x[-1:], (nbuf,) + x.shape[1:])], axis=0)

    h2ovmr = ext(qv1d * amdw)
    ones = xp.ones_like(play)
    o3mmr = inirad(bk, plev=plev[:nlayers])
    o3vmr = o3mmr * amdo
    emis = xp.broadcast_to(emiss[:, None], (emiss.shape[0], nbndlw))

    cldfrac = cldfra3d
    landfrac = 2.0 - xland
    landm = landfrac
    snowh = 0.001 * snow
    icefrac = xice
    gicewp = (qi1d + qs1d) * pdel * 100.0 / g * 1000.0
    gliqwp = qc1d * pdel * 100.0 / g * 1000.0
    cicewp = gicewp / xp.clip(cldfrac, 0.01, None)
    cliqwp = gliqwp / xp.clip(cldfrac, 0.01, None)
    tl = tlay[:nk]
    reliq = relcalc(
        bk, t=tl, landfrac=landfrac, landm=landm, icefrac=icefrac, snowh=snowh
    )
    reice = reicalc(bk, t=tl)
    reice = xp.clip(reice * 1.0315, None, 140.0)

    def pad(x, v):
        return xp.concat(
            [x, xp.full((nbuf,) + tuple(x.shape[1:]), v, dtype=x.dtype)], axis=0
        )

    mc = mcica_subcol_lw(
        bk,
        permuteseed=150,
        play=play,
        cldfrac=pad(cldfrac, 0.0),
        ciwp=pad(cicewp, 0.0),
        clwp=pad(cliqwp, 0.0),
        cswp=xp.zeros_like(play),
        rei=pad(reice, 10.0),
        rel=pad(reliq, 10.0),
        res=10.0 * ones,
        tauc=xp.zeros(tuple(play.shape) + (nbndlw,), dtype=play.dtype),
        T=T,
    )
    r = rrtmg_lw(
        bk,
        play=play,
        plev=plev,
        tlay=tlay,
        tlev=tlev,
        tsfc=tsk,
        h2ovmr=h2ovmr,
        o3vmr=o3vmr,
        co2vmr=co2 * ones,
        ch4vmr=ch4 * ones,
        n2ovmr=n2o * ones,
        o2vmr=o2 * ones,
        cfc11vmr=cfc11 * ones,
        cfc12vmr=cfc12 * ones,
        cfc22vmr=cfc22 * ones,
        ccl4vmr=ccl4 * ones,
        emis=emis,
        cldfmcl=mc["cldfmcl"],
        taucmcl=mc["taucmcl"],
        ciwpmcl=mc["ciwpmcl"],
        clwpmcl=mc["clwpmcl"],
        cswpmcl=mc["cswpmcl"],
        reicmcl=mc["reicmcl"],
        relqmcl=mc["relqmcl"],
        resnmcl=mc["resnmcl"],
        T=T,
        trace=trace,
    )
    if trace is not None:
        trace.update(
            play=play, plev=plev, tlay=tlay, tlev=tlev, o3vmr=o3vmr, h2ovmr=h2ovmr,
            cicewp=cicewp, cliqwp=cliqwp, reliq=reliq, reice=reice, mcica=mc,
        )  # fmt: skip
    uflx, dflx, uflxc, dflxc = r["uflx"], r["dflx"], r["uflxc"], r["dflxc"]
    top = nlayers
    return {
        "rthratenlw": r["hr"][:nk] / 86400.0,
        "lwtenc": r["hrc"][:nk] / 86400.0,
        "o33d": o3vmr[:nk],
        "glw": dflx[0],
        "olr": uflx[top],
        "lwcf": uflxc[top] - uflx[top],
        "lwupt": uflx[top],
        "lwuptc": uflxc[top],
        "lwdnt": dflx[top],
        "lwdntc": dflxc[top],
        "lwupb": uflx[0],
        "lwupbc": uflxc[0],
        "lwdnb": dflx[0],
        "lwdnbc": dflxc[0],
        "co2out": co2,
    }


def _e0(x):
    return x[None]
