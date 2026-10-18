"""RRTMG shortwave radiation (src/module_ra_rrtmg_sw.F) as CM1 uses it
(radopt=2: inflgsw=2, iceflgsw=3, liqflgsw=1, McICA maximum-random overlap
with the KISS generator, icpr=1, no aerosols: aer_opt=0, iaer=10 with zero
aerosol optical depth).

Layout as in :mod:`cm1x.module_ra_rrtmg_lw`: layer arrays ``[L, C]``
(bottom-up, ``L = nlayers``), level arrays ``[L+1, C]``, g-point arrays
``[L, C, G]`` (G = 112) and band arrays ``[..., 14]`` (bands 16..29).
Inside ``spcvmc_sw``, ``reftra_sw`` and ``vrtqdr_sw`` the layer axis runs
top-down, as in the Fortran (``jk = 1`` is the top layer, ``klev+1`` the
surface).  Recurrences over layers and the sums over g-points run in the
Fortran order (``bk.scan``); every ``if`` on array values is evaluated on
both sides and selected with ``xp.where``.

Only the broadband outputs CM1 uses are formed: the UV/near-IR split of
``spcvmc_sw`` (used for sf_surface_physics=8 only) is not computed.
"""

from __future__ import annotations

import functools

import numpy as np

from . import _ftables as ft
from . import module_ra_rrtmg_lw as lw
from .module_ra_rrtmg_lw import _e, clip, ffloat, fint, ity, take

SRC = "module_ra_rrtmg_sw.F"

nbndsw = 14
ngptsw = 112
mg = 16
jpb1 = 16
rrsw_scon = 1.36822e03
od_lo = 0.06

# rrsw_kgNN contents of RRTMG_SW_DATA, in READ order: name:Fortran shape
_SCHEMA = {
    16: "rayl: strrat1: layreffr:i kao:9,5,13,16 kbo:5,47,16 selfrefo:10,16 forrefo:3,16 sfluxrefo:16",
    17: "rayl: strrat: layreffr:i kao:9,5,13,16 kbo:5,5,47,16 selfrefo:10,16 forrefo:4,16 sfluxrefo:16,5",
    18: "rayl: strrat: layreffr:i kao:9,5,13,16 kbo:5,47,16 selfrefo:10,16 forrefo:3,16 sfluxrefo:16,9",
    19: "rayl: strrat: layreffr:i kao:9,5,13,16 kbo:5,47,16 selfrefo:10,16 forrefo:3,16 sfluxrefo:16,9",
    20: "rayl: layreffr:i absch4o:16 kao:5,13,16 kbo:5,47,16 selfrefo:10,16 forrefo:4,16 sfluxrefo:16",
    21: "rayl: strrat: layreffr:i kao:9,5,13,16 kbo:5,5,47,16 selfrefo:10,16 forrefo:4,16 sfluxrefo:16,9",
    22: "rayl: strrat: layreffr:i kao:9,5,13,16 kbo:5,47,16 selfrefo:10,16 forrefo:3,16 sfluxrefo:16,9",
    23: "raylo:16 givfac: layreffr:i kao:5,13,16 selfrefo:10,16 forrefo:3,16 sfluxrefo:16",
    24: "raylao:16,9 raylbo:16 strrat: layreffr:i abso3ao:16 abso3bo:16 kao:9,5,13,16 kbo:5,47,16 selfrefo:10,16 forrefo:3,16 sfluxrefo:16,9",
    25: "raylo:16 layreffr:i abso3ao:16 abso3bo:16 kao:5,13,16 sfluxrefo:16",
    26: "raylo:16 sfluxrefo:16",
    27: "raylo:16 scalekur: layreffr:i kao:5,13,16 kbo:5,47,16 sfluxrefo:16",
    28: "rayl: strrat: layreffr:i kao:9,5,13,16 kbo:5,5,47,16 sfluxrefo:16,5",
    29: "rayl: layreffr:i absh2oo:16 absco2o:16 kao:5,13,16 kbo:5,47,16 selfrefo:10,16 forrefo:4,16 sfluxrefo:16",
}

_NSPA = [9, 9, 9, 9, 1, 9, 9, 1, 9, 1, 0, 1, 9, 1]
_NSPB = [1, 5, 1, 1, 1, 5, 1, 0, 1, 0, 0, 1, 5, 1]


@functools.cache
def _wvn():
    """swdatinit / swcmbdat constants (rrsw_wvn)."""
    d = {"nspa": _NSPA, "nspb": _NSPB}
    d["ngc"] = [6, 12, 8, 8, 10, 10, 2, 10, 8, 6, 6, 8, 6, 12]
    d["ngs"] = list(np.cumsum(d["ngc"]))
    txt = " ".join(ft.subroutine_body(SRC, "swcmbdat"))

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
def rrtmg_swinit(cpdair=1005.7, grav=9.81):
    """``rrtmg_swinit`` -> ``rrtmg_swlookuptable`` + ``rrtmg_sw_ini``: numpy
    tables (float64) of the reduced 112-g-point k-distribution, the cloud
    optics, the reference atmosphere and the exponential table."""
    w = _wvn()
    recs = ft.read_unformatted(ft.cm1_root() / "run" / "RRTMG_SW_DATA")
    if len(recs) != nbndsw:
        raise ValueError("RRTMG_SW_DATA: expected 14 records")
    ngc, ngn, ngm, wt = w["ngc"], w["ngn"], w["ngm"], w["wt"]
    # rrtmg_sw_ini: g-point weights (ng(ib) = 16 for every band)
    rwgt = np.zeros(nbndsw * mg)
    igcsm = 0
    for ibnd in range(nbndsw):
        iprsm = 0
        if ngc[ibnd] < mg:
            wtsm = np.zeros(mg)
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
    for i in range(nbndsw):
        b = jpb1 + i
        items = lw.parse_schema(_SCHEMA[b])
        raw = ft.unpack_record(recs[i], items)
        red = lw.cmbgb(
            raw, items, ngc[i], ngn[off : off + ngc[i]],
            rwgt[i * mg : (i + 1) * mg], ("sfluxref", "rayla"),
        )  # fmt: skip
        off += ngc[i]
        if "ka" in red:  # equivalence absa(65|585, ng) <-> ka(.., 13, ng)
            red["absa"] = red["ka"].reshape((-1, ngc[i]), order="F")
        if "kb" in red:
            red["absb"] = red["kb"].reshape((-1, ngc[i]), order="F")
        if "layreffr" in red:
            red["layreffr"] = int(red["layreffr"])
        bands[b] = red
    T = {"bands": bands, "rwgt": rwgt}
    T.update({k: w[k] for k in ("nspa", "nspb", "ngc", "ngs", "ngb")})
    # swdatinit
    T["grav"] = grav
    T["avogad"] = 6.02214199e23
    T["secdy"] = 8.64e4
    T["oneminus"] = 1.0 - 1.0e-06
    T["heatfac"] = grav * T["secdy"] / (cpdair * 1.0e2)
    # rrsw_tbl
    ntbl, pade, expeps = 10000, 0.278, 1.0e-20
    bpade = 1.0 / pade
    ex = np.zeros(ntbl + 1)
    ex[0], ex[ntbl] = 1.0, expeps
    for itr in range(1, ntbl):
        tfn = float(itr) / float(ntbl)
        tau = bpade * tfn / (1.0 - tfn)
        ex[itr] = max(np.exp(-tau), expeps)
    T.update(exp_tbl=ex, bpade=bpade, tblint=10000.0)
    T.update(swcldpr())
    T.update(swatmref())
    return T


def swcldpr():
    """Cloud optics of ``swcldpr`` used by liqflag 1 / iceflag 3 (band axis
    16..29 is the second axis)."""
    b = (16, 29)
    dims = {n: ((1, 58), b) for n in ("extliq1", "ssaliq1", "asyliq1")}
    dims.update({n: ((1, 46), b) for n in ("extice3", "ssaice3", "asyice3", "fdlice3")})
    return ft.assignments(SRC, "swcldpr", dims)


def swatmref():
    return ft.assignments(
        SRC, "swatmref", {"pref": ((1, 59),), "preflog": ((1, 59),), "tref": ((1, 59),)}
    )


_BK_CACHE = {}


def tables(bk, cpdair=1005.7, grav=9.81):
    """rrtmg_swinit tables as backend arrays (cached per backend/dtype)."""
    key = (bk.name, str(bk.dtype), str(bk.device), cpdair, grav)
    if key not in _BK_CACHE:
        T = rrtmg_swinit(cpdair, grav)

        def conv(v):
            if isinstance(v, np.ndarray) and v.dtype.kind == "f":
                return bk.asarray(v)
            if isinstance(v, dict):
                return {k: conv(x) for k, x in v.items()}
            return v

        _BK_CACHE[key] = conv(T)
    return _BK_CACHE[key]


def _bandidx(T):
    """0-based band index (0..13) of every g-point."""
    return np.array(T["ngb"]) - jpb1


# ---------------------------------------------------------------------------
# mcica_subcol_gen_sw
# ---------------------------------------------------------------------------
kissvec = lw.kissvec  # identical KISS generator


def generate_stochastic_clouds_sw(
    bk, *, pmid, cld, clwp, ciwp, cswp, tauc, ssac, asmc, fsfc, changeseed, nsubcol, ngb
):
    """``generate_stochastic_clouds_sw`` for icld=2, irng=0: the longwave
    algorithm (same KISS stream and overlap), with the band index of g-point
    ``isubcol`` taken as ``ngb(isubcol) - ngb(1) + 1`` and clear subcolumns
    getting ssac=1, asmc=fsfc=0.  ``tauc``/``ssac``/``asmc``/``fsfc`` are
    ``[L, C, nbnd]``; returns the ``*_stoch`` arrays ``[L, C, nsub]``."""
    xp = bk.xp
    nb = np.array(ngb) - ngb[0] + 1
    r = lw.generate_stochastic_clouds(
        bk, pmid=pmid, cld=cld, clwp=clwp, ciwp=ciwp, cswp=cswp, tauc=tauc,
        changeseed=changeseed, nsubcol=nsubcol, ngb=nb,
    )  # fmt: skip
    cloudy = r["cld_stoch"] == 1.0
    idx = xp.asarray(nb - 1)

    def sel(tab, clear):
        return xp.where(cloudy, xp.take(tab, idx, axis=-1), clear)

    r["ssac_stoch"] = sel(ssac, 1.0)
    r["asmc_stoch"] = sel(asmc, 0.0)
    r["fsfc_stoch"] = sel(fsfc, 0.0)
    return r


def mcica_subcol_sw(
    bk, *, permuteseed, play, cldfrac, ciwp, clwp, cswp, rei, rel, res,
    tauc, ssac, asmc, fsfc, T=None,
):  # fmt: skip
    """``mcica_subcol_sw`` (icld=2, irng=0).  ``[L, C]`` inputs (play in
    mb), band arrays ``[L, C, 14]``.  Returns cldfmcl, ciwpmcl, clwpmcl,
    cswpmcl, taucmcl, ssacmcl, asmcmcl, fsfcmcl ``[L, C, 112]`` and
    reicmcl, relqmcl, resnmcl ``[L, C]``."""
    T = tables(bk) if T is None else T
    r = generate_stochastic_clouds_sw(
        bk, pmid=play * 1.0e2, cld=cldfrac, clwp=clwp, ciwp=ciwp, cswp=cswp,
        tauc=tauc, ssac=ssac, asmc=asmc, fsfc=fsfc, changeseed=permuteseed,
        nsubcol=ngptsw, ngb=T["ngb"],
    )  # fmt: skip
    return {
        "cldfmcl": r["cld_stoch"],
        "clwpmcl": r["clwp_stoch"],
        "ciwpmcl": r["ciwp_stoch"],
        "cswpmcl": r["cswp_stoch"],
        "taucmcl": r["tauc_stoch"],
        "ssacmcl": r["ssac_stoch"],
        "asmcmcl": r["asmc_stoch"],
        "fsfcmcl": r["fsfc_stoch"],
        "reicmcl": rei,
        "relqmcl": rel,
        "resnmcl": res,
    }


# ---------------------------------------------------------------------------
# rrtmg_sw_rad: inatm_sw
# ---------------------------------------------------------------------------
def inatm_sw(
    bk, *, play, plev, tlay, tlev, tsfc, h2ovmr, o3vmr, co2vmr, ch4vmr, n2ovmr,
    o2vmr, adjes, scon, cldfmcl, taucmcl, ssacmcl, asmcmcl, fsfcmcl, ciwpmcl,
    clwpmcl, cswpmcl, reicmcl, relqmcl, resnmcl, T=None,
):  # fmt: skip
    """``inatm_sw`` (dyofyr=0, icld>=1, iceflag<5, zero aerosols).  Returns
    pavel, tavel, pz, tz, tbound, pdp, coldry, wkl {1..7}, adjflux [14] and
    the McICA arrays (cswpmc, resnmc zero as for iceflag<5)."""
    xp = bk.xp
    T = tables(bk) if T is None else T
    amd, amw = 28.9660, 18.0160
    pz, tz = plev, tlev
    pdp = pz[:-1] - pz[1:]
    w1 = h2ovmr
    amm = (1.0 - w1) * amd + w1 * amw
    coldry = (
        (pz[:-1] - pz[1:])
        * 1.0e3
        * T["avogad"]
        / (1.0e2 * T["grav"] * amm * (1.0 + w1))
    )
    z = xp.zeros_like(coldry)
    vmr = {1: h2ovmr, 2: co2vmr, 3: o3vmr, 4: n2ovmr, 5: z, 6: ch4vmr, 7: o2vmr}
    wkl = {m: coldry * v for m, v in vmr.items()}
    solvar = scon / rrsw_scon
    adjflux = adjes * solvar
    return {
        "pavel": play,
        "tavel": tlay,
        "pz": pz,
        "tz": tz,
        "tbound": tsfc,
        "pdp": pdp,
        "coldry": coldry,
        "wkl": wkl,
        "adjflux": adjflux,
        "cldfmc": cldfmcl,
        "taucmc": taucmcl,
        "ssacmc": ssacmcl,
        "asmcmc": asmcmcl,
        "fsfcmc": fsfcmcl,
        "ciwpmc": ciwpmcl,
        "clwpmc": clwpmcl,
        "cswpmc": xp.zeros_like(cswpmcl),
        "reicmc": reicmcl,
        "relqmc": relqmcl,
        "resnmc": xp.zeros_like(resnmcl),
    }


# ---------------------------------------------------------------------------
# rrtmg_sw_cldprmc
# ---------------------------------------------------------------------------
def _bandtab(bk, tab, idx, frac, T):
    """tab(index, ngb(ig)) + frac*(tab(index+1, ngb(ig)) - tab(index, ngb(ig)))
    for a [n, 14] table, 1-based ``idx`` [L, C, 1] -> [L, C, 112]."""
    xp = bk.xp
    gtab = xp.take(tab, xp.asarray(_bandidx(T)), axis=1)  # [n, 112]
    n = gtab.shape[0]
    g = xp.arange(ngptsw)
    flat = xp.reshape(gtab, (-1,))
    i0 = (clip(bk, idx, 1, n - 1) - 1) * ngptsw + g
    sh = tuple(i0.shape)
    a0 = xp.reshape(xp.take(flat, xp.reshape(i0, (-1,))), sh)
    a1 = xp.reshape(xp.take(flat, xp.reshape(i0 + ngptsw, (-1,))), sh)
    return a0, a0 + frac * (a1 - a0)


def cldprmc_sw(
    bk, *, cldfmc, ciwpmc, clwpmc, cswpmc, reicmc, relqmc, taucmc, ssacmc, asmcmc,
    fsfcmc, T=None,
):  # fmt: skip
    """``cldprmc_sw`` for inflag=2, iceflag=3, liqflag=1.  ``[L, C, 112]``
    arrays, effective sizes ``[L, C]``.  Returns taormc, taucmc, ssacmc,
    asmcmc.  (The Fortran stops when a size leaves the table range; here the
    table index is clamped instead.)"""
    xp = bk.xp
    T = tables(bk) if T is None else T
    cldmin = 1.0e-20
    cwp = ciwpmc + clwpmc + cswpmc
    active = (cldfmc >= cldmin) & ((cwp >= cldmin) | (taucmc >= cldmin))
    # ice, iceflag 3
    radice = _e(reicmc)
    factor = (radice - 2.0) / 3.0
    index = fint(bk, factor)
    index = xp.where(index == 46, 45, index)
    fi = factor - ffloat(bk, index)
    _, extcoice = _bandtab(bk, T["extice3"], index, fi, T)
    _, ssacoice = _bandtab(bk, T["ssaice3"], index, fi, T)
    _, gice = _bandtab(bk, T["asyice3"], index, fi, T)
    _, fdelta = _bandtab(bk, T["fdlice3"], index, fi, T)
    noice = (ciwpmc + cswpmc) == 0.0
    ssacoice_s = xp.where(noice, 1.0, ssacoice)
    forwice = fdelta + 0.5 / ssacoice_s
    forwice = xp.where(forwice > gice, gice, forwice)
    extcoice = xp.where(noice, 0.0, extcoice)
    ssacoice = xp.where(noice, 0.0, ssacoice)
    gice = xp.where(noice, 0.0, gice)
    forwice = xp.where(noice, 0.0, forwice)
    # liquid, liqflag 1
    radliq = _e(relqmc)
    index = fint(bk, radliq - 1.5)
    index = xp.where(index == 0, 1, index)
    index = xp.where(index == 58, 57, index)
    fl = radliq - 1.5 - ffloat(bk, index)
    _, extcoliq = _bandtab(bk, T["extliq1"], index, fl, T)
    s0, ssacoliq = _bandtab(bk, T["ssaliq1"], index, fl, T)
    ssacoliq = xp.where((fl < 0.0) & (ssacoliq > 1.0), s0, ssacoliq)
    _, gliq = _bandtab(bk, T["asyliq1"], index, fl, T)
    forwliq = gliq * gliq
    noliq = clwpmc == 0.0
    extcoliq = xp.where(noliq, 0.0, extcoliq)
    ssacoliq = xp.where(noliq, 0.0, ssacoliq)
    gliq = xp.where(noliq, 0.0, gliq)
    forwliq = xp.where(noliq, 0.0, forwliq)
    # delta-scaled optical properties (iceflag < 5)
    tauliqorig = clwpmc * extcoliq
    tauiceorig = ciwpmc * extcoice
    taormc = tauliqorig + tauiceorig
    ssaliq = ssacoliq * (1.0 - forwliq) / (1.0 - forwliq * ssacoliq)
    tauliq = (1.0 - forwliq * ssacoliq) * tauliqorig
    ssaice = ssacoice * (1.0 - forwice) / (1.0 - forwice * ssacoice)
    tauice = (1.0 - forwice * ssacoice) * tauiceorig
    scatliq = ssaliq * tauliq
    scatice = ssaice * tauice
    tau = tauliq + tauice
    tau = xp.where(tau == 0.0, cldmin, tau)
    scatice = xp.where(scatice == 0.0, cldmin, scatice)
    ssa = (scatliq + scatice) / tau
    asm = (1.0 / (scatliq + scatice)) * (
        scatliq * (gliq - forwliq) / (1.0 - forwliq)
        + scatice * ((gice - forwice) / (1.0 - forwice))
    )
    return {
        "taormc": xp.where(active, taormc, taucmc),
        "taucmc": xp.where(active, tau, taucmc),
        "ssacmc": xp.where(active, ssa, ssacmc),
        "asmcmc": xp.where(active, asm, asmcmc),
    }


# ---------------------------------------------------------------------------
# rrtmg_sw_setcoef
# ---------------------------------------------------------------------------
def setcoef_sw(bk, *, pavel, tavel, coldry, wkl, T=None):
    """``setcoef_sw``.  ``[L, C]`` inputs, ``wkl`` {1..7}.  Returns laytrop,
    laylow ([C] counts), jp, jt, jt1, the column amounts (colh2o, colco2,
    colo3, coln2o, colch4, colo2, colmol), co2mult, fac00..fac11, selffac,
    selffrac, indself (0 above laytrop, as in the Fortran), forfac,
    forfrac, indfor."""
    xp = bk.xp
    T = tables(bk) if T is None else T
    preflog, tref = T["preflog"], T["tref"]
    stpfac = 296.0 / 1013.0
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
    it = ity(bk)
    laytrop = xp.sum(xp.astype(lower, it), axis=0)
    laylow = xp.sum(xp.astype(lower & (plog >= 6.62), it), axis=0)

    forfac = scalefac / (1.0 + water)
    factor = (332.0 - tavel) / 36.0
    indfor_lo = clip(bk, fint(bk, factor), 1, 2)
    forfrac_lo = factor - ffloat(bk, indfor_lo)
    factor_up = (tavel - 188.0) / 36.0
    indfor = xp.where(lower, indfor_lo, 3)
    forfrac = xp.where(lower, forfrac_lo, factor_up - 1.0)
    selffac = xp.where(lower, water * forfac, 0.0)
    factor = (tavel - 188.0) / 7.2
    indself_lo = clip(bk, fint(bk, factor) - 7, 1, 9)
    selffrac = xp.where(lower, factor - ffloat(bk, indself_lo + 7), 0.0)
    indself = xp.where(lower, indself_lo, 0)

    col = {
        "colh2o": 1.0e-20 * wkl[1],
        "colco2": 1.0e-20 * wkl[2],
        "colo3": 1.0e-20 * wkl[3],
        "coln2o": 1.0e-20 * wkl[4],
        "colch4": 1.0e-20 * wkl[6],
        "colo2": 1.0e-20 * wkl[7],
    }
    col["colmol"] = 1.0e-20 * coldry + col["colh2o"]
    for name in ("colco2", "coln2o", "colch4", "colo2"):
        col[name] = xp.where(col[name] == 0.0, 1.0e-32 * coldry, col[name])
    co2reg = 3.55e-24 * coldry
    co2mult = (
        (col["colco2"] - co2reg)
        * 272.63
        * xp.exp(-1919.4 / tavel)
        / (8.7604e-4 * tavel)
    )
    compfp = 1.0 - fp
    out = {
        "laytrop": laytrop,
        "laylow": laylow,
        "jp": jp,
        "jt": jt,
        "jt1": jt1,
        "co2mult": co2mult,
        "fac10": compfp * ft,
        "fac00": compfp * (1.0 - ft),
        "fac11": fp * ft1,
        "fac01": fp * (1.0 - ft1),
        "selffac": selffac,
        "selffrac": selffrac,
        "indself": indself,
        "forfac": forfac,
        "forfrac": forfrac,
        "indfor": indfor,
    }
    out.update(col)
    return out


# ---------------------------------------------------------------------------
# rrtmg_sw_taumol
# ---------------------------------------------------------------------------
class _TauSW:
    """Shared look-up arithmetic of the ``taumolNN`` bands (one instance per
    ``taumol_sw`` call; ``d`` = setcoef_sw output)."""

    def __init__(self, bk, T, d):
        self.bk, self.xp, self.T, self.d = bk, bk.xp, T, d
        L = d["jp"].shape[0]
        self.L = L
        self.lay = self.xp.reshape(self.xp.arange(L, dtype=ity(bk)), (L, 1))  # 0-based
        self.lower = self.lay < d["laytrop"][None, :]

    def pick(self, lo, up):
        m = self.lower
        return self.xp.where(m[..., None] if lo.ndim == 3 else m, lo, up)

    def interp(self, tab, i, f):
        """tab(i) + f*(tab(i+1) - tab(i)) along the leading axis (1-based i)."""
        a = take(self.bk, tab, i - 1)
        b = take(self.bk, tab, i)
        return a + _e(f) * (b - a)

    def spec(self, cola, ratio, colb, mult):
        """speccomb, js = 1 + int(specmult), fs = mod(specmult, 1)."""
        bk, xp = self.bk, self.xp
        speccomb = cola + ratio * colb
        specparm = cola / speccomb
        om = self.T["oneminus"]
        specparm = xp.where(specparm >= om, om, specparm)
        specmult = mult * specparm
        ji = fint(bk, specmult)
        return speccomb, 1 + ji, specmult - ffloat(bk, ji)

    def ind_lo(self, nsp, js):
        d = self.d
        ind0 = ((d["jp"] - 1) * 5 + (d["jt"] - 1)) * nsp + js
        ind1 = (d["jp"] * 5 + (d["jt1"] - 1)) * nsp + js
        return ind0, ind1

    def ind_up(self, nsp, js):
        d = self.d
        ind0 = ((d["jp"] - 13) * 5 + (d["jt"] - 1)) * nsp + js
        ind1 = ((d["jp"] - 12) * 5 + (d["jt1"] - 1)) * nsp + js
        return ind0, ind1

    def maj8(self, tab, ind0, ind1, fs, o1, o2, o3):
        """fac000*tab(ind0) + fac100*tab(ind0+o1) + fac010*tab(ind0+o2) +
        fac110*tab(ind0+o3) + (same with ind1 and fac0x1/fac1x1)."""
        bk, d = self.bk, self.d
        f000 = (1.0 - fs) * d["fac00"]
        f010 = (1.0 - fs) * d["fac10"]
        f100 = fs * d["fac00"]
        f110 = fs * d["fac10"]
        f001 = (1.0 - fs) * d["fac01"]
        f011 = (1.0 - fs) * d["fac11"]
        f101 = fs * d["fac01"]
        f111 = fs * d["fac11"]
        i0, i1 = ind0 - 1, ind1 - 1
        t = _e(f000) * take(bk, tab, i0)
        t = t + _e(f100) * take(bk, tab, i0 + o1)
        t = t + _e(f010) * take(bk, tab, i0 + o2)
        t = t + _e(f110) * take(bk, tab, i0 + o3)
        t = t + _e(f001) * take(bk, tab, i1)
        t = t + _e(f101) * take(bk, tab, i1 + o1)
        t = t + _e(f011) * take(bk, tab, i1 + o2)
        return t + _e(f111) * take(bk, tab, i1 + o3)

    def s4(self, tab, ind0, ind1):
        d, bk = self.d, self.bk
        t = _e(d["fac00"]) * take(bk, tab, ind0 - 1)
        t = t + _e(d["fac10"]) * take(bk, tab, ind0)
        t = t + _e(d["fac01"]) * take(bk, tab, ind1 - 1)
        return t + _e(d["fac11"]) * take(bk, tab, ind1)

    def selfterm(self, B):
        d = self.d
        return _e(d["selffac"]) * self.interp(B["selfref"], d["indself"], d["selffrac"])

    def forterm(self, B):
        d = self.d
        return _e(d["forfac"]) * self.interp(B["forref"], d["indfor"], d["forfrac"])

    def forinterp(self, B):
        d = self.d
        return self.interp(B["forref"], d["indfor"], d["forfrac"])

    def lower_std(self, B, cola, ratio, colb):
        """speccomb*(8-term absa) + colh2o*(self + for), nspa = 9."""
        speccomb, js, fs = self.spec(cola, ratio, colb, 8.0)
        ind0, ind1 = self.ind_lo(9, js)
        t = _e(speccomb) * self.maj8(B["absa"], ind0, ind1, fs, 1, 9, 10)
        return t, js, fs

    def bcast(self, v):
        L, C = self.d["jp"].shape
        return self.xp.broadcast_to(v, (L, C, v.shape[-1]))

    def rayl(self, B):
        """taur = colmol*rayl (a scalar or one value per g-point)."""
        r, colmol = B["rayl"], self.d["colmol"]
        L, C = colmol.shape
        ng = B["sfluxref"].shape[-1]
        return self.xp.broadcast_to(_e(colmol) * r, (L, C, ng))

    def sflux_interp(self, B, js, fs):
        """sfluxref(ig, js) + fs*(sfluxref(ig, js+1) - sfluxref(ig, js))."""
        return self.interp(B["sfluxref"], js, fs)

    def _onehot(self, l0, valid):
        return (self.lay == l0[None, :]) & valid[None, :]

    def solfr_lower(self, layreffr):
        """One-hot [L, C] of the layer where the lower-atmosphere loop sets
        sfluxzen: laysolfr = laytrop, moved to min(lay+1, laytrop) for the
        last lay <= laytrop with jp(lay) < layreffr <= jp(lay+1)."""
        xp, d = self.xp, self.d
        jp, t, L = d["jp"], d["laytrop"], self.L
        if layreffr is None:
            last = xp.full(t.shape, -1, dtype=t.dtype)
        else:
            tr = (
                (self.lay[:-1] < t[None, :])
                & (jp[:-1] < layreffr)
                & (jp[1:] >= layreffr)
            )
            cand = xp.where(tr, self.lay[:-1], -1)
            last = (
                xp.max(cand, axis=0) if L > 1 else xp.full(t.shape, -1, dtype=t.dtype)
            )
        l0 = xp.where(last >= 0, xp.minimum(last + 1, t - 1), t - 1)
        return self._onehot(l0, t >= 1)

    def solfr_upper(self, layreffr):
        """One-hot [L, C] of the layer where the upper-atmosphere loop sets
        sfluxzen: the last lay > laytrop with jp(lay-1) < layreffr <=
        jp(lay), else nlayers."""
        xp, d = self.xp, self.d
        jp, t, L = d["jp"], d["laytrop"], self.L
        lay = self.lay[1:]
        tr = (lay >= t[None, :]) & (jp[:-1] < layreffr) & (jp[1:] >= layreffr)
        last = xp.max(xp.where(tr, lay, -1), axis=0)
        l0 = xp.where(last >= 0, last, L - 1)
        return self._onehot(l0, t < L)

    def select(self, onehot, cand):
        """cand at the one-hot layer (zero where none): [L, C, ng] -> [C, ng]."""
        xp = self.xp
        return xp.sum(xp.where(onehot[..., None], cand, 0.0), axis=0)


def taumol16(bk, tm):
    B, d = tm.T["bands"][16], tm.d
    lo, _, _ = tm.lower_std(B, d["colh2o"], B["strrat1"], d["colch4"])
    lo = lo + _e(d["colh2o"]) * (tm.selfterm(B) + tm.forterm(B))
    ind0, ind1 = tm.ind_up(1, 1)
    up = _e(d["colch4"]) * tm.s4(B["absb"], ind0, ind1)
    sfl = tm.select(tm.solfr_upper(B["layreffr"]), tm.bcast(B["sfluxref"]))
    return tm.pick(lo, up), tm.rayl(B), sfl


def taumol17(bk, tm):
    B, d = tm.T["bands"][17], tm.d
    lo, _, _ = tm.lower_std(B, d["colh2o"], B["strrat"], d["colco2"])
    lo = lo + _e(d["colh2o"]) * (tm.selfterm(B) + tm.forterm(B))
    speccomb, js, fs = tm.spec(d["colh2o"], B["strrat"], d["colco2"], 4.0)
    ind0, ind1 = tm.ind_up(5, js)
    up = _e(speccomb) * tm.maj8(B["absb"], ind0, ind1, fs, 1, 5, 6)
    up = up + _e(d["colh2o"] * d["forfac"]) * tm.forinterp(B)
    sfl = tm.select(tm.solfr_upper(B["layreffr"]), tm.sflux_interp(B, js, fs))
    return tm.pick(lo, up), tm.rayl(B), sfl


def _lower_l(tm, B, col):
    """Bands 18, 19: standard lower atmosphere (h2o + ``col``), col*(4-term
    absb) above."""
    d = tm.d
    lo, js, fs = tm.lower_std(B, d["colh2o"], B["strrat"], d[col])
    lo = lo + _e(d["colh2o"]) * (tm.selfterm(B) + tm.forterm(B))
    ind0, ind1 = tm.ind_up(1, 1)
    up = _e(d[col]) * tm.s4(B["absb"], ind0, ind1)
    sfl = tm.select(tm.solfr_lower(B["layreffr"]), tm.sflux_interp(B, js, fs))
    return tm.pick(lo, up), tm.rayl(B), sfl


def taumol18(bk, tm):
    return _lower_l(tm, tm.T["bands"][18], "colch4")


def taumol19(bk, tm):
    return _lower_l(tm, tm.T["bands"][19], "colco2")


def taumol20(bk, tm):
    B, d = tm.T["bands"][20], tm.d
    ind0, ind1 = tm.ind_lo(1, 1)
    ch4 = _e(d["colch4"]) * B["absch4"]
    lo = (
        _e(d["colh2o"])
        * (tm.s4(B["absa"], ind0, ind1) + tm.selfterm(B) + tm.forterm(B))
        + ch4
    )
    ind0, ind1 = tm.ind_up(1, 1)
    up = _e(d["colh2o"]) * (tm.s4(B["absb"], ind0, ind1) + tm.forterm(B)) + ch4
    sfl = tm.select(tm.solfr_lower(B["layreffr"]), tm.bcast(B["sfluxref"]))
    return tm.pick(lo, up), tm.rayl(B), sfl


def taumol21(bk, tm):
    B, d = tm.T["bands"][21], tm.d
    lo, js, fs = tm.lower_std(B, d["colh2o"], B["strrat"], d["colco2"])
    lo = lo + _e(d["colh2o"]) * (tm.selfterm(B) + tm.forterm(B))
    speccomb, jsu, fsu = tm.spec(d["colh2o"], B["strrat"], d["colco2"], 4.0)
    ind0, ind1 = tm.ind_up(5, jsu)
    up = _e(speccomb) * tm.maj8(B["absb"], ind0, ind1, fsu, 1, 5, 6)
    up = up + _e(d["colh2o"] * d["forfac"]) * tm.forinterp(B)
    sfl = tm.select(tm.solfr_lower(B["layreffr"]), tm.sflux_interp(B, js, fs))
    return tm.pick(lo, up), tm.rayl(B), sfl


def taumol22(bk, tm):
    B, d = tm.T["bands"][22], tm.d
    o2adj = 1.6
    o2cont = _e(4.35e-4 * d["colo2"] / (350.0 * 2.0))
    lo, js, fs = tm.lower_std(B, d["colh2o"], o2adj * B["strrat"], d["colo2"])
    lo = lo + _e(d["colh2o"]) * (tm.selfterm(B) + tm.forterm(B)) + o2cont
    ind0, ind1 = tm.ind_up(1, 1)
    up = _e(d["colo2"] * o2adj) * tm.s4(B["absb"], ind0, ind1) + o2cont
    sfl = tm.select(tm.solfr_lower(B["layreffr"]), tm.sflux_interp(B, js, fs))
    return tm.pick(lo, up), tm.rayl(B), sfl


def taumol23(bk, tm):
    B, d = tm.T["bands"][23], tm.d
    ind0, ind1 = tm.ind_lo(1, 1)
    lo = _e(d["colh2o"]) * (
        B["givfac"] * tm.s4(B["absa"], ind0, ind1) + tm.selfterm(B) + tm.forterm(B)
    )
    sfl = tm.select(tm.solfr_lower(B["layreffr"]), tm.bcast(B["sfluxref"]))
    return tm.pick(lo, 0.0 * lo), tm.rayl(B), sfl


def taumol24(bk, tm):
    B, d = tm.T["bands"][24], tm.d
    lo, js, fs = tm.lower_std(B, d["colh2o"], B["strrat"], d["colo2"])
    lo = lo + _e(d["colo3"]) * B["abso3a"]
    lo = lo + _e(d["colh2o"]) * (tm.selfterm(B) + tm.forterm(B))
    ray_lo = _e(d["colmol"]) * tm.interp(B["rayla"], js, fs)
    ind0, ind1 = tm.ind_up(1, 1)
    up = _e(d["colo2"]) * tm.s4(B["absb"], ind0, ind1) + _e(d["colo3"]) * B["abso3b"]
    ray_up = _e(d["colmol"]) * B["raylb"]
    sfl = tm.select(tm.solfr_lower(B["layreffr"]), tm.sflux_interp(B, js, fs))
    return tm.pick(lo, up), tm.pick(ray_lo, tm.bcast(ray_up)), sfl


def taumol25(bk, tm):
    B, d = tm.T["bands"][25], tm.d
    ind0, ind1 = tm.ind_lo(1, 1)
    lo = _e(d["colh2o"]) * tm.s4(B["absa"], ind0, ind1) + _e(d["colo3"]) * B["abso3a"]
    up = tm.bcast(_e(d["colo3"]) * B["abso3b"])
    sfl = tm.select(tm.solfr_lower(B["layreffr"]), tm.bcast(B["sfluxref"]))
    return tm.pick(lo, up), tm.rayl(B), sfl


def taumol26(bk, tm):
    B = tm.T["bands"][26]
    ray = tm.rayl(B)
    sfl = tm.select(tm.solfr_lower(None), tm.bcast(B["sfluxref"]))  # no layreffr
    return 0.0 * ray, ray, sfl


def taumol27(bk, tm):
    B, d = tm.T["bands"][27], tm.d
    ind0, ind1 = tm.ind_lo(1, 1)
    lo = _e(d["colo3"]) * tm.s4(B["absa"], ind0, ind1)
    ind0, ind1 = tm.ind_up(1, 1)
    up = _e(d["colo3"]) * tm.s4(B["absb"], ind0, ind1)
    sfl = tm.select(
        tm.solfr_upper(B["layreffr"]), tm.bcast(B["scalekur"] * B["sfluxref"])
    )
    return tm.pick(lo, up), tm.rayl(B), sfl


def taumol28(bk, tm):
    B, d = tm.T["bands"][28], tm.d
    lo, _, _ = tm.lower_std(B, d["colo3"], B["strrat"], d["colo2"])
    speccomb, js, fs = tm.spec(d["colo3"], B["strrat"], d["colo2"], 4.0)
    ind0, ind1 = tm.ind_up(5, js)
    up = _e(speccomb) * tm.maj8(B["absb"], ind0, ind1, fs, 1, 5, 6)
    sfl = tm.select(tm.solfr_upper(B["layreffr"]), tm.sflux_interp(B, js, fs))
    return tm.pick(lo, up), tm.rayl(B), sfl


def taumol29(bk, tm):
    B, d = tm.T["bands"][29], tm.d
    ind0, ind1 = tm.ind_lo(1, 1)
    lo = _e(d["colh2o"]) * (
        tm.s4(B["absa"], ind0, ind1) + tm.selfterm(B) + tm.forterm(B)
    )
    lo = lo + _e(d["colco2"]) * B["absco2"]
    ind0, ind1 = tm.ind_up(1, 1)
    up = _e(d["colco2"]) * tm.s4(B["absb"], ind0, ind1) + _e(d["colh2o"]) * B["absh2o"]
    sfl = tm.select(tm.solfr_upper(B["layreffr"]), tm.bcast(B["sfluxref"]))
    return tm.pick(lo, up), tm.rayl(B), sfl


_TAUMOL = (
    taumol16, taumol17, taumol18, taumol19, taumol20, taumol21, taumol22,
    taumol23, taumol24, taumol25, taumol26, taumol27, taumol28, taumol29,
)  # fmt: skip


def taumol_sw(bk, *, coef, T=None):
    """``taumol_sw``: gaseous (taug) and Rayleigh (taur) optical depths
    ``[L, C, 112]`` and the incoming solar source sfluxzen ``[C, 112]``.
    ``coef`` = setcoef_sw output."""
    xp = bk.xp
    T = tables(bk) if T is None else T
    tm = _TauSW(bk, T, coef)
    taug, taur, sfl = [], [], []
    for f in _TAUMOL:
        g, r, s = f(bk, tm)
        taug.append(g)
        taur.append(r)
        sfl.append(s)
    return {
        "taug": xp.concat(taug, axis=-1),
        "taur": xp.concat(taur, axis=-1),
        "sfluxzen": xp.concat(sfl, axis=-1),
    }


# ---------------------------------------------------------------------------
# rrtmg_sw_reftra
# ---------------------------------------------------------------------------
def _expo(bk, T, x):
    """exp(-x) as spcvmc_sw / reftra_sw evaluate it: 1 - x + x*x/2 for
    x <= od_lo, else exp_tbl(int(tblint*x/(bpade+x) + 0.5))."""
    xp = bk.xp
    small = 1.0 - x + 0.5 * x * x
    tblind = x / (T["bpade"] + x)
    itind = fint(bk, T["tblint"] * tblind + 0.5)
    return xp.where(x <= od_lo, small, take(bk, T["exp_tbl"], itind))


def reftra_sw(bk, *, lrtchk, pgg, prmuz, ptau, pw, T=None):
    """``reftra_sw`` (kmodts=2, PIFM): layer reflectivities and
    transmissivities for direct (pref, ptra) and diffuse (prefd, ptrad)
    light.  Arrays of any shape broadcasting with ``prmuz``; returns pref,
    prefd, ptra, ptrad (0, 0, 1, 1 where ``lrtchk`` is false)."""
    xp = bk.xp
    T = tables(bk) if T is None else T
    eps = 1.0e-08
    zwcrit = 0.9999995
    zto1, zw, zg = ptau, pw, pgg
    zg3 = 3.0 * zg
    zgamma1 = (8.0 - zw * (5.0 + zg3)) * 0.25
    zgamma2 = 3.0 * (zw * (1.0 - zg)) * 0.25
    zgamma3 = (2.0 - zg3 * prmuz) * 0.25
    zgamma4 = 1.0 - zgamma3
    one_g = xp.where(zg != 1.0, 1.0 - zg, 1.0)
    r = zg / one_g
    denom = xp.where(zg != 1.0, 1.0 - (1.0 - zw) * (r * r), 1.0)
    ok = (zw > 0.0) & (denom != 0.0)
    zwo = xp.where(ok, zw / xp.where(ok, denom, 1.0), 0.0)
    cons = zwo >= zwcrit

    # conservative scattering
    za = zgamma1 * prmuz
    za1 = za - zgamma3
    zgt = zgamma1 * zto1
    ze1 = xp.clip(zto1 / prmuz, None, 500.0)
    ze2 = _expo(bk, T, ze1)
    c_ref = (zgt - za1 * (1.0 - ze2)) / (1.0 + zgt)
    c_tra = 1.0 - c_ref
    c_refd = zgt / (1.0 + zgt)
    c_trad = 1.0 - c_refd
    noext = ze2 == 1.0
    c_ref = xp.where(noext, 0.0, c_ref)
    c_tra = xp.where(noext, 1.0, c_tra)
    c_refd = xp.where(noext, 0.0, c_refd)
    c_trad = xp.where(noext, 1.0, c_trad)

    # non-conservative scattering (inputs made harmless where unused)
    g1 = xp.where(cons, 1.0, zgamma1)
    g2 = xp.where(cons, 0.5, zgamma2)
    za1 = zgamma1 * zgamma4 + zgamma2 * zgamma3
    za2 = zgamma1 * zgamma3 + zgamma2 * zgamma4
    za1 = xp.where(cons, 0.0, za1)
    za2 = xp.where(cons, 0.0, za2)
    zrk = xp.sqrt(g1 * g1 - g2 * g2)
    zrp = zrk * prmuz
    zrp1 = 1.0 + zrp
    zrm1 = 1.0 - zrp
    zrk2 = 2.0 * zrk
    zrpp = 1.0 - zrp * zrp
    zrkg = zrk + g1
    zr1 = zrm1 * (za2 + zrk * zgamma3)
    zr2 = zrp1 * (za2 - zrk * zgamma3)
    zr3 = zrk2 * (zgamma3 - za2 * prmuz)
    zr4 = zrpp * zrkg
    zr5 = zrpp * (zrk - g1)
    zt1 = zrp1 * (za1 + zrk * zgamma4)
    zt2 = zrm1 * (za1 - zrk * zgamma4)
    zt3 = zrk2 * (zgamma4 + za1 * prmuz)
    zt4 = zr4
    zt5 = zr5
    zbeta = (g1 - zrk) / zrkg
    ze1 = xp.clip(zrk * zto1, None, 500.0)
    ze2 = xp.clip(zto1 / prmuz, None, 500.0)
    zem1 = _expo(bk, T, ze1)
    zep1 = 1.0 / zem1
    zem2 = _expo(bk, T, ze2)
    zep2 = 1.0 / zem2
    zdenr = zr4 * zep1 + zr5 * zem1
    zdent = zt4 * zep1 + zt5 * zem1
    tiny = (zdenr >= -eps) & (zdenr <= eps)
    zdenr_s = xp.where(tiny, 1.0, zdenr)
    zdent_s = xp.where(tiny, 1.0, zdent)
    n_ref = xp.where(tiny, eps, zw * (zr1 * zep1 - zr2 * zem1 - zr3 * zem2) / zdenr_s)
    n_tra = xp.where(
        tiny, zem2, zem2 - zem2 * zw * (zt1 * zep1 - zt2 * zem1 - zt3 * zep2) / zdent_s
    )
    zemm = zem1 * zem1
    zdend = 1.0 / ((1.0 - zbeta * zemm) * zrkg)
    n_refd = g2 * (1.0 - zemm) * zdend
    n_trad = zrk2 * zem1 * zdend

    def out(c, n, clear):
        v = xp.where(cons, c, n)
        return v if lrtchk is True else xp.where(lrtchk, v, clear)

    return {
        "pref": out(c_ref, n_ref, 0.0),
        "ptra": out(c_tra, n_tra, 1.0),
        "prefd": out(c_refd, n_refd, 0.0),
        "ptrad": out(c_trad, n_trad, 1.0),
    }


# ---------------------------------------------------------------------------
# rrtmg_sw_vrtqdr
# ---------------------------------------------------------------------------
def vrtqdr_sw(bk, *, pref, prefd, ptra, ptrad, pdbt, ptdbt):
    """``vrtqdr_sw``: adding of the layers.  Inputs ``[klev+1, ...]``
    top-down (index klev = surface; pref/prefd there are the albedos,
    pdbt/ptra/ptrad unused), ``ptdbt[klev+1, ...]`` the direct-beam
    transmission above each level.  Returns pfd, pfu, prup, prupd, prdnd
    ``[klev+1, ...]`` (level 0 = top of the atmosphere)."""
    xp = bk.xp

    def up(c, x):
        rup_n, rupd_n = c
        rf, rfd, tr, trd, db = x
        zreflect = 1.0 / (1.0 - rupd_n * rfd)
        rup = rf + (trd * ((tr - db) * rupd_n + db * rup_n)) * zreflect
        rupd = rfd + trd * trd * rupd_n * zreflect
        return (rup, rupd), (rup, rupd)

    (_, _), (rup, rupd) = bk.scan(
        up, (pref[-1], prefd[-1]), (pref[:-1], prefd[:-1], ptra[:-1], ptrad[:-1], pdbt[:-1]),
        reverse=True,
    )  # fmt: skip
    prup = xp.concat([rup, pref[-1:]], axis=0)
    prupd = xp.concat([rupd, prefd[-1:]], axis=0)

    def down(c, x):
        tdn, rdnd = c
        rf, rfd, tr, trd, tdb = x
        zreflect = 1.0 / (1.0 - rfd * rdnd)
        tdn1 = tdb * tr + (trd * ((tdn - tdb) + tdb * rf * rdnd)) * zreflect
        rdnd1 = rfd + trd * trd * rdnd * zreflect
        return (tdn1, rdnd1), (tdn1, rdnd1)

    one = xp.ones_like(ptra[0])
    _, (tdn, rdnd) = bk.scan(
        down, (ptra[0], prefd[0]),
        (pref[1:-1], prefd[1:-1], ptra[1:-1], ptrad[1:-1], ptdbt[1:-1]),
    )  # fmt: skip
    ztdn = xp.concat([one[None], ptra[:1], tdn], axis=0)
    prdnd = xp.concat([0.0 * one[None], prefd[:1], rdnd], axis=0)
    zreflect = 1.0 / (1.0 - prdnd * prupd)
    pfu = (ptdbt * prup + (ztdn - ptdbt) * prupd) * zreflect
    pfd = ptdbt + (ztdn - ptdbt + ptdbt * prup * prdnd) * zreflect
    return {"pfd": pfd, "pfu": pfu, "prup": prup, "prupd": prupd, "prdnd": prdnd}


# ---------------------------------------------------------------------------
# rrtmg_sw_spcvmc
# ---------------------------------------------------------------------------
def _cumprod(bk, x):
    """[1, x0, x0*x1, ...] along axis 0, multiplied in the Fortran order."""
    xp = bk.xp

    def f(c, xi):
        n = xi * c
        return n, n

    one = xp.ones_like(x[0])
    _, r = bk.scan(f, one, x)
    return xp.concat([one[None], r], axis=0)


def spcvmc_sw(
    bk, *, palbd, palbp, pcldfmc, ptaucmc, pasycmc, pomgcmc, ptaormc, prmu0,
    adjflux, taumol, T=None,
):  # fmt: skip
    """``spcvmc_sw`` (istart=16, iend=29, icpr=1, iout=0, zero aerosols):
    two-stream McICA shortwave transfer.  Cloud arrays ``[L, C, 112]``
    (bottom-up, after cldprmc_sw), albedos ``[C, 14]``, ``prmu0[C]``,
    ``taumol`` = taumol_sw output.  Returns pbbfd, pbbfu, pbbcd, pbbcu,
    pbbfddir, pbbcddir ``[L+1, C]`` (level 0 = surface)."""
    xp = bk.xp
    T = tables(bk) if T is None else T
    bidx = xp.asarray(_bandidx(T))
    flip = lambda a: xp.flip(a, axis=0)
    mu = prmu0[None, :, None]
    ztaur, ztaug = flip(taumol["taur"]), flip(taumol["taug"])
    cldf = flip(pcldfmc)
    taucmc, omgcmc, asycmc = flip(ptaucmc), flip(pomgcmc), flip(pasycmc)
    taormc = flip(ptaormc)
    ptaua, pomga, pasya = 0.0, 1.0, 0.0
    zincflx = adjflux * taumol["sfluxzen"] * prmu0[:, None]  # [C, G]

    ztauc = ztaur + ztaug + ptaua
    zomcc = ztaur * 1.0 + ptaua * pomga
    zgcc = pasya * pomga * ptaua / zomcc
    zomcc = zomcc / ztauc
    zclear = 1.0 - cldf
    zcloud = cldf
    zdbtmc = _expo(bk, T, ztauc / mu)
    ztdbtc_nodel = _cumprod(bk, zdbtmc)
    tauorig = ztauc + taormc
    zdbtmo = _expo(bk, T, tauorig / mu)
    zdbt_nodel = zclear * zdbtmc + zcloud * zdbtmo
    ztdbt_nodel = _cumprod(bk, zdbt_nodel)
    zf = zgcc * zgcc
    zwf = zomcc * zf
    ztauc = (1.0 - zwf) * ztauc
    zomcc = (zomcc - zwf) / (1.0 - zwf)
    zgcc = (zgcc - zf) / (1.0 - zf)
    # icpr = 1
    ztauo = ztauc + taucmc
    zomco = ztauc * zomcc + taucmc * omgcmc
    zgco = (taucmc * omgcmc * asycmc + ztauc * zomcc * zgcc) / zomco
    zomco = zomco / ztauo

    clr = reftra_sw(bk, lrtchk=True, pgg=zgcc, prmuz=mu, ptau=ztauc, pw=zomcc, T=T)
    cld = reftra_sw(
        bk, lrtchk=cldf > 1.0e-12, pgg=zgco, prmuz=mu, ptau=ztauo, pw=zomco, T=T
    )
    zref = zclear * clr["pref"] + zcloud * cld["pref"]
    zrefd = zclear * clr["prefd"] + zcloud * cld["prefd"]
    ztra = zclear * clr["ptra"] + zcloud * cld["ptra"]
    ztrad = zclear * clr["ptrad"] + zcloud * cld["ptrad"]
    zdbtc = _expo(bk, T, ztauc / mu)
    ztdbtc = _cumprod(bk, zdbtc)
    zdbt = zclear * _expo(bk, T, ztauc / mu) + zcloud * _expo(bk, T, ztauo / mu)
    ztdbt = _cumprod(bk, zdbt)

    albp = xp.take(palbp, bidx, axis=-1)[None]  # [1, C, G]
    albd = xp.take(palbd, bidx, axis=-1)[None]
    zero = 0.0 * albp

    def sfc(x, v):
        return xp.concat([x, v], axis=0)

    qc = vrtqdr_sw(
        bk, pref=sfc(clr["pref"], albp), prefd=sfc(clr["prefd"], albd),
        ptra=sfc(clr["ptra"], zero), ptrad=sfc(clr["ptrad"], zero),
        pdbt=sfc(zdbtc, zero), ptdbt=ztdbtc,
    )  # fmt: skip
    qf = vrtqdr_sw(
        bk, pref=sfc(zref, albp), prefd=sfc(zrefd, albd), ptra=sfc(ztra, zero),
        ptrad=sfc(ztrad, zero), pdbt=sfc(zdbt, zero), ptdbt=ztdbt,
    )  # fmt: skip

    # sum over g-points in order; levels back to bottom-up
    def acc(c, x):
        inc, fu, fd, cu, cd, fdd, cdd = x
        return tuple(
            ci + inc * xi for ci, xi in zip(c, (fu, fd, cu, cd, fdd, cdd))
        ), None

    def g_first(a):
        return xp.permute_dims(flip(a), (2, 0, 1))  # [G, L+1, C]

    xs = (
        xp.permute_dims(zincflx, (1, 0))[:, None, :],
        g_first(qf["pfu"]),
        g_first(qf["pfd"]),
        g_first(qc["pfu"]),
        g_first(qc["pfd"]),
        g_first(ztdbt_nodel),
        g_first(ztdbtc_nodel),
    )
    z = xp.zeros_like(xs[1][0])
    sums, _ = bk.scan(acc, (z,) * 6, xs)
    names = ("pbbfu", "pbbfd", "pbbcu", "pbbcd", "pbbfddir", "pbbcddir")
    return dict(zip(names, sums))


# ---------------------------------------------------------------------------
# rrtmg_sw_rad: rrtmg_sw
# ---------------------------------------------------------------------------
def rrtmg_sw(
    bk, *, play, plev, tlay, tlev, tsfc, h2ovmr, o3vmr, co2vmr, ch4vmr, n2ovmr,
    o2vmr, asdir, asdif, aldir, aldif, coszen, adjes, scon, cldfmcl, taucmcl,
    ssacmcl, asmcmcl, fsfcmcl, ciwpmcl, clwpmcl, cswpmcl, reicmcl, relqmcl,
    resnmcl, T=None, trace=None,
):  # fmt: skip
    """``rrtmg_sw`` (dyofyr=0, aer_opt=0, icpr=1): inatm_sw -> cldprmc_sw
    -> setcoef_sw -> spcvmc_sw (taumol_sw, reftra_sw, vrtqdr_sw) for all
    columns.  Returns swuflx, swdflx, swuflxc, swdflxc, swdkdir,
    swdkdif, swdkdirc ``[L+1, C]`` and swhr, swhrc ``[L, C]``."""
    xp = bk.xp
    T = tables(bk) if T is None else T
    a = inatm_sw(
        bk, play=play, plev=plev, tlay=tlay, tlev=tlev, tsfc=tsfc, h2ovmr=h2ovmr,
        o3vmr=o3vmr, co2vmr=co2vmr, ch4vmr=ch4vmr, n2ovmr=n2ovmr, o2vmr=o2vmr,
        adjes=adjes, scon=scon, cldfmcl=cldfmcl, taucmcl=taucmcl, ssacmcl=ssacmcl,
        asmcmcl=asmcmcl, fsfcmcl=fsfcmcl, ciwpmcl=ciwpmcl, clwpmcl=clwpmcl,
        cswpmcl=cswpmcl, reicmcl=reicmcl, relqmcl=relqmcl, resnmcl=resnmcl, T=T,
    )  # fmt: skip
    cp = cldprmc_sw(
        bk, cldfmc=a["cldfmc"], ciwpmc=a["ciwpmc"], clwpmc=a["clwpmc"],
        cswpmc=a["cswpmc"], reicmc=a["reicmc"], relqmc=a["relqmc"],
        taucmc=a["taucmc"], ssacmc=a["ssacmc"], asmcmc=a["asmcmc"],
        fsfcmc=a["fsfcmc"], T=T,
    )  # fmt: skip
    coef = setcoef_sw(
        bk, pavel=a["pavel"], tavel=a["tavel"], coldry=a["coldry"], wkl=a["wkl"], T=T
    )
    tau = taumol_sw(bk, coef=coef, T=T)
    cossza = xp.where(coszen <= 1.0e-10, 1.0e-10, coszen)
    # albedos: bands 16..24 and 29 near-IR (aldir/aldif), 25..28 UV/visible
    vis = np.zeros(nbndsw, dtype=bool)
    vis[9:13] = True
    vis = xp.asarray(vis)
    albdir = xp.where(vis, asdir[:, None], aldir[:, None])
    albdif = xp.where(vis, asdif[:, None], aldif[:, None])
    r = spcvmc_sw(
        bk, palbd=albdif, palbp=albdir, pcldfmc=a["cldfmc"], ptaucmc=cp["taucmc"],
        pasycmc=cp["asmcmc"], pomgcmc=cp["ssacmc"], ptaormc=cp["taormc"],
        prmu0=cossza, adjflux=a["adjflux"], taumol=tau, T=T,
    )  # fmt: skip
    swnflxc = r["pbbcd"] - r["pbbcu"]
    swnflx = r["pbbfd"] - r["pbbfu"]
    zdpgcp = T["heatfac"] / a["pdp"]
    swhrc = (swnflxc[1:] - swnflxc[:-1]) * zdpgcp
    swhr = (swnflx[1:] - swnflx[:-1]) * zdpgcp
    zl = xp.zeros_like(swhr[-1:])
    swhrc = xp.concat([swhrc[:-1], zl], axis=0)
    swhr = xp.concat([swhr[:-1], zl], axis=0)
    if trace is not None:
        trace.update(inatm=a, cldprmc=cp, setcoef=coef, taumol=tau, spcvmc=r)
    return {
        "swuflx": r["pbbfu"],
        "swdflx": r["pbbfd"],
        "swuflxc": r["pbbcu"],
        "swdflxc": r["pbbcd"],
        "swdkdir": r["pbbfddir"],
        "swdkdif": r["pbbfd"] - r["pbbfddir"],
        "swdkdirc": r["pbbcddir"],
        "swhr": swhr,
        "swhrc": swhrc,
    }


# ---------------------------------------------------------------------------
# module_ra_rrtmg_sw: RRTMG_SWRAD
# ---------------------------------------------------------------------------
def RRTMG_SWRAD(
    bk, *, coszen, solcon, albedo, t3d, t8w, tsk, p3d, p8w, dz8w, cldfra3d,
    xland, xice, snow, qv3d, qc3d, qr3d, yr, g=9.81, swtenc=None, gsw=None,
    T=None, trace=None,
):  # fmt: skip
    """``RRTMG_SWRAD`` as CM1's radopt=2 calls it for warm-rain (ptype=1)
    microphysics (f_qv, f_qc, f_qr; cloud/rain below 273.15 K become
    ice/snow), icloud=1, cldovrlp=2, has_reqc=has_reqi=has_reqs=0,
    o3input=0, aer_opt=0, sf_surface_physics=1.  One extra layer above the
    model top (nlay = nk + 1).

    Layout: ``[nk, C]`` layers, ``[nk+1, C]`` levels (p8w, t8w), ``[C]``
    surface arrays; ``coszen`` from calc_coszen.  Where coszen <= 0 the
    Fortran skips the column: rthratensw and the TOA/surface fluxes are 0
    and swtenc, gsw keep their previous values (``swtenc``, ``gsw``
    arguments, default 0).  Returns rthratensw, swtenc ``[nk, C]``, gsw,
    swcf, swupt, swuptc, swdnt, swdntc, swupb, swupbc, swdnb, swdnbc,
    swddir, swddni, swddif, swdownc, swddirc, swddnic ``[C]``, coszr and
    co2out."""
    xp = bk.xp
    T = tables(bk) if T is None else T
    nk = p3d.shape[0]
    co2 = (280.0 + 90.0 * np.exp(0.02 * (yr - 2000))) * 1.0e-6
    ch4, n2o, o2 = 1774.0e-9, 319.0e-9, 0.209488
    amdw, amdo = 1.607793, 0.603461
    day = coszen > 0.0
    pw1d = p8w * 0.01
    tw1d = t8w
    qv1d = xp.clip(qv3d, 0.0, None)
    t1d = t3d
    p1d = p3d / 100.0
    qc1d = xp.clip(qc3d, 0.0, None)
    qr1d = xp.clip(qr3d, 0.0, None)
    cold = t1d < 273.15
    qi1d = xp.where(cold, qc1d, 0.0)
    qs1d = xp.where(cold, qr1d, 0.0)
    qc1d = xp.where(cold, 0.0, qc1d)
    qv1d = xp.clip(qv1d, 1.0e-12, None)

    plev = xp.concat([pw1d, xp.full_like(pw1d[:1], 1.0e-5)], axis=0)  # nk+2
    play = xp.concat([p1d, 0.5 * pw1d[nk:]], axis=0)  # nk+1
    tlev = xp.concat([tw1d, tw1d[nk:] + 0.0], axis=0)
    tlay = xp.concat([t1d, tw1d[nk:] + 0.0], axis=0)
    pdel = plev[:nk] - plev[1 : nk + 1]
    h2o = qv1d * amdw
    h2ovmr = xp.concat([h2o, h2o[-1:]], axis=0)
    ones = xp.ones_like(play)
    o3vmr = lw.inirad(bk, plev=plev[: nk + 1]) * amdo
    asdir = asdif = aldir = aldif = albedo

    cldfrac = cldfra3d
    landfrac = 2.0 - xland
    gicewp = (qi1d + qs1d) * pdel * 100.0 / g * 1000.0
    gliqwp = qc1d * pdel * 100.0 / g * 1000.0
    cicewp = gicewp / xp.clip(cldfrac, 0.01, None)
    cliqwp = gliqwp / xp.clip(cldfrac, 0.01, None)
    tl = tlay[:nk]
    reliq = lw.relcalc(
        bk, t=tl, landfrac=landfrac, landm=landfrac, icefrac=xice, snowh=0.001 * snow
    )
    reice = lw.reicalc(bk, t=tl)
    reice = xp.clip(reice * 1.0315, None, 140.0)

    def pad(x, v):
        return xp.concat([x, xp.full_like(x[:1], v)], axis=0)

    zb = xp.zeros(tuple(play.shape) + (nbndsw,), dtype=play.dtype)
    mc = mcica_subcol_sw(
        bk, permuteseed=1, play=play, cldfrac=pad(cldfrac, 0.0), ciwp=pad(cicewp, 0.0),
        clwp=pad(cliqwp, 0.0), cswp=xp.zeros_like(play), rei=pad(reice, 10.0),
        rel=pad(reliq, 10.0), res=10.0 * ones, tauc=zb, ssac=zb + 1.0, asmc=zb,
        fsfc=zb, T=T,
    )  # fmt: skip
    r = rrtmg_sw(
        bk, play=play, plev=plev, tlay=tlay, tlev=tlev, tsfc=tsk, h2ovmr=h2ovmr,
        o3vmr=o3vmr, co2vmr=co2 * ones, ch4vmr=ch4 * ones, n2ovmr=n2o * ones,
        o2vmr=o2 * ones, asdir=asdir, asdif=asdif, aldir=aldir, aldif=aldif,
        coszen=coszen, adjes=1.0, scon=solcon, cldfmcl=mc["cldfmcl"],
        taucmcl=mc["taucmcl"], ssacmcl=mc["ssacmcl"], asmcmcl=mc["asmcmcl"],
        fsfcmcl=mc["fsfcmcl"], ciwpmcl=mc["ciwpmcl"], clwpmcl=mc["clwpmcl"],
        cswpmcl=mc["cswpmcl"], reicmcl=mc["reicmcl"], relqmcl=mc["relqmcl"],
        resnmcl=mc["resnmcl"], T=T, trace=trace,
    )  # fmt: skip
    if trace is not None:
        trace.update(
            play=play, plev=plev, tlay=tlay, tlev=tlev, o3vmr=o3vmr, h2ovmr=h2ovmr,
            cicewp=cicewp, cliqwp=cliqwp, reliq=reliq, reice=reice, mcica=mc,
        )  # fmt: skip
    uf, df, ufc, dfc = r["swuflx"], r["swdflx"], r["swuflxc"], r["swdflxc"]
    top = nk + 1
    zc = xp.zeros_like(coszen)
    swtenc = xp.zeros_like(t3d) if swtenc is None else swtenc
    gsw = zc if gsw is None else gsw
    safe_mu = xp.where(day, coszen, 1.0)

    def dsel(v, night=zc):
        return xp.where(day, v, night)

    swddir = dsel(r["swdkdir"][0])
    swddirc = dsel(r["swdkdirc"][0])
    return {
        "rthratensw": xp.where(day, r["swhr"][:nk] / 86400.0, 0.0),
        "swtenc": xp.where(day, r["swhrc"][:nk] / 86400.0, swtenc),
        "gsw": dsel(df[0] - uf[0], gsw),
        "swcf": dsel((df[top] - uf[top]) - (dfc[top] - ufc[top])),
        "swupt": dsel(uf[top]),
        "swuptc": dsel(ufc[top]),
        "swdnt": dsel(df[top]),
        "swdntc": dsel(dfc[top]),
        "swupb": dsel(uf[0]),
        "swupbc": dsel(ufc[0]),
        "swdnb": dsel(df[0]),
        "swdnbc": dsel(dfc[0]),
        "swddir": swddir,
        "swddni": dsel(swddir / safe_mu),
        "swddif": dsel(r["swdkdif"][0]),
        "swdownc": dsel(dfc[0]),
        "swddirc": swddirc,
        "swddnic": dsel(swddirc / safe_mu),
        "coszr": coszen,
        "co2out": co2,
    }
