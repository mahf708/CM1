"""Surface initialization, ported from src/init_surface.F (initsfc=1,
sfcmodel>=1, no ocean mixed layer): the namelist skin temperature and the
MM5/WRF land-use parameters of LANDUSE.TBL (``getlanduse``)."""

from __future__ import annotations

from pathlib import Path

from ._ftables import cm1_root


def _opt(cfg, name, default):
    get = getattr(cfg, "get", None)
    return get(name, default) if get else getattr(cfg, name, default)


def getlanduse(season, lu_index, path=None):
    """init_surface.F ``getlanduse`` for a uniform land-use category: albedo,
    emissivity, roughness length, thermal inertia, moisture availability and
    xland (2 for water, category 16, else 1) from LANDUSE.TBL."""
    path = cm1_root() / "run" / "LANDUSE.TBL" if path is None else path
    lines = Path(path).read_text().splitlines()
    pos, tab = 2, {}
    for isn in (1, 2):
        pos += 1  # season name
        for _ in range(33):
            f = lines[pos].split(",")
            pos += 1
            tab[(int(f[0]), isn)] = [float(x) for x in f[1:8]]
    iswater = 16
    is_ = iswater if int(lu_index) == 0 else int(lu_index)
    albd, slmo, sfem, sfz0, therin, _scfx, _sfhc = tab[(is_, int(season))]
    if albd <= 0.0:
        raise ValueError("getlanduse: category not found")
    return {
        "albd": albd / 100.0,
        "thc": therin / 100.0,
        "emiss": sfem,
        "znt": sfz0 / 100.0,
        "mavail": slmo,
        "xland": 2.0 if is_ == iswater else 1.0,
    }


def init_surface(bk, g, cfg):
    """The parts of init_surface.F that radiation and the surface layer read (sfcmodel >= 1,
    initsfc = 1): tsk = tsk0 on (ib:ie, jb:je), and albd, emiss, xland from
    getlanduse(season, lu0) (also the roughness length znt), plus xice =
    xsnow = 0.  2-D [j, i] arrays with halos."""
    if int(_opt(cfg, "sfcmodel", 0)) < 1:
        raise NotImplementedError("init_surface: needs sfcmodel>=1")
    if int(_opt(cfg, "initsfc", 1)) != 1:
        raise NotImplementedError("init_surface: only initsfc=1 is ported")
    if int(_opt(cfg, "oceanmodel", 1)) not in (0, 1):
        raise NotImplementedError("init_surface: oceanmodel not ported")
    lu = getlanduse(int(cfg.season), int(cfg.lu0))
    shp = g.s_shape[1:]
    z = bk.zeros(shp)
    return {
        "tsk": z + float(cfg.tsk0),
        "albd": z + lu["albd"],
        "emiss": z + lu["emiss"],
        "xland": z + lu["xland"],
        "xice": z,
        "xsnow": z,
        "znt": z + lu["znt"],
    }
