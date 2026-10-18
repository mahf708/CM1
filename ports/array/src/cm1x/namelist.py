"""Read a CM1 ``namelist.input`` into a config object.

CM1 keeps every namelist variable as a global in Fortran module ``input``; the
group names (``&param0`` ...) only matter for reading. ``Config`` therefore
flattens all groups into one case-insensitive attribute namespace
(``cfg.nx``, ``cfg.isnd``, ``cfg.dx`` ...). Variables not given in the file take
CM1's default: the values assigned in ``cm1.F`` before ``param`` reads the
namelist (``DEFAULTS`` below), otherwise 0 / ``.false.`` (gfortran's zeroed
module storage, which CM1 relies on).

A few derived settings that ``param.F`` computes right after reading are applied
too (``ni,nj,nk``, ``imove=0 => umove=vmove=0``, ``ngxy``, ``ngz``, and the
moisture-array tables of param.F for ptype=1 (Kessler), 3 (Thompson) and
5 (Morrison): numq, qname, qmag and the index variables).
"""

from __future__ import annotations

import re
from pathlib import Path

# assigned in cm1.F before the namelist is read (single-precision build)
DEFAULTS = {
    "outfile": 6,
    "outunits": 1,
    "smeps": 1.0e-30,
    "tsmall": 0.0001,
    "qsmall": 1.0e-14,
    "ndcnst": 0.0,
    "nt_c": 0.0,
    "csound": 0.0,
    "cstar": 0.0,
    "nodex": 1,
    "nodey": 1,
    "ngxy": 3,
    "ngz": 1,
    "terrain_flag": False,
    "procfiles": False,
}


class Config:
    """Flat, case-insensitive view of all namelist variables (plus derived ones)."""

    def __init__(self, values: dict | None = None, groups: dict | None = None):
        object.__setattr__(self, "_v", {})
        object.__setattr__(self, "groups", groups or {})
        for k, v in DEFAULTS.items():
            self._v[k] = v
        for k, v in (values or {}).items():
            self._v[k.lower()] = v
        _derive(self._v)

    def __getattr__(self, name):
        try:
            return self._v[name.lower()]
        except KeyError:
            if name.startswith("__"):
                raise AttributeError(name) from None
            return 0  # Fortran module default (zero-initialised storage)

    def __setattr__(self, name, value):
        self._v[name.lower()] = value

    def __getitem__(self, name):
        return getattr(self, name)

    def __contains__(self, name):
        return name.lower() in self._v

    def get(self, name, default=None):
        return self._v.get(name.lower(), default)

    def replace(self, **kw):
        """Return a copy with some variables changed (derived values recomputed)."""
        v = {k: x for k, x in self._v.items()}
        v.update({k.lower(): x for k, x in kw.items()})
        return Config(v, self.groups)

    def as_dict(self):
        return dict(self._v)

    def __repr__(self):
        return f"Config({len(self._v)} variables)"


def _derive(v: dict):
    v["ni"] = v.get("nx", 0)
    v["nj"] = v.get("ny", 0)
    v["nk"] = v.get("nz", 0)
    if v.get("imove", 0) == 0:
        v["umove"] = 0.0
        v["vmove"] = 0.0
    # moisture-array tables (param.F)
    if v.get("imoist", 0) == 1 and v.get("ptype", 0) in MOISTURE_TABLES:
        v.update(moisture_table(v["ptype"], int(v.get("inum", 1))))
    elif v.get("imoist", 0) == 0:
        v.update(numq=0, nqv=0, nqc=0, nqr=0, iice=0, idm=0, qname=(), qmag=(), nql1=0)


# param.F: the q arrays of each microphysics scheme (qname, qmag).  The
# index variables are derived from them as param.F does (nqc = the 'qc'
# entry, ...); nql1..nql2 / nqs1..nqs2 / nnc1..nnc2 are the liquid, solid
# and number-concentration ranges (1-based, 0 or -2 when absent, as CM1).
MOISTURE_TABLES = {
    # Kessler
    1: dict(
        qname=("qv", "qc", "qr"),
        qmag=(1.0e-2,) * 3,
        nql=(2, 3),
        nqs=(-2, -2),
        nnc=(0, 0),
        iice=0,
        idm=0,
    ),
    # Thompson
    3: dict(
        qname=("qv", "qc", "qr", "qi", "qs", "qg", "nci", "ncr"),
        qmag=(1.0e-2,) * 6 + (1.0e6,) * 2,
        nql=(2, 3),
        nqs=(4, 6),
        nnc=(7, 8),
        iice=1,
        idm=1,
    ),
    # Morrison (inum=1: constant droplet number; GRAUPEL_INIT hard-wires it)
    5: dict(
        qname=("qv", "qc", "qr", "qi", "qs", "qg", "nci", "ncs", "ncr", "ncg"),
        qmag=(1.0e-2,) * 6 + (1.0e6,) * 4,
        nql=(2, 3),
        nqs=(4, 6),
        nnc=(7, 10),
        iice=1,
        idm=1,
    ),
}
# Morrison with predicted droplet number (inum=0): 11 arrays, smaller qmag
_MORRISON_INUM0 = dict(
    MOISTURE_TABLES[5],
    qname=MOISTURE_TABLES[5]["qname"] + ("ncc",),
    qmag=(1.0e-2,) * 6 + (1.0e5,) * 5,
    nnc=(7, 11),
)
_INDEX_NAMES = {
    "qc": "nqc",
    "ql": "nqc",
    "qr": "nqr",
    "qi": "nqi",
    "qi1": "nqi",
    "qs": "nqs",
    "qg": "nqg",
    "nci": "nnci",
    "ncc": "nncc",
    "ncs": "nncs",
    "ncr": "nncr",
    "ncg": "nncg",
}


def moisture_table(ptype: int, inum: int = 1) -> dict:
    """param.F's moisture settings for ``ptype``: numq, qname, qmag, iice,
    idm, nqv, nql1/nql2, nqs1/nqs2, nnc1/nnc2 and the named indices (nqc,
    nqr, nqi, nqs, nqg, nnci, nncc and, for cm1x's convenience, nncs, nncr,
    nncg; 0 when absent).  ``inum`` (Morrison only) selects the
    predicted-droplet-number variant (inum=0); CM1's GRAUPEL_INIT always
    sets inum=1."""
    t = _MORRISON_INUM0 if (ptype == 5 and inum == 0) else MOISTURE_TABLES[ptype]
    out = dict(
        numq=len(t["qname"]),
        qname=t["qname"],
        qmag=t["qmag"],
        iice=t["iice"],
        idm=t["idm"],
        nqv=1,
        nql1=t["nql"][0],
        nql2=t["nql"][1],
        nqs1=t["nqs"][0],
        nqs2=t["nqs"][1],
        nnc1=t["nnc"][0],
        nnc2=t["nnc"][1],
    )
    if ptype == 5:
        out["inum"] = inum
    for name in set(_INDEX_NAMES.values()):
        out[name] = 0
    for n, q in enumerate(t["qname"], start=1):
        if q in _INDEX_NAMES:
            out[_INDEX_NAMES[q]] = n
    return out


# ---------------------------------------------------------------------------
# minimal Fortran namelist parser

_TOKEN = re.compile(
    r"""\s*(?:
        (?P<str>'(?:[^']|'')*'|"(?:[^"]|"")*")
      | (?P<other>[^,\s'"]+)
    )\s*,?""",
    re.X,
)


def _strip_comment(line: str) -> str:
    out, q = [], None
    for ch in line:
        if q:
            out.append(ch)
            if ch == q:
                q = None
        elif ch in "'\"":
            q = ch
            out.append(ch)
        elif ch == "!":
            break
        else:
            out.append(ch)
    return "".join(out)


def _value(tok: str):
    t = tok.strip()
    if t[0] in "'\"":
        return t[1:-1].replace(t[0] * 2, t[0])
    low = t.lower()
    if low in (".true.", "t", ".t.", "true"):
        return True
    if low in (".false.", "f", ".f.", "false"):
        return False
    try:
        return int(t)
    except ValueError:
        pass
    try:
        return float(low.replace("d", "e"))
    except ValueError:
        return t


def _parse_values(text: str):
    vals = []
    pos = 0
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m or m.end() == pos:
            break
        pos = m.end()
        tok = m.group("str") or m.group("other")
        if tok is None:
            continue
        # repeat count r*value
        rm = re.fullmatch(r"(\d+)\*(.+)", tok)
        if rm:
            vals.extend([_value(rm.group(2))] * int(rm.group(1)))
        else:
            vals.append(_value(tok))
    return vals


def parse_namelist(text: str) -> dict:
    """Parse Fortran namelist text -> {group: {var: value}} (lower-case names)."""
    groups: dict = {}
    body = "\n".join(_strip_comment(line) for line in text.splitlines())
    for gm in re.finditer(r"&(\w+)(.*?)(?:^\s*/|/\s*$|&end)", body, re.S | re.M | re.I):
        gname = gm.group(1).lower()
        content = gm.group(2)
        g = groups.setdefault(gname, {})
        # split on "name =" (optionally name(idx) =)
        parts = re.split(r"(?:^|,|\s)\s*([A-Za-z_]\w*(?:\([^)]*\))?)\s*=", content)
        # parts = [junk, name1, vals1, name2, vals2, ...]
        for name, vals in zip(parts[1::2], parts[2::2]):
            v = _parse_values(vals.strip().rstrip(","))
            g[name.lower()] = v[0] if len(v) == 1 else v
    return groups


# cm1.F's values when compiled with -DDP (double precision)
DP_DEFAULTS = {"smeps": 1.0e-60, "tsmall": 1.0e-12, "qsmall": 1.0e-24}


def read_namelist(path, dp: bool = False, overrides: dict | None = None) -> Config:
    """Read a CM1 namelist.input file into a ``Config``.

    ``dp=True`` uses the thresholds of a ``-DDP`` CM1 build (see cm1.F).
    ``overrides`` ({name: value}, e.g. ``{"nx": 80, "ny": 80}``) replace
    namelist values before the derived settings are computed.
    """
    groups = parse_namelist(Path(path).read_text())
    flat = dict(DP_DEFAULTS) if dp else {}
    for g in groups.values():
        flat.update(g)
    for k, v in (overrides or {}).items():
        flat[k.lower()] = v
    return Config(flat, groups)
