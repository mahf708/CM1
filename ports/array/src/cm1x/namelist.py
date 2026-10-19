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
moisture-array indices for the Kessler scheme ``ptype=1``).

Like CM1 (gfortran's namelist read stops the model), a name that is not a
variable of its namelist group, or a group CM1 does not read, is an error,
and so is looking up a name that is neither a namelist variable nor a
variable of CM1's module ``input`` (``cm1_input.MODULE_VARS``) nor one of
cm1x's own knobs (``CM1X_KNOBS``): a misspelled option never reads as 0.
"""

from __future__ import annotations

import re
from pathlib import Path

from .cm1_input import MODULE_VARS, NAMELIST_GROUPS

# cm1x-only settings (not in CM1): accepted as overrides and lookups
CM1X_KNOBS = frozenset({"nfall_max", "satadj_maxiter"})

_GROUP_OF = {n: g for g, names in NAMELIST_GROUPS.items() for n in names}
KNOWN = frozenset(_GROUP_OF) | MODULE_VARS | CM1X_KNOBS

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
    # also assigned in cm1.F before param reads the namelist
    "cflmax": 0.0,
    "ksmax": 0.0,
    "nudgeobc": 0,
    "alphobc": 60.0,
    "top_cd": 0.0,
    "side_cd": 0.0,
    "ramp_sgs": 0,
    "ramp_time": 0.0,
    "t2p_avg": 1,
    "dotimeavg": False,
    "restart_format": 1,
    "restart_filetype": 1,
    "restart_reset_frqtim": True,
    "interp_on_restart": False,
    "run_time": -999.0,
    "hurr_rad": 1.0e30,
    "dolsw": False,
    "dohifrq": False,
    "iusetke": False,
    "do_adapt_move": False,
    "cuparam": 0,
    "idoles": False,
    "idopbl": False,
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
            if name.startswith("__") or name.lower() not in KNOWN:
                raise AttributeError(
                    f"{name!r} is not a CM1 namelist or module-input variable"
                ) from None
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
    # moisture indices (param.F, ptype = 1: Kessler, qv/qc/qr)
    if v.get("imoist", 0) == 1 and v.get("ptype", 0) == 1:
        v.update(numq=3, nqv=1, nqc=2, nqr=3, nql1=2, nql2=3, iice=0)
    elif v.get("imoist", 0) == 0:
        v.update(numq=0, nqv=0, nqc=0, nqr=0, iice=0)
    # param.F: with stretch_z = 2 the nominal dz (mh/mf normalization) is
    # ztop/nk
    if v.get("stretch_z", 0) == 2 and v.get("nz", 0):
        v["dz"] = float(v.get("ztop", 0.0)) / float(v["nz"])


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


def parse_namelist(text: str, strict: bool = True) -> dict:
    """Parse Fortran namelist text -> {group: {var: value}} (lower-case names).

    ``strict`` (default): groups and variables must be ones CM1 reads."""
    groups: dict = {}
    body = "\n".join(_strip_comment(line) for line in text.splitlines())
    for gm in re.finditer(r"&(\w+)(.*?)(?:^\s*/|/\s*$|&end)", body, re.S | re.M | re.I):
        gname = gm.group(1).lower()
        content = gm.group(2)
        if strict and gname not in NAMELIST_GROUPS:
            raise ValueError(f"namelist group &{gname} is not read by CM1")
        g = groups.setdefault(gname, {})
        # split on "name =" (optionally name(idx) =)
        parts = re.split(r"(?:^|,|\s)\s*([A-Za-z_]\w*(?:\([^)]*\))?)\s*=", content)
        # parts = [junk, name1, vals1, name2, vals2, ...]
        for name, vals in zip(parts[1::2], parts[2::2]):
            base = name.split("(")[0].lower()
            if strict and base not in NAMELIST_GROUPS[gname]:
                where = _GROUP_OF.get(base)
                hint = f" (it belongs to &{where})" if where else ""
                raise ValueError(f"&{gname}: unknown namelist variable {name!r}{hint}")
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
        if k.lower() not in KNOWN:
            raise ValueError(f"override {k!r} is not a CM1 namelist variable")
        flat[k.lower()] = v
    # extra input files (input_grid_z, ...) are read next to the namelist
    flat.setdefault("namelist_dir", str(Path(path).resolve().parent))
    return Config(flat, groups)
