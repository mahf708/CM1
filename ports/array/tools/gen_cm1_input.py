"""Generate ``cm1x/cm1_input.py`` from CM1's ``src/input.F`` (+ param.F).

    python ports/array/tools/gen_cm1_input.py > ports/array/src/cm1x/cm1_input.py
    ruff format ports/array/src/cm1x/cm1_input.py

``tests/test_namelist.py`` checks that the committed file matches the source.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _join(text: str) -> str:
    text = "\n".join(line.split("!")[0] for line in text.splitlines())
    return re.sub(r"&\s*\n", " ", text)


def namelist_groups(src: Path = ROOT / "src") -> dict:
    groups = {}
    for fname in ("input.F", "param.F"):
        text = _join((src / fname).read_text())
        for m in re.finditer(r"namelist\s*/\s*(\w+)\s*/([^\n]*)", text, re.I):
            names = [n.lower() for n in re.split(r"[,\s]+", m.group(2)) if n]
            groups[m.group(1).lower()] = tuple(names)
    return groups


def module_vars(src: Path = ROOT / "src") -> tuple:
    text = _join((src / "input.F").read_text())
    text = text.split("CONTAINS")[0]
    names = set()
    decl = re.compile(
        r"^\s*(?:integer|real|logical|double precision|character)\b[^:\n]*?(?:::)?\s*(.*)$",
        re.I | re.M,
    )
    for m in decl.finditer(text):
        body = re.sub(r"\([^)]*\)", "", m.group(1)).lstrip(": ")
        for item in body.split(","):
            n = item.split("=")[0].strip().lower()
            if re.fullmatch(r"[a-z_]\w*", n):
                names.add(n)
    return tuple(sorted(names))


def render() -> str:
    g = namelist_groups()
    out = [
        '"""CM1\'s namelist groups and module ``input`` variables.',
        "",
        "Generated from src/input.F and src/param.F by",
        "ports/array/tools/gen_cm1_input.py; do not edit by hand.",
        '"""',
        "",
        "NAMELIST_GROUPS = {",
    ]
    for k, names in g.items():
        out.append(f"    {k!r}: (")
        for n in names:
            out.append(f"        {n!r},")
        out.append("    ),")
    out.append("}")
    out.append("")
    out.append("MODULE_VARS = frozenset(")
    out.append("    (")
    for n in module_vars():
        out.append(f"        {n!r},")
    out.append("    )")
    out.append(")")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    sys.stdout.write(render())
