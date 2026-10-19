#!/usr/bin/env python3
"""Write cxx/src/cm1k/share/cm1k_namelist_groups.inc: every namelist group
CM1 reads and its variables, from src/input.F and src/param.F.

    cxx/scripts/gen_namelist_groups.py          # rewrite the file
    cxx/scripts/gen_namelist_groups.py --check  # exit 1 if it is stale
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "cxx/src/cm1k/share/cm1k_namelist_groups.inc"


def groups():
    out = {}
    for fname in ("input.F", "param.F"):
        text = (ROOT / "src" / fname).read_text()
        text = "\n".join(line.split("!")[0] for line in text.splitlines())
        text = re.sub(r"&\s*\n", " ", text)
        for m in re.finditer(r"namelist\s*/\s*(\w+)\s*/([^\n]*)", text, re.I):
            out[m.group(1).lower()] = [
                n.lower() for n in re.split(r"[,\s]+", m.group(2)) if n
            ]
    return out


def render():
    lines = [
        "// Generated from src/input.F and src/param.F by cxx/scripts/gen_namelist_groups.py;",
        "// do not edit. One {group, variable} pair per namelist variable CM1 reads.",
    ]
    for g, names in groups().items():
        for n in names:
            lines.append(f'{{"{g}", "{n}"}},')
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    text = render()
    if "--check" in sys.argv[1:]:
        if OUT.read_text() != text:
            sys.exit(f"{OUT} is stale: run {sys.argv[0]}")
    else:
        OUT.write_text(text)
