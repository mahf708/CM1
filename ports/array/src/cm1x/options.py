"""Whole-model option checks: settings that no single routine sees, so a
routine-level check cannot catch them, and that the Model would otherwise
run as something different from CM1.  Each raises NotImplementedError."""

from __future__ import annotations


def check_unported(cfg) -> None:
    """Raise for namelist settings the Model does not implement."""
    bad = []
    if int(cfg.irst) != 0:
        bad.append(f"irst={cfg.irst} (restarts)")
    if bool(cfg.do_adapt_move):
        bad.append("do_adapt_move=.true.")
    if int(cfg.adapt_dt) != 0:
        bad.append(f"adapt_dt={cfg.adapt_dt}")
    if int(cfg.cm1setup) not in (0, 1):
        bad.append(f"cm1setup={cfg.cm1setup} (only 0, 1)")
    if int(cfg.cm1setup) == 1 and int(cfg.sgsmodel) not in (0, 1):
        bad.append(f"sgsmodel={cfg.sgsmodel} (only 0, 1)")
    if int(cfg.ipbl) != 0:
        bad.append(f"ipbl={cfg.ipbl}")
    if bool(cfg.terrain_flag) and int(cfg.sfcmodel) >= 1:
        # the surface layer's za (height of the lowest level above ground)
        # and stress terms have terrain branches that are not ported
        bad.append(f"sfcmodel={cfg.sfcmodel} with terrain")
    if int(cfg.iptra) != 0:
        bad.append(f"iptra={cfg.iptra} (passive tracers)")
    if int(cfg.cuparam) != 0:
        bad.append(f"cuparam={cfg.cuparam}")
    if bad:
        raise NotImplementedError("Model: options not ported: " + ", ".join(bad))
