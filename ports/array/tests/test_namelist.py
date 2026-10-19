"""Namelist strictness: names CM1 does not know are errors, never 0."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from cm1x import cm1_input
from cm1x.model import Model
from cm1x.namelist import Config, parse_namelist, read_namelist

ROOT = Path(__file__).resolve().parents[3]
REF = Path(__file__).resolve().parents[2] / "reference"
NML = REF / "kessler_periodic" / "namelist.input"


def _gen():
    spec = importlib.util.spec_from_file_location(
        "gen_cm1_input",
        Path(__file__).resolve().parents[1] / "tools" / "gen_cm1_input.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not (ROOT / "src" / "input.F").exists(), reason="no CM1 source")
def test_cm1_input_matches_source():
    """cm1x/cm1_input.py is generated from src/input.F and src/param.F."""
    gen = _gen()
    assert cm1_input.NAMELIST_GROUPS == gen.namelist_groups()
    assert set(cm1_input.MODULE_VARS) == set(gen.module_vars())


def test_every_shipped_namelist_parses():
    files = sorted(ROOT.glob("run/config_files/*/namelist.input"))
    files += sorted(REF.glob("*/namelist.input"))
    files += sorted(ROOT.glob("cxx/cases/*/namelist.input"))
    assert files
    for f in files:
        read_namelist(f)


def test_unknown_name_in_file_raises(tmp_path):
    text = NML.read_text().replace(" isnd ", " isnf ", 1)
    assert text != NML.read_text()
    p = tmp_path / "namelist.input"
    p.write_text(text)
    with pytest.raises(ValueError, match="isnf"):
        read_namelist(p)


def test_name_in_wrong_group_raises():
    with pytest.raises(ValueError, match="belongs to &param2"):
        parse_namelist("&param1\n dx = 1.0, isnd = 5,\n/\n")


def test_unknown_group_raises():
    with pytest.raises(ValueError, match="&param99"):
        parse_namelist("&param99\n dx = 1.0,\n/\n")


def test_array_element_names_parse():
    g = parse_namelist("&param9\n output_format = 2,\n/\n&param10\n stat_w = 1,\n/\n")
    assert g["param9"]["output_format"] == 2


def test_unknown_lookup_raises_known_defaults_to_zero():
    cfg = Config({"nx": 4})
    assert cfg.isnd == 0  # namelist variable not given: CM1's zeroed storage
    assert cfg.dolsw is False  # module input variable assigned in cm1.F
    assert cfg.alphobc == 60.0  # cm1.F default
    with pytest.raises(AttributeError, match="isnf"):
        _ = cfg.isnf


def test_unknown_override_raises():
    with pytest.raises(ValueError, match="nxx"):
        read_namelist(NML, overrides={"nxx": 8})
    read_namelist(NML, overrides={"nfall_max": 4})  # cm1x knob


@pytest.mark.parametrize(
    "ov, msg",
    [
        ({"irst": 1}, "irst"),
        ({"do_adapt_move": True}, "do_adapt_move"),
        ({"cm1setup": 2}, "cm1setup"),
        ({"cm1setup": 3}, "cm1setup"),
        ({"ipbl": 1}, "ipbl"),
        ({"sgsmodel": 2}, "sgsmodel"),
        ({"iptra": 1}, "iptra"),
    ],
)
def test_model_rejects_unported_settings(ov, msg):
    with pytest.raises(NotImplementedError, match=msg):
        Model(NML, overrides={"nx": 8, "ny": 8, **ov})


def test_set_znt_sets_roughness():
    from cm1x.backend import get_backend
    from cm1x.grid import Grid
    from cm1x.init_surface import init_surface

    bk = get_backend("numpy", "float64")
    cfg = read_namelist(NML, overrides={"sfcmodel": 1, "set_znt": 1, "cnst_znt": 0.16})
    sf = init_surface(bk, Grid(4, 4, 8, 1.0, 1.0, 1.0), cfg)
    assert float(sf["znt"].min()) == float(sf["znt"].max()) == 0.16
