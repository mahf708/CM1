import pytest

from conftest import requires_ext

pytestmark = requires_ext

NML = """
 &param0
 nx = 16, ny = 8,   nz = 10,   ! comment
 /
 &param1
 dx = 2000.0, dtl = 6.0d0, timax = 1.2e2,
 /
 &param2
 terrain_flag = .false., procfiles = T,
 output_path = './out/', output_basename = 'cm1out',
 zs = 3*0.0, 1.5,
 /
"""


def test_from_string_get():
    import cm1k

    nl = cm1k.Namelist.from_string(NML)
    assert nl["nx"] == 16 and isinstance(nl["nx"], int)
    assert nl.get("dtl") == 6.0 and nl["timax"] == 120.0
    assert nl["terrain_flag"] is False and nl["procfiles"] is True
    assert nl["output_path"] == "./out/"
    assert nl["zs"] == [0.0, 0.0, 0.0, 1.5]
    assert nl.get("missing") is None and nl.get("missing", 3) == 3
    assert "NX" in nl and "missing" not in nl
    assert nl.group_of("dx") == "param1"
    assert nl.geti("nz", 0) == 10 and nl.getr("dx", 0.0) == 2000.0
    assert nl.getb("terrain_flag", True) is False
    with pytest.raises(KeyError):
        nl["missing"]
    d = nl.to_dict()
    assert d["ny"] == 8


def test_set_and_copy():
    import cm1k

    nl = cm1k.Namelist.from_string(NML)
    c = nl.copy()
    c["dtl"] = 3.0
    c["nx"] = 32
    c["terrain_flag"] = True
    c["output_basename"] = "it's"
    c["zs"] = [1, 2]
    c.set_raw("timax", "60.0d0")
    assert c["dtl"] == 3.0 and c.getr("dtl", 0) == 3.0
    assert c["nx"] == 32 and c["terrain_flag"] is True
    assert c["output_basename"] == "it''s"   # Fortran-escaped quote is kept as written
    assert c["zs"] == [1, 2]
    assert c["timax"] == 60.0
    assert nl["dtl"] == 6.0 and nl["nx"] == 16    # original untouched
    with pytest.raises(TypeError):
        c["nx"] = object()


def test_from_file(cases_dir):
    import cm1k

    nl = cm1k.Namelist.from_file(str(cases_dir / "dry_bubble" / "namelist.input"))
    assert nl["nx"] > 0 and nl["dtl"] > 0
    with pytest.raises(RuntimeError, match="cannot open"):
        cm1k.Namelist.from_file("/nonexistent/namelist.input")


def test_bad_typed_get():
    import cm1k

    nl = cm1k.Namelist.from_string(NML)
    with pytest.raises(RuntimeError, match="not an integer"):
        nl.geti("output_path", 0)
