"""cm1k.dump and cm1k.field: pure numpy, no extension needed."""
import numpy as np
import pytest

from cm1k import dump
from cm1k.field import FieldView


def _fields(rng):
    return {
        "ua": FieldView("ua", np.asfortranarray(rng.standard_normal((7, 6, 5))), (-2, -2, 0)),
        "zh_like": (rng.standard_normal((4, 3, 2)), (1, 1, 1)),
        "xh": rng.standard_normal(9),
    }


def test_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    f = _fields(rng)
    d = dump.write(dump.step_dir(tmp_path, 12), f, nstep=12, mtime=72.5)
    assert d.endswith("step_000012")
    r = dump.Dump(tmp_path, nstep=12)
    assert (r.nstep, r.mtime) == (12, 72.5)
    assert set(r.field_names()) == {"ua", "zh_like", "xh"}
    ua = r["ua"]
    assert ua.lbounds == (-2, -2, 0) and ua.ubounds == (4, 3, 4)
    np.testing.assert_array_equal(ua.data, f["ua"].data)
    np.testing.assert_array_equal(r["zh_like"].data, f["zh_like"][0])
    assert r["xh"].lbounds == (1,)
    assert dump.list_steps(tmp_path) == [12]
    # raw layout: Fortran order on disk
    raw = np.fromfile(tmp_path / "step_000012" / "ua.bin", dtype="<f8")
    np.testing.assert_array_equal(raw, f["ua"].data.ravel(order="F"))


def test_index_format_matches_fortran(tmp_path):
    dump.write(tmp_path, {"xh": (np.zeros(38), (-2,))}, nstep=0, mtime=0.0)
    lines = (tmp_path / "index.txt").read_text().splitlines()
    # src/cm1dump.F: '(a,1x,i10,1x,es24.16)' and '(a,1x,i2,1x,i1,2(1x,i6))'
    assert lines[0] == "#step_time          0   0.0000000000000000E+00"
    assert lines[1] == "xh  8 1     -2     35"


def test_float32_and_errors(tmp_path):
    (tmp_path / "index.txt").write_text("#step_time 3 1.5\na 4 2 1 2 0 2\nb 8 1 1 4\n")
    np.arange(6, dtype="<f4").tofile(tmp_path / "a.bin")
    np.arange(3, dtype="<f8").tofile(tmp_path / "b.bin")   # short file
    r = dump.read(tmp_path)
    a = r["a"]
    assert a.dtype == np.float64 and a.shape == (2, 3)
    assert a.at(2, 0) == 1.0 and a.at(1, 2) == 4.0
    with pytest.raises(ValueError, match="expected 4 values"):
        r["b"]
    with pytest.raises(KeyError):
        r["nope"]


def test_compare(tmp_path):
    rng = np.random.default_rng(1)
    f = _fields(rng)
    dump.write(tmp_path / "a", f)
    g = {k: (v.copy() if isinstance(v, FieldView) else v) for k, v in f.items()}
    g["ua"].f[1, 2, 3] += 1e-3
    dump.write(tmp_path / "b", g)
    st = dump.compare(tmp_path / "b", tmp_path / "a")
    assert st["xh"].max_abs == 0 and st["zh_like"].max_abs == 0
    assert st["ua"].where == (1, 2, 3)
    assert st["ua"].max_abs == pytest.approx(1e-3)
    assert st["ua"].rel == pytest.approx(1e-3 / np.abs(f["ua"].data).max())
    # NaNs count as infinitely different
    g["ua"].f[0, 0, 0] = np.nan
    assert dump.diff_fields(g["ua"], f["ua"]).max_abs == np.inf


def test_fieldview_indexing():
    a = np.asfortranarray(np.arange(4 * 3 * 2, dtype=float).reshape((4, 3, 2), order="F"))
    fv = FieldView("x", a, (-1, 0, 1))
    assert fv.ubounds == (2, 2, 2)
    assert fv.at(-1, 0, 1) == a[0, 0, 0]
    assert fv.at(2, 2, 2) == a[3, 2, 1]
    np.testing.assert_array_equal(fv.f[1:3, :, 2], a[2:4, :, 1])
    fv.f[0, 1, 1] = -5
    assert a[1, 1, 0] == -5
    with pytest.raises(IndexError):
        fv.at(3, 0, 1)
    with pytest.raises(IndexError):
        fv.f[-2, 0, 1]
    assert np.asarray(fv) is a
    assert "x(-1:2, 0:2, 1:2)" in repr(fv)


def test_read_fortran_reference(case_ref):
    from conftest import ref_cases

    names = ref_cases()
    if not names:
        pytest.skip("no reference cases")
    root = case_ref(names[0])
    steps = dump.list_steps(root)
    assert steps[0] == 0
    r = dump.read(root, steps[-1])
    assert r.nstep == steps[-1]
    for n in ("ua", "va", "wa", "tha", "ppi", "xh", "zh"):
        fv = r[n]
        assert np.all(np.isfinite(fv.data)), n
    assert r["ua"].shape[0] == r["tha"].shape[0] + 1
