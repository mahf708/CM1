import subprocess
import sys

import numpy as np
import pytest

from conftest import ref_cases, requires_ext

pytestmark = requires_ext


@pytest.fixture
def nml(cases_dir):
    return str(cases_dir / "kessler_bubble" / "namelist.input")


@pytest.fixture
def model(nml):
    import cm1k

    return cm1k.Model(nml)


def test_kokkos_info():
    import cm1k

    assert cm1k.execution_space() in (
        "Serial",
        "OpenMP",
        "Threads",
        "Cuda",
        "HIP",
        "SYCL",
    )
    assert isinstance(cm1k.host_accessible(), bool)
    assert cm1k.initialize() in (True, False)
    assert cm1k.is_initialized()
    assert cm1k.initialize() is False  # idempotent
    assert cm1k.concurrency() >= 1
    assert "Kokkos" in cm1k.kokkos_configuration()


def test_construct(model, nml):
    import cm1k

    b = model.bounds
    nl = cm1k.Namelist.from_file(nml)
    assert (b["nx"], b["ny"], b["nz"]) == (nl["nx"], nl["ny"], nl["nz"])
    assert model.dt == nl["dtl"] and model.nstep == 0 and model.mtime == 0.0
    names = model.field_names()
    for n in (
        "ua",
        "va",
        "wa",
        "tha",
        "ppi",
        "rho",
        "xh",
        "zh",
        "th0",
        "qa_qv",
        "qa_qc",
        "qa_qr",
    ):
        assert n in names and n in model
    assert model.qname == ["qv", "qc", "qr"]
    assert model.field_rank("xh") == 1 and model.field_rank("ua") == 3
    lb, ub = model.field_bounds("ua")
    assert lb == (b["ib"], b["jb"], b["kb"]) and ub == (b["ie"] + 1, b["je"], b["ke"])
    assert "OpenMP" in repr(model) or cm1k.execution_space() in repr(model)


def test_overrides(nml):
    import cm1k

    m = cm1k.Model(nml, nx=8, ny=4, dtl=1.5)
    assert m.bounds["nx"] == 8 and m.bounds["ny"] == 4 and m.dt == 1.5
    assert m.namelist["nx"] == 8
    assert m.field("tha").shape[0] == 8 + 2 * m.bounds["ngxy"]
    with pytest.raises(TypeError):
        cm1k.Model(42)
    with pytest.raises(RuntimeError):
        cm1k.Model(cm1k.Namelist.from_string("&param0\n/\n"))  # nx/ny/nz missing


def test_field_get_set(model):
    import cm1k

    tha = model.field("tha")
    b = model.bounds
    assert tha.data.flags.f_contiguous and tha.dtype == np.float64
    assert tha.lbounds == (b["ib"], b["jb"], b["kb"])
    assert tha.shape == (
        b["ie"] - b["ib"] + 1,
        b["je"] - b["jb"] + 1,
        b["ke"] - b["kb"] + 1,
    )

    rng = np.random.default_rng(2)
    new = rng.standard_normal(tha.shape)
    model.set_field("tha", new)
    np.testing.assert_array_equal(model.get_field("tha", copy=True), new)
    # C-ordered and integer input are converted
    model["tha"] = np.ascontiguousarray(new * 2)
    np.testing.assert_array_equal(model["tha"], new * 2)
    model.set_field("xh", np.arange(model.field("xh").shape[0]))
    assert model.field("xh").at(model.bounds["ib"]) == 0.0

    # qa_<q> is a slice of the 4D qa array
    qc = model.field("qa_qc")
    model["qa_qc"] = np.full(qc.shape, 1e-3)
    assert np.all(model.field("qa_qc", copy=True).data == 1e-3)

    copy = model.field("tha", copy=True)
    assert not copy.is_view
    copy.data[...] = 0
    assert np.any(model["tha"] != 0)

    if cm1k.host_accessible():
        view = model.field("tha")
        assert view.is_view
        view.f[1, 1, 1] = 123.0
        assert model.field("tha", copy=True).at(1, 1, 1) == 123.0
        assert np.shares_memory(view.data, model["tha"])
        model["tha"] = view.data  # setting a field from its own view is a no-op


def test_field_errors(model):
    with pytest.raises(KeyError, match="unknown field"):
        model.field("nope")
    with pytest.raises(KeyError):
        model["qa_qi"]
    with pytest.raises(KeyError):
        model.set_field("nope", np.zeros(3))
    shape = model.field("ua").shape
    with pytest.raises(ValueError, match="shape mismatch"):
        model.set_field("ua", np.zeros(shape[::-1]))
    with pytest.raises(ValueError, match="shape mismatch"):
        model.set_field("ua", np.zeros(shape[:2]))
    with pytest.raises(ValueError, match="shape mismatch"):
        model.set_field("xh", np.zeros(3))
    with pytest.raises(RuntimeError, match="cannot open dump"):
        model.init_from_dump("/nonexistent/step_000000")
    with pytest.raises(ValueError):
        model.step(-1)


def test_step_and_init_do_not_crash(nml):
    """step()/init() either work or raise a Python exception (never abort the interpreter)."""
    code = f"""
import cm1k
m = cm1k.Model({nml!r})
for fn in (m.init, m.step):
    try:
        fn()
    except RuntimeError as e:
        print("raised:", e)
print("ok", m.nstep)
"""
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=600
    )
    assert r.returncode == 0, r.stderr
    assert "ok" in r.stdout


def test_write_dump_roundtrip(model, tmp_path):
    from cm1k import dump

    rng = np.random.default_rng(3)
    for n in ("ua", "tha", "qa_qr", "zh", "xh"):
        model[n] = rng.standard_normal(model.field(n).shape)
    d = model.write_dump(str(tmp_path))
    assert d.endswith("step_000000")
    r = dump.read(d)
    assert set(r.field_names()) == set(model.field_names())
    for n in model.field_names():
        a = model.field(n, copy=True)
        assert r[n].lbounds == a.lbounds, n
        np.testing.assert_array_equal(r[n].data, a.data)
    # and back into a fresh model
    import cm1k

    m2 = cm1k.Model(model.namelist)
    m2.init_from_dump(d)
    assert all(s.max_abs == 0 for s in dump.compare(m2, model).values())


@pytest.mark.parametrize("case", ref_cases() or ["<no refs>"])
def test_init_from_reference(case, case_ref, cases_dir, tmp_path):
    import cm1k
    from cm1k import dump

    root = case_ref(case)
    m = cm1k.Model(cases_dir / case / "namelist.input")
    m.init_from_dump(root, nstep=0)
    ref = dump.read(root, 0)
    assert m.nstep == ref.nstep and m.mtime == ref.mtime
    common = set(ref.field_names()) & set(m.field_names())
    assert {"ua", "va", "wa", "tha", "ppi", "zh"} <= common
    st = dump.compare(m, ref)
    assert all(s.bounds_match and s.max_abs == 0 for s in st.values()), {
        k: v for k, v in st.items() if v.max_abs or not v.bounds_match
    }
    # write_dump reproduces the Fortran dump bit for bit
    out = m.write_dump(str(tmp_path))
    st = dump.compare(out, ref, names=sorted(common))
    assert all(s.max_abs == 0 for s in st.values())
    # zero-copy views see init_from_dump updates (arrays are not reallocated)
    tha = m.field("tha")
    tha.data[...] = 0
    m.init_from_dump(root, nstep=0)
    assert np.array_equal(np.asarray(tha), ref["tha"].data) or not tha.is_view


def test_finalize_lifecycle(nml):
    code = f"""
import cm1k, numpy as np
m = cm1k.Model({nml!r})
keep = m["tha"]                      # numpy view that outlives finalize
keep[...] = 1.0
cm1k.finalize()
cm1k.finalize()                      # idempotent
assert cm1k.is_finalized() and not cm1k.is_initialized()
assert m.released
assert float(keep.sum()) == keep.size   # memory still valid
try:
    m.field("tha")
except RuntimeError as e:
    print("released:", e)
try:
    cm1k.initialize()
except RuntimeError as e:
    print("reinit:", e)
try:
    cm1k.Model({nml!r})
except RuntimeError as e:
    print("newmodel:", e)
print(repr(m))
"""
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
    )
    assert r.returncode == 0, r.stderr
    assert "released:" in r.stdout and "reinit:" in r.stdout and "newmodel:" in r.stdout
    assert "(released)" in r.stdout


def test_exit_with_live_objects(nml):
    """Interpreter exit with live models/views finalizes Kokkos cleanly."""
    code = f"""
import cm1k
m = cm1k.Model({nml!r}); m2 = cm1k.Model({nml!r})
v = m.field("ua"); q = m2["qa_qv"]
print("done")
"""
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
    )
    assert r.returncode == 0, r.stderr
    assert "done" in r.stdout
    assert "deallocat" not in r.stderr.lower()
