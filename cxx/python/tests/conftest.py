import os
import pathlib

import pytest

CASES_DIR = pathlib.Path(__file__).resolve().parents[2] / "cases"


def ref_root():
    r = os.environ.get("CM1K_REF_DIR", "")
    return pathlib.Path(r) if r else None


@pytest.fixture
def cases_dir():
    return CASES_DIR


@pytest.fixture
def case_ref():
    """case_ref(name) -> path to the reference dumps of a case (skips if unavailable)."""

    def get(name):
        r = ref_root()
        if r is None:
            pytest.skip("CM1K_REF_DIR not set")
        p = r / name
        if not (p / "step_000000" / "index.txt").is_file():
            pytest.skip(f"no reference dumps for case {name} in {r}")
        return p

    return get


def ref_cases():
    """Cases available in CM1K_REF_DIR that also have a namelist in cxx/cases."""
    r = ref_root()
    if r is None or not r.is_dir():
        return []
    return sorted(
        d.name
        for d in r.iterdir()
        if (d / "step_000000" / "index.txt").is_file()
        and (CASES_DIR / d.name / "namelist.input").is_file()
    )


def ext_available():
    try:
        import cm1k
    except ImportError:
        return False
    return getattr(cm1k, "_cm1k", None) is not None


requires_ext = pytest.mark.skipif(
    not ext_available(), reason="cm1k._cm1k extension not built"
)
