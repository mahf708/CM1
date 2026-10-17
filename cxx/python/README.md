# cm1k Python bindings

`cm1k` is a pybind11 module for driving the C++/Kokkos port of CM1 from
Python, in the spirit of E3SM's pyeamxx: create a model from a namelist, load
or initialize its state, step it, and read or write fields as numpy arrays.
`cm1k.dump` reads and writes the reference dump format with numpy only, so you
can compare runs with the Fortran model without building the C++.

## Install

From the repository root, after `git submodule update --init --recursive`
(see `cxx/README.md`):

```bash
pip install ./cxx                                              # Serial backend
pip install ./cxx -Ccmake.define.Kokkos_ENABLE_OPENMP=ON       # OpenMP
pip install ./cxx -Ccmake.define.Kokkos_ENABLE_CUDA=ON \
                  -Ccmake.define.Kokkos_ARCH_AMPERE80=ON       # CUDA (A100)
```

`cxx/pyproject.toml` uses scikit-build-core. It builds only the `_cm1k`
target, installs only the `python` CMake component, and turns off the tests
and the Fortran bridge. You can pass any CMake variable with
`-Ccmake.define.VAR=VALUE` or through `CMAKE_ARGS`. Set
`CMAKE_BUILD_PARALLEL_LEVEL` to choose the number of build jobs. Add
`--no-build-isolation` if scikit-build-core and pybind11 are already
installed.

**Without pip (CMake build tree):** the build directory can be imported
directly.

```bash
cmake -S cxx -B build -DCM1K_ENABLE_PYTHON=ON -DCM1K_ENABLE_TESTS=OFF \
      -DCM1K_ENABLE_FORTRAN_BRIDGE=OFF -DKokkos_ENABLE_OPENMP=ON
cmake --build build --target _cm1k -j
export PYTHONPATH=$PWD/build/python
python -c "import cm1k; print(cm1k.execution_space())"
```

To install the package into a prefix, run
`cmake --install build --component python`. The destination is
`lib/pythonX.Y/site-packages/cm1k`, which you can change with
`CM1K_PY_INSTALL_DIR`.

**Tests:** `cxx/scripts/ci_python.sh` builds the module and runs pytest. You
can also run pytest yourself with `pytest cxx/python/tests`. Tests that need
reference dumps run only when `CM1K_REF_DIR` points at the output of
`cxx/scripts/make_refs.sh`; otherwise they are skipped.

## Example

```python
import numpy as np
import cm1k
from cm1k import dump

ref = "refs/kessler_bubble"                      # from cxx/scripts/make_refs.sh

nl = cm1k.Namelist.from_file(f"{ref}/namelist.input")
nl["dtl"]                                        # 6.0 (typed: int/float/bool/str/list)
m = cm1k.Model(nl)                               # or cm1k.Model(path, dtl=3.0, nx=64)
m.init_from_dump(ref, nstep=0)                   # or m.init() (native init)
print(m, cm1k.execution_space())

tha = m.field("tha")                             # FieldView: numpy array + Fortran bounds
tha.lbounds, tha.ubounds                         # e.g. ((-2, -2, 0), (35, 35, 21))
tha.f[1:m.bounds["ni"] + 1, 1, 1] += 0.5         # Fortran-origin indexing (Python slice semantics)
tha.at(1, 1, 1)                                  # one element, Fortran indices
m["qa_qv"] = np.maximum(m["qa_qv"], 0.0)         # set_field, with a shape check

m.step(10)                                       # advance 10 large steps
out = m.write_dump("out")                        # out/step_000010 (same format as the Fortran)
for name, s in dump.compare(out, dump.read(ref, 10)).items():
    print(f"{name:8s} max_abs={s.max_abs:.3e} rel={s.rel:.3e} at {s.where}")
```

## API

**Kokkos lifecycle**

| Function | What it does |
|---|---|
| `cm1k.initialize(num_threads=-1, device_id=-1, args=())` | Initializes Kokkos. It is idempotent: it returns `False` if Kokkos is already initialized. Creating a `Model` calls it on demand, so you only need it to choose settings. |
| `cm1k.finalize()` | Releases every live `Model`, then finalizes Kokkos. It is idempotent and registered with `atexit`. Kokkos cannot be initialized again afterwards. A released model raises `RuntimeError` when used. numpy views that outlive finalize stay valid, because their memory is deliberately leaked. |
| `is_initialized()`, `is_finalized()` | Report the Kokkos state. |
| `execution_space()`, `memory_space()` | Return `'Serial'`, `'OpenMP'`, `'Cuda'`, and so on. |
| `host_accessible()` | Reports whether field arrays can be viewed without copying. |
| `concurrency()` | Kokkos concurrency of the execution space. |
| `kokkos_configuration()` | The Kokkos configuration as text. |

**`cm1k.Namelist`**

- Construct with `from_file(path)` or `from_string(text)`.
- `nl[key]` and `nl.get(key, default)` convert values to Python types. Fortran
  logicals become `bool`. `3*0.0` expands to `[0.0, 0.0, 0.0]`. A key with
  several values becomes a list.
- `nl[key] = value` writes a Python value: `bool` becomes `.true.`, `str` is
  quoted, and a list becomes a comma list. `set_raw(key, "fortran text")`
  stores the text as written.
- Also available: `geti`, `getr`, `getb`, `gets`, `getv` (with C++
  semantics), `keys()`, `to_dict()`, `group_of(key)`, `copy()`, and
  `key in nl`.

**`cm1k.Model(namelist_or_path, **overrides)`**

- Setup and stepping: `init()`, `init_from_dump(step_dir)` (or
  `init_from_dump(root, nstep=N)`), and `step(n=1)`. `step` releases the GIL.
  C++ errors become `RuntimeError`.
- Output: `write_dump(root)` returns the `root/step_NNNNNN` directory it wrote.
- Read-only properties: `nstep`, `mtime`, `dt`, `namelist`, `bounds` (a dict
  with `nx`, `ni`, `ib`, `ie`, `ibm`, ...), `qname`, and `released`.
- Field metadata: `field_names()`, `has_field(name)` (or `name in m`),
  `field_rank(name)`, and `field_bounds(name)`, which returns
  `(lbounds, ubounds)`.
- `field(name, copy=False)` returns a `FieldView`. `fields()` returns
  `{name: FieldView}`.
- `get_field(name, copy=False)` and `m[name]` return a Fortran-ordered
  float64 ndarray. When the memory space is host accessible (Serial, OpenMP,
  CUDA UVM) and `copy=False`, it is a zero-copy view: writing to it changes
  the model. The view keeps the allocation alive. Otherwise you get a copy.
- `set_field(name, array)` and `m[name] = array` copy into the model. Any
  dtype or memory order is accepted, but the shape must match exactly.
- Errors: an unknown field raises `KeyError`, a wrong shape raises
  `ValueError`, and C++ failures raise `RuntimeError`.
- Field names are the reference-dump names:
  - grid metrics: `xh`, `uh`, ..., `zh`, `mh`, ...
  - base state: `th0`, `pi0`, `rho0`, ...
  - prognostics: `ua`, `va`, `wa`, `ppi`, `tha`, `rho`, `prs`, `tkea`, `kmh`, `kmv`, `khh`, `khv`, `rain`
  - moisture: `qa_<qname>` (for example `qa_qv`), each a view of one slice of the 4D `qa`

**`cm1k.FieldView`** (pure numpy)

- `.data` is the ndarray. `.name`, `.lbounds`, `.ubounds` and `.bounds` describe it.
- `.is_view` tells you whether `.data` aliases model memory.
- Indexing: `.at(i, j, k)` and `.index(i, j, k)` use Fortran indices. `.f[...]`
  indexes with a Fortran origin but keeps Python slice semantics (the stop is
  exclusive). `fv[...]` is plain numpy, zero-based.
- `np.asarray(fv)` returns `.data`.

**`cm1k.dump`** (pure numpy, works without the extension)

- `step_dir(root, n)`, `list_steps(root)`, and `read_index(dir)`, which
  returns `(nstep, mtime, entries)`.
- `Dump(dir)` or `read(root, nstep)` opens a dump. `dump[name]` returns a
  `FieldView`, read lazily. A `Dump` has the same `field_names()` and `field()`
  interface as `Model`.
- `write(dir, {name: FieldView | (array, lbounds) | array}, nstep, mtime)`
  writes the same `index.txt` format as `src/cm1dump.F`.
- `compare(a, ref, names=None)` takes dump directories, `Dump` objects or
  `Model`s and returns `{name: DiffStats(max_abs, ref_max, rel, where, bounds_match)}`.
  `where` holds the Fortran indices of the largest difference, the same as
  `cm1k::diff`. NaNs count as `inf`. `diff_fields(a, ref, region=None)`
  compares two fields directly.
