"""Performance sweep of the cm1x time step: backends x {eager, jit} x grids.

    python ports/array/bench/sweep.py                      # default sweep
    python ports/array/bench/sweep.py --backends numpy,jax --grids 20,40 \\
        --steps 5 --out ports/array/bench/results_cpu

Each (backend, mode, grid) runs in its own subprocess (fresh jax/torch state,
bounded by --timeout) on the supercell_kessler case with nx/ny/nz overridden.
Measured per run: model set-up time, the first step (includes compilation for
mode=jit: jax.jit / torch.compile), and the median of the following
``--steps`` steps ("steady-state"), from which grid points per second follow.
Results go to ``<out>.json`` and a Markdown table ``<out>.md`` with the CPU
model, thread count and library versions.

mode=jit means ``Model(..., jit=True)``: on jax one ``jax.jit`` function
per time step; on torch the hot routines (advection, acoustic small step,
subgrid turbulence, Kessler) each go through ``torch.compile`` (inductor) and
the rest of the step runs eagerly (``Backend.kernel``).  Backends without a
compiler are run eagerly only.  Each torch run gets an empty inductor cache
(``TORCHINDUCTOR_CACHE_DIR``), so its compile time is a cold start (unless
``--warm-cache``).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CASE = REPO / "ports" / "reference" / "supercell_kessler" / "namelist.input"
COMPILABLE = {"jax", "torch"}


def _grid(s):
    p = [int(x) for x in s.lower().split("x")]
    return tuple(p * 3) if len(p) == 1 else tuple(p)


def run_one(backend, mode, grid, steps, case, threads):
    """Child process body: time one configuration, print a JSON line."""
    if threads and backend == "torch":
        import torch

        torch.set_num_threads(threads)
    # (jax: thread count set through XLA_FLAGS by the parent, see _child_env)
    from cm1x.model import Model

    nx, ny, nz = grid
    t0 = time.perf_counter()
    m = Model(
        case,
        backend=backend,
        jit=(mode == "jit"),
        overrides={"nx": nx, "ny": ny, "nz": nz},
    )
    _sync(m)
    t_init = time.perf_counter() - t0
    t0 = time.perf_counter()
    m.step()
    _sync(m)
    t_first = time.perf_counter() - t0
    times = []
    for _ in range(steps):
        t0 = time.perf_counter()
        m.step()
        _sync(m)
        times.append(time.perf_counter() - t0)
    f = m.fields()
    finite = all(bool((abs(v) < 1e30).all()) for v in f.values())
    per = statistics.median(times)
    return {
        "backend": backend,
        "mode": mode,
        "grid": list(grid),
        "points": nx * ny * nz,
        "steps": steps,
        "init_s": t_init,
        "first_step_s": t_first,
        "compile_s": max(t_first - per, 0.0) if mode == "jit" else None,
        "step_s": per,
        "step_min_s": min(times),
        "threads": threads or None,
        "points_per_s": nx * ny * nz / per,
        "max_w": float(abs(f["w"]).max()),
        "finite": finite,
    }


def _sync(m):
    """Block until the backend has finished the queued work."""
    s = m.state
    if m.bk.name == "jax":
        import jax

        jax.block_until_ready(s)
    elif m.bk.name == "dask":
        pass  # Model persists dask state every step
    # numpy / torch (CPU) are synchronous


def _child_env(backend, threads, cache_dir=None):
    env = dict(os.environ)
    if cache_dir:
        env["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
    if threads and backend == "jax":
        env["XLA_FLAGS"] = (
            env.get("XLA_FLAGS", "")
            + " --xla_cpu_multi_thread_eigen=false"
            + f" intra_op_parallelism_threads={threads}"
        ).strip()
    if threads:
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            env[v] = str(threads)
    return env


def env_info():
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": _cpu_model(),
        "ncpu": os.cpu_count(),
    }
    for mod in ("numpy", "jax", "jaxlib", "torch", "dask", "array_api_strict"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            info[mod] = None
    with contextlib.suppress(ImportError):
        import torch

        info["torch_threads"] = torch.get_num_threads()
    return info


def _cpu_model():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def to_markdown(results, info, cmd):
    lines = [
        "# cm1x CPU performance sweep",
        "",
        (
            "Case: supercell_kessler (WENO5, TKE, open radiative BCs, Kessler), "
            f"nx/ny/nz overridden; float64. Command: `{cmd}`"
        ),
        "",
        f"- CPU: {info['cpu']} ({info['ncpu']} logical CPUs)",
        f"- Python {info['python']}, "
        + ", ".join(
            f"{k} {info[k]}"
            for k in ("numpy", "jax", "jaxlib", "torch", "dask", "array_api_strict")
            if info.get(k)
        ),
        (
            f"- default torch threads: {info.get('torch_threads')}; numpy runs "
            "single-threaded elementwise; jax (XLA CPU) uses its own thread pool. "
            "`threads` = 1 pins torch (set_num_threads) / XLA (intra-op) to one "
            "thread for a like-for-like comparison with numpy."
        ),
        "",
        (
            "`first step` includes compilation for jit; `compile` = first step - "
            "steady step. `step` = median of the timed steps. jax jit = whole "
            "step in one jax.jit; torch jit = torch.compile (inductor) of each "
            "hot routine, cold inductor cache, rest of the step eager."
        ),
        "",
        (
            "| backend | mode | threads | grid | points | init (s) | first step (s) "
            "| compile (s) | step (s) | Mpoints/s | speedup vs numpy |"
        ),
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ref = {
        tuple(r["grid"]): r["step_s"]
        for r in results
        if r.get("backend") == "numpy" and r.get("mode") == "eager" and "step_s" in r
    }
    for r in results:
        g = "x".join(str(x) for x in r["grid"])
        th = r.get("threads") or "default"
        if "error" in r:
            lines.append(
                f"| {r['backend']} | {r['mode']} | {th} | {g} | | | | | "
                f"failed: {r['error'][:80]} | | |"
            )
            continue
        comp = "" if r["compile_s"] is None else f"{r['compile_s']:.1f}"
        base = ref.get(tuple(r["grid"]))
        sp = f"{base / r['step_s']:.2f}x" if base else ""
        lines.append(
            f"| {r['backend']} | {r['mode']} | {th} | {g} | {r['points']} "
            f"| {r['init_s']:.2f} | {r['first_step_s']:.2f} | {comp} "
            f"| {r['step_s']:.4f} | {r['points_per_s'] / 1e6:.3f} | {sp} |"
        )
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--backends", default="numpy,jax,torch")
    ap.add_argument("--modes", default="eager,jit")
    ap.add_argument("--grids", default="20,40,40x40x80,80")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--case", default=str(CASE))
    ap.add_argument(
        "--threads",
        default="0",
        help="comma list of thread counts for torch/jax runs (0: library default)",
    )
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument(
        "--skip-eager-jax",
        action="store_true",
        help="skip eager jax (op-by-op dispatch, very slow on large grids)",
    )
    ap.add_argument(
        "--skip",
        default="",
        help="comma list of backend:mode[:grid] entries to leave out, e.g. torch:jit",
    )
    ap.add_argument(
        "--warm-cache",
        action="store_true",
        help="torch: keep the default (persistent) inductor cache instead of "
        "a fresh one per run (cold compile)",
    )
    ap.add_argument("--out", default=str(HERE / "results_cpu"))
    ap.add_argument("--child", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)

    if a.child:
        b, mode, g = a.child.split(":")
        res = run_one(b, mode, _grid(g), a.steps, a.case, int(a.threads))
        print("RESULT " + json.dumps(res), flush=True)
        return 0

    results = []
    for g in [_grid(s) for s in a.grids.split(",")]:
        for b in a.backends.split(","):
            for mode, nt in [
                (m, int(t)) for m in a.modes.split(",") for t in a.threads.split(",")
            ]:
                if mode == "jit" and b not in COMPILABLE:
                    continue
                if mode == "eager" and b == "jax" and a.skip_eager_jax:
                    continue
                if nt and b not in COMPILABLE:
                    continue  # thread counts only matter for torch / jax
                gs = "x".join(map(str, g))
                skips = set(a.skip.split(",")) - {""}
                if f"{b}:{mode}" in skips or f"{b}:{mode}:{gs}" in skips:
                    continue
                cmd = [
                    sys.executable,
                    __file__,
                    "--child",
                    f"{b}:{mode}:{gs}",
                    "--steps",
                    str(a.steps),
                    "--case",
                    a.case,
                    "--threads",
                    str(nt),
                ]
                t0 = time.time()
                tmp = None
                if b == "torch" and mode == "jit" and not a.warm_cache:
                    tmp = tempfile.TemporaryDirectory(prefix="cm1x_inductor_")
                try:
                    p = subprocess.run(
                        cmd,
                        capture_output=True,
                        check=False,
                        text=True,
                        timeout=a.timeout,
                        env=_child_env(b, nt, tmp and tmp.name),
                    )
                    line = [x for x in p.stdout.splitlines() if x.startswith("RESULT ")]
                    if line:
                        r = json.loads(line[-1][7:])
                    else:
                        err = (p.stderr.strip().splitlines() or ["no output"])[-1]
                        r = {"backend": b, "mode": mode, "grid": list(g), "error": err}
                        r["threads"] = nt or None
                except subprocess.TimeoutExpired:
                    r = {
                        "backend": b,
                        "mode": mode,
                        "grid": list(g),
                        "error": f"timeout after {a.timeout:.0f} s",
                        "threads": nt or None,
                    }
                if tmp is not None:
                    tmp.cleanup()
                results.append(r)
                msg = r.get("error") or (
                    f"first {r['first_step_s']:.2f}s step {r['step_s']:.4f}s"
                )
                th = f"t={nt}" if nt else "t=dflt"
                print(
                    f"{b:6s} {mode:5s} {th:6s} {gs:9s} {msg}  [{time.time() - t0:.0f}s]"
                )
                sys.stdout.flush()

    info = env_info()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmdline = "python ports/array/bench/sweep.py " + " ".join(argv or sys.argv[1:])
    out.with_suffix(".json").write_text(
        json.dumps({"env": info, "command": cmdline, "results": results}, indent=1)
        + "\n"
    )
    out.with_suffix(".md").write_text(to_markdown(results, info, cmdline))
    print(f"wrote {out.with_suffix('.md')} and .json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
