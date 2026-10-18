"""Run a CUDA build of cm1k on a cloud GPU (Modal), for CI.

GitHub-hosted runners have no GPU, so the CUDA build is compiled on the
runner (nvidia/cuda container) and only the run happens on Modal: the
workspace (build tree, reference dumps, scripts; no object files) is
mounted at the same absolute path, so ctest's absolute paths still hold,
and cxx/scripts/gpu_check.sh runs there.

    pip install modal          # needs MODAL_TOKEN_ID / MODAL_TOKEN_SECRET
    CM1K_MODAL_GPU=T4 modal run cxx/scripts/modal_gpu.py \\
        --build build --refs refs --out gpu_out --cases kessler_periodic,kessler_tke

CM1K_MODAL_GPU picks the GPU (T4, L4, A10G, A100, ...; the build's
Kokkos_ARCH must match).  Exit status is gpu_check.sh's.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import modal

# Modal re-imports this file inside the container (as /root/modal_gpu.py),
# where there is no checkout around it; the workspace path only matters
# locally, and check() is handed it as an argument.
_HERE = Path(__file__).resolve()
WS = Path(
    os.environ.get("GITHUB_WORKSPACE")
    or (_HERE.parents[2] if len(_HERE.parents) > 2 else _HERE.parent)
).resolve()
GPU = os.environ.get("CM1K_MODAL_GPU", "T4")
CUDA_IMAGE = os.environ.get("CM1K_CUDA_IMAGE", "nvidia/cuda:12.6.3-devel-ubuntu24.04")

image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.12")
    .apt_install("libopenmpi-dev", "openmpi-bin", "libgomp1")
    .pip_install("numpy", "cmake")  # cmake: ctest
    .add_local_dir(
        WS,
        remote_path=str(WS),
        ignore=[
            ".git",
            ".ccache",
            "**/*.o",
            "**/*.a",
            "**/CMakeFiles/**/*.d",
            "cxx/externals/**",
            "build/externals/**",
            "build/_deps/**",
            "gpu_out/**",
        ],
    )
)

app = modal.App("cm1k-gpu-check", image=image)


@app.function(gpu=GPU, timeout=3600, cpu=2.0, memory=8192)
def check(
    ws: str, build: str, refs: str, out: str, cases: list[str], max_steps: str
) -> dict:
    # ws: the workspace path, as mounted (same absolute path as on the runner)
    ws = Path(ws)
    env = dict(os.environ, OMP_PROC_BIND="false", PYTHON=sys.executable)
    if max_steps:
        env["CM1K_GPU_STEPS"] = max_steps
    subprocess.run(["nvidia-smi"], check=False)
    cmd = [
        "bash",
        str(ws / "cxx/scripts/gpu_check.sh"),
        str(ws / build),
        str(ws / refs),
        out,
        *cases,
    ]
    rc = subprocess.run(cmd, env=env, cwd=str(ws), check=False).returncode
    files = {
        p.name: p.read_text(errors="replace")
        for p in Path(out).glob("*")
        if p.is_file()
    }
    return {"rc": rc, "files": files}


@app.local_entrypoint()
def main(
    build: str = "build",
    refs: str = "refs",
    out: str = "gpu_out",
    cases: str = "kessler_periodic",
    max_steps: str = "",
):
    res = check.remote(
        str(WS),
        build,
        refs,
        "/tmp/gpu_out",
        [c for c in cases.split(",") if c],
        max_steps,
    )
    od = Path(out)
    od.mkdir(parents=True, exist_ok=True)
    for name, text in res["files"].items():
        (od / name).write_text(text)
    print(
        (od / "report.md").read_text() if (od / "report.md").exists() else "no report"
    )
    if res["rc"] != 0:
        sys.exit(res["rc"])
