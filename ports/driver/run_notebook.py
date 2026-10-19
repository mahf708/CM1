#!/usr/bin/env python3
"""Execute a notebook and save it (and an html copy) even when a cell fails.

    python ports/driver/run_notebook.py ports/driver/supercell_backends.ipynb OUT_DIR

Exit status 1 if a cell raised.
"""

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
from nbconvert import HTMLExporter


def main(src, out_dir):
    src, out = Path(src), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    nb = nbformat.read(src, as_version=4)
    client = NotebookClient(
        nb, timeout=3600, kernel_name="python3", resources={"metadata": {"path": "."}}
    )
    status = 0
    try:
        client.execute()
    except CellExecutionError as e:
        print(e, file=sys.stderr)
        status = 1
    nbformat.write(nb, out / src.name)
    html, _ = HTMLExporter().from_notebook_node(nb)
    (out / src.with_suffix(".html").name).write_text(html)
    print(f"wrote {out / src.name} and its html")
    return status


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
