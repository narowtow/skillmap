"""Execute notebooks from notebooks/ with this venv's Python; fail on any error."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from time import perf_counter

import nbformat
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    runtime = root / ".notebook-runtime"
    kernel = runtime / "kernels" / "skillmap-venv"
    kernel.mkdir(parents=True, exist_ok=True)
    (kernel / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Skillmap venv", "language": "python",
    }), encoding="utf-8")
    environment = dict(os.environ, IPYTHONDIR=str(runtime / "ipython"),
                       JUPYTER_RUNTIME_DIR=str(runtime), MPLCONFIGDIR=str(runtime / "matplotlib"))
    for path in sorted((root / "notebooks").glob("*.ipynb")):
        started = perf_counter()
        manager = KernelManager(
            kernel_name="skillmap-venv", connection_file=str(runtime / f"{path.stem}.json"),
            kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(runtime / "kernels")]),
        )
        notebook = nbformat.read(path, as_version=4)
        client = NotebookClient(notebook, km=manager, timeout=180, allow_errors=False,
                                resources={"metadata": {"path": str(root / "notebooks")}})
        try:
            client.execute(env=environment)
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
        nbformat.write(notebook, path)
        print(f"Executed {path.name} in {perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
