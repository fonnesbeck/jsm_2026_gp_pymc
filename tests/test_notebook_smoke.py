import os
import subprocess
import time
from pathlib import Path

import pytest

NB = Path(__file__).resolve().parents[1] / "notebooks"

_BROKEN_NOTEBOOK_SOURCE = """\
import marimo

__generated_with = "0.23.14"
app = marimo.App()


@app.cell
def _():
    raise RuntimeError("intentional failure for smoke-test regression check")


if __name__ == "__main__":
    app.run()
"""


def run_notebook(path: Path, timeout_s: int) -> float:
    """Execute a marimo notebook headlessly; return elapsed seconds. Raise on error."""
    start = time.time()
    proc = subprocess.run(
        ["marimo", "export", "html", str(path), "--no-include-code", "-o", os.devnull],
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    if proc.returncode != 0:
        raise AssertionError(f"{path.name} failed:\n{proc.stderr[-2000:]}")
    return time.time() - start


def test_00_environment_check():
    run_notebook(NB / "00_environment_check.py", timeout_s=120)


def test_01_foundations():
    run_notebook(NB / "01_foundations.py", timeout_s=180)


def test_02_marginal_latent():
    run_notebook(NB / "02_marginal_latent_gps.py", timeout_s=180)


def test_03_kernels_hierarchy():
    run_notebook(NB / "03_kernels_and_hierarchy.py", timeout_s=240)


def test_04_scaling_workflow():
    run_notebook(NB / "04_scaling_and_workflow.py", timeout_s=240)


def test_run_notebook_detects_cell_error(tmp_path):
    """Regression guard: run_notebook must raise when a notebook cell errors.

    marimo is only range-pinned in the environment spec, so its exit-code
    behavior on a failing cell could drift across versions. This pins that
    behavior with a throwaway notebook that always raises.
    """
    broken = tmp_path / "broken_notebook.py"
    broken.write_text(_BROKEN_NOTEBOOK_SOURCE)

    with pytest.raises(AssertionError):
        run_notebook(broken, timeout_s=60)
