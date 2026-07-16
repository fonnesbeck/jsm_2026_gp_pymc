import subprocess
import time
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks"


def run_notebook(path: Path, timeout_s: int) -> float:
    """Execute a marimo notebook headlessly; return elapsed seconds. Raise on error."""
    start = time.time()
    proc = subprocess.run(
        ["marimo", "export", "html", str(path), "--no-include-code", "-o", "/dev/null"],
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
