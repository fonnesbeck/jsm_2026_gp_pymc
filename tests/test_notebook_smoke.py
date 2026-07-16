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
