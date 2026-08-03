import os
import subprocess
import sys
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


def run_notebook(
    path: Path, timeout_s: int, *, expected_stderr: str | None = None
) -> float:
    """Execute a marimo notebook headlessly; return elapsed seconds. Raise on error."""
    start = time.time()
    proc = subprocess.run(
        [
            "marimo",
            "export",
            "html",
            str(path),
            "--no-include-code",
            "-o",
            os.devnull,
        ],
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    stderr = proc.stderr
    if proc.returncode != 0:
        raise AssertionError(f"{path.name} failed:\n{stderr[-2000:]}")
    if expected_stderr is not None and expected_stderr not in stderr:
        raise AssertionError(
            f"{path.name} did not emit {expected_stderr!r} to stderr:\n{stderr[-2000:]}"
        )
    return time.time() - start


@pytest.mark.slow
def test_00_environment_check():
    run_notebook(NB / "00_environment_check.py", timeout_s=120)


_ENV_CONTRACT_SCRIPT = """\
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("environment_check", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# app.run() executes the notebook's own `import pymc as pm` cell first. Importing
# pymc here in the driver *before* app.run() deadlocks: pymc's import triggers
# pytensor setup that conflicts with preliz's IPython/pygments import chain when
# it later runs inside app.run(). Importing pymc only after app.run() reuses the
# module the notebook already imported and avoids the conflict entirely.
_, definitions = module.app.run()
import pymc as pm

model = definitions["env_check_model"]
gp = definitions["gp"]

assert model.coords == {"obs": tuple(range(20)), "feature": ("x",)}
assert model.named_vars_to_dims == {
    "X": ("obs", "feature"),
    "y_obs": ("obs",),
    "y": ("obs",),
}
assert model.named_vars["X"].get_value().shape == (20, 1)
assert model.named_vars["y_obs"].get_value().shape == (20,)
assert model.named_vars["ell"].owner.op.name == "lognormal"
assert model.named_vars["eta"].owner.op.name == "halfnormal"
assert model.named_vars["sigma"].owner.op.name == "halfnormal"
matern = gp.cov_func._factor_list[0]
assert isinstance(matern, pm.gp.cov.Matern52)
assert matern.ls is model.named_vars["ell"]
assert model.compile_logp()(model.initial_point()).shape == ()

print("CONTRACT OK")
"""


def test_00_environment_check_gp_contract():
    # Runs in a subprocess so the notebook's real `import preliz` never runs
    # inside the pytest process. See the comment in _ENV_CONTRACT_SCRIPT for
    # why pymc must not be imported in this script before app.run().
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _ENV_CONTRACT_SCRIPT,
            str(NB / "00_environment_check.py"),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "CONTRACT OK" in result.stdout


@pytest.mark.slow
def test_01_foundations():
    run_notebook(NB / "01_foundations.py", timeout_s=900)


@pytest.mark.slow
def test_03_marginal_and_latent():
    run_notebook(
        NB / "03_marginal_and_latent_gps.py",
        timeout_s=1800,
        expected_stderr="Naive MAP optimization complete",
    )


@pytest.mark.slow
def test_02_gp_priors_and_kernels():
    run_notebook(NB / "02_gp_priors_and_kernels.py", timeout_s=240)


@pytest.mark.slow
def test_04_scaling_workflow():
    run_notebook(NB / "04_scaling_and_workflow.py", timeout_s=900)


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
