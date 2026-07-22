import os
import subprocess
import time
from pathlib import Path
import importlib.util

import pymc as pm


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
            "--",
            "--execute-models",
            "true",
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


def test_00_environment_check():
    run_notebook(NB / "00_environment_check.py", timeout_s=120)


def test_00_environment_check_gp_contract():
    spec = importlib.util.spec_from_file_location(
        "environment_check",
        NB / "00_environment_check.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    _, definitions = module.app.run()
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


def assert_foundations_artifact_contract(idata, free_rv_names: set[str]) -> None:
    import arviz as az

    posterior = idata["posterior"]
    assert free_rv_names <= set(posterior.data_vars)
    assert {"chain", "draw"} <= set(posterior.dims)
    assert posterior.sizes["chain"] == 4, "expected four chains"

    divergences = idata["sample_stats"]["diverging"].sum().item()
    assert divergences == 0

    diagnostics = az.summary(
        idata,
        var_names=sorted(free_rv_names),
        kind="diagnostics",
        round_to="none",
    )
    assert diagnostics["r_hat"].notna().all()
    assert (diagnostics["r_hat"] <= 1.01).all()
    assert (diagnostics["ess_bulk"] >= 400).all()
    assert (diagnostics["ess_tail"] >= 400).all()

    assert idata["observed_data"]["conc_obs"].dims == ("observation",)
    assert idata["posterior_predictive"]["conc_obs"].dims == (
        "chain",
        "draw",
        "observation",
    )


def test_01_foundations():
    run_notebook(NB / "01_foundations.py", timeout_s=900)

    import arviz as az

    for filename, free_rv_names in (
        ("01_warmup.nc", {"mu", "sigma"}),
        ("01_piecewise.nc", {"peak", "rise", "decay", "tau", "sigma"}),
    ):
        idata = az.from_netcdf(NB.parent / "results" / filename)
        assert_foundations_artifact_contract(idata, free_rv_names)



def test_foundations_artifact_contract_rejects_nonfour_chain_datatree():
    import arviz as az
    import numpy as np

    idata = az.from_dict(
        {
            "posterior": {"mu": np.zeros((3, 500))},
            "sample_stats": {"diverging": np.zeros((3, 500), dtype=bool)},
            "posterior_predictive": {"conc_obs": np.zeros((3, 500, 11))},
            "observed_data": {"conc_obs": np.zeros(11)},
        }
    )

    with pytest.raises(AssertionError, match="chain"):
        assert_foundations_artifact_contract(idata, {"mu"})

def assert_marginal_latent_artifact_contract(
    idata, free_rv_names: set[str], observed_name: str, observed_dim: str
) -> None:
    import arviz as az

    posterior = idata["posterior"]
    assert free_rv_names <= set(posterior.data_vars)
    assert posterior.sizes["chain"] == 4, "expected four chains"
    assert idata["sample_stats"]["diverging"].sum().item() == 0

    diagnostics = az.summary(
        idata,
        var_names=sorted(free_rv_names),
        kind="diagnostics",
        round_to="none",
    )
    assert diagnostics["r_hat"].notna().all()
    assert (diagnostics["r_hat"] <= 1.01).all()
    assert (diagnostics["ess_bulk"] >= 400).all()
    assert (diagnostics["ess_tail"] >= 400).all()
    assert idata["observed_data"][observed_name].dims == (observed_dim,)
    assert idata["posterior_predictive"][observed_name].dims == (
        "chain",
        "draw",
        observed_dim,
    )


def test_02_marginal_latent():
    run_notebook(
        NB / "02_marginal_latent_gps.py",
        timeout_s=180,
        expected_stderr="Naive MAP optimization complete",
    )
    import arviz as az

    results_dir = NB.parent / "results"
    structured = az.from_netcdf(results_dir / "02_marginal_gp.nc")
    coal = az.from_netcdf(results_dir / "02_coal_latent_gp.nc")
    assert_marginal_latent_artifact_contract(
        structured,
        {"intercept", "beta", "ell", "eta", "sigma"},
        "y",
        "observation",
    )
    assert_marginal_latent_artifact_contract(
        coal,
        {"alpha", "ell", "eta", "f_rotated_"},
        "y",
        "year",
    )


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
