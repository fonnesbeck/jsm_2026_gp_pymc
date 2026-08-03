import arviz as az
import numpy as np

import pymc as pm
import xarray as xr

from inference_contract import (
    eti,
    eti_bounds,
    inference_health,
    posterior_subset,
    sampled_rv_names,
)


def test_eti_returns_labeled_equal_tailed_bounds():
    data = xr.DataArray(
        np.arange(2 * 10 * 3).reshape(2, 10, 3),
        dims=("chain", "draw", "location"),
        coords={"location": ["a", "b", "c"]},
    )

    interval = eti(data)

    assert interval.dims == ("quantile", "location")
    assert interval.coords["location"].values.tolist() == ["a", "b", "c"]
    np.testing.assert_allclose(interval["quantile"], [0.055, 0.945])


def test_eti_bounds_select_by_order_without_float_coordinate_matching():
    data = xr.DataArray(
        np.arange(2 * 10 * 3).reshape(2, 10, 3),
        dims=("chain", "draw", "location"),
        coords={"location": ["a", "b", "c"]},
    )

    lower, upper = eti_bounds(data)

    assert lower.equals(eti(data).isel(quantile=0))
    assert upper.equals(eti(data).isel(quantile=1))


def test_posterior_subset_preserves_every_chain_and_even_draws():
    idata = az.from_dict(
        {"posterior": {"theta": np.arange(3 * 10).reshape(3, 10)}},
        coords={"chain_label": ["a", "b", "c"]},
        dims={"theta": []},
    )

    subset = posterior_subset(idata, draws_per_chain=4)

    assert subset["posterior"].sizes["chain"] == 3
    assert subset["posterior"].sizes["draw"] == 4
    assert subset["posterior"]["draw"].values.tolist() == [0, 3, 6, 9]


def test_inference_health_uses_all_free_variables_and_diagnostics():
    with pm.Model() as model:
        alpha = pm.Normal("alpha")
        beta = pm.HalfNormal("beta")
        pm.Normal("obs", mu=alpha, sigma=beta, observed=np.array([0.0]))

    idata = az.from_dict(
        {
            "posterior": {
                "alpha": np.random.default_rng(42).normal(size=(4, 500)),
                "beta": np.random.default_rng(43).normal(size=(4, 500)),
            },
            "sample_stats": {"diverging": np.zeros((4, 500), dtype=bool)},
        }
    )

    diagnostics, passed = inference_health(idata, model, ess_floor=1, rhat_ceiling=1.1)

    assert set(diagnostics.index) == {"alpha", "beta"}
    assert passed is True


def test_sampled_rv_names_skips_variables_added_after_sampling():
    with pm.Model() as model:
        alpha = pm.Normal("alpha")
        beta = pm.HalfNormal("beta")
        pm.Normal("obs", mu=alpha, sigma=beta, observed=np.array([0.0]))

    idata = az.from_dict(
        {
            "posterior": {
                "alpha": np.zeros((2, 5)),
                "beta": np.ones((2, 5)),
            }
        }
    )

    with model:
        pm.Normal("f_pred")

    assert sampled_rv_names(idata, model) == ["alpha", "beta"]
