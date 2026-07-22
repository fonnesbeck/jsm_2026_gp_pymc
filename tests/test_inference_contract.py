import arviz as az
import numpy as np
import pymc as pm
import xarray as xr

from inference_contract import eti, eti_bounds, inference_health, posterior_subset


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

    diagnostics, passed = inference_health(
        idata, model, ess_floor=1, rhat_ceiling=1.1
    )

    assert set(diagnostics.index) == {"alpha", "beta"}
    assert passed is True


def test_prediction_sampling_uses_a_fresh_model_without_mutating_fit_model():
    from inference_contract import sample_fresh_model_predictions

    x_observed = np.array([[0.0], [1.0]])
    y_observed = np.array([0.1, -0.2])
    with pm.Model(coords={"observation": [0, 1], "feature": ["x"]}) as fitted_model:
        x_data = pm.Data("X", x_observed, dims=("observation", "feature"))
        y_data = pm.Data("y_data", y_observed, dims="observation")
        ell = pm.HalfNormal("ell", sigma=1)
        eta = pm.HalfNormal("eta", sigma=1)
        sigma = pm.HalfNormal("sigma", sigma=1)
        gp = pm.gp.Marginal(cov_func=eta**2 * pm.gp.cov.Matern52(1, ls=ell))
        gp.marginal_likelihood("y", X=x_data, y=y_data, sigma=sigma, dims="observation")

    fitted_idata = az.from_dict(
        {
            "posterior": {
                "ell": np.array([[0.7, 0.8]]),
                "eta": np.array([[1.1, 1.2]]),
                "sigma": np.array([[0.2, 0.3]]),
            }
        }
    )
    fitted_names = set(fitted_model.named_vars)

    def build_prediction_model():
        with pm.Model(
            coords={"observation": [0, 1], "feature": ["x"], "prediction": [0, 1]}
        ) as prediction_model:
            x_data = pm.Data("X", x_observed, dims=("observation", "feature"))
            y_data = pm.Data("y_data", y_observed, dims="observation")
            x_pred = pm.Data("X_pred", [[0.5], [1.5]], dims=("prediction", "feature"))
            ell = pm.HalfNormal("ell", sigma=1)
            eta = pm.HalfNormal("eta", sigma=1)
            sigma = pm.HalfNormal("sigma", sigma=1)
            gp = pm.gp.Marginal(
                cov_func=eta**2 * pm.gp.cov.Matern52(1, ls=ell)
            )
            gp.marginal_likelihood(
                "y", X=x_data, y=y_data, sigma=sigma, dims="observation"
            )
            gp.conditional("f_pred", x_pred, dims="prediction")
        return prediction_model

    predictions = sample_fresh_model_predictions(
        fitted_idata,
        build_prediction_model,
        var_names=["f_pred"],
        random_seed=42,
    )

    assert set(fitted_model.named_vars) == fitted_names
    assert "f_pred" not in fitted_model.named_vars
    assert "/predictions" in predictions.groups
    assert predictions["predictions"]["f_pred"].sizes["prediction"] == 2
