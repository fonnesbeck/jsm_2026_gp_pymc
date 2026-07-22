"""Shared Bayesian inference helpers for the workshop notebooks."""

import arviz as az
import pymc as pm


def sample_fresh_model_predictions(
    idata, build_prediction_model, *, var_names, random_seed
):
    """Draw named predictions from a model newly built for the prediction grid."""
    prediction_model = build_prediction_model()
    with prediction_model:
        return pm.sample_posterior_predictive(
            idata,
            var_names=var_names,
            random_seed=random_seed,
            predictions=True,
        )


import numpy as np


def eti(data, prob=0.89):
    """Return a labeled equal-tailed interval over chain and draw dimensions."""
    tail_probability = (1 - prob) / 2
    return data.quantile(
        [tail_probability, 1 - tail_probability], dim=("chain", "draw")
    )


def eti_bounds(data, prob=0.89):
    """Return the lower and upper equal-tailed interval endpoints by position."""
    interval = eti(data, prob=prob)
    return interval.isel(quantile=0), interval.isel(quantile=1)


def posterior_subset(idata, draws_per_chain=100):
    """Select evenly spaced draws while retaining every chain and other dimensions."""
    draw_count = idata["posterior"].sizes.get("draw", 0)
    if draw_count == 0:
        return idata
    indices = np.linspace(
        0, draw_count - 1, num=min(draws_per_chain, draw_count), dtype=int
    )
    return idata.isel(draw=indices, missing_dims="ignore")


def inference_health(idata, model, ess_floor=400, rhat_ceiling=1.01):
    """Summarize every free RV and report whether sampler diagnostics pass."""
    free_rv_names = [rv.name for rv in model.free_RVs]
    diagnostics = az.summary(idata, var_names=free_rv_names, kind="diagnostics")
    divergences = int(idata["sample_stats"]["diverging"].sum().item())
    rhat = diagnostics["r_hat"].to_numpy(dtype=float)
    ess_bulk = diagnostics["ess_bulk"].to_numpy(dtype=float)
    ess_tail = diagnostics["ess_tail"].to_numpy(dtype=float)
    passed = bool(
        divergences == 0
        and np.isfinite(rhat).all()
        and (rhat <= rhat_ceiling).all()
        and (ess_bulk >= ess_floor).all()
        and (ess_tail >= ess_floor).all()
    )
    diagnostics.attrs["divergences"] = divergences
    diagnostics.attrs["passed"] = passed
    return diagnostics, passed
