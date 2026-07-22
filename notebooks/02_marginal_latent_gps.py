import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Marginal and Latent Gaussian Processes

    Notebook 1 built up the Gaussian process (GP) as a distribution over
    functions, fully specified by a mean function and a covariance
    function. This notebook puts that machinery to work in PyMC, in the
    two settings you will meet again and again.

    **Part A — the conjugate case.** With a Gaussian likelihood, the
    latent function $f$ can be integrated out analytically. PyMC exposes
    this as `pm.gp.Marginal`: fast, exact, and the right default whenever
    the observation noise is (approximately) Gaussian. We fit it to a
    single Theophylline subject's concentration curve.

    **Part B — the non-conjugate case.** With a non-Gaussian likelihood
    (here, Poisson counts), $f$ cannot be integrated out in closed form,
    so it must be sampled directly. PyMC exposes this as `pm.gp.Latent`.
    We fit it to the classic British coal-mining-disasters count series.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    import sys
    from pathlib import Path

    notebook_dir = mo.notebook_dir()
    if notebook_dir is None:
        raise RuntimeError("Marimo could not determine this notebook's directory.")
    project_root = notebook_dir.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from inference_contract import (
        eti,
        eti_bounds,
        inference_health,
        posterior_subset,
        sample_fresh_model_predictions,
    )
    from time import perf_counter

    import arviz as az
    import numpy as np
    import plotly.graph_objects as go
    import polars as pl
    import pymc as pm

    # PyMC brand colors, used throughout for consistency across notebooks.
    PYMC_BLUE = "#154A72"
    PYMC_GREEN = "#81C240"
    PYMC_LIGHT_BLUE = "#4A9EDE"
    PYMC_DARK_GREEN = "#40611F"

    RANDOM_SEED = 42
    is_script_mode = mo.app_meta().mode == "script"
    execute_models = is_script_mode or bool(
        mo.cli_args().get("execute-models", False)
    )
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    data_dir = project_root / "data"

    def z(a):
        """Standardize an array: (a - mean) / population std."""
        return (a - a.mean()) / a.std(ddof=0)

    return (
        PYMC_BLUE,
        PYMC_GREEN,
        PYMC_LIGHT_BLUE,
        RANDOM_SEED,
        az,
        data_dir,
        eti_bounds,
        execute_models,
        go,
        inference_health,
        np,
        perf_counter,
        pl,
        pm,
        posterior_subset,
        results_dir,
        sample_fresh_model_predictions,
        z,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part A: Marginal GP on the Theophylline curve

    Recall the Theophylline dataset from Notebook 1: 12 subjects, each
    with 11 serum-concentration measurements over 24 hours after a
    single oral dose. We again work with a single subject's curve — a
    smooth rise-then-decay shape with no natural parametric form.
    """)
    return


@app.cell
def _(data_dir, pl):
    theoph = pl.read_csv(data_dir / "theophylline.csv")
    return (theoph,)


@app.cell
def _(pl, theoph, z):
    subject_id = 1
    subj = theoph.filter(pl.col("subject") == subject_id).sort("time")

    time_vals = subj["time"].to_numpy()
    conc_vals = subj["conc"].to_numpy()

    time_mean, time_std = time_vals.mean(), time_vals.std(ddof=0)
    conc_mean, conc_std = conc_vals.mean(), conc_vals.std(ddof=0)

    X = z(time_vals).reshape(-1, 1)  # GP inputs are 2D: (n, 1)
    y = z(conc_vals)
    return (
        X,
        conc_mean,
        conc_std,
        conc_vals,
        subject_id,
        time_mean,
        time_std,
        time_vals,
        y,
    )


@app.cell
def _(PYMC_BLUE, conc_vals, go, subject_id, time_vals):
    data_fig = go.Figure()
    data_fig.add_trace(
        go.Scatter(
            x=time_vals,
            y=conc_vals,
            mode="markers+lines",
            name=f"Subject {subject_id}",
            line=dict(color=PYMC_BLUE),
            marker=dict(color=PYMC_BLUE, size=8),
        )
    )
    data_fig.update_layout(
        title=f"Theophylline concentration over time — Subject {subject_id}",
        xaxis_title="Time since dose (hours)",
        yaxis_title="Concentration (mg/L)",
        template="plotly_white",
    )
    data_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A naive first attempt

    Let's fit the most straightforward possible GP: a zero-mean
    `pm.gp.Marginal` with a `Matern52` covariance, evaluated directly on
    the **standardized raw time** axis. `pm.gp.Marginal` marginalizes
    the latent function analytically, so `.marginal_likelihood` gives us
    an exact Gaussian likelihood in the hyperparameters ($\ell$, $\eta$,
    $\sigma$) — no MCMC over $f$ itself is needed. We optimize with
    `pm.find_MAP` for a quick first look before committing to full
    sampling.
    """)
    return


@app.cell
def _(X, np, pm, y):
    def build_naive_marginal_gp_model(raw_X, raw_y):
        coords = {"observation": np.arange(len(raw_y)), "feature": ["time"]}
        with pm.Model(coords=coords) as model:
            x_data = pm.Data("X", raw_X, dims=("observation", "feature"))
            concentration = pm.Data("concentration", raw_y, dims="observation")
            ell = pm.LogNormal("ell", mu=0, sigma=1)
            eta = pm.HalfNormal("eta", sigma=1)
            sigma = pm.HalfNormal("sigma", sigma=0.5)
            gp = pm.gp.Marginal(
                mean_func=pm.gp.mean.Zero(),
                cov_func=eta**2 * pm.gp.cov.Matern52(1, ls=ell),
            )
            gp.marginal_likelihood(
                "y", X=x_data, y=concentration, sigma=sigma, dims="observation"
            )
        return model, gp

    naive_model, gp_naive = build_naive_marginal_gp_model(X, y)
    return gp_naive, naive_model


@app.cell
def _(RANDOM_SEED, naive_model, pm):
    with naive_model:
        naive_prior = pm.sample_prior_predictive(draws=300, random_seed=RANDOM_SEED)
    naive_prior_y = naive_prior["prior_predictive"]["y"]
    print(
        "Naive prior predictive 89% pointwise interval spans "
        f"{float(naive_prior_y.quantile(0.055)):.2f} to "
        f"{float(naive_prior_y.quantile(0.945)):.2f} on the standardized scale."
    )
    return


@app.cell
def _(mo):
    naive_map_button = mo.ui.run_button(label="Optimize naive MAP")
    naive_map_button
    return (naive_map_button,)


@app.cell
def _(execute_models, mo, naive_map_button, naive_model, pm):
    mo.stop(not (naive_map_button.value or execute_models))
    naive_model.compile_logp()(naive_model.initial_point())
    with naive_model:
        map_naive = pm.find_MAP(progressbar=False)
    print("Naive MAP optimization complete")
    return (map_naive,)


@app.cell
def _(
    PYMC_LIGHT_BLUE,
    conc_mean,
    conc_std,
    conc_vals,
    go,
    gp_naive,
    map_naive,
    naive_model,
    np,
    time_mean,
    time_std,
    time_vals,
):
    naive_grid = np.linspace(time_vals.min(), time_vals.max(), 200)
    naive_Xnew = ((naive_grid - time_mean) / time_std).reshape(-1, 1)

    with naive_model:
        naive_mu, naive_var = gp_naive.predict(naive_Xnew, point=map_naive, diag=True)

    naive_mu_orig = naive_mu * conc_std + conc_mean
    naive_sd_orig = np.sqrt(naive_var) * conc_std

    naive_fig = go.Figure()
    naive_fig.add_trace(
        go.Scatter(
            x=np.concatenate([naive_grid, naive_grid[::-1]]),
            y=np.concatenate(
                [
                    naive_mu_orig + 2 * naive_sd_orig,
                    (naive_mu_orig - 2 * naive_sd_orig)[::-1],
                ]
            ),
            fill="toself",
            fillcolor="rgba(74,158,222,0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="MAP mean ± 2 SD",
        )
    )
    naive_fig.add_trace(
        go.Scatter(
            x=naive_grid,
            y=naive_mu_orig,
            mode="lines",
            name="naive GP (MAP)",
            line=dict(color=PYMC_LIGHT_BLUE, width=3),
        )
    )
    naive_fig.add_trace(
        go.Scatter(
            x=time_vals,
            y=conc_vals,
            mode="markers",
            name="observed",
            marker=dict(color="black", size=8),
        )
    )
    naive_fig.update_layout(
        title="Naive GP (raw standardized time) — struggles with rise + decay",
        xaxis_title="Time since dose (hours)",
        yaxis_title="Concentration (mg/L)",
        template="plotly_white",
    )
    naive_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Why the naive fit struggles

    The observed time points are very unevenly spaced: dense near
    $t=0$ (0, 0.25, 0.57, 1.12, 2.02 hours — where concentration is
    rising fastest) and sparse in the tail (9.05, 12.12, 24.37 hours —
    where it is decaying slowly). A single global lengthscale
    $\ell$ has to compromise between these two regimes: fine enough to
    catch the fast rise, coarse enough not to overfit noise in the long,
    sparsely-sampled tail. The MAP fit above visibly overshoots the
    early rise and is under-constrained past $t \approx 12$.

    Two standard fixes: **transform the input** so that the regions of
    genuinely different curvature are given comparable spacing (here,
    `log1p(time)` compresses the long tail without needing a second
    lengthscale), and/or **add a mean function** so the GP only needs to
    model the residual departure from a simple parametric trend rather
    than the whole curve. We'll use both below.
    """)
    return


@app.cell
def _(np, time_vals, z):
    log_time = np.log1p(time_vals)
    log_time_mean, log_time_std = log_time.mean(), log_time.std(ddof=0)
    X_log = z(log_time).reshape(-1, 1)
    return X_log, log_time_mean, log_time_std


@app.cell
def _(np, pm):
    def build_marginal_gp_model(X, y, *, X_pred=None, pred_coord=None):
        coords = {"observation": np.arange(len(y)), "feature": ["log_time"]}
        if X_pred is not None:
            coords["prediction"] = (
                np.asarray(pred_coord)
                if pred_coord is not None
                else np.arange(len(X_pred))
            )
        with pm.Model(coords=coords) as model:
            x_data = pm.Data("X", X, dims=("observation", "feature"))
            concentration = pm.Data("concentration", y, dims="observation")
            beta = pm.Normal("beta", mu=0, sigma=1, dims="feature")
            intercept = pm.Normal("intercept", mu=0, sigma=1)
            ell = pm.LogNormal("ell", mu=0, sigma=1)
            eta = pm.HalfNormal("eta", sigma=1)
            sigma = pm.HalfNormal("sigma", sigma=0.5)
            gp = pm.gp.Marginal(
                mean_func=pm.gp.mean.Linear(coeffs=beta, intercept=intercept),
                cov_func=eta**2 * pm.gp.cov.Matern52(1, ls=ell),
            )
            gp.marginal_likelihood(
                "y", X=x_data, y=concentration, sigma=sigma, dims="observation"
            )
            if X_pred is not None:
                x_pred = pm.Data("X_pred", X_pred, dims=("prediction", "feature"))
                gp.conditional("f_pred", x_pred, dims="prediction")
                gp.conditional(
                    "f_pred_noise", x_pred, pred_noise=True, dims="prediction"
                )
        return model, gp

    return (build_marginal_gp_model,)


@app.cell
def _(X_log, build_marginal_gp_model, y):
    gp_model, _structured_gp = build_marginal_gp_model(X_log, y)
    return (gp_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Prior predictive check

    As always, we look at what the model implies *before* fitting.
    """)
    return


@app.cell
def _(RANDOM_SEED, gp_model, pm):
    with gp_model:
        prior_pred = pm.sample_prior_predictive(draws=500, random_seed=RANDOM_SEED)
    return (prior_pred,)


@app.cell
def _(PYMC_LIGHT_BLUE, RANDOM_SEED, X_log, go, np, prior_pred, y):
    prior_draws = (
        prior_pred["prior_predictive"]["y"]
        .stack(sample=("chain", "draw"))
        .transpose("sample", "observation")
    )
    prior_fig = go.Figure()
    rng_plot = np.random.default_rng(RANDOM_SEED)
    for prior_draw_index in rng_plot.choice(prior_draws.sizes["sample"], size=50, replace=False):
        prior_fig.add_trace(
            go.Scatter(
                x=X_log[:, 0],
                y=prior_draws.isel(sample=prior_draw_index).values,
                mode="lines",
                line=dict(color=PYMC_LIGHT_BLUE, width=1),
                opacity=0.25,
                showlegend=False,
            )
        )
    prior_fig.add_trace(
        go.Scatter(
            x=X_log[:, 0],
            y=y,
            mode="markers",
            marker=dict(color="black", size=8),
            name="observed (standardized)",
        )
    )
    prior_fig.update_layout(
        title="Prior predictive draws vs. standardized observed data",
        xaxis_title="log1p(time), standardized",
        yaxis_title="conc (standardized)",
        template="plotly_white",
    )
    prior_fig
    return (prior_draws,)


@app.cell(hide_code=True)
def _(mo, prior_draws, y):
    mo.md(
        f"""
        **Prior implications:** the 500 draws span
        [{float(prior_draws.min()):.2f}, {float(prior_draws.max()):.2f}] on the
        standardized scale, compared with observed values from
        [{y.min():.2f}, {y.max():.2f}]. This is a prior-range check, not evidence
        that the model fits the data.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Conjugacy: why the latent function integrates out

    `pm.gp.Marginal` earns its name from a piece of algebra worth seeing
    once. Write the model at the observed inputs $X$ as a latent Gaussian
    vector $\mathbf f \sim \mathcal N(\mathbf m,\ K)$ with $K = k(X, X)$,
    and a Gaussian observation layer
    $\mathbf y \mid \mathbf f \sim \mathcal N(\mathbf f,\ \sigma^2 I)$.
    Because *both* pieces are Gaussian, the joint over
    $(\mathbf f, \mathbf y)$ is Gaussian, and a Gaussian integrated over one
    of its blocks is Gaussian again — the **marginalization** property from
    Notebook 1. Integrating $\mathbf f$ out gives the **marginal
    likelihood** in closed form:

    $$\mathbf y \sim \mathcal N\big(\mathbf m,\ K + \sigma^2 I\big).$$

    No latent $\mathbf f$ appears — the function values have been absorbed
    into the $n \times n$ covariance $K + \sigma^2 I$. This is exactly what
    `.marginal_likelihood("y", X=X, y=y, sigma=sigma)` evaluates. The only
    free unknowns left are the handful of hyperparameters
    $(\ell, \eta, \sigma)$ plus the two mean-function coefficients — **five
    scalars**, instead of $\mathbf f$'s 11 values *and* the hyperparameters.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The same conjugacy delivers predictions in closed form. For test inputs
    $X_*$, the posterior over the latent function is Gaussian with

    $$\mathbb E[\mathbf f_* \mid \mathbf y]
    = \mathbf m_* + K_*(K + \sigma^2 I)^{-1}(\mathbf y - \mathbf m),$$

    $$\operatorname{Cov}[\mathbf f_* \mid \mathbf y]
    = K_{**} - K_*(K + \sigma^2 I)^{-1}K_*^\top,$$

    with $K_* = k(X_*, X)$ and $K_{**} = k(X_*, X_*)$ — the identical
    conditioning formula you applied by hand in Notebook 1, now with a
    learned mean function $\mathbf m$ subtracted off first. Two consequences
    we lean on below:

    - **Every hyperparameter draw yields an *exact* predictive Gaussian.**
      There is no Monte-Carlo error in $\mathbf f$ given
      $(\ell, \eta, \sigma)$; the only thing we sample over is the
      hyperparameters themselves.
    - **The $(K + \sigma^2 I)^{-1}$ solve is $O(n^3)$.** With $n = 11$ that
      is nothing, but it is the exact cost that motivates the sparse and
      HSGP approximations in Hour 4. Conjugacy buys exactness at a cubic
      price.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### MAP vs. full posterior

    `pm.gp.Marginal` only ever samples the **hyperparameters**
    ($\ell$, $\eta$, $\sigma$, and the two mean-function coefficients) —
    five scalars, since $f$ itself is marginalized out analytically.
    That makes both a MAP optimization and full MCMC cheap; let's do
    both and compare.
    """)
    return


@app.cell
def _(execute_models, gp_model, mo, pm, structured_map_button):
    mo.stop(not (structured_map_button.value or execute_models))
    gp_model.compile_logp()(gp_model.initial_point())
    with gp_model:
        map_estimate = pm.find_MAP(progressbar=False)
    return (map_estimate,)


@app.cell
def _(mo):
    structured_map_button = mo.ui.run_button(label="Optimize structured MAP")
    structured_map_button
    return (structured_map_button,)


@app.cell
def _(mo):
    structured_fit_button = mo.ui.run_button(label="Fit structured GP")
    structured_fit_button
    return (structured_fit_button,)


@app.cell
def _(execute_models, gp_model, mo, pm, results_dir, structured_fit_button):
    mo.stop(not (structured_fit_button.value or execute_models))
    gp_model.compile_logp()(gp_model.initial_point())
    with gp_model:
        idata = pm.sample(
            chains=4,
            draws=500,
            tune=500,
            # The default geometry produced 23 divergences at this model's
            # near-zero-noise boundary; 0.99 resolves that observed failure.
            target_accept=0.99,
            init="adapt_diag",
        )
    idata.to_netcdf(results_dir / "02_marginal_gp.nc")
    return (idata,)


@app.cell
def _(RANDOM_SEED, gp_model, idata, pm, posterior_subset, results_dir):
    with gp_model:
        observed_ppc = pm.sample_posterior_predictive(
            posterior_subset(idata, draws_per_chain=100),
            var_names=["y"],
            random_seed=RANDOM_SEED,
        )
    idata_with_ppc = idata.copy()
    idata_with_ppc["posterior_predictive"] = observed_ppc["posterior_predictive"]
    idata_with_ppc.to_netcdf(results_dir / "02_marginal_gp.nc")
    return (idata_with_ppc,)


@app.cell
def _(mo):
    coal_fit_button = mo.ui.run_button(label="Fit coal latent GP")
    coal_fit_button
    return (coal_fit_button,)


@app.cell
def _(gp_model, idata, inference_health):
    summary, health_passed = inference_health(idata, gp_model)
    n_div = summary.attrs["divergences"]
    n_draws_total = idata["posterior"].sizes["chain"] * idata["posterior"].sizes["draw"]
    min_ess_bulk = float(summary["ess_bulk"].min())
    min_ess_tail = float(summary["ess_tail"].min())
    max_rhat = float(summary["r_hat"].astype(float).max())
    print(f"Divergences: {n_div} / {n_draws_total}; health passed: {health_passed}")
    summary
    return (
        health_passed,
        max_rhat,
        min_ess_bulk,
        min_ess_tail,
        n_div,
        n_draws_total,
    )


@app.cell
def _(az, gp_model, idata):
    posterior_summary = az.summary(
        idata, var_names=[rv.name for rv in gp_model.free_RVs], kind="stats"
    )
    return (posterior_summary,)


@app.cell(hide_code=True)
def _(
    health_passed,
    map_estimate,
    max_rhat,
    min_ess_bulk,
    min_ess_tail,
    mo,
    n_div,
    n_draws_total,
    posterior_summary,
):
    mo.md(
        f"""
        **Diagnostics:** {n_div} divergence(s) out of {n_draws_total} draws.
        Minimum `ess_bulk` is {min_ess_bulk:.0f}, minimum `ess_tail` is
        {min_ess_tail:.0f}, and maximum `r_hat` is {max_rhat:.3f}; the computed
        all-free-variable health status is **{health_passed}**. Interpret the
        posterior only if that status is true.

        **MAP vs. posterior mean** (lengthscale $\\ell$): MAP gives
        {map_estimate["ell"]:.3f}; the full posterior mean is
        {float(posterior_summary.loc["ell", "mean"]):.3f}. MAP and its predictive curve
        are plug-in summaries: they condition on one hyperparameter point rather
        than integrating hyperparameter uncertainty.
        """
    )
    return


@app.cell
def _(az, gp_model, idata):
    _structured_free_rv_names = [rv.name for rv in gp_model.free_RVs]
    az.plot_trace_dist(idata, var_names=_structured_free_rv_names, compact=True)
    az.plot_rank(idata, var_names=_structured_free_rv_names)
    return


@app.cell
def _(map_estimate, np, pl, posterior_summary):
    beta_row = next(index for index in posterior_summary.index if index.startswith("beta"))
    map_vs_post = pl.DataFrame(
        {
            "parameter": ["ell", "eta", "sigma", "intercept", beta_row],
            "MAP (plug-in)": [
                round(float(map_estimate["ell"]), 3),
                round(float(map_estimate["eta"]), 3),
                round(float(map_estimate["sigma"]), 3),
                round(float(map_estimate["intercept"]), 3),
                round(float(np.ravel(map_estimate["beta"])[0]), 3),
            ],
            "posterior_mean": [
                round(float(posterior_summary.loc["ell", "mean"]), 3),
                round(float(posterior_summary.loc["eta", "mean"]), 3),
                round(float(posterior_summary.loc["sigma", "mean"]), 3),
                round(float(posterior_summary.loc["intercept", "mean"]), 3),
                round(float(posterior_summary.loc[beta_row, "mean"]), 3),
            ],
        }
    )
    map_vs_post
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The table lines up the two point summaries for all five
    hyperparameters. They agree to within a posterior standard deviation
    here, which is the *good* case: a low-dimensional, well-identified
    hyperparameter posterior that is roughly symmetric, so its mode (MAP)
    and its mean nearly coincide. When should you trust MAP, and when not?

    - **MAP is appropriate** as a fast first look, for initialization, or
      when the hyperparameter posterior is tight and unimodal and you
      genuinely only need a point estimate. It is a single optimization —
      cheap and deterministic.
    - **MAP misses** everything about the *shape* of the posterior. It
      reports no uncertainty, so it cannot tell you the lengthscale is only
      known to within some range; it sits at the mode, which for a skewed
      posterior (lengthscales often are) can be far from the mean; it can
      settle on a local optimum; and — the subtle one — a MAP *prediction*
      plugs in a single hyperparameter value, so its predictive band
      understates uncertainty because it ignores the spread over
      $(\ell, \eta, \sigma)$ entirely.

    Full MCMC propagates hyperparameter uncertainty into every prediction:
    we draw many $(\ell, \eta, \sigma)$, form the exact predictive Gaussian
    for each (conjugacy again), and mix them. That mixture is wider — and
    more honest — than any single MAP curve, which is why the fit below uses
    the full posterior rather than `map_estimate`.
    """)
    return


@app.cell
def _(log_time_mean, log_time_std, np, time_vals):
    time_grid = np.linspace(time_vals.min(), time_vals.max(), 200)
    Xnew = ((np.log1p(time_grid) - log_time_mean) / log_time_std).reshape(-1, 1)
    return Xnew, time_grid


@app.cell
def _(
    RANDOM_SEED,
    X_log,
    Xnew,
    build_marginal_gp_model,
    idata,
    sample_fresh_model_predictions,
    time_grid,
    y,
):
    structured_predictions = sample_fresh_model_predictions(
        idata,
        lambda: build_marginal_gp_model(
            X_log, y, X_pred=Xnew, pred_coord=time_grid
        )[0],
        var_names=["f_pred", "f_pred_noise"],
        random_seed=RANDOM_SEED,
    )
    return (structured_predictions,)


@app.cell
def _(
    PYMC_BLUE,
    conc_mean,
    conc_std,
    conc_vals,
    eti_bounds,
    go,
    np,
    structured_predictions,
    time_grid,
    time_vals,
):
    latent_draws = structured_predictions["predictions"]["f_pred"].rename(
        {"prediction": "time_grid"}
    )
    structured_noisy_draws = structured_predictions["predictions"]["f_pred_noise"].rename(
        {"prediction": "time_grid"}
    )
    latent_draws = latent_draws * conc_std + conc_mean
    structured_noisy_draws = structured_noisy_draws * conc_std + conc_mean
    latent_mean = latent_draws.mean(dim=("chain", "draw"))
    latent_low, latent_high = eti_bounds(latent_draws)
    structured_noisy_low, structured_noisy_high = eti_bounds(structured_noisy_draws)

    pred_fig = go.Figure()
    pred_fig.add_trace(
        go.Scatter(
            x=np.concatenate([time_grid, time_grid[::-1]]),
            y=np.concatenate([structured_noisy_high.values, structured_noisy_low.values[::-1]]),
            fill="toself",
            fillcolor="rgba(74,158,222,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="89% ETI (new measurement)",
        )
    )
    pred_fig.add_trace(
        go.Scatter(
            x=np.concatenate([time_grid, time_grid[::-1]]),
            y=np.concatenate([latent_high.values, latent_low.values[::-1]]),
            fill="toself",
            fillcolor="rgba(21,74,114,0.35)",
            line=dict(color="rgba(255,255,255,0)"),
            name="89% ETI (latent f)",
        )
    )
    pred_fig.add_trace(
        go.Scatter(
            x=time_grid,
            y=latent_mean.values,
            mode="lines",
            name="full-posterior mean",
            line=dict(color=PYMC_BLUE, width=3),
        )
    )
    pred_fig.add_trace(
        go.Scatter(
            x=time_vals,
            y=conc_vals,
            mode="markers",
            name="observed",
            marker=dict(color="black", size=8),
        )
    )
    pred_fig.update_layout(
        title="Full-posterior marginal-GP prediction",
        xaxis_title="Time since dose (hours)",
        yaxis_title="Concentration (mg/L)",
        template="plotly_white",
    )
    pred_fig
    return


@app.cell
def _(az, idata_with_ppc):
    az.plot_ppc_dist(idata_with_ppc, var_names=["y"], kind="ecdf", num_samples=50)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The narrower band is uncertainty in the latent function $f$ alone
    (`pred_noise=False`); the wider band additionally folds in the
    observation noise $\sigma$ (`pred_noise=True`) and is the right one
    to compare against *new, unobserved measurements*. The transformed
    input plus linear mean function now trace the rise, peak, and decay
    far more faithfully than the naive fit.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What `pred_noise` actually changes

    The two bands differ by exactly one term, and it is worth being precise
    about which. Writing the predictive variance at a test point $x_*$:

    - `pred_noise=False` returns $\operatorname{Var}[f(x_*) \mid \mathbf y]$
      — the uncertainty in the *latent function* $f$ at $x_*$. Use this when
      you care about the underlying smooth curve ("what is the true
      concentration trajectory?").
    - `pred_noise=True` returns
      $\operatorname{Var}[f(x_*) \mid \mathbf y] + \sigma^2$ — the latent
      uncertainty *plus* the observation-noise variance. Use this when you
      care about a *new measurement* ("what will the assay read if I draw
      blood at 5 hours?"), because a real measurement carries the same
      $\sigma$ scatter the training points did.

    The gap between the bands is therefore a **constant $\sigma^2$ in
    variance** — uniform across the whole grid, since observation noise does
    not depend on *where* you predict. That is why the outer band sits a
    fixed vertical distance outside the inner one even where the data are
    dense and the latent band is tight. The cell below reads off that
    constant gap in the original mg/L units.
    """)
    return


@app.cell
def _(conc_std, posterior_summary):
    sigma_post_mean = float(posterior_summary.loc["sigma", "mean"])
    noise_var_mgL = (sigma_post_mean * conc_std) ** 2
    print(f"Posterior-mean sigma (standardized): {sigma_post_mean:.3f}")
    print(f"Constant variance gap between the bands: {noise_var_mgL:.3f} (mg/L)^2")
    print(f"  i.e. a noise standard deviation of {sigma_post_mean * conc_std:.3f} mg/L")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise: extrapolate with full posterior uncertainty

    Use the fresh prediction model and posterior draws to predict to 40 hours
    past dose — well beyond the last observation at 24.37 hours. Contrast the
    full-posterior 89% ETI with a plug-in MAP prediction: which uncertainty
    sources does the latter omit, and toward what specified mean function does
    the full prediction return?
    """)
    return


@app.cell(hide_code=True)
def _(
    PYMC_BLUE,
    RANDOM_SEED,
    X_log,
    build_marginal_gp_model,
    conc_mean,
    conc_std,
    conc_vals,
    eti_bounds,
    go,
    idata,
    log_time_mean,
    log_time_std,
    mo,
    np,
    sample_fresh_model_predictions,
    time_vals,
    y,
):
    extrap_grid = np.linspace(0, 40, 200)
    extrap_X = ((np.log1p(extrap_grid) - log_time_mean) / log_time_std).reshape(-1, 1)
    extrap_predictions = sample_fresh_model_predictions(
        idata,
        lambda: build_marginal_gp_model(
            X_log, y, X_pred=extrap_X, pred_coord=extrap_grid
        )[0],
        var_names=["f_pred_noise"],
        random_seed=RANDOM_SEED,
    )
    extrap_noisy_draws = (
        extrap_predictions["predictions"]["f_pred_noise"]
        .rename({"prediction": "time"})
        * conc_std
        + conc_mean
    )
    extrap_noisy_mean = extrap_noisy_draws.mean(dim=("chain", "draw"))
    extrap_noisy_low, extrap_noisy_high = eti_bounds(extrap_noisy_draws)

    extrap_fig = go.Figure()
    extrap_fig.add_trace(
        go.Scatter(
            x=np.concatenate([extrap_grid, extrap_grid[::-1]]),
            y=np.concatenate([extrap_noisy_high.values, extrap_noisy_low.values[::-1]]),
            fill="toself",
            fillcolor="rgba(21,74,114,0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="full-posterior 89% ETI",
        )
    )
    extrap_fig.add_trace(
        go.Scatter(
            x=extrap_grid,
            y=extrap_noisy_mean.values,
            mode="lines",
            name="full-posterior mean",
            line=dict(color=PYMC_BLUE, width=2),
        )
    )
    extrap_fig.add_trace(
        go.Scatter(
            x=time_vals,
            y=conc_vals,
            mode="markers",
            name="observed",
            marker=dict(color="black", size=8),
        )
    )
    extrap_fig.add_vline(x=time_vals.max(), line=dict(dash="dash", color="gray"))
    extrap_fig.update_layout(
        title="Full-posterior extrapolation past the last observation",
        xaxis_title="Time since dose (hours)",
        yaxis_title="Concentration (mg/L)",
        template="plotly_white",
    )
    mo.md(
        "Past the observed range, the full-posterior prediction returns toward "
        "the specified linear mean and its uncertainty widens."
    )
    mo.accordion({"Solution": extrap_fig})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md(
                r"""
                **Exercise — what does the linear mean function buy us?**
                Suppose you removed `pm.gp.mean.Linear` and used a zero-mean GP
                on the same `log1p(time)` input. Predict the consequences both
                within the observed range and beyond 24 hours before expanding
                the discussion.
                """
            ),
            mo.accordion(
                {
                    "Discussion": mo.md(
                        r"""
                        Within the data, a flexible GP could still trace much of
                        the curve. The difference is clearest in extrapolation
                        and in the division of labor: a zero-mean GP returns
                        toward standardized zero, while this model returns toward
                        its learned linear mean. Without that mean, the
                        covariance must explain both broad drift and local
                        curvature. The linear mean encodes the named global
                        trend and leaves the Matérn residual GP to express
                        departures from it; full-posterior prediction retains
                        uncertainty in both pieces.
                        """
                    )
                }
            ),
            mo.md(
                r"""
                **Exercise — assess the Matérn 5/2 prior implication.** Before
                changing kernels, compare prior draws and the observed-data
                posterior-predictive discrepancy. What feature of the
                pharmacokinetic curve would make an infinitely smooth ExpQuad
                prior implausible, and what would you look for in the PPC after
                the change?
                """
            ),
            mo.accordion(
                {
                    "Discussion": mo.md(
                        r"""
                        ExpQuad imposes exceptionally smooth latent functions,
                        whereas the early absorption rise can require more local
                        flexibility. Matérn 5/2 permits a continuous but less
                        rigid curve. This is not a claim that one kernel is
                        universally better: inspect the prior curves first, then
                        compare the observed residual and replicated-data
                        discrepancies. A visually smoother posterior mean alone
                        is not evidence that the observation model improved.
                        """
                    )
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part B: Latent Poisson GP on coal-mining disasters

    ### Background

    The `coal_disasters` dataset records the number of British
    coal-mining disasters causing 10 or more deaths, by year, from 1851
    to 1962 (112 annual counts) — a classic changepoint / intensity-
    estimation dataset in the Bayesian literature (Raftery & Akman,
    1986). The rate of disasters is believed to have declined over this
    period (improved safety regulation), but not necessarily smoothly
    or via any particular parametric form — another natural fit for a
    GP, this time on the log-rate of a **Poisson** count process.
    """)
    return


@app.cell
def _(data_dir, pl):
    coal = pl.read_csv(data_dir / "coal_disasters.csv")
    return (coal,)


@app.cell
def _(coal, z):
    year_vals = coal["year"].to_numpy()
    disaster_counts = coal["disasters"].to_numpy()

    t = z(year_vals).reshape(-1, 1)  # GP inputs are 2D: (n, 1)
    return disaster_counts, t, year_vals


@app.cell
def _(PYMC_BLUE, disaster_counts, go, year_vals):
    coal_fig = go.Figure()
    coal_fig.add_trace(
        go.Bar(
            x=year_vals,
            y=disaster_counts,
            marker=dict(color=PYMC_BLUE),
            name="disasters",
        )
    )
    coal_fig.update_layout(
        title="British coal-mining disasters (≥10 deaths) per year, 1851–1962",
        xaxis_title="Year",
        yaxis_title="Disasters",
        template="plotly_white",
    )
    coal_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Why we can't marginalize here

    `pm.gp.Marginal` relies on the Gaussian likelihood's conjugacy: if
    $y \mid f \sim \mathcal{N}(f, \sigma^2)$ and $f$ is a GP, then $f$
    can be integrated out in closed form, leaving a tractable marginal
    likelihood in $y$ alone. A Poisson likelihood,
    $y \mid f \sim \mathrm{Poisson}(e^f)$, has no such conjugate form —
    there is no analytic way to integrate $f$ out of the joint
    distribution. So instead of marginalizing, `pm.gp.Latent` keeps the
    vector of latent function values $f$ as an explicit set of
    parameters (one per observed year here) and lets NUTS sample them
    jointly with the covariance hyperparameters. This is more
    expensive — we are now sampling a 112-dimensional latent vector
    plus hyperparameters, rather than 2–5 hyperparameters alone — but
    it is the only exact option once the likelihood leaves the Gaussian
    family.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### From latent function to disaster rate: the exp link

    It is worth tracing the generative chain the model encodes, because it
    is the template for *every* non-Gaussian GP. For each year $t_i$:

    1. A **latent function value** $f(t_i)$ is drawn jointly from the GP
       prior — these are the 112 correlated values `f = gp.prior("f", X=t)`.
       On this scale $f$ is unconstrained: it can be any real number,
       positive or negative, and neighbouring years are correlated through
       the Matérn kernel.
    2. An **exp link** maps it to a positive rate,
       $\lambda(t_i) = \exp\!\big(f(t_i)\big)$. Exponentiating guarantees
       $\lambda > 0$ (a Poisson mean must be positive) and makes the GP a
       model of the **log-rate**, so a straight-line drop in $f$ is a
       constant *proportional* decline in the rate — the natural scale for
       counts.
    3. A **Poisson likelihood** turns the rate into observed counts,
       $y_i \sim \mathrm{Poisson}\big(\lambda(t_i)\big)$.

    Contrast this directly with Part A. There, the link was the identity and
    the noise was Gaussian, so step 1's latent values integrated out and we
    sampled only five hyperparameters. Here the exp link and Poisson
    likelihood break conjugacy: there is no closed form for
    $\int \prod_i \mathrm{Poisson}(y_i \mid e^{f_i})\,
    \mathcal N(\mathbf f \mid \mathbf 0, K)\, d\mathbf f$, so we have no
    choice but to **keep $\mathbf f$ in the model and let NUTS explore all
    112 values jointly** with $(\ell, \eta)$. Carrying $\mathbf f$ explicitly
    is the price of leaving the Gaussian family — and the reason latent GPs
    sample more slowly than marginal ones.
    """)
    return


@app.cell
def _(disaster_counts, np, pm, t, year_vals):
    coords = {"year": year_vals, "feature": ["standardized_year"]}
    with pm.Model(coords=coords) as coal_model:
        year_data = pm.Data("year_input", t, dims=("year", "feature"))
        count_data = pm.Data("disaster_count", disaster_counts, dims="year")
        alpha = pm.Normal("alpha", mu=np.log(1.5), sigma=0.5)
        ell = pm.LogNormal("ell", mu=0, sigma=0.5)
        eta = pm.HalfNormal("eta", sigma=1)
        coal_gp = pm.gp.Latent(cov_func=eta**2 * pm.gp.cov.Matern52(1, ls=ell))
        f = coal_gp.prior("f", X=year_data, dims="year")
        log_rate = pm.Deterministic("log_rate", alpha + f, dims="year")
        rate = pm.Deterministic("rate", pm.math.exp(log_rate), dims="year")
        pm.Poisson("y", mu=rate, observed=count_data, dims="year")
    return (coal_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Prior predictive check
    """)
    return


@app.cell
def _(RANDOM_SEED, coal_model, pm):
    with coal_model:
        coal_prior_pred = pm.sample_prior_predictive(draws=500, random_seed=RANDOM_SEED)
    return (coal_prior_pred,)


@app.cell
def _(coal_prior_pred, disaster_counts, eti_bounds, go, np, year_vals):
    coal_prior_counts = coal_prior_pred["prior_predictive"]["y"]
    coal_prior_rate = coal_prior_pred["prior"]["rate"]
    prior_count_low, prior_count_high = eti_bounds(coal_prior_counts)
    rate_low, rate_high = eti_bounds(coal_prior_rate)

    coal_prior_fig = go.Figure()
    coal_prior_fig.add_trace(
        go.Scatter(
            x=np.concatenate([year_vals, year_vals[::-1]]),
            y=np.concatenate([prior_count_high.values, prior_count_low.values[::-1]]),
            fill="toself",
            fillcolor="rgba(129,194,64,0.25)",
            line=dict(color="rgba(255,255,255,0)"),
            name="89% prior-predictive count ETI",
        )
    )
    coal_prior_fig.add_trace(
        go.Scatter(
            x=year_vals,
            y=disaster_counts,
            mode="markers",
            marker=dict(color="black", size=6),
            name="observed counts",
        )
    )
    coal_prior_fig.add_trace(
        go.Scatter(
            x=year_vals,
            y=coal_prior_rate.mean(dim=("chain", "draw")).values,
            mode="lines",
            line=dict(color="#154A72", dash="dot"),
            name="prior mean rate",
        )
    )
    coal_prior_fig.update_layout(
        title="Prior rate and count trajectories",
        xaxis_title="Year",
        yaxis_title="Disasters / year",
        template="plotly_white",
    )
    coal_prior_fig
    return (coal_prior_counts,)


@app.cell(hide_code=True)
def _(coal_prior_counts, disaster_counts, mo):
    mo.md(
        f"""
        **Prior implications:** simulated counts range from
        {float(coal_prior_counts.min()):.0f} to {float(coal_prior_counts.max()):.0f};
        the observed range is [{disaster_counts.min()}, {disaster_counts.max()}].
        This checks whether the stated priors generate plausible rate and count
        scales before conditioning on the data.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Sampling

    112 latent function values plus 2 covariance hyperparameters. We use
    four chains and the default PyMC sampler settings, then inspect
    divergences, R-hat, and effective sample sizes across all 112 latent
    values rather than treating a sampler setting as a substitute for
    identified geometry.
    """)
    return


@app.cell
def _(
    RANDOM_SEED,
    coal_fit_button,
    coal_model,
    execute_models,
    mo,
    perf_counter,
    pm,
    results_dir,
):
    mo.stop(not (coal_fit_button.value or execute_models))
    coal_model.compile_logp()(coal_model.initial_point())
    with coal_model:
        _start = perf_counter()
        coal_idata = pm.sample(
            random_seed=RANDOM_SEED,
            draws=800,
            tune=800,
            chains=4,
            init="adapt_diag",
            # correlated latent field; 0.99 resolves that observed failure.
            target_accept=0.99,
        )
        coal_sample_seconds = perf_counter() - _start
    coal_idata.to_netcdf(results_dir / "02_coal_latent_gp.nc")
    print(f"Coal latent-GP sampling wall-time: {coal_sample_seconds:.1f}s")
    return coal_idata, coal_sample_seconds


@app.cell
def _(coal_idata, coal_model, inference_health):
    coal_summary, coal_health_passed = inference_health(coal_idata, coal_model)
    coal_n_div = coal_summary.attrs["divergences"]
    coal_n_draws_total = (
        coal_idata["posterior"].sizes["chain"] * coal_idata["posterior"].sizes["draw"]
    )
    coal_min_ess_bulk = float(coal_summary["ess_bulk"].min())
    coal_min_ess_tail = float(coal_summary["ess_tail"].min())
    coal_max_rhat = float(coal_summary["r_hat"].astype(float).max())
    print(f"Divergences: {coal_n_div} / {coal_n_draws_total}")
    print(
        f"Min ess_bulk / ess_tail (all free RVs): {coal_min_ess_bulk:.0f} / {coal_min_ess_tail:.0f}; health passed: {coal_health_passed}"
    )
    return (
        coal_health_passed,
        coal_max_rhat,
        coal_min_ess_bulk,
        coal_min_ess_tail,
        coal_n_div,
        coal_n_draws_total,
    )


@app.cell(hide_code=True)
def _(
    coal_health_passed,
    coal_max_rhat,
    coal_min_ess_bulk,
    coal_min_ess_tail,
    coal_n_div,
    coal_n_draws_total,
    coal_sample_seconds,
    mo,
):
    mo.md(
        f"""
        **Diagnostics:** {coal_n_div} divergences out of
        {coal_n_draws_total} draws in {coal_sample_seconds:.1f}s. Minimum
        `ess_bulk` is {coal_min_ess_bulk:.0f}, minimum `ess_tail` is
        {coal_min_ess_tail:.0f}, and maximum `r_hat` is {coal_max_rhat:.3f}.
        The all-free-variable health status is **{coal_health_passed}**; any
        substantive interpretation is conditional on this status.
        """
    )
    return


@app.cell
def _(PYMC_BLUE, coal_idata, disaster_counts, eti_bounds, go, np, year_vals):
    _rate_posterior = coal_idata["posterior"]["rate"]
    _rate_mean = _rate_posterior.mean(dim=("chain", "draw"))
    _rate_lo, _rate_hi = eti_bounds(_rate_posterior)

    rate_fig = go.Figure()
    rate_fig.add_trace(
        go.Scatter(
            x=np.concatenate([year_vals, year_vals[::-1]]),
            y=np.concatenate([_rate_hi.values, _rate_lo.values[::-1]]),
            fill="toself",
            fillcolor="rgba(21,74,114,0.25)",
            line=dict(color="rgba(255,255,255,0)"),
            name="89% ETI",
        )
    )
    rate_fig.add_trace(
        go.Scatter(
            x=year_vals,
            y=_rate_mean.values,
            mode="lines",
            name="posterior mean rate",
            line=dict(color=PYMC_BLUE, width=3),
        )
    )
    rate_fig.add_trace(
        go.Scatter(
            x=year_vals,
            y=disaster_counts,
            mode="markers",
            name="observed disasters",
            marker=dict(color="black", size=6),
        )
    )
    rate_fig.update_layout(
        title="Posterior latent rate exp(f) vs. observed disaster counts",
        xaxis_title="Year",
        yaxis_title="Disasters / rate",
        template="plotly_white",
    )
    rate_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The posterior is a distribution over rate *functions*

    The 89% ETI band above summarizes the posterior, but it can hide that each
    posterior draw is a whole *function*. Plotting sixty individual draws of
    $\exp(f)$ makes the object concrete: the model is uncertain not about a
    few numbers but about the entire trajectory, and the draws fan out where
    data are sparse (the early years, few disasters to pin the rate) and
    pull together where the counts are more informative.
    """)
    return


@app.cell
def _(PYMC_GREEN, RANDOM_SEED, coal_idata, go, np, year_vals):
    rate_draws = (
        coal_idata["posterior"]["rate"]
        .stack(sample=("chain", "draw"))
        .transpose("sample", "year")
    )
    rng = np.random.default_rng(RANDOM_SEED)
    spaghetti_fig = go.Figure()
    for rate_draw_index in rng.choice(rate_draws.sizes["sample"], size=60, replace=False):
        spaghetti_fig.add_trace(
            go.Scatter(
                x=year_vals,
                y=rate_draws.isel(sample=rate_draw_index).values,
                mode="lines",
                line=dict(color=PYMC_GREEN, width=1),
                opacity=0.15,
                showlegend=False,
            )
        )
    spaghetti_fig.update_layout(
        title="Sixty posterior draws of the annual rate",
        xaxis_title="Year",
        yaxis_title="Disasters / year (rate)",
        template="plotly_white",
    )
    spaghetti_fig
    return


@app.cell
def _(coal_idata, eti_bounds):
    selected_rates = coal_idata["posterior"]["rate"].sel(year=[1851, 1900, 1962])

    def summarize_rate(year):
        rate_at_year = selected_rates.sel(year=year)
        lower, upper = eti_bounds(rate_at_year)
        return (
            float(rate_at_year.mean(dim=("chain", "draw"))),
            (float(lower), float(upper)),
        )

    rate_1851 = summarize_rate(1851)
    rate_1900 = summarize_rate(1900)
    rate_1962 = summarize_rate(1962)
    pct_decline = 100 * (1 - rate_1962[0] / rate_1851[0])
    print(
        f"Mean rates at 1851, 1900, 1962: {rate_1851[0]:.2f}, "
        f"{rate_1900[0]:.2f}, {rate_1962[0]:.2f} disasters/year."
    )
    return pct_decline, rate_1851, rate_1900, rate_1962


@app.cell(hide_code=True)
def _(mo, pct_decline, rate_1851, rate_1900, rate_1962):
    mo.md(
        f"""
        **What the trajectory says about the history.** The posterior puts the
        disaster rate near {rate_1851[0]:.1f} per year in 1851, roughly
        {rate_1900[0]:.1f} by 1900, and about {rate_1962[0]:.1f} by 1962 — an
        overall decline of about {pct_decline:.0f}%. Crucially, the model was
        *not told* to look for a change around 1890–1900; it inferred a smooth,
        sustained drop concentrated in the late-19th-century decades directly
        from the counts. Historically this tracks the tightening of British
        mine-safety regulation and inspection over that period. Two modelling
        points worth stressing to a class:

        - The GP reports the decline **with uncertainty everywhere** (the 89% ETIs
          above), rather than a single point estimate of one changepoint year.
          It stays agnostic about *whether* the change was abrupt or gradual and
          lets the data decide — and the data prefer gradual.
        - Because $f$ is the log-rate, the decline reads naturally as
          *multiplicative*: the rate roughly halved and then halved again across
          the record, a structure a linear-in-rate model would describe far less
          gracefully.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The latent rate shows a clear decline over the period with no
    sharp changepoint imposed — the GP infers the shape of the decline
    (which looks like it happens over some decades around the turn of
    the century) directly from the data, unlike a two-regime
    changepoint model.

    ### Posterior predictive check

    Finally, we simulate new counts from the fitted model and check
    that they bracket the observed series.
    """)
    return


@app.cell
def _(RANDOM_SEED, coal_idata, coal_model, pm, posterior_subset, results_dir):
    with coal_model:
        coal_ppc = pm.sample_posterior_predictive(
            posterior_subset(coal_idata, draws_per_chain=100),
            var_names=["y"],
            random_seed=RANDOM_SEED,
        )
    coal_idata["posterior_predictive"] = coal_ppc["posterior_predictive"]
    coal_idata.to_netcdf(results_dir / "02_coal_latent_gp.nc")
    return (coal_ppc,)


@app.cell
def _(
    PYMC_GREEN,
    coal_ppc,
    disaster_counts,
    eti_bounds,
    go,
    np,
    pl,
    year_vals,
):
    ppc_counts = coal_ppc["posterior_predictive"]["y"]
    ppc_count_low, ppc_count_high = eti_bounds(ppc_counts)
    count_mean = ppc_counts.mean(dim=("chain", "draw"))
    ppc_fig = go.Figure()
    ppc_fig.add_trace(
        go.Scatter(
            x=np.concatenate([year_vals, year_vals[::-1]]),
            y=np.concatenate([ppc_count_high.values, ppc_count_low.values[::-1]]),
            fill="toself",
            fillcolor="rgba(129,194,64,0.25)",
            line=dict(color="rgba(255,255,255,0)"),
            name="89% posterior-predictive ETI",
        )
    )
    ppc_fig.add_trace(
        go.Scatter(
            x=year_vals,
            y=count_mean.values,
            mode="lines",
            name="posterior-predictive mean",
            line=dict(color=PYMC_GREEN, width=2),
        )
    )
    ppc_fig.add_trace(
        go.Scatter(
            x=year_vals,
            y=disaster_counts,
            mode="markers",
            name="observed",
            marker=dict(color="black", size=6),
        )
    )
    ppc_fig.update_layout(
        title="Posterior-predictive annual counts",
        xaxis_title="Year",
        yaxis_title="Disasters",
        template="plotly_white",
    )

    mean = ppc_counts.mean(dim="year")
    discrepancy_draws = {
        "zero fraction": (ppc_counts == 0).mean(dim="year"),
        "mean": mean,
        "variance / mean": ppc_counts.var(dim="year") / mean,
        "maximum": ppc_counts.max(dim="year"),
    }
    observed_discrepancies = {
        "zero fraction": float((disaster_counts == 0).mean()),
        "mean": float(disaster_counts.mean()),
        "variance / mean": float(disaster_counts.var() / disaster_counts.mean()),
        "maximum": float(disaster_counts.max()),
    }
    ppc_discrepancies = pl.DataFrame(
        {
            "statistic": list(discrepancy_draws),
            "observed": list(observed_discrepancies.values()),
            "predictive_percentile": [
                float((draws <= observed_discrepancies[name]).mean())
                for name, draws in discrepancy_draws.items()
            ],
        }
    )
    ppc_fig
    ppc_discrepancies
    return (ppc_discrepancies,)


@app.cell
def _(az, coal_idata):
    az.plot_ppc_dist(coal_idata, var_names=["y"], kind="ecdf", num_samples=100)
    return


@app.cell(hide_code=True)
def _(mo, ppc_discrepancies):
    mo.md(
        "The ECDF and table compare observed zero fraction, mean, variance-to-mean "
        "ratio, and maximum with posterior-predictive distributions. They describe "
        "these checked discrepancies; they do not establish general adequacy."
    )
    ppc_discrepancies
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Observed counts fall consistently within the posterior predictive
    interval, and the mean tracks the visible decline in disaster
    frequency — the model captures the broad structure of the series.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise: inspect the lengthscale prior implication

    The fitted coal model uses `ell ~ LogNormal(0, 0.5)` on standardized year.
    Before fitting anything, replace it in a **prior-predictive** scratch model
    with `LogNormal(-1, 0.5)`. Predict how the labeled `rate(year)` trajectories
    and implied counts change, and why a more flexible prior does not by itself
    justify a more flexible posterior.
    """)
    return


@app.cell(hide_code=True)
def _(PYMC_GREEN, RANDOM_SEED, disaster_counts, go, mo, np, pm, t, year_vals):
    _alt_coords = {"year": year_vals, "feature": ["standardized_year"]}
    with pm.Model(coords=_alt_coords):
        _alt_year_input = pm.Data("year_input", t, dims=("year", "feature"))
        _alt_disaster_count = pm.Data(
            "disaster_count", disaster_counts, dims="year"
        )
        _alpha_alt = pm.Normal("alpha", mu=np.log(1.5), sigma=0.5)
        _ell_alt = pm.LogNormal("ell", mu=-1, sigma=0.5)
        _eta_alt = pm.HalfNormal("eta", sigma=1)
        _cov_alt = _eta_alt**2 * pm.gp.cov.Matern52(1, ls=_ell_alt)
        _gp_alt = pm.gp.Latent(cov_func=_cov_alt)
        _f_alt = _gp_alt.prior("f", X=_alt_year_input, dims="year")
        _rate_alt = pm.Deterministic(
            "rate", pm.math.exp(_alpha_alt + _f_alt), dims="year"
        )
        pm.Poisson("y", mu=_rate_alt, observed=_alt_disaster_count, dims="year")
        alt_prior = pm.sample_prior_predictive(draws=200, random_seed=RANDOM_SEED)
    alt_rate_draws = (
        alt_prior["prior"]["rate"]
        .stack(sample=("chain", "draw"))
        .transpose("sample", "year")
    )

    alt_fig = go.Figure()
    rng_alt = np.random.default_rng(RANDOM_SEED)
    for alt_draw_index in rng_alt.choice(
        alt_rate_draws.sizes["sample"], size=30, replace=False
    ):
        alt_fig.add_trace(
            go.Scatter(
                x=year_vals,
                y=alt_rate_draws.isel(sample=alt_draw_index).values,
                mode="lines",
                line=dict(color=PYMC_GREEN, width=1),
                opacity=0.3,
                showlegend=False,
            )
        )
    alt_fig.add_trace(
        go.Scatter(
            x=year_vals,
            y=disaster_counts,
            mode="markers",
            marker=dict(color="black", size=6),
            name="observed",
        )
    )
    alt_fig.update_layout(
        title="Prior draws of rate under a shorter-lengthscale prior",
        xaxis_title="Year",
        yaxis_title="rate",
        template="plotly_white",
        showlegend=False,
    )

    mo.accordion(
        {
            "Discussion": mo.vstack(
                [
                    mo.md(
                        """
                        ```python
                        ell_alt = pm.LogNormal("ell", mu=-1, sigma=0.5)
                        ```

                        Shifting the LogNormal location downward makes shorter
                        lengthscales more plausible, so prior rate trajectories
                        can move more quickly from year to year. With only 112
                        annual counts, that additional flexibility can chase
                        noise rather than reveal sustained rate change. Inspect
                        these labeled prior trajectories before refitting; then
                        evaluate posterior-predictive discrepancies rather than
                        treating a wavier posterior curve as an improvement.
                        """
                    ),
                    alt_fig,
                ]
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md(
                r"""
                **Exercise — does the Poisson likelihood capture spread?**
                Use the existing posterior-predictive distributions of zero
                fraction, mean, variance-to-mean ratio, and maximum. Where
                would the observed variance-to-mean ratio need to sit before a
                Negative Binomial observation model becomes a motivated
                alternative?
                """
            ),
            mo.accordion(
                {
                    "Discussion": mo.md(
                        r"""
                        A Poisson likelihood has
                        $\operatorname{Var}[y]=\mathbb E[y]$. Compare the
                        observed variance-to-mean ratio to its replicated
                        distribution, alongside the other available
                        discrepancies. An observed value in the upper tail
                        would indicate over-dispersion relative to the fitted
                        model and motivate a Negative Binomial likelihood with
                        an estimable dispersion parameter. Coverage of one
                        band—or a pleasing rate trajectory—does not establish
                        the count model's adequacy.
                        """
                    )
                }
            ),
            mo.md(
                r"""
                **Exercise — why retain a latent count GP?** Consider replacing
                the Poisson latent GP with a transformed Gaussian marginal GP.
                For the observed low counts, zeros, and rate interpretation,
                identify the approximation that would be introduced and the
                posterior-predictive quantity you would use to expose it.
                """
            ),
            mo.accordion(
                {
                    "Discussion": mo.md(
                        r"""
                        A variance-stabilizing transformation can be useful for
                        uniformly large counts, but these small counts include
                        many zeros where the Gaussian approximation is weakest.
                        It changes the observation model and makes honest
                        count-scale predictions indirect. The latent Poisson GP
                        preserves the discrete likelihood and rate-scale
                        interpretation. If evaluating a transformed alternative,
                        compare its replicated zero fraction, spread, and
                        maximum on the original count scale rather than judging
                        only a transformed-space fit.
                        """
                    )
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Where we are, and what's next

    You have now fit both faces of GP inference on real data. In the
    **marginal** (conjugate) case, a Gaussian likelihood let PyMC integrate
    the latent function out analytically: `pm.gp.Marginal` sampled only a
    handful of hyperparameters, predictions came in closed form, and MAP and
    full MCMC nearly agreed. In the **latent** (non-conjugate) case, a
    Poisson likelihood broke conjugacy, so `pm.gp.Latent` carried all 112
    latent function values through the exp link and NUTS sampled them jointly
    with the kernel hyperparameters — slower, but exact and fully
    uncertainty-aware.

    The dividing question is always the likelihood: **Gaussian noise ⇒ reach
    for `Marginal`; anything else (counts, binary, heavy tails) ⇒ reach for
    `Latent`.** Both share the same covariance-function vocabulary, which is
    exactly what **Notebook 3** expands next — a tour of kernels (the Matérn
    family, periodic, rational-quadratic, linear), how to *combine* them
    additively and multiplicatively, GPs on multi-dimensional inputs, and
    hierarchical GPs that share structure across groups.
    """)
    return


if __name__ == "__main__":
    app.run()
