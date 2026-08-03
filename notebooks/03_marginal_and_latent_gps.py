import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo

    import sys
    from pathlib import Path

    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from inference_contract import (
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
    import pytensor.tensor as pt
    from scipy.spatial import Delaunay
    import preliz as pz

    PYMC_BLUE = "#154A72"
    PYMC_GREEN = "#81C240"
    PYMC_LIGHT_BLUE = "#4A9EDE"
    PYMC_DARK_GREEN = "#40611F"

    RANDOM_SEED = 42
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    data_dir = project_root / "data"

    def z(a):
        """Standardize an array: (a - mean) / population std."""
        return (a - a.mean()) / a.std(ddof=0)


@app.cell(hide_code=True)
def _():
    import inspect
    from exercises import exercise

    return exercise, inspect


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Marginal and Latent Gaussian Processes

    This notebook applies Gaussian-process models in PyMC. With a Gaussian
    likelihood, `pm.gp.Marginal` integrates the latent function out exactly.
    With a non-Gaussian likelihood, `pm.gp.Latent` represents and samples it
    directly.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Meet the marginal-GP API, on data with known truth

    Notebook 2 built a GP posterior by hand: write the joint Gaussian over
    training and test points, condition on the training rows, done.
    PyMC's `pm.gp` module wraps exactly that machinery, you supply a mean
    function, a covariance function, and how the observations relate to
    the latent function, and it assembles the linear algebra for you.

    Before trusting the API on real data, we check it against
    **simulated** data: a single draw from a GP with known lengthscale
    $\ell$, amplitude $\eta$, and noise scale $\sigma$. That gives the
    posterior a very direct question to answer, does inference recover
    the values used to generate the data?
    """)
    return


@app.cell
def _():
    def simulate_gp_data(seed, n=100, ell_true=1.0, eta_true=3.0, sigma_true=2.0):
        """Draw one exact GP realization plus Gaussian noise, with known truth."""
        rng = np.random.default_rng(seed)
        sim_x = np.linspace(0, 10, n)
        sim_X = sim_x[:, None]
        mean_func = pm.gp.mean.Zero()
        cov_func = eta_true**2 * pm.gp.cov.Matern52(1, ell_true)
        f_true = rng.multivariate_normal(
            mean_func(sim_X).eval(),
            cov_func(sim_X).eval() + 1e-8 * np.eye(n),
            1,
        ).flatten()
        sim_y = f_true + sigma_true * rng.standard_normal(n)
        return sim_x, sim_X, f_true, sim_y

    sim_ell_true, sim_eta_true, sim_sigma_true = 1.0, 3.0, 2.0
    sim_x, sim_X, sim_f_true, sim_y = simulate_gp_data(
        RANDOM_SEED,
        ell_true=sim_ell_true,
        eta_true=sim_eta_true,
        sigma_true=sim_sigma_true,
    )
    return (
        sim_X,
        sim_ell_true,
        sim_eta_true,
        sim_f_true,
        sim_sigma_true,
        sim_x,
        sim_y,
    )


@app.cell(hide_code=True)
def _(sim_f_true, sim_x, sim_y):
    sim_data_fig = go.Figure()
    sim_data_fig.add_trace(
        go.Scatter(
            x=sim_x,
            y=sim_f_true,
            mode="lines",
            name="true f",
            line=dict(color=PYMC_BLUE, width=3),
        )
    )
    sim_data_fig.add_trace(
        go.Scatter(
            x=sim_x,
            y=sim_y,
            mode="markers",
            name="observed data",
            marker=dict(color="black", size=6, opacity=0.6),
        )
    )
    sim_data_fig.update_layout(
        title="Simulated data: known f, known noise",
        xaxis_title="x",
        yaxis_title="y",
        template="plotly_white",
    )
    sim_data_fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### The `.marginal_likelihood` method

    `pm.gp.Marginal` implements the conjugate case: observations are the
    GP plus Gaussian noise,

    $$
    \begin{aligned}
    f(x) &\sim \mathcal{GP}\big(0,\ k(x, x')\big) \\
    y &= f(x) + \epsilon, \qquad \epsilon \sim \mathcal N(0, \sigma^2 I)
    \end{aligned}
    $$

    Because both pieces are Gaussian, $f$ can be integrated out of the
    joint analytically, the **marginal likelihood**:

    $$p(y \mid X) = \int p(y \mid f, X)\, p(f \mid X)\, df$$

    $$\log p(y \mid X) = -\tfrac12\, y^\top (K + \sigma^2 I)^{-1} y - \tfrac12 \log\lvert K + \sigma^2 I \rvert - \tfrac{n}{2}\log 2\pi$$

    The class exposes three methods that mirror this structure directly:
    `.marginal_likelihood(name, X, y, sigma)` builds the expression above
    as the model's likelihood, `.conditional(name, Xnew)` builds the
    posterior over the latent function at new inputs, and `.predict(Xnew,
    point)` returns a closed-form mean and variance at a single
    hyperparameter point instead of a full random variable. We use the
    first two below; `.predict` reappears with the fastball-spin data.
    """)
    return


@app.cell
def _(sim_X, sim_y):
    def build_sim_marginal_model(X, y, *, X_pred=None, pred_coord=None):
        coords = {"observation": np.arange(len(y)), "feature": ["x"]}
        if X_pred is not None:
            coords["prediction"] = (
                np.asarray(pred_coord)
                if pred_coord is not None
                else np.arange(len(X_pred))
            )
        with pm.Model(coords=coords) as sim_model:
            x_data = pm.Data("X", X, dims=("observation", "feature"))
            y_data = pm.Data("y_obs", y, dims="observation")
            ell = pm.Gamma("ell", alpha=2, beta=1)
            eta = pm.HalfCauchy("eta", beta=5)
            sigma = pm.HalfCauchy("sigma", beta=5)
            cov_func = eta**2 * pm.gp.cov.Matern52(1, ls=ell)
            sim_gp = pm.gp.Marginal(mean_func=pm.gp.mean.Zero(), cov_func=cov_func)
            sim_gp.marginal_likelihood(
                "y", X=x_data, y=y_data, sigma=sigma, dims="observation"
            )
            if X_pred is not None:
                x_pred = pm.Data("X_pred", X_pred, dims=("prediction", "feature"))
                sim_gp.conditional("f_pred", x_pred, dims="prediction")
        return sim_model, sim_gp

    sim_model, sim_gp = build_sim_marginal_model(sim_X, sim_y)
    return build_sim_marginal_model, sim_model


@app.cell
def _(sim_model):
    sim_model.compile_logp()(sim_model.initial_point())
    with sim_model:
        sim_sample_start = perf_counter()
        sim_idata = pm.sample(chains=4, draws=500, tune=500, random_seed=RANDOM_SEED)
        sim_sample_seconds = perf_counter() - sim_sample_start
    sim_idata.to_netcdf(results_dir / "03_sim_marginal_gp.nc")
    print(f"Simulated marginal-GP sampling wall-time: {sim_sample_seconds:.1f}s")
    return sim_idata, sim_sample_seconds


@app.cell(hide_code=True)
def _(sim_idata, sim_model):
    sim_summary, sim_health_passed = inference_health(sim_idata, sim_model)
    sim_n_div = sim_summary.attrs["divergences"]
    sim_n_draws_total = (
        sim_idata["posterior"].sizes["chain"] * sim_idata["posterior"].sizes["draw"]
    )
    sim_min_ess_bulk = float(sim_summary["ess_bulk"].min())
    sim_min_ess_tail = float(sim_summary["ess_tail"].min())
    sim_max_rhat = float(sim_summary["r_hat"].astype(float).max())
    print(
        f"Divergences: {sim_n_div} / {sim_n_draws_total}; health passed: {sim_health_passed}"
    )
    sim_summary.round(4)
    return (
        sim_health_passed,
        sim_max_rhat,
        sim_min_ess_bulk,
        sim_min_ess_tail,
        sim_n_div,
        sim_n_draws_total,
    )


@app.cell
def _(sim_idata):
    sim_posterior_summary = az.summary(
        sim_idata, var_names=["ell", "eta", "sigma"], kind="stats"
    )
    return (sim_posterior_summary,)


@app.cell(hide_code=True)
def _(
    sim_ell_true,
    sim_eta_true,
    sim_idata,
    sim_posterior_summary,
    sim_sigma_true,
):
    sim_ell_mean = round(float(sim_posterior_summary.loc["ell", "mean"]), 3)
    sim_eta_mean = round(float(sim_posterior_summary.loc["eta", "mean"]), 3)
    sim_sigma_mean = round(float(sim_posterior_summary.loc["sigma", "mean"]), 3)
    sim_recovery = pl.DataFrame(
        {
            "parameter": ["ell", "eta", "sigma"],
            "true_value": [sim_ell_true, sim_eta_true, sim_sigma_true],
            "posterior_mean": [sim_ell_mean, sim_eta_mean, sim_sigma_mean],
        }
    )
    sim_ell_lo, sim_ell_hi = eti_bounds(sim_idata["posterior"]["ell"])
    sim_eta_lo, sim_eta_hi = eti_bounds(sim_idata["posterior"]["eta"])
    sim_sigma_lo, sim_sigma_hi = eti_bounds(sim_idata["posterior"]["sigma"])
    sim_recovered = {
        "ell": bool(float(sim_ell_lo) <= sim_ell_true <= float(sim_ell_hi)),
        "eta": bool(float(sim_eta_lo) <= sim_eta_true <= float(sim_eta_hi)),
        "sigma": bool(float(sim_sigma_lo) <= sim_sigma_true <= float(sim_sigma_hi)),
    }
    print(f"True value within 89% ETI: {sim_recovered}")
    sim_recovery
    return sim_ell_mean, sim_eta_mean, sim_recovered, sim_sigma_mean


@app.cell(hide_code=True)
def _(
    sim_ell_mean,
    sim_ell_true,
    sim_eta_mean,
    sim_eta_true,
    sim_health_passed,
    sim_max_rhat,
    sim_min_ess_bulk,
    sim_min_ess_tail,
    sim_n_div,
    sim_n_draws_total,
    sim_recovered,
    sim_sample_seconds,
    sim_sigma_mean,
    sim_sigma_true,
):
    mo.md(f"""
    **Diagnostics:** {sim_n_div} divergence(s) out of {sim_n_draws_total} draws
    in {sim_sample_seconds:.1f}s. Minimum `ess_bulk` is {sim_min_ess_bulk:.0f},
    minimum `ess_tail` is {sim_min_ess_tail:.0f}, and maximum `r_hat` is
    {sim_max_rhat:.3f}; health status **{sim_health_passed}**.

    **Recovery check:** posterior means ({sim_ell_mean}, {sim_eta_mean},
    {sim_sigma_mean}) for (ell, eta, sigma) sit close to the simulating
    values ({sim_ell_true}, {sim_eta_true}, {sim_sigma_true}), and the 89% ETI
    for every parameter contains its true value: {sim_recovered}. **Inference
    recovers the generating hyperparameters.** This is the check to repeat
    mentally on real data, where the truth is unknown and only the diagnostics
    and predictive checks are available.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### The `.conditional` distribution

    After fitting, `.conditional(name, Xnew)` builds the posterior over
    the latent function at new inputs $x_*$:

    $$
    \begin{aligned}
    m^*(x_*) &= k(x_*, x)^\top [k(x,x) + \sigma^2 I]^{-1}\, y \\
    k^*(x_*) &= k(x_*,x_*) - k(x_*,x)^\top [k(x,x)+\sigma^2 I]^{-1}\, k(x_*,x)
    \end{aligned}
    $$

    Its `pred_noise` flag defaults to `False`, so the conditional
    represents the underlying function $f$ rather than a new noisy
    measurement. We predict on a grid running past the observed range ,
    $x \in [0, 12]$ against data on $[0, 10]$, to see the band widen
    under extrapolation, right where the true function is still known and
    checkable.
    """)
    return


@app.cell(hide_code=True)
def _(build_sim_marginal_model, sim_X, sim_idata, sim_y):
    sim_x_grid = np.linspace(0, 12, 400)
    sim_X_new = sim_x_grid[:, None]
    sim_predictions = sample_fresh_model_predictions(
        sim_idata,
        lambda: build_sim_marginal_model(
            sim_X, sim_y, X_pred=sim_X_new, pred_coord=sim_x_grid
        )[0],
        var_names=["f_pred"],
        random_seed=RANDOM_SEED,
    )
    return sim_predictions, sim_x_grid


@app.cell(hide_code=True)
def _(sim_predictions):
    sim_pred_draws = sim_predictions["predictions"]["f_pred"]
    sim_pred_mean = sim_pred_draws.mean(dim=("chain", "draw"))
    sim_pred_low, sim_pred_high = eti_bounds(sim_pred_draws)
    return sim_pred_high, sim_pred_low, sim_pred_mean


@app.cell(hide_code=True)
def _(
    sim_f_true,
    sim_pred_high,
    sim_pred_low,
    sim_pred_mean,
    sim_x,
    sim_x_grid,
    sim_y,
):
    sim_pred_fig = go.Figure()
    sim_pred_fig.add_trace(
        go.Scatter(
            x=np.concatenate([sim_x_grid, sim_x_grid[::-1]]),
            y=np.concatenate([sim_pred_high.values, sim_pred_low.values[::-1]]),
            fill="toself",
            fillcolor="rgba(21,74,114,0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="89% ETI",
        )
    )
    sim_pred_fig.add_trace(
        go.Scatter(
            x=sim_x_grid,
            y=sim_pred_mean.values,
            mode="lines",
            name="posterior mean",
            line=dict(color=PYMC_BLUE, width=3),
        )
    )
    sim_pred_fig.add_trace(
        go.Scatter(
            x=sim_x,
            y=sim_f_true,
            mode="lines",
            name="true f",
            line=dict(color=PYMC_DARK_GREEN, width=2, dash="dash"),
        )
    )
    sim_pred_fig.add_trace(
        go.Scatter(
            x=sim_x,
            y=sim_y,
            mode="markers",
            name="observed data",
            marker=dict(color="black", size=5, opacity=0.6),
        )
    )
    sim_pred_fig.add_vline(x=10, line=dict(dash="dot", color="gray"))
    sim_pred_fig.update_layout(
        title="Marginal-GP posterior conditional: mean, 89% ETI, and the true function",
        xaxis_title="x",
        yaxis_title="y",
        template="plotly_white",
    )
    sim_pred_fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Marginal GP on the Theophylline curve

    The simulated example confirmed the API recovers known hyperparameters;
    now we turn to real data, where the truth is unknown and a GP has to
    earn its fit on its own. Recall the Theophylline dataset from
    Notebook 1: 12 subjects, each with 11 serum-concentration measurements
    over 24 hours after a single oral dose. We again work with a single
    subject's curve, a smooth rise-then-decay shape with no natural
    parametric form.
    """)
    return


@app.cell(hide_code=True)
def _():
    theoph = pl.read_csv(data_dir / "theophylline.csv")
    return (theoph,)


@app.cell(hide_code=True)
def _(theoph):
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


@app.cell(hide_code=True)
def _(conc_vals, subject_id, time_vals):
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
def _():
    mo.md(r"""
    ### A naive first attempt

    Let's fit the most straightforward possible GP: a zero-mean
    `pm.gp.Marginal` with a `Matern52` covariance, evaluated directly on
    the **standardized raw time** axis. `pm.gp.Marginal` marginalizes
    the latent function analytically, so `.marginal_likelihood` gives us
    an exact Gaussian likelihood in the hyperparameters ($\ell$, $\eta$,
    $\sigma$), no MCMC over $f$ itself is needed. We optimize with
    `pm.find_MAP` for a quick first look before committing to full
    sampling.
    """)
    return


@app.cell
def _(X, y):
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
def _(naive_model):
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
def _(naive_model):
    naive_model.compile_logp()(naive_model.initial_point())
    with naive_model:
        map_naive = pm.find_MAP(progressbar=False)
    print("Naive MAP optimization complete")
    return (map_naive,)


@app.cell(hide_code=True)
def _(
    conc_mean,
    conc_std,
    gp_naive,
    map_naive,
    naive_model,
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
    return naive_grid, naive_mu_orig, naive_sd_orig


@app.cell(hide_code=True)
def _(conc_vals, naive_grid, naive_mu_orig, naive_sd_orig, time_vals):
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
def _():
    mo.md(r"""
    ### Why the naive fit struggles

    The observed time points are very unevenly spaced: dense near
    $t=0$ (0, 0.25, 0.57, 1.12, 2.02 hours, where concentration is
    rising fastest) and sparse in the tail (9.05, 12.12, 24.37 hours ,
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


@app.cell(hide_code=True)
def _(time_vals):
    log_time = np.log1p(time_vals)
    log_time_mean, log_time_std = log_time.mean(), log_time.std(ddof=0)
    X_log = z(log_time).reshape(-1, 1)
    return X_log, log_time_mean, log_time_std


@app.function
def build_marginal_gp_model(X, y, *, X_pred=None, pred_coord=None):
    coords = {"observation": np.arange(len(y)), "feature": ["log_time"]}
    if X_pred is not None:
        coords["prediction"] = (
            np.asarray(pred_coord) if pred_coord is not None else np.arange(len(X_pred))
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
            gp.conditional("f_pred_noise", x_pred, pred_noise=True, dims="prediction")
    return model, gp


@app.cell
def _(X_log, y):
    gp_model, structured_gp = build_marginal_gp_model(X_log, y)
    return (gp_model,)


@app.cell(hide_code=True)
def _():
    mo.vstack(
        [
            mo.md(
                r"""
                        **Question, what does the linear mean function buy us?**
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
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Prior predictive check

    As always, we look at what the model implies *before* fitting.
    """)
    return


@app.cell
def _(gp_model):
    with gp_model:
        prior_pred = pm.sample_prior_predictive(draws=500, random_seed=RANDOM_SEED)
    return (prior_pred,)


@app.cell(hide_code=True)
def _(prior_pred):
    prior_draws = (
        prior_pred["prior_predictive"]["y"]
        .stack(sample=("chain", "draw"))
        .transpose("sample", "observation")
    )
    return (prior_draws,)


@app.cell(hide_code=True)
def _(X_log, prior_draws, y):
    prior_fig = go.Figure()
    rng_plot = np.random.default_rng(RANDOM_SEED)
    for prior_draw_index in rng_plot.choice(
        prior_draws.sizes["sample"], size=50, replace=False
    ):
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
    return


@app.cell(hide_code=True)
def _(prior_draws, y):
    mo.md(f"""
    **Prior implications:** the 500 draws span
    [{float(prior_draws.min()):.2f}, {float(prior_draws.max()):.2f}] on the
    standardized scale, compared with observed values from
    [{y.min():.2f}, {y.max():.2f}]. This is a prior-range check, not evidence
    that the model fits the data.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.vstack(
        [
            mo.md(
                r"""
                        **Question, assess the Matérn 5/2 prior implication.** Before
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
def _():
    mo.md(r"""
    ### Conjugacy: why the latent function integrates out

    `pm.gp.Marginal` earns its name from a piece of algebra worth seeing
    once. Write the model at the observed inputs $X$ as a latent Gaussian
    vector $\mathbf f \sim \mathcal N(\mathbf m,\ K)$ with $K = k(X, X)$,
    and a Gaussian observation layer
    $\mathbf y \mid \mathbf f \sim \mathcal N(\mathbf f,\ \sigma^2 I)$.
    Because *both* pieces are Gaussian, the joint over
    $(\mathbf f, \mathbf y)$ is Gaussian, and a Gaussian integrated over one
    of its blocks is Gaussian again, the **marginalization** property from
    Notebook 2. Integrating $\mathbf f$ out gives the **marginal
    likelihood** in closed form:

    $$\mathbf y \sim \mathcal N\big(\mathbf m,\ K + \sigma^2 I\big).$$

    No latent $\mathbf f$ appears, the function values have been absorbed
    into the $n \times n$ covariance $K + \sigma^2 I$. This is exactly what
    `.marginal_likelihood("y", X=X, y=y, sigma=sigma)` evaluates. The only
    free unknowns left are the handful of hyperparameters
    $(\ell, \eta, \sigma)$ plus the two mean-function coefficients, **five
    scalars**, instead of $\mathbf f$'s 11 values *and* the hyperparameters.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The same conjugacy delivers predictions in closed form. For test inputs
    $X_*$, the posterior over the latent function is Gaussian with

    $$\mathbb E[\mathbf f_* \mid \mathbf y]
    = \mathbf m_* + K_*(K + \sigma^2 I)^{-1}(\mathbf y - \mathbf m),$$

    $$\operatorname{Cov}[\mathbf f_* \mid \mathbf y]
    = K_{**} - K_*(K + \sigma^2 I)^{-1}K_*^\top,$$

    with $K_* = k(X_*, X)$ and $K_{**} = k(X_*, X_*)$, the identical
    conditioning formula you applied by hand in Notebook 2, now with a
    learned mean function $\mathbf m$ subtracted off first. Two consequences
    we lean on below:

    - **Every hyperparameter draw yields an *exact* predictive Gaussian.**
      There is no Monte-Carlo error in $\mathbf f$ given
      $(\ell, \eta, \sigma)$; the only thing we sample over is the
      hyperparameters themselves.
    - **The $(K + \sigma^2 I)^{-1}$ solve is $O(n^3)$.** With $n = 11$ that
      is nothing, but it is the exact cost that motivates the sparse and
      HSGP approximations in notebook 4. Conjugacy buys exactness at a cubic
      price.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### MAP vs. full posterior

    `pm.gp.Marginal` only ever samples the **hyperparameters**
    ($\ell$, $\eta$, $\sigma$, and the two mean-function coefficients) ,
    five scalars, since $f$ itself is marginalized out analytically.
    That makes both a MAP optimization and full MCMC cheap; let's do
    both and compare.
    """)
    return


@app.cell
def _(gp_model):
    gp_model.compile_logp()(gp_model.initial_point())
    with gp_model:
        map_estimate = pm.find_MAP(progressbar=False)
    return (map_estimate,)


@app.cell
def _(gp_model):
    gp_model.compile_logp()(gp_model.initial_point())
    with gp_model:
        idata = pm.sample(
            chains=4,
            draws=500,
            tune=500,
            # 0.9 clears the 23 divergences the default produced at the noise boundary.
            target_accept=0.9,
            init="adapt_diag",
        )
    idata.to_netcdf(results_dir / "03_theoph_marginal_gp.nc")
    return (idata,)


@app.cell
def _(gp_model, idata):
    with gp_model:
        observed_ppc = pm.sample_posterior_predictive(
            posterior_subset(idata, draws_per_chain=100),
            var_names=["y"],
            random_seed=RANDOM_SEED,
        )
    idata_with_ppc = idata.copy()
    idata_with_ppc["posterior_predictive"] = observed_ppc["posterior_predictive"]
    idata_with_ppc.to_netcdf(results_dir / "03_theoph_marginal_gp.nc")
    return (idata_with_ppc,)


@app.cell(hide_code=True)
def _(gp_model, idata):
    summary, health_passed = inference_health(idata, gp_model)
    n_div = summary.attrs["divergences"]
    n_draws_total = idata["posterior"].sizes["chain"] * idata["posterior"].sizes["draw"]
    min_ess_bulk = float(summary["ess_bulk"].min())
    min_ess_tail = float(summary["ess_tail"].min())
    max_rhat = float(summary["r_hat"].astype(float).max())
    print(f"Divergences: {n_div} / {n_draws_total}; health passed: {health_passed}")
    summary.round(4)
    return (
        health_passed,
        max_rhat,
        min_ess_bulk,
        min_ess_tail,
        n_div,
        n_draws_total,
    )


@app.cell
def _(gp_model, idata):
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
    n_div,
    n_draws_total,
    posterior_summary,
):
    mo.md(f"""
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
    """)
    return


@app.cell
def _():
    def _(gp_model, idata):
        structured_free_rv_names = [rv.name for rv in gp_model.free_RVs]
        trace_plot = az.plot_trace_dist(
            idata,
            var_names=structured_free_rv_names,
            compact=True,
            figure_kwargs={"figsize": (5, 3.5)},
        )
        rank_plot = az.plot_rank(
            idata,
            var_names=structured_free_rv_names,
            figure_kwargs={"figsize": (5, 3.5)},
        )
        mo.hstack(
            [
                mo.mpl.interactive(trace_plot.viz["figure"].item()),
                mo.mpl.interactive(rank_plot.viz["figure"].item()),
            ],
            gap=1,
            justify="center",
        )
        return

    return


@app.cell(hide_code=True)
def _(map_estimate, posterior_summary):
    beta_row = next(
        index for index in posterior_summary.index if index.startswith("beta")
    )
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
def _():
    mo.md(r"""
    The table lines up the two point summaries for all five
    hyperparameters. They agree to within a posterior standard deviation
    here, which is the *good* case: a low-dimensional, well-identified
    hyperparameter posterior that is roughly symmetric, so its mode (MAP)
    and its mean nearly coincide. When should you trust MAP, and when not?

    - **MAP is appropriate** as a fast first look, for initialization, or
      when the hyperparameter posterior is tight and unimodal and you
      genuinely only need a point estimate. It is a single optimization ,
      cheap and deterministic.
    - **MAP misses** everything about the *shape* of the posterior. It
      reports no uncertainty, so it cannot tell you the lengthscale is only
      known to within some range; it sits at the mode, which for a skewed
      posterior (lengthscales often are) can be far from the mean; it can
      settle on a local optimum; and, the subtle one, a MAP *prediction*
      plugs in a single hyperparameter value, so its predictive band
      understates uncertainty because it ignores the spread over
      $(\ell, \eta, \sigma)$ entirely.

    Full MCMC propagates hyperparameter uncertainty into every prediction:
    we draw many $(\ell, \eta, \sigma)$, form the exact predictive Gaussian
    for each (conjugacy again), and mix them. That mixture is wider, and
    more honest, than any single MAP curve, which is why the fit below uses
    the full posterior rather than `map_estimate`.
    """)
    return


@app.cell(hide_code=True)
def _(log_time_mean, log_time_std, time_vals):
    time_grid = np.linspace(time_vals.min(), time_vals.max(), 200)
    Xnew = ((np.log1p(time_grid) - log_time_mean) / log_time_std).reshape(-1, 1)
    return Xnew, time_grid


@app.cell(hide_code=True)
def _(X_log, Xnew, idata, time_grid, y):
    structured_predictions = sample_fresh_model_predictions(
        idata,
        lambda: build_marginal_gp_model(X_log, y, X_pred=Xnew, pred_coord=time_grid)[0],
        var_names=["f_pred", "f_pred_noise"],
        random_seed=RANDOM_SEED,
    )
    return (structured_predictions,)


@app.cell(hide_code=True)
def _(conc_mean, conc_std, structured_predictions):
    latent_draws = structured_predictions["predictions"]["f_pred"].rename(
        {"prediction": "time_grid"}
    )
    structured_noisy_draws = structured_predictions["predictions"][
        "f_pred_noise"
    ].rename({"prediction": "time_grid"})
    latent_draws = latent_draws * conc_std + conc_mean
    structured_noisy_draws = structured_noisy_draws * conc_std + conc_mean
    latent_mean = latent_draws.mean(dim=("chain", "draw"))
    latent_low, latent_high = eti_bounds(latent_draws)
    structured_noisy_low, structured_noisy_high = eti_bounds(structured_noisy_draws)
    return (
        latent_high,
        latent_low,
        latent_mean,
        structured_noisy_high,
        structured_noisy_low,
    )


@app.cell(hide_code=True)
def _(
    conc_vals,
    latent_high,
    latent_low,
    latent_mean,
    structured_noisy_high,
    structured_noisy_low,
    time_grid,
    time_vals,
):
    pred_fig = go.Figure()
    pred_fig.add_trace(
        go.Scatter(
            x=np.concatenate([time_grid, time_grid[::-1]]),
            y=np.concatenate(
                [structured_noisy_high.values, structured_noisy_low.values[::-1]]
            ),
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
def _(idata_with_ppc):
    az.plot_ppc_dist(idata_with_ppc, var_names=["y"], kind="ecdf", num_samples=50)
    return


@app.cell(hide_code=True)
def _():
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
def _():
    mo.md(r"""
    ### What `pred_noise` actually changes

    The two bands differ by exactly one term, and it is worth being precise
    about which. Writing the predictive variance at a test point $x_*$:

    - `pred_noise=False` returns $\operatorname{Var}[f(x_*) \mid \mathbf y]$
     , the uncertainty in the *latent function* $f$ at $x_*$. Use this when
      you care about the underlying smooth curve ("what is the true
      concentration trajectory?").
    - `pred_noise=True` returns
      $\operatorname{Var}[f(x_*) \mid \mathbf y] + \sigma^2$, the latent
      uncertainty *plus* the observation-noise variance. Use this when you
      care about a *new measurement* ("what will the assay read if I draw
      blood at 5 hours?"), because a real measurement carries the same
      $\sigma$ scatter the training points did.

    The gap between the bands is therefore a **constant $\sigma^2$ in
    variance**, uniform across the whole grid, since observation noise does
    not depend on *where* you predict. That is why the outer band sits a
    fixed vertical distance outside the inner one even where the data are
    dense and the latent band is tight. The cell below reads off that
    constant gap in the original mg/L units.
    """)
    return


@app.cell(hide_code=True)
def _(conc_std, posterior_summary):
    sigma_post_mean = float(posterior_summary.loc["sigma", "mean"])
    noise_var_mgL = (sigma_post_mean * conc_std) ** 2
    print(f"Posterior-mean sigma (standardized): {sigma_post_mean:.3f}")
    print(f"Constant variance gap between the bands: {noise_var_mgL:.3f} (mg/L)^2")
    print(f"  i.e. a noise standard deviation of {sigma_post_mean * conc_std:.3f} mg/L")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Exercise: extrapolate with full posterior uncertainty

    Use the fresh prediction model and posterior draws to predict to 40 hours
    past dose, well beyond the last observation at 24.37 hours. Contrast the
    full-posterior 89% ETI with a plug-in MAP prediction: which uncertainty
    sources does the latter omit, and toward what specified mean function does
    the full prediction return?
    """)
    return


@app.cell
def _(exercise):
    @exercise
    def exercise_full_posterior_extrapolation():
        # Build a fresh prediction model on a 0--40 hour grid, draw from the
        # full posterior predictive distribution, and plot its 89% ETI.
        # Explain which uncertainty the MAP plug-in omits and its limiting mean.
        ...

    exercise_full_posterior_extrapolation()
    return


@app.cell(hide_code=True)
def _(
    X_log,
    conc_mean,
    conc_std,
    conc_vals,
    idata,
    inspect,
    log_time_mean,
    log_time_std,
    time_vals,
    y,
):
    def solution_full_posterior_extrapolation():
        extrap_grid = np.linspace(0, 40, 200)
        extrap_X = ((np.log1p(extrap_grid) - log_time_mean) / log_time_std).reshape(
            -1, 1
        )
        extrap_predictions = sample_fresh_model_predictions(
            idata,
            lambda: build_marginal_gp_model(
                X_log, y, X_pred=extrap_X, pred_coord=extrap_grid
            )[0],
            var_names=["f_pred_noise"],
            random_seed=RANDOM_SEED,
        )
        extrap_noisy_draws = (
            extrap_predictions["predictions"]["f_pred_noise"].rename(
                {"prediction": "time"}
            )
            * conc_std
            + conc_mean
        )
        extrap_noisy_mean = extrap_noisy_draws.mean(dim=("chain", "draw"))
        extrap_noisy_low, extrap_noisy_high = eti_bounds(extrap_noisy_draws)

        extrap_fig = go.Figure()
        extrap_fig.add_trace(
            go.Scatter(
                x=np.concatenate([extrap_grid, extrap_grid[::-1]]),
                y=np.concatenate(
                    [extrap_noisy_high.values, extrap_noisy_low.values[::-1]]
                ),
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
        return mo.as_html(extrap_fig)

    mo.accordion(
        {
            "Solution": mo.vstack(
                [
                    mo.md(
                        f"```python\n"
                        f"{inspect.getsource(solution_full_posterior_extrapolation)}"
                        f"\n```"
                    ),
                    mo.lazy(
                        solution_full_posterior_extrapolation,
                        show_loading_indicator=True,
                    ),
                    mo.md(
                        "The full-posterior forecast returns toward the specified "
                        "linear mean and averages conditional latent-function and "
                        "observation-noise uncertainty over the hyperparameter "
                        "posterior; a MAP plug-in fixes one hyperparameter point."
                    ),
                ]
            )
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Real-world example: fastball spin rates

    That was contrived data; here is a real irregularly-sampled time
    series. Consider game-averaged fastball spin rate (rpm) for a single
    Major League pitcher across a season. Games are not evenly spaced ,
    off days, rotation skips, and injuries leave gaps from a few days to
    several weeks between starts, exactly the setting where a GP's
    covariance-based notion of "nearby" (in elapsed time, not in
    observation index) matters: two starts eleven days apart should be
    treated as less informative about each other than two starts three
    days apart, and a GP's kernel encodes that directly, with no need to
    resample onto a regular grid.
    """)
    return


@app.cell(hide_code=True)
def _():
    spin_rates = pl.read_csv(data_dir / "fastball_spin_rates.csv")
    spin_rates = spin_rates.with_columns(
        pl.col("game_date").str.strptime(pl.Date, "%Y-%m-%d")
    )
    spin_rates = spin_rates.with_columns(
        pl.col("game_date").dt.ordinal_day().alias("day_of_year")
    )
    kopech_spin = spin_rates.filter(pl.col("pitcher") == "Kopech, Michael").sort(
        "day_of_year"
    )
    kopech_day = kopech_spin["day_of_year"].to_numpy().astype(float)
    kopech_rate = kopech_spin["spin_rate"].to_numpy()
    kopech_day_mean, kopech_day_std = kopech_day.mean(), kopech_day.std(ddof=0)
    kopech_rate_mean, kopech_rate_std = kopech_rate.mean(), kopech_rate.std(ddof=0)
    kopech_day_z = z(kopech_day)
    kopech_rate_z = z(kopech_rate)
    return (
        kopech_day,
        kopech_day_mean,
        kopech_day_std,
        kopech_day_z,
        kopech_rate,
        kopech_rate_mean,
        kopech_rate_std,
        kopech_rate_z,
        spin_rates,
    )


@app.cell(hide_code=True)
def _(kopech_day, kopech_rate):
    kopech_fig = go.Figure()
    kopech_fig.add_trace(
        go.Scatter(
            x=kopech_day,
            y=kopech_rate,
            mode="markers",
            name="Kopech",
            marker=dict(color=PYMC_BLUE, size=8),
        )
    )
    kopech_fig.update_layout(
        title="Kopech fastball spin rate by game",
        xaxis_title="Day of year",
        yaxis_title="Spin rate (rpm)",
        template="plotly_white",
    )
    kopech_fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We can build a similar model to what we used for the simulated data ,
    same `pm.gp.Marginal` and `.marginal_likelihood`, same Matérn 5/2
    covariance, just with a constant mean function set at the pitcher's
    average spin rate, since there is no reason here to expect a linear
    trend across the season.
    """)
    return


@app.cell
def _(kopech_day_z, kopech_rate_z):
    def build_kopech_spin_model(day, rate, *, day_pred=None):
        coords = {"observation": np.arange(len(rate)), "feature": ["day_of_year"]}
        if day_pred is not None:
            coords["prediction"] = np.arange(len(day_pred))
        with pm.Model(coords=coords) as kopech_model:
            day_data = pm.Data(
                "day_of_year", day[:, None], dims=("observation", "feature")
            )
            rate_data = pm.Data("spin_rate", rate, dims="observation")
            ell = pm.LogNormal("ell", mu=0, sigma=1)
            eta = pm.HalfNormal("eta", sigma=1)
            sigma = pm.HalfNormal("sigma", sigma=1)
            cov_fn = eta**2 * pm.gp.cov.Matern52(1, ls=ell)
            kopech_gp = pm.gp.Marginal(mean_func=pm.gp.mean.Zero(), cov_func=cov_fn)
            kopech_gp.marginal_likelihood(
                "spin", X=day_data, y=rate_data, sigma=sigma, dims="observation"
            )
            if day_pred is not None:
                day_pred_data = pm.Data(
                    "day_of_year_pred",
                    day_pred[:, None],
                    dims=("prediction", "feature"),
                )
                kopech_gp.conditional("spin_pred", day_pred_data, dims="prediction")
        return kopech_model, kopech_gp

    kopech_model, kopech_gp = build_kopech_spin_model(kopech_day_z, kopech_rate_z)
    return build_kopech_spin_model, kopech_gp, kopech_model


@app.cell
def _(kopech_model):
    kopech_model.compile_logp()(kopech_model.initial_point())
    with kopech_model:
        kopech_sample_start = perf_counter()
        kopech_idata = pm.sample(
            chains=4, draws=500, tune=500, target_accept=0.9, random_seed=RANDOM_SEED
        )
        kopech_sample_seconds = perf_counter() - kopech_sample_start
    kopech_idata.to_netcdf(results_dir / "03_kopech_spin_marginal_gp.nc")
    print(f"Kopech spin-rate GP sampling wall-time: {kopech_sample_seconds:.1f}s")
    return kopech_idata, kopech_sample_seconds


@app.cell(hide_code=True)
def _(kopech_idata, kopech_model):
    kopech_summary, kopech_health_passed = inference_health(kopech_idata, kopech_model)
    kopech_n_div = kopech_summary.attrs["divergences"]
    kopech_n_draws_total = (
        kopech_idata["posterior"].sizes["chain"]
        * kopech_idata["posterior"].sizes["draw"]
    )
    kopech_min_ess_bulk = float(kopech_summary["ess_bulk"].min())
    kopech_min_ess_tail = float(kopech_summary["ess_tail"].min())
    kopech_max_rhat = float(kopech_summary["r_hat"].astype(float).max())
    print(
        f"Divergences: {kopech_n_div} / {kopech_n_draws_total}; "
        f"health passed: {kopech_health_passed}"
    )
    kopech_summary.round(4)
    return (
        kopech_health_passed,
        kopech_max_rhat,
        kopech_min_ess_bulk,
        kopech_min_ess_tail,
        kopech_n_div,
        kopech_n_draws_total,
    )


@app.cell(hide_code=True)
def _(
    kopech_health_passed,
    kopech_max_rhat,
    kopech_min_ess_bulk,
    kopech_min_ess_tail,
    kopech_n_div,
    kopech_n_draws_total,
    kopech_sample_seconds,
):
    mo.md(f"""
    **Diagnostics:** {kopech_n_div} divergence(s) out of {kopech_n_draws_total} draws
    in {kopech_sample_seconds:.1f}s. Minimum `ess_bulk` is {kopech_min_ess_bulk:.0f},
    minimum `ess_tail` is {kopech_min_ess_tail:.0f}, and maximum `r_hat` is
    {kopech_max_rhat:.3f}; health status **{kopech_health_passed}**.
    """)
    return


@app.cell(hide_code=True)
def _(
    build_kopech_spin_model,
    kopech_day,
    kopech_day_mean,
    kopech_day_std,
    kopech_day_z,
    kopech_idata,
    kopech_rate_z,
):
    kopech_day_grid = np.arange(kopech_day.min(), kopech_day.max() + 1)
    kopech_day_grid_z = (kopech_day_grid - kopech_day_mean) / kopech_day_std
    kopech_predictions = sample_fresh_model_predictions(
        kopech_idata,
        lambda: build_kopech_spin_model(
            kopech_day_z, kopech_rate_z, day_pred=kopech_day_grid_z
        )[0],
        var_names=["spin_pred"],
        random_seed=RANDOM_SEED,
    )
    return kopech_day_grid, kopech_day_grid_z, kopech_predictions


@app.cell(hide_code=True)
def _(kopech_predictions, kopech_rate_mean, kopech_rate_std):
    kopech_pred_draws = (
        kopech_predictions["predictions"]["spin_pred"] * kopech_rate_std
        + kopech_rate_mean
    )
    kopech_pred_mean = kopech_pred_draws.mean(dim=("chain", "draw"))
    kopech_pred_low, kopech_pred_high = eti_bounds(kopech_pred_draws)
    return kopech_pred_high, kopech_pred_low, kopech_pred_mean


@app.cell(hide_code=True)
def _(
    kopech_day,
    kopech_day_grid,
    kopech_pred_high,
    kopech_pred_low,
    kopech_pred_mean,
    kopech_rate,
):
    kopech_pred_fig = go.Figure()
    kopech_pred_fig.add_trace(
        go.Scatter(
            x=np.concatenate([kopech_day_grid, kopech_day_grid[::-1]]),
            y=np.concatenate([kopech_pred_high.values, kopech_pred_low.values[::-1]]),
            fill="toself",
            fillcolor="rgba(21,74,114,0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="89% ETI",
        )
    )
    kopech_pred_fig.add_trace(
        go.Scatter(
            x=kopech_day_grid,
            y=kopech_pred_mean.values,
            mode="lines",
            name="posterior mean",
            line=dict(color=PYMC_BLUE, width=3),
        )
    )
    kopech_pred_fig.add_trace(
        go.Scatter(
            x=kopech_day,
            y=kopech_rate,
            mode="markers",
            name="observed",
            marker=dict(color="black", size=6),
        )
    )
    kopech_pred_fig.update_layout(
        title="Kopech fastball spin rate: full-season prediction",
        xaxis_title="Day of year",
        yaxis_title="Spin rate (rpm)",
        template="plotly_white",
    )
    kopech_pred_fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### `.predict`: a single point estimate

    For comparison, `.predict` returns a closed-form mean and variance at
    a single hyperparameter point (e.g. the MAP) rather than the full
    posterior-averaged conditional above, the same shortcut used for the
    naive Theophylline fit earlier. It is cheap (one linear solve, no
    MCMC over predictions) but, like any plug-in estimate, understates
    uncertainty, since it ignores the spread over $(\ell, \eta, \sigma)$.
    """)
    return


@app.cell
def _(
    kopech_day_grid_z,
    kopech_gp,
    kopech_model,
    kopech_rate_mean,
    kopech_rate_std,
):
    with kopech_model:
        kopech_map = pm.find_MAP(progressbar=False)
        kopech_predict_mean_z, kopech_predict_var_z = kopech_gp.predict(
            kopech_day_grid_z[:, None], point=kopech_map, diag=True
        )
    kopech_predict_mean = kopech_predict_mean_z * kopech_rate_std + kopech_rate_mean
    kopech_predict_sd = np.sqrt(kopech_predict_var_z) * kopech_rate_std
    return kopech_predict_mean, kopech_predict_sd


@app.cell(hide_code=True)
def _(
    kopech_day,
    kopech_day_grid,
    kopech_predict_mean,
    kopech_predict_sd,
    kopech_rate,
):
    kopech_map_fig = go.Figure()
    kopech_map_fig.add_trace(
        go.Scatter(
            x=np.concatenate([kopech_day_grid, kopech_day_grid[::-1]]),
            y=np.concatenate(
                [
                    kopech_predict_mean + 2 * kopech_predict_sd,
                    (kopech_predict_mean - 2 * kopech_predict_sd)[::-1],
                ]
            ),
            fill="toself",
            fillcolor="rgba(129,194,64,0.25)",
            line=dict(color="rgba(255,255,255,0)"),
            name="MAP mean ± 2 SD",
        )
    )
    kopech_map_fig.add_trace(
        go.Scatter(
            x=kopech_day_grid,
            y=kopech_predict_mean,
            mode="lines",
            name="MAP prediction",
            line=dict(color=PYMC_GREEN, width=2),
        )
    )
    kopech_map_fig.add_trace(
        go.Scatter(
            x=kopech_day,
            y=kopech_rate,
            mode="markers",
            name="observed",
            marker=dict(color="black", size=6),
        )
    )
    kopech_map_fig.update_layout(
        title="Kopech fastball spin rate: `.predict` at the MAP",
        xaxis_title="Day of year",
        yaxis_title="Spin rate (rpm)",
        template="plotly_white",
    )
    kopech_map_fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Composed kernels on real data: NOAA tide gauge

    Every fit so far has used a single covariance function. This one uses a
    **sum** of three, which is the composition idea from notebook 2 put to
    work: there we drew from this kernel's prior and saw tide-shaped curves
    appear before any data were involved. Here we fit it.

    The API does not change. It is the same `pm.gp.Marginal` and
    `.marginal_likelihood` as above, handed a covariance function that
    happens to be a sum. What changes is that the kernel now encodes
    specific physical knowledge, and inference has to apportion the signal
    between its parts.

    ### Background

    NOAA CO-OPS station 9414290 (San Francisco, CA) is a long-record
    **mixed semidiurnal** tide station: the water level shows two
    superimposed periodic components, a roughly 12.42-hour
    **semidiurnal** (twice-daily) tide driven mainly by the moon, and a
    roughly 23.93-hour **diurnal** (once-daily) tide, riding on top of
    a slower background trend. Values below are hourly water levels in
    meters relative to the MLLW (mean lower low water) datum for a
    slice of 2019.
    """)
    return


@app.cell(hide_code=True)
def _():
    N_EXACT = 200
    tides = pl.read_csv(data_dir / "noaa_tides_hourly.csv")
    tides = tides.with_columns(
        pl.col("time").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M")
    )
    tides_slice = tides.head(N_EXACT)
    tides_slice.head()
    return (tides_slice,)


@app.cell(hide_code=True)
def _(tides_slice):
    tide_t0 = tides_slice["time"][0]
    tide_hours = (tides_slice["time"] - tide_t0).dt.total_minutes().to_numpy() / 60.0
    tide_level = tides_slice["water_level"].to_numpy()

    tide_hours_std = tide_hours.std(ddof=0)
    tide_level_mean, tide_level_std = tide_level.mean(), tide_level.std(ddof=0)

    X_tide = z(tide_hours).reshape(-1, 1)  # GP inputs are 2D: (n, 1)
    y_tide = z(tide_level)
    return (
        X_tide,
        tide_hours,
        tide_hours_std,
        tide_level,
        tide_level_mean,
        tide_level_std,
        y_tide,
    )


@app.cell(hide_code=True)
def _(tide_hours, tide_level):
    tide_fig = go.Figure()
    tide_fig.add_trace(
        go.Scatter(
            x=tide_hours,
            y=tide_level,
            mode="lines",
            line=dict(color=PYMC_BLUE),
        )
    )
    tide_fig.update_layout(
        title="San Francisco hourly water level, first slice of 2019",
        xaxis_title="Hours since slice start",
        yaxis_title="Water level (m, MLLW)",
        template="plotly_white",
    )
    tide_fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### What is fixed and what is free

    The kernel is a long-lengthscale `Matern52` trend plus two `Periodic`
    components, one per known physical cycle. Each `Periodic` component's
    period and within-cycle lengthscale are **fixed**: 12.42h and 23.93h
    are astronomical constants rather than things to estimate, and the
    within-cycle lengthscale is set to give each cycle a smooth, roughly
    sinusoidal shape rather than a sharp spike.

    That restraint is what makes the model samplable. The two periods are
    nearly commensurate, 23.93h being 1.93 times 12.42h, close enough to a
    2:1 ratio that each component can partly stand in for the other but
    not close enough for one to replace it. Free both periods and both
    within-cycle lengthscales and the posterior becomes strongly
    correlated and hard to explore. Leaving only the trend lengthscale and
    the three amplitudes free keeps the geometry well behaved while still
    letting the data decide how strong each tidal component is.
    """)
    return


@app.cell
def _(X_tide, tide_hours_std, y_tide):
    tide_semi_period_std = 12.42 / tide_hours_std
    tide_diurnal_period_std = 23.93 / tide_hours_std
    tide_coords = {
        "observation": np.arange(len(y_tide)),
        "feature": ["time"],
    }


    def build_tide_model(X_pred=None):
        coords = dict(tide_coords)
        if X_pred is not None:
            coords["prediction"] = np.arange(len(X_pred))
        with pm.Model(coords=coords) as tide_model:
            X_data = pm.Data("X", X_tide, dims=("observation", "feature"))
            tide_level_data = pm.Data("tide_level", y_tide, dims="observation")
            ell_trend = pm.LogNormal("ell_trend", mu=0, sigma=1)
            eta_trend = pm.HalfNormal("eta_trend", sigma=1)
            cov_trend = eta_trend**2 * pm.gp.cov.Matern52(1, ls=ell_trend)
            eta_semi = pm.HalfNormal("eta_semi", sigma=1)
            cov_semi = eta_semi**2 * pm.gp.cov.Periodic(
                1, period=tide_semi_period_std, ls=0.5
            )
            eta_diurnal = pm.HalfNormal("eta_diurnal", sigma=0.5)
            cov_diurnal = eta_diurnal**2 * pm.gp.cov.Periodic(
                1, period=tide_diurnal_period_std, ls=0.5
            )
            sigma_tide = pm.HalfNormal("sigma_tide", sigma=0.5)
            gp_tide = pm.gp.Marginal(cov_func=cov_trend + cov_semi + cov_diurnal)
            gp_tide.marginal_likelihood(
                "y",
                X=X_data,
                y=tide_level_data,
                sigma=sigma_tide,
                dims="observation",
            )
            if X_pred is not None:
                X_pred_data = pm.Data("X_pred", X_pred, dims=("prediction", "feature"))
                gp_tide.conditional("f_tide_pred", X_pred_data, dims="prediction")
        return tide_model


    tide_model = build_tide_model()
    return build_tide_model, tide_model


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Prior predictive check
    """)
    return


@app.cell
def _(tide_model):
    with tide_model:
        tide_prior_pred = pm.sample_prior_predictive(draws=500, random_seed=RANDOM_SEED)
    return (tide_prior_pred,)


@app.cell(hide_code=True)
def _(tide_prior_pred):
    tide_prior_draws = (
        tide_prior_pred["prior_predictive"]["y"]
        .stack(sample=("chain", "draw"))
        .transpose("sample", "observation")
    )
    return (tide_prior_draws,)


@app.cell(hide_code=True)
def _(X_tide, tide_prior_draws, y_tide):
    tide_prior_fig = go.Figure()
    rng_plot_tide = np.random.default_rng(0)
    for _i in rng_plot_tide.choice(
        tide_prior_draws.sizes["sample"], size=50, replace=False
    ):
        tide_prior_fig.add_trace(
            go.Scatter(
                x=X_tide[:, 0],
                y=tide_prior_draws.isel(sample=_i).values,
                mode="lines",
                line=dict(color=PYMC_LIGHT_BLUE, width=1),
                opacity=0.2,
                showlegend=False,
            )
        )
    tide_prior_fig.add_trace(
        go.Scatter(
            x=X_tide[:, 0],
            y=y_tide,
            mode="markers",
            marker=dict(color="black", size=4),
            name="observed (standardized)",
        )
    )
    tide_prior_fig.update_layout(
        title="Noisy prior-predictive tide observations vs. data",
        xaxis_title="time (standardized)",
        yaxis_title="water level (standardized)",
        template="plotly_white",
    )
    tide_prior_fig
    return


@app.cell(hide_code=True)
def _(tide_prior_draws, y_tide):
    mo.md(f"""
    **Plausibility check:** prior predictive draws span
    [{tide_prior_draws.min():.2f}, {tide_prior_draws.max():.2f}] on the
    standardized scale, comfortably bracketing the observed range
    [{y_tide.min():.2f}, {y_tide.max():.2f}], broad but not absurd,
    reasonable to proceed to sampling.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Sampling

    5 free hyperparameters (trend $\ell,\eta$; two periodic amplitudes
    $\eta$; noise $\sigma$, periods and within-cycle shape are fixed,
    as discussed above) over 200 points.
    """)
    return


@app.cell
def _(tide_model):
    with tide_model:
        tide_start = perf_counter()
        tide_idata = pm.sample(draws=500, tune=500, chains=4, random_seed=RANDOM_SEED)
        tide_sample_seconds = perf_counter() - tide_start
    tide_idata.to_netcdf(results_dir / "03_tide_exact_gp.nc")
    return tide_idata, tide_sample_seconds


@app.cell(hide_code=True)
def _(tide_idata, tide_model):
    tide_summary, tide_health_passed = inference_health(tide_idata, tide_model)
    tide_n_div = tide_summary.attrs["divergences"]
    tide_n_draws_total = (
        tide_idata["posterior"].sizes["chain"] * tide_idata["posterior"].sizes["draw"]
    )
    tide_min_ess_bulk = float(tide_summary["ess_bulk"].min())
    tide_min_ess_tail = float(tide_summary["ess_tail"].min())
    tide_max_rhat = float(tide_summary["r_hat"].astype(float).max())
    tide_summary.round(4)
    return (
        tide_health_passed,
        tide_max_rhat,
        tide_min_ess_bulk,
        tide_min_ess_tail,
        tide_n_div,
        tide_n_draws_total,
    )


@app.cell(hide_code=True)
def _(
    tide_health_passed,
    tide_max_rhat,
    tide_min_ess_bulk,
    tide_min_ess_tail,
    tide_n_div,
    tide_n_draws_total,
    tide_sample_seconds,
):
    mo.md(f"""
    **Diagnostics:** {tide_n_div} divergence(s) out of
    {tide_n_draws_total} draws in {tide_sample_seconds:.1f}s. Minimum
    `ess_bulk` is {tide_min_ess_bulk:.0f} and minimum `ess_tail` is
    {tide_min_ess_tail:.0f}, both above the 400 threshold, and
    maximum `r_hat` is {tide_max_rhat:.3f}. Health gate passed:
    **{tide_health_passed}**.
    """)
    return


@app.cell(hide_code=True)
def _(X_tide, build_tide_model, tide_idata):
    tide_ppc = sample_fresh_model_predictions(
        tide_idata,
        lambda: build_tide_model(X_tide),
        var_names=["f_tide_pred"],
        random_seed=RANDOM_SEED,
    )
    return (tide_ppc,)


@app.cell
def _(build_tide_model, tide_idata):
    tide_observed_model = build_tide_model()
    with tide_observed_model:
        tide_observed_ppc = pm.sample_posterior_predictive(
            posterior_subset(tide_idata),
            var_names=["y"],
            random_seed=RANDOM_SEED,
        )
    return (tide_observed_ppc,)


@app.cell(hide_code=True)
def _(tide_observed_ppc, y_tide):
    tide_replicated = tide_observed_ppc["posterior_predictive"]["y"]
    tide_ppc_location = tide_replicated.mean(dim="observation")
    tide_ppc_spread = tide_replicated.std(dim="observation")
    tide_observed_location = float(y_tide.mean())
    tide_observed_spread = float(y_tide.std(ddof=0))
    {
        "observed_mean": tide_observed_location,
        "predictive_mean_eti": tide_ppc_location.quantile(
            [0.055, 0.945], dim=("chain", "draw")
        ),
        "observed_sd": tide_observed_spread,
        "predictive_sd_eti": tide_ppc_spread.quantile(
            [0.055, 0.945], dim=("chain", "draw")
        ),
    }
    return


@app.cell(hide_code=True)
def _(tide_hours, tide_level_mean, tide_level_std, tide_ppc):
    tide_fit = tide_ppc["predictions"]["f_tide_pred"]
    tide_fit = tide_fit.rename({tide_fit.dims[-1]: "tide_hour"}).assign_coords(
        tide_hour=tide_hours
    )
    tide_fit = tide_fit * tide_level_std + tide_level_mean
    tide_fit_mean = tide_fit.mean(dim=("chain", "draw"))
    tide_fit_lo, tide_fit_hi = eti_bounds(tide_fit)
    return tide_fit_hi, tide_fit_lo, tide_fit_mean


@app.cell(hide_code=True)
def _(tide_fit_hi, tide_fit_lo, tide_fit_mean, tide_hours, tide_level):
    tide_fit_fig = go.Figure()
    tide_fit_fig.add_trace(
        go.Scatter(
            x=np.concatenate([tide_hours, tide_hours[::-1]]),
            y=np.concatenate([tide_fit_hi.values, tide_fit_lo.values[::-1]]),
            fill="toself",
            fillcolor="rgba(21,74,114,0.25)",
            line=dict(color="rgba(255,255,255,0)"),
            name="89% ETI",
        )
    )
    tide_fit_fig.add_trace(
        go.Scatter(
            x=tide_hours,
            y=tide_fit_mean.values,
            mode="lines",
            name="posterior mean fit",
            line=dict(color=PYMC_BLUE, width=2),
        )
    )
    tide_fit_fig.add_trace(
        go.Scatter(
            x=tide_hours,
            y=tide_level,
            mode="markers",
            name="observed",
            marker=dict(color="black", size=4),
        )
    )
    tide_fit_fig.update_layout(
        title="Additive-kernel GP fit: trend + semidiurnal + diurnal",
        xaxis_title="Hours since slice start",
        yaxis_title="Water level (m, MLLW)",
        template="plotly_white",
    )
    tide_fit_fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The additive kernel recovers the characteristic mixed-tide shape,
    alternating higher and lower high tides each day (the diurnal
    inequality) riding on a slowly drifting mean level, without ever
    being told the functional form of a tide curve, just its additive
    covariance structure.

    Fitting the **full year** (8,760 points) exactly this way is not
    practical. Exact `gp.Marginal` inference costs $O(n^3)$ per gradient
    evaluation, and this 200-point slice is already close to what that
    budget allows. Notebook 4 takes that constraint on directly, with two
    approximations whose cost grows far more slowly: sparse inducing-point
    GPs, and `pm.gp.HSGP`, a basis-function approximation that is *linear*
    in $n$.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Multi-output GPs

    We can generalize the single-pitcher model to **multiple outputs at
    once** using a multi-output Gaussian process: one GP that jointly
    models several related curves, sharing a single source of structure
    across them rather than fitting each pitcher separately. There are
    several approaches; we use the **Intrinsic Coregionalization Model**
    (ICM), which is the Hadamard product (element-wise product) of a
    `Coregion` kernel over the output index and an ordinary input kernel:

    $$\big[K_{\mathrm{ICM}}\big]_{ij}
    = B_{o_i o_j}\, K_{\mathrm{input}}(x_i, x_j), \qquad
    B = WW^\top + \mathrm{diag}(\kappa),$$

    where $o_i$ is the output (pitcher) index of observation $i$. (When every
    output is observed on the same input grid this matrix rearranges into the
    Kronecker product $B \otimes K_{\mathrm{input}}$, the form most ICM
    references write.)

    $B$ is a learned $p \times p$ output covariance ($p$ = number of
    pitchers): its diagonal sets each pitcher's own variance and its
    off-diagonal entries say how strongly two pitchers' curves co-vary.
    $W$ (a low-rank factor) and $\kappa$ (independent per-output
    variance) together parameterize $B$ so it is guaranteed positive
    semi-definite. Rather than picking on Kopech alone, we model the five
    pitchers with the most game appearances together.
    """)
    return


@app.function
def get_icm(input_dim, kernel, W=None, kappa=None, B=None, active_dims=None):
    """Hadamard product of a Coregion kernel and an input covariance function."""
    coreg = pm.gp.cov.Coregion(
        input_dim=input_dim, W=W, kappa=kappa, B=B, active_dims=active_dims
    )
    return kernel * coreg


@app.cell(hide_code=True)
def _(spin_rates):
    icm_pitcher_counts = (
        spin_rates.group_by("pitcher")
        .len()
        .sort(["len", "pitcher"], descending=[True, False])
    )
    icm_pitchers = icm_pitcher_counts.head(5)["pitcher"].to_list()
    icm_pitcher_index = {name: i for i, name in enumerate(icm_pitchers)}

    icm_data = (
        spin_rates.filter(pl.col("pitcher").is_in(icm_pitchers))
        .with_columns(
            pl.col("pitcher").replace_strict(icm_pitcher_index).alias("output_index")
        )
        .sort(["output_index", "day_of_year"])
    )

    icm_day = icm_data["day_of_year"].to_numpy().astype(float)
    icm_output = icm_data["output_index"].to_numpy().astype(float)
    icm_rate = icm_data["spin_rate"].to_numpy()

    icm_day_mean, icm_day_std = icm_day.mean(), icm_day.std(ddof=0)
    icm_rate_mean, icm_rate_std = icm_rate.mean(), icm_rate.std(ddof=0)
    icm_day_z = z(icm_day)
    icm_rate_z = z(icm_rate)
    icm_X = np.column_stack([icm_day_z, icm_output])
    return (
        icm_X,
        icm_day,
        icm_day_mean,
        icm_day_std,
        icm_output,
        icm_pitchers,
        icm_rate,
        icm_rate_mean,
        icm_rate_std,
        icm_rate_z,
    )


@app.cell(hide_code=True)
def _(icm_day, icm_output, icm_pitchers, icm_rate):
    icm_colors = ["#154A72", "#81C240", "#4A9EDE", "#40611F", "#8C1C13"]
    icm_raw_fig = go.Figure()
    for icm_plot_index, icm_plot_pitcher in enumerate(icm_pitchers):
        icm_plot_mask = icm_output == icm_plot_index
        icm_raw_fig.add_trace(
            go.Scatter(
                x=icm_day[icm_plot_mask],
                y=icm_rate[icm_plot_mask],
                mode="markers",
                name=icm_plot_pitcher,
                marker=dict(color=icm_colors[icm_plot_index], size=7),
            )
        )
    icm_raw_fig.update_layout(
        title="Fastball spin rate, five pitchers with the most game appearances",
        xaxis_title="Day of year",
        yaxis_title="Spin rate (rpm)",
        template="plotly_white",
    )
    icm_raw_fig
    return (icm_colors,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The model is essentially the same as the single-pitcher GP, except for
    the more complex covariance function that captures the dependence
    between pitchers, the assumption that they are governed by a similar
    underlying process, with pitcher-to-pitcher differences expressed
    through $B$ rather than five unrelated kernels.
    """)
    return


@app.cell
def _(icm_X, icm_pitchers, icm_rate_z):
    icm_coords = {
        "observation": np.arange(len(icm_rate_z)),
        "icm_feature": ["day_z", "output_index"],
        "pitcher": icm_pitchers,
        "pitcher_b": icm_pitchers,
        "coregion_rank": ["w1", "w2"],
    }

    def build_icm_model(X, y, *, X_pred=None, pred_coord=None):
        coords = dict(icm_coords)
        if X_pred is not None:
            coords["prediction"] = (
                np.asarray(pred_coord)
                if pred_coord is not None
                else np.arange(len(X_pred))
            )
        with pm.Model(coords=coords) as icm_model:
            X_data = pm.Data("X", X, dims=("observation", "icm_feature"))
            y_data = pm.Data("y_obs", y, dims="observation")
            ell = pm.Gamma("ell", alpha=2, beta=1)
            eta = pm.HalfNormal("eta", sigma=1)
            base_cov = eta**2 * pm.gp.cov.ExpQuad(input_dim=2, ls=ell, active_dims=[0])
            W = pm.Normal("W", mu=0, sigma=1, dims=("pitcher", "coregion_rank"))
            kappa = pm.Gamma("kappa", alpha=1.5, beta=1, dims="pitcher")
            B = pm.Deterministic(
                "B", pt.dot(W, W.T) + pt.diag(kappa), dims=("pitcher", "pitcher_b")
            )
            icm_cov = get_icm(input_dim=2, kernel=base_cov, B=B, active_dims=[1])
            sigma = pm.HalfNormal("sigma", sigma=0.5)
            icm_gp = pm.gp.Marginal(cov_func=icm_cov)
            icm_gp.marginal_likelihood(
                "y", X=X_data, y=y_data, sigma=sigma, dims="observation"
            )
            if X_pred is not None:
                X_pred_data = pm.Data(
                    "X_pred", X_pred, dims=("prediction", "icm_feature")
                )
                icm_gp.conditional("f_pred", X_pred_data, dims="prediction")
        return icm_model, icm_gp

    icm_model, icm_gp = build_icm_model(icm_X, icm_rate_z)
    return build_icm_model, icm_model


@app.cell
def _(icm_model):
    icm_model.compile_logp()(icm_model.initial_point())
    with icm_model:
        icm_sample_start = perf_counter()
        icm_idata = pm.sample(
            chains=4, draws=500, tune=500, target_accept=0.9, random_seed=RANDOM_SEED
        )
        icm_sample_seconds = perf_counter() - icm_sample_start
    icm_idata.to_netcdf(results_dir / "03_icm_multi_output_gp.nc")
    print(f"ICM multi-output GP sampling wall-time: {icm_sample_seconds:.1f}s")
    return icm_idata, icm_sample_seconds


@app.cell(hide_code=True)
def _(icm_idata, icm_model):
    icm_summary, icm_health_passed = inference_health(icm_idata, icm_model)
    icm_n_div = icm_summary.attrs["divergences"]
    icm_n_draws_total = (
        icm_idata["posterior"].sizes["chain"] * icm_idata["posterior"].sizes["draw"]
    )
    icm_min_ess_bulk = float(icm_summary["ess_bulk"].min())
    icm_min_ess_tail = float(icm_summary["ess_tail"].min())
    icm_max_rhat = float(icm_summary["r_hat"].astype(float).max())
    print(
        f"Divergences: {icm_n_div} / {icm_n_draws_total}; health passed: {icm_health_passed}"
    )
    icm_summary.round(4)
    return (
        icm_health_passed,
        icm_max_rhat,
        icm_min_ess_bulk,
        icm_min_ess_tail,
        icm_n_div,
        icm_n_draws_total,
    )


@app.cell(hide_code=True)
def _(
    icm_health_passed,
    icm_max_rhat,
    icm_min_ess_bulk,
    icm_min_ess_tail,
    icm_n_div,
    icm_n_draws_total,
    icm_sample_seconds,
):
    mo.md(rf"""
    **Diagnostics:** {icm_n_div} divergence(s) out of {icm_n_draws_total} draws in
    {icm_sample_seconds:.1f}s ({icm_sample_seconds / 60:.1f} min), this is the
    notebook's most expensive fit, five correlated curves and a $5\times 5$
    coregionalization matrix sampled jointly. Minimum `ess_bulk` is
    {icm_min_ess_bulk:.0f}, minimum `ess_tail` is {icm_min_ess_tail:.0f}, and
    maximum `r_hat` is {icm_max_rhat:.3f}; health status **{icm_health_passed}**.
    """)
    return


@app.cell(hide_code=True)
def _(
    build_icm_model,
    icm_X,
    icm_day,
    icm_day_mean,
    icm_day_std,
    icm_idata,
    icm_pitchers,
    icm_rate_z,
):
    icm_day_grid = np.arange(icm_day.min(), icm_day.max() + 1)
    icm_day_grid_z = (icm_day_grid - icm_day_mean) / icm_day_std
    icm_n_grid = len(icm_day_grid)
    icm_output_grid = np.repeat(np.arange(len(icm_pitchers)), icm_n_grid)
    icm_X_grid = np.column_stack(
        [np.tile(icm_day_grid_z, len(icm_pitchers)), icm_output_grid]
    )

    icm_predictions = sample_fresh_model_predictions(
        icm_idata,
        lambda: build_icm_model(icm_X, icm_rate_z, X_pred=icm_X_grid)[0],
        var_names=["f_pred"],
        random_seed=RANDOM_SEED,
    )
    return icm_day_grid, icm_n_grid, icm_predictions


@app.cell(hide_code=True)
def _(icm_predictions, icm_rate_mean, icm_rate_std):
    icm_pred_draws = (
        icm_predictions["predictions"]["f_pred"] * icm_rate_std + icm_rate_mean
    )
    icm_pred_mean = icm_pred_draws.mean(dim=("chain", "draw"))
    icm_pred_low, icm_pred_high = eti_bounds(icm_pred_draws)
    return icm_pred_high, icm_pred_low, icm_pred_mean


@app.cell(hide_code=True)
def _(
    icm_colors,
    icm_day,
    icm_day_grid,
    icm_n_grid,
    icm_output,
    icm_pitchers,
    icm_pred_high,
    icm_pred_low,
    icm_pred_mean,
    icm_rate,
):
    icm_prediction_plots = []
    for icm_row, icm_pitcher_name in enumerate(icm_pitchers, start=1):
        icm_slice = slice(icm_n_grid * (icm_row - 1), icm_n_grid * icm_row)
        icm_pred_fig = go.Figure()
        icm_pred_fig.add_trace(
            go.Scatter(
                x=np.concatenate([icm_day_grid, icm_day_grid[::-1]]),
                y=np.concatenate(
                    [
                        icm_pred_high.values[icm_slice],
                        icm_pred_low.values[icm_slice][::-1],
                    ]
                ),
                fill="toself",
                fillcolor="rgba(21,74,114,0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                showlegend=False,
            )
        )
        icm_pred_fig.add_trace(
            go.Scatter(
                x=icm_day_grid,
                y=icm_pred_mean.values[icm_slice],
                mode="lines",
                line=dict(color=icm_colors[icm_row - 1], width=2),
                showlegend=False,
            )
        )
        icm_obs_mask = icm_output == (icm_row - 1)
        icm_pred_fig.add_trace(
            go.Scatter(
                x=icm_day[icm_obs_mask],
                y=icm_rate[icm_obs_mask],
                mode="markers",
                marker=dict(color="black", size=5),
                showlegend=False,
            )
        )
        icm_pred_fig.update_layout(
            title=f"{icm_pitcher_name}: ICM posterior mean and 89% ETI",
            template="plotly_white",
            height=300,
            margin=dict(l=50, r=20, t=50, b=45),
        )
        icm_prediction_plots.append(mo.as_html(icm_pred_fig))
    mo.vstack(
        [
            mo.hstack(icm_prediction_plots[:3], widths="equal", gap=1),
            mo.hstack([*icm_prediction_plots[3:], mo.md("")], widths="equal", gap=1),
        ],
        gap=1,
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Sharing one covariance across pitchers assumes their spin-rate
    trajectories vary on a **common timescale** ($\ell$ is shared) and
    that any co-variation between pitchers is fully captured by the
    learned $5\times 5$ output covariance $B$, the same coregionalization
    structure applies to every pair of pitchers, rather than each pair
    having its own relationship. That is a strong, simplifying assumption:
    it buys statistical strength (a pitcher with few, noisy games borrows
    precision from the others through $B$) at the cost of flexibility (it
    cannot represent a pair of pitchers who are strongly linked while a
    third is unrelated to both, without a large enough rank in $W$).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Contrast: two ways to share structure across pitchers

    ICM and the hierarchical model below both let five (or three) pitchers'
    curves borrow strength from each other, but through different
    mechanisms. **ICM** shares structure through a single *learned
    cross-output covariance* $B$: every pitcher uses the same input kernel
    and the same lengthscale, and pairwise similarity is a free parameter
    fit directly from the data. The **hierarchical model** instead shares
    structure through an explicit *population curve*, one GP shape common
    to everyone, plus *sum-to-zero, constrained* per-pitcher deviations
    from it. ICM's assumption is symmetric and data-driven (it does not
    presuppose a "typical" pitcher); the hierarchical model's assumption is
    additive and interpretable (there is a definite population trajectory,
    and each pitcher is a labeled departure from it). Reach for ICM when
    you want the data to discover the co-variation structure itself; reach
    for the hierarchical form when a population-plus-deviation
    decomposition is scientifically meaningful and you want that structure
    named in the model rather than inferred as a free covariance.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Hierarchical GP, fastball spin rates

    ### Background

    Statcast fastball spin-rate (rpm) game means for three MLB pitchers,
    ten games each during **April–May 2021**. The 30 observations occupy a
    common grid of 24 dates. We model population time structure while
    retaining pitcher baselines and pitcher-specific functional departures.
    """)
    return


@app.cell(hide_code=True)
def _():
    spin = pl.read_csv(data_dir / "fastball_spin_rates.csv")
    spin = spin.with_columns(pl.col("game_date").str.strptime(pl.Date, "%Y-%m-%d"))

    # Three pitchers, first ten qualifying games: a small hierarchy to contrast with ICM.
    spin = (
        spin.filter(
            pl.col("pitcher").is_in(
                ["Buehler, Walker", "Hearn, Taylor", "Rodriguez, Richard"]
            )
        )
        .filter(pl.col("n_pitches") >= 10)
        .sort(["pitcher", "game_date"])
        .group_by("pitcher", maintain_order=True)
        .head(10)
    )
    spin.head()
    return (spin,)


@app.cell(hide_code=True)
def _(spin):
    pitchers = spin["pitcher"].unique(maintain_order=True).sort().to_list()
    pitcher_map = {name: i for i, name in enumerate(pitchers)}
    pitcher_idx_num = np.array([pitcher_map[p] for p in spin["pitcher"].to_list()])

    season_start = spin["game_date"].min()
    day_of_season = (
        (spin["game_date"] - season_start).dt.total_days().to_numpy().astype(float)
    )
    day_grid = np.unique(day_of_season)
    day_mean, day_std = day_of_season.mean(), day_of_season.std(ddof=0)
    day_z = z(day_of_season)
    day_grid_z = (day_grid - day_mean) / day_std
    date_map = {day: i for i, day in enumerate(day_grid)}
    date_idx = np.array([date_map[day] for day in day_of_season])

    spin_vals = spin["spin_rate"].to_numpy()
    n_pitches = spin["n_pitches"].to_numpy().astype(float)
    spin_mean, spin_std = spin_vals.mean(), spin_vals.std(ddof=0)
    spin_z = z(spin_vals)
    return (
        date_idx,
        day_grid,
        day_grid_z,
        day_of_season,
        day_z,
        n_pitches,
        pitcher_idx_num,
        pitchers,
        spin_mean,
        spin_std,
        spin_vals,
        spin_z,
    )


@app.cell(hide_code=True)
def _(day_of_season, pitcher_idx_num, pitchers, spin_vals):
    spin_raw_fig = go.Figure()
    _colors = ["#154A72", "#81C240", "#4A9EDE"]
    for _i, _pitcher in enumerate(pitchers):
        _mask = pitcher_idx_num == _i
        spin_raw_fig.add_trace(
            go.Scatter(
                x=np.asarray(day_of_season)[_mask],
                y=np.asarray(spin_vals)[_mask],
                mode="markers+lines",
                name=_pitcher,
                marker=dict(color=_colors[_i], size=7),
                line=dict(color=_colors[_i]),
            )
        )
    spin_raw_fig.update_layout(
        title="Fastball spin rate over the season, by pitcher",
        xaxis_title="Days since first game in dataset",
        yaxis_title="Spin rate (rpm)",
        template="plotly_white",
    )
    spin_raw_fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Hierarchical models: partial pooling

    These observations are grouped by pitcher. A fully pooled model would give
    every pitcher the same baseline and time pattern; separate models would fit
    each pitcher independently. With only ten games per pitcher, separate fits
    overreact to noise, while a pooled fit erases real differences.

    A hierarchical model gives each pitcher an intercept, but treats those
    intercepts as related draws from a population distribution:

    $$
    \alpha_j = \alpha_{\mathrm{pop}} + \tau_\alpha z_j,
    \qquad z_j \sim \mathcal{N}(0, 1).
    $$

    This is **partial pooling**. A pitcher with little information is pulled
    toward the population; a pitcher with clear data can depart from it. The
    data determine how much pooling is appropriate through $\tau_\alpha$.

    Here the same idea extends from one intercept to a whole trajectory:

    $$
    f_j(t) = \alpha_{\mathrm{pop}} + \alpha_j + f_{\mathrm{pop}}(t) + d_j(t).
    $$

    The population GP $f_{\mathrm{pop}}(t)$ describes shared time variation;
    $d_j(t)$ is pitcher $j$'s GP departure. Constraining the departures to
    average to zero makes the population curve the average trajectory rather
    than letting a deviation duplicate it.
    """)
    return


@app.cell
def _(
    date_idx,
    day_grid,
    day_grid_z,
    n_pitches,
    pitcher_idx_num,
    pitchers,
    spin_z,
):
    spin_coords = {
        "pitcher": pitchers,
        "day": day_grid,
        "feature": ["standardized_day"],
        "observation": np.arange(len(spin_z)),
    }

    def build_spin_model(eta_dev_sigma=0.5):
        """Build the identifiable population-plus-functional-deviation GP."""
        with pm.Model(coords=spin_coords) as spin_model:
            day_grid_data = pm.Data(
                "day_grid", day_grid_z[:, None], dims=("day", "feature")
            )
            date_idx_data = pm.Data("date_idx", date_idx, dims="observation")
            pitcher_idx_data = pm.Data(
                "pitcher_idx", pitcher_idx_num, dims="observation"
            )
            n_pitches_data = pm.Data("n_pitches", n_pitches, dims="observation")
            spin_data = pm.Data("spin_rate", spin_z, dims="observation")

            alpha_pop = pm.Normal("alpha_pop", mu=0, sigma=1)
            sigma_pitcher = pm.HalfNormal("sigma_pitcher", sigma=1)
            pitcher_offset_raw = pm.Normal(
                "pitcher_offset_raw", mu=0, sigma=1, dims="pitcher"
            )
            pitcher_intercept = pm.Deterministic(
                "pitcher_intercept",
                sigma_pitcher * (pitcher_offset_raw - pt.mean(pitcher_offset_raw)),
                dims="pitcher",
            )

            ell = pm.LogNormal("ell", mu=0, sigma=0.5)
            eta_pop = pm.HalfNormal("eta_pop", sigma=1)
            gp_pop = pm.gp.Latent(cov_func=eta_pop**2 * pm.gp.cov.Matern52(1, ls=ell))
            f_pop_raw = gp_pop.prior("f_pop_raw", X=day_grid_data, dims="day")
            f_pop = pm.Deterministic(
                "f_pop", f_pop_raw - pt.mean(f_pop_raw), dims="day"
            )

            eta_dev = pm.HalfNormal("eta_dev", sigma=eta_dev_sigma)
            gp_dev = pm.gp.Latent(cov_func=eta_dev**2 * pm.gp.cov.Matern52(1, ls=ell))
            f_dev_raw = gp_dev.prior(
                "f_dev_raw",
                X=day_grid_data,
                n_outputs=3,
                dims=("pitcher", "day"),
            )
            f_dev = pm.Deterministic(
                "f_dev",
                f_dev_raw
                - pt.mean(f_dev_raw, axis=0, keepdims=True)
                - pt.mean(f_dev_raw, axis=1, keepdims=True)
                + pt.mean(f_dev_raw),
                dims=("pitcher", "day"),
            )
            sigma_obs = pm.HalfNormal("sigma_obs", sigma=0.5)
            noise_scale = pm.math.sqrt(n_pitches_data / n_pitches.mean())
            pm.Normal(
                "spin_obs",
                mu=(
                    alpha_pop
                    + pitcher_intercept[pitcher_idx_data]
                    + f_pop[date_idx_data]
                    + f_dev[pitcher_idx_data, date_idx_data]
                ),
                sigma=sigma_obs / noise_scale,
                observed=spin_data,
                dims="observation",
            )
        return spin_model

    spin_model = build_spin_model()
    return build_spin_model, spin_model


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Prior predictive check
    """)
    return


@app.cell
def _(spin_model):
    with spin_model:
        spin_prior_pred = pm.sample_prior_predictive(draws=500, random_seed=RANDOM_SEED)
    return (spin_prior_pred,)


@app.cell(hide_code=True)
def _(spin_prior_pred):
    spin_prior_draws = (
        spin_prior_pred["prior_predictive"]["spin_obs"]
        .stack(sample=("chain", "draw"))
        .transpose("sample", "observation")
    )
    return (spin_prior_draws,)


@app.cell(hide_code=True)
def _(day_z, spin_prior_draws, spin_z):
    spin_prior_fig = go.Figure()
    rng_plot_spin = np.random.default_rng(0)
    for _i in rng_plot_spin.choice(
        spin_prior_draws.sizes["sample"], size=50, replace=False
    ):
        spin_prior_fig.add_trace(
            go.Scatter(
                x=day_z,
                y=spin_prior_draws.isel(sample=_i).values,
                mode="markers",
                marker=dict(color=PYMC_LIGHT_BLUE, size=4),
                opacity=0.15,
                showlegend=False,
            )
        )
    spin_prior_fig.add_trace(
        go.Scatter(
            x=day_z,
            y=spin_z,
            mode="markers",
            marker=dict(color="black", size=5),
            name="observed (standardized)",
        )
    )
    spin_prior_fig.update_layout(
        title="Prior predictive draws vs. standardized observed spin rate",
        xaxis_title="day of season (standardized)",
        yaxis_title="spin rate (standardized)",
        template="plotly_white",
    )
    spin_prior_fig
    return


@app.cell(hide_code=True)
def _(spin_prior_draws, spin_z):
    mo.md(f"""
    **Plausibility check:** prior predictive draws span
    [{spin_prior_draws.min():.2f}, {spin_prior_draws.max():.2f}] on the
    standardized scale, comfortably bracketing the observed range
    [{spin_z.min():.2f}, {spin_z.max():.2f}], broad but not absurd,
    reasonable to proceed to sampling.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Sampling

    Only 30 observations, but the non-centered hierarchy over a latent
    GP still has delicate geometry: 96 latent function values inferred
    from 30 data points, with two-way centering on top to keep the
    population and deviation GPs identified. A handful of divergences
    survive at this notebook's `target_accept=0.9` cap. Read them as a
    warning about how far to trust the per-pitcher deviation curves, not
    as a reason to keep raising the acceptance target until the warning
    goes away, which hides the geometry rather than fixing it.
    """)
    return


@app.cell
def _(spin_model):
    spin_model.compile_logp()(spin_model.initial_point())
    with spin_model:
        spin_start = perf_counter()
        # Divergences by acceptance target: 30 default, 16 at 0.90, 3 at 0.95.
        spin_idata = pm.sample(
            draws=500,
            tune=500,
            chains=4,
            target_accept=0.9,
            random_seed=RANDOM_SEED,
        )
        spin_sample_seconds = perf_counter() - spin_start
    spin_idata.to_netcdf(results_dir / "03_spin_hierarchical_gp.nc")
    print(f"Hierarchical spin-rate GP sampling wall-time: {spin_sample_seconds:.1f}s")
    return spin_idata, spin_sample_seconds


@app.cell(hide_code=True)
def _(spin_idata, spin_model):
    spin_summary, spin_health_passed = inference_health(spin_idata, spin_model)
    spin_n_div = spin_summary.attrs["divergences"]
    spin_n_draws_total = (
        spin_idata["posterior"].sizes["chain"] * spin_idata["posterior"].sizes["draw"]
    )
    spin_min_ess_bulk = float(spin_summary["ess_bulk"].min())
    spin_min_ess_tail = float(spin_summary["ess_tail"].min())
    spin_max_rhat = float(spin_summary["r_hat"].astype(float).max())
    print(
        f"Divergences: {spin_n_div} / {spin_n_draws_total}; "
        f"health passed: {spin_health_passed}"
    )
    spin_summary.round(4)
    return (
        spin_health_passed,
        spin_max_rhat,
        spin_min_ess_bulk,
        spin_min_ess_tail,
        spin_n_div,
        spin_n_draws_total,
    )


@app.cell(hide_code=True)
def _(
    spin_health_passed,
    spin_max_rhat,
    spin_min_ess_bulk,
    spin_min_ess_tail,
    spin_n_div,
    spin_n_draws_total,
    spin_sample_seconds,
):
    mo.md(f"""
    **Diagnostics:** {spin_n_div} divergence(s) out of
    {spin_n_draws_total} draws in {spin_sample_seconds:.1f}s. Computed
    over every free model variable. Minimum `ess_bulk` is
    {spin_min_ess_bulk:.0f}, minimum `ess_tail` is
    {spin_min_ess_tail:.0f}, and maximum `r_hat` is
    {spin_max_rhat:.3f}. Health gate passed: **{spin_health_passed}**.
    """)
    return


@app.cell
def _(spin_idata, spin_model):
    with spin_model:
        spin_observed_ppc = pm.sample_posterior_predictive(
            posterior_subset(spin_idata),
            var_names=["spin_obs"],
            random_seed=RANDOM_SEED,
        )
    return (spin_observed_ppc,)


@app.cell(hide_code=True)
def _(pitcher_idx_num, pitchers, spin_observed_ppc, spin_z):
    pitcher_labels = np.asarray(pitchers)[pitcher_idx_num]
    spin_replicated = spin_observed_ppc["posterior_predictive"][
        "spin_obs"
    ].assign_coords(pitcher_label=("observation", pitcher_labels))
    observed_spin = spin_replicated.isel(chain=0, draw=0).copy(data=spin_z)
    spin_ppc_by_pitcher = {
        "observed_group_mean": observed_spin.groupby("pitcher_label").mean(
            "observation"
        ),
        "predictive_group_mean_eti": spin_replicated.groupby("pitcher_label")
        .mean("observation")
        .quantile([0.055, 0.945], dim=("chain", "draw")),
    }
    spin_ppc_by_pitcher
    return


@app.cell(hide_code=True)
def _(spin_idata, spin_mean, spin_std):
    posterior = spin_idata["posterior"]
    population = posterior["alpha_pop"] + posterior["f_pop"]
    population_rpm = population * spin_std + spin_mean
    population_mean = population_rpm.mean(dim=("chain", "draw"))
    population_lo, population_hi = eti_bounds(population_rpm)
    return population_hi, population_lo, population_mean, posterior


@app.cell(hide_code=True)
def _(
    day_grid,
    day_of_season,
    pitcher_idx_num,
    pitchers,
    population_hi,
    population_lo,
    population_mean,
    posterior,
    spin_mean,
    spin_std,
    spin_vals,
):
    spin_traj_fig = go.Figure()
    spin_traj_fig.add_trace(
        go.Scatter(
            x=np.concatenate([day_grid, day_grid[::-1]]),
            y=np.concatenate([population_hi.values, population_lo.values[::-1]]),
            fill="toself",
            fillcolor="rgba(80,80,80,0.12)",
            line=dict(color="rgba(255,255,255,0)"),
            name="population 89% ETI",
        )
    )
    spin_traj_fig.add_trace(
        go.Scatter(
            x=day_grid,
            y=population_mean.values,
            mode="lines",
            name="population trajectory",
            line=dict(color="black", width=3, dash="dash"),
        )
    )
    _colors = ["#154A72", "#81C240", "#4A9EDE"]
    _fill_colors = [
        "rgba(21,74,114,0.15)",
        "rgba(129,194,64,0.15)",
        "rgba(74,158,222,0.15)",
    ]
    for _i, _pitcher in enumerate(pitchers):
        _trajectory = (
            posterior["alpha_pop"]
            + posterior["pitcher_intercept"].isel(pitcher=_i)
            + posterior["f_pop"]
            + posterior["f_dev"].isel(pitcher=_i)
        ) * spin_std + spin_mean
        _mean_traj = _trajectory.mean(dim=("chain", "draw"))
        _lo, _hi = eti_bounds(_trajectory)
        spin_traj_fig.add_trace(
            go.Scatter(
                x=np.concatenate([day_grid, day_grid[::-1]]),
                y=np.concatenate([_hi.values, _lo.values[::-1]]),
                fill="toself",
                fillcolor=_fill_colors[_i],
                line=dict(color="rgba(255,255,255,0)"),
                showlegend=False,
            )
        )
        spin_traj_fig.add_trace(
            go.Scatter(
                x=day_grid,
                y=_mean_traj.values,
                mode="lines",
                name=f"{_pitcher} trajectory",
                line=dict(color=_colors[_i], width=3),
            )
        )
        _mask = pitcher_idx_num == _i
        spin_traj_fig.add_trace(
            go.Scatter(
                x=np.asarray(day_of_season)[_mask],
                y=np.asarray(spin_vals)[_mask],
                mode="markers",
                marker=dict(color=_colors[_i], size=6, symbol="circle-open"),
                showlegend=False,
            )
        )
    spin_traj_fig.update_layout(
        title="Population and pitcher-specific functional-deviation trajectories",
        xaxis_title="Days since first game in dataset",
        yaxis_title="Spin rate (rpm)",
        template="plotly_white",
    )
    spin_traj_fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The solid trajectories combine the population curve, the sum-to-zero
    pitcher intercept, and each pitcher's two-way-centered functional
    deviation. Thus they can differ in shape, not merely vertical offset.

    ### Exercise: guided sensitivity for functional pooling

    `eta_dev` is the amplitude of each pitcher's GP deviation from the shared
    population curve: a smaller prior scale permits less pitcher-specific shape
    and therefore imposes stronger functional pooling. Use the already-defined
    `build_spin_model(eta_dev_sigma=...)` to refit with
    `eta_dev_sigma=0.1`, then compare its posterior-mean trajectories and
    `eta_dev` with the fitted default (`eta_dev_sigma=0.5`) model. Explain why
    this tighter setting pools functional shapes more strongly. You do not need
    to construct a posterior-predictive table.
    """)
    return


@app.cell
def _(exercise):
    @exercise
    def exercise_functional_pooling_sensitivity():
        # Build and fit the eta_dev_sigma=0.1 sensitivity model, then compare
        # its posterior-mean trajectories and eta_dev with the 0.5 baseline.
        # State why the smaller prior scale gives stronger functional pooling.
        ...

    exercise_functional_pooling_sensitivity()
    return


@app.cell(hide_code=True)
def _(
    build_spin_model,
    day_grid,
    inspect,
    pitchers,
    spin_idata,
    spin_mean,
    spin_std,
):
    sensitivity_results = {}

    def _fit_functional_pooling_sensitivity():
        if not sensitivity_results:
            sensitivity_model = build_spin_model(eta_dev_sigma=0.1)
            sensitivity_model.compile_logp()(sensitivity_model.initial_point())
            with sensitivity_model:
                sensitivity_results["idata"] = pm.sample(
                    draws=1000,
                    tune=1000,
                    chains=4,
                    target_accept=0.9,
                    random_seed=RANDOM_SEED,
                )
        return sensitivity_results["idata"]

    def solution_functional_pooling_sensitivity():
        sensitivity_idata = _fit_functional_pooling_sensitivity()
        baseline_posterior = spin_idata["posterior"]
        sensitivity_posterior = sensitivity_idata["posterior"]
        sensitivity_fig = go.Figure()
        colors = ["#154A72", "#81C240", "#4A9EDE"]
        for pitcher_index, pitcher in enumerate(pitchers):
            baseline_trajectory = (
                baseline_posterior["alpha_pop"]
                + baseline_posterior["pitcher_intercept"].isel(pitcher=pitcher_index)
                + baseline_posterior["f_pop"]
                + baseline_posterior["f_dev"].isel(pitcher=pitcher_index)
            ) * spin_std + spin_mean
            tighter_trajectory = (
                sensitivity_posterior["alpha_pop"]
                + sensitivity_posterior["pitcher_intercept"].isel(pitcher=pitcher_index)
                + sensitivity_posterior["f_pop"]
                + sensitivity_posterior["f_dev"].isel(pitcher=pitcher_index)
            ) * spin_std + spin_mean
            sensitivity_fig.add_trace(
                go.Scatter(
                    x=day_grid,
                    y=baseline_trajectory.mean(dim=("chain", "draw")).values,
                    mode="lines",
                    name=f"{pitcher}, baseline (0.5)",
                    line=dict(color=colors[pitcher_index], width=2, dash="dot"),
                )
            )
            sensitivity_fig.add_trace(
                go.Scatter(
                    x=day_grid,
                    y=tighter_trajectory.mean(dim=("chain", "draw")).values,
                    mode="lines",
                    name=f"{pitcher}, tighter (0.1)",
                    line=dict(color=colors[pitcher_index], width=3),
                )
            )
        sensitivity_fig.update_layout(
            title="Functional-pooling sensitivity: default versus tighter deviation prior",
            xaxis_title="Days since first game in dataset",
            yaxis_title="Spin rate (rpm)",
            template="plotly_white",
        )
        return mo.as_html(sensitivity_fig)

    def functional_pooling_sensitivity_details():
        sensitivity_idata = _fit_functional_pooling_sensitivity()
        return mo.vstack(
            [
                mo.md(
                    "With `eta_dev ~ HalfNormal(0.1)`, the prior places more "
                    "mass near zero than the default `HalfNormal(0.5)`. Each "
                    "pitcher's functional deviation is consequently pulled "
                    "closer to the shared population curve, so the solid "
                    "trajectories should be less distinct in shape than the "
                    "dotted default-fit trajectories."
                ),
                az.summary(
                    sensitivity_idata,
                    var_names=["eta_dev", "eta_pop", "sigma_pitcher", "sigma_obs"],
                ),
            ]
        )

    mo.accordion(
        {
            "Solution": mo.vstack(
                [
                    mo.md(
                        f"```python\n"
                        f"{inspect.getsource(_fit_functional_pooling_sensitivity)}"
                        f"\n{inspect.getsource(solution_functional_pooling_sensitivity)}"
                        f"\n{inspect.getsource(functional_pooling_sensitivity_details)}"
                        f"\n```"
                    ),
                    mo.lazy(
                        solution_functional_pooling_sensitivity,
                        show_loading_indicator=True,
                    ),
                    mo.lazy(functional_pooling_sensitivity_details),
                ]
            )
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Robust regression with a Student-t likelihood

    Every model so far has used a Gaussian observation layer. Real data
    sometimes have heavier tails than that, occasional large outliers a
    Gaussian likelihood cannot explain except by inflating $\sigma$ for
    every point. A **Student-t** likelihood has heavier tails than the
    Gaussian (controlled by its degrees-of-freedom parameter $\nu$: small
    $\nu$ means heavy tails, and $\nu \to \infty$ recovers the Gaussian),
    so it can treat a handful of large residuals as unsurprising rather
    than requiring a wider observation-noise band for every point. We
    generate data the same way as the very first example, a draw from a
    GP with known $\ell$, $\eta$, but now corrupt it with Student-t noise
    of known $\sigma$ and $\nu = 3$ instead of Gaussian noise, so the
    long-tailed outliers below are simulated, not real-data mess.
    """)
    return


@app.cell
def _():
    def simulate_robust_data(
        seed, n=100, ell_true=1.0, eta_true=3.0, sigma_true=2.0, nu_true=3.0
    ):
        """Draw one exact GP realization plus heavy-tailed Student-t noise."""
        rng = np.random.default_rng(seed)
        robust_X = np.linspace(0, 10, n)[:, None]
        mean_func = pm.gp.mean.Zero()
        cov_func = eta_true**2 * pm.gp.cov.Matern52(1, ell_true)
        f_true = rng.multivariate_normal(
            mean_func(robust_X).eval(),
            cov_func(robust_X).eval() + 1e-8 * np.eye(n),
            1,
        ).flatten()
        robust_y = f_true + sigma_true * rng.standard_t(nu_true, size=n)
        return robust_X, f_true, robust_y

    robust_ell_true, robust_eta_true, robust_sigma_true, robust_nu_true = (
        1.0,
        3.0,
        2.0,
        3.0,
    )
    robust_X, robust_f_true, robust_y = simulate_robust_data(
        RANDOM_SEED + 7,
        ell_true=robust_ell_true,
        eta_true=robust_eta_true,
        sigma_true=robust_sigma_true,
        nu_true=robust_nu_true,
    )
    return robust_X, robust_f_true, robust_y


@app.cell(hide_code=True)
def _(robust_X, robust_f_true, robust_y):
    robust_data_fig = go.Figure()
    robust_data_fig.add_trace(
        go.Scatter(
            x=robust_X[:, 0],
            y=robust_f_true,
            mode="lines",
            name="true f",
            line=dict(color=PYMC_BLUE, width=3),
        )
    )
    robust_data_fig.add_trace(
        go.Scatter(
            x=robust_X[:, 0],
            y=robust_y,
            mode="markers",
            name="observed data (Student-t noise)",
            marker=dict(color="black", size=6, opacity=0.6),
        )
    )
    robust_data_fig.update_layout(
        title="Simulated data with Student-t (heavy-tailed) noise, nu=3",
        xaxis_title="x",
        yaxis_title="y",
        template="plotly_white",
    )
    robust_data_fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### `gp.Latent` and `.prior`

    `gp.Latent` represents the GP's finite vector of function values as
    explicit latent variables, rather than integrating them out. Unlike
    `gp.Marginal`, it can be paired with **any** likelihood, Gaussian,
    Student-t, Poisson, Bernoulli. With inputs of finite size, `.prior`
    places a multivariate-normal prior directly on $\mathbf f$:

    $$\mathbf f \sim \text{MvNormal}(\mathbf m_x,\, \mathbf K_{xx}).$$

    By default PyMC uses a non-centered Cholesky representation
    ($\mathbf v \sim \mathcal N(0, 1)$,
    $\mathbf f = \mathbf m_x + \mathbf L \mathbf v$ with $\mathbf L$ the
    Cholesky factor of $\mathbf K_{xx}$), which reduces posterior
    dependence between $\mathbf f$ and the covariance hyperparameters. A
    Student-t observation layer breaks the Gaussian conjugacy that let
    $\mathbf f$ integrate out back in Part B, so here $\mathbf f$ must be
    carried as 100 explicit parameters and sampled jointly with
    $(\ell, \eta, \sigma, \nu)$, the same trade PyMC makes for the
    coal-disasters counts later in this notebook.
    """)
    return


@app.cell
def _(robust_X, robust_y):
    def build_robust_model(X, y, *, X_pred=None):
        coords = {"observation": np.arange(len(y)), "feature": ["x"]}
        if X_pred is not None:
            coords["prediction"] = np.arange(len(X_pred))
        with pm.Model(coords=coords) as robust_model:
            X_data = pm.Data("X", X, dims=("observation", "feature"))
            y_data = pm.Data("y_obs", y, dims="observation")
            ell = pm.Gamma("ell", alpha=2, beta=1)
            eta = pm.HalfCauchy("eta", beta=2)
            cov = eta**2 * pm.gp.cov.Matern52(1, ls=ell)
            robust_gp = pm.gp.Latent(cov_func=cov)
            f = robust_gp.prior("f", X=X_data, dims="observation")
            sigma = pm.HalfCauchy("sigma", beta=2)
            nu = pm.Gamma("nu", alpha=2, beta=0.1)
            pm.StudentT(
                "y", mu=f, sigma=sigma, nu=nu, observed=y_data, dims="observation"
            )
            if X_pred is not None:
                X_pred_data = pm.Data("X_pred", X_pred, dims=("prediction", "feature"))
                robust_gp.conditional("f_pred", X_pred_data, dims="prediction")
        return robust_model, robust_gp

    robust_model, robust_gp = build_robust_model(robust_X, robust_y)
    return (robust_model,)


@app.cell
def _(robust_model):
    robust_model.compile_logp()(robust_model.initial_point())
    with robust_model:
        robust_sample_start = perf_counter()
        robust_idata = pm.sample(
            chains=4,
            draws=500,
            tune=500,
            target_accept=0.9,
            random_seed=RANDOM_SEED,
        )
        robust_sample_seconds = perf_counter() - robust_sample_start
    robust_idata.to_netcdf(results_dir / "03_robust_studentt_gp.nc")
    print(f"Robust Student-t GP sampling wall-time: {robust_sample_seconds:.1f}s")
    return robust_idata, robust_sample_seconds


@app.cell(hide_code=True)
def _(robust_idata, robust_model):
    robust_summary, robust_health_passed = inference_health(robust_idata, robust_model)
    robust_n_div = robust_summary.attrs["divergences"]
    robust_n_draws_total = (
        robust_idata["posterior"].sizes["chain"]
        * robust_idata["posterior"].sizes["draw"]
    )
    robust_min_ess_bulk = float(robust_summary["ess_bulk"].min())
    robust_min_ess_tail = float(robust_summary["ess_tail"].min())
    robust_max_rhat = float(robust_summary["r_hat"].astype(float).max())
    print(
        f"Divergences: {robust_n_div} / {robust_n_draws_total}; "
        f"health passed: {robust_health_passed}"
    )
    robust_summary.round(4)
    return (
        robust_health_passed,
        robust_max_rhat,
        robust_min_ess_bulk,
        robust_min_ess_tail,
        robust_n_div,
        robust_n_draws_total,
    )


@app.cell(hide_code=True)
def _(
    robust_health_passed,
    robust_max_rhat,
    robust_min_ess_bulk,
    robust_min_ess_tail,
    robust_n_div,
    robust_n_draws_total,
    robust_sample_seconds,
):
    mo.md(rf"""
    **Diagnostics:** {robust_n_div} divergence(s) out of {robust_n_draws_total} draws
    in {robust_sample_seconds:.1f}s, over 104 free-RV rows (dominated by the
    100-element latent field $\mathbf f$). Minimum `ess_bulk` is
    {robust_min_ess_bulk:.0f}, minimum `ess_tail` is {robust_min_ess_tail:.0f},
    and maximum `r_hat` is {robust_max_rhat:.3f}; health status
    **{robust_health_passed}**.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Latent Poisson GP on coal-mining disasters

    ### Background

    The `coal_disasters` dataset records the number of British
    coal-mining disasters causing 10 or more deaths, by year, from 1851
    to 1962 (112 annual counts), a classic changepoint / intensity-
    estimation dataset in the Bayesian literature (Raftery & Akman,
    1986). The rate of disasters is believed to have declined over this
    period (improved safety regulation), but not necessarily smoothly
    or via any particular parametric form, another natural fit for a
    GP, this time on the log-rate of a **Poisson** count process.
    """)
    return


@app.cell(hide_code=True)
def _():
    coal = pl.read_csv(data_dir / "coal_disasters.csv")
    return (coal,)


@app.cell(hide_code=True)
def _(coal):
    year_vals = coal["year"].to_numpy()
    disaster_counts = coal["disasters"].to_numpy()
    return disaster_counts, year_vals


@app.cell(hide_code=True)
def _(disaster_counts, year_vals):
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
def _():
    mo.md(r"""
    ### Why we can't marginalize here

    `pm.gp.Marginal` relies on the Gaussian likelihood's conjugacy: if
    $y \mid f \sim \mathcal{N}(f, \sigma^2)$ and $f$ is a GP, then $f$
    can be integrated out in closed form, leaving a tractable marginal
    likelihood in $y$ alone. A Poisson likelihood,
    $y \mid f \sim \mathrm{Poisson}(e^f)$, has no such conjugate form ,
    there is no analytic way to integrate $f$ out of the joint
    distribution. So instead of marginalizing, `pm.gp.Latent` keeps the
    vector of latent function values $f$ as an explicit set of
    parameters (one per observed year here) and lets NUTS sample them
    jointly with the covariance hyperparameters. This is more
    expensive, we are now sampling a 112-dimensional latent vector
    plus hyperparameters, rather than 2–5 hyperparameters alone, but
    it is the only exact option once the likelihood leaves the Gaussian
    family.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.vstack(
        [
            mo.md(
                r"""
                **Question, why retain a latent count GP?** Write down two
                numbers from the series above before reading on: **33 of the
                112 years have zero disasters** (a zero fraction of 0.29),
                and the largest annual count is **6**. Now suppose you
                replaced the Poisson latent GP with a Gaussian marginal GP
                on $\sqrt{y}$, the classic variance-stabilizing transform,
                and a genuinely reasonable move for large counts. Predict
                what that model's replicated data would look like at the low
                end: what fraction of exact zeros can a continuous
                predictive distribution produce, and what happens to draws
                that land below zero when you square them back? Keep your
                answer; the posterior-predictive table later in this
                notebook reports the replicated zero fraction, and you can
                check yourself against it there.
                """
            ),
            mo.accordion(
                {
                    "Discussion": mo.md(
                        r"""
                        A continuous predictive distribution puts zero
                        probability on the value 0 exactly, so a transformed
                        Gaussian model can match the *average* count while
                        never reproducing the 29% of years that are exactly
                        zero, the discrepancy the table exposes directly.
                        Back-transforming makes it worse in a second way:
                        $\sqrt{y}$ draws below zero square back into
                        positive counts, folding the lower tail on itself
                        and quietly biasing low-count years upward. The
                        transform is fine when counts are uniformly large
                        and far from the boundary; here the data sit against
                        the boundary. The latent Poisson GP keeps the
                        discrete likelihood and the rate-scale
                        interpretation, so `rate(year)` means events per
                        year rather than a transformed-space artifact. If
                        you do evaluate a transformed alternative, compare
                        replicated zero fraction, spread, and maximum on the
                        original count scale, a good fit in transformed
                        space is not evidence about counts.
                        """
                    )
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### From latent function to disaster rate: the exp link

    It is worth tracing the generative chain the model encodes, because it
    is the template for *every* non-Gaussian GP. For each year $t_i$:

    1. A **latent function value** $f(t_i)$ is drawn jointly from the GP
       prior, these are the 112 correlated values `f = gp.prior("f", X=t)`.
       On this scale $f$ is unconstrained: it can be any real number,
       positive or negative, and neighbouring years are correlated through
       the Matérn kernel.
    2. An **exp link** maps it to a positive rate,
       $\lambda(t_i) = \exp\!\big(f(t_i)\big)$. Exponentiating guarantees
       $\lambda > 0$ (a Poisson mean must be positive) and makes the GP a
       model of the **log-rate**, so a straight-line drop in $f$ is a
       constant *proportional* decline in the rate, the natural scale for
       counts.
    3. A **Poisson likelihood** turns the rate into observed counts,
       $y_i \sim \mathrm{Poisson}\big(\lambda(t_i)\big)$.

    Contrast this directly with Part B. There, the link was the identity and
    the noise was Gaussian, so step 1's latent values integrated out and we
    sampled only five hyperparameters. Here the exp link and Poisson
    likelihood break conjugacy: there is no closed form for
    $\int \prod_i \mathrm{Poisson}(y_i \mid e^{f_i})\,
    \mathcal N(\mathbf f \mid \mathbf 0, K)\, d\mathbf f$, so we have no
    choice but to **keep $\mathbf f$ in the model and let NUTS explore all
    112 values jointly** with $(\ell, \eta)$. Carrying $\mathbf f$ explicitly
    is the price of leaving the Gaussian family, and the reason latent GPs
    sample more slowly than marginal ones.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.callout(
        mo.md(r"""

        **Exercise: latent Poisson GP for coal-mining disasters**

        1. Build a `pm.gp.Latent` model for `disaster_counts` using
           `year_vals[:, None]` as the input, raw years, not standardized,
           since the elicited lengthscale prior is in year units.
        2. Use the supplied `ell_prior` for the Matérn-5/2 lengthscale.
           It already encodes 94% prior mass in `[2, 10]` years: enough
           flexibility for multi-year changes, not one-year noise.
        3. Put a weakly informative prior on the GP amplitude, exponentiate
           the latent GP to get a positive `rate`, and use a `Poisson`
           likelihood.
        4. Sample the model and plot the posterior mean disaster rate with
           an 89% ETI over the observed counts.
        5. Optional: draw posterior-predictive replicated counts and compare
           their distribution visually with the observed counts. Does the
           Poisson likelihood look broadly plausible?
        """),
        kind="info",
    )
    return


@app.cell
def _():
    ell_prior = pz.maxent(pz.Gamma(), 2, 10, 0.94, plot=False)
    print(
        f"Elicited Gamma(alpha={ell_prior.alpha:.2f}, beta={ell_prior.beta:.2f}) "
        "lengthscale prior, 94% mass in [2, 10] years."
    )
    return (ell_prior,)


@app.cell
def _(exercise):
    @exercise
    def exercise_latent_poisson_coal_gp():
        # Use the supplied ell_prior to build a pm.gp.Latent Poisson model.
        # Sample it, return a posterior-rate plot, and optionally use
        # az.plot_ppc_dist for a qualitative posterior predictive check.
        ...

    exercise_latent_poisson_coal_gp()
    return


@app.cell(hide_code=True)
def _(disaster_counts, ell_prior, inspect, year_vals):
    coal_results = {}

    def _fit_latent_poisson_coal_gp():
        if not coal_results:
            coords = {"year": year_vals}
            with pm.Model(coords=coords) as coal_model:
                year_data = pm.Data("year_input", year_vals[:, None])
                count_data = pm.Data("disaster_count", disaster_counts, dims="year")
                ell = pm.Gamma("ell", alpha=ell_prior.alpha, beta=ell_prior.beta)
                eta = pm.HalfNormal("eta", sigma=1)
                cov = eta**2 * pm.gp.cov.Matern52(1, ls=ell)
                coal_gp = pm.gp.Latent(cov_func=cov)
                f = coal_gp.prior("f", X=year_data, dims="year")
                rate = pm.Deterministic("rate", pm.math.exp(f), dims="year")
                pm.Poisson("y", mu=rate, observed=count_data, dims="year")

                coal_model.compile_logp()(coal_model.initial_point())
                coal_results["idata"] = pm.sample(
                    chains=4,
                    draws=1000,
                    tune=1000,
                    target_accept=0.9,
                    random_seed=RANDOM_SEED,
                )
                coal_results["ppc"] = pm.sample_posterior_predictive(
                    coal_results["idata"], var_names=["y"], random_seed=RANDOM_SEED
                )
        return coal_results["idata"], coal_results["ppc"]

    def solution_latent_poisson_coal_gp():
        coal_idata, _ = _fit_latent_poisson_coal_gp()
        rate_draws = coal_idata["posterior"]["rate"]
        rate_mean = rate_draws.mean(dim=("chain", "draw"))
        rate_low, rate_high = eti_bounds(rate_draws)
        rate_fig = go.Figure()
        rate_fig.add_trace(
            go.Scatter(
                x=np.concatenate([year_vals, year_vals[::-1]]),
                y=np.concatenate([rate_high.values, rate_low.values[::-1]]),
                fill="toself",
                fillcolor="rgba(21,74,114,0.25)",
                line=dict(color="rgba(255,255,255,0)"),
                name="89% ETI",
            )
        )
        rate_fig.add_trace(
            go.Scatter(
                x=year_vals,
                y=rate_mean.values,
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
            title="Posterior disaster rate vs. observed counts",
            xaxis_title="Year",
            yaxis_title="Disasters / rate",
            template="plotly_white",
            height=350,
            margin=dict(l=55, r=20, t=50, b=45),
        )
        return mo.as_html(rate_fig)

    def latent_poisson_coal_gp_details():
        coal_idata, _ = _fit_latent_poisson_coal_gp()
        ppc_ax = az.plot_ppc_dist(coal_idata, var_names=["y"], num_samples=50)
        ppc_ax.figure.set_size_inches(6, 3.5)
        return mo.as_html(ppc_ax)

    def coal_solution_plots():
        return mo.hstack(
            [
                solution_latent_poisson_coal_gp(),
                latent_poisson_coal_gp_details(),
            ],
            widths="equal",
            gap=1,
        )

    mo.accordion(
        {
            "Solution": mo.vstack(
                [
                    mo.md(
                        f"```python\n"
                        f"{inspect.getsource(_fit_latent_poisson_coal_gp)}"
                        f"\n{inspect.getsource(solution_latent_poisson_coal_gp)}"
                        f"\n{inspect.getsource(latent_poisson_coal_gp_details)}"
                        f"\n{inspect.getsource(coal_solution_plots)}"
                        f"\n```"
                    ),
                    mo.lazy(coal_solution_plots, show_loading_indicator=True),
                ]
            )
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Multi-dimensional inputs, CDC PLACES county diabetes

    ### Background

    CDC PLACES ("Local Data for Better Health") publishes **model-based
    small-area estimates** of chronic-disease prevalence for every U.S.
    county, produced by combining BRFSS survey data with census
    covariates via multilevel regression and poststratification. These
    are *not* raw county censuses, they carry their own modeling
    uncertainty, but they are a standard source for county-level
    health geography. Below: diagnosed-diabetes prevalence (%) among
    adults for North Carolina's 100 counties, located by county-centroid
    longitude/latitude.
    """)
    return


@app.cell(hide_code=True)
def _():
    places = pl.read_csv(data_dir / "places_diabetes.csv")
    places.head()
    return (places,)


@app.cell(hide_code=True)
def _(places):
    places_lon = places["lon"].to_numpy()
    places_lat = places["lat"].to_numpy()
    places_diabetes = places["diabetes_pct"].to_numpy()
    places_obesity_z = z(places["obesity_pct"].to_numpy())

    # A local equirectangular plane gives both ARD axes the unit kilometres.
    latitude_origin = places_lat.mean()
    east_km = (
        (places_lon - places_lon.mean()) * 111.320 * np.cos(np.deg2rad(latitude_origin))
    )
    north_km = (places_lat - latitude_origin) * 110.574
    places_diabetes_mean = places_diabetes.mean()
    places_diabetes_std = places_diabetes.std(ddof=0)

    X_places = np.column_stack([east_km, north_km, places_obesity_z])
    y_places = z(places_diabetes)
    return (
        X_places,
        east_km,
        north_km,
        places_diabetes,
        places_diabetes_mean,
        places_diabetes_std,
        places_lat,
        places_lon,
        y_places,
    )


@app.cell(hide_code=True)
def _(places, places_diabetes, places_lat, places_lon):
    places_fig = go.Figure()
    places_fig.add_trace(
        go.Scatter(
            x=places_lon,
            y=places_lat,
            mode="markers",
            marker=dict(
                size=10,
                color=places_diabetes,
                colorscale="Blues",
                colorbar=dict(title="Diabetes %"),
                line=dict(color=PYMC_BLUE, width=1),
            ),
            text=places["county"].to_list(),
            hovertemplate="%{text}: %{marker.color:.1f}%<extra></extra>",
        )
    )
    places_fig.update_layout(
        title="NC county diabetes prevalence by centroid location",
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        template="plotly_white",
    )
    places_fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### ARD: one lengthscale per input dimension

    So far every kernel has had a single scalar lengthscale $\ell$. When
    the input is multi-dimensional, nothing stops us from giving each
    dimension its **own** lengthscale, a vector
    $\boldsymbol{\ell} = (\ell_{\text{east}}, \ell_{\text{north}})$ instead
    of a scalar. This is called **automatic relevance determination**
    (ARD): the posterior for each $\ell_d$ tells you how quickly
    correlation decays *in that direction alone*. A short lengthscale
    in one direction and a long one in the other means the underlying
    surface varies faster east-west than north-south (or vice versa) ,
    exactly the kind of anisotropy you'd expect from a health outcome
    correlated with, say, a north-south urban/rural gradient. PyMC's
    stationary kernels accept a vector for `ls` directly.
    """)
    return


@app.cell
def _(X_places, y_places):
    places_coords = {
        "county": np.arange(len(y_places)),
        "input": ["east_km", "north_km", "obesity_z"],
        "axis": ["east_km", "north_km"],
    }

    def build_places_model(X_pred=None):
        """Build the covariate-adjusted spatial GP on its kilometre plane."""
        coords = dict(places_coords)
        if X_pred is not None:
            coords["prediction"] = np.arange(len(X_pred))
        with pm.Model(coords=coords) as places_model:
            county_inputs = pm.Data("county_inputs", X_places, dims=("county", "input"))
            diabetes_data = pm.Data("diabetes", y_places, dims="county")
            alpha = pm.Normal("alpha", mu=0, sigma=1)
            beta_obesity = pm.Normal("beta_obesity", mu=0, sigma=1)
            ell_axis = pm.LogNormal("ell_axis", mu=np.log(150), sigma=0.5, dims="axis")
            eta = pm.HalfNormal("eta", sigma=1)
            sigma = pm.HalfNormal("sigma", sigma=0.5)
            mean_func = pm.gp.mean.Linear(coeffs=[0, 0, beta_obesity], intercept=alpha)
            gp_places = pm.gp.Marginal(
                mean_func=mean_func,
                cov_func=eta**2
                * pm.gp.cov.Matern52(3, active_dims=[0, 1], ls=ell_axis),
            )
            gp_places.marginal_likelihood(
                "y",
                X=county_inputs,
                y=diabetes_data,
                sigma=sigma,
                dims="county",
            )
            if X_pred is not None:
                prediction_inputs = pm.Data(
                    "prediction_inputs", X_pred, dims=("prediction", "input")
                )
                gp_places.conditional("f_grid", prediction_inputs, dims="prediction")
        return places_model

    places_model = build_places_model()
    return build_places_model, places_model


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Prior predictive check
    """)
    return


@app.cell
def _(places_model):
    with places_model:
        places_prior_pred = pm.sample_prior_predictive(
            draws=500, random_seed=RANDOM_SEED
        )
    return (places_prior_pred,)


@app.cell(hide_code=True)
def _(places_prior_pred):
    places_prior_draws = (
        places_prior_pred["prior_predictive"]["y"]
        .stack(sample=("chain", "draw"))
        .transpose("sample", "county")
    )
    return (places_prior_draws,)


@app.cell(hide_code=True)
def _(places_prior_draws, y_places):
    places_prior_fig = go.Figure()
    rng_plot_places = np.random.default_rng(0)
    # Markers, not lines: county index has no natural ordering, unlike the time series elsewhere.
    for _i in rng_plot_places.choice(
        places_prior_draws.sizes["sample"], size=50, replace=False
    ):
        places_prior_fig.add_trace(
            go.Scatter(
                x=np.arange(len(y_places)),
                y=places_prior_draws.isel(sample=_i).values,
                mode="markers",
                marker=dict(color=PYMC_LIGHT_BLUE, size=4),
                opacity=0.15,
                showlegend=False,
            )
        )
    places_prior_fig.add_trace(
        go.Scatter(
            x=np.arange(len(y_places)),
            y=y_places,
            mode="markers",
            marker=dict(color="black", size=5),
            name="observed (standardized)",
        )
    )
    places_prior_fig.update_layout(
        title="Prior predictive draws vs. standardized observed diabetes prevalence",
        xaxis_title="county index",
        yaxis_title="diabetes % (standardized)",
        template="plotly_white",
    )
    places_prior_fig
    return


@app.cell(hide_code=True)
def _(places_prior_draws, y_places):
    mo.md(f"""
    **Plausibility check:** prior predictive draws span
    [{places_prior_draws.min():.2f}, {places_prior_draws.max():.2f}] on
    the standardized scale, comfortably bracketing the observed range
    [{y_places.min():.2f}, {y_places.max():.2f}], broad but not absurd,
    reasonable to proceed to sampling.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Sampling

    100 counties is tiny for a GP, the training covariance is only
    $100 \times 100$, so its Cholesky decomposition is essentially
    instantaneous, and sampling is fast even with the extra ARD
    lengthscale dimension.
    """)
    return


@app.cell
def _(places_model):
    places_model.compile_logp()(places_model.initial_point())
    with places_model:
        places_start = perf_counter()
        # 0.90 removes the single divergence a 500-draw diagnostic run exposed.
        places_idata = pm.sample(
            draws=500,
            tune=500,
            chains=4,
            target_accept=0.9,
            random_seed=RANDOM_SEED,
        )
        places_sample_seconds = perf_counter() - places_start
    places_idata.to_netcdf(results_dir / "03_places_spatial_gp.nc")
    print(f"PLACES ARD-GP sampling wall-time: {places_sample_seconds:.1f}s")
    return places_idata, places_sample_seconds


@app.cell(hide_code=True)
def _(places_idata, places_model):
    places_summary, places_health_passed = inference_health(places_idata, places_model)
    places_n_div = places_summary.attrs["divergences"]
    places_n_draws_total = (
        places_idata["posterior"].sizes["chain"]
        * places_idata["posterior"].sizes["draw"]
    )
    places_min_ess_bulk = float(places_summary["ess_bulk"].min())
    places_min_ess_tail = float(places_summary["ess_tail"].min())
    places_max_rhat = float(places_summary["r_hat"].astype(float).max())
    print(
        f"Divergences: {places_n_div} / {places_n_draws_total}; "
        f"health passed: {places_health_passed}"
    )
    places_summary.round(4)
    return (
        places_health_passed,
        places_max_rhat,
        places_min_ess_bulk,
        places_min_ess_tail,
        places_n_div,
        places_n_draws_total,
    )


@app.cell(hide_code=True)
def _(
    places_health_passed,
    places_max_rhat,
    places_min_ess_bulk,
    places_min_ess_tail,
    places_n_div,
    places_n_draws_total,
    places_sample_seconds,
):
    mo.md(f"""
    **Diagnostics:** {places_n_div} divergence(s) out of
    {places_n_draws_total} draws in {places_sample_seconds:.1f}s.
    Minimum `ess_bulk` is {places_min_ess_bulk:.0f} and minimum
    `ess_tail` is {places_min_ess_tail:.0f}, both above the 400
    threshold, and maximum `r_hat` is {places_max_rhat:.3f}. Health gate
    passed: **{places_health_passed}**.

    Directional structure is reported below as the full posterior
    contrast between kilometre-scale east-west and north-south
    lengthscales, rather than a posterior-mean-only anisotropy claim.
    """)
    return


@app.cell(hide_code=True)
def _(places_idata):
    places_ell = places_idata["posterior"]["ell_axis"]
    places_directional_contrast = places_ell.sel(axis="east_km") - places_ell.sel(
        axis="north_km"
    )
    places_contrast_lo, places_contrast_hi = eti_bounds(places_directional_contrast)
    places_directional_summary = {
        "P(east-west lengthscale > north-south lengthscale)": float(
            (places_directional_contrast > 0).mean(dim=("chain", "draw"))
        ),
        "89% ETI (km)": (
            float(places_contrast_lo),
            float(places_contrast_hi),
        ),
    }
    places_directional_summary
    return


@app.cell
def _(build_places_model, places_idata):
    places_observed_model = build_places_model()
    with places_observed_model:
        places_observed_ppc = pm.sample_posterior_predictive(
            posterior_subset(places_idata),
            var_names=["y"],
            random_seed=RANDOM_SEED,
        )
    return (places_observed_ppc,)


@app.cell(hide_code=True)
def _(places_observed_ppc, y_places):
    places_replicated = places_observed_ppc["posterior_predictive"]["y"]
    {
        "observed_mean": float(y_places.mean()),
        "predictive_mean_eti": places_replicated.mean(dim="county").quantile(
            [0.055, 0.945], dim=("chain", "draw")
        ),
        "observed_sd": float(y_places.std(ddof=0)),
        "predictive_sd_eti": places_replicated.std(dim="county").quantile(
            [0.055, 0.945], dim=("chain", "draw")
        ),
    }
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Predicting on a spatial grid

    We evaluate the fitted GP's **full posterior-predictive**
    conditional on a regular longitude/latitude grid covering the
    county centroids, the same `.conditional` +
    `pm.sample_posterior_predictive` pattern used for `f_tide_pred` and
    `f_pred` above, rather than a single
    plug-in point estimate, so the heatmap below reflects the actual
    posterior mean over hyperparameters. The grid is kept modest
    (20x20 = 400 points) since `.conditional` draws a full-covariance
    sample per posterior draw and cost grows quickly with grid size.
    """)
    return


@app.cell(hide_code=True)
def _(east_km, north_km):
    grid_n = 20
    east_grid = np.linspace(east_km.min(), east_km.max(), grid_n)
    north_grid = np.linspace(north_km.min(), north_km.max(), grid_n)
    EAST_MESH, NORTH_MESH = np.meshgrid(east_grid, north_grid)
    spatial_grid = np.column_stack([EAST_MESH.ravel(), NORTH_MESH.ravel()])
    in_hull = (
        Delaunay(np.column_stack([east_km, north_km])).find_simplex(spatial_grid) >= 0
    )
    # Zero is average obesity on the standardized covariate scale.
    X_grid = np.column_stack([spatial_grid, np.zeros(len(spatial_grid))])
    in_hull_grid = in_hull.reshape(grid_n, grid_n)
    return EAST_MESH, NORTH_MESH, X_grid, in_hull_grid


@app.cell(hide_code=True)
def _(X_grid, build_places_model, places_idata):
    places_grid_ppc = sample_fresh_model_predictions(
        places_idata,
        lambda: build_places_model(X_grid),
        var_names=["f_grid"],
        random_seed=RANDOM_SEED,
    )
    return (places_grid_ppc,)


@app.cell(hide_code=True)
def _(
    EAST_MESH,
    NORTH_MESH,
    in_hull_grid,
    places_diabetes,
    places_diabetes_mean,
    places_diabetes_std,
    places_grid_ppc,
):
    grid_mean = places_grid_ppc["predictions"]["f_grid"].mean(dim=("chain", "draw"))
    grid_diabetes = grid_mean * places_diabetes_std + places_diabetes_mean
    grid_diabetes_2d = (
        grid_diabetes.assign_coords(
            east_km=("prediction", EAST_MESH.ravel()),
            north_km=("prediction", NORTH_MESH.ravel()),
        )
        .set_index(prediction=("north_km", "east_km"))
        .unstack("prediction")
        .where(in_hull_grid)
    )
    diabetes_cmin = min(float(grid_diabetes_2d.min()), float(places_diabetes.min()))
    diabetes_cmax = max(float(grid_diabetes_2d.max()), float(places_diabetes.max()))
    return diabetes_cmax, diabetes_cmin, grid_diabetes_2d


@app.cell(hide_code=True)
def _(
    diabetes_cmax,
    diabetes_cmin,
    east_km,
    grid_diabetes_2d,
    north_km,
    places,
    places_diabetes,
):
    heatmap_fig = go.Figure()
    heatmap_fig.add_trace(
        go.Heatmap(
            x=grid_diabetes_2d["east_km"].values,
            y=grid_diabetes_2d["north_km"].values,
            z=grid_diabetes_2d.values,
            colorscale="Blues",
            zmin=diabetes_cmin,
            zmax=diabetes_cmax,
            colorbar=dict(title="Diabetes %"),
        )
    )
    heatmap_fig.add_trace(
        go.Scatter(
            x=east_km,
            y=north_km,
            mode="markers",
            marker=dict(
                size=7,
                color=places_diabetes,
                colorscale="Blues",
                cmin=diabetes_cmin,
                cmax=diabetes_cmax,
                line=dict(color=PYMC_BLUE, width=1),
                showscale=False,
            ),
            text=places["county"].to_list(),
            hovertemplate="%{text}: %{marker.color:.1f}%<extra></extra>",
            name="county centroids",
        )
    )
    heatmap_fig.update_layout(
        title=(
            "Average-obesity conditional interpolation within the county convex hull"
        ),
        xaxis_title="Local east coordinate (km)",
        yaxis_title="Local north coordinate (km)",
        template="plotly_white",
    )
    heatmap_fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Where we are, and what's next

    Across this notebook's models you have now used both faces of GP
    inference repeatedly: `pm.gp.Marginal` whenever the likelihood is
    Gaussian (the simulated check, Theophylline, the single-pitcher and
    multi-output spin-rate models, PLACES), and `pm.gp.Latent` whenever it
    is not (the robust Student-t model, the coal-disasters exercise, and
    the hierarchical spin-rate model's population-plus-deviation
    structure).
    The dividing question was always the same: **Gaussian noise ⇒
    `Marginal`; anything else ⇒ `Latent`.**

    You have also seen the price both implementations pay. Every fit here
    inverts (or Cholesky-factors) an $n \times n$ covariance matrix, an
    $\mathcal{O}(n^3)$ operation, cheap at $n=11$ (Theophylline) or even
    $n=100$ (PLACES), but the wall-clock times already showed its shape:
    the 112-year coal series took several minutes to sample as a latent
    Poisson GP, and the five-pitcher ICM model, a $251 \times 251$
    covariance evaluated at every leapfrog step, took nearly six minutes.
    This cubic cost is the hard constraint on exact GPs, `Marginal` and
    `Latent` alike, and it does not improve by adding more compute per
    step; the $n \times n$ matrix itself is the bottleneck.

    $$
    k^*(x^*) = k(x^*,x^*) - k(x^*,x)^\top
    \underbrace{[k(x,x) + \sigma^2 I]^{-1}}_{\text{this is the problem}}
    k(x^*,x)
    $$

    **Notebook 4** takes on that constraint directly: sparse approximations
    (`gp.MarginalApprox`, where a small set of $m \ll n$ inducing points
    stands in for the full training covariance) and the Hilbert-space GP
    (`gp.HSGP`, a basis-function approximation with $\mathcal{O}(nm)$ cost
    instead of $\mathcal{O}(n^3)$), plus the workflow question every
    approximation raises: when is the bias it introduces worth the speed,
    versus simply waiting for the exact fit?
    """)
    return


if __name__ == "__main__":
    app.run()
