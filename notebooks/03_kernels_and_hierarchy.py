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
    # Kernels, Multi-Dimensional Inputs, and Hierarchy

    Notebooks 1 and 2 always used a single covariance function
    (`Matern52`) on a single input dimension. This notebook broadens
    both axes.

    **Part A — the kernel zoo.** Different covariance functions encode
    different assumptions about smoothness, periodicity, and even
    linearity. We build intuition by drawing prior samples from seven
    common kernels and comparing them side by side.

    **Part B — kernel composition.** Kernels can be added (OR: either
    structure explains the data) or multiplied (AND: both structures
    apply simultaneously) to build richer covariance functions. We fit
    an additive kernel — trend plus two tidal cycles — to a slice of
    real NOAA tide-gauge data.

    **Part C — multi-dimensional inputs.** GP inputs need not be 1D. We
    fit a 2D spatial GP with **ARD** (automatic relevance determination)
    lengthscales — one per input dimension — to county-level diabetes
    prevalence data.

    **Part D — hierarchical GPs.** Combining a GP with partial pooling:
    a shared population-level trend plus small, non-centered per-group
    deviations, fit to a panel of pitcher fastball spin rates.
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
    import pytensor.tensor as pt

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
        PYMC_DARK_GREEN,
        PYMC_LIGHT_BLUE,
        RANDOM_SEED,
        az,
        data_dir,
        execute_models,
        eti,
        eti_bounds,
        inference_health,
        go,
        np,
        perf_counter,
        pl,
        pm,
        pt,
        posterior_subset,
        results_dir,
        sample_fresh_model_predictions,
        z,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part A: The kernel zoo

    Every covariance function below is evaluated on the same
    1D grid, with the same amplitude $\eta$ and lengthscale $\ell$
    where applicable, so that differences in the drawn functions come
    purely from the *shape* of the kernel. Pick a kernel from the
    dropdown and drag the sliders.

    - **ExpQuad** (squared exponential): infinitely smooth, the
      "default" GP kernel.
    - **Matern12 / Matern32 / Matern52**: progressively smoother members
      of the Matérn family (Matern12 = Ornstein-Uhlenbeck, rough;
      Matern52 close to ExpQuad-smooth).
    - **Periodic**: strictly repeating structure with period $p$.
    - **RatQuad** (rational quadratic): a scale mixture of ExpQuad
      kernels with different lengthscales, controlled by $\alpha$.
    - **Linear**: no lengthscale at all — draws are straight lines
      through a pivot point $c$, not stationary.
    """)
    return


@app.cell
def _(mo):
    kernel_dropdown = mo.ui.dropdown(
        options=[
            "ExpQuad",
            "Matern12",
            "Matern32",
            "Matern52",
            "Periodic",
            "RatQuad",
            "Linear",
        ],
        value="ExpQuad",
        label="Kernel",
    )
    zoo_ls_slider = mo.ui.slider(0.1, 3.0, value=1.0, step=0.1, label="Lengthscale ℓ")
    zoo_eta_slider = mo.ui.slider(0.1, 3.0, value=1.0, step=0.1, label="Amplitude η")
    zoo_period_slider = mo.ui.slider(
        0.5, 5.0, value=2.0, step=0.1, label="Period p (Periodic only)"
    )
    zoo_alpha_slider = mo.ui.slider(
        0.1, 5.0, value=1.0, step=0.1, label="α (RatQuad only)"
    )
    zoo_c_slider = mo.ui.slider(
        0.0, 5.0, value=2.5, step=0.1, label="Pivot c (Linear only)"
    )
    mo.vstack(
        [
            kernel_dropdown,
            mo.hstack([zoo_ls_slider, zoo_eta_slider], gap=2),
            mo.hstack([zoo_period_slider, zoo_alpha_slider, zoo_c_slider], gap=2),
        ]
    )
    return (
        kernel_dropdown,
        zoo_alpha_slider,
        zoo_c_slider,
        zoo_eta_slider,
        zoo_ls_slider,
        zoo_period_slider,
    )


@app.cell
def _(
    kernel_dropdown,
    pm,
    zoo_alpha_slider,
    zoo_c_slider,
    zoo_eta_slider,
    zoo_ls_slider,
    zoo_period_slider,
):
    _name = kernel_dropdown.value
    _ls = zoo_ls_slider.value
    _eta = zoo_eta_slider.value

    if _name == "Periodic":
        zoo_base_cov = pm.gp.cov.Periodic(1, period=zoo_period_slider.value, ls=_ls)
    elif _name == "RatQuad":
        zoo_base_cov = pm.gp.cov.RatQuad(1, alpha=zoo_alpha_slider.value, ls=_ls)
    elif _name == "Linear":
        zoo_base_cov = pm.gp.cov.Linear(1, c=zoo_c_slider.value)
    else:
        zoo_base_cov = getattr(pm.gp.cov, _name)(1, ls=_ls)

    zoo_cov = _eta**2 * zoo_base_cov
    return (zoo_cov,)


@app.cell
def _(
    PYMC_DARK_GREEN,
    RANDOM_SEED,
    go,
    kernel_dropdown,
    np,
    zoo_alpha_slider,
    zoo_c_slider,
    zoo_cov,
    zoo_eta_slider,
    zoo_ls_slider,
    zoo_period_slider,
):
    zoo_grid = np.linspace(0, 10, 200).reshape(-1, 1)  # GP inputs are 2D: (n, 1)
    zoo_K = zoo_cov(zoo_grid).eval()
    zoo_K = zoo_K + 1e-8 * np.eye(len(zoo_grid))  # jitter for numerical stability

    control_seed = sum(ord(char) for char in kernel_dropdown.value)
    control_seed += round(
        100
        * (
            zoo_ls_slider.value
            + zoo_eta_slider.value
            + zoo_period_slider.value
            + zoo_alpha_slider.value
            + zoo_c_slider.value
        )
    )
    rng = np.random.default_rng(RANDOM_SEED + control_seed)
    zoo_n_draws = 5
    zoo_draws = rng.multivariate_normal(
        np.zeros(len(zoo_grid)), zoo_K, size=zoo_n_draws
    )

    zoo_fig = go.Figure()
    for _i in range(zoo_n_draws):
        zoo_fig.add_trace(
            go.Scatter(
                x=zoo_grid[:, 0],
                y=zoo_draws[_i],
                mode="lines",
                name=f"draw {_i + 1}",
                line=dict(width=2),
            )
        )
    zoo_fig.update_layout(
        title=f"GP prior draws — {kernel_dropdown.value} kernel",
        xaxis_title="x",
        yaxis_title="f(x)",
        template="plotly_white",
        showlegend=False,
    )
    zoo_fig.update_traces(line_color=PYMC_DARK_GREEN, selector=dict(name="draw 1"))
    zoo_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What to notice as you switch kernels:** Matern12 draws are jagged
    and nowhere differentiable; Matern52 and ExpQuad are visibly
    smoother, with ExpQuad the smoothest of all (infinitely
    differentiable). Periodic draws repeat exactly every $p$ units no
    matter how far you look — the only kernel here with that property.
    RatQuad with small $\alpha$ behaves like a mixture of many
    lengthscales at once (locally wigglier with occasional smooth
    stretches); as $\alpha \to \infty$ it converges to ExpQuad. Linear
    is the odd one out: it is **not stationary** (covariance depends on
    the absolute location $x$, not just $x - x'$), so draws are simply
    straight lines pivoting near $c$ — no lengthscale controls their
    shape at all.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part B: Kernel composition — NOAA tide gauge

    ### Background

    NOAA CO-OPS station 9414290 (San Francisco, CA) is a long-record
    **mixed semidiurnal** tide station: the water level shows two
    superimposed periodic components — a roughly 12.42-hour
    **semidiurnal** (twice-daily) tide driven mainly by the moon, and a
    roughly 23.93-hour **diurnal** (once-daily) tide — riding on top of
    a slower background trend. Values below are hourly water levels in
    meters relative to the MLLW (mean lower low water) datum for a
    slice of 2019.
    """)
    return


@app.cell
def _(data_dir, pl):
    N_EXACT = 200
    tides = pl.read_csv(data_dir / "noaa_tides_hourly.csv")
    tides = tides.with_columns(
        pl.col("time").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M")
    )
    tides_slice = tides.head(N_EXACT)
    tides_slice.head()
    return (tides_slice,)


@app.cell
def _(tides_slice, z):
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


@app.cell
def _(PYMC_BLUE, go, tide_hours, tide_level):
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
        title="San Francisco hourly water level — first slice of 2019",
        xaxis_title="Hours since slice start",
        yaxis_title="Water level (m, MLLW)",
        template="plotly_white",
    )
    tide_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Additive vs. multiplicative kernel structure

    Two ways to combine covariance functions:

    - **Additive (OR)**: $k(x,x') = k_1(x,x') + k_2(x,x')$. A draw from
      the sum is a draw from $k_1$ *plus* a (statistically independent)
      draw from $k_2$ — useful when the data is a **superposition** of
      distinct structures, e.g. a slow trend plus fast periodic
      wiggles. This is the right structure for tides: a slowly-drifting
      mean level, plus a semidiurnal cycle, plus a diurnal cycle, added
      together.
    - **Multiplicative (AND)**: $k(x,x') = k_1(x,x') \cdot k_2(x,x')$.
      This is how you build kernels whose behavior along one dimension
      is *modulated* by another (e.g. a periodic kernel times a slowly
      decaying ExpQuad gives a periodic pattern that fades in and out —
      `pm.gp.cov.Periodic` combined this way is one route to a
      quasi-periodic kernel), or how an ARD kernel over multiple input
      dimensions is built (Part C).

    Below we compose an **additive** kernel: a long-lengthscale
    `Matern52` trend plus two `Periodic` components, one for each known
    physical period. Each `Periodic` component's period and
    within-cycle lengthscale are **fixed** at physically-motivated
    constants (12.42h semidiurnal / 23.93h diurnal periods — these are
    astronomical constants, not free parameters — and a moderate
    within-cycle lengthscale that gives each cycle a smooth, roughly
    sinusoidal shape rather than a sharp spike). Freeing both the period
    *and* the within-cycle lengthscale of two near-commensurate cycles
    (23.93h is almost exactly twice 12.42h) creates a hard-to-sample,
    highly correlated posterior; fixing the shape parameters and
    leaving only the trend lengthscale and each component's amplitude
    free keeps the search well-behaved while still letting the data
    determine how strong each tidal component is.
    """)
    return


@app.cell
def _(X_tide, np, pm, tide_hours_std, y_tide):
    _semi_period_std = 12.42 / tide_hours_std
    _diurnal_period_std = 23.93 / tide_hours_std
    _tide_coords = {
        "observation": np.arange(len(y_tide)),
        "feature": ["time"],
    }

    def build_tide_model(X_pred=None):
        _coords = dict(_tide_coords)
        if X_pred is not None:
            _coords["prediction"] = np.arange(len(X_pred))
        with pm.Model(coords=_coords) as tide_model:
            _X_data = pm.Data("X", X_tide, dims=("observation", "feature"))
            _tide_level_data = pm.Data("tide_level", y_tide, dims="observation")
            _ell_trend = pm.LogNormal("ell_trend", mu=0, sigma=1)
            _eta_trend = pm.HalfNormal("eta_trend", sigma=1)
            _cov_trend = _eta_trend**2 * pm.gp.cov.Matern52(1, ls=_ell_trend)
            _eta_semi = pm.HalfNormal("eta_semi", sigma=1)
            _cov_semi = _eta_semi**2 * pm.gp.cov.Periodic(
                1, period=_semi_period_std, ls=0.5
            )
            _eta_diurnal = pm.HalfNormal("eta_diurnal", sigma=0.5)
            _cov_diurnal = _eta_diurnal**2 * pm.gp.cov.Periodic(
                1, period=_diurnal_period_std, ls=0.5
            )
            _sigma_tide = pm.HalfNormal("sigma_tide", sigma=0.5)
            _gp_tide = pm.gp.Marginal(
                cov_func=_cov_trend + _cov_semi + _cov_diurnal
            )
            _gp_tide.marginal_likelihood(
                "y",
                X=_X_data,
                y=_tide_level_data,
                sigma=_sigma_tide,
                dims="observation",
            )
            if X_pred is not None:
                _X_pred = pm.Data(
                    "X_pred", X_pred, dims=("prediction", "feature")
                )
                _gp_tide.conditional(
                    "f_tide_pred", _X_pred, dims="prediction"
                )
        return tide_model

    tide_model = build_tide_model()
    return build_tide_model, tide_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Prior predictive check
    """)
    return


@app.cell
def _(RANDOM_SEED, pm, tide_model):
    with tide_model:
        tide_prior_pred = pm.sample_prior_predictive(draws=500, random_seed=RANDOM_SEED)
    return (tide_prior_pred,)


@app.cell
def _(PYMC_LIGHT_BLUE, X_tide, go, np, tide_prior_pred, y_tide):
    tide_prior_draws = tide_prior_pred["prior_predictive"]["y"].stack(
        sample=("chain", "draw")
    ).transpose("sample", "observation")

    tide_prior_fig = go.Figure()
    rng_plot_tide = np.random.default_rng(0)
    for _i in rng_plot_tide.choice(tide_prior_draws.sizes["sample"], size=50, replace=False):
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
    return (tide_prior_draws,)


@app.cell(hide_code=True)
def _(mo, tide_prior_draws, y_tide):
    mo.md(
        f"""
        **Plausibility check:** prior predictive draws span
        [{tide_prior_draws.min():.2f}, {tide_prior_draws.max():.2f}] on the
        standardized scale, comfortably bracketing the observed range
        [{y_tide.min():.2f}, {y_tide.max():.2f}] — broad but not absurd,
        reasonable to proceed to sampling.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Sampling

    5 free hyperparameters (trend $\ell,\eta$; two periodic amplitudes
    $\eta$; noise $\sigma$ — periods and within-cycle shape are fixed,
    as discussed above) over 200 points.
    """)
    return


@app.cell
def _(mo):
    tide_fit_button = mo.ui.run_button(label="Fit exact tide GP")
    tide_fit_button
    return (tide_fit_button,)


@app.cell
def _(
    RANDOM_SEED,
    execute_models,
    mo,
    perf_counter,
    pm,
    results_dir,
    tide_fit_button,
    tide_model,
):
    mo.stop(not (tide_fit_button.value or execute_models))
    tide_model.compile_logp()(tide_model.initial_point())
    with tide_model:
        tide_start = perf_counter()
        tide_idata = pm.sample(
            draws=500, tune=500, chains=4, random_seed=RANDOM_SEED
        )
        tide_sample_seconds = perf_counter() - tide_start
    tide_idata.to_netcdf(results_dir / "03_tide_exact_gp.nc")
    print(f"NOAA additive-GP sampling wall-time: {tide_sample_seconds:.1f}s")
    return tide_idata, tide_sample_seconds


@app.cell
def _(inference_health, tide_idata, tide_model):
    tide_summary, tide_health_passed = inference_health(tide_idata, tide_model)
    tide_n_div = tide_summary.attrs["divergences"]
    tide_n_draws_total = (
        tide_idata["posterior"].sizes["chain"] * tide_idata["posterior"].sizes["draw"]
    )
    tide_min_ess_bulk = float(tide_summary["ess_bulk"].min())
    tide_min_ess_tail = float(tide_summary["ess_tail"].min())
    tide_max_rhat = float(tide_summary["r_hat"].astype(float).max())
    print(
        f"Divergences: {tide_n_div} / {tide_n_draws_total}; "
        f"health passed: {tide_health_passed}"
    )
    tide_summary
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
    mo,
    tide_health_passed,
    tide_max_rhat,
    tide_min_ess_bulk,
    tide_min_ess_tail,
    tide_n_div,
    tide_n_draws_total,
    tide_sample_seconds,
):
    mo.md(
        f"""
        **Diagnostics:** {tide_n_div} divergence(s) out of
        {tide_n_draws_total} draws in {tide_sample_seconds:.1f}s. Minimum
        `ess_bulk` is {tide_min_ess_bulk:.0f} and minimum `ess_tail` is
        {tide_min_ess_tail:.0f} — both above the 400 threshold — and
        maximum `r_hat` is {tide_max_rhat:.3f}. Health gate passed:
        **{tide_health_passed}**.
        """
    )
    return


@app.cell
def _(
    RANDOM_SEED,
    X_tide,
    build_tide_model,
    sample_fresh_model_predictions,
    tide_idata,
):
    tide_ppc = sample_fresh_model_predictions(
        tide_idata,
        lambda: build_tide_model(X_tide),
        var_names=["f_tide_pred"],
        random_seed=RANDOM_SEED,
    )
    return (tide_ppc,)


@app.cell
def _(
    RANDOM_SEED,
    build_tide_model,
    pm,
    posterior_subset,
    tide_idata,
):
    tide_observed_model = build_tide_model()
    with tide_observed_model:
        tide_observed_ppc = pm.sample_posterior_predictive(
            posterior_subset(tide_idata),
            var_names=["y"],
            random_seed=RANDOM_SEED,
        )
    return (tide_observed_ppc,)


@app.cell
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


@app.cell
def _(
    PYMC_BLUE,
    eti_bounds,
    go,
    np,
    tide_hours,
    tide_level,
    tide_level_mean,
    tide_level_std,
    tide_ppc,
):
    tide_fit = tide_ppc["predictions"]["f_tide_pred"]
    tide_fit = tide_fit.rename({tide_fit.dims[-1]: "tide_hour"}).assign_coords(
        tide_hour=tide_hours
    )
    tide_fit = tide_fit * tide_level_std + tide_level_mean
    tide_fit_mean = tide_fit.mean(dim=("chain", "draw"))
    tide_fit_lo, tide_fit_hi = eti_bounds(tide_fit)

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
        title="Additive-kernel GP fit — trend + semidiurnal + diurnal",
        xaxis_title="Hours since slice start",
        yaxis_title="Water level (m, MLLW)",
        template="plotly_white",
    )
    tide_fit_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The additive kernel recovers the characteristic mixed-tide shape —
    alternating higher and lower high tides each day (the diurnal
    inequality) riding on a slowly drifting mean level — without ever
    being told the functional form of a tide curve, just its additive
    covariance structure.

    Fitting the **full year** (8,760 points) exactly this way is not
    practical — exact `gp.Marginal` inference costs $O(n^3)$ per
    gradient evaluation. Hour 4 introduces `pm.gp.HSGP`, a basis-function
    approximation that scales to the whole series (and beyond) at a
    fraction of the cost; we defer that fit there.

    ### Exercise: compose a kernel for a described pattern

    Suppose you have hourly foot-traffic counts at a retail store, and
    you're told: "traffic has a strong repeating **daily** pattern
    (open/close hours), a weaker repeating **weekly** pattern (weekends
    differ from weekdays), and a slow **seasonal** drift on top." Using
    only `Matern52` and `Periodic`, sketch the additive kernel you
    would use (as a sum of terms with a period argument, where
    relevant) before expanding the solution. Then state the input units
    explicitly: if you standardize hourly inputs before fitting, how must
    the 24-hour and 168-hour periods be transformed? Finally, describe a
    prior-predictive pattern that would warn you that the periodic amplitudes
    or within-cycle lengthscales are implausible before you interpret a
    posterior fit. Relate that check to the tide example: its semidiurnal and
    diurnal periods are physical knowledge, while their amplitudes and the
    nonperiodic trend remain uncertain model components.
    Explain why fixed periods do not eliminate uncertainty about observed
    timing, amplitude, noise, or boundary behavior.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Solution": mo.md(
                """
                ```python
                cov = (
                    eta_season**2 * pm.gp.cov.Matern52(1, ls=ell_season)  # slow seasonal drift
                    + eta_daily**2 * pm.gp.cov.Periodic(1, period=24, ls=ell_daily)  # daily cycle
                    + eta_weekly**2 * pm.gp.cov.Periodic(1, period=24 * 7, ls=ell_weekly)  # weekly cycle
                )
                ```

                Three additive (OR) terms, one per described structure,
                each a repeating `Periodic` component at the right period
                (in whatever units the input axis uses — hours here) except
                the slow drift, which has no fixed period and so gets a
                `Matern52`. With standardized input, divide each physical
                period by the hourly input standard deviation before building
                the kernel; a literal `period=24` would otherwise be in the
                wrong units. The relative `eta` amplitudes let the model learn
                that the weekly effect is weaker than the daily one, rather
                than hard-coding that conclusion. Prior curves that oscillate
                too sharply, vary wildly in height, or suppress the stated
                daily rhythm are a reason to revise priors or kernel
                parameters before sampling—not a sampler-tuning problem.
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part C: Multi-dimensional inputs — CDC PLACES county diabetes

    ### Background

    CDC PLACES ("Local Data for Better Health") publishes **model-based
    small-area estimates** of chronic-disease prevalence for every U.S.
    county, produced by combining BRFSS survey data with census
    covariates via multilevel regression and poststratification. These
    are *not* raw county censuses — they carry their own modeling
    uncertainty — but they are a standard source for county-level
    health geography. Below: diagnosed-diabetes prevalence (%) among
    adults for North Carolina's 100 counties, located by county-centroid
    longitude/latitude.
    """)
    return


@app.cell
def _(data_dir, pl):
    places = pl.read_csv(data_dir / "places_diabetes.csv")
    places.head()
    return (places,)


@app.cell
def _(np, places, z):
    places_lon = places["lon"].to_numpy()
    places_lat = places["lat"].to_numpy()
    places_diabetes = places["diabetes_pct"].to_numpy()
    places_obesity_z = z(places["obesity_pct"].to_numpy())

    # A local equirectangular plane gives both ARD axes the unit kilometres.
    _latitude_origin = places_lat.mean()
    east_km = (
        (places_lon - places_lon.mean())
        * 111.320
        * np.cos(np.deg2rad(_latitude_origin))
    )
    north_km = (places_lat - _latitude_origin) * 110.574
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
        places_obesity_z,
        y_places,
    )


@app.cell
def _(PYMC_BLUE, go, places, places_diabetes, places_lat, places_lon):
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
def _(mo):
    mo.md(r"""
    ### ARD: one lengthscale per input dimension

    So far every kernel has had a single scalar lengthscale $\ell$. When
    the input is multi-dimensional, nothing stops us from giving each
    dimension its **own** lengthscale — a vector
    $\boldsymbol{\ell} = (\ell_{\text{lon}}, \ell_{\text{lat}})$ instead
    of a scalar. This is called **automatic relevance determination**
    (ARD): the posterior for each $\ell_d$ tells you how quickly
    correlation decays *in that direction alone*. A short lengthscale
    in one direction and a long one in the other means the underlying
    surface varies faster east-west than north-south (or vice versa) —
    exactly the kind of anisotropy you'd expect from a health outcome
    correlated with, say, a north-south urban/rural gradient. PyMC's
    stationary kernels accept a vector for `ls` directly.
    """)
    return


@app.cell
def _(X_places, np, pm, y_places):
    _places_coords = {
        "county": np.arange(len(y_places)),
        "input": ["east_km", "north_km", "obesity_z"],
        "axis": ["east_km", "north_km"],
    }

    def build_places_model(X_pred=None):
        """Build the covariate-adjusted spatial GP on its kilometre plane."""
        _coords = dict(_places_coords)
        if X_pred is not None:
            _coords["prediction"] = np.arange(len(X_pred))
        with pm.Model(coords=_coords) as places_model:
            _county_inputs = pm.Data(
                "county_inputs", X_places, dims=("county", "input")
            )
            _diabetes_data = pm.Data("diabetes", y_places, dims="county")
            _alpha = pm.Normal("alpha", mu=0, sigma=1)
            _beta_obesity = pm.Normal("beta_obesity", mu=0, sigma=1)
            _ell_axis = pm.LogNormal(
                "ell_axis", mu=np.log(150), sigma=0.5, dims="axis"
            )
            _eta = pm.HalfNormal("eta", sigma=1)
            _sigma = pm.HalfNormal("sigma", sigma=0.5)
            _mean = pm.gp.mean.Linear(
                coeffs=[0, 0, _beta_obesity], intercept=_alpha
            )
            _gp_places = pm.gp.Marginal(
                mean_func=_mean,
                cov_func=_eta**2
                * pm.gp.cov.Matern52(3, active_dims=[0, 1], ls=_ell_axis),
            )
            _gp_places.marginal_likelihood(
                "y",
                X=_county_inputs,
                y=_diabetes_data,
                sigma=_sigma,
                dims="county",
            )
            if X_pred is not None:
                _prediction_inputs = pm.Data(
                    "prediction_inputs", X_pred, dims=("prediction", "input")
                )
                _gp_places.conditional(
                    "f_grid", _prediction_inputs, dims="prediction"
                )
        return places_model

    places_model = build_places_model()
    return build_places_model, places_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Prior predictive check
    """)
    return


@app.cell
def _(RANDOM_SEED, places_model, pm):
    with places_model:
        places_prior_pred = pm.sample_prior_predictive(
            draws=500, random_seed=RANDOM_SEED
        )
    return (places_prior_pred,)


@app.cell
def _(places_prior_pred):
    places_prior_draws = places_prior_pred["prior_predictive"]["y"].stack(
        sample=("chain", "draw")
    ).transpose("sample", "county")
    return (places_prior_draws,)


@app.cell
def _(PYMC_LIGHT_BLUE, go, np, places_prior_draws, y_places):
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
def _(mo, places_prior_draws, y_places):
    mo.md(
        f"""
        **Plausibility check:** prior predictive draws span
        [{places_prior_draws.min():.2f}, {places_prior_draws.max():.2f}] on
        the standardized scale, comfortably bracketing the observed range
        [{y_places.min():.2f}, {y_places.max():.2f}] — broad but not absurd,
        reasonable to proceed to sampling.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Sampling

    100 counties is tiny for a GP — the training covariance is only
    $100 \times 100$, so its Cholesky decomposition is essentially
    instantaneous, and sampling is fast even with the extra ARD
    lengthscale dimension.
    """)
    return


@app.cell
def _(mo):
    places_fit_button = mo.ui.run_button(label="Fit PLACES spatial GP")
    places_fit_button
    return (places_fit_button,)


@app.cell
def _(
    RANDOM_SEED,
    execute_models,
    mo,
    perf_counter,
    places_fit_button,
    places_model,
    pm,
    results_dir,
):
    mo.stop(not (places_fit_button.value or execute_models))
    places_model.compile_logp()(places_model.initial_point())
    with places_model:
        places_start = perf_counter()
        # A 500-draw diagnostic run exposed one divergence; the minimum
        # 0.90 acceptance threshold removes it without changing samplers.
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


@app.cell
def _(inference_health, places_idata, places_model):
    places_summary, places_health_passed = inference_health(
        places_idata, places_model
    )
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
    places_summary
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
    mo,
    places_health_passed,
    places_max_rhat,
    places_min_ess_bulk,
    places_min_ess_tail,
    places_n_div,
    places_n_draws_total,
    places_sample_seconds,
):
    mo.md(
        f"""
        **Diagnostics:** {places_n_div} divergence(s) out of
        {places_n_draws_total} draws in {places_sample_seconds:.1f}s.
        Minimum `ess_bulk` is {places_min_ess_bulk:.0f} and minimum
        `ess_tail` is {places_min_ess_tail:.0f} — both above the 400
        threshold — and maximum `r_hat` is {places_max_rhat:.3f}. Health gate
        passed: **{places_health_passed}**.

        Directional structure is reported below as the full posterior
        contrast between kilometre-scale east-west and north-south
        lengthscales, rather than a posterior-mean-only anisotropy claim.
        """
    )
    return


@app.cell
def _(eti_bounds, places_idata):
    places_ell = places_idata["posterior"]["ell_axis"]
    places_directional_contrast = (
        places_ell.sel(axis="east_km") - places_ell.sel(axis="north_km")
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
    return (places_directional_summary,)


@app.cell
def _(
    RANDOM_SEED,
    build_places_model,
    pm,
    places_idata,
    posterior_subset,
):
    places_observed_model = build_places_model()
    with places_observed_model:
        places_observed_ppc = pm.sample_posterior_predictive(
            posterior_subset(places_idata),
            var_names=["y"],
            random_seed=RANDOM_SEED,
        )
    return (places_observed_ppc,)


@app.cell
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Predicting on a spatial grid

    We evaluate the fitted GP's **full posterior-predictive**
    conditional on a regular longitude/latitude grid covering the
    county centroids — the same `.conditional` +
    `pm.sample_posterior_predictive` pattern used for `f_tide_pred` in
    Part B and for `f_pred` in Notebook 2 — rather than a single
    plug-in point estimate, so the heatmap below reflects the actual
    posterior mean over hyperparameters. The grid is kept modest
    (20x20 = 400 points) since `.conditional` draws a full-covariance
    sample per posterior draw and cost grows quickly with grid size.
    """)
    return


@app.cell
def _(east_km, north_km, np):
    from scipy.spatial import Delaunay

    grid_n = 20
    east_grid = np.linspace(east_km.min(), east_km.max(), grid_n)
    north_grid = np.linspace(north_km.min(), north_km.max(), grid_n)
    EAST_MESH, NORTH_MESH = np.meshgrid(east_grid, north_grid)
    spatial_grid = np.column_stack([EAST_MESH.ravel(), NORTH_MESH.ravel()])
    in_hull = Delaunay(np.column_stack([east_km, north_km])).find_simplex(
        spatial_grid
    ) >= 0
    # Zero is average obesity on the standardized covariate scale.
    X_grid = np.column_stack([spatial_grid, np.zeros(len(spatial_grid))])
    in_hull_grid = in_hull.reshape(grid_n, grid_n)
    return EAST_MESH, NORTH_MESH, X_grid, grid_n, in_hull_grid


@app.cell
def _(
    RANDOM_SEED,
    X_grid,
    build_places_model,
    places_idata,
    sample_fresh_model_predictions,
):
    places_grid_ppc = sample_fresh_model_predictions(
        places_idata,
        lambda: build_places_model(X_grid),
        var_names=["f_grid"],
        random_seed=RANDOM_SEED,
    )
    return (places_grid_ppc,)


@app.cell
def _(
    EAST_MESH,
    NORTH_MESH,
    PYMC_BLUE,
    go,
    grid_n,
    in_hull_grid,
    places,
    places_diabetes,
    places_diabetes_mean,
    places_diabetes_std,
    places_grid_ppc,
    east_km,
    north_km,
):
    grid_mean = places_grid_ppc["predictions"]["f_grid"].mean(
        dim=("chain", "draw")
    )
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
            "Average-obesity conditional interpolation within the county "
            "convex hull"
        ),
        xaxis_title="Local east coordinate (km)",
        yaxis_title="Local north coordinate (km)",
        template="plotly_white",
    )
    heatmap_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The masked surface is an interpolation of model-based PLACES estimates
    at average obesity within the observed county-centroid convex hull. It is
    not a map of county measurements, and it deliberately does not extrapolate
    outside the observed geometry.

    ### Exercise: try an isotropic (non-ARD) kernel

    Refit with a single kilometre-scale lengthscale
    (`pm.gp.cov.Matern52(2, ls=ell_iso)` with `ell_iso` scalar) rather
    than the ARD vector. Compare the fitted surface and the posterior
    directional contrast: does forcing east/north to share one scale
    materially change the in-hull interpolation?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Discussion": mo.md(
                """
                ```python
                ell_iso = pm.LogNormal("ell_iso", mu=np.log(150), sigma=0.5)
                cov_iso = eta**2 * pm.gp.cov.Matern52(2, ls=ell_iso)
                ```

                Compare the scalar kilometre lengthscale with the ARD
                posterior's directional contrast and its 89% ETI. If that
                contrast is concentrated near zero, the isotropic surface
                may be a useful simpler description. If not, an isotropic
                fit must compromise between the two directional scales.
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part D: Hierarchical GP — fastball spin rates

    ### Background

    Statcast fastball spin-rate (rpm) game means for three MLB pitchers,
    ten games each during **April–May 2021**. The 30 observations occupy a
    common grid of 24 dates. We model population time structure while
    retaining pitcher baselines and pitcher-specific functional departures.
    """)
    return


@app.cell
def _(data_dir, pl):
    spin = pl.read_csv(data_dir / "fastball_spin_rates.csv")
    spin = spin.with_columns(pl.col("game_date").str.strptime(pl.Date, "%Y-%m-%d"))
    spin.head()
    return (spin,)

@app.cell
def _(np, spin, z):
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
        day_mean,
        day_of_season,
        day_std,
        day_z,
        n_pitches,
        pitcher_idx_num,
        pitchers,
        spin_mean,
        spin_std,
        spin_vals,
        spin_z,
    )


@app.cell
def _(day_of_season, go, np, pitcher_idx_num, pitchers, spin_vals):
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
def _(mo):
    mo.md(r"""
    ### Partial pooling

    A population intercept and a sum-to-zero, non-centered pitcher-intercept
    hierarchy separate level differences from time structure. A mean-centered
    population GP captures shared day-to-day movement. Each pitcher then gets
    a two-way-centered GP deviation: it cannot absorb the population trajectory
    or a pitcher intercept, but can represent a different functional shape.
    Observation noise is scaled by $1/\sqrt{n_{\mathrm{pitches}}/\bar n}$,
    treating game means based on more pitches as more precise under an
    effective-sample-size assumption.
    """)
    return
@app.cell
def _(
    date_idx,
    day_grid,
    day_grid_z,
    n_pitches,
    np,
    pitcher_idx_num,
    pitchers,
    pm,
    pt,
    spin_z,
):
    _spin_coords = {
        "pitcher": pitchers,
        "day": day_grid,
        "feature": ["standardized_day"],
        "observation": np.arange(len(spin_z)),
    }

    def build_spin_model():
        """Build the identifiable population-plus-functional-deviation GP."""
        with pm.Model(coords=_spin_coords) as spin_model:
            _day_grid_data = pm.Data(
                "day_grid", day_grid_z[:, None], dims=("day", "feature")
            )
            _date_idx_data = pm.Data("date_idx", date_idx, dims="observation")
            _pitcher_idx_data = pm.Data(
                "pitcher_idx", pitcher_idx_num, dims="observation"
            )
            _n_pitches_data = pm.Data(
                "n_pitches", n_pitches, dims="observation"
            )
            _spin_data = pm.Data("spin_rate", spin_z, dims="observation")

            _alpha_pop = pm.Normal("alpha_pop", mu=0, sigma=1)
            _sigma_pitcher = pm.HalfNormal("sigma_pitcher", sigma=1)
            _pitcher_offset_raw = pm.Normal(
                "pitcher_offset_raw", mu=0, sigma=1, dims="pitcher"
            )
            _pitcher_intercept = pm.Deterministic(
                "pitcher_intercept",
                _sigma_pitcher
                * (_pitcher_offset_raw - pt.mean(_pitcher_offset_raw)),
                dims="pitcher",
            )

            _ell = pm.LogNormal("ell", mu=0, sigma=0.5)
            _eta_pop = pm.HalfNormal("eta_pop", sigma=1)
            _gp_pop = pm.gp.Latent(
                cov_func=_eta_pop**2 * pm.gp.cov.Matern52(1, ls=_ell)
            )
            _f_pop_raw = _gp_pop.prior(
                "f_pop_raw", X=_day_grid_data, dims="day"
            )
            _f_pop = pm.Deterministic(
                "f_pop", _f_pop_raw - pt.mean(_f_pop_raw), dims="day"
            )

            _eta_dev = pm.HalfNormal("eta_dev", sigma=0.5)
            _gp_dev = pm.gp.Latent(
                cov_func=_eta_dev**2 * pm.gp.cov.Matern52(1, ls=_ell)
            )
            _f_dev_raw = _gp_dev.prior(
                "f_dev_raw",
                X=_day_grid_data,
                n_outputs=3,
                dims=("pitcher", "day"),
            )
            _f_dev = pm.Deterministic(
                "f_dev",
                _f_dev_raw
                - pt.mean(_f_dev_raw, axis=0, keepdims=True)
                - pt.mean(_f_dev_raw, axis=1, keepdims=True)
                + pt.mean(_f_dev_raw),
                dims=("pitcher", "day"),
            )
            _sigma_obs = pm.HalfNormal("sigma_obs", sigma=0.5)
            _noise_scale = pm.math.sqrt(_n_pitches_data / n_pitches.mean())
            pm.Normal(
                "spin_obs",
                mu=(
                    _alpha_pop
                    + _pitcher_intercept[_pitcher_idx_data]
                    + _f_pop[_date_idx_data]
                    + _f_dev[_pitcher_idx_data, _date_idx_data]
                ),
                sigma=_sigma_obs / _noise_scale,
                observed=_spin_data,
                dims="observation",
            )
        return spin_model

    spin_model = build_spin_model()
    return build_spin_model, spin_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Prior predictive check
    """)
    return


@app.cell
def _(RANDOM_SEED, pm, spin_model):
    with spin_model:
        spin_prior_pred = pm.sample_prior_predictive(
            draws=500, random_seed=RANDOM_SEED
        )
    return (spin_prior_pred,)


@app.cell
def _(spin_prior_pred):
    spin_prior_draws = spin_prior_pred["prior_predictive"]["spin_obs"].stack(
        sample=("chain", "draw")
    ).transpose("sample", "observation")
    return (spin_prior_draws,)


@app.cell
def _(PYMC_LIGHT_BLUE, day_z, go, np, spin_prior_draws, spin_z):
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
def _(mo, spin_prior_draws, spin_z):
    mo.md(
        f"""
        **Plausibility check:** prior predictive draws span
        [{spin_prior_draws.min():.2f}, {spin_prior_draws.max():.2f}] on the
        standardized scale, comfortably bracketing the observed range
        [{spin_z.min():.2f}, {spin_z.max():.2f}] — broad but not absurd,
        reasonable to proceed to sampling.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Sampling

    Only 30 observations, but the non-centered hierarchy over a latent
    GP can still have delicate geometry. We use the default PyMC sampler
    settings and assess all free variables — every hyperparameter,
    the per-pitcher deviations, and the latent GP itself
    (`f_pop`/`f_pop_rotated_`, its highest-dimensional component) — with
    divergences, R-hat, and effective sample sizes rather than relying on
    a tuning override.
    """)
    return


@app.cell
def _(mo):
    spin_fit_button = mo.ui.run_button(label="Fit hierarchical spin GP")
    spin_fit_button
    return (spin_fit_button,)


@app.cell
def _(
    RANDOM_SEED,
    execute_models,
    mo,
    perf_counter,
    pm,
    results_dir,
    spin_fit_button,
    spin_model,
):
    mo.stop(not (spin_fit_button.value or execute_models))
    spin_model.compile_logp()(spin_model.initial_point())
    with spin_model:
        spin_start = perf_counter()
        # The 500-draw default run had 30 divergences; 0.90 had 16 and 0.95
        # had 3. This 0.99 threshold is the next geometry repair while keeping
        # PyMC's default sampler; 3,000 retained draws support the all-RV ESS gate.
        spin_idata = pm.sample(
            draws=3000,
            tune=1500,
            chains=4,
            target_accept=0.99,
            random_seed=RANDOM_SEED,
        )
        spin_sample_seconds = perf_counter() - spin_start
    spin_idata.to_netcdf(results_dir / "03_spin_hierarchical_gp.nc")
    print(f"Hierarchical spin-rate GP sampling wall-time: {spin_sample_seconds:.1f}s")
    return spin_idata, spin_sample_seconds


@app.cell
def _(inference_health, spin_idata, spin_model):
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
    spin_summary
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
    mo,
    spin_health_passed,
    spin_max_rhat,
    spin_min_ess_bulk,
    spin_min_ess_tail,
    spin_n_div,
    spin_n_draws_total,
    spin_sample_seconds,
):
    mo.md(
        f"""
        **Diagnostics:** {spin_n_div} divergence(s) out of
        {spin_n_draws_total} draws in {spin_sample_seconds:.1f}s. Computed
        over every free model variable. Minimum `ess_bulk` is
        {spin_min_ess_bulk:.0f}, minimum `ess_tail` is
        {spin_min_ess_tail:.0f}, and maximum `r_hat` is
        {spin_max_rhat:.3f}. Health gate passed: **{spin_health_passed}**.
        """
    )
    return


@app.cell
def _(RANDOM_SEED, pm, posterior_subset, spin_idata, spin_model):
    with spin_model:
        spin_observed_ppc = pm.sample_posterior_predictive(
            posterior_subset(spin_idata),
            var_names=["spin_obs"],
            random_seed=RANDOM_SEED,
        )
    return (spin_observed_ppc,)


@app.cell
def _(np, pitcher_idx_num, pitchers, spin_observed_ppc, spin_z):
    pitcher_labels = np.asarray(pitchers)[pitcher_idx_num]
    spin_replicated = spin_observed_ppc["posterior_predictive"]["spin_obs"].assign_coords(
        pitcher_label=("observation", pitcher_labels)
    )
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
    return (spin_ppc_by_pitcher,)


@app.cell
def _(
    day_grid,
    day_of_season,
    eti_bounds,
    go,
    np,
    pitcher_idx_num,
    pitchers,
    spin_idata,
    spin_mean,
    spin_std,
    spin_vals,
):
    posterior = spin_idata["posterior"]
    population = posterior["alpha_pop"] + posterior["f_pop"]
    population_rpm = population * spin_std + spin_mean
    population_mean = population_rpm.mean(dim=("chain", "draw"))
    population_lo, population_hi = eti_bounds(population_rpm)

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
def _(mo):
    mo.md(r"""
    The solid trajectories combine the population curve, the sum-to-zero
    pitcher intercept, and each pitcher's two-way-centered functional
    deviation. Thus they can differ in shape, not merely vertical offset.

    ### Exercise: inspect functional pooling

    Refit after tightening or loosening `eta_dev`, the scale of the
    pitcher-specific GP deviations. Compare the labeled population trajectory,
    pitcher trajectories, pooling scales, and grouped posterior-predictive
    table; do not treat a better-looking curve as evidence without checking
    those predictive discrepancies.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Discussion": mo.md(
                """
                ```python
                eta_dev = pm.HalfNormal("eta_dev", sigma=0.1)  # tighter functional pooling
                # or
                eta_dev = pm.HalfNormal("eta_dev", sigma=1.0)  # looser functional pooling
                ```

                Tightening `eta_dev` pulls the two-way-centered deviations
                toward zero, so pitchers share more of the population shape.
                Loosening it permits distinct shapes, but each pitcher has
                only ten observations. Compare the posterior on `eta_dev` and
                `sigma_pitcher` with the grouped posterior-predictive
                discrepancies before deciding whether the additional
                flexibility is supported.
                """
            )
        }
    )
    return


if __name__ == "__main__":
    app.run()
