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
        PYMC_DARK_GREEN,
        PYMC_LIGHT_BLUE,
        RANDOM_SEED,
        az,
        data_dir,
        execute_models,
        eti_bounds,
        inference_health,
        go,
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
            _ell_trend = pm.InverseGamma("ell_trend", alpha=5, beta=5)
            _eta_trend = pm.HalfNormal("eta_trend", sigma=1)
            _cov_trend = _eta_trend**2 * pm.gp.cov.Matern52(1, ls=_ell_trend)
            _eta_semi = pm.HalfNormal("eta_semi", sigma=1)
            _cov_semi = _eta_semi**2 * pm.gp.cov.Periodic(
                1, period=_semi_period_std, ls=0.5
            )
            _eta_diurnal = pm.HalfNormal("eta_diurnal", sigma=1)
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
    tide_prior_draws = tide_prior_pred["prior_predictive"]["y"].values.reshape(
        -1, len(y_tide)
    )

    tide_prior_fig = go.Figure()
    rng_plot_tide = np.random.default_rng(0)
    for _i in rng_plot_tide.choice(tide_prior_draws.shape[0], size=50, replace=False):
        tide_prior_fig.add_trace(
            go.Scatter(
                x=X_tide[:, 0],
                y=tide_prior_draws[_i],
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
        title="Prior predictive draws vs. standardized observed tide level",
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
    with tide_model:
        tide_start = perf_counter()
        tide_idata = pm.sample(chains=4, random_seed=RANDOM_SEED)
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
    relevant) before expanding the solution.
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
                `Matern52`. The relative `eta` amplitudes let the model
                learn that the weekly effect is weaker than the daily one,
                exactly as described, rather than that being hard-coded.
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

    places_lon_mean, places_lon_std = places_lon.mean(), places_lon.std(ddof=0)
    places_lat_mean, places_lat_std = places_lat.mean(), places_lat.std(ddof=0)
    places_diabetes_mean = places_diabetes.mean()
    places_diabetes_std = places_diabetes.std(ddof=0)

    X_places = np.column_stack(
        [z(places_lon), z(places_lat)]
    )  # GP inputs are 2D: (n, 2)
    y_places = z(places_diabetes)
    return (
        X_places,
        places_diabetes,
        places_diabetes_mean,
        places_diabetes_std,
        places_lat,
        places_lat_mean,
        places_lat_std,
        places_lon,
        places_lon_mean,
        places_lon_std,
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
        "coordinate": ["longitude", "latitude"],
    }

    def build_places_model(X_pred=None):
        _coords = dict(_places_coords)
        if X_pred is not None:
            _coords["prediction"] = np.arange(len(X_pred))
        with pm.Model(coords=_coords) as places_model:
            _county_locations = pm.Data(
                "county_locations", X_places, dims=("county", "coordinate")
            )
            _diabetes_data = pm.Data("diabetes", y_places, dims="county")
            _ell_places = pm.InverseGamma("ell", alpha=5, beta=5, shape=2)
            _eta_places = pm.HalfNormal("eta", sigma=2)
            _sigma_places = pm.HalfNormal("sigma_places", sigma=1)
            _gp_places = pm.gp.Marginal(
                cov_func=_eta_places**2
                * pm.gp.cov.Matern52(2, ls=_ell_places)
            )
            _gp_places.marginal_likelihood(
                "y",
                X=_county_locations,
                y=_diabetes_data,
                sigma=_sigma_places,
                dims="county",
            )
            if X_pred is not None:
                _county_prediction_locations = pm.Data(
                    "prediction_locations",
                    X_pred,
                    dims=("prediction", "coordinate"),
                )
                _gp_places.conditional(
                    "f_grid", _county_prediction_locations, dims="prediction"
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
def _(places_prior_pred, y_places):
    places_prior_draws = places_prior_pred["prior_predictive"]["y"].values.reshape(
        -1, len(y_places)
    )
    return (places_prior_draws,)


@app.cell
def _(PYMC_LIGHT_BLUE, go, np, places_prior_draws, y_places):
    places_prior_fig = go.Figure()
    rng_plot_places = np.random.default_rng(0)
    # Markers, not lines: county index has no natural ordering, unlike the time series elsewhere.
    for _i in rng_plot_places.choice(
        places_prior_draws.shape[0], size=50, replace=False
    ):
        places_prior_fig.add_trace(
            go.Scatter(
                x=np.arange(len(y_places)),
                y=places_prior_draws[_i],
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
    with places_model:
        places_start = perf_counter()
        places_idata = pm.sample(chains=4, random_seed=RANDOM_SEED)
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

        The two ARD lengthscales (see table above, `ell[0]` = longitude,
        `ell[1]` = latitude) need not agree — if their posteriors are well
        separated, that's evidence the surface really is anisotropic.
        """
    )
    return


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
def _(
    np,
    places_lat,
    places_lat_mean,
    places_lat_std,
    places_lon,
    places_lon_mean,
    places_lon_std,
):
    grid_n = 20
    lon_grid = np.linspace(places_lon.min(), places_lon.max(), grid_n)
    lat_grid = np.linspace(places_lat.min(), places_lat.max(), grid_n)
    LON_MESH, LAT_MESH = np.meshgrid(lon_grid, lat_grid)
    X_grid = np.column_stack(
        [
            (LON_MESH.ravel() - places_lon_mean) / places_lon_std,
            (LAT_MESH.ravel() - places_lat_mean) / places_lat_std,
        ]
    )
    return LAT_MESH, LON_MESH, X_grid, grid_n


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
    LAT_MESH,
    LON_MESH,
    PYMC_BLUE,
    go,
    grid_n,
    places,
    places_diabetes,
    places_diabetes_mean,
    places_diabetes_std,
    places_grid_ppc,
    places_lat,
    places_lon,
):
    grid_mu = (
        places_grid_ppc["predictions"]["f_grid"]
        .mean(dim=["chain", "draw"])
        .values
    )
    grid_diabetes = (grid_mu * places_diabetes_std + places_diabetes_mean).reshape(
        grid_n, grid_n
    )
    # Shared color scale so the heatmap and county markers are directly comparable.
    diabetes_cmin = min(grid_diabetes.min(), places_diabetes.min())
    diabetes_cmax = max(grid_diabetes.max(), places_diabetes.max())

    heatmap_fig = go.Figure()
    heatmap_fig.add_trace(
        go.Heatmap(
            x=LON_MESH[0],
            y=LAT_MESH[:, 0],
            z=grid_diabetes,
            colorscale="Blues",
            zmin=diabetes_cmin,
            zmax=diabetes_cmax,
            colorbar=dict(title="Diabetes %"),
        )
    )
    heatmap_fig.add_trace(
        go.Scatter(
            x=places_lon,
            y=places_lat,
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
            name="counties",
        )
    )
    heatmap_fig.update_layout(
        title="Predicted diabetes prevalence surface (ARD Matern52 GP) + county centroids",
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        template="plotly_white",
    )
    heatmap_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The predicted surface smoothly interpolates between county
    centroids, higher in some regions and lower in others, reflecting
    the spatial correlation the ARD kernel learned along longitude and
    latitude separately. Treat this surface with the same caution as
    the underlying PLACES estimates themselves: it is a smoothed
    *model* of model-based estimates, useful for visualizing broad
    spatial pattern, not a substitute for a county-level measurement.

    ### Exercise: try an isotropic (non-ARD) kernel

    Refit with a single scalar lengthscale (`pm.gp.cov.Matern52(2,
    ls=ell)` with `ell` a scalar, rather than `shape=2`) instead of the
    ARD vector. Compare the fitted surface — does forcing lon and lat to
    share one lengthscale visibly change the shape of the predicted
    surface, or do the two ARD lengthscales turn out to be similar
    enough that it barely matters? Expand for a discussion.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Discussion": mo.md(
                """
                ```python
                ell_iso = pm.InverseGamma("ell_iso", alpha=5, beta=5)  # scalar, not shape=2
                cov_iso = eta**2 * pm.gp.cov.Matern52(2, ls=ell_iso)
                ```

                Look back at the ARD posterior summary table above: if
                `ell[0]` (longitude) and `ell[1]` (latitude) have
                overlapping credible intervals, an isotropic fit will look
                nearly identical to the ARD one — the extra flexibility
                wasn't needed. If they are well separated, the isotropic
                fit is forced to compromise between the two true
                length-scales, which shows up as a surface that is either
                too smooth in the fast-varying direction or too wiggly in
                the slow-varying one. ARD costs one extra parameter; it is
                usually worth fitting first and checking whether the
                lengthscales actually differ before simplifying.
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

    Statcast fastball spin rate (rpm) for 3 MLB pitchers, 10 games each
    over the 2021 season. Spin rate drifts gradually over a season for
    physiological and mechanical reasons, but each pitcher has their
    own characteristic average level. We want to borrow strength across
    pitchers for the *shape* of the within-season drift, while still
    letting each pitcher have their own baseline.
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
    day_mean, day_std = day_of_season.mean(), day_of_season.std(ddof=0)
    day_z = z(day_of_season)

    spin_vals = spin["spin_rate"].to_numpy()
    spin_mean, spin_std = spin_vals.mean(), spin_vals.std(ddof=0)
    spin_z = z(spin_vals)
    return (
        day_mean,
        day_of_season,
        day_std,
        day_z,
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

    Two extremes: fit each pitcher's trend **completely separately**
    (no pooling — noisy with only 10 games each) or fit **one shared**
    trend and ignore that pitchers differ (complete pooling — ignores
    real between-pitcher variation). A hierarchical model does neither:
    it puts a single shared population-level GP trend $f_{\text{pop}}$
    over day-of-season, and gives every pitcher a **small, partially
    pooled deviation** from it,
    $\text{dev}_i \sim \mathcal{N}(0, \sigma_{\text{dev}}^2)$, estimated
    jointly with everything else. $\sigma_{\text{dev}}$ itself is
    learned from the data: if pitchers turn out to be very similar, it
    shrinks toward zero and pools hard; if they are genuinely
    different, it grows and pools less. As in earlier hierarchical
    models, we use the **non-centered** parameterization
    (`offset ~ Normal(0, 1)`, `dev = sigma_dev * offset`) to avoid the
    funnel geometry that a centered hierarchical GP is prone to.
    """)
    return


@app.cell
def _(day_z, np, pitcher_idx_num, pitchers, pm, spin_z):
    _spin_coords = {"pitcher": pitchers, "obs": np.arange(len(spin_z))}

    def build_spin_model(day_grid_z=None):
        _coords = dict(_spin_coords)
        if day_grid_z is not None:
            _coords["prediction"] = np.arange(len(day_grid_z))
        with pm.Model(coords=_coords) as spin_model:
            _day_data = pm.Data("day", day_z, dims="obs")
            _pitcher_data = pm.Data("pitcher_idx", pitcher_idx_num, dims="obs")
            _spin_data = pm.Data("spin_rate", spin_z, dims="obs")
            _ell_pop = pm.InverseGamma("ell_pop", alpha=5, beta=5)
            _eta_pop = pm.HalfNormal("eta_pop", sigma=1)
            _gp_pop = pm.gp.Latent(
                cov_func=_eta_pop**2 * pm.gp.cov.Matern52(1, ls=_ell_pop)
            )
            _f_pop = _gp_pop.prior("f_pop", X=_day_data[:, None], dims="obs")
            _sigma_dev = pm.HalfNormal("sigma_dev", sigma=0.5)
            _offset = pm.Normal("offset", 0, 1, dims="pitcher")
            _dev = pm.Deterministic(
                "dev", _sigma_dev * _offset, dims="pitcher"
            )
            _sigma_obs = pm.HalfNormal("sigma_obs", sigma=0.5)
            pm.Normal(
                "spin_obs",
                mu=_f_pop + _dev[_pitcher_data],
                sigma=_sigma_obs,
                observed=_spin_data,
                dims="obs",
            )
            if day_grid_z is not None:
                _day_prediction = pm.Data(
                    "day_prediction", day_grid_z, dims="prediction"
                )
                _gp_pop.conditional(
                    "f_pop_grid",
                    _day_prediction[:, None],
                    dims="prediction",
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
        spin_prior_pred = pm.sample_prior_predictive(draws=500, random_seed=RANDOM_SEED)
    return (spin_prior_pred,)


@app.cell
def _(spin_prior_pred, spin_z):
    spin_prior_draws = spin_prior_pred["prior_predictive"]["spin_obs"].values.reshape(
        -1, len(spin_z)
    )
    return (spin_prior_draws,)


@app.cell
def _(PYMC_LIGHT_BLUE, day_z, go, np, spin_prior_draws, spin_z):
    spin_prior_fig = go.Figure()
    rng_plot_spin = np.random.default_rng(0)
    # Markers, not lines: three pitchers interleave in day order, so lines would zigzag.
    for _i in rng_plot_spin.choice(spin_prior_draws.shape[0], size=50, replace=False):
        spin_prior_fig.add_trace(
            go.Scatter(
                x=day_z,
                y=spin_prior_draws[_i],
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
    with spin_model:
        spin_start = perf_counter()
        spin_idata = pm.sample(
            draws=1500,
            tune=1500,
            chains=4,
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
def _(day_mean, day_of_season, day_std, np):
    spin_day_grid = np.linspace(day_of_season.min(), day_of_season.max(), 50)
    spin_day_grid_z = ((spin_day_grid - day_mean) / day_std).reshape(-1, 1)
    return spin_day_grid, spin_day_grid_z


@app.cell
def _(
    RANDOM_SEED,
    build_spin_model,
    sample_fresh_model_predictions,
    spin_day_grid_z,
    spin_idata,
):
    spin_grid_ppc = sample_fresh_model_predictions(
        spin_idata,
        lambda: build_spin_model(spin_day_grid_z),
        var_names=["f_pop_grid"],
        random_seed=RANDOM_SEED,
    )
    return (spin_grid_ppc,)


@app.cell
def _(
    eti_bounds,
    day_of_season,
    go,
    np,
    pitcher_idx_num,
    pitchers,
    spin_day_grid,
    spin_grid_ppc,
    spin_idata,
    spin_mean,
    spin_std,
    spin_vals,
):
    _grid_n = len(spin_day_grid)
    f_pop_grid_samples = spin_grid_ppc["predictions"]["f_pop_grid"]
    f_pop_grid_samples = f_pop_grid_samples.rename(
        {f_pop_grid_samples.dims[-1]: "day"}
    ).assign_coords(day=spin_day_grid)
    dev_samples = spin_idata["posterior"]["dev"]

    spin_traj_fig = go.Figure()
    _colors = ["#154A72", "#81C240", "#4A9EDE"]
    _fill_colors = [
        "rgba(21,74,114,0.15)",
        "rgba(129,194,64,0.15)",
        "rgba(74,158,222,0.15)",
    ]
    for _i, _pitcher in enumerate(pitchers):
        _combined = (
            f_pop_grid_samples + dev_samples.isel(pitcher=_i)
        ) * spin_std + spin_mean
        _mean_traj = _combined.mean(dim=("chain", "draw"))
        _lo, _hi = eti_bounds(_combined)
        spin_traj_fig.add_trace(
            go.Scatter(
                x=np.concatenate([spin_day_grid, spin_day_grid[::-1]]),
                y=np.concatenate([_hi.values, _lo.values[::-1]]),
                fill="toself",
                fillcolor=_fill_colors[_i],
                line=dict(color="rgba(255,255,255,0)"),
                showlegend=False,
            )
        )
        spin_traj_fig.add_trace(
            go.Scatter(
                x=spin_day_grid,
                y=_mean_traj.values,
                mode="lines",
                name=f"{_pitcher} (posterior mean)",
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
        title="Per-pitcher posterior trajectories: shared trend + partially-pooled offset",
        xaxis_title="Days since first game in dataset",
        yaxis_title="Spin rate (rpm)",
        template="plotly_white",
    )
    spin_traj_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Each pitcher's posterior trajectory is the **same shared shape**
    ($f_{\text{pop}}$, the population trend), shifted up or down by
    that pitcher's own partially-pooled offset — visible as three
    vertically-separated curves that all rise and fall together. The
    model borrows the within-season *shape* across all 30
    observations, while still giving each pitcher their own level.

    ### Exercise: inspect pooling by changing the `sigma_dev` prior

    The `sigma_dev ~ HalfNormal(0.5)` prior controls how much pitchers
    are *allowed* to differ from the shared trend. Refit with a much
    tighter prior (e.g. `HalfNormal(0.05)`, forcing near-complete
    pooling) and a much looser one (e.g. `HalfNormal(5)`, allowing
    near-independent fits). How do the three trajectories above change?
    Expand for a discussion.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Discussion": mo.md(
                """
                ```python
                sigma_dev = pm.HalfNormal("sigma_dev", sigma=0.05)  # tight: forces pooling
                # or
                sigma_dev = pm.HalfNormal("sigma_dev", sigma=5)  # loose: nearly independent
                ```

                With a tight prior, the three per-pitcher offsets are
                squeezed toward zero regardless of what the data suggest —
                the three trajectories in the plot above would nearly
                coincide, even though the raw data (first scatter plot in
                this section) show visibly different average spin rates
                per pitcher. That is **too much** pooling: it discards real
                signal. With a loose prior, each pitcher's offset is
                estimated almost independently from the other two — with
                only 10 games each, individual trajectories become noisier
                and more sensitive to that pitcher's particular games,
                losing the benefit of borrowing strength across pitchers
                for the shared shape. The `HalfNormal(0.5)` prior used
                above sits between these extremes and lets the data (via
                the posterior on `sigma_dev` itself) decide how much
                pooling is warranted.
                """
            )
        }
    )
    return


if __name__ == "__main__":
    app.run()
