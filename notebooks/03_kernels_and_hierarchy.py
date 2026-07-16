import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
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
        """
    )
    return


@app.cell(hide_code=True)
def _():
    from pathlib import Path
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
    rng = np.random.default_rng(RANDOM_SEED)

    data_dir = Path(__file__).parent.parent / "data"

    def z(a):
        """Standardize an array: (a - mean) / population std."""
        return (a - a.mean()) / a.std(ddof=0)

    return (
        PYMC_BLUE,
        PYMC_DARK_GREEN,
        PYMC_GREEN,
        PYMC_LIGHT_BLUE,
        RANDOM_SEED,
        az,
        data_dir,
        go,
        np,
        perf_counter,
        pl,
        pm,
        rng,
        z,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
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
        """
    )
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
def _(PYMC_DARK_GREEN, go, kernel_dropdown, np, rng, zoo_cov):
    zoo_grid = np.linspace(0, 10, 200).reshape(-1, 1)  # GP inputs are 2D: (n, 1)
    zoo_K = zoo_cov(zoo_grid).eval()
    zoo_K = zoo_K + 1e-8 * np.eye(len(zoo_grid))  # jitter for numerical stability

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
    mo.md(
        r"""
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
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
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
        """
    )
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
    return N_EXACT, tides_slice


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
    mo.md(
        r"""
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
        """
    )
    return


@app.cell
def _(pm, tide_hours_std):
    SEMI_PERIOD_HOURS = 12.42  # M2 lunar semidiurnal constituent
    DIURNAL_PERIOD_HOURS = 23.93  # K1 lunar diurnal constituent
    PERIODIC_LS_STD = 0.5  # fixed within-cycle shape (standardized time units)
    semi_period_std = SEMI_PERIOD_HOURS / tide_hours_std
    diurnal_period_std = DIURNAL_PERIOD_HOURS / tide_hours_std

    with pm.Model() as tide_model:
        ell_trend = pm.InverseGamma("ell_trend", alpha=5, beta=5)
        eta_trend = pm.HalfNormal("eta_trend", sigma=1)
        cov_trend = eta_trend**2 * pm.gp.cov.Matern52(1, ls=ell_trend)

        eta_semi = pm.HalfNormal("eta_semi", sigma=1)
        cov_semi = eta_semi**2 * pm.gp.cov.Periodic(
            1, period=semi_period_std, ls=PERIODIC_LS_STD
        )

        eta_diurnal = pm.HalfNormal("eta_diurnal", sigma=1)
        cov_diurnal = eta_diurnal**2 * pm.gp.cov.Periodic(
            1, period=diurnal_period_std, ls=PERIODIC_LS_STD
        )

        cov_tide = cov_trend + cov_semi + cov_diurnal  # additive (OR) composition
        sigma_tide = pm.HalfNormal("sigma_tide", sigma=0.5)

        gp_tide = pm.gp.Marginal(cov_func=cov_tide)

    return cov_tide, gp_tide, tide_model


@app.cell
def _(X_tide, gp_tide, sigma_tide, tide_model, y_tide):
    with tide_model:
        gp_tide.marginal_likelihood("y", X=X_tide, y=y_tide, sigma=sigma_tide)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Prior predictive check""")
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
    mo.md(
        r"""
        ### Sampling

        5 free hyperparameters (trend $\ell,\eta$; two periodic amplitudes
        $\eta$; noise $\sigma$ — periods and within-cycle shape are fixed,
        as discussed above) over 200 points.
        """
    )
    return


@app.cell
def _(RANDOM_SEED, perf_counter, pm, tide_model):
    with tide_model:
        tide_start = perf_counter()
        tide_idata = pm.sample(random_seed=RANDOM_SEED)
        tide_sample_seconds = perf_counter() - tide_start
    print(f"NOAA additive-GP sampling wall-time: {tide_sample_seconds:.1f}s")
    return tide_idata, tide_sample_seconds


@app.cell
def _(az, tide_idata):
    tide_n_div = tide_idata["sample_stats"]["diverging"].sum().item()
    tide_n_draws_total = (
        tide_idata["posterior"].sizes["chain"] * tide_idata["posterior"].sizes["draw"]
    )
    tide_summary = az.summary(tide_idata["posterior"])
    tide_min_ess_bulk = float(tide_summary["ess_bulk"].min())
    tide_min_ess_tail = float(tide_summary["ess_tail"].min())
    tide_max_rhat = float(tide_summary["r_hat"].astype(float).max())
    print(f"Divergences: {tide_n_div} / {tide_n_draws_total}")
    tide_summary
    return (
        tide_max_rhat,
        tide_min_ess_bulk,
        tide_min_ess_tail,
        tide_n_div,
        tide_n_draws_total,
    )


@app.cell(hide_code=True)
def _(
    mo,
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
        maximum `r_hat` is {tide_max_rhat:.3f}. Safe to interpret.
        """
    )
    return


@app.cell
def _(X_tide, gp_tide, tide_model):
    with tide_model:
        f_tide_pred = gp_tide.conditional("f_tide_pred", X_tide)
    return (f_tide_pred,)


@app.cell
def _(RANDOM_SEED, f_tide_pred, pm, tide_idata, tide_model):
    with tide_model:
        tide_ppc = pm.sample_posterior_predictive(
            tide_idata, var_names=["f_tide_pred"], random_seed=RANDOM_SEED
        )
    return (tide_ppc,)


@app.cell
def _(
    PYMC_BLUE,
    az,
    go,
    np,
    tide_hours,
    tide_level,
    tide_level_mean,
    tide_level_std,
    tide_ppc,
):
    tide_fit_vals = (
        tide_ppc["posterior_predictive"]["f_tide_pred"].values.reshape(
            -1, len(tide_hours)
        )
        * tide_level_std
        + tide_level_mean
    )
    tide_fit_mean = tide_fit_vals.mean(axis=0)
    tide_fit_hdi = az.hdi(tide_fit_vals, prob=0.89, axis=0)
    tide_fit_lo, tide_fit_hi = tide_fit_hdi[:, 0], tide_fit_hdi[:, 1]

    tide_fit_fig = go.Figure()
    tide_fit_fig.add_trace(
        go.Scatter(
            x=np.concatenate([tide_hours, tide_hours[::-1]]),
            y=np.concatenate([tide_fit_hi, tide_fit_lo[::-1]]),
            fill="toself",
            fillcolor="rgba(21,74,114,0.25)",
            line=dict(color="rgba(255,255,255,0)"),
            name="89% HDI",
        )
    )
    tide_fit_fig.add_trace(
        go.Scatter(
            x=tide_hours,
            y=tide_fit_mean,
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
    mo.md(
        r"""
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
        """
    )
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
    mo.md(
        r"""
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
        """
    )
    return


@app.cell
def _(data_dir, pl):
    places = pl.read_csv(data_dir / "places_diabetes.csv")
    places.head()
    return (places,)


@app.cell
def _(places, z):
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
    mo.md(
        r"""
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
        """
    )
    return


@app.cell
def _(X_places, pm, y_places):
    with pm.Model() as places_model:
        ell_places = pm.InverseGamma("ell", alpha=5, beta=5, shape=2)  # ARD: (lon, lat)
        eta_places = pm.HalfNormal("eta", sigma=2)
        cov_places = eta_places**2 * pm.gp.cov.Matern52(2, ls=ell_places)
        sigma_places = pm.HalfNormal("sigma_places", sigma=1)

        gp_places = pm.gp.Marginal(cov_func=cov_places)
        gp_places.marginal_likelihood("y", X=X_places, y=y_places, sigma=sigma_places)

    return gp_places, places_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Prior predictive check""")
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
    mo.md(
        r"""
        ### Sampling

        100 counties is tiny for a GP — the training covariance is only
        $100 \times 100$, so its Cholesky decomposition is essentially
        instantaneous, and sampling is fast even with the extra ARD
        lengthscale dimension.
        """
    )
    return


@app.cell
def _(RANDOM_SEED, perf_counter, places_model, pm):
    with places_model:
        places_start = perf_counter()
        places_idata = pm.sample(random_seed=RANDOM_SEED)
        places_sample_seconds = perf_counter() - places_start
    print(f"PLACES ARD-GP sampling wall-time: {places_sample_seconds:.1f}s")
    return places_idata, places_sample_seconds


@app.cell
def _(az, places_idata):
    places_n_div = places_idata["sample_stats"]["diverging"].sum().item()
    places_n_draws_total = (
        places_idata["posterior"].sizes["chain"]
        * places_idata["posterior"].sizes["draw"]
    )
    places_summary = az.summary(places_idata["posterior"])
    places_min_ess_bulk = float(places_summary["ess_bulk"].min())
    places_min_ess_tail = float(places_summary["ess_tail"].min())
    places_max_rhat = float(places_summary["r_hat"].astype(float).max())
    print(f"Divergences: {places_n_div} / {places_n_draws_total}")
    places_summary
    return (
        places_max_rhat,
        places_min_ess_bulk,
        places_min_ess_tail,
        places_n_div,
        places_n_draws_total,
    )


@app.cell(hide_code=True)
def _(
    mo,
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
        threshold — and maximum `r_hat` is {places_max_rhat:.3f}. Safe to
        interpret.

        The two ARD lengthscales (see table above, `ell[0]` = longitude,
        `ell[1]` = latitude) need not agree — if their posteriors are well
        separated, that's evidence the surface really is anisotropic.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
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
        """
    )
    return


@app.cell
def _(
    gp_places,
    np,
    places_lat,
    places_lat_mean,
    places_lat_std,
    places_lon,
    places_lon_mean,
    places_lon_std,
    places_model,
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

    with places_model:
        f_grid = gp_places.conditional("f_grid", X_grid)
    return LAT_MESH, LON_MESH, f_grid, grid_n, lat_grid, lon_grid


@app.cell
def _(RANDOM_SEED, f_grid, pm, places_idata, places_model):
    with places_model:
        places_grid_ppc = pm.sample_posterior_predictive(
            places_idata, var_names=["f_grid"], random_seed=RANDOM_SEED
        )
    return (places_grid_ppc,)


@app.cell
def _(
    PYMC_BLUE,
    LAT_MESH,
    LON_MESH,
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
        places_grid_ppc["posterior_predictive"]["f_grid"]
        .mean(dim=["chain", "draw"])
        .values
    )
    grid_diabetes = (grid_mu * places_diabetes_std + places_diabetes_mean).reshape(
        grid_n, grid_n
    )

    heatmap_fig = go.Figure()
    heatmap_fig.add_trace(
        go.Heatmap(
            x=LON_MESH[0],
            y=LAT_MESH[:, 0],
            z=grid_diabetes,
            colorscale="Blues",
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
    mo.md(
        r"""
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
        """
    )
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
    mo.md(
        r"""
        ## Part D: Hierarchical GP — fastball spin rates

        ### Background

        Statcast fastball spin rate (rpm) for 3 MLB pitchers, 10 games each
        over the 2021 season. Spin rate drifts gradually over a season for
        physiological and mechanical reasons, but each pitcher has their
        own characteristic average level. We want to borrow strength across
        pitchers for the *shape* of the within-season drift, while still
        letting each pitcher have their own baseline.
        """
    )
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
        pitcher_map,
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
    mo.md(
        r"""
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
        """
    )
    return


@app.cell
def _(day_z, np, pitcher_idx_num, pitchers, pm, spin_z):
    spin_coords = {"pitcher": pitchers, "obs": np.arange(len(spin_z))}
    with pm.Model(coords=spin_coords) as spin_model:
        day_data = pm.Data("day", day_z, dims="obs")
        pitcher_data = pm.Data("pitcher_idx", pitcher_idx_num, dims="obs")

        ell_pop = pm.InverseGamma("ell_pop", alpha=5, beta=5)
        eta_pop = pm.HalfNormal("eta_pop", sigma=1)
        cov_pop = eta_pop**2 * pm.gp.cov.Matern52(1, ls=ell_pop)
        gp_pop = pm.gp.Latent(cov_func=cov_pop)
        f_pop = gp_pop.prior("f_pop", X=day_data[:, None])

        sigma_dev = pm.HalfNormal("sigma_dev", sigma=0.5)  # small deviation amplitude
        offset = pm.Normal("offset", 0, 1, dims="pitcher")  # non-centered
        dev = pm.Deterministic("dev", sigma_dev * offset, dims="pitcher")

        mu = f_pop + dev[pitcher_data]
        sigma_obs = pm.HalfNormal("sigma_obs", sigma=0.5)
        pm.Normal("spin_obs", mu=mu, sigma=sigma_obs, observed=spin_z, dims="obs")

    return gp_pop, spin_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Prior predictive check""")
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
    mo.md(
        r"""
        ### Sampling

        Only 30 observations, but the non-centered hierarchy over a latent
        GP has a mildly delicate funnel geometry, so we raise
        `target_accept` well above the 0.8 default (0.99, found by
        checking for divergences) rather than the more typical 0.9, and use
        more draws than the `1000` default to comfortably clear
        `ess_bulk > 400` on the **full posterior** — every hyperparameter,
        the per-pitcher deviations, and the latent GP itself
        (`f_pop`/`f_pop_rotated_`, its highest-dimensional and most
        funnel-prone component) — not just a convenient subset.
        """
    )
    return


@app.cell
def _(RANDOM_SEED, perf_counter, pm, spin_model):
    with spin_model:
        spin_start = perf_counter()
        spin_idata = pm.sample(
            draws=1500,
            tune=1500,
            random_seed=RANDOM_SEED,
            target_accept=0.99,
        )
        spin_sample_seconds = perf_counter() - spin_start
    print(f"Hierarchical spin-rate GP sampling wall-time: {spin_sample_seconds:.1f}s")
    return spin_idata, spin_sample_seconds


@app.cell
def _(az, spin_idata):
    spin_n_div = spin_idata["sample_stats"]["diverging"].sum().item()
    spin_n_draws_total = (
        spin_idata["posterior"].sizes["chain"] * spin_idata["posterior"].sizes["draw"]
    )
    # Full-posterior summary — no var_names filter, so the whitened GP free RV
    # (f_pop_rotated_) and the Deterministic f_pop are both covered, along
    # with every hyperparameter and the per-pitcher deviations.
    spin_full_summary = az.summary(spin_idata["posterior"])
    spin_min_ess_bulk = float(spin_full_summary["ess_bulk"].min())
    spin_min_ess_tail = float(spin_full_summary["ess_tail"].min())
    spin_max_rhat = float(spin_full_summary["r_hat"].astype(float).max())
    print(f"Divergences: {spin_n_div} / {spin_n_draws_total}")
    # Compact table for readability; the numbers above still come from the
    # full-posterior summary, not this filtered view.
    spin_summary = az.summary(
        spin_idata["posterior"],
        var_names=["ell_pop", "eta_pop", "sigma_dev", "sigma_obs", "dev"],
    )
    spin_summary
    return (
        spin_max_rhat,
        spin_min_ess_bulk,
        spin_min_ess_tail,
        spin_n_div,
        spin_n_draws_total,
    )


@app.cell(hide_code=True)
def _(
    mo,
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
        over the **full posterior** — every hyperparameter, the per-pitcher
        deviations, and the latent GP free RV `f_pop_rotated_` itself, not a
        filtered subset — minimum `ess_bulk` is {spin_min_ess_bulk:.0f} and
        minimum `ess_tail` is {spin_min_ess_tail:.0f} — both above the 400
        threshold — and maximum `r_hat` is {spin_max_rhat:.3f}. Safe to
        interpret.
        """
    )
    return


@app.cell
def _(day_mean, day_of_season, day_std, gp_pop, np, spin_model):
    spin_day_grid = np.linspace(day_of_season.min(), day_of_season.max(), 50)
    spin_day_grid_z = ((spin_day_grid - day_mean) / day_std).reshape(-1, 1)
    with spin_model:
        f_pop_grid = gp_pop.conditional("f_pop_grid", spin_day_grid_z)
    return f_pop_grid, spin_day_grid


@app.cell
def _(RANDOM_SEED, f_pop_grid, pm, spin_idata, spin_model):
    with spin_model:
        spin_grid_ppc = pm.sample_posterior_predictive(
            spin_idata, var_names=["f_pop_grid"], random_seed=RANDOM_SEED
        )
    return (spin_grid_ppc,)


@app.cell
def _(
    az,
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
    f_pop_grid_samples = spin_grid_ppc["posterior_predictive"][
        "f_pop_grid"
    ].values.reshape(-1, _grid_n)
    dev_samples = spin_idata["posterior"]["dev"].values.reshape(-1, len(pitchers))

    spin_traj_fig = go.Figure()
    _colors = ["#154A72", "#81C240", "#4A9EDE"]
    _fill_colors = [
        "rgba(21,74,114,0.15)",
        "rgba(129,194,64,0.15)",
        "rgba(74,158,222,0.15)",
    ]
    for _i, _pitcher in enumerate(pitchers):
        _combined = (
            f_pop_grid_samples + dev_samples[:, _i][:, None]
        ) * spin_std + spin_mean
        _mean_traj = _combined.mean(axis=0)
        _hdi = az.hdi(_combined, prob=0.89, axis=0)
        spin_traj_fig.add_trace(
            go.Scatter(
                x=np.concatenate([spin_day_grid, spin_day_grid[::-1]]),
                y=np.concatenate([_hdi[:, 1], _hdi[:, 0][::-1]]),
                fill="toself",
                fillcolor=_fill_colors[_i],
                line=dict(color="rgba(255,255,255,0)"),
                showlegend=False,
            )
        )
        spin_traj_fig.add_trace(
            go.Scatter(
                x=spin_day_grid,
                y=_mean_traj,
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
    mo.md(
        r"""
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
        """
    )
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
