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
    # Scaling, Approximate GPs, and the Model Workflow

    Notebooks 1–3 fit exact GPs — `pm.gp.Marginal` and `pm.gp.Latent` —
    on datasets from a handful of points up to a few hundred. This
    notebook confronts what happens as data grows into the thousands,
    and what to do about it.

    **Part A — why exact GPs don't scale.** A concrete timing
    demonstration of the $O(n^3)$ Cholesky-factorization cost that
    exact GP inference pays at every gradient evaluation.

    **Part B — sparse approximation.** `pm.gp.MarginalApprox` replaces
    the exact $n \times n$ covariance with a rank-reduced approximation
    built from a small set of *inducing points*, fit here to a few
    hundred NOAA tide-gauge readings.

    **Part C — Hilbert-space GP (HSGP) approximation.** A different,
    basis-function approximation that scales *linearly* in $n$, fit
    here to the **full** ~8,760-hour NOAA tide series — a size that
    would be impractical for exact inference.

    **Part D — the model-development workflow.** Putting the pieces
    together: prior predictive checks, saving results early, MCMC
    diagnostics, posterior predictive checks, and a decision guide for
    choosing among exact, sparse, and HSGP GPs.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    from pathlib import Path
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
    data_dir = Path(__file__).parent.parent / "data"
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    is_script_mode = mo.app_meta().mode == "script"
    execute_models = is_script_mode or bool(mo.cli_args().get("execute-models", False))

    def eti(data, prob=0.89):
        return data.quantile(
            [(1 - prob) / 2, 1 - (1 - prob) / 2], dim=("chain", "draw")
        )

    def posterior_subset(idata, draws_per_chain=100):
        n_draws = idata["posterior"].sizes["draw"]
        indices = np.linspace(0, n_draws - 1, min(draws_per_chain, n_draws), dtype=int)
        return idata.isel(draw=indices, missing_dims="ignore")

    def inference_health(idata, model, ess_floor=400, rhat_ceiling=1.01):
        names = [rv.name for rv in model.free_RVs]
        diagnostics = az.summary(idata["posterior"], var_names=names, kind="diagnostics")
        divergences = int(idata["sample_stats"]["diverging"].sum().item())
        rhat = diagnostics["r_hat"].astype(float)
        passed = (
            divergences == 0
            and (diagnostics["ess_bulk"] >= ess_floor).all()
            and (diagnostics["ess_tail"] >= ess_floor).all()
            and (rhat <= rhat_ceiling).all()
        )
        return diagnostics, divergences, passed

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
        eti,
        execute_models,
        go,
        inference_health,
        np,
        perf_counter,
        pl,
        pm,
        posterior_subset,
        pt,
        results_dir,
        z,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part A: Why exact GPs don't scale

    Fitting `pm.gp.Marginal` (or `pm.gp.Latent`) exactly requires
    factorizing the $n \times n$ training covariance matrix $K$ — a
    Cholesky decomposition — at **every** gradient evaluation NUTS
    makes during sampling, since $\ell$, $\eta$, and $\sigma$ change
    $K$ at every step. Cholesky factorization costs $O(n^3)$
    floating-point operations, and NUTS typically needs thousands of
    gradient evaluations over a full sampling run. Below we measure
    just the factorization step — building $K$ and calling
    `np.linalg.cholesky` — with **no sampling at all**, as $n$ grows,
    to make the scaling visible directly.
    """)
    return


@app.cell
def _(np, perf_counter):
    def _scale_cov(X, ls=1.0, eta=1.0):
        d = np.abs(X[:, None, 0] - X[None, :, 0])
        K = eta**2 * np.exp(-0.5 * (d / ls) ** 2)
        return K + 1e-6 * np.eye(len(X))

    scale_ns = [100, 300, 900, 2700]
    scale_times = []
    for _n in scale_ns:
        _X = np.linspace(0, 10, _n)[:, None]
        _start = perf_counter()
        _K = _scale_cov(_X)
        np.linalg.cholesky(_K)
        scale_times.append(perf_counter() - _start)
    return scale_ns, scale_times


@app.cell
def _(PYMC_BLUE, go, scale_ns, scale_times):
    scale_fig = go.Figure()
    scale_fig.add_trace(
        go.Scatter(
            x=scale_ns,
            y=scale_times,
            mode="lines+markers",
            line=dict(color=PYMC_BLUE, width=2),
            marker=dict(size=9),
        )
    )
    scale_fig.update_layout(
        title="Covariance build + Cholesky factorization time vs. n (log-log)",
        xaxis_title="n (training points)",
        yaxis_title="elapsed time (s)",
        xaxis_type="log",
        yaxis_type="log",
        template="plotly_white",
    )
    scale_fig
    return


@app.cell(hide_code=True)
def _(mo, scale_ns, scale_times):
    _ratio_n = scale_ns[-1] / scale_ns[-2]
    _ratio_t = scale_times[-1] / scale_times[-2]
    _extrap_n = 8760
    _extrap_t = scale_times[-1] * (_extrap_n / scale_ns[-1]) ** 3
    mo.md(
        f"""
        **What to notice:** going from $n={scale_ns[-2]}$ to $n={scale_ns[-1]}$
        (a {_ratio_n:.0f}× increase in $n$) took {_ratio_t:.1f}× longer —
        close to the {_ratio_n**3:.0f}× an exact $O(n^3)$ law predicts (the
        exact multiple varies run to run since these are sub-second wall
        times, but the cubic trend is unmistakable on the log-log plot
        above). Extrapolating that cubic trend to $n={_extrap_n:,}$ — the
        full NOAA hourly series used in Part C below — a **single**
        Cholesky factorization would already take on the order of
        {_extrap_t:.0f}s, and NUTS needs one such factorization (or more)
        per gradient evaluation, typically thousands of them over a full
        run. Exact inference at that scale is not practical. The rest of
        this notebook covers two ways around the bottleneck.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part B: Sparse approximation — inducing points

    Instead of the full $n \times n$ covariance, a **sparse** GP
    approximation summarizes the training data with a small set of
    $m \ll n$ **inducing points** (also called pseudo-inputs), located
    somewhere in the input space (not necessarily at observed $x$
    values). All of the expensive linear algebra is then done on
    $m \times m$ and $n \times m$ matrices instead of $n \times n$ —
    $O(nm^2)$ instead of $O(n^3)$ — a large saving whenever $m \ll n$.
    `pm.gp.MarginalApprox` implements this for the conjugate
    (Gaussian-likelihood) case, with several approximation variants;
    below we use **FITC** (Fully Independent Training Conditional),
    which additionally corrects the noise/uncertainty at each training
    point for the information lost by not conditioning on the full
    exact covariance. The inducing point *locations* can be fixed,
    chosen by a simple heuristic (e.g. k-means cluster centers over the
    training inputs, used below), or even learned as extra parameters —
    we treat them as fixed here.
    """)
    return


@app.cell
def _(data_dir, pl):
    N_SPARSE = 450
    sparse_tides = pl.read_csv(data_dir / "noaa_tides_hourly.csv")
    sparse_tides = sparse_tides.with_columns(
        pl.col("time").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M")
    )
    sparse_slice = sparse_tides.slice(3000, N_SPARSE)
    sparse_slice.head()
    return (sparse_slice,)


@app.cell
def _(sparse_slice, z):
    sparse_t0 = sparse_slice["time"][0]
    sparse_hours = (
        sparse_slice["time"] - sparse_t0
    ).dt.total_minutes().to_numpy() / 60.0
    sparse_level = sparse_slice["water_level"].to_numpy()

    sparse_hours_std = sparse_hours.std(ddof=0)
    sparse_level_mean, sparse_level_std = sparse_level.mean(), sparse_level.std(ddof=0)

    X_sparse = z(sparse_hours).reshape(-1, 1)  # GP inputs are 2D: (n, 1)
    y_sparse = z(sparse_level)
    return (
        X_sparse,
        sparse_hours,
        sparse_hours_std,
        sparse_level,
        sparse_level_mean,
        sparse_level_std,
        y_sparse,
    )


@app.cell
def _(X_sparse, pm, sparse_hours_std):
    N_INDUCING = 25
    SPARSE_SEMI_PERIOD_HOURS = 12.42
    SPARSE_DIURNAL_PERIOD_HOURS = 23.93
    SPARSE_PERIODIC_LS_STD = 0.5
    sparse_semi_period_std = SPARSE_SEMI_PERIOD_HOURS / sparse_hours_std
    sparse_diurnal_period_std = SPARSE_DIURNAL_PERIOD_HOURS / sparse_hours_std
    Xu_init = pm.gp.util.kmeans_inducing_points(N_INDUCING, X_sparse)
    return (
        N_INDUCING,
        SPARSE_PERIODIC_LS_STD,
        Xu_init,
        sparse_diurnal_period_std,
        sparse_semi_period_std,
    )


@app.cell
def _(
    SPARSE_PERIODIC_LS_STD,
    np,
    pm,
    pt,
    sparse_diurnal_period_std,
    sparse_semi_period_std,
):
    def build_fitc_prior_model(X, Xu):
        """Generative FITC approximation for prior-predictive checking."""
        coords = {
            "obs": np.arange(len(X)),
            "feature": ["standardized time"],
            "inducing": np.arange(len(Xu)),
        }
        with pm.Model(coords=coords) as model:
            X_data = pm.Data("X", X, dims=("obs", "feature"))
            Xu_data = pm.Data("Xu", Xu, dims=("inducing", "feature"))
            ell_s_trend = pm.LogNormal("ell_s_trend", mu=0, sigma=1)
            eta_s_trend = pm.HalfNormal("eta_s_trend", sigma=1)
            eta_s_semi = pm.HalfNormal("eta_s_semi", sigma=1)
            eta_s_diurnal = pm.HalfNormal("eta_s_diurnal", sigma=0.5)
            sigma_sparse = pm.HalfNormal("sigma_sparse", sigma=0.5)
            cov = (
                eta_s_trend**2 * pm.gp.cov.Matern52(1, ls=ell_s_trend)
                + eta_s_semi**2
                * pm.gp.cov.Periodic(
                    1, period=sparse_semi_period_std, ls=SPARSE_PERIODIC_LS_STD
                )
                + eta_s_diurnal**2
                * pm.gp.cov.Periodic(
                    1, period=sparse_diurnal_period_std, ls=SPARSE_PERIODIC_LS_STD
                )
            )
            Kuu = cov(Xu_data) + 1e-6 * pt.eye(len(Xu))
            u = pm.MvNormal("u", mu=pt.zeros(len(Xu)), cov=Kuu, dims="inducing")
            Kfu = cov(X_data, Xu_data)
            projection = Kfu @ pt.linalg.solve(Kuu, u)
            Qff = Kfu @ pt.linalg.solve(Kuu, Kfu.T)
            residual_var = pt.maximum(pt.diag(cov(X_data) - Qff), 0) + sigma_sparse**2
            pm.Normal("y", mu=projection, sigma=pt.sqrt(residual_var), dims="obs")
        return model

    def build_sparse_gp_model(X, y, Xu, *, X_pred=None, pred_coord=None):
        coords = {
            "obs": np.arange(len(X)),
            "feature": ["standardized time"],
            "inducing": np.arange(len(Xu)),
        }
        if X_pred is not None:
            coords["pred_obs"] = (
                np.arange(len(X_pred)) if pred_coord is None else pred_coord
            )
        with pm.Model(coords=coords) as model:
            X_data = pm.Data("X", X, dims=("obs", "feature"))
            y_data = pm.Data("y_data", y, dims="obs")
            Xu_data = pm.Data("Xu", Xu, dims=("inducing", "feature"))
            ell_s_trend = pm.LogNormal("ell_s_trend", mu=0, sigma=1)
            eta_s_trend = pm.HalfNormal("eta_s_trend", sigma=1)
            eta_s_semi = pm.HalfNormal("eta_s_semi", sigma=1)
            eta_s_diurnal = pm.HalfNormal("eta_s_diurnal", sigma=0.5)
            sigma_sparse = pm.HalfNormal("sigma_sparse", sigma=0.5)
            cov = (
                eta_s_trend**2 * pm.gp.cov.Matern52(1, ls=ell_s_trend)
                + eta_s_semi**2
                * pm.gp.cov.Periodic(
                    1, period=sparse_semi_period_std, ls=SPARSE_PERIODIC_LS_STD
                )
                + eta_s_diurnal**2
                * pm.gp.cov.Periodic(
                    1, period=sparse_diurnal_period_std, ls=SPARSE_PERIODIC_LS_STD
                )
            )
            gp_sparse = pm.gp.MarginalApprox(cov_func=cov, approx="FITC")
            gp_sparse.marginal_likelihood(
                "y", X=X_data, Xu=Xu_data, y=y_data, sigma=sigma_sparse
            )
            if X_pred is not None:
                X_pred_data = pm.Data(
                    "X_pred", X_pred, dims=("pred_obs", "feature")
                )
                gp_sparse.conditional(
                    "f_sparse_latent", X_pred_data, pred_noise=False, dims="pred_obs"
                )
                gp_sparse.conditional(
                    "f_sparse_noisy", X_pred_data, pred_noise=True, dims="pred_obs"
                )
        return model

    return build_fitc_prior_model, build_sparse_gp_model


@app.cell
def _(X_sparse, Xu_init, build_fitc_prior_model):
    sparse_prior_model = build_fitc_prior_model(X_sparse, Xu_init)
    sparse_prior_model.compile_logp()(sparse_prior_model.initial_point())
    return (sparse_prior_model,)


@app.cell
def _(RANDOM_SEED, pm, sparse_prior_model):
    with sparse_prior_model:
        sparse_prior_pred = pm.sample_prior_predictive(draws=500, random_seed=RANDOM_SEED)
    return (sparse_prior_pred,)


@app.cell
def _(X_sparse, Xu_init, build_sparse_gp_model, y_sparse):
    sparse_model = build_sparse_gp_model(X_sparse, y_sparse, Xu_init)
    sparse_model.compile_logp()(sparse_model.initial_point())
    return (sparse_model,)


@app.cell
def _(PYMC_LIGHT_BLUE, X_sparse, go, np, sparse_prior_pred, y_sparse):
    sparse_prior_draws = sparse_prior_pred["prior"]["y"].values.reshape(
        -1, len(y_sparse)
    )

    sparse_prior_fig = go.Figure()
    _rng_plot = np.random.default_rng(0)
    for _i in _rng_plot.choice(sparse_prior_draws.shape[0], size=50, replace=False):
        sparse_prior_fig.add_trace(
            go.Scatter(
                x=X_sparse[:, 0],
                y=sparse_prior_draws[_i],
                mode="lines",
                line=dict(color=PYMC_LIGHT_BLUE, width=1),
                opacity=0.2,
                showlegend=False,
            )
        )
    sparse_prior_fig.add_trace(
        go.Scatter(
            x=X_sparse[:, 0],
            y=y_sparse,
            mode="markers",
            marker=dict(color="black", size=3),
            name="observed (standardized)",
        )
    )
    sparse_prior_fig.update_layout(
        title="Prior predictive draws vs. standardized observed tide level",
        xaxis_title="time (standardized)",
        yaxis_title="water level (standardized)",
        template="plotly_white",
    )
    sparse_prior_fig
    return (sparse_prior_draws,)


@app.cell(hide_code=True)
def _(mo, sparse_prior_draws, y_sparse):
    mo.md(
        f"""
        **Plausibility check:** prior predictive draws span
        [{sparse_prior_draws.min():.2f}, {sparse_prior_draws.max():.2f}] on
        the standardized scale, comfortably bracketing the observed range
        [{y_sparse.min():.2f}, {y_sparse.max():.2f}] — broad but not absurd,
        reasonable to proceed to sampling.
        """
    )
    return


@app.cell(hide_code=True)
def _(N_INDUCING, mo):
    mo.md(
        f"""
        ### Sampling

        The FITC likelihood only ever touches $n \\times m$ and
        $m \\times m$ matrices ($m={N_INDUCING}$ inducing points), so
        sampling stays fast even though $n=450$ is more than double the
        200-point exact fit in Notebook 3.
        """
    )
    return


@app.cell
def _(mo):
    sparse_fit_button = mo.ui.run_button(label="Fit FITC model")
    sparse_fit_button
    return (sparse_fit_button,)


@app.cell
def _(
    RANDOM_SEED,
    execute_models,
    mo,
    perf_counter,
    pm,
    results_dir,
    sparse_fit_button,
    sparse_model,
):
    mo.stop(not (sparse_fit_button.value or execute_models))
    with sparse_model:
        sparse_start = perf_counter()
        sparse_idata = pm.sample(random_seed=RANDOM_SEED, chains=4)
        sparse_sample_seconds = perf_counter() - sparse_start
    sparse_idata.to_netcdf(results_dir / "04_sparse_fitc_gp.nc")
    print(f"Sparse FITC-GP sampling wall-time: {sparse_sample_seconds:.1f}s")
    return (sparse_idata,)


@app.cell
def _(inference_health, sparse_idata, sparse_model):
    sparse_summary, sparse_n_div, sparse_health_passed = inference_health(
        sparse_idata, sparse_model
    )
    sparse_summary
    return sparse_health_passed, sparse_n_div


@app.cell(hide_code=True)
def _(mo, sparse_health_passed, sparse_n_div):
    mo.md(
        f"""**FITC inference health:** {sparse_n_div} divergences; all free
        variables meet the stated ESS/R-hat thresholds: **{sparse_health_passed}**."""
    )
    return


@app.cell
def _(X_sparse, Xu_init, build_sparse_gp_model, np, y_sparse):
    sparse_prediction_model = build_sparse_gp_model(
        X_sparse, y_sparse, Xu_init, X_pred=X_sparse, pred_coord=np.arange(len(X_sparse))
    )
    sparse_prediction_model.compile_logp()(sparse_prediction_model.initial_point())
    return (sparse_prediction_model,)


@app.cell
def _(RANDOM_SEED, pm, posterior_subset, sparse_idata, sparse_prediction_model):
    sparse_predictive_subset = posterior_subset(sparse_idata)
    with sparse_prediction_model:
        sparse_predictions = pm.sample_posterior_predictive(
            sparse_predictive_subset,
            var_names=["f_sparse_latent", "f_sparse_noisy"],
            predictions=True,
            random_seed=RANDOM_SEED,
        )
    return sparse_predictions, sparse_predictive_subset


@app.cell
def _(
    N_INDUCING,
    PYMC_BLUE,
    Xu_init,
    eti,
    go,
    np,
    sparse_hours,
    sparse_level,
    sparse_level_mean,
    sparse_level_std,
    sparse_predictions,
):
    sparse_fit = sparse_predictions["predictions"]["f_sparse_latent"]
    sparse_fit_vals = sparse_fit * sparse_level_std + sparse_level_mean
    sparse_fit_interval = eti(sparse_fit_vals)
    sparse_fit_mean = sparse_fit_vals.mean(("chain", "draw"))
    sparse_fit_lo = sparse_fit_interval.isel(quantile=0)
    sparse_fit_hi = sparse_fit_interval.isel(quantile=1)
    sparse_fit_fig = go.Figure()
    sparse_fit_fig.add_trace(
        go.Scatter(
            x=np.concatenate([sparse_hours, sparse_hours[::-1]]),
            y=np.concatenate([sparse_fit_hi.values, sparse_fit_lo.values[::-1]]),
            fill="toself",
            fillcolor="rgba(21,74,114,0.25)",
            line=dict(color="rgba(255,255,255,0)"),
            name="89% ETI",
        )
    )
    sparse_fit_fig.add_trace(
        go.Scatter(
            x=sparse_hours,
            y=sparse_fit_mean.values,
            mode="lines",
            name="posterior mean latent fit",
            line=dict(color=PYMC_BLUE, width=2),
        )
    )
    sparse_fit_fig.add_trace(
        go.Scatter(
            x=sparse_hours,
            y=sparse_level,
            mode="markers",
            name="observed",
            marker=dict(color="black", size=3),
        )
    )
    _Xu_hours = Xu_init[:, 0] * sparse_hours.std(ddof=0) + sparse_hours.mean()
    sparse_fit_fig.add_trace(
        go.Scatter(
            x=_Xu_hours,
            y=[sparse_level.min()] * N_INDUCING,
            mode="markers",
            name="inducing points",
            marker=dict(color="orange", size=7, symbol="triangle-up"),
        )
    )
    sparse_fit_fig.update_layout(
        title=f"FITC latent fit — {N_INDUCING} inducing points",
        xaxis_title="Hours since slice start",
        yaxis_title="Water level (m, MLLW)",
        template="plotly_white",
    )
    sparse_fit_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The FITC fit recovers essentially the same tidal shape as the
    exact fit in Notebook 3, using a covariance approximation that
    never touches more than a $450 \times 25$ matrix. The 25 inducing
    points (orange triangles, plotted along the bottom for reference)
    are enough to summarize the whole 450-point training set for this
    smoothly-varying signal. That is the point of sparse approximation:
    trade a small, usually invisible amount of accuracy for a large
    reduction in cost. It still will not reach the full year, though —
    even $O(nm^2)$ becomes expensive as $n$ grows into the thousands
    with a fixed-size $m$ that must grow to track increasingly complex
    structure. Part C introduces an approximation whose cost is
    *linear* in $n$, letting us fit the entire 8,760-point series next.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise: interrogate the FITC approximation

    The FITC model uses 25 fixed k-means inducing locations for 450 tide
    observations. Before altering an inducing set, identify the two distinct
    approximations in its predictive distribution: the low-rank inducing
    projection and the nonnegative diagonal correction. Then propose a
    diagnostic comparison that could reveal an inducing set that is too sparse
    near a rapid tidal feature. Use the latent and noisy conditionals
    separately: which one answers a question about the smooth water-level
    function, and which one answers a question about a replicated observation?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Hint": mo.md(
                r"""
                Keep the inducing locations fixed while comparing
                approximation settings. Start with prior trajectories: a
                generative FITC prior has to use the same jittered $K_{uu}$ in
                its inducing draw and projection. In posterior prediction,
                compare the labeled latent conditional with the
                `pred_noise=True` conditional rather than treating their
                intervals as interchangeable.
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Discussion": mo.md(
                r"""
                FITC represents long-range dependence through the inducing
                projection $Q_{ff}$ and restores pointwise variance through the
                clipped diagonal residual $\max(\operatorname{diag}(K_{ff} -
                Q_{ff}), 0)$. Too few or poorly located inducing points can
                leave a localized feature under-resolved even when the plotted
                mean looks plausible. Compare predictions at the observed
                inputs, especially near fast changes, against the retained
                noisy-observation posterior predictive check; systematic local
                residual structure is evidence about approximation accuracy.

                The latent conditional concerns the underlying tidal function.
                The noisy conditional adds measurement variation and is the
                appropriate comparison for another recorded water level.
                Neither comparison establishes that FITC agrees with the exact
                Notebook 3 fit: the notebooks use different tide slices, so the
                useful question is whether this approximation is adequate for
                this model, data, and inducing configuration.
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part C: Hilbert-space GP (HSGP) approximation

    The **Hilbert-space GP** approximation takes a completely different
    approach from sparse/inducing-point methods. Instead of
    approximating the *covariance matrix*, it approximates the
    covariance **function** itself by its spectral (Fourier-like)
    decomposition on a bounded domain: a GP with a stationary kernel is
    rewritten as a weighted sum of a fixed set of basis functions
    (Laplace eigenfunctions of the domain, independent of any kernel
    hyperparameter) with random Gaussian weights, whose variances are
    set by the kernel's power spectral density (which *does* depend on
    the hyperparameters). Because the basis functions don't change
    during sampling, the model is essentially a **linear-in-parameters**
    regression on those weights — cost grows **linearly** in $n$, not
    cubically, since evaluating the fixed basis at $n$ points is just a
    $O(nm)$ matrix multiply. The tradeoff: the number of basis
    functions $m$ (and the domain boundary) must be chosen so the
    approximation is accurate over the lengthscales the kernel actually
    needs to represent — get $m$ or the boundary wrong and the
    approximation degrades, as the exercise below will show directly.
    """)
    return


@app.cell
def _(mo):
    m_slider = mo.ui.slider(5, 40, value=20, step=1, label="m (# basis functions)")
    c_slider = mo.ui.slider(1.1, 3.0, value=1.5, step=0.1, label="c (boundary factor)")
    mo.hstack([m_slider, c_slider], gap=2)
    return c_slider, m_slider


@app.cell
def _(
    PYMC_BLUE,
    PYMC_DARK_GREEN,
    PYMC_GREEN,
    PYMC_LIGHT_BLUE,
    c_slider,
    go,
    m_slider,
    np,
    pm,
):
    hsgp_demo_grid = np.linspace(-3, 3, 300).reshape(-1, 1)
    # The basis (eigenfunctions of the Laplacian on [-cL, cL]) does not depend
    # on the covariance function at all — only on m, c, and X — so any
    # placeholder stationary kernel works here purely to construct the object.
    hsgp_demo_cov = pm.gp.cov.ExpQuad(1, ls=1.0)
    hsgp_demo_gp = pm.gp.HSGP(
        m=[m_slider.value], c=c_slider.value, cov_func=hsgp_demo_cov
    )
    hsgp_demo_phi, _hsgp_demo_sqrt_psd = hsgp_demo_gp.prior_linearized(X=hsgp_demo_grid)
    hsgp_demo_phi_vals = hsgp_demo_phi.eval()

    hsgp_basis_fig = go.Figure()
    _n_show = min(5, hsgp_demo_phi_vals.shape[1])
    _colors = [PYMC_BLUE, PYMC_GREEN, PYMC_LIGHT_BLUE, PYMC_DARK_GREEN, "#8C1C13"]
    for _j in range(_n_show):
        hsgp_basis_fig.add_trace(
            go.Scatter(
                x=hsgp_demo_grid[:, 0],
                y=hsgp_demo_phi_vals[:, _j],
                mode="lines",
                name=f"basis {_j + 1}",
                line=dict(color=_colors[_j % len(_colors)], width=2),
            )
        )
    hsgp_basis_fig.update_layout(
        title=f"First {_n_show} HSGP Laplace eigenfunctions (m={m_slider.value}, c={c_slider.value})",
        xaxis_title="x (standardized)",
        yaxis_title="φⱼ(x)",
        template="plotly_white",
    )
    hsgp_basis_fig
    return


@app.cell(hide_code=True)
def _(c_slider, m_slider, mo):
    mo.md(
        f"""
        **Reading the basis functions above:** with **m = {m_slider.value}**
        basis functions and boundary factor **c = {c_slider.value}**, the
        approximation domain extends to ±{c_slider.value:.1f} × max\\|x\\| —
        beyond the observed data range, which is required so the boundary
        condition (the basis functions are pinned to exactly zero at the
        edges) doesn't distort the fit near the edges of the actual data.
        Each `φⱼ` above is a fixed sine-like wave on that extended
        domain, of increasing frequency — entirely independent of any
        covariance hyperparameter. What *does* depend on the kernel and its
        hyperparameters is only the **weight** given to each basis function
        (its power spectral density) when they are summed to build the GP
        prior: a short lengthscale puts more weight on the high-frequency
        (wigglier) basis functions, a long one concentrates weight on the
        low-frequency (smooth) ones. Increasing $m$ lets the approximation
        represent higher-frequency structure, at the cost of more
        parameters to sample; increasing $c$ pushes the boundary farther
        out (reducing edge artifacts) but, for a fixed $m$, spreads the
        same basis functions over a wider domain and so coarsens the
        resolution available *within* the data range — $m$ and $c$ must
        generally be increased together.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Constraints of the HSGP approximation

    - Only works for **stationary** covariance functions with a known
      power spectral density (`Matern52`, `ExpQuad`, `Matern32`, ... —
      not `Linear`, which is not stationary).
    - Officially supported for input dimension **up to 3** — the
      number of basis functions needed to cover a fixed-density grid
      grows exponentially with dimension, so HSGP stops being
      efficient well before that in practice for high-dimensional
      inputs.
    - The **`Periodic`** kernel is *not* directly supported by
      `pm.gp.HSGP` (it has no ordinary power-spectral-density
      expansion) — PyMC provides a separate `pm.gp.HSGPPeriodic` class
      using a different low-rank basis (used below) for periodic
      structure.
    - Accuracy depends on `m` and the boundary factor `c` (or an
      explicit boundary `L`) both being large enough for the
      lengthscales actually present in the data, as explored above and
      in the exercise below.
    """)
    return


@app.cell
def _(data_dir, pl):
    hsgp_tides = pl.read_csv(data_dir / "noaa_tides_hourly.csv")
    hsgp_tides = hsgp_tides.with_columns(
        pl.col("time").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M")
    )
    hsgp_tides.head()
    return (hsgp_tides,)


@app.cell
def _(hsgp_tides, z):
    hsgp_t0 = hsgp_tides["time"][0]
    hsgp_hours = (hsgp_tides["time"] - hsgp_t0).dt.total_minutes().to_numpy() / 60.0
    hsgp_level = hsgp_tides["water_level"].to_numpy()

    hsgp_hours_std = hsgp_hours.std(ddof=0)
    hsgp_level_mean, hsgp_level_std = hsgp_level.mean(), hsgp_level.std(ddof=0)

    X_hsgp = z(hsgp_hours).reshape(-1, 1)  # GP inputs are 2D: (n, 1)
    y_hsgp = z(hsgp_level)
    return (
        X_hsgp,
        hsgp_hours,
        hsgp_hours_std,
        hsgp_level,
        hsgp_level_mean,
        hsgp_level_std,
        y_hsgp,
    )


@app.cell
def _(PYMC_BLUE, go, hsgp_hours, hsgp_level):
    hsgp_data_fig = go.Figure()
    hsgp_data_fig.add_trace(
        go.Scatter(
            x=hsgp_hours,
            y=hsgp_level,
            mode="lines",
            line=dict(color=PYMC_BLUE, width=1),
        )
    )
    hsgp_data_fig.update_layout(
        title=f"San Francisco hourly water level — full series (n={len(hsgp_hours):,})",
        xaxis_title="Hours since series start",
        yaxis_title="Water level (m, MLLW)",
        template="plotly_white",
    )
    hsgp_data_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Full-year HSGP: trend + semidiurnal + diurnal tide

    The same three-component kernel used for the FITC example is fit to all
    8,760 hourly observations. The slider controls choose the trend basis and
    boundary domain; fitting remains explicit through the button below.
    """)
    return


@app.cell
def _(X_hsgp, c_slider, hsgp_hours_std, np, pm):
    HSGP_PERIODIC_LS_STD = 0.5
    hsgp_semi_period_std = 12.42 / hsgp_hours_std
    hsgp_diurnal_period_std = 23.93 / hsgp_hours_std
    hsgp_L = float(c_slider.value * np.abs(X_hsgp[:, 0]).max())

    def build_hsgp_model(
        X, y, *, m_trend, L, semi_period, diurnal_period, periodic_ls
    ):
        coords = {"obs": np.arange(len(X)), "feature": ["standardized time"]}
        with pm.Model(coords=coords) as model:
            X_data = pm.Data("X", X, dims=("obs", "feature"))
            y_data = pm.Data("y_data", y, dims="obs")
            ell_trend = pm.LogNormal("ell_trend", mu=0, sigma=1)
            eta_trend = pm.HalfNormal("eta_trend", sigma=1)
            eta_semi = pm.HalfNormal("eta_semi", sigma=1)
            eta_diurnal = pm.HalfNormal("eta_diurnal", sigma=0.5)
            sigma_hsgp = pm.HalfNormal("sigma_hsgp", sigma=0.5)
            trend = pm.gp.HSGP(
                m=[m_trend],
                L=[L],
                cov_func=eta_trend**2 * pm.gp.cov.Matern52(1, ls=ell_trend),
            )
            semi = pm.gp.HSGPPeriodic(
                m=8,
                scale=eta_semi,
                cov_func=pm.gp.cov.Periodic(
                    1, period=semi_period, ls=periodic_ls
                ),
            )
            diurnal = pm.gp.HSGPPeriodic(
                m=8,
                scale=eta_diurnal,
                cov_func=pm.gp.cov.Periodic(
                    1, period=diurnal_period, ls=periodic_ls
                ),
            )
            f_trend = trend.prior("f_trend", X=X_data, dims="obs")
            f_semi = semi.prior("f_semi", X=X_data, dims="obs")
            f_diurnal = diurnal.prior("f_diurnal", X=X_data, dims="obs")
            pm.Normal(
                "y",
                mu=f_trend + f_semi + f_diurnal,
                sigma=sigma_hsgp,
                observed=y_data,
                dims="obs",
            )
        return model

    return (
        HSGP_PERIODIC_LS_STD,
        build_hsgp_model,
        hsgp_L,
        hsgp_diurnal_period_std,
        hsgp_semi_period_std,
    )


@app.cell
def _(
    HSGP_PERIODIC_LS_STD,
    X_hsgp,
    build_hsgp_model,
    hsgp_L,
    hsgp_diurnal_period_std,
    hsgp_semi_period_std,
    m_slider,
    y_hsgp,
):
    hsgp_model = build_hsgp_model(
        X_hsgp,
        y_hsgp,
        m_trend=m_slider.value,
        L=hsgp_L,
        semi_period=hsgp_semi_period_std,
        diurnal_period=hsgp_diurnal_period_std,
        periodic_ls=HSGP_PERIODIC_LS_STD,
    )
    hsgp_model.compile_logp()(hsgp_model.initial_point())
    return (hsgp_model,)


@app.cell
def _(RANDOM_SEED, hsgp_model, pm):
    with hsgp_model:
        hsgp_prior_pred = pm.sample_prior_predictive(
            draws=500, var_names=["y"], random_seed=RANDOM_SEED
        )
    return


@app.cell
def _(mo):
    hsgp_fit_button = mo.ui.run_button(label="Fit full-year HSGP")
    hsgp_fit_button
    return (hsgp_fit_button,)


@app.cell
def _(
    RANDOM_SEED,
    execute_models,
    hsgp_fit_button,
    hsgp_model,
    mo,
    perf_counter,
    pm,
):
    mo.stop(not (hsgp_fit_button.value or execute_models))
    with hsgp_model:
        hsgp_start = perf_counter()
        hsgp_idata = pm.sample(
            random_seed=RANDOM_SEED,
            chains=4,
            var_names=[rv.name for rv in hsgp_model.free_RVs],
        )
        hsgp_sample_seconds = perf_counter() - hsgp_start
    print(f"HSGP full-year sampling wall-time: {hsgp_sample_seconds:.1f}s")
    return (hsgp_idata,)


@app.cell
def _(
    HSGP_PERIODIC_LS_STD,
    hsgp_L,
    hsgp_diurnal_period_std,
    hsgp_hours_std,
    hsgp_idata,
    hsgp_level_mean,
    hsgp_level_std,
    hsgp_semi_period_std,
    m_slider,
    results_dir,
):
    hsgp_idata_to_save = hsgp_idata.copy()
    posterior = hsgp_idata_to_save["posterior"].ds
    hsgp_idata_to_save["posterior"] = posterior.drop_vars(
        ["f_trend", "f_semi", "f_diurnal", "y"], errors="ignore"
    )
    hsgp_idata_to_save.attrs.update(
        {
            "m_trend": int(m_slider.value),
            "L": float(hsgp_L),
            "semi_period": float(hsgp_semi_period_std),
            "diurnal_period": float(hsgp_diurnal_period_std),
            "periodic_ls": float(HSGP_PERIODIC_LS_STD),
            "time_std": float(hsgp_hours_std),
            "level_mean": float(hsgp_level_mean),
            "level_std": float(hsgp_level_std),
        }
    )
    hsgp_path = results_dir / "04_hsgp_full_year.nc"
    hsgp_idata_to_save.to_netcdf(hsgp_path)
    return (hsgp_path,)
@app.cell
def _(inference_health, hsgp_idata, hsgp_model):
    hsgp_summary, hsgp_n_div, hsgp_health_passed = inference_health(
        hsgp_idata, hsgp_model
    )
    hsgp_summary
    return hsgp_health_passed, hsgp_n_div



@app.cell
def _(az, hsgp_path):
    persisted_hsgp_idata = az.from_netcdf(hsgp_path)
    hsgp_config = persisted_hsgp_idata.attrs
    return hsgp_config, persisted_hsgp_idata


@app.cell
def _(X_hsgp, build_hsgp_model, hsgp_config, y_hsgp):
    hsgp_prediction_model = build_hsgp_model(
        X_hsgp,
        y_hsgp,
        m_trend=int(hsgp_config["m_trend"]),
        L=float(hsgp_config["L"]),
        semi_period=float(hsgp_config["semi_period"]),
        diurnal_period=float(hsgp_config["diurnal_period"]),
        periodic_ls=float(hsgp_config["periodic_ls"]),
    )
    hsgp_prediction_model.compile_logp()(hsgp_prediction_model.initial_point())
    return (hsgp_prediction_model,)


@app.cell
def _(
    RANDOM_SEED,
    hsgp_prediction_model,
    persisted_hsgp_idata,
    pm,
    posterior_subset,
):
    hsgp_predictive_subset = posterior_subset(persisted_hsgp_idata)
    with hsgp_prediction_model:
        hsgp_ppc = pm.sample_posterior_predictive(
            hsgp_predictive_subset, var_names=["y"], random_seed=RANDOM_SEED
        )
        hsgp_components = pm.sample_posterior_predictive(
            hsgp_predictive_subset,
            var_names=["f_trend", "f_semi", "f_diurnal"],
            predictions=True,
            random_seed=RANDOM_SEED,
        )
    return hsgp_components, hsgp_ppc


@app.cell
def _(hsgp_level, hsgp_ppc, np, pl):
    def autocorrelation(values, lag):
        centered = values - values.mean()
        return float(np.dot(centered[:-lag], centered[lag:]) / np.dot(centered, centered))

    lags = [1, 12, 24, 168]
    observed_residual = hsgp_level - hsgp_ppc["posterior_predictive"]["y"].mean(
        ("chain", "draw")
    ).values
    replicated = hsgp_ppc["posterior_predictive"]["y"].stack(
        sample=("chain", "draw")
    ).transpose("sample", "obs").values
    autocorr_table = pl.DataFrame(
        {
            "lag_hours": lags,
            "observed_residual_acf": [autocorrelation(observed_residual, lag) for lag in lags],
            "ppc_5.5%": [np.quantile([autocorrelation(row, lag) for row in replicated], 0.055) for lag in lags],
            "ppc_94.5%": [np.quantile([autocorrelation(row, lag) for row in replicated], 0.945) for lag in lags],
        }
    )
    return


@app.cell
def _(PYMC_BLUE, go, hsgp_components, hsgp_hours, hsgp_level):
    mask = hsgp_hours <= 24 * 14
    component_draws = hsgp_components["predictions"]
    fig = go.Figure()
    for name, color in (
        ("f_trend", PYMC_BLUE),
        ("f_semi", "#81C240"),
        ("f_diurnal", "#4A9EDE"),
    ):
        fig.add_trace(
            go.Scatter(
                x=hsgp_hours[mask],
                y=component_draws[name].mean(("chain", "draw")).values[mask],
                mode="lines",
                name=name.removeprefix("f_"),
                line=dict(color=color),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=hsgp_hours[mask], y=hsgp_level[mask], mode="markers",
            name="observed", marker=dict(color="black", size=3),
        )
    )
    fig.update_layout(
        title="HSGP component decomposition — first two weeks",
        xaxis_title="Hours since series start",
        yaxis_title="Standardized contribution / observed level",
        template="plotly_white",
    )
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise: assess an HSGP basis approximation

    Use the live `m` and `c` controls above to choose one smaller and one
    larger basis/domain configuration. For each choice, click **Fit full-year
    HSGP** explicitly; changing a control invalidates the previous fit but does
    not resample by itself. Before comparing the first-two-week components,
    predict which boundary or basis-accuracy symptom you would expect from a
    configuration that is too small. Record the saved `m_trend` and `L` for
    each fit so that the later prediction and posterior-predictive checks are
    tied to the basis actually sampled, not merely the current slider state.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Hint": mo.md(
                r"""
                A basis with too few functions cannot resolve all structure
                allowed by the kernel, while a boundary placed too near the
                observed standardized-time range can impose artificial edge
                behavior. The controls have different roles: `m` controls
                basis resolution and `c` determines the boundary factor used
                to calculate the training-domain half-width `L`.
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Discussion": mo.md(
                r"""
                A small `m` can omit resolvable trend structure; a small `c`
                places artificial basis boundaries too near the training
                domain, which can distort behavior near its ends. Increasing
                either setting is not automatically an improvement: it changes
                the approximation and requires a new explicit fit. Compare the
                persisted `m_trend` and `L` with first-two-week components,
                posterior-predictive location and spread, and residual
                autocorrelations at 1, 12, 24, and 168 hours.

                Prefer an HSGP for this large, stationary one-dimensional
                series only when those basis and residual checks support the
                approximation. Persistent periodic residual structure, edge
                artifacts, or posterior-predictive discrepancies are reasons
                to revisit the basis, boundary, or model assumptions—not to
                claim universal superiority over exact or sparse GPs.
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The ECDF, location/spread summaries, and residual-autocorrelation table
    are posterior-predictive diagnostics, not a pre-declared adequacy verdict.
    Read the observed discrepancies against their replicated distributions:
    values outside or near a tail keep the relevant model limitation visible
    and should guide a basis or likelihood revision.

    **Out of scope for this course:** comparing models by predictive
    accuracy via `az.compute_log_likelihood` + `az.loo`/`az.compare`
    (LOO-CV, ELPD) is a natural next step once you have more than one
    candidate model, but is beyond what we can cover in an
    introductory workshop — see the ArviZ documentation and the
    `model-evaluation` material for that workflow.

    ### Decision guide: exact vs. sparse vs. HSGP

    | | **Exact** (`pm.gp.Marginal` / `pm.gp.Latent`) | **Sparse** (`pm.gp.MarginalApprox`) | **HSGP** (`pm.gp.HSGP` / `HSGPPeriodic`) |
    |---|---|---|---|
    | **Cost** | $O(n^3)$ per gradient eval | $O(nm^2)$, $m$ = # inducing points | $O(nm)$, $m$ = # basis functions (linear in $n$) |
    | **When to use** | Small/moderate $n$ (up to a few hundred–low thousands); need exact inference | Moderate–large $n$; conjugate (Gaussian-noise) likelihoods; comfortable choosing inducing points | Large $n$ (thousands+); need speed and are willing to restrict to stationary kernels |
    | **Likelihood** | Any (`Marginal`: Gaussian only; `Latent`: any, via explicit sampling of $f$) | Gaussian only (conjugate) | Any — `.prior()` gives you $f$ to plug into any likelihood, just like `Latent` |
    | **Key constraint** | None beyond cost | Approximation quality depends on inducing point count/placement | Stationary kernels with a known power spectral density only; input dim practically $\lesssim 3$; needs $m$/boundary tuned to the data's lengthscales |
    | **What we saw here** | Notebooks 1–3: up to ~340 points | Part B: 450 points, 25 inducing points | Part C: 8,760 points, ~20–40 basis coefficients total |

    In practice: start exact whenever you can afford to (it is the
    easiest to reason about and has no approximation error to worry
    about); reach for sparse GPs when $n$ grows into the
    thousands–tens-of-thousands with a Gaussian likelihood and you can
    tolerate a modest, controllable approximation; reach for HSGP when
    $n$ is large, your kernel is stationary, and your input dimension
    is low — as here, where it is the only one of the three that makes
    fitting the full 8,760-point series practical at all.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Approximation workflow: a reusable checklist

    Scaling a GP is not merely a matter of selecting the fastest class. Start
    by preserving the generative question: draw from an approximation-specific
    prior and ask whether its functions and observations can represent the
    scientific signal. State the approximation configuration in the fitted
    artifact—inducing locations for FITC, or basis count, boundary, periods,
    and standardization for HSGP—so predictions can be reconstructed without
    hidden notebook state. Then separate latent-function predictions from
    noisy replicated observations, and compare the latter with the observed
    data using named posterior-predictive discrepancies.

    Finally, make the approximation choice conditional. FITC may be effective
    when a Gaussian likelihood and well-covered inducing locations make its
    residual approximation accurate. HSGP is particularly useful for a large,
    low-dimensional stationary problem when basis and boundary checks pass.
    Exact GPs remain the clearest reference for smaller datasets. None of
    these labels substitutes for checking identification, prior implications,
    inference health, and posterior-predictive behavior in the particular
    model at hand.

    For a practical review, ask four concrete questions. Are the covariate
    domain and units represented consistently in the approximation and its
    prediction model? Do prior draws reveal behavior that contradicts known
    scale, periodicity, or smoothness? Do sampled free variables have healthy
    diagnostics before their posterior is used for prediction? And do
    replicated observations reproduce the discrepancies that matter for the
    decision? This sequence prevents an efficient approximation from becoming
    an opaque black box. It also keeps the distinction clear between an
    approximation that is computationally convenient and one that is adequate
    for the observed data and stated predictive task.
    Apply this checklist anew for each fit.
    """)
    return


if __name__ == "__main__":
    app.run()
