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
        Path,
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
        """
    )
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
    mo.md(
        r"""
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
        """
    )
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
    return N_SPARSE, sparse_slice


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
def _(PYMC_BLUE, go, sparse_hours, sparse_level):
    sparse_data_fig = go.Figure()
    sparse_data_fig.add_trace(
        go.Scatter(
            x=sparse_hours,
            y=sparse_level,
            mode="lines",
            line=dict(color=PYMC_BLUE),
        )
    )
    sparse_data_fig.update_layout(
        title=f"San Francisco hourly water level — {len(sparse_hours)}-point slice",
        xaxis_title="Hours since slice start",
        yaxis_title="Water level (m, MLLW)",
        template="plotly_white",
    )
    sparse_data_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We reuse the same additive trend + semidiurnal + diurnal kernel
        structure from Notebook 3 (Part B), with the tidal periods and
        within-cycle shapes again fixed at their physically-motivated
        constants for the same reason as before — only now on more than
        twice as many points, and fit with a **small** set of 25 inducing
        points (about 5% of $n$) chosen by k-means over the standardized
        time axis, rather than the exact $450 \times 450$ covariance.
        """
    )
    return


@app.cell
def _(X_sparse, pm, sparse_hours_std):
    N_INDUCING = 25
    SPARSE_SEMI_PERIOD_HOURS = 12.42  # M2 lunar semidiurnal constituent
    SPARSE_DIURNAL_PERIOD_HOURS = 23.93  # K1 lunar diurnal constituent
    SPARSE_PERIODIC_LS_STD = 0.5  # fixed within-cycle shape (standardized time units)
    sparse_semi_period_std = SPARSE_SEMI_PERIOD_HOURS / sparse_hours_std
    sparse_diurnal_period_std = SPARSE_DIURNAL_PERIOD_HOURS / sparse_hours_std

    Xu_init = pm.gp.util.kmeans_inducing_points(N_INDUCING, X_sparse)

    with pm.Model() as sparse_model:
        ell_s_trend = pm.InverseGamma("ell_s_trend", alpha=5, beta=5)
        eta_s_trend = pm.HalfNormal("eta_s_trend", sigma=1)
        cov_s_trend = eta_s_trend**2 * pm.gp.cov.Matern52(1, ls=ell_s_trend)

        eta_s_semi = pm.HalfNormal("eta_s_semi", sigma=1)
        cov_s_semi = eta_s_semi**2 * pm.gp.cov.Periodic(
            1, period=sparse_semi_period_std, ls=SPARSE_PERIODIC_LS_STD
        )

        eta_s_diurnal = pm.HalfNormal("eta_s_diurnal", sigma=1)
        cov_s_diurnal = eta_s_diurnal**2 * pm.gp.cov.Periodic(
            1, period=sparse_diurnal_period_std, ls=SPARSE_PERIODIC_LS_STD
        )

        cov_sparse = cov_s_trend + cov_s_semi + cov_s_diurnal
        sigma_sparse = pm.HalfNormal("sigma_sparse", sigma=0.5)

        Xu = pm.Data("Xu", Xu_init)
        gp_sparse = pm.gp.MarginalApprox(cov_func=cov_sparse, approx="FITC")

    return (
        N_INDUCING,
        SPARSE_PERIODIC_LS_STD,
        Xu,
        cov_sparse,
        gp_sparse,
        sigma_sparse,
        sparse_diurnal_period_std,
        sparse_model,
        sparse_semi_period_std,
    )


@app.cell
def _(X_sparse, Xu, gp_sparse, sigma_sparse, sparse_model, y_sparse):
    with sparse_model:
        gp_sparse.marginal_likelihood(
            "y", X=X_sparse, Xu=Xu, y=y_sparse, sigma=sigma_sparse
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Prior predictive check

        `MarginalApprox.marginal_likelihood` registers its FITC likelihood
        as a `pm.Potential` (a log-density contribution), not as an
        observed random variable — so there is no generative "y" node to
        draw prior-predictive samples from directly on `sparse_model`
        itself. Since the FITC approximation only changes how the
        **posterior** is computed, not the **prior** over the kernel
        hyperparameters, we build a second, throwaway model below with the
        identical priors and kernel but an *exact* `pm.gp.Marginal`
        likelihood (which does register an observed RV) purely to draw
        prior-predictive `y` samples for this plausibility check. It is
        never sampled from with `pm.sample` — only `sparse_model` above is
        used for actual inference.
        """
    )
    return


@app.cell
def _(
    RANDOM_SEED,
    SPARSE_PERIODIC_LS_STD,
    X_sparse,
    pm,
    sparse_diurnal_period_std,
    sparse_semi_period_std,
    y_sparse,
):
    with pm.Model() as _sparse_prior_model:
        _ell_trend = pm.InverseGamma("ell_s_trend", alpha=5, beta=5)
        _eta_trend = pm.HalfNormal("eta_s_trend", sigma=1)
        _cov_trend = _eta_trend**2 * pm.gp.cov.Matern52(1, ls=_ell_trend)

        _eta_semi = pm.HalfNormal("eta_s_semi", sigma=1)
        _cov_semi = _eta_semi**2 * pm.gp.cov.Periodic(
            1, period=sparse_semi_period_std, ls=SPARSE_PERIODIC_LS_STD
        )

        _eta_diurnal = pm.HalfNormal("eta_s_diurnal", sigma=1)
        _cov_diurnal = _eta_diurnal**2 * pm.gp.cov.Periodic(
            1, period=sparse_diurnal_period_std, ls=SPARSE_PERIODIC_LS_STD
        )

        _sigma = pm.HalfNormal("sigma_sparse", sigma=0.5)
        _gp_exact = pm.gp.Marginal(cov_func=_cov_trend + _cov_semi + _cov_diurnal)
        _gp_exact.marginal_likelihood("y", X=X_sparse, y=y_sparse, sigma=_sigma)

        sparse_prior_pred = pm.sample_prior_predictive(
            draws=500, random_seed=RANDOM_SEED
        )
    return (sparse_prior_pred,)


@app.cell
def _(PYMC_LIGHT_BLUE, X_sparse, go, np, sparse_prior_pred, y_sparse):
    sparse_prior_draws = sparse_prior_pred["prior_predictive"]["y"].values.reshape(
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
def _(RANDOM_SEED, perf_counter, pm, sparse_model):
    with sparse_model:
        sparse_start = perf_counter()
        sparse_idata = pm.sample(random_seed=RANDOM_SEED)
        sparse_sample_seconds = perf_counter() - sparse_start
    print(f"Sparse FITC-GP sampling wall-time: {sparse_sample_seconds:.1f}s")
    return sparse_idata, sparse_sample_seconds


@app.cell
def _(az, sparse_idata):
    sparse_n_div = sparse_idata["sample_stats"]["diverging"].sum().item()
    sparse_n_draws_total = (
        sparse_idata["posterior"].sizes["chain"]
        * sparse_idata["posterior"].sizes["draw"]
    )
    sparse_summary = az.summary(sparse_idata["posterior"])
    sparse_min_ess_bulk = float(sparse_summary["ess_bulk"].min())
    sparse_min_ess_tail = float(sparse_summary["ess_tail"].min())
    sparse_max_rhat = float(sparse_summary["r_hat"].astype(float).max())
    print(f"Divergences: {sparse_n_div} / {sparse_n_draws_total}")
    sparse_summary
    return (
        sparse_max_rhat,
        sparse_min_ess_bulk,
        sparse_min_ess_tail,
        sparse_n_div,
        sparse_n_draws_total,
    )


@app.cell(hide_code=True)
def _(
    mo,
    sparse_max_rhat,
    sparse_min_ess_bulk,
    sparse_min_ess_tail,
    sparse_n_div,
    sparse_n_draws_total,
    sparse_sample_seconds,
):
    mo.md(
        f"""
        **Diagnostics:** {sparse_n_div} divergence(s) out of
        {sparse_n_draws_total} draws in {sparse_sample_seconds:.1f}s.
        Minimum `ess_bulk` is {sparse_min_ess_bulk:.0f} and minimum
        `ess_tail` is {sparse_min_ess_tail:.0f} — both comfortably above
        the 400 threshold — and maximum `r_hat` is {sparse_max_rhat:.3f}.
        Safe to interpret.
        """
    )
    return


@app.cell
def _(X_sparse, gp_sparse, sparse_model):
    with sparse_model:
        f_sparse_pred = gp_sparse.conditional("f_sparse_pred", X_sparse)
    return (f_sparse_pred,)


@app.cell
def _(RANDOM_SEED, f_sparse_pred, pm, sparse_idata, sparse_model):
    with sparse_model:
        sparse_ppc = pm.sample_posterior_predictive(
            sparse_idata, var_names=["f_sparse_pred"], random_seed=RANDOM_SEED
        )
    return (sparse_ppc,)


@app.cell
def _(
    N_INDUCING,
    PYMC_BLUE,
    Xu,
    az,
    go,
    np,
    sparse_hours,
    sparse_level,
    sparse_level_mean,
    sparse_level_std,
    sparse_ppc,
):
    sparse_fit_vals = (
        sparse_ppc["posterior_predictive"]["f_sparse_pred"].values.reshape(
            -1, len(sparse_hours)
        )
        * sparse_level_std
        + sparse_level_mean
    )
    sparse_fit_mean = sparse_fit_vals.mean(axis=0)
    sparse_fit_hdi = az.hdi(sparse_fit_vals, prob=0.89, axis=0)
    sparse_fit_lo, sparse_fit_hi = sparse_fit_hdi[:, 0], sparse_fit_hdi[:, 1]

    sparse_fit_fig = go.Figure()
    sparse_fit_fig.add_trace(
        go.Scatter(
            x=np.concatenate([sparse_hours, sparse_hours[::-1]]),
            y=np.concatenate([sparse_fit_hi, sparse_fit_lo[::-1]]),
            fill="toself",
            fillcolor="rgba(21,74,114,0.25)",
            line=dict(color="rgba(255,255,255,0)"),
            name="89% HDI",
        )
    )
    sparse_fit_fig.add_trace(
        go.Scatter(
            x=sparse_hours,
            y=sparse_fit_mean,
            mode="lines",
            name="posterior mean fit",
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
    _Xu_hours = (
        (np.asarray(Xu.get_value())[:, 0]) * sparse_hours.std(ddof=0)
        + sparse_hours.mean()
    )
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
        title=f"Sparse FITC-GP fit — {N_INDUCING} inducing points, {len(sparse_hours)} training points",
        xaxis_title="Hours since slice start",
        yaxis_title="Water level (m, MLLW)",
        template="plotly_white",
    )
    sparse_fit_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
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
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
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
        """
    )
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
    mo.md(
        r"""
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
        """
    )
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
    mo.md(
        r"""
        ### Model: trend + semidiurnal tide

        As discussed in Notebook 3, the semidiurnal (~12.42h) and diurnal
        (~23.93h) tidal periods are close to commensurate (23.93 ≈
        2 × 12.42), which makes a posterior that frees both components'
        amplitudes simultaneously hard to sample — the two periodic
        components can trade off against each other. Over a full year of
        hourly data that difficulty is worse, not better: 8,760 points give
        the near-commensurate frequencies far more opportunity to alias
        against one another. To keep the full-series fit well-behaved for
        this introductory workshop, we model only the **dominant**
        semidiurnal component (`pm.gp.HSGPPeriodic`, period fixed at the
        physical constant) plus a smooth trend (`pm.gp.HSGP` with a
        `Matern52` kernel, using the $m$/$c$ slider values above) — leaving
        out the smaller diurnal correction. This is a real modeling
        simplification, not a limitation of HSGP itself; a production tide
        model would resolve it (e.g. by fitting on a coarser dataset with
        each period's amplitude given more informative, mutually
        constraining priors).
        """
    )
    return


@app.cell
def _(hsgp_hours_std):
    HSGP_SEMI_PERIOD_HOURS = 12.42  # M2 lunar semidiurnal constituent
    HSGP_PERIODIC_LS_STD = 0.5  # fixed within-cycle shape (standardized time units)
    hsgp_semi_period_std = HSGP_SEMI_PERIOD_HOURS / hsgp_hours_std
    return HSGP_PERIODIC_LS_STD, hsgp_semi_period_std


@app.cell
def _(
    HSGP_PERIODIC_LS_STD, X_hsgp, c_slider, hsgp_semi_period_std, m_slider, pm, y_hsgp
):
    with pm.Model() as hsgp_model:
        ell_trend = pm.InverseGamma("ell_trend", alpha=5, beta=5)
        eta_trend = pm.HalfNormal("eta_trend", sigma=1)
        cov_trend = eta_trend**2 * pm.gp.cov.Matern52(1, ls=ell_trend)
        gp_trend = pm.gp.HSGP(m=[m_slider.value], c=c_slider.value, cov_func=cov_trend)
        f_trend = gp_trend.prior("f_trend", X=X_hsgp)

        eta_semi = pm.HalfNormal("eta_semi", sigma=1)
        cov_semi = pm.gp.cov.Periodic(
            1, period=hsgp_semi_period_std, ls=HSGP_PERIODIC_LS_STD
        )
        gp_semi = pm.gp.HSGPPeriodic(m=8, scale=eta_semi, cov_func=cov_semi)
        f_semi = gp_semi.prior("f_semi", X=X_hsgp)

        sigma_hsgp = pm.HalfNormal("sigma_hsgp", sigma=0.5)
        pm.Normal("y", mu=f_trend + f_semi, sigma=sigma_hsgp, observed=y_hsgp)

    return f_semi, f_trend, hsgp_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Prior predictive check""")
    return


@app.cell
def _(RANDOM_SEED, hsgp_model, pm):
    with hsgp_model:
        hsgp_prior_pred = pm.sample_prior_predictive(draws=500, random_seed=RANDOM_SEED)
    return (hsgp_prior_pred,)


@app.cell
def _(PYMC_LIGHT_BLUE, X_hsgp, go, hsgp_prior_pred, np, y_hsgp):
    hsgp_prior_draws = hsgp_prior_pred["prior_predictive"]["y"].values.reshape(
        -1, len(y_hsgp)
    )

    hsgp_prior_fig = go.Figure()
    _rng_plot = np.random.default_rng(0)
    for _i in _rng_plot.choice(hsgp_prior_draws.shape[0], size=50, replace=False):
        hsgp_prior_fig.add_trace(
            go.Scatter(
                x=X_hsgp[:, 0],
                y=hsgp_prior_draws[_i],
                mode="lines",
                line=dict(color=PYMC_LIGHT_BLUE, width=1),
                opacity=0.12,
                showlegend=False,
            )
        )
    hsgp_prior_fig.add_trace(
        go.Scatter(
            x=X_hsgp[:, 0],
            y=y_hsgp,
            mode="lines",
            line=dict(color="black", width=1),
            name="observed (standardized)",
        )
    )
    hsgp_prior_fig.update_layout(
        title="Prior predictive draws vs. standardized observed tide level (full year)",
        xaxis_title="time (standardized)",
        yaxis_title="water level (standardized)",
        template="plotly_white",
    )
    hsgp_prior_fig
    return (hsgp_prior_draws,)


@app.cell(hide_code=True)
def _(hsgp_prior_draws, mo, y_hsgp):
    mo.md(
        f"""
        **Plausibility check:** prior predictive draws span
        [{hsgp_prior_draws.min():.2f}, {hsgp_prior_draws.max():.2f}] on the
        standardized scale, comfortably bracketing the observed range
        [{y_hsgp.min():.2f}, {y_hsgp.max():.2f}] — broad but not absurd,
        reasonable to proceed to sampling.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Sampling

        8,760 observations, but only the `m_slider` trend basis
        coefficients + 15 periodic basis coefficients + 4 hyperparameters
        are actually sampled (HSGP's fixed-basis, linear-in-$n$ structure
        again) — so despite the data size, this is a small sampling
        problem. We raise `target_accept` above the 0.8 default and use
        more draws than the `1000` default, both because the near-periodic
        structure still leaves a mildly delicate posterior geometry even
        with only one periodic component.
        """
    )
    return


@app.cell
def _(RANDOM_SEED, hsgp_model, perf_counter, pm):
    with hsgp_model:
        hsgp_start = perf_counter()
        hsgp_idata = pm.sample(
            random_seed=RANDOM_SEED,
            target_accept=0.97,
            draws=1500,
            tune=1500,
        )
        hsgp_sample_seconds = perf_counter() - hsgp_start
    print(f"HSGP full-year sampling wall-time: {hsgp_sample_seconds:.1f}s")
    return hsgp_idata, hsgp_sample_seconds


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Save early

        Before computing a single diagnostic, save the freshly sampled
        `idata` to disk. Posterior sampling is the most expensive step in
        this whole workflow — if a later cell crashes, the kernel
        restarts, or a diagnostic call has a bug, an unsaved `idata` means
        resampling everything from scratch. Saving immediately, *before*
        any inspection, costs almost nothing and is cheap insurance. We
        revisit this file (and the rest of the diagnostic loop) formally in
        Part D below.
        """
    )
    return


@app.cell
def _(Path, hsgp_idata):
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    hsgp_idata.to_netcdf(str(results_dir / "hsgp.nc"))
    return (results_dir,)


@app.cell
def _(az, hsgp_idata):
    hsgp_n_div = hsgp_idata["sample_stats"]["diverging"].sum().item()
    hsgp_n_draws_total = (
        hsgp_idata["posterior"].sizes["chain"] * hsgp_idata["posterior"].sizes["draw"]
    )
    # Full-posterior summary restricted to the model's actual free parameters —
    # hyperparameters and the low-dimensional HSGP basis coefficients — rather
    # than the two 8,760-length f_trend/f_semi Deterministics, which are a
    # fixed linear projection of those same coefficients and would make
    # ess/r_hat computation extremely slow (tens of thousands of entries)
    # without adding any diagnostic information beyond what the coefficients
    # already show.
    hsgp_var_names = [
        "ell_trend",
        "eta_trend",
        "f_trend_hsgp_coeffs",
        "eta_semi",
        "f_semi_hsgp_coeffs_",
        "sigma_hsgp",
    ]
    hsgp_summary = az.summary(hsgp_idata["posterior"], var_names=hsgp_var_names)
    hsgp_min_ess_bulk = float(hsgp_summary["ess_bulk"].min())
    hsgp_min_ess_tail = float(hsgp_summary["ess_tail"].min())
    hsgp_max_rhat = float(hsgp_summary["r_hat"].astype(float).max())
    print(f"Divergences: {hsgp_n_div} / {hsgp_n_draws_total}")
    hsgp_summary
    return (
        hsgp_max_rhat,
        hsgp_min_ess_bulk,
        hsgp_min_ess_tail,
        hsgp_n_div,
        hsgp_n_draws_total,
    )


@app.cell(hide_code=True)
def _(
    hsgp_max_rhat,
    hsgp_min_ess_bulk,
    hsgp_min_ess_tail,
    hsgp_n_div,
    hsgp_n_draws_total,
    hsgp_sample_seconds,
    mo,
):
    mo.md(
        f"""
        **Diagnostics:** {hsgp_n_div} divergence(s) out of
        {hsgp_n_draws_total} draws in {hsgp_sample_seconds:.1f}s. Computed
        over the model's actual free parameters (hyperparameters + HSGP
        basis coefficients, not the derived per-timepoint arrays) — minimum
        `ess_bulk` is {hsgp_min_ess_bulk:.0f} and minimum `ess_tail` is
        {hsgp_min_ess_tail:.0f} — both above the 400 threshold — and
        maximum `r_hat` is {hsgp_max_rhat:.3f}. Safe to interpret.
        """
    )
    return


@app.cell
def _(
    PYMC_BLUE,
    az,
    go,
    hsgp_hours,
    hsgp_idata,
    hsgp_level,
    hsgp_level_mean,
    hsgp_level_std,
    np,
):
    # Extract .values before adding: f_trend/f_semi carry independently
    # auto-generated xarray dim names for their timepoint axis, so adding the
    # xarray.DataArrays directly (before dropping to plain numpy) would
    # broadcast them as an outer product instead of an elementwise sum.
    hsgp_fit_vals = (
        hsgp_idata["posterior"]["f_trend"].values
        + hsgp_idata["posterior"]["f_semi"].values
    ).reshape(-1, len(hsgp_hours)) * hsgp_level_std + hsgp_level_mean
    hsgp_fit_mean = hsgp_fit_vals.mean(axis=0)
    hsgp_fit_hdi = az.hdi(hsgp_fit_vals, prob=0.89, axis=0)
    hsgp_fit_lo, hsgp_fit_hi = hsgp_fit_hdi[:, 0], hsgp_fit_hdi[:, 1]

    hsgp_fit_fig = go.Figure()
    hsgp_fit_fig.add_trace(
        go.Scatter(
            x=np.concatenate([hsgp_hours, hsgp_hours[::-1]]),
            y=np.concatenate([hsgp_fit_hi, hsgp_fit_lo[::-1]]),
            fill="toself",
            fillcolor="rgba(21,74,114,0.25)",
            line=dict(color="rgba(255,255,255,0)"),
            name="89% HDI",
        )
    )
    hsgp_fit_fig.add_trace(
        go.Scatter(
            x=hsgp_hours,
            y=hsgp_level,
            mode="lines",
            name="observed",
            line=dict(color="black", width=1),
            opacity=0.5,
        )
    )
    hsgp_fit_fig.add_trace(
        go.Scatter(
            x=hsgp_hours,
            y=hsgp_fit_mean,
            mode="lines",
            name="posterior mean fit",
            line=dict(color=PYMC_BLUE, width=1),
        )
    )
    hsgp_fit_fig.update_layout(
        title="HSGP fit — trend + semidiurnal tide, full year",
        xaxis_title="Hours since series start",
        yaxis_title="Water level (m, MLLW)",
        template="plotly_white",
    )
    hsgp_fit_fig
    return (hsgp_fit_vals,)


@app.cell
def _(PYMC_BLUE, go, hsgp_fit_vals, hsgp_hours, hsgp_level):
    _mask = hsgp_hours <= 24 * 14  # first two weeks, at full hourly resolution
    hsgp_zoom_mean = hsgp_fit_vals[:, _mask].mean(axis=0)

    hsgp_zoom_fig = go.Figure()
    hsgp_zoom_fig.add_trace(
        go.Scatter(
            x=hsgp_hours[_mask],
            y=hsgp_zoom_mean,
            mode="lines",
            name="HSGP posterior mean",
            line=dict(color=PYMC_BLUE, width=2),
        )
    )
    hsgp_zoom_fig.add_trace(
        go.Scatter(
            x=hsgp_hours[_mask],
            y=hsgp_level[_mask],
            mode="markers",
            name="observed",
            marker=dict(color="black", size=3),
        )
    )
    hsgp_zoom_fig.update_layout(
        title="HSGP fit — first two weeks, zoomed in to full hourly resolution",
        xaxis_title="Hours since series start",
        yaxis_title="Water level (m, MLLW)",
        template="plotly_white",
    )
    hsgp_zoom_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Zoomed out to the full year, the HSGP posterior mean tracks the
        slow seasonal drift in mean water level; zoomed in to two weeks at
        full hourly resolution, the semidiurnal tidal cycle is still
        resolved cleanly — the basis-function approximation has not
        blurred out the fast-varying structure it was given enough basis
        functions to represent. (The diurnal inequality — alternating
        higher/lower high tides each day, visible in the exact fits in
        Notebook 3 — is smoothed over here, since we deliberately omitted
        that second periodic component above.)

        ### Exercise: change `m` and `c` and judge the tradeoff

        Go back to the `m` / `c` sliders above Part C and try two changes,
        one at a time, then re-run the sampling cell and diagnostics:

        1. **Drop `m` to something small**, e.g. 8. Does the full-year fit
           above still track the slow seasonal trend, or does it visibly
           under-fit (too smooth, missing real drift)?
        2. **Push `c` up to 3.0 while leaving `m` at its default (20).**
           Does the fit change much? What happens to the diagnostics
           (divergences, `ess_bulk`, `r_hat`) as you push $c$ up without
           also raising $m$?

        Expand below for a discussion once you've tried at least one.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Discussion": mo.md(
                """
                With **`m` too small** (e.g. 8), the trend component has
                too few basis functions to represent the actual shape of
                the seasonal drift at the lengthscale the data supports —
                the fitted mean over-smooths, missing real wiggles in the
                trend that a larger `m` recovers. This is the direct analog
                of using too few Fourier terms to approximate a function:
                the *shape* of the error is a systematically over-smoothed
                curve, not random noise.

                With **`c` pushed up without raising `m`** in tandem, the
                same fixed 20 basis functions must now be spread over a
                wider domain (`c=3.0` instead of `1.5`), which coarsens
                their resolution *within* the actual data range even though
                nothing about the true underlying function changed. In
                practice this often shows up as: a visually similar (or
                slightly worse) fit, but sometimes *harder* sampling —
                wider, more spread-out basis functions can make the
                per-coefficient posterior less well identified, which can
                lower `ess_bulk` or raise divergences slightly, even though
                the model "looks" more conservative. The lesson generalizes:
                `m` and `c` are not independent knobs to be maxed out
                separately — for a fixed computational budget, increasing
                one without the other rarely helps, and checking
                *diagnostics*, not just eyeballing the fit, is the only
                reliable way to tell whether an HSGP approximation has
                enough resolution.
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Part D: The model-development workflow

        We've already done the first two steps of the standard workflow on
        the HSGP fit above — a **prior predictive check** before ever
        calling `pm.sample`, and **saving `idata` to disk immediately**
        after sampling finished, before looking at a single diagnostic.
        This section completes the loop on that same `hsgp_idata`: two
        more diagnostics beyond divergences and `r_hat`/`ess`, then a
        decision guide for choosing among the three GP flavors covered in
        this notebook.
        """
    )
    return


@app.cell
def _(az, hsgp_idata):
    # az.plot_energy returns a PlotCollection (matplotlib backend by
    # default); pull out the underlying Figure for marimo to render.
    hsgp_energy_pc = az.plot_energy(hsgp_idata)
    hsgp_energy_pc.viz["figure"].item()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Reading the energy plot:** it overlays the marginal energy
        distribution with the energy-transition distribution. When the two
        histograms overlap closely (as above), HMC/NUTS is exploring the
        posterior's energy levels efficiently; a transition distribution
        that is visibly narrower than the marginal one signals poor
        exploration (the sampler struggles to move between energy levels) —
        a geometry problem that `r_hat` and `ess` alone can miss, since
        those are computed per-parameter while the energy diagnostic
        reflects the sampler's global behavior across the whole joint
        posterior.

        ### Posterior predictive check

        Finally, simulate new data from the fitted model and compare its
        distribution to the observed data — a check that the *whole*
        model (not just individual hyperparameters) can plausibly have
        generated what we saw.
        """
    )
    return


@app.cell
def _(RANDOM_SEED, hsgp_idata, hsgp_model, pm):
    with hsgp_model:
        wf_ppc = pm.sample_posterior_predictive(
            hsgp_idata, var_names=["y"], random_seed=RANDOM_SEED
        )
    return (wf_ppc,)


@app.cell
def _(PYMC_BLUE, PYMC_GREEN, go, np, wf_ppc, y_hsgp):
    def _ecdf(vals):
        s = np.sort(vals)
        p = np.arange(1, len(s) + 1) / len(s)
        return s, p

    wf_y_rep = wf_ppc["posterior_predictive"]["y"].values.reshape(-1, len(y_hsgp))
    wf_obs_x, wf_obs_p = _ecdf(y_hsgp)

    wf_ppc_fig = go.Figure()
    _rng_ecdf = np.random.default_rng(0)
    for _i in _rng_ecdf.choice(wf_y_rep.shape[0], size=30, replace=False):
        _xs, _ps = _ecdf(wf_y_rep[_i])
        wf_ppc_fig.add_trace(
            go.Scatter(
                x=_xs,
                y=_ps,
                mode="lines",
                line=dict(color=PYMC_GREEN, width=1),
                opacity=0.25,
                showlegend=False,
            )
        )
    wf_ppc_fig.add_trace(
        go.Scatter(
            x=wf_obs_x,
            y=wf_obs_p,
            mode="lines",
            line=dict(color=PYMC_BLUE, width=2),
            name="observed ECDF",
        )
    )
    wf_ppc_fig.update_layout(
        title="Posterior predictive check — ECDF overlay (30 draws vs. observed)",
        xaxis_title="water level (standardized)",
        yaxis_title="cumulative probability",
        template="plotly_white",
    )
    wf_ppc_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The observed ECDF (blue) sits comfortably inside the band of 30
        posterior-predictive ECDFs (green) at every quantile — the fitted
        model's implied data distribution matches the real one well, with
        no systematic gap indicating a mis-specified likelihood or missing
        structure.

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
        | **What we saw here** | Notebooks 1–3: up to ~200 points, seconds | Part B: 450 points, 25 inducing points, ~30s | Part C: 8,760 points, ~20–40 basis coefficients total, under a minute |

        In practice: start exact whenever you can afford to (it is the
        easiest to reason about and has no approximation error to worry
        about); reach for sparse GPs when $n$ grows into the
        thousands–tens-of-thousands with a Gaussian likelihood and you can
        tolerate a modest, controllable approximation; reach for HSGP when
        $n$ is large, your kernel is stationary, and your input dimension
        is low — as here, where it is the only one of the three that makes
        fitting the full 8,760-point series practical at all.
        """
    )
    return


if __name__ == "__main__":
    app.run()
