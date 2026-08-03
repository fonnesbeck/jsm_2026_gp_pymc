import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo

    import sys
    from pathlib import Path
    from time import perf_counter

    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from inference_contract import inference_health

    import arviz as az
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import polars as pl
    import pymc as pm
    import preliz as pz

    PYMC_BLUE = "#154A72"
    PYMC_GREEN = "#81C240"
    PYMC_LIGHT_BLUE = "#4A9EDE"
    PYMC_DARK_GREEN = "#40611F"

    RANDOM_SEED = 42
    data_dir = project_root / "data"

    def eti(data, prob=0.89):
        return data.quantile(
            [(1 - prob) / 2, 1 - (1 - prob) / 2], dim=("chain", "draw")
        )

    def posterior_subset(idata, draws_per_chain=100):
        n_draws = idata["posterior"].sizes["draw"]
        indices = np.linspace(0, n_draws - 1, min(draws_per_chain, n_draws), dtype=int)
        return idata.isel(draw=indices, missing_dims="ignore")

    def z(a):
        """Standardize an array: (a - mean) / population std."""
        return (a - a.mean()) / a.std(ddof=0)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Scaling, Approximate GPs, and the Model Workflow

    Exact Gaussian processes become impractical as the number of observations
    grows because covariance factorization scales cubically with data size.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Why exact GPs don't scale

    The cost is visible directly in the GP posterior. For a marginal GP,
    the posterior covariance at a new input $x^*$ is

    $$k^*(x^*) = k(x^*,x^*) + \sigma^2 - k(x^*,x)^T \underbrace{\left[k(x,x) + \sigma^2 I\right]^{-1}}_{\rule{0pt}{0.9em}\raisebox{-4pt}{😢}}\, k(x^*,x)$$

    The braced term is the bottleneck: it inverts the $n \times n$
    training covariance. In practice we never form that inverse, we take a
    Cholesky factorization of $k(x,x) + \sigma^2 I$ and solve, but the
    cost is the same $O(n^3)$, and the matrix itself is $O(n^2)$ to store.

    Fitting `pm.gp.Marginal` (or `pm.gp.Latent`) exactly pays that cost at
    **every** gradient evaluation NUTS makes during sampling, since
    $\ell$, $\eta$, and $\sigma$ change $K$ at every step, and NUTS
    typically needs thousands of gradient evaluations over a full run.
    Below we measure just the factorization step, building $K$ and calling
    `np.linalg.cholesky`, with **no sampling at all**, as $n$ grows, to
    make the scaling visible directly.
    """)
    return


@app.cell(hide_code=True)
def _():
    def scale_cov(X, ls=1.0, eta=1.0):
        d = np.abs(X[:, None, 0] - X[None, :, 0])
        K = eta**2 * np.exp(-0.5 * (d / ls) ** 2)
        return K + 1e-6 * np.eye(len(X))

    scale_ns = [100, 300, 900, 2700]
    scale_times = []
    for scale_n in scale_ns:
        scale_X = np.linspace(0, 10, scale_n)[:, None]
        scale_start = perf_counter()
        scale_K = scale_cov(scale_X)
        np.linalg.cholesky(scale_K)
        scale_times.append(perf_counter() - scale_start)
    return scale_ns, scale_times


@app.cell(hide_code=True)
def _(scale_ns, scale_times):
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
def _(scale_ns, scale_times):
    _ratio_n = scale_ns[-1] / scale_ns[-2]
    _ratio_t = scale_times[-1] / scale_times[-2]
    _extrap_n = 8760
    _extrap_t = scale_times[-1] * (_extrap_n / scale_ns[-1]) ** 3
    mo.md(
        f"""
        **What to notice:** going from $n={scale_ns[-2]}$ to $n={scale_ns[-1]}$
        (a {_ratio_n:.0f}× increase in $n$) took {_ratio_t:.1f}× longer ,
        close to the {_ratio_n**3:.0f}× an exact $O(n^3)$ law predicts (the
        exact multiple varies run to run since these are sub-second wall
        times, but the cubic trend is unmistakable on the log-log plot
        above). Extrapolating that cubic trend to $n={_extrap_n:,}$, a
        full year of hourly NOAA tide readings, a **single**
        Cholesky factorization would already take on the order of
        {_extrap_t:.0f}s, and NUTS needs one such factorization (or more)
        per gradient evaluation, typically thousands of them over a full
        run. Exact inference at that scale is not practical. The rest of
        this notebook covers two ways around the bottleneck.
        """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Two views of the same model

    Everything from here on is an approximation, so it pays to be precise about
    *what* is being approximated. A GP can be written down two equivalent ways
    (Rasmussen & Williams 2006, ch. 2).

    **Function space** is what we have used so far: put the prior directly on
    functions, $f \sim \mathcal{GP}\bigl(m(x),\, k(x,x')\bigr)$, and let the
    kernel say which functions are plausible.

    **Weight space** instead fixes a set of basis functions $\phi(x)$ and puts
    the prior on their coefficients, which is just Bayesian linear regression in
    a transformed space:

    $$f(x) = \phi(x)^\top w, \qquad w \sim \mathcal{N}(0, \Sigma_p)$$

    These are not two models. Ask what covariance the second one implies and the
    kernel falls out immediately:

    $$\operatorname{cov}\bigl(f(x), f(x')\bigr) = \phi(x)^\top\, \mathbb{E}[w w^\top]\, \phi(x') = \phi(x)^\top \Sigma_p\, \phi(x')$$

    $$\boxed{\;k(x,x') = \phi(x)^\top \Sigma_p\, \phi(x')\;}$$

    Choosing a basis and a weight prior **is** choosing a kernel. Choosing a
    kernel **is** choosing a basis, and that is the direction that matters here.

    ### Why this is the key to scaling

    The same posterior has two computational routes, and they cost differently.
    With $n$ observations and $N$ basis functions, one route inverts an
    $n \times n$ matrix and the other an $N \times N$ matrix. You take whichever
    is smaller. Exact GPs are stuck with the first, which is the $O(n^3)$ wall
    above.

    The catch is that a kernel like `ExpQuad` corresponds to a weight-space model
    with **infinitely many** basis functions, so no finite basis reproduces it
    exactly. Keep only $m$ of them and the implied covariance has rank $m$: with
    $n > m$ it is singular, a *degenerate* GP. That is the precise sense in which
    everything below is approximate.

    Both approximations in this notebook route the covariance through an
    $m$-dimensional subspace. They differ in how they choose it, and in what
    they do about what the subspace misses:

    | | covariance used | subspace chosen by |
    |---|---|---|
    | Exact GP | $K_{nn}$, rank $n$ | nothing truncated |
    | Sparse (FITC) | rank-$m$ term plus a diagonal correction | placement of $m$ inducing points |
    | HSGP | rank $m$ | first $m$ basis functions of the domain |

    So $m$ is an **accuracy-versus-cost** dial in both cases, not a
    flexibility-versus-overfitting one. Raising $m$ moves you toward the exact
    GP, never past it.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Sparse approximation, inducing points

    Instead of the full $n \times n$ covariance, a **sparse** GP
    approximation summarizes the training data with a small set of
    $m \ll n$ **inducing points** (also called pseudo-inputs), located
    somewhere in the input space (not necessarily at observed $x$
    values).

    ### What the approximation actually assumes

    Let $u = f(X_u)$ be the (unobserved) GP values at the $m$ inducing
    locations. Because $f$ and $u$ come from the *same* GP, the exact
    model factors as $p(f) = \int p(f \mid u)\, p(u)\, du$, with no
    approximation yet: $u$ is just a slice of the same function.

    The approximation is a single assumption about that conditional.
    **FITC** (Fully Independent Training Conditional), the variant used
    below, replaces the exact $p(f \mid u)$, which has a full $n \times n$
    covariance, with a product of independent conditionals:

    $$p(f \mid u) \;\approx\; \prod_{i=1}^{n} p(f_i \mid u)$$

    which is exactly what its name describes. Every
    training point is allowed to talk to every other one **only through
    $u$**. The $m$ inducing values become a bottleneck that the whole
    dataset's information has to pass through.

    ### Why that makes the algebra cheap

    Writing $K_{nu}$ for the $n \times m$ cross-covariance and $K_{uu}$
    for the $m \times m$ covariance among inducing inputs, that assumption
    replaces the training covariance with its **Nyström** form

    $$K_{nn} \;\approx\; Q_{nn} = K_{nu}\,K_{uu}^{-1}\,K_{un}$$

    and FITC restores the exact variance on the diagonal:

    $$K_{\text{FITC}} = Q_{nn} + \operatorname{diag}\!\left(K_{nn} - Q_{nn}\right)$$

    $Q_{nn}$ is still $n \times n$, but it has **rank at most $m$**, and
    that is what the Woodbury identity needs. For any diagonal $\Lambda$,

    $$\left(Q_{nn} + \Lambda\right)^{-1} = \Lambda^{-1} - \Lambda^{-1} K_{nu} \left(K_{uu} + K_{un}\Lambda^{-1}K_{nu}\right)^{-1} K_{un}\Lambda^{-1}$$

    The only matrix inverted on the right is $m \times m$.

    `pm.gp.MarginalApprox` implements this for the conjugate
    (Gaussian-likelihood) case, with several approximation variants; the
    one used here is FITC.

    The inducing point *locations* can be fixed,
    chosen by a simple heuristic (e.g. k-means cluster centers over the
    training inputs, used below), or even learned as extra parameters ,
    we treat them as fixed here. Fixed has to mean *reproducibly* fixed:
    k-means starts from randomly drawn centroids, so its seed is set
    explicitly everywhere it is called below. Left unseeded, the same
    notebook gives a different set of inducing points, and a different
    fit, every time it is opened.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### The approximation, visualized

    Before fitting FITC to real tide data, look at what a sparse
    approximation actually does to a covariance matrix. Inducing points are
    **auxiliary variables**, not observations, extra locations $X_u$ the
    model is free to place anywhere in the input space, used only to build a
    compressed stand-in for the full $n \times n$ covariance. Below, a
    denser simulated dataset ($n=2{,}000$, past where exact inference is
    comfortable) stands in for "many observations"; a subset of $n=200$ of
    its points keeps the covariance matrices in the figure a readable size.
    """)
    return


@app.cell
def _():
    def simulate_dense_gp_data(seed, n=2000):
        """A dense simulated dataset matching the 2,000-observation tide
        slice below, used only to show the covariance approximation."""
        rng = np.random.default_rng(seed)
        X = 10 * np.sort(rng.uniform(size=n))[:, None]
        ls_true, eta_true = 1.0, 3.0
        cov_func = eta_true**2 * pm.gp.cov.Matern52(1, ls_true)
        K = cov_func(X).eval() + 1e-8 * np.eye(n)
        f_true = rng.multivariate_normal(np.zeros(n), K).flatten()
        sigma_true = 2.0
        y = f_true + sigma_true * rng.standard_normal(n)
        return X, f_true, y

    indviz_X, indviz_f_true, indviz_y = simulate_dense_gp_data(RANDOM_SEED)
    return (indviz_X,)


@app.cell(hide_code=True)
def _(indviz_X):
    indviz_rng = np.random.default_rng(RANDOM_SEED)
    indviz_idx = np.sort(indviz_rng.choice(indviz_X.shape[0], 200, replace=False))
    indviz_X_viz = indviz_X[indviz_idx]
    indviz_cov = 3.0**2 * pm.gp.cov.Matern52(1, 1.0)
    indviz_K_full = indviz_cov(indviz_X_viz).eval()

    indviz_Xu_poor = np.linspace(indviz_X.min(), indviz_X.max(), 4)[:, None]
    # scipy's k-means ignores RANDOM_SEED, so seed= is passed to pin the centroids.
    indviz_Xu_enough = pm.gp.util.kmeans_inducing_points(20, indviz_X, seed=RANDOM_SEED)

    def low_rank_covariance_approx(cov_func, X, Xu):
        """The Nystrom low-rank reconstruction K_nm K_mm^-1 K_mn used by FITC."""
        Kmm = cov_func(Xu).eval()
        Knm = cov_func(X, Xu).eval()
        return Knm @ np.linalg.solve(Kmm + 1e-8 * np.eye(Kmm.shape[0]), Knm.T)

    indviz_configs = [
        ("Too few inducing points (m = 4)", indviz_Xu_poor),
        (f"Enough inducing points (m = {len(indviz_Xu_enough)})", indviz_Xu_enough),
    ]
    indviz_approxes = [
        low_rank_covariance_approx(indviz_cov, indviz_X_viz, Xu)
        for _, Xu in indviz_configs
    ]
    indviz_errors = [indviz_K_full - approx for approx in indviz_approxes]
    return indviz_K_full, indviz_approxes, indviz_configs, indviz_errors


@app.cell(hide_code=True)
def _(indviz_K_full, indviz_approxes, indviz_configs, indviz_errors):
    _vmax = float(max(indviz_K_full.max(), max(a.max() for a in indviz_approxes)))
    _vemax = float(max(np.abs(e).max() for e in indviz_errors))
    _n_configs = len(indviz_configs)
    indviz_fig = make_subplots(
        rows=_n_configs,
        cols=3,
        subplot_titles=[
            _title
            for _label, _Xu in indviz_configs
            for _title in (
                "Full covariance $K_{nn}$",
                _label,
                f"Error ({_label.split(' (')[0].lower()})",
            )
        ],
        vertical_spacing=0.12,
    )
    for _row, ((_label, _Xu), _approx, _err) in enumerate(
        zip(indviz_configs, indviz_approxes, indviz_errors), start=1
    ):
        _colorbar_y = 1 - (_row - 0.5) / _n_configs
        _colorbar_len = 0.8 / _n_configs
        indviz_fig.add_trace(
            go.Heatmap(
                z=indviz_K_full,
                zmin=0,
                zmax=_vmax,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(x=0.28, y=_colorbar_y, len=_colorbar_len),
            ),
            row=_row,
            col=1,
        )
        indviz_fig.add_trace(
            go.Heatmap(
                z=_approx,
                zmin=0,
                zmax=_vmax,
                colorscale="Viridis",
                showscale=False,
            ),
            row=_row,
            col=2,
        )
        indviz_fig.add_trace(
            go.Heatmap(
                z=_err,
                zmin=-_vemax,
                zmax=_vemax,
                colorscale="RdBu",
                showscale=True,
                colorbar=dict(x=1.0, y=_colorbar_y, len=_colorbar_len),
            ),
            row=_row,
            col=3,
        )
    indviz_fig.update_layout(
        title="Sparse (Nystrom) covariance approximations",
        template="plotly_white",
        height=560,
    )
    indviz_fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Notice that with only 4 inducing points, the low-rank
    reconstruction misses covariance structure almost everywhere off the
    diagonal, the error panel is large and structured, not noise.

    With 20 K-means-located inducing points, the reconstruction is close to the full
    matrix and the error panel is small and unstructured; a smooth GP's
    covariance really is close to low-rank, so a modest $m$ recovers most of
    it.

    One problem remains even in the good case: the diagonal of a
    low-rank matrix $K_{nm}K_{mm}^{-1}K_{mn}$ is *smaller* than the diagonal
    of the true $K_{nn}$, it throws away some of every point's own
    variance, so plugging it directly into a Gaussian likelihood would
    understate the noise. This is what the FITC diagonal correction
    introduced above is for: it adds back the missing variance as
    an independent per-point term,
    $\max(\operatorname{diag}(K_{nn}-K_{nm}K_{mm}^{-1}K_{mn}), 0)$, so the
    approximate covariance stays valid and appropriately uncertain rather
    than overconfident. Nothing in `build_sparse_gp_model` below spells this
    out: `pm.gp.MarginalApprox(..., approx="FITC")` applies the correction
    internally, which is the whole difference between passing `"FITC"` and
    passing `"DTC"`.
    """)
    return


@app.cell(hide_code=True)
def _():
    N_SPARSE = 2000
    sparse_tides = pl.read_csv(data_dir / "noaa_tides_hourly.csv")
    sparse_tides = sparse_tides.with_columns(
        pl.col("time").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M")
    )
    sparse_slice = sparse_tides.slice(3000, N_SPARSE)
    sparse_slice.head()
    return N_SPARSE, sparse_slice


@app.cell(hide_code=True)
def _(sparse_slice):
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
def _(X_sparse, sparse_hours_std):
    N_INDUCING = 25
    SPARSE_SEMI_PERIOD_HOURS = 12.42
    SPARSE_DIURNAL_PERIOD_HOURS = 23.93
    SPARSE_PERIODIC_LS_STD = 0.5
    sparse_semi_period_std = SPARSE_SEMI_PERIOD_HOURS / sparse_hours_std
    sparse_diurnal_period_std = SPARSE_DIURNAL_PERIOD_HOURS / sparse_hours_std
    # Seeded: see the note at the covariance-approximation figure above.
    Xu_init = pm.gp.util.kmeans_inducing_points(N_INDUCING, X_sparse, seed=RANDOM_SEED)
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
    sparse_diurnal_period_std,
    sparse_semi_period_std,
):

    def build_sparse_gp_model(X, y, Xu):
        coords = {
            "obs": np.arange(len(X)),
            "feature": ["standardized time"],
            "inducing": np.arange(len(Xu)),
        }
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
        return model, gp_sparse

    return (build_sparse_gp_model,)


@app.cell
def _(X_sparse, Xu_init, build_sparse_gp_model, y_sparse):
    sparse_model, sparse_gp = build_sparse_gp_model(X_sparse, y_sparse, Xu_init)
    assert np.isfinite(sparse_model.compile_logp()(sparse_model.initial_point()))
    return sparse_gp, sparse_model


@app.cell(hide_code=True)
def _(N_INDUCING, N_SPARSE):
    mo.md(f"""
    ### Sampling

    The FITC likelihood only ever touches $n \\times m$ and
    $m \\times m$ matrices ($m={N_INDUCING}$ inducing points), so
    it can fit $n={N_SPARSE:,}$ hourly observations where exact GP
    inference is infeasible.
    """)
    return


@app.cell
def _(sparse_model):
    with sparse_model:
        sparse_start = perf_counter()
        sparse_idata = pm.sample(draws=500, tune=500, chains=4, random_seed=RANDOM_SEED)
        sparse_sample_seconds = perf_counter() - sparse_start
    print(f"Sparse FITC-GP sampling wall-time: {sparse_sample_seconds:.1f}s")
    return (sparse_idata,)


@app.cell(hide_code=True)
def _(sparse_idata, sparse_model):
    sparse_summary, sparse_health_passed = inference_health(sparse_idata, sparse_model)
    sparse_n_div = sparse_summary.attrs["divergences"]
    sparse_summary.round(4)
    return sparse_health_passed, sparse_n_div


@app.cell(hide_code=True)
def _(sparse_health_passed, sparse_n_div):
    mo.md(f"""
    **FITC inference health:** {sparse_n_div} divergences; all free
    variables meet the stated ESS/R-hat thresholds: **{sparse_health_passed}**.
    """)
    return


@app.cell
def _(X_sparse, sparse_gp, sparse_idata, sparse_model):
    with sparse_model:
        sparse_model.add_coords({"prediction": np.arange(len(X_sparse))})
        sparse_gp.conditional(
            "f_sparse_latent", X_sparse, pred_noise=False, dims="prediction"
        )
        sparse_gp.conditional(
            "f_sparse_noisy", X_sparse, pred_noise=True, dims="prediction"
        )
        sparse_predictions = pm.sample_posterior_predictive(
            posterior_subset(sparse_idata),
            var_names=["f_sparse_latent", "f_sparse_noisy"],
            predictions=True,
            random_seed=RANDOM_SEED,
        )
    return (sparse_predictions,)


@app.cell(hide_code=True)
def _(sparse_level_mean, sparse_level_std, sparse_predictions):
    sparse_fit = sparse_predictions["predictions"]["f_sparse_latent"]
    sparse_fit_vals = sparse_fit * sparse_level_std + sparse_level_mean
    sparse_fit_interval = eti(sparse_fit_vals)
    sparse_fit_mean = sparse_fit_vals.mean(("chain", "draw"))
    sparse_fit_lo = sparse_fit_interval.isel(quantile=0)
    sparse_fit_hi = sparse_fit_interval.isel(quantile=1)
    return sparse_fit_hi, sparse_fit_lo, sparse_fit_mean


@app.cell(hide_code=True)
def _(
    Xu_init,
    sparse_fit_hi,
    sparse_fit_lo,
    sparse_fit_mean,
    sparse_hours,
    sparse_level,
):
    plot_hours = 7 * 24
    plot_mask = sparse_hours <= plot_hours
    window_hours = sparse_hours[plot_mask]
    window_fit_hi = sparse_fit_hi.values[plot_mask]
    window_fit_lo = sparse_fit_lo.values[plot_mask]
    window_fit_mean = sparse_fit_mean.values[plot_mask]
    window_level = sparse_level[plot_mask]
    inducing_hours = Xu_init[:, 0] * sparse_hours.std(ddof=0) + sparse_hours.mean()
    visible_inducing_hours = inducing_hours[inducing_hours <= plot_hours]

    sparse_fit_fig = go.Figure()
    sparse_fit_fig.add_trace(
        go.Scatter(
            x=np.concatenate([window_hours, window_hours[::-1]]),
            y=np.concatenate([window_fit_hi, window_fit_lo[::-1]]),
            fill="toself",
            fillcolor="rgba(21,74,114,0.25)",
            line=dict(color="rgba(255,255,255,0)"),
            showlegend=False,
        )
    )
    sparse_fit_fig.add_trace(
        go.Scatter(
            x=window_hours,
            y=window_fit_mean,
            mode="lines",
            line=dict(color=PYMC_BLUE, width=2),
            showlegend=False,
        )
    )
    sparse_fit_fig.add_trace(
        go.Scatter(
            x=window_hours,
            y=window_level,
            mode="markers",
            marker=dict(color="black", size=4),
            showlegend=False,
        )
    )
    for inducing_hour in visible_inducing_hours:
        sparse_fit_fig.add_vline(
            x=inducing_hour,
            line=dict(color="rgba(190,85,0,0.9)", width=2, dash="dash"),
        )
    sparse_fit_fig.update_layout(
        title="FITC latent fit: first seven days",
        xaxis_title="Hours since slice start",
        yaxis_title="Water level (m, MLLW)",
        template="plotly_white",
        showlegend=False,
    )
    sparse_fit_fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The FITC fit captures the tide record without forming a
    $2{,}000 \times 2{,}000$ covariance matrix. The 25 inducing points
    summarize the 2,000-observation training set. Orange dashed lines mark
    the inducing locations within this seven-day window, making the fitted
    tidal cycles and their uncertainty readable. Sparse approximation trades a
    small, usually invisible amount of accuracy for a large reduction in cost.
    It does not scale indefinitely, though: even
    $O(nm^2)$ becomes expensive as $n$ grows with an $m$ that must itself grow
    to track increasingly complex structure, and every one of those $m$ points
    has to be placed. The HSGP section introduces an approximation whose cost
    is *linear* in $n$ and needs no inducing points at all.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Hilbert-space GP (HSGP) approximation

    ### The computational wall

    Sparse approximation dropped the per-gradient cost from $O(n^3)$ to
    $O(nm^2)$, a big win, but it still requires choosing and placing $m$
    inducing points, and $m$ itself has to grow as $n$ grows into the
    thousands and the underlying structure gets more complex. The
    2,412-batter swing-decision fit below is past where that tradeoff stays
    comfortable. The **Hilbert-space GP** (HSGP; Solin & Sarkka 2020,
    Riutort-Mayol et al. 2023) attacks the problem from the other side of
    the identity above: instead of approximating the covariance *matrix*,
    it works in weight space directly, and its cost is *linear* in $n$.

    ### The idea

    HSGP approximates the covariance **function** itself by its spectral
    (Fourier-like)
    decomposition on a bounded domain: a GP with a stationary kernel is
    rewritten as a weighted sum of a fixed set of basis functions (Laplace
    eigenfunctions of the domain, independent of any kernel hyperparameter
    and independent of any inducing points we would otherwise have to
    place) with random Gaussian weights, whose variances are set by the
    kernel's power spectral density (which *does* depend on the
    hyperparameters).
    """)
    return


@app.cell(hide_code=True)
def _():
    def plot_basis_functions():
        basisdemo_L = 5.0
        basisdemo_x = np.linspace(-basisdemo_L, basisdemo_L, 300)
        basisdemo_colors = [
            PYMC_BLUE,
            PYMC_GREEN,
            PYMC_LIGHT_BLUE,
            PYMC_DARK_GREEN,
            "#C2C240",
        ]

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                "First 5 basis functions",
                "Weighted sum of 20 basis → smooth function",
            ),
        )
        for j, color in zip(range(1, 6), basisdemo_colors):
            phi_j = np.sin(
                np.pi * j * (basisdemo_x + basisdemo_L) / (2 * basisdemo_L)
            ) / np.sqrt(basisdemo_L)
            fig.add_trace(
                go.Scatter(
                    x=basisdemo_x,
                    y=phi_j,
                    mode="lines",
                    name=f"φ_{j}",
                    line=dict(color=color, width=2),
                ),
                row=1,
                col=1,
            )

        basisdemo_rng = np.random.default_rng(RANDOM_SEED)
        basisdemo_weights = basisdemo_rng.normal(0, 1, 20) * np.exp(
            -0.3 * np.arange(20)
        )
        basisdemo_f_approx = np.zeros_like(basisdemo_x)
        for j in range(20):
            phi_j = np.sin(
                np.pi * (j + 1) * (basisdemo_x + basisdemo_L) / (2 * basisdemo_L)
            ) / np.sqrt(basisdemo_L)
            basisdemo_f_approx += basisdemo_weights[j] * phi_j
        fig.add_trace(
            go.Scatter(
                x=basisdemo_x,
                y=basisdemo_f_approx,
                mode="lines",
                line=dict(color=PYMC_BLUE, width=2.5),
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        fig.update_xaxes(title_text="x", row=1, col=1)
        fig.update_xaxes(title_text="x", row=1, col=2)
        fig.update_yaxes(title_text="φⱼ(x)", row=1, col=1)
        fig.update_yaxes(title_text="f(x)", row=1, col=2)
        fig.update_layout(
            template="plotly_white",
            title="HSGP basis functions: fixed shapes, learned weights",
        )
        return fig

    plot_basis_functions()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### How does it work?

    $$f \sim \mathcal{GP}\bigl(0,\, k(x, x'; \ell)\bigr) \;\longrightarrow\; f \approx \phi(x)\,\beta(\ell)$$

    The two panels above show what that expansion means concretely: the
    left panel is the fixed basis $\phi_j(x)$, sinusoidal-like functions
    of increasing frequency that depend only on the domain, never on a
    kernel hyperparameter. The right panel is a single draw of
    $f(x) = \sum_j \phi_j(x)\,\beta_j$, a weighted sum of 20 such basis
    functions with weights $\beta_j$ shrinking geometrically, smoother
    functions correspond to weights concentrated on the low-frequency
    (small-$j$) basis functions, which is exactly what a longer kernel
    lengthscale does to the spectral density in a real model. Because the
    basis functions don't change during sampling, the model is essentially
    a **linear-in-parameters** regression on those weights, cost grows
    **linearly** in $n$, not cubically, since evaluating the fixed basis at
    $n$ points is just an $O(nm)$ matrix multiply. This also makes HSGP a
    genuinely *parametric* model: once the basis is precomputed there is no
    `.conditional` step the way `Marginal` or `MarginalApprox` need ,
    `gp.prior()` hands you $f$ directly, a drop-in component for any
    likelihood, exactly like `pm.gp.Latent`. The tradeoff: the number of
    basis functions $m$ (and the domain boundary) must be chosen so the
    approximation is accurate over the lengthscales the kernel actually
    needs to represent, get $m$ or the boundary wrong and the
    approximation degrades, as the next section shows directly.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### How `L` and `c` reshape the basis

    The basis vectors above are sinusoids on an interval $[-L, L]$, which
    means they are forced to *pinch to zero* at the boundary. If $L$ sits
    right at the edge of the data, the approximation degrades there. PyMC
    builds $L$ from the data half-range $S = \max|x|$ and a boundary
    multiplier $c$:

    $$L = c \cdot S.$$

    Rather than describe what that does, drive it. The figure below is live:
    $m$ sets how many basis functions exist, $c$ sets how far past the data
    the domain extends. Work through these three settings in order and watch
    the edges, and the *first* basis function in particular.
    """)
    return


@app.cell(hide_code=True)
def _():
    m_slider = mo.ui.slider(5, 40, value=5, step=1, label="m (# basis functions)")
    c_slider = mo.ui.slider(1.0, 4.0, value=1.0, step=0.1, label="c (boundary factor)")
    return c_slider, m_slider


@app.cell
def _(c_slider, m_slider):
    hsgp_demo_grid = np.linspace(-3, 3, 300).reshape(-1, 1)
    hsgp_demo_centered = hsgp_demo_grid - hsgp_demo_grid.mean(axis=0)
    hsgp_demo_L = c_slider.value * np.max(np.abs(hsgp_demo_centered), axis=0)
    hsgp_demo_eigvals = pm.gp.hsgp_approx.calc_eigenvalues(
        hsgp_demo_L, [m_slider.value]
    )
    hsgp_demo_phi_vals = np.sin(
        np.sqrt(hsgp_demo_eigvals[:, 0])[None, :]
        * (hsgp_demo_centered[:, 0, None] + hsgp_demo_L[0])
    ) / np.sqrt(hsgp_demo_L[0])
    return hsgp_demo_grid, hsgp_demo_phi_vals


@app.cell(hide_code=True)
def _(c_slider, hsgp_demo_grid, hsgp_demo_phi_vals, m_slider):
    hsgp_basis_fig = go.Figure()
    _n_basis = hsgp_demo_phi_vals.shape[1]
    for _j in range(_n_basis):
        _shade = _j / max(_n_basis - 1, 1)
        hsgp_basis_fig.add_trace(
            go.Scatter(
                x=hsgp_demo_grid[:, 0],
                y=hsgp_demo_phi_vals[:, _j],
                mode="lines",
                name=f"basis {_j + 1}",
                showlegend=False,
                line=dict(
                    color=f"rgb({int(21 + 108 * _shade)},{int(74 + 120 * _shade)},{int(114 - 50 * _shade)})",
                    width=1.2,
                ),
            )
        )
    hsgp_basis_fig.update_layout(
        title=(
            f"All {_n_basis} HSGP Laplace eigenfunctions "
            f"(m={m_slider.value}, c={c_slider.value})"
        ),
        xaxis_title="x (standardized)",
        yaxis_title="φⱼ(x)",
        template="plotly_white",
    )

    # Controls render with the figure so their effect is visible at a glance.
    mo.vstack([mo.hstack([m_slider, c_slider], gap=2), hsgp_basis_fig])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Try these three, in order.** Leave $m$ at 5 so the low-frequency basis
    functions stay easy to see, and move only $c$.

    1. **$c = 1.0$.** The domain boundary sits exactly at the edge of the
       data. Every basis function is pinned to zero right where your data
       are, so the approximation degrades at both edges, and how badly
       depends on the lengthscale. This is why $c = 1$ is not a usable
       setting.
    2. **$c = 1.2$.** The Riutort-Mayol *et al.* recommended minimum. The
       pinch is pushed just outside the data range, which is usually enough
       to keep it from distorting the fit.
    3. **$c = 4.0$.** The boundary is far away, which accommodates long
       lengthscales, but the *period* of every basis function has grown, so
       the same $m$ now resolves far less detail inside the data range. Watch
       the first basis function in particular: it flattens almost completely.
       In that regime it can become unidentifiable with a model intercept,
       which is why `drop_first=True` matters whenever the model carries its
       own intercept.

    Now raise $m$ while holding $c = 4.0$ and watch the short-wavelength
    functions come back.

    Those are the two knobs, and this is the whole tradeoff between them.
    **`m`** is the number of basis functions; **`c`** is the boundary
    extension factor, with $L = c \cdot S$. Increasing $m$ lets the HSGP
    approximate GPs with **smaller lengthscales**, at the cost of
    computation. Increasing $c$ (or $L$) lets it approximate **larger
    lengthscales**, but spreads the same basis over a wider domain, so it
    generally requires a larger $m$ to compensate. The two therefore have to
    be raised together.
    """)
    return


@app.cell(hide_code=True)
def _(c_slider, m_slider):
    mo.md(f"""
    **Reading the basis functions above:** with **m = {m_slider.value}**
    basis functions and boundary factor **c = {c_slider.value}**, the
    approximation domain extends to ±{c_slider.value:.1f} × max\\|x\\| ,
    beyond the observed data range, which is required so the boundary
    condition (the basis functions are pinned to exactly zero at the
    edges) doesn't distort the fit near the edges of the actual data.
    Each `φⱼ` above is a fixed sine-like wave on that extended
    domain, of increasing frequency, entirely independent of any
    covariance hyperparameter. What *does* depend on the kernel and its
    hyperparameters is only the **weight** given to each basis function
    (its power spectral density) when they are summed to build the GP
    prior: a short lengthscale puts more weight on the high-frequency
    (wigglier) basis functions, a long one concentrates weight on the
    low-frequency (smooth) ones. That is the division of labour worth
    holding on to: the sliders change the basis, the kernel changes only
    how much each basis function counts.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Constraints of the HSGP approximation

    - Only works for **stationary** covariance functions with a known
      power spectral density (`Matern52`, `ExpQuad`, `Matern32`, ... ,
      not `Linear`, which is not stationary).
    - Officially supported for input dimension **up to 3**, the
      number of basis functions needed to cover a fixed-density grid
      grows exponentially with dimension, so HSGP stops being
      efficient well before that in practice for high-dimensional
      inputs.
    - The **`Periodic`** kernel is *not* directly supported by
      `pm.gp.HSGP` (it has no ordinary power-spectral-density
      expansion), PyMC provides a separate `pm.gp.HSGPPeriodic` class
      using a different low-rank basis for periodic structure.
    - Accuracy depends on `m` and the boundary factor `c` (or an
      explicit boundary `L`) both being large enough for the
      lengthscales actually present in the data, as explored above and
      in the question below.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Swing decisions by age, revisited

    Notebook 1 fit a **hand-built spline basis** to 2023 batter
    swing-decision grades against age: choose a degree, place the knots at
    quantiles of age, build the design matrix, put a prior on the weights.
    The same data returns here, fit with a **spectral basis** instead. The
    comparison is the point. A spline and an HSGP are both linear models in
    a fixed basis, $f(x) = \sum_j \phi_j(x)\,\beta_j$ in both cases ,
    and the difference is where the basis comes from and what sets the
    weights. The spline's knots were a modelling choice made by hand; the
    HSGP's basis is the Laplace eigenfunctions of the domain, and the
    weights' prior variances come from the kernel's spectral density, so
    the smoothness is controlled by a lengthscale with an interpretable
    prior rather than by a knot count.

    There are well over 2,000 batter-season-levels here, which is past the
    point where an exact GP is comfortable and where FITC would still need
    inducing points chosen and placed.
    """)
    return


@app.cell(hide_code=True)
def _():
    # Same filter and columns as notebook 1's spline fit: identical data, two bases.
    swing_decisions = (
        pl.read_csv(data_dir / "batter_grades_2023.csv")
        .filter((pl.col("throws") == "R") & (pl.col("n_pa") > 100))
        .select("batter_id", "batter", "age", "swing_decision")
        .drop_nulls()
    )
    swing_ages_obs = swing_decisions["age"].to_numpy()
    swing_grades = swing_decisions["swing_decision"].to_numpy()
    swing_age_grid = np.sort(np.unique(swing_ages_obs))
    swing_age_index = np.searchsorted(swing_age_grid, swing_ages_obs)
    print(
        f"n = {len(swing_grades):,} batter-season-levels across "
        f"{len(swing_age_grid)} distinct ages "
        f"({swing_age_grid.min()}-{swing_age_grid.max()})."
    )
    return swing_age_grid, swing_age_index, swing_ages_obs, swing_grades


@app.cell(hide_code=True)
def _(swing_ages_obs, swing_grades):
    swing_data_fig = go.Figure()
    swing_data_fig.add_trace(
        go.Scatter(
            x=swing_ages_obs,
            y=swing_grades,
            mode="markers",
            marker=dict(color=PYMC_BLUE, size=5, opacity=0.3),
            name="batter-season-levels",
        )
    )
    swing_data_fig.update_layout(
        title="Swing decision grade by age, 2023",
        xaxis_title="Age (years)",
        yaxis_title="Swing decision grade",
        template="plotly_white",
    )
    swing_data_fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Before choosing a basis we have to say what lengthscales the kernel is
    allowed to represent, because that, not $n$, and not taste, is what
    sets `m` and `c`. Age spans 17 to 40, so a lengthscale below about a year would be finer
    than whole-year data can resolve, and one much above 15 years would
    flatten the whole curve. We elicit on $\ell \in [1, 15]$ years.

    `pz.maxent` turns that stated interval into the **most diffuse**
    LogNormal consistent with it, maximum entropy subject to the
    constraint, so we are not smuggling in more information than the two
    bounds we can actually defend.
    """)
    return


@app.cell
def _():
    # LogNormal lengthscale prior with 90% of its mass between 1 and 15 years.
    SWING_ELL_LO, SWING_ELL_HI, SWING_ELL_MASS = 1.0, 15.0, 0.9
    swing_ell_prior = pz.maxent(
        pz.LogNormal(), SWING_ELL_LO, SWING_ELL_HI, SWING_ELL_MASS, plot=False
    )
    return SWING_ELL_HI, SWING_ELL_LO, swing_ell_prior


@app.cell(hide_code=True)
def _(swing_ell_prior):
    swing_ell_mu = float(swing_ell_prior.mu)
    swing_ell_sigma = float(swing_ell_prior.sigma)
    return swing_ell_mu, swing_ell_sigma


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Choosing the two knobs

    `m` and `c` are the same two knobs the sliders above drive. As rough
    orders of magnitude in 1-D, `m` is typically 20–200 and `c` typically
    1.3–4.0, and if the kernel's lengthscale is much longer than the data
    spacing a small `m` is fine.

    Rather than guess, `pm.gp.hsgp_approx.approx_hsgp_hyperparams` takes
    the data's x-range and a plausible lengthscale range and returns
    recommended `m` and `c` from the Riutort-Mayol et al. (2023)
    approximation-error bounds. It only needs the lengthscale *prior*, not
    the data itself, the choice is a modelling decision tied to what the
    kernel is allowed to represent, not a number to tune by trial and
    error after the fact.

    One related default worth knowing: `pm.gp.HSGP` uses
    `parametrization="noncentered"`, the same reparameterization trick
    that tames hierarchical funnels, applied here to decorrelate the basis
    coefficients from the scale set by the spectral density, which keeps
    NUTS from stalling when the noise level is uncertain.

    `drop_first=True` removes the lowest-frequency basis function. It is not
    a constant, the basis is a sine series indexed from the first nonzero
    mode, but over the observed range it is close enough to flat to trade
    off against a model intercept, so dropping it is worth doing whenever the
    model carries its own intercept. That is the case below.
    """)
    return


@app.cell
def _(SWING_ELL_HI, SWING_ELL_LO, swing_age_grid):
    swing_m, swing_c = pm.gp.hsgp_approx.approx_hsgp_hyperparams(
        x_range=[float(swing_age_grid.min()), float(swing_age_grid.max())],
        lengthscale_range=[SWING_ELL_LO, SWING_ELL_HI],
        cov_func="matern52",
    )
    print(
        f"approx_hsgp_hyperparams recommends m = {swing_m}, c = {swing_c:.2f} "
        f"for lengthscales in [{SWING_ELL_LO:.0f}, {SWING_ELL_HI:.0f}] years."
    )
    return swing_c, swing_m


@app.cell(hide_code=True)
def _(c_slider, m_slider):
    mo.md(f"""
    The sliders above are currently set to m = {m_slider.value}, "
        f"c = {c_slider.value}.
    """)
    return


@app.cell(hide_code=True)
def _(swing_age_grid):
    def plot_hsgp_parameter_curves(S, ell_grid):
        c_list = np.array([1.2, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0])
        fig = go.Figure()
        for c_val in c_list:
            m_curve = 2.65 * (c_val / ell_grid) * S
            valid = c_val >= (4.1 * (ell_grid / S))
            fig.add_trace(
                go.Scatter(
                    x=ell_grid,
                    y=np.where(valid, m_curve, np.nan),
                    mode="lines",
                    name=f"c = {c_val}",
                    line=dict(width=2),
                )
            )
        fig.update_yaxes(
            type="log",
            range=[np.log10(5), np.log10(1000)],
            title_text="number of basis functions (m)",
        )
        fig.update_xaxes(title_text="lengthscale (ℓ, years of age)")
        fig.update_layout(
            title="Matérn-5/2 HSGP approximation parameter curves",
            template="plotly_white",
        )
        return fig

    # S is the half-range of the centered inputs, which is what L = c * S uses.
    swing_param_S = float(np.abs(swing_age_grid - swing_age_grid.mean()).max())
    swing_param_ell_grid = np.linspace(0.5, 25.0, 500)
    plot_hsgp_parameter_curves(swing_param_S, swing_param_ell_grid)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Each curve marks where the HSGP approximation is reliable for that
    value of $c$: below the curve for a given $c$, the pair $(\ell, m)$ is
    in the valid region. **The right value depends on your prior over
    $\ell$**, not on $n$ or on taste, a curve is not "the answer," it's a
    map from a lengthscale prior to a basis-size requirement. Our elicited
    prior puts 90% of its mass on 1 to 15 years, and
    `approx_hsgp_hyperparams` did exactly this calculation to turn that
    range into the `m` and `c` the fit below actually uses. Smaller $m$ is
    cheaper; $c$ has no effect on cost, only on where the boundary pinch
    sits.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Fitting the HSGP

    `m` and `c` are now fixed by the lengthscale prior rather than chosen
    by hand, so the model is short. `gp.prior()` hands back $f$ evaluated
    on the age grid, there is no `.conditional` step, because the basis is
    the model. The GP is evaluated once per **distinct age** (24 values)
    and indexed out to the observations, which is both cheaper and a
    statement about the model: swing decision is a function of age alone,
    and everything else is observation noise.
    """)
    return


@app.cell
def _(
    swing_age_grid,
    swing_age_index,
    swing_c,
    swing_ell_mu,
    swing_ell_sigma,
    swing_grades,
    swing_m,
):
    def build_swing_hsgp_model(age_grid, age_index, grades, *, m, c, ell_mu, ell_sigma):
        coords = {
            "age": age_grid,
            "observation": np.arange(len(grades)),
            "feature": ["age"],
        }
        with pm.Model(coords=coords) as model:
            age_data = pm.Data(
                "age_grid", age_grid[:, None].astype(float), dims=("age", "feature")
            )
            age_index_data = pm.Data("age_index", age_index, dims="observation")
            grade_data = pm.Data("grade", grades, dims="observation")

            ell = pm.LogNormal("ell", mu=ell_mu, sigma=ell_sigma)
            eta = pm.Exponential("eta", lam=1)
            gp = pm.gp.HSGP(
                m=[m],
                c=c,
                cov_func=eta**2 * pm.gp.cov.Matern52(1, ls=ell),
                parametrization="noncentered",
                drop_first=True,
            )
            f = gp.prior("f", X=age_data, dims="age")

            intercept = pm.Normal("intercept", mu=float(grades.mean()), sigma=5)
            sigma = pm.HalfNormal("sigma", sigma=10)
            pm.Normal(
                "y",
                mu=intercept + f[age_index_data],
                sigma=sigma,
                observed=grade_data,
                dims="observation",
            )
        return model, gp

    swing_model, swing_gp = build_swing_hsgp_model(
        swing_age_grid,
        swing_age_index,
        swing_grades,
        m=swing_m,
        c=swing_c,
        ell_mu=swing_ell_mu,
        ell_sigma=swing_ell_sigma,
    )
    swing_model.compile_logp()(swing_model.initial_point())
    return (swing_model,)


@app.cell
def _(swing_model):
    with swing_model:
        swing_prior_pred = pm.sample_prior_predictive(
            draws=500, random_seed=RANDOM_SEED
        )
    return (swing_prior_pred,)


@app.cell
def _(swing_prior_pred):
    swing_prior_y = swing_prior_pred["prior_predictive"]["y"]
    swing_prior_lo, swing_prior_hi = np.quantile(swing_prior_y.values, [0.055, 0.945])
    return swing_prior_hi, swing_prior_lo, swing_prior_y


@app.cell(hide_code=True)
def _(swing_age_grid, swing_ages_obs, swing_grades, swing_prior_pred):
    swing_prior_fig = go.Figure()
    swing_prior_f = swing_prior_pred["prior"]["f"].stack(sample=("chain", "draw"))
    _rng_swing = np.random.default_rng(RANDOM_SEED)
    for _i in _rng_swing.choice(swing_prior_f.sizes["sample"], size=60, replace=False):
        swing_prior_fig.add_trace(
            go.Scatter(
                x=swing_age_grid,
                y=swing_prior_f.isel(sample=int(_i)).values,
                mode="lines",
                line=dict(color=PYMC_LIGHT_BLUE, width=1),
                opacity=0.25,
                showlegend=False,
            )
        )
    swing_prior_fig.add_trace(
        go.Scatter(
            x=swing_ages_obs,
            y=swing_grades,
            mode="markers",
            marker=dict(color="black", size=4, opacity=0.25),
            name="observed",
        )
    )
    swing_prior_fig.update_layout(
        title="Prior draws of the HSGP function f(age), against the observed grades",
        xaxis_title="Age (years)",
        yaxis_title="Swing decision grade",
        template="plotly_white",
    )
    swing_prior_fig
    return


@app.cell(hide_code=True)
def _(swing_grades, swing_prior_hi, swing_prior_lo, swing_prior_y):
    mo.md(f"""
    **Prior implications:** the middle 89% of prior predictive draws falls
    in [{swing_prior_lo:.2f}, {swing_prior_hi:.2f}], against an observed
    range of [{swing_grades.min():.2f}, {swing_grades.max():.2f}]. That is
    the right comparison to make: the outright minimum and maximum over
    {swing_prior_y.size:,} draws are
    [{float(swing_prior_y.min()):.1f}, {float(swing_prior_y.max()):.1f}],
    but those are extreme tail draws of the observation noise, not a
    statement about what the model believes. The interval brackets the data
    without the prior functions being asked to represent grades no batter
    could post. Reasonable to proceed to sampling.
    """)
    return


@app.cell
def _(swing_model):
    with swing_model:
        swing_start = perf_counter()
        swing_idata = pm.sample(
            draws=500,
            tune=1000,
            chains=4,
            target_accept=0.97,
            random_seed=RANDOM_SEED,
        )
        swing_sample_seconds = perf_counter() - swing_start
    print(f"HSGP swing-decision sampling wall-time: {swing_sample_seconds:.1f}s")
    return (swing_idata,)


@app.cell(hide_code=True)
def _(swing_idata, swing_model):
    swing_summary, swing_health_passed = inference_health(swing_idata, swing_model)
    swing_n_div = swing_summary.attrs["divergences"]
    swing_n_draws_total = (
        swing_idata["posterior"].sizes["chain"] * swing_idata["posterior"].sizes["draw"]
    )
    swing_min_ess_bulk = float(swing_summary["ess_bulk"].min())
    swing_min_ess_tail = float(swing_summary["ess_tail"].min())
    swing_max_rhat = float(swing_summary["r_hat"].astype(float).max())
    print(
        f"Divergences: {swing_n_div} / {swing_n_draws_total}; "
        f"health passed: {swing_health_passed}"
    )
    mo.ui.table(swing_summary.round(4), pagination=True, page_size=10)
    return (
        swing_health_passed,
        swing_max_rhat,
        swing_min_ess_bulk,
        swing_min_ess_tail,
        swing_n_div,
        swing_n_draws_total,
    )


@app.cell(hide_code=True)
def _(
    swing_age_grid,
    swing_grades,
    swing_health_passed,
    swing_m,
    swing_max_rhat,
    swing_min_ess_bulk,
    swing_min_ess_tail,
    swing_n_div,
    swing_n_draws_total,
):
    mo.md(f"""
    **Inference health:** {swing_n_div} of {swing_n_draws_total} draws
    diverged, maximum `r_hat` {swing_max_rhat:.3f}, minimum `ess_bulk`
    {swing_min_ess_bulk:.0f} and minimum `ess_tail`
    {swing_min_ess_tail:.0f} (`inference_health` threshold check:
    **{swing_health_passed}**, an advisory flag, not a pass/fail gate).
    What the approximation bought here is the interface, not the size. The
    {len(swing_grades):,} observations sit on only {len(swing_age_grid)}
    distinct ages, so an exact `gp.Latent` over those ages would need a
    {len(swing_age_grid)} × {len(swing_age_grid)} covariance, smaller than
    this {swing_m}-coefficient basis. Read the fit as a worked example of
    setting `m` and `c` from a lengthscale prior with no inducing points to
    place; the cost argument only starts paying when the inputs are
    genuinely continuous, as in the 2-D example below. The tight
    `target_accept=0.97` is the same point from the sampler's side: the
    basis is large relative to the structure the data can support.
    """)
    return


@app.cell
def _(swing_idata, swing_model):
    with swing_model:
        swing_ppc = pm.sample_posterior_predictive(
            posterior_subset(swing_idata), var_names=["y"], random_seed=RANDOM_SEED
        )
    swing_idata_with_ppc = swing_idata.copy()
    swing_idata_with_ppc["posterior_predictive"] = swing_ppc["posterior_predictive"]

    swing_curve = swing_idata["posterior"]["intercept"] + swing_idata["posterior"]["f"]
    swing_curve_mean = swing_curve.mean(("chain", "draw")).values
    swing_curve_eti = eti(swing_curve)
    return swing_curve_eti, swing_curve_mean, swing_idata_with_ppc


@app.cell(hide_code=True)
def _(
    swing_age_grid,
    swing_ages_obs,
    swing_curve_eti,
    swing_curve_mean,
    swing_grades,
):
    swing_fit_fig = go.Figure()
    swing_fit_fig.add_trace(
        go.Scatter(
            x=np.concatenate([swing_age_grid, swing_age_grid[::-1]]),
            y=np.concatenate(
                [
                    swing_curve_eti.isel(quantile=1).values,
                    swing_curve_eti.isel(quantile=0).values[::-1],
                ]
            ),
            fill="toself",
            fillcolor="rgba(21,74,114,0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="89% ETI",
        )
    )
    swing_fit_fig.add_trace(
        go.Scatter(
            x=swing_ages_obs,
            y=swing_grades,
            mode="markers",
            marker=dict(color="black", size=4, opacity=0.2),
            name="observed",
        )
    )
    swing_fit_fig.add_trace(
        go.Scatter(
            x=swing_age_grid,
            y=swing_curve_mean,
            mode="lines",
            line=dict(color=PYMC_BLUE, width=3),
            name="posterior mean",
        )
    )
    swing_fit_fig.update_layout(
        title="HSGP posterior for swing decision as a function of age",
        xaxis_title="Age (years)",
        yaxis_title="Swing decision grade",
        template="plotly_white",
    )
    swing_fit_fig
    return


@app.cell(hide_code=True)
def _(swing_ages_obs):
    mo.md(f"""
    The posterior mean rises steeply through the early twenties and
    flattens after that, with the interval widening at both ends where the
    data thin out, {int((swing_ages_obs >= 35).sum())} of
    {len(swing_ages_obs):,} batter-season-levels are 35 or older. That
    widening is the GP saying what it does not know, and it is the part a
    fixed spline basis has to be coaxed into expressing.

    Notebook 1's caveats about these data still apply, and the HSGP does
    nothing to fix them: this is one season's snapshot, players who stop
    hitting stop appearing, and competition level is very nearly a proxy
    for age at the young end. What changed is only the basis, not the
    identification problem.
    """)
    return


@app.cell
def _(swing_idata_with_ppc):
    az.plot_ppc_dist(swing_idata_with_ppc, var_names=["y"], kind="ecdf", num_samples=50)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The replicated ECDFs bracket the observed one across the whole range of
    grades, which says the Normal observation layer and the fitted age
    curve together reproduce the marginal distribution of the data. That is
    a weak check on its own, it would look much the same for a model with
    no age structure at all, because almost all of the variance here *is*
    observation noise, so read it together with the posterior curve above
    rather than as a verdict on the age effect.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Question: assess an HSGP basis approximation

    The fit above used the `m` and `c` that `approx_hsgp_hyperparams`
    derived from a 1-to-15-year lengthscale prior. Before running anything:

    1. Use the `m` and `c` sliders further up to set a basis far smaller
       than the recommended one. Reading the basis-function figure alone,
       what structure in the age curve could that basis *not* represent?
    2. Predict what `approx_hsgp_hyperparams` returns if you *narrow* the
       elicited range to 3-8 years, and what that does to the sampling
       cost. Then set `SWING_ELL_LO = 3.0` and `SWING_ELL_HI = 8.0` and
       read both the printed recommendation and the diagnostics.
       (Change them back afterwards.)
    3. Which of the two knobs would you expect to change the sampling
       *cost*, and which only the boundary behaviour?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.accordion(
        {
            "Hint": mo.md(
                r"""
                The two knobs have different jobs. `m` sets how many
                frequencies the basis carries, so it bounds the *shortest*
                lengthscale the approximation can represent. `c` sets how
                far past the data the boundary, where every basis function
                is pinned to zero, is pushed, so it governs behaviour at
                the edges of the age range and the *longest* lengthscale
                that fits inside the domain. Only one of them adds
                parameters to sample.
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.accordion(
        {
            "Discussion": mo.md(
                r"""
                1. A basis with too few functions carries only low
                frequencies, so it can trace the broad rise across the
                twenties but not any sharper feature, a kink at a
                particular age, or a late decline confined to a few years,
                would be smoothed away. Crucially it would not *look*
                broken: you would get a smooth, confident curve that had
                silently ruled out the structure you were asking about.

                2. Narrowing the range to 3-8 years drops the
                recommendation from **m = 162** to **m = 28**, because the
                basis size is driven by the *shortest* lengthscale the
                prior allows. That is a large cut in the number of
                coefficients NUTS has to sample, and the fit gets easier
                for it. The point is not that narrower is always better ,
                it is that the recommendation is only ever as good as the
                lengthscale range you feed it, so the range has to be one
                you can defend against the data. Here ages are whole
                numbers spanning 23 years, which is worth weighing against
                a prior that reaches down to one year.

                3. `m` is the cost knob, it is literally the number of
                basis coefficients sampled. `c` costs nothing; it only
                moves the boundary pinch, and increasing it without raising
                `m` spreads the same basis functions over a wider domain,
                coarsening resolution inside the data range.
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### When NOT to reach for HSGP

    The swing-decision fit above is a genuine success, but HSGP is not a
    universal replacement for the exact and sparse GPs above. Two of
    its restrictions are already implicit in the constraints
    listed above, stationary kernels only, and a basis count that grows
    quickly with input dimension. Two more are worth stating before
    reaching for it on the next problem:

    - **Rapidly-varying processes are expensive.** If $f$ changes
      quickly relative to the width of the domain, matching that with
      basis functions can require an impractically large $m$.
    - **Small data: prefer the exact GP.** For a few hundred
      observations, exact `pm.gp.Marginal` is simple, fast enough, and
      removes an entire source of approximation error, there is no
      reason to pay HSGP's setup cost.

    The example below pushes HSGP to the edge of where it remains
    practical: two input dimensions instead of one.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Multi-input GPs: called-strike probability

    Every GP fit in *this* notebook has had a single input, time or player
    age. Multiple inputs are not new, Notebook 3's PLACES model already put
    a GP over two spatial coordinates with a separate lengthscale per axis.
    What changes under HSGP is the basis: it becomes a product over the
    axes, so the coefficient count is the product of the per-axis $m$
    values rather than their sum.

    A natural baseball application is **called-strike probability**:
    given where a pitch crosses the plate, $(x, z)$, what is the
    probability an umpire calls it a strike? This is the course's only
    classification example, a Bernoulli likelihood driven by the
    HSGP's latent function through `invlogit`, and its only 2-D
    approximation, which is close to where HSGP's practical usefulness
    ends (the constraints above: officially supported up to three input
    dimensions, with the basis count growing quickly in each).
    """)
    return


@app.cell(hide_code=True)
def _():
    strike_pitches = pl.read_csv(data_dir / "taken_pitches_walker.csv")
    called_strike = strike_pitches.select(
        ["bats", "location_x", "location_z", "is_strike"]
    )
    called_strike.head()
    return (called_strike,)


@app.cell(hide_code=True)
def _(called_strike):
    strike_plot_fig = go.Figure()
    for _flag, _color, _label in [
        (0, PYMC_LIGHT_BLUE, "ball"),
        (1, PYMC_GREEN, "strike"),
    ]:
        _sub = called_strike.filter(pl.col("is_strike") == _flag)
        strike_plot_fig.add_trace(
            go.Scatter(
                x=_sub["location_x"],
                y=_sub["location_z"],
                mode="markers",
                marker=dict(size=6, color=_color, opacity=0.6),
                name=_label,
            )
        )
    strike_plot_fig.add_shape(
        type="rect",
        x0=-0.83,
        x1=0.83,
        y0=1.4,
        y1=3.4,
        line=dict(color="black"),
    )
    strike_plot_fig.update_layout(
        title=f"Taken pitches (n={called_strike.height:,}): calls by location",
        xaxis_title="location_x (ft from plate center)",
        yaxis_title="location_z (ft above ground)",
        yaxis=dict(scaleanchor="x", scaleratio=1, constrain="domain"),
        template="plotly_white",
        width=450,
        height=420,
        legend=dict(
            title=dict(text="Call"),
            orientation="h",
            x=0.5,
            xanchor="center",
            y=1.02,
            yanchor="bottom",
        ),
    )
    return (strike_plot_fig,)


@app.cell(hide_code=True)
def _(called_strike):
    strike_X = called_strike.select(["location_x", "location_z"]).to_numpy()
    strike_y = called_strike["is_strike"].to_numpy().astype(int)
    strike_m = (25, 25)
    strike_X_center = (strike_X.max(axis=0) + strike_X.min(axis=0)) / 2
    strike_X_centered = strike_X - strike_X_center
    strike_L = 4.0 * (strike_X_centered.max(axis=0) - strike_X_centered.min(axis=0)) / 2
    print(
        f"strike_X shape: {strike_X.shape}; basis functions: {int(np.prod(strike_m))}"
    )
    print(f"strike_L (boundary half-widths): {strike_L}")
    return strike_L, strike_X_center, strike_X_centered, strike_m, strike_y


@app.function
def hsgp_posterior_mean_probability(X, L, m, basis_weights, chunk_size=50):
    """Posterior-mean invlogit(f) from linearized HSGP weights, batched in numpy.

    The HSGP is linear in its basis weights, so the fixed grid basis is
    evaluated once in numpy and invlogit(f) is averaged over posterior
    draws in chunks. This is far cheaper than re-running PyTensor once
    per draw, and avoids materializing a full draw-by-grid
    posterior-predictive array.
    """
    eigvals = pm.gp.hsgp_approx.calc_eigenvalues(np.asarray(L), list(m))
    phi = np.ones((X.shape[0], int(np.prod(m))))
    for dim in range(len(m)):
        phi *= np.sin(
            np.sqrt(eigvals[:, dim])[None, :] * (X[:, dim, None] + L[dim])
        ) / np.sqrt(L[dim])

    probability_sum = np.zeros(X.shape[0])
    for start in range(0, basis_weights.shape[0], chunk_size):
        batch = basis_weights[start : start + chunk_size]
        logits = batch @ phi.T
        probability_sum += (1.0 / (1.0 + np.exp(-logits))).sum(axis=0)
    return probability_sum / basis_weights.shape[0]


@app.cell
def _(strike_L, strike_X_centered, strike_m, strike_y):
    def build_called_strike_model(X_centered, y, m, L):
        coords = {
            "basis": np.arange(int(np.prod(m))),
            "obs": np.arange(y.size),
        }
        with pm.Model(coords=coords) as called_strike_model:
            ls = pm.Gamma("ls", alpha=2, beta=2, shape=2)
            eta = pm.HalfNormal("eta", sigma=2)
            cov = eta**2 * pm.gp.cov.Matern52(2, ls=ls)

            gp = pm.gp.HSGP(m=list(m), L=list(L), cov_func=cov)
            phi, sqrt_psd = gp.prior_linearized(X_centered)
            f_hsgp_coeffs = pm.Normal("f_hsgp_coeffs", dims="basis")
            basis_weights = pm.Deterministic(
                "basis_weights", f_hsgp_coeffs * sqrt_psd, dims="basis"
            )
            f = pm.Deterministic("f", phi @ basis_weights, dims="obs")

            strike_prob = pm.math.invlogit(f)
            pm.Bernoulli("strike", p=strike_prob, observed=y, dims="obs")
        return called_strike_model

    called_strike_model = build_called_strike_model(
        strike_X_centered, strike_y, strike_m, strike_L
    )
    called_strike_model.compile_logp()(called_strike_model.initial_point())
    return (called_strike_model,)


@app.cell
def _(called_strike_model, strike_y):
    with called_strike_model:
        called_strike_prior_pred = pm.sample_prior_predictive(
            draws=500, var_names=["strike"], random_seed=RANDOM_SEED
        )
    print(
        "Prior-predictive mean strike rate:",
        float(called_strike_prior_pred["prior_predictive"]["strike"].mean()),
        "  Observed strike rate:",
        float(strike_y.mean()),
    )
    return


@app.cell
def _(called_strike_model):
    with called_strike_model:
        strike_start = perf_counter()
        strike_idata = pm.sample(
            draws=500,
            tune=1000,
            chains=4,
            target_accept=0.9,
            random_seed=RANDOM_SEED,
        )
        strike_sample_seconds = perf_counter() - strike_start
    print(f"Called-strike HSGP sampling wall-time: {strike_sample_seconds:.1f}s")
    return strike_idata, strike_sample_seconds


@app.cell(hide_code=True)
def _(called_strike_model, strike_idata):
    strike_summary, strike_health_passed = inference_health(
        strike_idata, called_strike_model
    )
    strike_n_div = strike_summary.attrs["divergences"]
    mo.ui.table(strike_summary.round(4), pagination=True, page_size=10)
    return strike_health_passed, strike_n_div


@app.cell(hide_code=True)
def _(strike_health_passed, strike_n_div, strike_sample_seconds, strike_y):
    mo.md(f"""
    **Called-strike HSGP inference health:** {strike_n_div} divergences;
    all free variables meet the stated ESS/R-hat thresholds:
    **{strike_health_passed}**. Sampling took **{strike_sample_seconds:.0f}s**
    for a $25\times25 = 625$-basis-function HSGP over 2 input dimensions
    and {strike_y.size:,} pitches, the notebook's most expensive fit.
    Unlike the 1-D fits above, this one needs a raised
    `target_accept=0.9`: at PyMC's default acceptance target the sampler
    leaves divergences and mildly elevated r-hat on a handful of the 625
    basis-weight coefficients, which is the geometry cost of a basis this
    large relative to the data.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    A key step in visualizing the fit is predicting over a 2-D grid.
    Because the HSGP is linear in its basis weights, each draw's
    effective weights (`basis_weights`, saved above) were kept during
    fitting; the fixed grid basis is evaluated once in NumPy, and
    `invlogit(f)` is averaged over every retained posterior draw. This
    avoids draw-by-draw PyTensor evaluation and avoids materializing a
    full draw-by-grid posterior-predictive array.
    """)
    return


@app.cell(hide_code=True)
def _(strike_L, strike_X_center, strike_idata, strike_m):
    strike_grid_x = np.linspace(-1.5, 1.5, 100)
    strike_grid_z = np.linspace(0.5, 4.5, 100)
    strike_grid_xx, strike_grid_zz = np.meshgrid(strike_grid_x, strike_grid_z)
    strike_X_grid = np.column_stack([strike_grid_xx.ravel(), strike_grid_zz.ravel()])

    strike_basis_weights = (
        strike_idata["posterior"]["basis_weights"]
        .stack(sample=("chain", "draw"))
        .transpose("sample", "basis")
        .values
    )
    strike_prob_grid = hsgp_posterior_mean_probability(
        strike_X_grid - strike_X_center,
        strike_L,
        strike_m,
        strike_basis_weights,
    )
    strike_prob_surface = strike_prob_grid.reshape(strike_grid_zz.shape)
    return strike_grid_x, strike_grid_z, strike_prob_surface


@app.cell(hide_code=True)
def _(strike_grid_x, strike_grid_z, strike_plot_fig, strike_prob_surface):
    strike_surface_fig = go.Figure()
    strike_surface_fig.add_trace(
        go.Heatmap(
            x=strike_grid_x,
            y=strike_grid_z,
            z=strike_prob_surface,
            colorscale="RdBu",
            reversescale=True,
            zmin=0,
            zmax=1,
            colorbar=dict(title="P(strike)"),
        )
    )
    strike_surface_fig.add_shape(
        type="rect",
        x0=-0.83,
        x1=0.83,
        y0=1.4,
        y1=3.4,
        line=dict(color="black", width=2),
    )
    strike_surface_fig.update_layout(
        title="Posterior mean called-strike probability over the zone",
        xaxis_title="location_x (ft from plate center)",
        yaxis_title="location_z (ft above ground)",
        yaxis=dict(scaleanchor="x", scaleratio=1, constrain="domain"),
        template="plotly_white",
        width=450,
        height=420,
    )
    mo.hstack([strike_plot_fig, strike_surface_fig], gap=1, justify="center")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Prior draws, divergence/R-hat/ESS summaries and posterior-predictive
    comparisons are diagnostics, not a pre-declared adequacy verdict. Read
    each observed discrepancy against its replicated distribution: a value
    outside or near a tail keeps the relevant model limitation visible and
    should guide a basis or likelihood revision.

    **Out of scope for this course:** comparing models by predictive
    accuracy via `pm.compute_log_likelihood` + `az.loo`/`az.compare`
    (LOO-CV, ELPD) is a natural next step once you have more than one
    candidate model, but is beyond what we can cover in an
    introductory workshop, see the ArviZ documentation and the
    `model-evaluation` material for that workflow.

    ### Decision guide: exact vs. sparse vs. HSGP

    | | **Exact** (`pm.gp.Marginal` / `pm.gp.Latent`) | **Sparse** (`pm.gp.MarginalApprox`) | **HSGP** (`pm.gp.HSGP` / `HSGPPeriodic`) |
    |---|---|---|---|
    | **Cost** | $O(n^3)$ per gradient eval | $O(nm^2)$, $m$ = # inducing points | $O(nm)$, $m$ = # basis functions (linear in $n$) |
    | **When to use** | Small/moderate $n$ (up to a few hundred–low thousands); need exact inference | Moderate–large $n$; conjugate (Gaussian-noise) likelihoods; comfortable choosing inducing points | Large $n$ (thousands+); need speed and are willing to restrict to stationary kernels |
    | **Likelihood** | Any (`Marginal`: Gaussian only; `Latent`: any, via explicit sampling of $f$) | Gaussian only (conjugate) | Any, `.prior()` gives you $f$ to plug into any likelihood, just like `Latent` |
    | **Key constraint** | None beyond cost | Approximation quality depends on inducing point count/placement | Stationary kernels with a known power spectral density only; input dim practically $\lesssim 3$; needs $m$/boundary tuned to the data's lengthscales |
    | **What we saw here** | Notebooks 2–3: up to ~250 points | Part B: 450 points, 25 inducing points | Part C: 2,412 grades on 24 distinct ages, 162 basis functions; then 2-D pitch locations, 625 |

    In practice: start exact whenever you can afford to (it is the
    easiest to reason about and has no approximation error to worry
    about); reach for sparse GPs when $n$ grows into the
    thousands–tens-of-thousands with a Gaussian likelihood and you can
    tolerate a modest, controllable approximation; reach for HSGP when
    $n$ is large, your kernel is stationary, and your input dimension
    is low, as here, where it turned a 2,412-observation problem into a
    regression on 162 basis coefficients with no inducing points
    to place, and then took a 2-D input in stride. Exact and sparse GPs
    accept multidimensional inputs too; what separates them here is cost,
    not capability.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Question: choose the approximation

    Use the table above. For each of these three problems, name the class
    you would reach for **first**, and, more importantly, the one check
    that would tell you the choice was wrong. Commit to all three before
    expanding the solution.

    1. **300** hourly water-level readings, Gaussian noise, and the fitted
       intervals go into a regulatory report.
    2. **40,000** soil-moisture measurements across a two-dimensional
       field, Gaussian noise, and you want a smooth map of the field.
    3. **6,000** daily event counts on a single time axis, many of them
       zero, with seasonal and trend structure.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.accordion(
        {
            "Solution": mo.md(r"""
            **1. Exact `pm.gp.Marginal`.** At $n=300$ the $O(n^3)$ cost is
            nothing, the likelihood is Gaussian, and exactness is worth
            paying for when someone will audit the intervals, there is no
            approximation error to defend. What would change your mind is
            not accuracy but scope creep: if the series grows by an order
            of magnitude, revisit. The checks that matter here are the
            ordinary ones, prior implications, inference health,
            posterior-predictive discrepancies.

            **2. HSGP or sparse, and the deciding factor is the
            lengthscale, not $n$.** Both handle 40,000 points; the table's
            constraint row is what separates them. HSGP is limited to
            stationary kernels and low input dimension, and in 2-D the
            basis count *multiplies* across dimensions ($m_1 \times m_2$),
            so a field with short lengthscales relative to its extent needs
            a basis large enough to erase the speed advantage. Estimate the
            required $m$ per dimension first. If it is modest, HSGP; if it
            explodes, sparse `MarginalApprox` with inducing points placed
            where the field varies. The falsifying check is the same in
            both cases: draw from the approximation-specific prior and ask
            whether it can represent structure at the scale you care about
           , a basis too coarse or an inducing set too sparse will show up
            there, before any fit.

            **3. HSGP feeding a Poisson likelihood.** The counts rule out
            two of the three columns immediately: sparse `MarginalApprox`
            is conjugate-only, and exact `Latent` at $n=6{,}000$ means
            $O(n^3)$ inside every gradient evaluation. `HSGP.prior()`
            returns $f$, which you plug into `pm.Poisson` exactly as
            `Latent` would. This is the one case where the likelihood, not
            the sample size, makes the decision. Check it on the count
            scale: replicated zero fraction and variance-to-mean against
            the observed values, plus the usual basis and boundary
            diagnostics.

            The pattern across all three: sample size narrows the field,
            but the likelihood and the input dimension usually decide, and
            the prior-predictive draw under the chosen approximation is
            what catches a bad choice before it costs you a fit.
            """)
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Approximation workflow: a reusable checklist

    Scaling a GP is not merely a matter of selecting the fastest class. Start
    by preserving the generative question: draw from an approximation-specific
    prior and ask whether its functions and observations can represent the
    scientific signal. State the approximation configuration in the fitted
    artifact: inducing locations for FITC, or basis count, boundary, periods,
    and standardization for HSGP, so predictions can be reconstructed without
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

    Ask four concrete questions.

    1. Are the covariate
    domain and units represented consistently in the approximation and its
    prediction model?

    2. Do prior draws reveal behavior that contradicts known
    scale, periodicity, or smoothness?

    3. Do sampled free variables have healthy
    diagnostics before their posterior is used for prediction?

    4. Do replicated observations reproduce the discrepancies that matter for the
    decision?

    This sequence prevents an efficient approximation from becoming
    an opaque black box. It also keeps the distinction clear between an
    approximation that is computationally convenient and one that is adequate
    for the observed data and stated predictive task.
    Apply this checklist anew for each fit.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Want to learn more?

    - Rasmussen, C. E. & Williams, C. K. I. (2005). *Gaussian Processes
      for Machine Learning.* MIT Press.
    - Quinonero-Candela, J. & Rasmussen, C. E. (2005). A Unifying View
      of Sparse Approximate Gaussian Process Regression. *Journal of
      Machine Learning Research* 6, 1939-1959.
    - Duvenaud, D. *The Kernel Cookbook: Advice on Covariance
      functions.* https://www.cs.toronto.edu/~duvenaud/cookbook/index.html
    - Riutort-Mayol, G., Burkner, P., Andersen, M., Solin, A., &
      Vehtari, A. (2023). Practical Hilbert space approximate Bayesian
      Gaussian processes for probabilistic programming. *Statistics and
      Computing* 33, 17. https://arxiv.org/abs/2004.11408
    """)
    return


@app.cell(hide_code=True)
def _():
    pymc_labs_logo = mo.image(
        project_root / "assets" / "pymc-labs-logo.png",
        alt="PyMC Labs",
        width=260,
    )
    mo.vstack(
        [
            pymc_labs_logo,
            mo.md(r"""
            ## Continue with PyMC Labs

            [PyMC Labs](https://www.pymc-labs.com/) is the Bayesian AI consultancy
            founded by the creators of PyMC. The team builds and optimizes custom
            Bayesian models, develops AI systems, and provides technical advising
            and hands-on training for organizations working on difficult data and
            decision problems.

            For consulting, training, or follow-up questions, contact
            [Chris Fonnesbeck](mailto:chris.fonnesbeck@pymc-labs.com) at
            [chris.fonnesbeck@pymc-labs.com](mailto:chris.fonnesbeck@pymc-labs.com).
            """),
        ],
        gap=1,
    )
    return


if __name__ == "__main__":
    app.run()
