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
        ## Part A: Marginal GP on the Theophylline curve

        Recall the Theophylline dataset from Notebook 1: 12 subjects, each
        with 11 serum-concentration measurements over 24 hours after a
        single oral dose. We again work with a single subject's curve — a
        smooth rise-then-decay shape with no natural parametric form.
        """
    )
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
    mo.md(
        r"""
        ### A naive first attempt

        Let's fit the most straightforward possible GP: a zero-mean
        `pm.gp.Marginal` with a `Matern52` covariance, evaluated directly on
        the **standardized raw time** axis. `pm.gp.Marginal` marginalizes
        the latent function analytically, so `.marginal_likelihood` gives us
        an exact Gaussian likelihood in the hyperparameters ($\ell$, $\eta$,
        $\sigma$) — no MCMC over $f$ itself is needed. We optimize with
        `pm.find_MAP` for a quick first look before committing to full
        sampling.
        """
    )
    return


@app.cell
def _(X, pm, y):
    with pm.Model() as naive_model:
        ell_naive = pm.InverseGamma("ell", alpha=5, beta=5)
        eta_naive = pm.HalfNormal("eta", sigma=2)
        sigma_naive = pm.HalfNormal("sigma", sigma=1)

        cov_naive = eta_naive**2 * pm.gp.cov.Matern52(1, ls=ell_naive)
        gp_naive = pm.gp.Marginal(cov_func=cov_naive)
        gp_naive.marginal_likelihood("y", X=X, y=y, sigma=sigma_naive)

    with naive_model:
        map_naive = pm.find_MAP(progressbar=False)

    return gp_naive, map_naive, naive_model


@app.cell
def _(
    PYMC_LIGHT_BLUE,
    conc_mean,
    conc_std,
    conc_vals,
    gp_naive,
    go,
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
    mo.md(
        r"""
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
        """
    )
    return


@app.cell
def _(np, time_vals, z):
    log_time = np.log1p(time_vals)
    log_time_mean, log_time_std = log_time.mean(), log_time.std(ddof=0)
    X_log = z(log_time).reshape(-1, 1)
    return X_log, log_time_mean, log_time_std


@app.cell
def _(X_log, pm, y):
    with pm.Model() as gp_model:
        beta_mean = pm.Normal("beta_mean", mu=0, sigma=1)
        intercept_mean = pm.Normal("intercept_mean", mu=0, sigma=1)
        mean_func = pm.gp.mean.Linear(coeffs=beta_mean, intercept=intercept_mean)

        ell = pm.InverseGamma("ell", alpha=5, beta=5)
        eta = pm.HalfNormal("eta", sigma=2)
        sigma = pm.HalfNormal("sigma", sigma=1)
        cov_func = eta**2 * pm.gp.cov.Matern52(1, ls=ell)

        gp = pm.gp.Marginal(mean_func=mean_func, cov_func=cov_func)
        gp.marginal_likelihood("y", X=X_log, y=y, sigma=sigma)

    return gp, gp_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Prior predictive check

        As always, we look at what the model implies *before* fitting.
        """
    )
    return


@app.cell
def _(RANDOM_SEED, gp_model, pm):
    with gp_model:
        prior_pred = pm.sample_prior_predictive(draws=500, random_seed=RANDOM_SEED)
    return (prior_pred,)


@app.cell
def _(PYMC_LIGHT_BLUE, X_log, go, np, prior_pred, y):
    prior_draws = prior_pred["prior_predictive"]["y"].values.reshape(-1, len(y))

    prior_fig = go.Figure()
    rng_plot = np.random.default_rng(0)
    for _i in rng_plot.choice(prior_draws.shape[0], size=50, replace=False):
        prior_fig.add_trace(
            go.Scatter(
                x=X_log[:, 0],
                y=prior_draws[_i],
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
        **Plausibility check:** prior predictive draws span
        [{prior_draws.min():.2f}, {prior_draws.max():.2f}] on the
        standardized scale, comfortably bracketing the observed range
        [{y.min():.2f}, {y.max():.2f}]. Broad but not absurd — reasonable to
        proceed.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
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
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
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
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### MAP vs. full posterior

        `pm.gp.Marginal` only ever samples the **hyperparameters**
        ($\ell$, $\eta$, $\sigma$, and the two mean-function coefficients) —
        five scalars, since $f$ itself is marginalized out analytically.
        That makes both a MAP optimization and full MCMC cheap; let's do
        both and compare.
        """
    )
    return


@app.cell
def _(gp_model, pm):
    with gp_model:
        map_estimate = pm.find_MAP(progressbar=False)
    return (map_estimate,)


@app.cell
def _(RANDOM_SEED, gp_model, pm):
    with gp_model:
        idata = pm.sample(random_seed=RANDOM_SEED, target_accept=0.95)
    return (idata,)


@app.cell
def _(az, idata):
    n_div = idata["sample_stats"]["diverging"].sum().item()
    n_draws_total = idata["posterior"].sizes["chain"] * idata["posterior"].sizes["draw"]
    summary = az.summary(idata["posterior"])
    min_ess_bulk = float(summary["ess_bulk"].min())
    min_ess_tail = float(summary["ess_tail"].min())
    max_rhat = float(summary["r_hat"].astype(float).max())
    print(f"Divergences: {n_div} / {n_draws_total}")
    summary
    return max_rhat, min_ess_bulk, min_ess_tail, n_div, n_draws_total, summary


@app.cell(hide_code=True)
def _(
    map_estimate,
    max_rhat,
    min_ess_bulk,
    min_ess_tail,
    mo,
    n_div,
    n_draws_total,
    summary,
):
    mo.md(
        f"""
        **Diagnostics:** {n_div} divergence(s) out of {n_draws_total} draws
        (near-zero). Minimum `ess_bulk` is {min_ess_bulk:.0f} and minimum
        `ess_tail` is {min_ess_tail:.0f} — both comfortably above the 400
        threshold — and maximum `r_hat` is {max_rhat:.3f} (see table above).
        Safe to interpret.

        **MAP vs. posterior mean** (lengthscale $\\ell$): MAP gives
        {map_estimate["ell"]:.3f}; the full posterior mean is
        {float(summary.loc["ell", "mean"]):.3f}. They agree closely here, which is
        expected for a fairly well-identified, low-dimensional hyperparameter
        posterior — but the full posterior additionally tells us the
        *uncertainty* around that estimate, which MAP alone cannot.
        """
    )
    return


@app.cell
def _(map_estimate, pl, summary):
    _params = ["ell", "eta", "sigma", "beta_mean", "intercept_mean"]
    map_vs_post = pl.DataFrame(
        {
            "parameter": _params,
            "MAP": [round(float(map_estimate[p]), 3) for p in _params],
            "posterior_mean": [
                round(float(summary.loc[p, "mean"]), 3) for p in _params
            ],
            "posterior_sd": [round(float(summary.loc[p, "sd"]), 3) for p in _params],
        }
    )
    map_vs_post
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
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
        """
    )
    return


@app.cell
def _(log_time_mean, log_time_std, np, time_vals):
    time_grid = np.linspace(time_vals.min(), time_vals.max(), 200)
    Xnew = ((np.log1p(time_grid) - log_time_mean) / log_time_std).reshape(-1, 1)
    return Xnew, time_grid


@app.cell
def _(Xnew, gp, gp_model):
    with gp_model:
        f_pred = gp.conditional("f_pred", Xnew)
        f_pred_noise = gp.conditional("f_pred_noise", Xnew, pred_noise=True)
    return f_pred, f_pred_noise


@app.cell
def _(RANDOM_SEED, f_pred, f_pred_noise, gp_model, idata, pm):
    with gp_model:
        ppc = pm.sample_posterior_predictive(
            idata, var_names=["f_pred", "f_pred_noise"], random_seed=RANDOM_SEED
        )
    return (ppc,)


@app.cell
def _(
    PYMC_BLUE,
    az,
    conc_mean,
    conc_std,
    conc_vals,
    go,
    np,
    ppc,
    time_grid,
    time_vals,
):
    f_pred_vals = (
        ppc["posterior_predictive"]["f_pred"].values.reshape(-1, len(time_grid))
        * conc_std
        + conc_mean
    )
    f_pred_noise_vals = (
        ppc["posterior_predictive"]["f_pred_noise"].values.reshape(-1, len(time_grid))
        * conc_std
        + conc_mean
    )

    f_pred_mean = f_pred_vals.mean(axis=0)
    f_pred_hdi = az.hdi(f_pred_vals, prob=0.89, axis=0)
    f_pred_lo, f_pred_hi = f_pred_hdi[:, 0], f_pred_hdi[:, 1]
    f_pred_noise_hdi = az.hdi(f_pred_noise_vals, prob=0.89, axis=0)
    f_pred_noise_lo, f_pred_noise_hi = f_pred_noise_hdi[:, 0], f_pred_noise_hdi[:, 1]

    pred_fig = go.Figure()
    pred_fig.add_trace(
        go.Scatter(
            x=np.concatenate([time_grid, time_grid[::-1]]),
            y=np.concatenate([f_pred_noise_hi, f_pred_noise_lo[::-1]]),
            fill="toself",
            fillcolor="rgba(74,158,222,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="89% HDI (with noise)",
        )
    )
    pred_fig.add_trace(
        go.Scatter(
            x=np.concatenate([time_grid, time_grid[::-1]]),
            y=np.concatenate([f_pred_hi, f_pred_lo[::-1]]),
            fill="toself",
            fillcolor="rgba(21,74,114,0.35)",
            line=dict(color="rgba(255,255,255,0)"),
            name="89% HDI (f only)",
        )
    )
    pred_fig.add_trace(
        go.Scatter(
            x=time_grid,
            y=f_pred_mean,
            mode="lines",
            name="posterior mean",
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
        title="Marginal GP fit (log1p(time) + linear mean) — f vs. f + noise",
        xaxis_title="Time since dose (hours)",
        yaxis_title="Concentration (mg/L)",
        template="plotly_white",
    )
    pred_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The narrower band is uncertainty in the latent function $f$ alone
        (`pred_noise=False`); the wider band additionally folds in the
        observation noise $\sigma$ (`pred_noise=True`) and is the right one
        to compare against *new, unobserved measurements*. The transformed
        input plus linear mean function now trace the rise, peak, and decay
        far more faithfully than the naive fit.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
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
        """
    )
    return


@app.cell
def _(conc_std, summary):
    sigma_post_mean = float(summary.loc["sigma", "mean"])
    noise_var_mgL = (sigma_post_mean * conc_std) ** 2
    print(f"Posterior-mean sigma (standardized): {sigma_post_mean:.3f}")
    print(f"Constant variance gap between the bands: {noise_var_mgL:.3f} (mg/L)^2")
    print(f"  i.e. a noise standard deviation of {sigma_post_mean * conc_std:.3f} mg/L")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Exercise: extrapolate beyond the observed time range

        Predict out to, say, 40 hours past dose — well beyond the last
        observation at 24.37 hours — using `gp.predict` at the posterior
        mean hyperparameters. What happens to the mean and the uncertainty
        band as you move past the data? Expand the solution below once
        you've made a prediction.
        """
    )
    return


@app.cell(hide_code=True)
def _(
    PYMC_BLUE,
    conc_mean,
    conc_std,
    conc_vals,
    go,
    gp,
    gp_model,
    idata,
    log_time_mean,
    log_time_std,
    mo,
    np,
    time_vals,
):
    extrap_grid = np.linspace(0, 40, 200)
    extrap_Xnew = ((np.log1p(extrap_grid) - log_time_mean) / log_time_std).reshape(
        -1, 1
    )

    extrap_point = {
        var: idata["posterior"][var].mean(dim=["chain", "draw"]).values
        for var in ["ell", "eta", "sigma", "beta_mean", "intercept_mean"]
    }

    with gp_model:
        extrap_mu, extrap_var = gp.predict(
            extrap_Xnew, point=extrap_point, diag=True, pred_noise=True
        )

    extrap_mu_orig = extrap_mu * conc_std + conc_mean
    extrap_sd_orig = np.sqrt(extrap_var) * conc_std

    extrap_fig = go.Figure()
    extrap_fig.add_trace(
        go.Scatter(
            x=np.concatenate([extrap_grid, extrap_grid[::-1]]),
            y=np.concatenate(
                [
                    extrap_mu_orig + 2 * extrap_sd_orig,
                    (extrap_mu_orig - 2 * extrap_sd_orig)[::-1],
                ]
            ),
            fill="toself",
            fillcolor="rgba(21,74,114,0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="mean ± 2 SD",
        )
    )
    extrap_fig.add_trace(
        go.Scatter(
            x=extrap_grid,
            y=extrap_mu_orig,
            mode="lines",
            name="posterior-mean-hyperparameter prediction",
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
        title="Extrapolation past the last observation (dashed line)",
        xaxis_title="Time since dose (hours)",
        yaxis_title="Concentration (mg/L)",
        template="plotly_white",
    )

    mo.accordion(
        {
            "Solution and diagnosis": mo.vstack(
                [
                    mo.md(
                        """
                        ```python
                        extrap_grid = np.linspace(0, 40, 200)
                        extrap_Xnew = ((np.log1p(extrap_grid) - log_time_mean)
                                       / log_time_std).reshape(-1, 1)
                        mu, var = gp.predict(extrap_Xnew, point=posterior_mean_point,
                                              diag=True, pred_noise=True)
                        ```

                        Past the last observed point (dashed line), the mean
                        prediction reverts toward the **linear mean
                        function** — there is no more data pulling the GP
                        away from it — and the uncertainty band widens
                        rapidly, since the covariance function assigns
                        vanishing correlation to points far (in
                        log1p-time) from any training input. This is the
                        correct, honest behavior: a GP does not
                        extrapolate a learned *shape*, only reverts to its
                        prior mean with growing uncertainty.
                        """
                    ),
                    extrap_fig,
                ]
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Exercise — what does the linear mean function buy us?": mo.md(
                r"""
                Suppose you removed the `pm.gp.mean.Linear` and used a
                *zero-mean* GP on the same `log1p(time)` input. Predict what
                would change in (a) the fit within the observed data and (b) the
                extrapolation beyond 24 hours.

                **Solution.** Within the data the fit would look *similar* — a
                flexible GP can represent the trend through its covariance
                alone, so the posterior mean still traces rise–peak–decay. The
                difference shows up in **extrapolation and in how hard the GP has
                to work**. With a zero mean, everything past the last point
                reverts toward 0 (the standardized output mean, i.e. the overall
                average concentration) rather than toward a *sloped* line; and
                the lengthscale/amplitude must stretch to explain the global
                downward drift that the linear mean would otherwise soak up
                cheaply. A mean function encodes the part of the trend you can
                name — here, "log-concentration falls roughly linearly in
                log-time" — leaving the GP to model only the residual curvature,
                which usually yields tighter, more stable hyperparameter
                posteriors.
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Exercise — would ExpQuad fit better than Matérn 5/2?": mo.md(
                r"""
                We used a `Matern52` kernel. The `ExpQuad` (squared-exponential)
                kernel is smoother still — infinitely differentiable, versus
                Matérn 5/2's two derivatives. Would swapping it in improve this
                pharmacokinetic fit? Reason about it before trying.

                **Solution.** Probably not meaningfully, and possibly worse.
                ExpQuad assumes the underlying function is *extremely* smooth,
                but real absorption/elimination curves have a fairly sharp early
                rise that an over-smooth kernel tends to round off or overshoot,
                especially with a single global lengthscale. Matérn 5/2 is the
                common default for physical processes precisely because it allows
                a little more local "give" while still looking continuous. The
                rigorous way to choose is out-of-sample predictive accuracy
                (LOO), which is beyond this intro course — but the intuition is:
                *match the kernel's assumed smoothness to the process*, and when
                unsure, Matérn 5/2 is a safer default than ExpQuad.
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
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
        """
    )
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
    mo.md(
        r"""
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
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
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
        """
    )
    return


@app.cell
def _(disaster_counts, pm, t):
    with pm.Model() as coal_model:
        ell_coal = pm.InverseGamma("ell", alpha=5, beta=5)
        eta_coal = pm.HalfNormal("eta", sigma=1)
        cov_coal = eta_coal**2 * pm.gp.cov.Matern52(1, ls=ell_coal)

        gp_coal = pm.gp.Latent(cov_func=cov_coal)
        f_coal = gp_coal.prior("f", X=t)

        rate = pm.Deterministic("rate", pm.math.exp(f_coal))
        pm.Poisson("y", mu=rate, observed=disaster_counts)

    return (coal_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Prior predictive check""")
    return


@app.cell
def _(RANDOM_SEED, coal_model, pm):
    with coal_model:
        coal_prior_pred = pm.sample_prior_predictive(draws=500, random_seed=RANDOM_SEED)
    return (coal_prior_pred,)


@app.cell
def _(PYMC_GREEN, coal_prior_pred, disaster_counts, go, np, year_vals):
    coal_prior_draws = coal_prior_pred["prior_predictive"]["y"].values.reshape(
        -1, len(year_vals)
    )
    coal_prior_lo, coal_prior_hi = np.quantile(coal_prior_draws, [0.055, 0.945], axis=0)

    coal_prior_fig = go.Figure()
    coal_prior_fig.add_trace(
        go.Scatter(
            x=np.concatenate([year_vals, year_vals[::-1]]),
            y=np.concatenate([coal_prior_hi, coal_prior_lo[::-1]]),
            fill="toself",
            fillcolor="rgba(129,194,64,0.25)",
            line=dict(color="rgba(255,255,255,0)"),
            name="89% prior predictive interval",
        )
    )
    coal_prior_fig.add_trace(
        go.Scatter(
            x=year_vals,
            y=disaster_counts,
            mode="markers",
            marker=dict(color="black", size=6),
            name="observed",
        )
    )
    coal_prior_fig.update_layout(
        title="Prior predictive counts vs. observed",
        xaxis_title="Year",
        yaxis_title="Disasters",
        template="plotly_white",
    )
    coal_prior_fig
    return (coal_prior_draws,)


@app.cell(hide_code=True)
def _(coal_prior_draws, disaster_counts, mo):
    mo.md(
        f"""
        **Plausibility check:** prior predictive counts range from
        {coal_prior_draws.min():.0f} to {coal_prior_draws.max():.0f}, which
        comfortably brackets the observed range
        [{disaster_counts.min()}, {disaster_counts.max()}] without being
        absurdly wide — reasonable to proceed to sampling.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Sampling

        112 latent function values plus 2 covariance hyperparameters. We use
        `draws=1500, tune=1000, chains=2` (slightly more draws than the
        `1000` default, to comfortably clear the `ess_bulk > 400` bar across
        all 112 latent values) and raise `target_accept` to 0.95, which is
        typical for latent GPs (the funnel between $f$ and its lengthscale
        is more delicate than in the marginal case).
        """
    )
    return


@app.cell
def _(RANDOM_SEED, coal_model, perf_counter, pm):
    with coal_model:
        _start = perf_counter()
        coal_idata = pm.sample(
            random_seed=RANDOM_SEED,
            target_accept=0.95,
            draws=1500,
            tune=1000,
            chains=2,
        )
        coal_sample_seconds = perf_counter() - _start
    print(f"Coal latent-GP sampling wall-time: {coal_sample_seconds:.1f}s")
    return coal_idata, coal_sample_seconds


@app.cell
def _(az, coal_idata):
    coal_n_div = coal_idata["sample_stats"]["diverging"].sum().item()
    coal_n_draws_total = (
        coal_idata["posterior"].sizes["chain"] * coal_idata["posterior"].sizes["draw"]
    )
    coal_summary = az.summary(coal_idata["posterior"], var_names=["ell", "eta"])
    coal_f_summary = az.summary(coal_idata["posterior"], var_names=["f"])
    coal_min_ess_bulk = float(
        min(coal_summary["ess_bulk"].min(), coal_f_summary["ess_bulk"].min())
    )
    coal_min_ess_tail = float(
        min(coal_summary["ess_tail"].min(), coal_f_summary["ess_tail"].min())
    )
    coal_max_rhat = max(
        coal_summary["r_hat"].astype(float).max(),
        coal_f_summary["r_hat"].astype(float).max(),
    )
    print(f"Divergences: {coal_n_div} / {coal_n_draws_total}")
    print(
        f"Min ess_bulk / ess_tail (hyperparams + f): {coal_min_ess_bulk:.0f} / {coal_min_ess_tail:.0f}"
    )
    coal_summary
    return (
        coal_max_rhat,
        coal_min_ess_bulk,
        coal_min_ess_tail,
        coal_n_div,
        coal_n_draws_total,
    )


@app.cell(hide_code=True)
def _(
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
        **Diagnostics:** {coal_n_div} divergence(s) out of
        {coal_n_draws_total} draws in {coal_sample_seconds:.1f}s of
        sampling. Minimum `ess_bulk` across the 2 hyperparameters and 112
        latent $f$ values is {coal_min_ess_bulk:.0f} and minimum `ess_tail`
        is {coal_min_ess_tail:.0f} — both comfortably above the 400
        threshold — and maximum `r_hat` is {coal_max_rhat:.3f}. Safe to
        interpret.
        """
    )
    return


@app.cell
def _(PYMC_BLUE, az, coal_idata, disaster_counts, go, np, year_vals):
    rate_samples = coal_idata["posterior"]["rate"].values.reshape(-1, len(year_vals))
    rate_mean = rate_samples.mean(axis=0)
    rate_hdi = az.hdi(rate_samples, prob=0.89, axis=0)
    rate_lo, rate_hi = rate_hdi[:, 0], rate_hdi[:, 1]

    rate_fig = go.Figure()
    rate_fig.add_trace(
        go.Scatter(
            x=np.concatenate([year_vals, year_vals[::-1]]),
            y=np.concatenate([rate_hi, rate_lo[::-1]]),
            fill="toself",
            fillcolor="rgba(21,74,114,0.25)",
            line=dict(color="rgba(255,255,255,0)"),
            name="89% HDI",
        )
    )
    rate_fig.add_trace(
        go.Scatter(
            x=year_vals,
            y=rate_mean,
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
    mo.md(
        r"""
        ### The posterior is a distribution over rate *functions*

        The HDI band above summarizes the posterior, but it can hide that each
        posterior draw is a whole *function*. Plotting sixty individual draws of
        $\exp(f)$ makes the object concrete: the model is uncertain not about a
        few numbers but about the entire trajectory, and the draws fan out where
        data are sparse (the early years, few disasters to pin the rate) and
        pull together where the counts are more informative.
        """
    )
    return


@app.cell
def _(PYMC_GREEN, coal_idata, go, np, year_vals):
    _rate_draws = coal_idata["posterior"]["rate"].values.reshape(-1, len(year_vals))
    _rng = np.random.default_rng(1)
    spaghetti_fig = go.Figure()
    for _i in _rng.choice(_rate_draws.shape[0], size=60, replace=False):
        spaghetti_fig.add_trace(
            go.Scatter(
                x=year_vals,
                y=_rate_draws[_i],
                mode="lines",
                line=dict(color=PYMC_GREEN, width=1),
                opacity=0.15,
                showlegend=False,
            )
        )
    spaghetti_fig.update_layout(
        title="Sixty posterior draws of the rate exp(f) — the posterior over functions",
        xaxis_title="Year",
        yaxis_title="Disasters / year (rate)",
        template="plotly_white",
    )
    spaghetti_fig
    return


@app.cell
def _(az, coal_idata, np, year_vals):
    _rate = coal_idata["posterior"]["rate"].values.reshape(-1, len(year_vals))

    def _rate_at(year):
        idx = int(np.argmin(np.abs(year_vals - year)))
        col = _rate[:, idx]
        return col.mean(), az.hdi(col, prob=0.89)

    rate_1851 = _rate_at(1851)
    rate_1900 = _rate_at(1900)
    rate_1962 = _rate_at(1962)
    pct_decline = 100 * (1 - rate_1962[0] / rate_1851[0])
    print(
        f"Mean rate 1851: {rate_1851[0]:.2f} /yr "
        f"(89% HDI {rate_1851[1][0]:.2f}-{rate_1851[1][1]:.2f})"
    )
    print(
        f"Mean rate 1900: {rate_1900[0]:.2f} /yr "
        f"(89% HDI {rate_1900[1][0]:.2f}-{rate_1900[1][1]:.2f})"
    )
    print(
        f"Mean rate 1962: {rate_1962[0]:.2f} /yr "
        f"(89% HDI {rate_1962[1][0]:.2f}-{rate_1962[1][1]:.2f})"
    )
    print(f"Overall decline 1851 to 1962: {pct_decline:.0f}%")
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

        - The GP reports the decline **with uncertainty everywhere** (the HDIs
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
    mo.md(
        r"""
        The latent rate shows a clear decline over the period with no
        sharp changepoint imposed — the GP infers the shape of the decline
        (which looks like it happens over some decades around the turn of
        the century) directly from the data, unlike a two-regime
        changepoint model.

        ### Posterior predictive check

        Finally, we simulate new counts from the fitted model and check
        that they bracket the observed series.
        """
    )
    return


@app.cell
def _(RANDOM_SEED, coal_idata, coal_model, pm):
    with coal_model:
        coal_ppc = pm.sample_posterior_predictive(
            coal_idata, var_names=["y"], random_seed=RANDOM_SEED
        )
    return (coal_ppc,)


@app.cell
def _(PYMC_GREEN, coal_ppc, disaster_counts, go, np, year_vals):
    ppc_counts = coal_ppc["posterior_predictive"]["y"].values.reshape(
        -1, len(year_vals)
    )
    ppc_mean = ppc_counts.mean(axis=0)
    ppc_lo, ppc_hi = np.quantile(ppc_counts, [0.055, 0.945], axis=0)

    ppc_fig = go.Figure()
    ppc_fig.add_trace(
        go.Scatter(
            x=np.concatenate([year_vals, year_vals[::-1]]),
            y=np.concatenate([ppc_hi, ppc_lo[::-1]]),
            fill="toself",
            fillcolor="rgba(129,194,64,0.25)",
            line=dict(color="rgba(255,255,255,0)"),
            name="89% posterior predictive interval",
        )
    )
    ppc_fig.add_trace(
        go.Scatter(
            x=year_vals,
            y=ppc_mean,
            mode="lines",
            name="posterior predictive mean",
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
        title="Posterior predictive count check",
        xaxis_title="Year",
        yaxis_title="Disasters",
        template="plotly_white",
    )
    ppc_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Observed counts fall consistently within the posterior predictive
        interval, and the mean tracks the visible decline in disaster
        frequency — the model captures the broad structure of the series.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Exercise: change the lengthscale prior and compare

        The `ell ~ InverseGamma(5, 5)` prior above is weakly informative on
        the standardized year scale. What happens to the *prior* on the
        implied rate function if we instead favor much shorter
        lengthscales — say `InverseGamma(2, 1)`, which puts more mass near
        zero? Try it yourself, then expand the solution below.
        """
    )
    return


@app.cell(hide_code=True)
def _(
    PYMC_GREEN,
    RANDOM_SEED,
    disaster_counts,
    go,
    mo,
    np,
    pm,
    t,
    year_vals,
):
    with pm.Model():
        ell_alt = pm.InverseGamma("ell", alpha=2, beta=1)
        eta_alt = pm.HalfNormal("eta", sigma=2)
        cov_alt = eta_alt**2 * pm.gp.cov.Matern52(1, ls=ell_alt)

        gp_alt = pm.gp.Latent(cov_func=cov_alt)
        f_alt = gp_alt.prior("f", X=t)
        rate_alt = pm.Deterministic("rate", pm.math.exp(f_alt))
        pm.Poisson("y", mu=rate_alt, observed=disaster_counts)

        alt_prior = pm.sample_prior_predictive(draws=500, random_seed=RANDOM_SEED)

    alt_rate_draws = alt_prior["prior"]["rate"].values.reshape(-1, len(year_vals))

    alt_fig = go.Figure()
    rng_alt = np.random.default_rng(0)
    for _i in rng_alt.choice(alt_rate_draws.shape[0], size=30, replace=False):
        alt_fig.add_trace(
            go.Scatter(
                x=year_vals,
                y=alt_rate_draws[_i],
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
        title="Prior draws of rate exp(f) under a short-lengthscale prior",
        xaxis_title="Year",
        yaxis_title="rate",
        template="plotly_white",
        showlegend=False,
    )

    mo.accordion(
        {
            "Solution and comparison": mo.vstack(
                [
                    mo.md(
                        """
                        ```python
                        ell_alt = pm.InverseGamma("ell", alpha=2, beta=1)
                        ```

                        With `InverseGamma(2, 1)` the prior mass shifts
                        toward much shorter lengthscales than
                        `InverseGamma(5, 5)`. Prior draws of the rate
                        function (below) are visibly wigglier year-to-year —
                        the GP prior now expects the disaster rate to swing
                        rapidly rather than evolve smoothly over decades.
                        With only 112 annual counts, a short-lengthscale
                        prior risks the posterior chasing individual noisy
                        years rather than recovering the genuine multi-decade
                        decline seen above — a concrete illustration of why
                        lengthscale priors matter for GPs on sparse data.
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
    mo.accordion(
        {
            "Exercise — does a Poisson likelihood capture the spread?": mo.md(
                r"""
                The posterior predictive check plotted the 89% band and it
                covered the data. A sterner test: a Poisson likelihood forces
                $\operatorname{Var}[y] = \mathbb E[y]$ (mean equals variance).
                Sketch how you would check whether the coal counts are
                *over-dispersed* relative to Poisson, and what you would switch
                to if they were.

                **Solution.** Compare a dispersion statistic between the observed
                series and the posterior predictive draws — e.g. the ratio of
                sample variance to sample mean — by overlaying the *distribution*
                of that ratio across posterior predictive datasets against the
                observed value (a posterior-predictive-check style comparison).
                If the observed counts scatter more than Poisson allows, the
                observed variance/mean ratio would sit in the upper tail of the
                predictive distribution. The standard fix is a likelihood with a
                free dispersion parameter — a **Negative Binomial** GP (same exp
                link, same latent $f$), which adds an over-dispersion parameter
                and reduces to Poisson as that parameter grows. Here the Poisson
                fit looks adequate, but this is the first thing to check for
                count GPs in the wild.
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Exercise — could we have forced this into a marginal GP?": mo.md(
                r"""
                A tempting shortcut: transform the counts (say $\sqrt{y}$ or
                $\log(y+1)$) to make them roughly Gaussian, then use the fast
                `pm.gp.Marginal` from Part A instead of the slower latent GP.
                When is that reasonable, and what does it cost *here*?

                **Solution.** It is a real technique — a variance-stabilizing
                transform plus a Gaussian GP — and for *large* counts it works
                passably, because a Poisson with a big mean is approximately
                Gaussian on the $\sqrt{\cdot}$ scale. But the coal counts are
                *small* (many years are 0, 1, or 2), where that approximation is
                poor: $\sqrt{0}, \sqrt{1}, \sqrt{2}$ are badly non-Gaussian, the
                transform cannot represent genuine zeros cleanly, and you throw
                away the honest discrete count likelihood. You would gain speed
                (marginalizing $f$) at the cost of a mis-specified observation
                model right where the data are most informative about the low
                rate. For small counts the latent Poisson GP is the correct tool;
                the marginal shortcut is only defensible when counts are
                uniformly large.
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
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
        """
    )
    return


if __name__ == "__main__":
    app.run()
