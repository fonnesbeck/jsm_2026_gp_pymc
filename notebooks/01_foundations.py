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
        # Foundations: A PyMC Primer and an Introduction to Gaussian Processes

        This notebook is in two parts.

        **Part A** is a fast primer on the Bayesian workflow in PyMC: build a
        model, check the prior, sample the posterior, summarize it. If you
        already know PyMC well, skim it.

        **Part B** builds up the idea of a Gaussian process (GP) from the
        multivariate normal distribution you already know, and has you
        implement a covariance function from scratch.

        Throughout the workshop we use a single running dataset —
        theophylline concentrations in blood plasma after an oral dose —
        introduced below.
        """
    )
    return


@app.cell(hide_code=True)
def _():
    from pathlib import Path

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
        pl,
        pm,
        rng,
        z,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Background: the Theophylline dataset

        `Theoph` is a classic pharmacokinetic dataset built into R (Boeckmann,
        Sheiner & Beal 1994, *NONMEM Users Guide*): 12 subjects each received
        a single oral dose of the asthma drug theophylline, and their serum
        concentration was measured at 11 time points over the following
        24 hours.

        After an oral dose, concentration first **rises** as the drug is
        absorbed from the gut, peaks, and then **decays** as it is cleared
        from the body — a smooth, non-linear curve that is a poor fit for a
        straight line but a natural fit for a Gaussian process. That makes
        it a good running example for this workshop.

        - `subject`: subject id (1–12)
        - `time`: hours since dose
        - `conc`: serum theophylline concentration (mg/L)
        - `dose`: administered dose (mg/kg)
        - `weight`: subject body weight (kg)
        """
    )
    return


@app.cell
def _(data_dir, pl):
    theoph = pl.read_csv(data_dir / "theophylline.csv")
    theoph.head()
    return (theoph,)


@app.cell
def _(PYMC_BLUE, go, pl, theoph):
    subject_id = 1
    subject_df = theoph.filter(pl.col("subject") == subject_id).sort("time")

    subject_fig = go.Figure()
    subject_fig.add_trace(
        go.Scatter(
            x=subject_df["time"].to_list(),
            y=subject_df["conc"].to_list(),
            mode="markers+lines",
            name=f"Subject {subject_id}",
            line=dict(color=PYMC_BLUE),
            marker=dict(color=PYMC_BLUE, size=8),
        )
    )
    subject_fig.update_layout(
        title=f"Theophylline concentration over time — Subject {subject_id}",
        xaxis_title="Time since dose (hours)",
        yaxis_title="Concentration (mg/L)",
        template="plotly_white",
    )
    subject_fig
    return subject_df, subject_id


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Part A: The Bayesian paradigm in PyMC

        Bayesian inference combines a **prior** — what we believe about
        parameters before seeing data — with a **likelihood** — how likely
        the observed data are under different parameter values — to produce
        a **posterior**: an updated, data-informed belief about the
        parameters.

        $$p(\theta \mid y) \propto p(y \mid \theta)\, p(\theta)$$

        In PyMC, you write down the prior and the likelihood as a
        `pm.Model`, and PyMC's sampler (as of PyMC 6, this defaults to the
        fast Rust-based `nutpie` sampler automatically — no argument
        needed) draws samples from the posterior for you.

        As a first example, let's fit a **simple linear regression** of
        concentration on time, using only Subject 1's data. This won't be a
        good model for this curve (the true relationship rises then decays),
        but it's a clean way to see the full PyMC workflow: standardize the
        data, specify priors, check the prior predictive, sample, and
        summarize.
        """
    )
    return


@app.cell
def _(subject_df, z):
    time_vals = subject_df["time"].to_numpy()
    conc_vals = subject_df["conc"].to_numpy()

    time_mean, time_std = time_vals.mean(), time_vals.std(ddof=0)
    conc_mean, conc_std = conc_vals.mean(), conc_vals.std(ddof=0)

    time_z = z(time_vals)
    conc_z = z(conc_vals)

    return (
        conc_mean,
        conc_std,
        conc_vals,
        conc_z,
        time_mean,
        time_std,
        time_vals,
        time_z,
    )


@app.cell
def _(conc_z, pm, time_z):
    with pm.Model() as linreg_model:
        alpha = pm.Normal("alpha", mu=0, sigma=1)
        beta = pm.Normal("beta", mu=0, sigma=1)
        sigma = pm.HalfNormal("sigma", sigma=1)

        mu = alpha + beta * time_z
        pm.Normal("conc_obs", mu=mu, sigma=sigma, observed=conc_z)

    linreg_model
    return (linreg_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Prior predictive check

        Before sampling the posterior, we simulate data **from the prior
        alone** (`pm.sample_prior_predictive`). This is a sanity check: does
        the prior imply a plausible range of outcomes, on the scale of the
        (standardized) data, *before* we let the data have any say?
        """
    )
    return


@app.cell
def _(RANDOM_SEED, linreg_model, pm):
    with linreg_model:
        prior_pred = pm.sample_prior_predictive(draws=500, random_seed=RANDOM_SEED)
    return (prior_pred,)


@app.cell
def _(PYMC_LIGHT_BLUE, conc_z, go, np, prior_pred, time_z):
    prior_draws = prior_pred["prior_predictive"]["conc_obs"].values.reshape(
        -1, len(time_z)
    )

    prior_fig = go.Figure()
    rng_plot = np.random.default_rng(0)
    for _i in rng_plot.choice(prior_draws.shape[0], size=50, replace=False):
        prior_fig.add_trace(
            go.Scatter(
                x=time_z,
                y=prior_draws[_i],
                mode="lines",
                line=dict(color=PYMC_LIGHT_BLUE, width=1),
                opacity=0.25,
                showlegend=False,
            )
        )
    prior_fig.add_trace(
        go.Scatter(
            x=time_z,
            y=conc_z,
            mode="markers",
            marker=dict(color="black", size=8),
            name="observed (standardized)",
        )
    )
    prior_fig.update_layout(
        title="Prior predictive draws vs. standardized observed data",
        xaxis_title="time (standardized)",
        yaxis_title="conc (standardized)",
        template="plotly_white",
    )
    prior_fig
    return (prior_draws,)


@app.cell(hide_code=True)
def _(conc_z, mo, prior_draws):
    mo.md(
        f"""
        **Plausibility check:** the prior predictive draws span
        [{prior_draws.min():.2f}, {prior_draws.max():.2f}] on the
        standardized scale, comfortably bracketing the observed standardized
        range of [{conc_z.min():.2f}, {conc_z.max():.2f}]. The prior is
        broad but not absurd — reasonable to proceed to sampling.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Sampling the posterior

        Now we sample the posterior with `pm.sample`. PyMC 6 selects the
        `nutpie` sampler automatically, so no `nuts_sampler=` argument is
        needed. With one subject's ~11 points and a two-parameter linear
        model, this is tiny — a couple hundred draws per chain is plenty.
        """
    )
    return


@app.cell
def _(RANDOM_SEED, linreg_model, pm):
    with linreg_model:
        idata = pm.sample(draws=500, tune=500, chains=2, random_seed=RANDOM_SEED)
    return (idata,)


@app.cell
def _(az, idata):
    n_div = idata["sample_stats"]["diverging"].sum().item()
    print(f"Divergences: {n_div}")
    az.summary(idata["posterior"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### How well does a straight line fit?

        Let's transform the posterior mean line back to the original units
        and overlay it on the raw data.
        """
    )
    return


@app.cell
def _(
    PYMC_BLUE,
    PYMC_GREEN,
    conc_mean,
    conc_std,
    conc_vals,
    go,
    idata,
    np,
    time_mean,
    time_std,
    time_vals,
):
    alpha_mean = idata["posterior"]["alpha"].mean().item()
    beta_mean = idata["posterior"]["beta"].mean().item()

    time_grid = np.linspace(time_vals.min(), time_vals.max(), 100)
    time_grid_z = (time_grid - time_mean) / time_std
    fit_z = alpha_mean + beta_mean * time_grid_z
    fit_orig = fit_z * conc_std + conc_mean

    fit_fig = go.Figure()
    fit_fig.add_trace(
        go.Scatter(
            x=time_vals,
            y=conc_vals,
            mode="markers",
            name="observed",
            marker=dict(color=PYMC_BLUE, size=8),
        )
    )
    fit_fig.add_trace(
        go.Scatter(
            x=time_grid,
            y=fit_orig,
            mode="lines",
            name="posterior mean fit",
            line=dict(color=PYMC_GREEN, width=3),
        )
    )
    fit_fig.update_layout(
        title="Linear fit vs. data — notice the systematic misfit",
        xaxis_title="Time since dose (hours)",
        yaxis_title="Concentration (mg/L)",
        template="plotly_white",
    )
    fit_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The straight line cannot capture the rise-then-decay shape of the
        curve — it systematically undershoots near the peak and overshoots
        at both ends. This is exactly the kind of smooth, non-linear
        structure that Gaussian processes are built to model without
        committing to a fixed functional form. Part B introduces the
        machinery.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Part B: From multivariate normals to Gaussian processes

        ### Marginalization and conditioning

        You already know the multivariate normal (MVN) distribution. Two of
        its properties turn out to be exactly what's needed to make Gaussian
        processes tractable.

        Suppose $(x, y)$ jointly follow an MVN:

        $$p(x,y) = \mathcal{N}\!\left(\begin{bmatrix}\mu_x \\ \mu_y\end{bmatrix},
        \begin{bmatrix}\Sigma_x & \Sigma_{xy} \\ \Sigma_{xy}^T & \Sigma_y\end{bmatrix}\right)$$

        **Marginalization**: the distribution of a subset of the variables,
        ignoring the rest, is also normal:

        $$p(x) = \int p(x,y)\,dy = \mathcal{N}(\mu_x, \Sigma_x)$$

        **Conditioning**: the distribution of a subset *given* the rest is
        also normal:

        $$p(x \mid y) = \mathcal{N}\!\left(\mu_x + \Sigma_{xy}\Sigma_y^{-1}(y-\mu_y),\;\;
        \Sigma_x - \Sigma_{xy}\Sigma_y^{-1}\Sigma_{xy}^T\right)$$

        Below is a correlated 2D Gaussian. The contour plot is the joint
        density $p(x_1, x_2)$; slicing it at a fixed $x_2$ and renormalizing
        gives the conditional $p(x_1 \mid x_2)$, and integrating out $x_2$
        gives the marginal $p(x_1)$.
        """
    )
    return


@app.cell
def _(go, np):
    from scipy.stats import multivariate_normal

    mvn_mean = np.array([0.0, 0.0])
    mvn_cov = np.array([[1.0, 0.7], [0.7, 1.0]])

    mvn_grid = np.linspace(-3, 3, 120)
    mvn_x1, mvn_x2 = np.meshgrid(mvn_grid, mvn_grid)
    mvn_pos = np.dstack((mvn_x1, mvn_x2))
    mvn_density = multivariate_normal(mvn_mean, mvn_cov).pdf(mvn_pos)

    mvn_fig = go.Figure(
        data=go.Contour(
            x=mvn_grid,
            y=mvn_grid,
            z=mvn_density,
            colorscale="Blues",
            contours=dict(showlabels=False),
        )
    )
    mvn_fig.update_layout(
        title="Correlated bivariate normal: joint density p(x1, x2)",
        xaxis_title="x1",
        yaxis_title="x2",
        template="plotly_white",
    )
    mvn_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### A Gaussian process is a distribution over functions

        A **Gaussian process (GP)** generalizes the MVN to infinitely many
        variables: it is defined as a collection of random variables, any
        finite subset of which has a joint Gaussian distribution. If we
        think of a function $f$ as an (infinitely long) vector indexed by
        its input $x$, a GP is a distribution over such functions:

        $$f(x) \sim \mathcal{GP}\bigl(m(x),\, k(x, x')\bigr)$$

        Just as an MVN is fully specified by a mean vector and covariance
        matrix, a GP is fully specified by a **mean function** $m(x)$ and a
        **covariance function** $k(x, x')$ (also called a *kernel*). It is
        the marginalization property above that makes this workable in
        practice: evaluated at any finite set of input points, a GP reduces
        to an ordinary MVN with mean $m(x)$ and covariance $k(x, x')$
        evaluated on that grid — we never need to store the infinite object,
        only compute it where we need it.

        The mean function is very often just zero; nearly all of the
        interesting behavior comes from the covariance function, which
        controls how smoothly and how strongly nearby function values
        co-vary.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Exercise: implement the ExpQuad kernel from scratch

        The exponential quadratic (a.k.a. squared-exponential, RBF) kernel
        is the most common covariance function:

        $$k(x, x') = \eta^2 \exp\!\left(-\frac{(x-x')^2}{2\ell^2}\right)$$

        where $\eta$ (amplitude) scales the output and $\ell$ (lengthscale)
        controls how quickly correlation decays with distance.

        Write a function `expquad(x1, x2, ls, eta)` that takes two 1-D
        arrays of input locations plus a lengthscale and amplitude, and
        returns the `(len(x1), len(x2))` covariance matrix. Hint:
        `np.subtract.outer(x1, x2)` gives you every pairwise difference at
        once.

        Try it yourself, then expand the solution below to check your work.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo, np):
    def expquad(x1, x2, ls, eta):
        dist = np.subtract.outer(x1, x2)
        return eta**2 * np.exp(-0.5 * dist**2 / ls**2)

    mo.accordion(
        {
            "Solution": mo.md(
                """
                ```python
                def expquad(x1, x2, ls, eta):
                    dist = np.subtract.outer(x1, x2)
                    return eta**2 * np.exp(-0.5 * dist**2 / ls**2)
                ```
                """
            )
        }
    )
    return (expquad,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        As a sanity check, PyMC ships this same kernel as
        `pm.gp.cov.ExpQuad`. PyMC's GP covariance functions always expect
        **2D inputs** of shape `(n, 1)` (one row per input point), even for
        a single input dimension — unlike our from-scratch version above,
        which works directly on 1D arrays.
        """
    )
    return


@app.cell
def _(expquad, np, pm):
    check_x = np.linspace(0, 5, 6)
    check_x_2d = check_x.reshape(-1, 1)  # PyMC GP inputs are 2D: (n, 1)

    pymc_cov = (1.5**2 * pm.gp.cov.ExpQuad(1, ls=1.0))(check_x_2d).eval()
    scratch_cov = expquad(check_x, check_x, ls=1.0, eta=1.5)

    kernels_match = np.allclose(pymc_cov, scratch_cov)
    print(f"from-scratch ExpQuad matches pm.gp.cov.ExpQuad: {kernels_match}")
    assert kernels_match
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Drawing functions from a GP prior

        With a covariance function in hand, we can draw sample functions
        from a GP prior over a grid of $x$ values: evaluate $k(x, x')$ on
        the grid to get a covariance matrix $K$, then draw from
        $\mathcal{N}(0, K)$. Adjust the sliders below and watch how the
        lengthscale and amplitude reshape the sampled functions.
        """
    )
    return


@app.cell
def _(mo):
    lengthscale_slider = mo.ui.slider(
        0.1, 3.0, value=1.0, step=0.1, label="Lengthscale ℓ"
    )
    amplitude_slider = mo.ui.slider(0.1, 3.0, value=1.0, step=0.1, label="Amplitude η")
    mo.hstack([lengthscale_slider, amplitude_slider], gap=2)
    return amplitude_slider, lengthscale_slider


@app.cell
def _(PYMC_DARK_GREEN, amplitude_slider, expquad, go, lengthscale_slider, np, rng):
    gp_grid = np.linspace(0, 10, 200)
    gp_cov = expquad(
        gp_grid, gp_grid, ls=lengthscale_slider.value, eta=amplitude_slider.value
    )
    gp_cov = gp_cov + 1e-8 * np.eye(len(gp_grid))  # jitter for numerical stability

    n_draws = 5
    gp_draws = rng.multivariate_normal(np.zeros(len(gp_grid)), gp_cov, size=n_draws)

    gp_fig = go.Figure()
    for _i in range(n_draws):
        gp_fig.add_trace(
            go.Scatter(
                x=gp_grid,
                y=gp_draws[_i],
                mode="lines",
                name=f"draw {_i + 1}",
                line=dict(width=2),
            )
        )
    gp_fig.update_layout(
        title=(
            f"GP prior draws — lengthscale={lengthscale_slider.value:.1f}, "
            f"amplitude={amplitude_slider.value:.1f}"
        ),
        xaxis_title="x",
        yaxis_title="f(x)",
        template="plotly_white",
        showlegend=False,
    )
    gp_fig.update_traces(line_color=PYMC_DARK_GREEN, selector=dict(name="draw 1"))
    gp_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **What changed?** Increasing the lengthscale $\ell$ makes the drawn
        functions smoother and slower-varying (nearby points stay strongly
        correlated over longer distances); decreasing it makes them
        wigglier. The amplitude $\eta$ scales how far the functions wander
        vertically from the (zero) mean, without changing their smoothness.
        Try dragging the lengthscale slider down toward 0.1 and back up
        toward 3.0 to see the effect.

        This is the last piece we need: in the next notebook we'll put a
        prior on $\ell$ and $\eta$ themselves and condition a GP on the
        theophylline data.
        """
    )
    return


if __name__ == "__main__":
    app.run()
