import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")

with app.setup:
    import inspect
    import sys
    from pathlib import Path
    from time import perf_counter

    import arviz as az
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    import polars as pl
    import pymc as pm
    import xarray as xr
    from patsy import dmatrix
    from scipy.stats import gaussian_kde, norm

    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from inference_contract import eti_bounds, inference_health, posterior_subset

    PYMC_BLUE = "#154A72"
    PYMC_GREEN = "#81C240"
    PYMC_LIGHT_BLUE = "#4A9EDE"

    RANDOM_SEED = 42

    data_dir = project_root / "data"
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)


    def z(a):
        """Standardize an array: (a - mean) / population std."""
        return (a - a.mean()) / a.std(ddof=0)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Foundations: A PyMC Primer and an Introduction to Gaussian Processes

    Welcome to the first hour of the workshop. Nothing here assumes you
    have seen a Gaussian process before; it *does* assume you are
    comfortable reading a little probability notation and some Python.

    This notebook builds the working vocabulary the rest of the workshop
    relies on: the PyMC model-building API, and the Bayesian workflow ,
    specify a model, check its prior, sample, check convergence, interpret
    , exercised on two models whose hand-chosen functional forms fail in
    instructive ways. Notebook 2 builds the Gaussian process itself.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The Bayesian workflow in PyMC
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### The Bayesian paradigm

    Everything in this workshop is Bayesian, so it is worth stating the
    core idea plainly. Bayesian inference treats unknown quantities as
    **random variables with distributions** and updates those
    distributions in light of data. Three objects do all the work:

    - The **prior** $p(\theta)$, what we believe about the parameters
      $\theta$ *before* seeing the data. It encodes scale, sign,
      plausibility: "a concentration is positive and probably a
      single-digit number of mg/L", not "a concentration could be
      $10^{12}$".
    - The **likelihood** $p(y \mid \theta)$, how probable the observed
      data $y$ are for each candidate value of $\theta$. This is the
      model's description of the data-generating process.
    - The **posterior** $p(\theta \mid y)$, the updated belief about
      $\theta$ *after* folding in the data. It is what we report.

    These are tied together by **Bayes' rule**:

    $$p(\theta \mid y) = \frac{p(y \mid \theta)\, p(\theta)}{p(y)}
    \;\propto\; p(y \mid \theta)\, p(\theta).$$

    The denominator $p(y) = \int p(y \mid \theta)\,p(\theta)\,d\theta$ is
    just the constant that makes the posterior integrate to one; for
    inference we usually work with the proportionality on the right. In
    words: **posterior $\propto$ likelihood $\times$ prior**. The
    posterior is a *compromise* between what you believed and what the
    data say, data-rich regions pull it toward the likelihood, and where
    data are scarce the prior still speaks.

    Only for a few textbook "conjugate" models can you write the
    posterior in closed form. For everything else, including every GP in
    this workshop, we *sample* from the posterior instead, drawing many
    representative parameter values with Markov chain Monte Carlo (MCMC).
    PyMC does this for us. The figure below shows the one conjugate case
    we can compute by hand, purely to build intuition for the
    prior → posterior update: an unknown mean $\theta$ for a Normal
    whose standard deviation is *known*, which makes
    prior $\times$ likelihood $\to$ posterior exact.
    """)
    return


@app.cell(hide_code=True)
def _():
    prior_m0, prior_s0 = 0.0, 1.0  # prior: theta ~ Normal(0, 1)
    known_sd = 1.0
    fake_data = np.array([1.6, 2.1, 1.9, 2.4, 1.7])  # a small "sample"
    n_obs = len(fake_data)
    ybar = fake_data.mean()

    # Conjugate Normal-Normal update for the mean.
    post_var = 1.0 / (1.0 / prior_s0**2 + n_obs / known_sd**2)
    post_mean = post_var * (prior_m0 / prior_s0**2 + n_obs * ybar / known_sd**2)
    post_sd = np.sqrt(post_var)

    theta_grid = np.linspace(-2, 4, 400)
    prior_pdf = norm.pdf(theta_grid, prior_m0, prior_s0)
    # Likelihood as a function of theta (up to a constant), scaled to plot.
    like_pdf = norm.pdf(ybar, theta_grid, known_sd / np.sqrt(n_obs))
    post_pdf = norm.pdf(theta_grid, post_mean, post_sd)
    return like_pdf, post_mean, post_pdf, post_sd, prior_pdf, theta_grid


@app.cell(hide_code=True)
def _(like_pdf, post_pdf, prior_pdf, theta_grid):
    bayes_fig = go.Figure()
    bayes_fig.add_trace(
        go.Scatter(
            x=theta_grid,
            y=prior_pdf,
            mode="lines",
            name="prior p(θ)",
            line=dict(color=PYMC_LIGHT_BLUE, width=3),
        )
    )
    bayes_fig.add_trace(
        go.Scatter(
            x=theta_grid,
            y=like_pdf,
            mode="lines",
            name="likelihood p(y | θ)",
            line=dict(color=PYMC_GREEN, width=3, dash="dash"),
        )
    )
    bayes_fig.add_trace(
        go.Scatter(
            x=theta_grid,
            y=post_pdf,
            mode="lines",
            name="posterior p(θ | y)",
            line=dict(color=PYMC_BLUE, width=4),
        )
    )
    bayes_fig.update_layout(
        title="Bayes' rule in one dimension: posterior sits between prior and likelihood",
        xaxis_title="θ (the unknown mean)",
        yaxis_title="density",
        template="plotly_white",
    )
    bayes_fig
    return


@app.cell(hide_code=True)
def _(post_mean, post_sd):
    mo.md(f"""
    The **prior** (light blue) is centred
    at 0 and fairly broad. The **likelihood** (green, dashed) is centred
    near the data mean of about 1.9 and is tighter because five
    observations already pin the mean down reasonably well. The
    **posterior** (dark blue) lands *between* them, mean
    ≈ {post_mean:.2f}, sd ≈ {post_sd:.2f}, pulled most of the way toward
    the data but still nudged toward the prior, and narrower than either
    input because it combines both sources of information.

    Two lessons carry through the whole workshop. First, **the posterior
    is a compromise**, and where data are sparse the prior matters, a
    fact that becomes vivid for GPs, whose priors are over entire
    functions. Second, **priors have consequences you should check
    before you fit**, which is exactly what the prior predictive check
    below is for.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Example: Theophylline dataset

    Before building a model of it, meet the dataset properly.
    `Theoph` is a classic pharmacokinetic dataset built into R
    (Boeckmann, Sheiner & Beal 1994, *NONMEM Users Guide*): **12
    subjects** each received a single oral dose of the asthma drug
    theophylline, and their serum concentration was measured at **11 time
    points** over the following ~24 hours (**132 observations** in all).

    After an oral dose, concentration follows a characteristic
    **rise → peak → decay** shape. It first *rises* as the drug is
    absorbed from the gut into the bloodstream, reaches a *peak* when
    absorption and elimination balance, then *decays* roughly
    exponentially as the liver and kidneys clear it. That smooth,
    asymmetric, single-humped curve is a poor fit for any straight line ,
    yet a completely natural fit for a Gaussian process, which makes it
    the ideal running example for this workshop.

    The columns:

    - `subject`: subject id (1–12)
    - `time`: hours since dose
    - `conc`: serum theophylline concentration (mg/L)
    - `dose`: administered dose (mg/kg)
    - `weight`: subject body weight (kg)
    """)
    return


@app.cell(hide_code=True)
def _():
    theoph = pl.read_csv(data_dir / "theophylline.csv")
    theoph.head()
    return (theoph,)


@app.cell(hide_code=True)
def _(theoph):
    # One subject, sorted by time. We reuse this prep for the piecewise model below.
    subject_id = 1
    subject_df = theoph.filter(pl.col("subject") == subject_id).sort("time")

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
        subject_id,
        time_mean,
        time_std,
        time_vals,
        time_z,
    )


@app.cell(hide_code=True)
def _(theoph):
    eda_fig = go.Figure()
    _subjects = theoph["subject"].unique(maintain_order=True).to_list()
    for _sid in _subjects:
        _sdf = theoph.filter(pl.col("subject") == _sid).sort("time")
        eda_fig.add_trace(
            go.Scatter(
                x=_sdf["time"].to_list(),
                y=_sdf["conc"].to_list(),
                mode="markers+lines",
                name=f"subject {_sid}",
                line=dict(width=1),
                opacity=0.7,
            )
        )
    eda_fig.update_layout(
        title="All 12 theophylline subjects — the shared rise-peak-decay shape",
        xaxis_title="Time since dose (hours)",
        yaxis_title="Concentration (mg/L)",
        template="plotly_white",
    )
    eda_fig
    return


@app.cell(hide_code=True)
def _(subject_id):
    mo.md(f"""
    Every subject traces the same qualitative arc, a fast early rise to
    a peak within the first couple of hours, then a slow decay over the
    rest of the day, but the *height* of the peak, its *timing*, and the
    *rate* of decay differ from person to person (driven partly by dose
    and body weight). Later notebooks exploit that shared-shape-with-
    individual-variation structure directly with hierarchical GPs. For
    the rest of this section we focus on a **single subject** (subject
    {subject_id}) and ask: can a simple parametric curve capture even one
    of these traces?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The building blocks of a PyMC model

    Five ideas make up essentially the whole PyMC API this workshop uses;
    everything later, including the GP module, is composition of them.

    **A model is a container.** `with pm.Model() as model:` is a context manager
    that records every variable created inside it to create a *symbolic graph* of
    the relationships you declare; from
    that graph PyMC automatically derives the joint log-probability and its
    gradient, the two functions MCMC needs. You never write either one by
    hand.
    """)
    return


@app.cell
def _():
    with pm.Model() as demo_model:
        demo_mu = pm.Normal("mu", mu=0, sigma=1)
        demo_sigma = pm.HalfNormal("sigma", sigma=1)
    demo_model
    return (demo_mu,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Random variables are symbolic.** Running the cell above drew no
    numbers: `demo_mu` is a node in the graph, not a value. Every random
    variable supports exactly two operations, and all of Bayesian
    computation is built out of them:

    - `pm.draw`, **simulate** values from the variable (prior predictive
      sampling is this, applied to the whole model at once);
    - `pm.logp`, **score** a proposed value against the variable's
      density (summing these across the model gives the joint
      log-probability the sampler climbs).

    Both build graphs themselves; `.eval()` compiles and runs one, handy
    for spot checks like comparing PyMC's answer against scipy's:
    """)
    return


@app.cell
def _(demo_mu):
    pm.draw(demo_mu, draws=5, random_seed=RANDOM_SEED).round(2)
    return


@app.cell
def _(demo_mu):
    float(pm.logp(demo_mu, 0).eval())
    return


@app.cell
def _():
    norm.logpdf(0, loc=0, scale=1)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Observed variables are the likelihood.** Passing `observed=` pins a
    random variable to data. It still has a distribution, it is *scored*,
    contributing $p(\text{data} \mid \text{parents})$ to the joint, but
    the fitting algorithm does change its value. Every model therefore splits into
    `free_RVs`, the unknowns MCMC explores, and `observed_RVs`, the data
    terms that score them.

    Also note that `sigma` must be
    positive, so PyMC actually samples `sigma_log__`, its logarithm, on
    the unconstrained scale and transforms back automatically. That is why
    sampler output sometimes mentions variables with `_log__` or
    `_interval__` suffixes you never defined.
    """)
    return


@app.cell
def _(conc_z):
    with pm.Model() as demo_obs_model:
        demo_obs_mu = pm.Normal("mu", mu=0, sigma=1)
        demo_obs_sigma = pm.HalfNormal("sigma", sigma=1)
        pm.Normal("conc", mu=demo_obs_mu, sigma=demo_obs_sigma, observed=conc_z)

    demo_obs_model.free_RVs
    return (demo_obs_model,)


@app.cell
def _(demo_obs_model):
    demo_obs_model.observed_RVs
    return


@app.cell
def _(demo_obs_model):
    demo_obs_model.value_vars
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **`coords` and `dims` name the axes.** Variables can be vectors or
    matrices. Give the model `coords`, labels for each dimension, and
    pass `dims=` when declaring a variable, and the posterior arrives with
    named, labeled dimensions instead of anonymous integers: ArviZ plots
    come out labeled, and a shape mistake fails at model-build time with a
    message naming the dimension, rather than at sample time with a
    broadcasting error. One normal per subject, all twelve at once:
    """)
    return


@app.cell
def _(theoph):
    model_coords = {"subject": theoph["subject"].unique(maintain_order=True).to_list()}

    with pm.Model(coords=model_coords) as demo_dims_model:
        demo_subject_mean = pm.Normal("subject_mean", mu=0, sigma=1, dims="subject")

    pm.draw(demo_subject_mean, random_seed=RANDOM_SEED).shape
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Deterministic variables**: Any expression of model
    variables is itself a model quantity, adding no randomness beyond what
    its inputs carry. Written as a plain expression it stays *anonymous*:
    usable inside the model, but not recorded in the output. Wrapping it in
    `pm.Deterministic("name", expr)` records its value with every posterior
    draw, so a fitted quantity arrives alongside the parameters instead of
    having to be recomputed afterwards. Use the named form for anything you
    want to plot or report and leave intermediate algebra anonymous in order to reduce unnecessary storage.

    **Data nodes** Wrapping an input array in
    `pm.Data` places it inside the graph as a named, replaceable node.
    After fitting, `pm.set_data({"name": new_values})` swaps it, which is
    how a fitted model predicts at new inputs without being rebuilt.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### A piecewise-linear baseline: two slopes meeting at a peak

    Every model in this half of the notebook works on a **standardized**
    scale: `z()` subtracts a variable's mean and divides by its standard
    deviation, so `time_z` and `conc_z` are both unitless with mean 0 and
    sd 1. That keeps a prior scale like `HalfNormal(1)` comparable across
    variables with very different natural units, hours versus mg/L ,
    instead of being dogmatic in one unit and vague in another; we convert
    back to hours and mg/L only when interpreting results. (The spline
    model later works in raw age units instead, since age does not span
    the orders of magnitude that time and concentration do.)

    The simplest model that respects the rise-then-decay shape is a
    **piecewise-linear** (broken-stick) curve: a straight line going *up*
    during absorption, a kink at the peak, and a straight line going
    *down* during elimination. We do not know where the peak is, so we
    make its location a parameter, $\tau$, and let the data estimate it.

    A continuous one-knot curve makes the interpretation unambiguous. On
    standardized time $t$,

    $$
    \mu(t) =
    \begin{cases}
    \text{peak} - \text{rise}\cdot(\tau - t) & \text{if } t \le \tau
    \quad\text{(absorption)} \\
    \text{peak} - \text{decay}\cdot(t - \tau) & \text{if } t > \tau
    \quad\text{(elimination)}
    \end{cases}
    $$

    Both branches equal `peak` at $t=\tau$, so the curve is continuous at
    the knot.

    `peak` is the value at the knot, `rise` is positive before it, and
    `decay` is positive after it, so the post-knot slope is strictly negative.
    The slope priors are easiest to reason about in the original units and
    then converted, since the model works on the standardized axis: this
    drug absorbs on the order of 10 mg/L per hour and eliminates about
    twenty times slower, giving scales
    $s_{\text{rise}}=10\,t_{\text{sd}}/y_{\text{sd}}$ and
    $s_{\text{decay}}=0.5\,t_{\text{sd}}/y_{\text{sd}}$. We use

    $$
    \begin{aligned}
    \text{peak} &\sim \mathcal N(0, 1) \\
    \text{rise} &\sim \text{HalfNormal}(s_{\text{rise}}) \\
    \text{decay} &\sim \text{HalfNormal}(s_{\text{decay}}) \\
    \tau &\sim \text{Uniform}(t_{\min},\ t_{\max}) \\
    \sigma &\sim \text{HalfNormal}(0.5)
    \end{aligned}
    $$

    (PyMC's `Normal` and `HalfNormal` take a standard deviation as their
    scale argument, not a variance, $\mathcal N(0,1)$ above has sd 1, and
    the same convention holds wherever a Normal prior appears later in
    this notebook.)

    Setting a slope prior directly on the standardized scale is an easy way
    to be accidentally dogmatic: a `HalfNormal(1)` on `rise` would put the
    climb this data actually shows about twenty times the prior's own
    scale into the tail, and the fit would be strangled by the prior
    rather than by the functional form we are trying to interrogate.

    $\tau$ remains on the observed standardized-time domain and is converted
    to hours only for interpretation.
    """)
    return


@app.cell
def _(conc_std, conc_z, time_std, time_z):
    pw_coords = {"observation": np.arange(len(conc_z))}
    pw_rise_scale = 10.0 * time_std / conc_std
    pw_decay_scale = 0.5 * time_std / conc_std

    with pm.Model(coords=pw_coords) as pw_model:
        pw_time_data = pm.Data("time", time_z, dims="observation")
        pw_concentration_data = pm.Data(
            "concentration_data", conc_z, dims="observation"
        )
        t_lo, t_hi = float(time_z.min()), float(time_z.max())
    
        peak = pm.Normal("peak", mu=0, sigma=1)
        rise = pm.HalfNormal("rise", sigma=pw_rise_scale)
        decay = pm.HalfNormal("decay", sigma=pw_decay_scale)
        tau = pm.Uniform("tau", lower=t_lo, upper=t_hi)
        pw_sigma = pm.HalfNormal("sigma", sigma=0.5)

        mu_pw = (
            peak
            - rise * pm.math.maximum(tau - pw_time_data, 0.0)
            - decay * pm.math.maximum(pw_time_data - tau, 0.0)
        )
        pm.Deterministic("mu_pw", mu_pw, dims="observation")
        pm.Normal(
            "conc_obs",
            mu=mu_pw,
            sigma=pw_sigma,
            observed=pw_concentration_data,
            dims="observation",
        )

    pw_model
    return (pw_model,)


@app.cell
def _(pw_model):
    pm.model_to_graphviz(pw_model)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Prior predictive check

    Before fitting, we look at what the model implies, and because these
    priors are over a whole *curve*, that means whole trajectories, not
    just a spread of points.

    We draw parameters from the priors,
    evaluate $\mu(t)$ across the time grid for each, and plot the implied
    standardized-concentration trajectories against the standardized
    data.
    """)
    return


@app.cell
def _(pw_model):
    with pw_model:
        pw_prior_pred = pm.sample_prior_predictive(draws=400, random_seed=RANDOM_SEED)
    return (pw_prior_pred,)


@app.cell(hide_code=True)
def _(conc_mean, conc_std, conc_z, pw_prior_pred, time_z):
    pw_prior_curves = pw_prior_pred["prior"]["mu_pw"]
    pw_prior_mu_mgl = pw_prior_curves * conc_std + conc_mean
    pw_prior_obs_mgl = pw_prior_pred["prior_predictive"]["conc_obs"] * conc_std + conc_mean
    pw_negative_fraction = float((pw_prior_obs_mgl < 0).mean())
    pw_mean_negative_fraction = float((pw_prior_mu_mgl < 0).mean())
    _mu_nonneg_mask = pw_prior_mu_mgl.values >= 0
    pw_residual_likelihood_fraction = float(
        (pw_prior_obs_mgl.values[_mu_nonneg_mask] < 0).mean()
    )
    print(f"prior predictive draws below zero (mg/L): {pw_negative_fraction:.1%}")
    print(f"prior mean function below zero (mg/L): {pw_mean_negative_fraction:.1%}")
    print(
        "observations below zero given mean function >= 0: "
        f"{pw_residual_likelihood_fraction:.1%}"
    )
    _order = np.argsort(time_z)

    pw_prior_fig = go.Figure()
    _rng_plot = np.random.default_rng(0)
    _draw_indices = _rng_plot.choice(pw_prior_curves.sizes["draw"], size=60, replace=False)
    for _i in _draw_indices:
        _curve = pw_prior_curves.isel(chain=0, draw=int(_i))
        pw_prior_fig.add_trace(
            go.Scatter(
                x=time_z[_order],
                y=_curve.values[_order],
                mode="lines",
                line=dict(color=PYMC_LIGHT_BLUE, width=1),
                opacity=0.2,
                showlegend=False,
            )
        )
    pw_prior_fig.add_trace(
        go.Scatter(
            x=time_z[_order],
            y=conc_z[_order],
            mode="markers",
            marker=dict(color="black", size=8),
            name="observed (standardized)",
        )
    )
    pw_prior_fig.update_layout(
        title="Prior predictive piecewise curves vs. standardized observed data",
        xaxis_title="time (standardized)",
        yaxis_title="conc (standardized)",
        template="plotly_white",
    )
    pw_prior_fig
    return (
        pw_mean_negative_fraction,
        pw_negative_fraction,
        pw_prior_curves,
        pw_prior_mu_mgl,
        pw_residual_likelihood_fraction,
    )


@app.cell(hide_code=True)
def _(
    conc_z,
    pw_mean_negative_fraction,
    pw_negative_fraction,
    pw_prior_curves,
    pw_prior_mu_mgl,
    pw_residual_likelihood_fraction,
):
    mo.md(f"""
    **Plausibility check, and it fails.** The prior mean function spans
    roughly [{float(pw_prior_curves.min()):.1f}, {float(pw_prior_curves.max()):.1f}]
    on the standardized scale, against an observed range of
    [{conc_z.min():.2f}, {conc_z.max():.2f}]. Containing the observed range
    is not the criterion: on the original scale that low end is
    {float(pw_prior_mu_mgl.min()):.0f} mg/L. Nothing in the model pins the
    curve near zero at dose time, so `rise` extrapolated backward from a
    free $\\tau$ can pull the early part of the curve arbitrarily negative.
    A modeller who saw this would anchor $\\mu(0)$ at (or near) zero, so
    `rise` is set by the peak and its timing rather than left free, a
    change to the model itself, out of scope for this notebook.

    That same structural gap, not the likelihood, is why so many prior
    predictive draws are negative. **{pw_negative_fraction:.1%}** of the
    observations this model expects to generate fall below zero, and
    **{pw_mean_negative_fraction:.1%}** of the prior *mean* functions are
    already negative somewhere, essentially all of the problem is there
    before the likelihood adds any noise. Only
    **{pw_residual_likelihood_fraction:.1%}** of observations are negative
    when the mean function itself is non-negative, the most a
    positive-support likelihood could ever fix here, and a small share of
    the problem. The verdict of this check is about $\\mu(t)$: a mean
    function that reaches {float(pw_prior_mu_mgl.min()):.0f} mg/L is a
    property of the functional form, and no change of observation
    distribution can repair it.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.vstack(
        [
            mo.md(
                r"""**Question 1, prior sensitivity.** The `rise` prior above
                is scaled from a belief in mg/L per hour. Suppose you ignored
                the units and wrote `rise ~ HalfNormal(1)` straight onto the
                standardized axis. Before rerunning, predict the prior curves,
                whether the posterior could reach the observed peak, and where
                the unexplained variation would end up."""
            ),
            mo.accordion(
                {
                    "Solution": mo.md(
                        r"""The prior curves climb far too slowly: the observed
                        absorption slope is about 21 on the standardized scale,
                        roughly twenty times `HalfNormal(1)`'s own scale out,
                        so the likelihood cannot overcome it. The posterior for
                        `rise` piles up against the prior, the fitted curve
                        under-shoots the peak badly, and `sigma` inflates ,
                        from about 0.6 mg/L to 2.6 mg/L, to absorb the
                        misfit. $\tau$ then drifts anywhere along the day,
                        because no knot position helps a curve that cannot
                        climb."""
                    )
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Sampling

    Five parameters (`peak`, `rise`, `decay`, `tau`, `sigma`) over 11
    points. We use `draws=500, tune=500, chains=4` with PyMC's default
    sampler settings. The sharp knot is an intentional structural limitation,
    whose identifiability and predictive consequences we inspect rather than
    attempting to hide with sampler tuning.
    """)
    return


@app.cell
def _(pw_model):
    with pw_model:
        pw_idata = pm.sample(
            draws=500,
            tune=500,
            chains=4,
            random_seed=RANDOM_SEED,
        )
    pw_idata.to_netcdf(results_dir / "01_piecewise.nc")
    return (pw_idata,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Optimization, and why we do not stop there

    Sampling is not the only way to fit this model. `pm.find_MAP` runs an
    optimizer to the single most probable parameter vector, the *maximum a
    posteriori* estimate. It is fast, it is deterministic, and for a model
    this small it takes a fraction of a second.
    """)
    return


@app.cell
def _(pw_model):
    with pw_model:
        pw_map = pm.find_MAP(progressbar=False)
    return (pw_map,)


@app.cell(hide_code=True)
def _(pw_idata, pw_map):
    pw_posterior_means = {
        name: float(pw_idata["posterior"][name].mean())
        for name in ("peak", "rise", "decay", "tau", "sigma")
    }
    pl.DataFrame(
        {
            "parameter": list(pw_posterior_means),
            "MAP": [round(float(pw_map[name]), 3) for name in pw_posterior_means],
            "posterior mean": [round(value, 3) for value in pw_posterior_means.values()],
        }
    )
    return (pw_posterior_means,)


@app.cell(hide_code=True)
def _(pw_idata):
    # The marginal mode of sigma alone, not the sigma coordinate of the joint MAP.
    pw_sigma_samples = pw_idata["posterior"]["sigma"].values.ravel()
    pw_sigma_grid = np.linspace(pw_sigma_samples.min(), pw_sigma_samples.max(), 2000)
    pw_sigma_mode = float(pw_sigma_grid[np.argmax(gaussian_kde(pw_sigma_samples)(pw_sigma_grid))])
    pw_sigma_mode
    return (pw_sigma_mode,)


@app.cell(hide_code=True)
def _(pw_map, pw_posterior_means, pw_sigma_mode):
    mo.md(f"""
    Four of the five parameters agree to within a few percent. `sigma`
    does not: its joint MAP is **{pw_map["sigma"]:.3f}**, while its
    posterior mean is **{pw_posterior_means["sigma"]:.3f}**, a gap of
    roughly a third.

    That gap has two sources, and it is easy to collapse them into one:
    call them the **MAP-to-mode gap** (joint MAP up to the marginal's own
    mode) and the **mode-to-mean gap** (that mode up to the posterior
    mean). The *marginal* posterior for `sigma` alone, the density
    `az.plot_trace_dist` draws, has a long right tail with its own mode
    around **{pw_sigma_mode:.3f}**: already above the joint MAP, because
    with only 11 observations a scale parameter's marginal skews right,
    and the mode of a skewed density sits left of its mean. But
    {pw_sigma_mode:.3f} is still short of {pw_posterior_means["sigma"]:.3f}
    by about as much as the joint MAP is short of {pw_sigma_mode:.3f} ,
    the two gaps are roughly the same size. Marginal skew accounts for
    only the mode-to-mean gap. The MAP-to-mode gap is a different
    phenomenon: `find_MAP` optimizes the joint density over all five
    parameters at once, and the `sigma` coordinate of that joint optimum
    has no obligation to land where `sigma`'s own marginal happens to
    peak.

    This is the general problem with a point estimate, in miniature, and
    this model's own numbers show it concretely:
    **{pw_map["sigma"]:.3f}**, **{pw_sigma_mode:.3f}** and
    **{pw_posterior_means["sigma"]:.3f}** are three different answers to
    "what is `sigma`?", in a model with only five parameters. Add more
    parameters and it gets worse: with more directions for the joint
    optimum to be pulled away from any one coordinate's marginal peak,
    the mode can end up in a region carrying almost no probability mass,
    while the posterior itself lives in the volume around it.

    `find_MAP` earns its place as a fast check that a model compiles and
    lands somewhere sensible before you spend minutes sampling it. We
    use it that way in notebook 3, on a GP that takes real time to fit.
    It is not a substitute for the posterior.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Checking the fit before reading it

    Three quantities decide whether a posterior is worth interpreting, and
    they are the same three every time.

    **Divergences** count the transitions where the sampler's simulated
    trajectory broke down, almost always because the posterior has a
    geometry the sampler cannot follow at its adapted step size. Divergences
    are not noise to be tuned away: they mark regions the chains could not
    explore, so the posterior you get is biased toward the regions they
    could. The standard is zero.

    **$\widehat{R}$** compares the variance between chains with the variance
    within them. If the chains explored the same distribution, the ratio is
    near 1. Values above **1.01** mean the chains disagree, so at least one
    of them has not found the posterior yet.

    **Effective sample size** is how many independent draws your correlated
    chain is worth. It bounds the Monte Carlo error on everything you
    report: a posterior mean from an ESS of 40 carries roughly three times
    the error of one from an ESS of 400. We ask for at least **400** in both
    the bulk (for means and medians) and the tail (for interval endpoints).

    Compute all three for every free variable in the model, not just the
    one you care about, a badly behaved nuisance parameter contaminates
    the ones you plan to quote.
    """)
    return


@app.cell(hide_code=True)
def _(pw_idata, pw_model):
    pw_free_rv_names = [rv.name for rv in pw_model.free_RVs]
    pw_diagnostics = az.summary(
        pw_idata,
        var_names=pw_free_rv_names,
        kind="diagnostics",
        round_to="none",
    )
    pw_divergences = int(pw_idata["sample_stats"]["diverging"].sum())
    pw_max_rhat = float(pw_diagnostics["r_hat"].astype(float).max())
    pw_min_ess_bulk = float(pw_diagnostics["ess_bulk"].min())
    pw_min_ess_tail = float(pw_diagnostics["ess_tail"].min())
    return pw_divergences, pw_max_rhat, pw_min_ess_bulk, pw_min_ess_tail


@app.cell
def _(pw_idata):
    az.summary(
        pw_idata,
        var_names=["peak", "rise", "decay", "tau", "sigma"],
        kind="diagnostics",
    ).round(4)
    return


@app.cell(hide_code=True)
def _(pw_divergences, pw_max_rhat, pw_min_ess_bulk, pw_min_ess_tail):
    mo.md(f"""
    This fit reports **{pw_divergences} divergences**, a maximum
    $\\widehat{{R}}$ of **{pw_max_rhat:.3f}**, and a minimum effective
    sample size of **{pw_min_ess_bulk:.0f}** in the bulk and
    **{pw_min_ess_tail:.0f}** in the tail. Every one of those clears the
    standard above, so the sampler did its job on this model.

    That is a narrower claim than it sounds. These checks ask whether the
    chains explored the posterior of *the model as written*. They say
    nothing about whether that model can represent the data, the piecewise
    form could be hopeless and these three numbers would look exactly the
    same.
    """)
    return


@app.cell
def _(pw_idata):
    az.summary(pw_idata, var_names=["peak", "rise", "decay", "tau", "sigma"])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The default `az.summary` adds the estimates to the diagnostics: `mean`
    and `sd` for each parameter, and `eti89_lb` / `eti89_ub`, the lower
    and upper bounds of the **89% equal-tailed interval** (ArviZ 1.2's
    default is `ci_kind="eti"`, `ci_prob=0.89`). An equal-tailed interval is
    just the 5.5th and 94.5th percentiles of the posterior: 5.5% of the
    mass sits below the lower bound, 5.5% above the upper one, the same
    quantity `eti_bounds` computes by hand elsewhere in this workshop. A
    highest-density interval is a different, generally narrower
    construction for skewed posteriors, and this table does not report
    one.

    Read the table in this order every time: diagnostics first, and only if
    they pass do the estimates mean anything. A tidy `mean` column from
    chains that never converged is a number with no referent.
    """)
    return


@app.cell
def _(pw_idata):
    trace_plot = az.plot_trace_dist(
        pw_idata,
        var_names=["peak", "rise", "decay", "tau", "sigma"],
        compact=True,
        figure_kwargs={"figsize": (5, 3.5)},
    )
    rank_plot = az.plot_rank(
        pw_idata,
        var_names=["peak", "rise", "decay", "tau", "sigma"],
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


@app.cell(hide_code=True)
def _(conc_mean, conc_std, pw_idata, time_mean, time_std, time_vals):
    # Reconstruct mu(t) on a fine grid for every posterior draw, in original units.
    pw_posterior = pw_idata["posterior"]
    time_grid = np.linspace(time_vals.min(), time_vals.max(), 200)
    time_grid_z = xr.DataArray(
        (time_grid - time_mean) / time_std,
        dims="time_grid",
        coords={"time_grid": time_grid},
    )
    pw_pre_peak = np.maximum(pw_posterior["tau"] - time_grid_z, 0.0)
    pw_post_peak = np.maximum(time_grid_z - pw_posterior["tau"], 0.0)
    mu_orig = (
        pw_posterior["peak"] - pw_posterior["rise"] * pw_pre_peak - pw_posterior["decay"] * pw_post_peak
    ) * conc_std + conc_mean

    pw_fit_mean = mu_orig.mean(dim=("chain", "draw"))
    pw_fit_lo, pw_fit_hi = eti_bounds(mu_orig)
    return mu_orig, pw_fit_hi, pw_fit_lo, pw_fit_mean, time_grid


@app.cell(hide_code=True)
def _(
    conc_vals,
    mu_orig,
    pw_fit_hi,
    pw_fit_lo,
    pw_fit_mean,
    time_grid,
    time_vals,
):
    pw_fit_fig = go.Figure()
    pw_fit_fig.add_trace(
        go.Scatter(
            x=np.concatenate([time_grid, time_grid[::-1]]),
            y=np.concatenate([pw_fit_hi.values, pw_fit_lo.values[::-1]]),
            fill="toself",
            fillcolor="rgba(21,74,114,0.25)",
            line=dict(color="rgba(255,255,255,0)"),
            name="89% ETI",
        )
    )

    _stacked = mu_orig.stack(sample=("chain", "draw"))
    _draw_choice = np.random.default_rng(0).choice(
        _stacked.sizes["sample"], size=60, replace=False
    )
    for _rank, _sample in enumerate(_draw_choice):
        pw_fit_fig.add_trace(
            go.Scatter(
                x=time_grid,
                y=_stacked.isel(sample=int(_sample)).values,
                mode="lines",
                line=dict(color=PYMC_LIGHT_BLUE, width=1),
                opacity=0.35,
                showlegend=_rank == 0,
                name="posterior draws",
            )
        )
    pw_fit_fig.add_trace(
        go.Scatter(
            x=time_grid,
            y=pw_fit_mean.values,
            mode="lines",
            name="posterior mean fit",
            line=dict(color=PYMC_GREEN, width=3),
        )
    )
    pw_fit_fig.add_trace(
        go.Scatter(
            x=time_vals,
            y=conc_vals,
            mode="markers",
            name="observed",
            marker=dict(color=PYMC_BLUE, size=9),
        )
    )
    pw_fit_fig.update_layout(
        title="Piecewise-linear fit — a sharp corner the drug does not have",
        xaxis_title="Time since dose (hours)",
        yaxis_title="Concentration (mg/L)",
        template="plotly_white",
    )
    pw_fit_fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The individual draws, not the averaged line, are what the model can
    actually represent: each one is a broken stick with a single knot.
    Averaging them rounds the corner off, which makes the fit look
    smoother than any function the model can actually produce.
    """)
    return


@app.cell
def _(pw_idata, pw_model):
    with pw_model:
        pw_ppc = pm.sample_posterior_predictive(
            posterior_subset(pw_idata),
            var_names=["conc_obs"],
            random_seed=RANDOM_SEED,
        )
    pw_idata["posterior_predictive"] = pw_ppc["posterior_predictive"]
    pw_idata.to_netcdf(results_dir / "01_piecewise.nc")
    return (pw_ppc,)


@app.cell(hide_code=True)
def _(conc_z, pw_ppc):
    pw_ppc_draws = pw_ppc["posterior_predictive"]["conc_obs"]
    pw_ppc_mean = pw_ppc_draws.mean(dim=("chain", "draw"))
    pw_residuals = conc_z - pw_ppc_mean
    pw_rmse_draws = ((pw_ppc_draws - conc_z) ** 2).mean(dim="observation") ** 0.5
    pw_ppc_table = {
        "replicate_RMSE_5.5%": float(pw_rmse_draws.quantile(0.055)),
        "replicate_RMSE_50%": float(pw_rmse_draws.quantile(0.5)),
        "replicate_RMSE_94.5%": float(pw_rmse_draws.quantile(0.945)),
    }
    pl.DataFrame(
        {
            "quantile": ["5.5%", "50%", "94.5%"],
            "replicate RMSE": [round(value, 3) for value in pw_ppc_table.values()],
        }
    )
    return pw_ppc_table, pw_residuals


@app.cell
def _(pw_ppc):
    az.plot_ppc_dist(
        pw_ppc,
        var_names=["conc_obs"],
        num_samples=50,
    )
    return


@app.cell(hide_code=True)
def _(pw_residuals, time_vals):
    pw_residual_fig = go.Figure()
    pw_residual_fig.add_hline(y=0, line_dash="dash", line_color="black")
    pw_residual_fig.add_trace(
        go.Scatter(
            x=time_vals,
            y=pw_residuals.values,
            mode="markers+lines",
            marker=dict(color=PYMC_BLUE, size=9),
            name="observed minus PPC mean",
        )
    )
    pw_residual_fig.update_layout(
        title="Piecewise posterior-predictive residuals show remaining curvature",
        xaxis_title="Time since dose (hours)",
        yaxis_title="Standardized concentration residual",
        template="plotly_white",
    )
    pw_residual_fig
    return


@app.cell(hide_code=True)
def _(pw_ppc_table, pw_residuals):
    mo.md(f"""
    The posterior-predictive RMSE across replicate datasets is printed
    above (median {pw_ppc_table["replicate_RMSE_50%"]:.2f} on the
    standardized scale, 89% range {pw_ppc_table["replicate_RMSE_5.5%"]:.2f}
    to {pw_ppc_table["replicate_RMSE_94.5%"]:.2f}). More revealingly, the
    residual sequence ranges from {float(pw_residuals.min()):.2f} to
    {float(pw_residuals.max()):.2f} and retains systematic curvature
    around the sharp knot. This is a model discrepancy, not a
    sampler-tuning problem: the next section explains why a smooth
    functional prior is needed.
    """)
    return


@app.cell(hide_code=True)
def _(tau_hi, tau_lo):
    mo.md(rf"""
    ### Diagnosis: why the piecewise model is inadequate

    **1. The kink is unphysical.** A real absorption/elimination curve is
    *smooth*, it eases through its peak, it does not turn a hard corner.
    The broken stick puts an infinitely sharp vertex at $\tau$, which no
    drug's pharmacokinetics actually do. The model can only ever
    approximate a smooth hump with two straight lines and an angle.

    **2. The knot is confidently placed and still an artifact.** The data
    pin $\tau$ down tightly: the 89% interval computed below is only
    {tau_hi - tau_lo:.2f} hours wide (about {(tau_hi - tau_lo) * 60:.0f}
    minutes), and the light-blue draws above all break at nearly the same
    place, so the posterior mean carries the corner too. But a sharp
    estimate is only the model answering the question it was built to ask.
    This functional form *requires* a corner to exist somewhere; the
    sampler obliges and reports the best available location with tight
    uncertainty. Nothing in that number tests whether the drug has a
    corner at all. The plot below shows how narrow the interval is, which
    is the point: the piecewise model states a peak time more confidently
    than the shape of the data warrants.

    **3. Straight segments miss the curvature.** Between the knots the
    model is forced to be exactly linear, so it *undershoots* the rounded
    top of the rise and cannot follow the gentle concave-up flattening of
    the decay tail. The residual structure you can see around the fitted
    line is the curvature the model has no vocabulary for.

    Ultimately, **we had to choose a functional form, and the form
    we chose is wrong in ways the data cannot address.** We could keep
    patching, add more change points, swap in an exponential decay, bolt on an
    absorption compartment, but each patch is another hand-specified
    commitment. What we actually want is a model that says only "the
    function is smooth" and lets the data supply the shape.
    """)
    return


@app.cell(hide_code=True)
def _(pw_idata, time_mean, time_std):
    # Convert the tau posterior back to hours for interpretation.
    tau_hours = pw_idata["posterior"]["tau"] * time_std + time_mean
    tau_lo, tau_hi = (float(value) for value in eti_bounds(tau_hours))

    tau_fig = go.Figure()
    tau_fig.add_trace(
        go.Histogram(
            x=tau_hours.values.ravel(),
            histnorm="probability density",
            marker=dict(color=PYMC_GREEN),
            opacity=0.75,
            name="τ posterior",
        )
    )
    tau_fig.add_vline(x=float(tau_lo), line=dict(color="black", dash="dash"))
    tau_fig.add_vline(x=float(tau_hi), line=dict(color="black", dash="dash"))
    tau_fig.update_layout(
        title="Posterior of the peak time τ (hours) — a narrow interval for a corner that is not real",
        xaxis_title="Estimated peak time τ (hours)",
        yaxis_title="density",
        template="plotly_white",
    )
    tau_fig
    return tau_hi, tau_lo


@app.cell(hide_code=True)
def _(tau_hi, tau_lo):
    mo.md(f"""
    The 89% posterior interval for the peak time runs from about
    {tau_lo:.2f} to {tau_hi:.2f} hours, a span of only
    {tau_hi - tau_lo:.2f} hours. That
    precision is worth examining critically: it is a confident answer to the
    question *where is the corner*, asked of a curve that has no corner.
    A narrow posterior reports how well a parameter is determined **given
    the model**, not whether the model deserves the parameter.
    """)
    return


@app.cell(hide_code=True)
def _(tau_hi, tau_lo):
    mo.vstack(
        [
            mo.md(
                rf"""**Question 3, a sharp estimate of the wrong thing.**
                The posterior for $\tau$ above is narrow, only
                {tau_hi - tau_lo:.2f} hours wide. Does that narrowness
                license the claim "the peak occurs at $\tau$ hours"? What
                would you have to check first, and what would the same plot
                look like under a model with no knot at all?"""
            ),
            mo.accordion(
                {
                    "Solution": mo.md(
                        r"""No. The interval is conditional on a functional
                        form that forces a corner to exist somewhere, so the
                        sampler must place one, and it places it precisely.
                        Check the posterior-predictive residuals for the
                        curvature the two straight segments cannot follow
                        before quoting the number. A smooth model has no
                        $\tau$ to report at all: it estimates the whole curve
                        and lets the peak be wherever the function turns over,
                        with uncertainty that reflects the sparse data near
                        the top."""
                    )
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## A more flexible basis: splines

    The piecewise model failed in a specific way: we picked the shape, and
    the shape was wrong. The obvious response is a more flexible shape.

    **Spline models** are the standard answer, the classic first stop in
    *nonparametric* regression, where the curve's shape is meant to come
    from the data rather than from a formula we assert. A spline is a sum
    of piecewise polynomials, one per region of the input axis, tied
    together at boundaries called **knots**, so the pieces meet smoothly
    instead of at a corner. Instead of one hinge we get many, and instead
    of straight segments we get cubics. The machinery stays modest: once
    the basis is built, the model is an ordinary linear regression on the
    transformed predictors. But note what that means, with the knots and
    degree fixed, we are back to a finite set of weights, and the model's
    flexibility grows only if *we* add basis functions by hand.

    We switch datasets here, to one where the shape of the curve is the
    scientific question rather than a nuisance.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Swing decisions by age

    The data are 2023 batter grades: for each batter, a **swing decision**
    score summarizing how well they chose which pitches to offer at. We
    restrict to right-handed throwers with more than 100 plate appearances,
    so that every grade rests on a reasonable sample.

    The question is how swing decision varies with **age**. There is an
    obvious story, young players are raw, experience helps, and eventually
    reflexes decline, but no formula that says what the curve looks like.
    That is the same predicament as the theophylline curve, on data where
    we cannot fall back on pharmacology.

    One caveat worth stating plainly: this is a single season's snapshot,
    and players who stop hitting stop appearing. The old players in these
    data are the ones who were good enough to still be playing, which
    flatters the right-hand end of any curve we fit. We are ignoring that
    selection effect, not solving it.

    A second caveat matters more, and it is visible in the data. These
    batters are spread across six levels of competition, from rookie
    ball to the majors, and level is very nearly a proxy for age: the
    17-to-21 group is mostly Rookie, A and A+ ball (90.6% of it) and
    averages -0.56, while the 30-and-over group is almost all AAA and
    MLB and averages +0.05. Rookie-level batters average -1.17 on this grade against
    about -0.03 in the majors.

    So a curve fit to age alone is not measuring aging. Much of the
    climb across the young end is players moving up levels, graded
    against progressively better pitching, and we cannot separate the
    two from this snapshot. We fit it anyway, because the point here is
    the *shape* of an unknown function and how a model represents it ,
    but the parameter is age, and the interpretation is not "getting
    older makes you better at this".
    """)
    return


@app.cell(hide_code=True)
def _():
    # Batter swing-decision grades, 2023. One row per batter-season-level.
    swing_decisions = (
        pl.read_csv(data_dir / "batter_grades_2023.csv")
        .filter((pl.col("throws") == "R") & (pl.col("n_pa") > 100))
        .select("batter_id", "batter", "age", "swing_decision")
        .drop_nulls()
    )
    swing_ages = np.sort(swing_decisions["age"].unique().to_numpy())
    swing_decisions.head()
    return swing_ages, swing_decisions


@app.cell(hide_code=True)
def _(swing_decisions):
    swing_fig = go.Figure()
    swing_fig.add_trace(
        go.Scatter(
            x=swing_decisions["age"].to_numpy(),
            y=swing_decisions["swing_decision"].to_numpy(),
            mode="markers",
            marker=dict(color=PYMC_BLUE, size=5, opacity=0.3),
            name="batter-season-levels",
        )
    )
    swing_fig.update_layout(
        title="Swing decision grade by age, 2023",
        xaxis_title="Age (years)",
        yaxis_title="Swing decision grade",
        template="plotly_white",
        showlegend=False,
    )
    swing_fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Choosing knots

    A spline needs knots: the ages where one polynomial piece hands over to
    the next. Their number sets the model's flexibility, and their
    placement sets *where* that flexibility goes.

    We take seven quantiles of the observed ages. The two extremes are the
    boundary; the five interior quantiles are the internal knots. Using
    quantiles rather than an even grid puts the knots where the data are ,
    more resolution through the crowded middle of the age range, less out
    in the sparse tails where there is nothing to resolve.
    """)
    return


@app.cell(hide_code=True)
def _(swing_decisions):
    num_knots = 7
    knot_list = np.quantile(swing_decisions["age"].to_numpy(), np.linspace(0, 1, num_knots))
    knot_list
    return (knot_list,)


@app.cell(hide_code=True)
def _(knot_list, swing_decisions):
    knot_fig = go.Figure()
    knot_fig.add_trace(
        go.Scatter(
            x=swing_decisions["age"].to_numpy(),
            y=swing_decisions["swing_decision"].to_numpy(),
            mode="markers",
            marker=dict(color=PYMC_BLUE, size=5, opacity=0.3),
            showlegend=False,
        )
    )
    for _knot in knot_list:
        knot_fig.add_vline(x=float(_knot), line=dict(color="grey", width=1, dash="dash"))
    knot_fig.update_layout(
        title="Knot placement over the observed ages",
        xaxis_title="Age (years)",
        yaxis_title="Swing decision grade",
        template="plotly_white",
    )
    knot_fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `patsy` turns knots into the **basis matrix** $B$. Each column is one
    basis function evaluated at every age, and `degree=3` makes them cubic.

    `include_intercept=True` keeps the full set of basis functions, and that
    has a consequence worth stating now because it decides how the model is
    written. At every age these columns sum to exactly one. A basis with
    that property already contains a constant: any overall level the curve
    needs can be produced by raising all nine weights together.

    So the model below has **no separate intercept**. Adding one would not
    be merely redundant in the loose sense, it would be exactly redundant,
    since adding $c$ to the intercept and subtracting $c$ from every weight
    leaves the fitted curve unchanged to machine precision. The posterior
    would contain a perfectly flat ridge that no amount of data could pin
    down, and the only thing holding the parameters in place would be their
    priors.

    That failure is quiet. It produces no divergences and a healthy-looking
    $\widehat{R}$; the only symptom is an effective sample size far below
    what the number of draws should buy, because the sampler spends its time
    sliding along the ridge instead of exploring. Keep that in mind
    whenever you read the ESS column for a basis-expansion model.

    That quietness is specific to an **exact** tie like this one. A model
    that is merely *near*-collinear rather than exactly redundant can
    instead announce itself as a real $\widehat{R}$ failure, so do not
    take "ESS is the only symptom" as a general rule for collinearity.

    We evaluate the basis on the *unique* ages rather than on all
    observations, because every batter of the same age gets the same basis
    row. The model indexes into it.
    """)
    return


@app.cell(hide_code=True)
def _(knot_list, swing_ages):
    spline_basis = np.asarray(
        dmatrix(
            "bs(age, knots=knots, degree=3, include_intercept=True) - 1",
            {"age": swing_ages, "knots": knot_list[1:-1]},
        ),
        order="F",
    )
    print(f"basis shape: {spline_basis.shape}, row sums span "
          f"{spline_basis.sum(axis=1).min():.4f} to {spline_basis.sum(axis=1).max():.4f}")
    spline_basis.shape
    return (spline_basis,)


@app.cell(hide_code=True)
def _(spline_basis, swing_ages):
    basis_fig = go.Figure()
    for _j in range(spline_basis.shape[1]):
        basis_fig.add_trace(
            go.Scatter(
                x=swing_ages,
                y=spline_basis[:, _j],
                mode="lines",
                name=f"{_j}",
                line=dict(width=2),
            )
        )
    basis_fig.update_layout(
        title="The cubic B-spline basis: one curve per weight",
        xaxis_title="Age (years)",
        yaxis_title="Basis function value",
        template="plotly_white",
        legend=dict(title="Basis index", orientation="h", y=1.02, yanchor="bottom"),
    )
    basis_fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Each curve is one column of $B$, and each will get its own weight. A
    curve's height at a given age is how much that weight influences the
    fit there, so a weight only affects the region where its basis
    function is non-zero. That locality is the whole point: the fit can
    bend in one part of the age range without dragging the rest with it.

    Where two curves overlap, both weights contribute, and the transition
    between regions is smooth rather than a corner. Those overlaps are the
    knots doing their work.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### The spline model

    With the basis in hand the model is an ordinary linear regression:

    $$
    \begin{aligned}
    G_i &\sim \mathcal{N}(\mu_{a(i)},\ \sigma) \\
    \mu &= B w \\
    w_j &\sim \mathcal{N}(0, 3) \\
    \sigma &\sim \text{Exponential}(1)
    \end{aligned}
    $$

    $G_i$ is batter $i$'s swing decision grade and $a(i)$ is their age. The
    mean is the basis matrix times a weight vector $w$, one weight per
    basis function, and no intercept, for the reason given above. The priors
    are deliberately weak: we are not claiming to know the shape, only its
    scale. (As with the Normal priors earlier, that 3 is a standard
    deviation, not a variance.)

    Note what is *not* in this model: any statement about smoothness. The
    weights are independent under the prior. Whatever smoothness the fit
    shows comes from the basis functions overlapping, not from the model
    believing neighbouring ages should be similar. That distinction is what
    notebook 2 is about.
    """)
    return


@app.cell
def _(spline_basis, swing_ages, swing_decisions):
    # Map each observation to its row in the basis matrix.
    age_index = np.searchsorted(swing_ages, swing_decisions["age"].to_numpy())

    spline_coords = {
        "basis": np.arange(spline_basis.shape[1]),
        "age_grid": swing_ages,
        "observation": np.arange(swing_decisions.height),
    }
    with pm.Model(coords=spline_coords) as spline_model:
        basis_data = pm.Data("basis_matrix", spline_basis, dims=("age_grid", "basis"))
        age_index_data = pm.Data("age_index", age_index, dims="observation")
        grade_data = pm.Data(
            "grade", swing_decisions["swing_decision"].to_numpy(), dims="observation"
        )

        weights = pm.Normal("weights", mu=0, sigma=3, dims="basis")
        spline_sigma = pm.Exponential("sigma", 1)

        # No intercept: the basis columns sum to one at every age, so they span it.
        mu_age = pm.Deterministic(
            "mu_age", pm.math.dot(basis_data, weights), dims="age_grid"
        )
        pm.Normal(
            "grade_obs",
            mu=mu_age[age_index_data],
            sigma=spline_sigma,
            observed=grade_data,
            dims="observation",
        )

    spline_model
    return (spline_model,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Prior predictive check

    Same discipline as before: before the data ever touches the model, we
    draw a full set of basis weights from their priors, push each draw
    through the basis matrix, and see what curves fall out. The comparison
    against the observed grades tells us whether "weak" priors are
    actually weak, or secretly ruling out the data we are about to fit.
    """)
    return


@app.cell
def _(spline_model, swing_decisions):
    with spline_model:
        spline_prior_pred = pm.sample_prior_predictive(draws=200, random_seed=RANDOM_SEED)

    spline_prior_curves = spline_prior_pred["prior"]["mu_age"]
    print(
        "prior mu_age spans "
        f"[{float(spline_prior_curves.min()):.1f}, {float(spline_prior_curves.max()):.1f}]"
        f" vs observed [{swing_decisions['swing_decision'].min():.1f}, "
        f"{swing_decisions['swing_decision'].max():.1f}]"
    )
    return (spline_prior_curves,)


@app.cell(hide_code=True)
def _(spline_prior_curves, swing_ages, swing_decisions):
    spline_prior_fig = go.Figure()
    _stacked_prior = spline_prior_curves.stack(sample=("chain", "draw"))
    _prior_choice = np.random.default_rng(0).choice(
        _stacked_prior.sizes["sample"], size=60, replace=False
    )
    for _rank, _sample in enumerate(_prior_choice):
        spline_prior_fig.add_trace(
            go.Scatter(
                x=swing_ages,
                y=_stacked_prior.isel(sample=int(_sample)).values,
                mode="lines",
                line=dict(color=PYMC_LIGHT_BLUE, width=1),
                opacity=0.25,
                showlegend=_rank == 0,
                name="prior draws",
            )
        )
    spline_prior_fig.add_trace(
        go.Scatter(
            x=swing_decisions["age"].to_numpy(),
            y=swing_decisions["swing_decision"].to_numpy(),
            mode="markers",
            marker=dict(color=PYMC_BLUE, size=4, opacity=0.3),
            name="observed",
        )
    )
    spline_prior_fig.update_layout(
        title="Prior predictive spline curves against the observed grades",
        xaxis_title="Age (years)",
        yaxis_title="Swing decision grade",
        template="plotly_white",
    )
    spline_prior_fig
    return


@app.cell
def _(spline_model):
    with spline_model:
        spline_start = perf_counter()
        spline_idata = pm.sample(
            draws=500, tune=500, chains=4, random_seed=RANDOM_SEED
        )
        spline_seconds = perf_counter() - spline_start

    spline_idata.to_netcdf(results_dir / "01_spline.nc")
    print(f"spline sampling wall-time: {spline_seconds:.1f}s")
    return (spline_idata,)


@app.cell(hide_code=True)
def _(spline_model):
    spline_free_rv_names = [rv.name for rv in spline_model.free_RVs]
    return (spline_free_rv_names,)


@app.cell
def _(spline_free_rv_names, spline_idata):
    spline_diagnostics = az.summary(
        spline_idata,
        var_names=spline_free_rv_names,
        kind="diagnostics",
        round_to="none",
    )
    spline_diagnostics.round(4)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### The same three checks, factored

    We have now written the same block twice, once for the piecewise fit,
    once for the spline. It will be needed for every model in this
    workshop, so it lives in `inference_contract.py` at the project root as
    `inference_health`.
    """)
    return


@app.cell(hide_code=True)
def _(spline_idata, spline_model):
    spline_health_summary, spline_health_passed = inference_health(
        spline_idata, spline_model
    )
    print(f"health passed: {spline_health_passed}")
    spline_health_summary.round(4)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Same numbers, one call. From here on the workshop uses
    `inference_health`, but nothing is hidden by it: it is the block you
    just read, applied to every free variable in the model, returning the
    diagnostics table and a single pass/fail.

    It is worth being clear about what "passed" means. It means the sampler
    explored this model's posterior adequately. It does not mean the model
    is right, and the rest of this workshop is largely about the difference.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### What the weights did

    The clearest way to see how a spline works is to plot each basis
    function multiplied by its posterior mean weight. Each coloured curve is
    one term of the sum; the black curve is the sum itself, which is the fitted $\mu$.
    """)
    return


@app.cell(hide_code=True)
def _(spline_basis, spline_idata):
    weight_means = spline_idata["posterior"]["weights"].mean(dim=("chain", "draw")).values
    weighted_basis = spline_basis * weight_means
    spline_mu_curve = weighted_basis.sum(axis=1)
    return spline_mu_curve, weighted_basis


@app.cell(hide_code=True)
def _(knot_list, spline_mu_curve, swing_ages, weighted_basis):
    weighted_fig = go.Figure()
    for _j in range(weighted_basis.shape[1]):
        weighted_fig.add_trace(
            go.Scatter(
                x=swing_ages,
                y=weighted_basis[:, _j],
                mode="lines",
                name=f"{_j}",
                line=dict(width=2),
            )
        )
    weighted_fig.add_trace(
        go.Scatter(
            x=swing_ages,
            y=spline_mu_curve,
            mode="lines",
            name="sum",
            line=dict(color="black", width=3),
        )
    )
    for _knot in knot_list:
        weighted_fig.add_vline(x=float(_knot), line=dict(color="grey", width=1, dash="dash"))
    weighted_fig.update_layout(
        title="Each basis function times its posterior mean weight, and their sum",
        xaxis_title="Age (years)",
        yaxis_title="Contribution to the mean",
        template="plotly_white",
        legend=dict(title="Basis index", orientation="h", y=1.02, yanchor="bottom"),
    )
    weighted_fig
    return


@app.cell(hide_code=True)
def _():
    mo.vstack(
        [
            mo.md(
                r"""**Question 4, how many knots?** We chose seven knots at
                age quantiles. Predict what changes if you refit with
                `num_knots = 20`: the shape of the fitted curve, the width of
                the 89% band, and the diagnostics. Then say which of those
                three would tell you the model had become too flexible, and
                which would look fine either way."""
            ),
            mo.accordion(
                {
                    "Solution": mo.md(
                        r"""The curve gains wiggles that track individual
                        clusters of batters rather than a trend, and the band
                        narrows where the extra knots sit, more parameters fit
                        the observed points more closely. The diagnostics are
                        the ones that will *not* warn you: with weak priors and
                        overlapping basis functions, more knots typically means
                        weaker identification of individual weights, so ESS may
                        drop, but a fit that overfits happily can still show
                        zero divergences and $\widehat{R}$ of 1.00. Flexibility
                        is not a sampling problem, so sampling diagnostics do
                        not detect it. Posterior-predictive checks on held-out
                        ages would."""
                    )
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Predicting at new inputs

    The model was fit at the unique ages present in the data, 24 integer
    values. To draw a smooth curve we want $\mu$ on a fine grid between
    them, which means feeding the model inputs it never saw.

    Because the basis went in as `pm.Data`, we can swap it with
    `pm.set_data` and sample the posterior predictive again, no
    refitting, no second model definition. The posterior draws are already
    in hand; we are only pushing them through the model at new inputs.
    """)
    return


@app.cell
def _(knot_list, spline_idata, spline_model, swing_ages):
    spline_pred_ages = np.linspace(swing_ages.min(), swing_ages.max(), 200)
    spline_pred_basis = np.asarray(
        dmatrix(
            "bs(age, knots=knots, degree=3, include_intercept=True) - 1",
            {"age": spline_pred_ages, "knots": knot_list[1:-1]},
        ),
        order="F",
    )
    with spline_model:
        pm.set_data({"basis_matrix": spline_pred_basis}, coords={"age_grid": spline_pred_ages})
        spline_post_pred = pm.sample_posterior_predictive(
            spline_idata, var_names=["mu_age"], random_seed=RANDOM_SEED
        )

    spline_mu_draws = spline_post_pred["posterior_predictive"]["mu_age"]
    spline_mu_mean = spline_mu_draws.mean(dim=("chain", "draw")).values
    spline_mu_lo, spline_mu_hi = (
        endpoint.values for endpoint in eti_bounds(spline_mu_draws)
    )
    return spline_mu_hi, spline_mu_lo, spline_mu_mean, spline_pred_ages


@app.cell(hide_code=True)
def _(
    knot_list,
    spline_mu_hi,
    spline_mu_lo,
    spline_mu_mean,
    spline_pred_ages,
    swing_decisions,
):
    spline_fit_fig = go.Figure()
    spline_fit_fig.add_trace(
        go.Scatter(
            x=np.concatenate([spline_pred_ages, spline_pred_ages[::-1]]),
            y=np.concatenate([spline_mu_hi, spline_mu_lo[::-1]]),
            fill="toself",
            fillcolor="rgba(21,74,114,0.25)",
            line=dict(color="rgba(255,255,255,0)"),
            name="89% ETI",
        )
    )
    spline_fit_fig.add_trace(
        go.Scatter(
            x=swing_decisions["age"].to_numpy(),
            y=swing_decisions["swing_decision"].to_numpy(),
            mode="markers",
            marker=dict(color=PYMC_BLUE, size=4, opacity=0.3),
            name="observed",
        )
    )
    spline_fit_fig.add_trace(
        go.Scatter(
            x=spline_pred_ages,
            y=spline_mu_mean,
            mode="lines",
            line=dict(color=PYMC_GREEN, width=3),
            name="posterior mean",
        )
    )
    for _knot in knot_list:
        spline_fit_fig.add_vline(x=float(_knot), line=dict(color="grey", width=1, dash="dash"))
    spline_fit_fig.update_layout(
        title="Spline fit with 89% posterior interval",
        xaxis_title="Age (years)",
        yaxis_title="Swing decision grade",
        template="plotly_white",
    )
    spline_fit_fig
    return


@app.cell
def _(spline_basis, spline_idata, spline_model, swing_ages):
    with spline_model:
        pm.set_data({"basis_matrix": spline_basis}, coords={"age_grid": swing_ages})
        spline_obs_ppc = pm.sample_posterior_predictive(
            spline_idata, var_names=["grade_obs"], random_seed=RANDOM_SEED
        )
    spline_idata["posterior_predictive"] = spline_obs_ppc["posterior_predictive"]
    spline_idata.to_netcdf(results_dir / "01_spline.nc")
    return


@app.cell(hide_code=True)
def _(spline_idata, swing_decisions):
    spline_ppc_sigma_mean = float(spline_idata["posterior"]["sigma"].mean())
    spline_ppc_total_sd = float(swing_decisions["swing_decision"].to_numpy().std(ddof=0))
    spline_ppc_r2 = 1 - (spline_ppc_sigma_mean / spline_ppc_total_sd) ** 2
    return spline_ppc_r2, spline_ppc_sigma_mean, spline_ppc_total_sd


@app.cell
def _(spline_idata):
    az.plot_ppc_dist(
        spline_idata,
        var_names=["grade_obs"],
        num_samples=50,
    )
    return


@app.cell(hide_code=True)
def _(spline_ppc_r2, spline_ppc_sigma_mean, spline_ppc_total_sd):
    mo.md(f"""
    The replicated grades reproduce the observed distribution's shape reasonably
    well, but posterior mean `sigma` ({spline_ppc_sigma_mean:.3f}) is nearly as
    large as the total observed standard deviation ({spline_ppc_total_sd:.3f}),
    so the age curve accounts for only about **{spline_ppc_r2:.0%}** of the
    variance in swing decision; most of the spread in this grade has nothing to
    do with age.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### What splines cost, and what comes next

    It fits without us having claimed to know the functional form. That is
    real progress over the piecewise model.

    But look at what we chose along the way. **How many knots?** Seven,
    because seven looked reasonable. **Where?** At quantiles, because that
    puts flexibility where the data are. **What degree?** Cubic, by
    convention. **How strong a prior on the weights?** $\mathcal{N}(0,3)$,
    weak enough not to fight the data. Every one of those is a modelling
    decision that shapes the answer, none is estimated from the data, and
    the fit gives no warning when one is wrong.

    There is also something the model never said. Nothing in it claims that
    a 27-year-old and a 28-year-old should have similar grades. The
    smoothness in that curve is a side effect of overlapping basis
    functions, a property of the *basis we built*, not of a belief we
    stated.

    A Gaussian process inverts that. You state the belief directly, "the
    function is smooth, on roughly this scale, with roughly this
    amplitude", as a **covariance function**, and the data inform its
    parameters. No knots to place, no degree to pick, no basis to design.

    Notebook 2 builds that machinery from the multivariate normal you
    already know.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Where we are, and what's next

    You now have the full Bayesian workflow, exercised twice: specify a
    model, check its prior predictive, sample, check convergence with the
    divergence + `r_hat` + ESS triad, check the posterior predictive
    against the data, and only then interpret what the posterior says. You
    saw it carried through PyMC's core API, `pm.Data` containers,
    `pm.Deterministic`, `pm.sample_prior_predictive`, `pm.sample`, and
    `pm.sample_posterior_predictive` with `pm.set_data`, on two different
    models.

    Both models also shared a limitation. The piecewise-linear curve
    committed to a functional form, a peak time and two slopes; the
    spline traded the form for hand-picked structure, knots, a degree,
    and a prior scale on the weights. Neither warns you when those
    choices are wrong: both fits converged cleanly regardless.

    **Notebook 2** removes that requirement. It builds the Gaussian process
    itself: the multivariate normal machinery it rests on, a covariance
    function, sample functions drawn from a GP prior, and GP regression as
    conditioning. **Notebook 3** then puts that machinery to work, first in
    the analytically convenient *marginal* (Gaussian-likelihood) case with
    `pm.gp.Marginal`, then in the *latent* (non-Gaussian) case with
    `pm.gp.Latent`.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Exercise: fit a spline to the pharmacokinetic curve

    The visible starter cell below prepares the same cubic B-spline basis for
    the single Theophylline subject introduced near the start of this notebook.
    The visible setup cell prepares the basis. The next two disabled cells are
    editable scaffolds: complete each marked step, enable it, and run it before
    continuing.

    1. Complete the model scaffold. The full B-spline basis already spans the
       constant function, so do not add a separate intercept. Enable the cell and
       run `pk_spline_model.debug()`.
    2. Complete the workflow scaffold: prior predictive check, posterior sampling,
       posterior predictive draws, and `az.plot_ppc_dist`.
    3. In one final code cell, plot the posterior mean curve and an 89% interval
       against time in hours.
       Compare its residual pattern with the piecewise fit: which limitation has
       the spline removed, and which modeling choices, knots and independent
       weights, remain?
    """)
    return


@app.cell
def _(conc_z, time_z):
    # Setup provided: one row per pharmacokinetic observation.
    pk_time = time_z
    pk_conc = conc_z
    pk_num_knots = 7
    pk_knot_list = np.quantile(pk_time, np.linspace(0, 1, pk_num_knots))
    pk_basis = np.asarray(
        dmatrix(
            "bs(time, knots=knots, degree=3, include_intercept=True) - 1",
            {"time": pk_time, "knots": pk_knot_list[1:-1]},
        ),
        order="F",
    )
    pk_coords = {
        "basis": np.arange(pk_basis.shape[1]),
        "observation": np.arange(pk_time.size),
    }
    print(f"Ready: {pk_basis.shape[0]} observations x {pk_basis.shape[1]} spline functions")
    return pk_basis, pk_conc, pk_coords


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Coding exercises

    Each coding exercise is a visible Python function with `...` placeholders. Replace every placeholder, then run its cell. A completed exercise shows the plot or table you return; an incomplete one tells you what remains; an error shows its traceback. Try the exercise before expanding its **Solution** accordion.
    """)
    return


@app.cell
def _(pk_basis, pk_conc, pk_coords):
    from exercises import exercise


    @exercise
    def exercise_pk_spline():
        with pm.Model(coords=pk_coords) as pk_spline_model:
            pk_basis_data = pm.Data("basis_matrix", pk_basis, dims=("observation", "basis"))
            pk_conc_data = pm.Data("conc_data", pk_conc, dims="observation")

            # Define Normal spline weights over the "basis" dimension.
            pk_weights = ...
            # Define a positive residual-scale prior.
            pk_sigma = ...
            # Define the spline mean and Normal concentration likelihood.
            pk_mu = ...
            ...

        # Run prior predictive, posterior, and posterior predictive sampling.
        ...


    exercise_pk_spline()
    return


@app.cell(hide_code=True)
def _(pk_basis, pk_conc, pk_coords):
    with pm.Model(coords=pk_coords) as pk_solution_model:
        pk_basis_data = pm.Data("basis_matrix", pk_basis, dims=("observation", "basis"))
        pk_conc_data = pm.Data("conc_data", pk_conc, dims="observation")
        pk_weights = pm.Normal("weights", mu=0, sigma=3, dims="basis")
        pk_sigma = pm.Exponential("sigma", 1)
        pk_mu = pm.Deterministic("mu", pm.math.dot(pk_basis_data, pk_weights), dims="observation")
        pm.Normal("conc_obs", mu=pk_mu, sigma=pk_sigma, observed=pk_conc_data, dims="observation")
        pk_solution_idata = pm.sample(draws=1000, tune=1000, chains=4, random_seed=RANDOM_SEED, progressbar=False)
        pk_solution_idata.update(pm.sample_posterior_predictive(pk_solution_idata, random_seed=RANDOM_SEED, progressbar=False))
    return (pk_solution_idata,)


@app.cell(hide_code=True)
def _(pk_solution_idata):
    def solution_pk_spline():
        return mo.as_html(
            az.plot_ppc_dist(pk_solution_idata, var_names=["conc_obs"], num_samples=50)
        )


    mo.accordion(
        {
            "Solution": mo.vstack([
                mo.md(f"```python\n{inspect.getsource(solution_pk_spline)}\n```"),
                mo.lazy(solution_pk_spline, show_loading_indicator=True),
                mo.md("The hidden reference fit supplies the completed pharmacokinetic spline posterior; the lazy result only renders its PPC."),
            ])
        }
    )
    return


if __name__ == "__main__":
    app.run()
