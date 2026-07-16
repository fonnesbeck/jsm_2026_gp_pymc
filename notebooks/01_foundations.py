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

        Welcome to the first hour of the workshop. Gaussian processes (GPs)
        can feel like a large conceptual jump, so this notebook deliberately
        starts from the ground and climbs one small step at a time. Nothing
        here assumes you have seen a GP before; it *does* assume you are
        comfortable reading a little probability notation and some Python.

        The notebook is in two parts.

        **Part A — the Bayesian workflow in PyMC.** Before we can *fit* a GP
        we need a shared language for building models, checking them, and
        sampling from them. We build up that language on the running dataset
        for the whole workshop — theophylline drug concentrations in blood
        plasma after a single oral dose — starting with the smallest model
        imaginable (a single mean and scale) and then fitting a
        **piecewise-linear** curve with an estimated peak time. That
        piecewise model is deliberately almost-good-enough: watching *how* it
        fails is what motivates everything that follows.

        **Part B — what a Gaussian process actually is.** We build the GP up
        from the multivariate normal distribution you already know, using two
        of its properties — marginalization and conditioning — and you will
        implement a covariance function from scratch, draw sample functions
        from a GP prior, and condition a GP on data by hand. By the end you
        will see that a GP is simply *the flexible function the
        piecewise-linear model could not be*.

        Throughout, we follow one disciplined workflow — specify a model,
        check its prior, sample, check convergence, interpret — so that by
        Part B the machinery feels routine and you can spend your attention
        on the GP ideas themselves.
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
    from plotly.subplots import make_subplots
    from scipy.stats import multivariate_normal, norm

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
        make_subplots,
        multivariate_normal,
        norm,
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
        ## Part A — the Bayesian workflow in PyMC

        ### The modeling problem: an unknown functional form

        Most introductory statistics teaches you to fit *forms you already
        know*: a straight line, a quadratic, an exponential decay, a
        logistic curve. You pick the form, estimate its handful of
        parameters, and you are done. This works beautifully when theory or
        long experience hands you the right form.

        But a great deal of real data has structure with **no obvious
        parametric form**. A drug concentration rises, peaks, and decays,
        but not along any textbook curve. A disease-rate surface varies
        across a map. A sensor drifts over a day in a way no polynomial
        captures cleanly. In each case you can *see* that the function is
        smooth and structured — you simply cannot write it down.

        This is the divide between **parametric** and **nonparametric**
        thinking:

        - A **parametric** model commits to a fixed functional form with a
          fixed, finite number of parameters (a line has two). Its
          flexibility is capped no matter how much data arrives.
        - A **nonparametric** model does not fix the form in advance. Its
          effective number of parameters can grow with the data, so it can
          represent whatever shape the data support — while still being
          regularized enough not to simply interpolate the noise.

        A Gaussian process is the flagship nonparametric model for
        *functions*. Instead of "which curve?" it asks "what do I believe
        about the function — how smooth is it, how far do its values wander
        — and what do the data then tell me?" That is a genuinely different
        way to think, which is why we spend a full hour on foundations
        before touching a GP. The rest of Part A builds the Bayesian
        workflow we will need; then we watch a parametric model
        (piecewise-linear) strain against a shape it cannot hold, which is
        exactly the itch a GP scratches.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### The Bayesian paradigm

        Everything in this workshop is Bayesian, so it is worth stating the
        core idea plainly. Bayesian inference treats unknown quantities as
        **random variables with distributions** and updates those
        distributions in light of data. Three objects do all the work:

        - The **prior** $p(\theta)$ — what we believe about the parameters
          $\theta$ *before* seeing the data. It encodes scale, sign,
          plausibility: "a concentration is positive and probably a
          single-digit number of mg/L", not "a concentration could be
          $10^{12}$".
        - The **likelihood** $p(y \mid \theta)$ — how probable the observed
          data $y$ are for each candidate value of $\theta$. This is the
          model's description of the data-generating process.
        - The **posterior** $p(\theta \mid y)$ — the updated belief about
          $\theta$ *after* folding in the data. It is what we report.

        These are tied together by **Bayes' rule**:

        $$p(\theta \mid y) = \frac{p(y \mid \theta)\, p(\theta)}{p(y)}
        \;\propto\; p(y \mid \theta)\, p(\theta).$$

        The denominator $p(y) = \int p(y \mid \theta)\,p(\theta)\,d\theta$ is
        just the constant that makes the posterior integrate to one; for
        inference we usually work with the proportionality on the right. In
        words: **posterior $\propto$ likelihood $\times$ prior**. The
        posterior is a *compromise* between what you believed and what the
        data say — data-rich regions pull it toward the likelihood, and where
        data are scarce the prior still speaks.

        Only for a few textbook "conjugate" models can you write the
        posterior in closed form. For everything else — including every GP in
        this workshop — we *sample* from the posterior instead, drawing many
        representative parameter values with Markov chain Monte Carlo (MCMC).
        PyMC does this for us. The figure below shows the one conjugate case
        we can compute by hand, purely to build intuition for the
        prior → posterior update.
        """
    )
    return


@app.cell
def _(PYMC_BLUE, PYMC_GREEN, PYMC_LIGHT_BLUE, go, norm, np):
    # A one-dimensional conjugate illustration: infer the unknown mean theta
    # of a Normal with KNOWN sd, so prior x likelihood -> posterior is exact.
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
    return post_mean, post_sd


@app.cell(hide_code=True)
def _(mo, post_mean, post_sd):
    mo.md(
        f"""
        Read the figure left to right. The **prior** (light blue) is centred
        at 0 and fairly broad. The **likelihood** (green, dashed) is centred
        near the data mean of about 1.9 and is tighter because five
        observations already pin the mean down reasonably well. The
        **posterior** (dark blue) lands *between* them — mean
        ≈ {post_mean:.2f}, sd ≈ {post_sd:.2f} — pulled most of the way toward
        the data but still nudged toward the prior, and narrower than either
        input because it combines both sources of information.

        Two lessons carry through the whole workshop. First, **the posterior
        is a compromise**, and where data are sparse the prior matters — a
        fact that becomes vivid for GPs, whose priors are over entire
        functions. Second, **priors have consequences you should check
        before you fit**, which is exactly what the prior predictive check
        below is for.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### The smallest possible model: a mean and a scale

        Let's turn the paradigm into PyMC code on the simplest question we
        can ask of real data. Take a single theophylline subject and ignore
        *time* entirely: model their concentration measurements as noisy
        draws around one unknown level. Concretely, for each measurement
        $y_i$,

        $$y_i \sim \mathcal{N}(\mu, \sigma), \qquad
        \mu \sim \mathcal{N}(5, 5), \qquad
        \sigma \sim \text{HalfNormal}(5).$$

        Here $\mu$ is the subject's typical concentration (mg/L) and $\sigma$
        is how much individual measurements scatter around it. This is a
        *bad* model — the true concentration rises then falls, so no single
        mean describes it — but it is the perfect vehicle for meeting every
        moving part of the PyMC workflow: the `pm.Model` context manager,
        priors, a likelihood, the prior predictive check, `pm.sample`, the
        returned inference object, `az.summary`, and convergence plots. We
        will keep this warm-up in raw mg/L so the numbers stay interpretable.
        """
    )
    return


@app.cell
def _(data_dir, pl):
    theoph = pl.read_csv(data_dir / "theophylline.csv")
    theoph.head()
    return (theoph,)


@app.cell
def _(pl, theoph, z):
    # One subject, sorted by time. We reuse this prep for both models in Part A.
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


@app.cell
def _(conc_vals, pm):
    with pm.Model() as warmup_model:
        mu = pm.Normal("mu", mu=5, sigma=5)
        sigma = pm.HalfNormal("sigma", sigma=5)
        pm.Normal("conc_obs", mu=mu, sigma=sigma, observed=conc_vals)

    warmup_model
    return (warmup_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The `with pm.Model() as warmup_model:` block is a **context manager**:
        every random variable created inside it is automatically registered
        with that model. `pm.Normal("mu", ...)` and
        `pm.HalfNormal("sigma", ...)` are the **priors**; the final
        `pm.Normal("conc_obs", ..., observed=conc_vals)` is the
        **likelihood**, marked as observed because we condition on those
        values. Rendering the model (above) shows the graph PyMC built.

        ### Prior predictive check

        Before sampling the posterior we simulate data **from the prior
        alone** with `pm.sample_prior_predictive`. This draws $\mu$ and
        $\sigma$ from their priors and then generates fake concentration
        datasets from them, *without ever looking at the observed values*.
        It answers a blunt question: do our priors, pushed through the
        likelihood, produce data in a remotely plausible range? A prior that
        implies negative concentrations, or concentrations in the thousands,
        is telling you something before you waste a single sampling second.
        """
    )
    return


@app.cell
def _(RANDOM_SEED, pm, warmup_model):
    with warmup_model:
        warmup_prior_pred = pm.sample_prior_predictive(
            draws=500, random_seed=RANDOM_SEED
        )
    return (warmup_prior_pred,)


@app.cell
def _(PYMC_LIGHT_BLUE, conc_vals, go, np, warmup_prior_pred):
    warmup_prior_draws = warmup_prior_pred["prior_predictive"][
        "conc_obs"
    ].values.reshape(-1, len(conc_vals))

    warmup_prior_fig = go.Figure()
    # Histogram of ALL simulated concentrations implied by the prior.
    warmup_prior_fig.add_trace(
        go.Histogram(
            x=warmup_prior_draws.ravel(),
            histnorm="probability density",
            marker=dict(color=PYMC_LIGHT_BLUE),
            opacity=0.7,
            name="prior predictive conc",
        )
    )
    for _c in conc_vals:
        warmup_prior_fig.add_vline(x=float(_c), line=dict(color="black", width=1))
    warmup_prior_fig.update_layout(
        title="Prior predictive concentrations (bars) vs. observed values (black lines)",
        xaxis_title="Concentration (mg/L)",
        yaxis_title="density",
        template="plotly_white",
    )
    warmup_prior_fig
    return (warmup_prior_draws,)


@app.cell(hide_code=True)
def _(conc_vals, mo, warmup_prior_draws):
    mo.md(
        f"""
        **Plausibility check:** the prior predictive concentrations span
        [{warmup_prior_draws.min():.1f}, {warmup_prior_draws.max():.1f}]
        mg/L, comfortably covering the observed range
        [{conc_vals.min():.2f}, {conc_vals.max():.2f}] mg/L without being
        wildly wider. Some prior draws stray slightly negative — a known
        artifact of a Normal likelihood on a positive quantity — but the mass
        sits in a sensible pharmacological range. Broad but not absurd:
        reasonable to proceed to sampling.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Sampling the posterior

        Now we draw from the posterior with `pm.sample`. As of PyMC 6 the
        sampler defaults to the fast Rust-based **nutpie** implementation of
        the No-U-Turn Sampler automatically — you do **not** pass a
        `nuts_sampler=` argument. We ask for `draws=1000` posterior samples
        per chain after `tune=1000` warm-up steps, across `chains=2`
        independent chains (running multiple chains is what lets us diagnose
        convergence). Passing `random_seed=RANDOM_SEED` makes the run
        reproducible.

        `pm.sample` returns an **inference object** — an
        [ArviZ `DataTree`](https://python.arviz.org) — that bundles the
        posterior draws, sampler statistics, prior, and observed data into
        one nested container. We access its groups by bracket, e.g.
        `idata["posterior"]` and `idata["sample_stats"]`.
        """
    )
    return


@app.cell
def _(RANDOM_SEED, pm, warmup_model):
    with warmup_model:
        warmup_idata = pm.sample(
            draws=1000, tune=1000, chains=2, random_seed=RANDOM_SEED
        )
    return (warmup_idata,)


@app.cell
def _(az, warmup_idata):
    warmup_n_div = warmup_idata["sample_stats"]["diverging"].sum().item()
    warmup_summary = az.summary(warmup_idata["posterior"])
    print(f"Divergences: {warmup_n_div}")
    warmup_summary
    return (warmup_n_div, warmup_summary)


@app.cell(hide_code=True)
def _(mo, warmup_n_div, warmup_summary):
    mo.md(
        f"""
        **Reading the diagnostics.** The table above (from `az.summary`) is
        the first thing to check after any fit. Two columns matter most:

        - **`r_hat`** compares variance *within* each chain to variance
          *between* chains. If the chains have converged to the same
          distribution it sits at **1.00**; values above ~1.01 warn that the
          chains disagree. Here the maximum `r_hat` is
          {float(warmup_summary["r_hat"].astype(float).max()):.3f}.
        - **`ess_bulk` / `ess_tail`** are *effective sample sizes* — how many
          truly independent draws your (autocorrelated) chains are worth, in
          the bulk and the tails of the distribution respectively. We want
          both comfortably above ~400. Here the minimum `ess_bulk` is
          {float(warmup_summary["ess_bulk"].min()):.0f} and the minimum
          `ess_tail` is {float(warmup_summary["ess_tail"].min()):.0f}.

        We also print the number of **divergences** — transitions where the
        sampler's numerical integrator broke down, each a red flag for
        biased results. Here there were **{warmup_n_div}**. With near-perfect
        `r_hat`, healthy ESS, and no divergences, this fit has converged and
        is safe to interpret. We will run exactly this three-part check
        (divergences + `r_hat` + ESS) after every model in the workshop.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Looking at the chains: trace and rank plots

        Summary numbers are necessary but not sufficient; it helps to *see*
        the chains. Two standard views, both built here directly in plotly:

        - A **trace plot** shows each chain's draws against iteration. Healthy
          chains look like a "fuzzy caterpillar" — stationary, overlapping,
          with no trends or stuck stretches.
        - A **rank plot** pools all draws, ranks them, and histograms those
          ranks *per chain*. If the chains are exploring the same
          distribution, every chain should be equally likely to hold any
          rank, so each histogram should look roughly **uniform**. Systematic
          departures (one chain owning the high ranks, say) reveal mixing
          problems that a trace plot can hide.
        """
    )
    return


@app.cell
def _(PYMC_BLUE, PYMC_GREEN, go, make_subplots, warmup_idata):
    trace_fig = make_subplots(rows=1, cols=2, subplot_titles=("μ trace", "σ trace"))
    _chain_colors = [PYMC_BLUE, PYMC_GREEN]
    for _p, _col in (("mu", 1), ("sigma", 2)):
        _da = warmup_idata["posterior"][_p]
        for _ci in range(_da.sizes["chain"]):
            trace_fig.add_trace(
                go.Scatter(
                    y=_da.isel(chain=_ci).values,
                    mode="lines",
                    line=dict(color=_chain_colors[_ci], width=1),
                    opacity=0.7,
                    name=f"chain {_ci}",
                    showlegend=(_col == 1),
                ),
                row=1,
                col=_col,
            )
    trace_fig.update_layout(
        title="Trace plot — each chain's draws over iteration",
        template="plotly_white",
    )
    trace_fig.update_xaxes(title_text="iteration")
    trace_fig
    return


@app.cell
def _(PYMC_BLUE, PYMC_GREEN, go, np, warmup_idata):
    # Rank plot for mu, built by hand: rank the pooled draws, then histogram
    # each chain's ranks. Uniform-looking histograms indicate good mixing.
    mu_draws = warmup_idata["posterior"]["mu"].values  # (chain, draw)
    n_chain, n_draw = mu_draws.shape
    flat_ranks = mu_draws.ravel().argsort().argsort().reshape(n_chain, n_draw)

    rank_fig = go.Figure()
    _chain_colors2 = [PYMC_BLUE, PYMC_GREEN]
    _edges = np.linspace(0, n_chain * n_draw, 21)
    for _ci in range(n_chain):
        rank_fig.add_trace(
            go.Histogram(
                x=flat_ranks[_ci],
                xbins=dict(start=_edges[0], end=_edges[-1], size=_edges[1] - _edges[0]),
                marker=dict(color=_chain_colors2[_ci]),
                opacity=0.6,
                name=f"chain {_ci}",
            )
        )
    rank_fig.update_layout(
        barmode="overlay",
        title="Rank plot for μ — roughly uniform per chain means good mixing",
        xaxis_title="rank of pooled draw",
        yaxis_title="count",
        template="plotly_white",
    )
    rank_fig
    return


@app.cell(hide_code=True)
def _(conc_vals, mo, warmup_summary):
    mo.md(
        f"""
        Both chains overlap and look stationary in the trace plot, and the
        two rank histograms sit on top of each other near uniform — the
        visual signature of a well-mixed, converged sampler that matches the
        clean summary numbers.

        **What did we learn?** The posterior mean of $\\mu$ is about
        {float(warmup_summary.loc["mu", "mean"]):.2f} mg/L — the subject's
        average concentration — with $\\sigma$ near
        {float(warmup_summary.loc["sigma", "mean"]):.2f} mg/L of scatter. But
        that scatter is enormous relative to the signal, because it is
        soaking up *real structure we ignored*: the concentration is not
        random noise around a constant, it **rises and falls with time**. The
        observed values run from {conc_vals.min():.2f} to
        {conc_vals.max():.2f} mg/L precisely because of that time course. A
        single mean throws all of it away. The obvious next step is to let
        the mean depend on time.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Background: the Theophylline dataset

        Before we add time to the model, meet the dataset properly.
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
        asymmetric, single-humped curve is a poor fit for any straight line —
        yet a completely natural fit for a Gaussian process — which makes it
        the ideal running example for this workshop.

        The columns:

        - `subject`: subject id (1–12)
        - `time`: hours since dose
        - `conc`: serum theophylline concentration (mg/L)
        - `dose`: administered dose (mg/kg)
        - `weight`: subject body weight (kg)
        """
    )
    return


@app.cell
def _(go, pl, theoph):
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
def _(mo, subject_id):
    mo.md(
        f"""
        Every subject traces the same qualitative arc — a fast early rise to
        a peak within the first couple of hours, then a slow decay over the
        rest of the day — but the *height* of the peak, its *timing*, and the
        *rate* of decay differ from person to person (driven partly by dose
        and body weight). Later notebooks exploit that shared-shape-with-
        individual-variation structure directly with hierarchical GPs. For
        the rest of Part A we focus on a **single subject** (subject
        {subject_id}) and ask: can a simple parametric curve capture even one
        of these traces?
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### A piecewise-linear baseline: two slopes meeting at a peak

        The warm-up model had no notion of time. The simplest model that
        respects the rise-then-decay shape is a **piecewise-linear** (broken-
        stick) curve: a straight line going *up* during absorption, a kink at
        the peak, and a straight line going *down* during elimination. We do
        not know where the peak is, so we make its location a parameter,
        $\tau$, and let the data estimate it.

        A clean way to write a one-knot broken stick uses a **hinge basis**.
        On the (standardized) time axis $t$,

        $$\mu(t) = \text{level} + \text{up}\cdot t
        + \text{down}\cdot \max(0,\, t - \tau).$$

        Read it piece by piece. Before the knot ($t < \tau$) the
        $\max(0, t-\tau)$ term is zero, so the slope is just `up`. After the
        knot ($t > \tau$) the hinge switches on and the slope becomes
        `up + down`. To bend a rise into a decay we expect `up` positive and
        `down` negative and larger in magnitude, so the post-peak slope turns
        downward. The priors:

        $$\text{level}\sim\mathcal N(0,1),\;
        \text{up}\sim\text{HalfNormal}(2),\;
        \text{down}\sim\mathcal N(0,2),\;
        \tau\sim\text{Uniform}(t_{\min}, t_{\max}),\;
        \sigma\sim\text{HalfNormal}(1).$$

        `up` is HalfNormal to encode "absorption makes it rise"; `down` is an
        unconstrained Normal so the data can decide how sharply it falls;
        $\tau$ is Uniform over the observed time window. We work on
        standardized time and concentration (numerically friendlier), and
        convert $\tau$ back to hours when we interpret it.
        """
    )
    return


@app.cell
def _(conc_z, np, pm, time_z):
    t_lo, t_hi = float(time_z.min()), float(time_z.max())
    with pm.Model() as pw_model:
        level = pm.Normal("level", mu=0, sigma=1)
        up = pm.HalfNormal("up", sigma=2)
        down = pm.Normal("down", mu=0, sigma=2)
        tau = pm.Uniform("tau", lower=t_lo, upper=t_hi)
        sigma_pw = pm.HalfNormal("sigma_pw", sigma=1)

        # Hinge / broken-stick mean function: slope is `up` before tau,
        # `up + down` after it. pm.math.maximum is the differentiable hinge.
        mu_pw = level + up * time_z + down * pm.math.maximum(0.0, time_z - tau)
        pm.Deterministic("mu_pw", mu_pw)
        pm.Normal("conc_obs", mu=mu_pw, sigma=sigma_pw, observed=conc_z)

    pw_model
    return pw_model, t_hi, t_lo


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Prior predictive check

        As before, we look at what the model believes *before* fitting —
        this time the priors imply whole *curves*, not just a spread of
        points. We draw parameter sets from the priors, evaluate $\mu(t)$
        across the time grid for each, and plot the implied
        standardized-concentration trajectories against the standardized
        data.
        """
    )
    return


@app.cell
def _(RANDOM_SEED, pm, pw_model):
    with pw_model:
        pw_prior_pred = pm.sample_prior_predictive(draws=400, random_seed=RANDOM_SEED)
    return (pw_prior_pred,)


@app.cell
def _(PYMC_LIGHT_BLUE, conc_z, go, np, pw_prior_pred, time_z):
    pw_prior_curves = pw_prior_pred["prior"]["mu_pw"].values.reshape(-1, len(time_z))
    _order = np.argsort(time_z)

    pw_prior_fig = go.Figure()
    _rng_plot = np.random.default_rng(0)
    for _i in _rng_plot.choice(pw_prior_curves.shape[0], size=60, replace=False):
        pw_prior_fig.add_trace(
            go.Scatter(
                x=time_z[_order],
                y=pw_prior_curves[_i][_order],
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
    return (pw_prior_curves,)


@app.cell(hide_code=True)
def _(conc_z, mo, pw_prior_curves):
    mo.md(
        f"""
        **Plausibility check:** the prior piecewise curves span roughly
        [{pw_prior_curves.min():.1f}, {pw_prior_curves.max():.1f}] on the
        standardized scale, bracketing the observed standardized range
        [{conc_z.min():.2f}, {conc_z.max():.2f}]. You can see the model's
        *inductive bias* directly in the plot: every draw is two straight
        segments with a single kink. Some rise then fall (the shape we
        hope for), some do the reverse, some barely bend — the prior is
        agnostic about the exact geometry but has already committed, hard, to
        "two lines and one corner". That commitment is the crux of what
        follows. Reasonable to proceed.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Sampling

        Five parameters (`level`, `up`, `down`, `tau`, `sigma_pw`) over 11
        points. We use the standard `draws=1000, tune=1000, chains=2` and
        nudge `target_accept` up to 0.9. The hinge introduces a *kink* in the
        likelihood surface as a function of $\tau$ (the gradient of
        $\max(0, t-\tau)$ jumps at each data point), which makes $\tau$
        intrinsically harder to sample than a smooth parameter — a first
        hint of the identifiability trouble we are about to diagnose.
        """
    )
    return


@app.cell
def _(RANDOM_SEED, pm, pw_model):
    with pw_model:
        pw_idata = pm.sample(
            draws=1000,
            tune=1000,
            chains=2,
            target_accept=0.9,
            random_seed=RANDOM_SEED,
        )
    return (pw_idata,)


@app.cell
def _(az, pw_idata):
    pw_n_div = pw_idata["sample_stats"]["diverging"].sum().item()
    pw_summary = az.summary(
        pw_idata["posterior"], var_names=["level", "up", "down", "tau", "sigma_pw"]
    )
    print(f"Divergences: {pw_n_div}")
    pw_summary
    return pw_n_div, pw_summary


@app.cell(hide_code=True)
def _(mo, pw_n_div, pw_summary):
    mo.md(
        f"""
        **Diagnostics:** {pw_n_div} divergence(s). Across the five
        parameters the maximum `r_hat` is
        {float(pw_summary["r_hat"].astype(float).max()):.3f} and the minimum
        `ess_bulk` is {float(pw_summary["ess_bulk"].min()):.0f}. The
        chains have converged well enough to interpret — but notice already
        in the table how *wide* the posterior for `tau` is relative to the
        others (compare its `sd` and its `hdi_3%`–`hdi_97%` interval to
        `level` or `up`). A converged fit can still be an inadequate one, as
        the next plots make plain.
        """
    )
    return


@app.cell
def _(
    PYMC_BLUE,
    PYMC_GREEN,
    az,
    conc_mean,
    conc_std,
    conc_vals,
    go,
    np,
    pw_idata,
    time_mean,
    time_std,
    time_vals,
):
    # Reconstruct mu(t) on a fine grid for every posterior draw, in original units.
    _post = pw_idata["posterior"]
    _level = _post["level"].values.ravel()
    _up = _post["up"].values.ravel()
    _down = _post["down"].values.ravel()
    _tau = _post["tau"].values.ravel()

    time_grid = np.linspace(time_vals.min(), time_vals.max(), 200)
    time_grid_z = (time_grid - time_mean) / time_std

    _hinge = np.maximum(0.0, time_grid_z[None, :] - _tau[:, None])
    _mu_z = (
        _level[:, None] + _up[:, None] * time_grid_z[None, :] + _down[:, None] * _hinge
    )
    mu_orig = _mu_z * conc_std + conc_mean

    pw_fit_mean = mu_orig.mean(axis=0)
    pw_fit_hdi = az.hdi(mu_orig, prob=0.89, axis=0)
    pw_fit_lo, pw_fit_hi = pw_fit_hdi[:, 0], pw_fit_hdi[:, 1]

    pw_fit_fig = go.Figure()
    pw_fit_fig.add_trace(
        go.Scatter(
            x=np.concatenate([time_grid, time_grid[::-1]]),
            y=np.concatenate([pw_fit_hi, pw_fit_lo[::-1]]),
            fill="toself",
            fillcolor="rgba(21,74,114,0.25)",
            line=dict(color="rgba(255,255,255,0)"),
            name="89% HDI",
        )
    )
    pw_fit_fig.add_trace(
        go.Scatter(
            x=time_grid,
            y=pw_fit_mean,
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
        title="Piecewise-linear fit — better than a mean, but note the sharp corner",
        xaxis_title="Time since dose (hours)",
        yaxis_title="Concentration (mg/L)",
        template="plotly_white",
    )
    pw_fit_fig
    return (time_grid,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This is a real improvement on a flat mean: the fit rises to a peak
        and then falls, roughly tracking the data. But look closely and three
        problems stand out — and they are not fixable by tuning this model,
        they are *baked into its form*. We diagnose them next.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Diagnosis: why the piecewise model is inadequate

        **1. The kink is unphysical.** A real absorption/elimination curve is
        *smooth* — it eases through its peak, it does not turn a hard corner.
        The broken stick puts an infinitely sharp vertex at $\tau$, which no
        drug's pharmacokinetics actually do. The model can only ever
        approximate a smooth hump with two straight lines and an angle.

        **2. The peak time $\tau$ is poorly identified.** Because only a few
        points sit near the peak, and because the hinge makes the likelihood
        nearly flat over a range of knot positions, the data cannot pin
        $\tau$ down. Its posterior is *wide* — a broad band of "the peak is
        somewhere in here" rather than a confident estimate. The plot below
        shows just how wide.

        **3. Straight segments miss the curvature.** Between the knots the
        model is forced to be exactly linear, so it *undershoots* the rounded
        top of the rise and cannot follow the gentle concave-up flattening of
        the decay tail. The residual structure you can see around the fitted
        line is the curvature the model has no vocabulary for.

        The common thread: **we had to choose a functional form, and the form
        we chose is wrong in ways the data cannot repair.** We could keep
        patching — add more knots, swap in an exponential decay, bolt on an
        absorption compartment — but each patch is another hand-specified
        commitment. What we actually want is a model that says only "the
        function is smooth" and lets the data supply the shape. That model is
        a Gaussian process, and Part B builds it.
        """
    )
    return


@app.cell
def _(PYMC_GREEN, go, np, pw_idata, time_mean, time_std):
    # Convert the tau posterior back to hours to show how wide it is.
    tau_hours = pw_idata["posterior"]["tau"].values.ravel() * time_std + time_mean
    tau_lo, tau_hi = np.quantile(tau_hours, [0.055, 0.945])

    tau_fig = go.Figure()
    tau_fig.add_trace(
        go.Histogram(
            x=tau_hours,
            histnorm="probability density",
            marker=dict(color=PYMC_GREEN),
            opacity=0.75,
            name="τ posterior",
        )
    )
    tau_fig.add_vline(x=float(tau_lo), line=dict(color="black", dash="dash"))
    tau_fig.add_vline(x=float(tau_hi), line=dict(color="black", dash="dash"))
    tau_fig.update_layout(
        title="Posterior of the peak time τ (hours) — dashed lines mark the 89% interval",
        xaxis_title="Estimated peak time τ (hours)",
        yaxis_title="density",
        template="plotly_white",
    )
    tau_fig
    return tau_hi, tau_lo


@app.cell(hide_code=True)
def _(mo, tau_hi, tau_lo):
    mo.md(
        f"""
        The 89% posterior interval for the peak time runs from about
        {tau_lo:.1f} to {tau_hi:.1f} hours — a span of roughly
        {tau_hi - tau_lo:.1f} hours for a curve that is essentially over
        within a day. That width is the model *telling us it does not know
        where its own corner belongs*, which is what happens when you force a
        sharp feature onto a smooth process. Contrast this with what we will
        get from the GP: no knot to locate, no corner to defend, just a
        smooth posterior over the whole curve with honest uncertainty
        everywhere.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Exercises — Part A

        Work these in your head or in a scratch cell before expanding the
        solutions. They reinforce the workflow habits — reading priors,
        judging identifiability, interpreting diagnostics — that the rest of
        the workshop assumes.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Exercise 1 — change a prior and predict the prior predictive": mo.md(
                r"""
                Suppose you tightened the rise-slope prior from
                `up ~ HalfNormal(2)` to `up ~ HalfNormal(0.2)`. *Before*
                rerunning, predict what the prior predictive curve plot would
                look like. Then reason about whether the posterior fit could
                still reach the observed peak.

                **Solution.** A HalfNormal(0.2) keeps `up` within roughly
                ±0.4 on the standardized scale, so the *rising* segments in
                the prior predictive would be far shallower — the fan of
                curves would climb only gently before the knot. Because the
                data show a steep early rise (0.74 → 10.5 mg/L in about an
                hour), such a prior fights the likelihood: the posterior
                could still pull `up` upward, but a genuinely informative-yet-
                wrong prior like this would bias the fit toward under-shooting
                the peak and inflate `sigma_pw` to absorb the misfit. The
                lesson: prior predictive plots let you catch an over-tight
                prior before it quietly distorts the posterior.
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Exercise 2 — widen vs. narrow the τ prior and think about identifiability": mo.md(
                r"""
                We used `tau ~ Uniform(t_min, t_max)` over the whole observed
                window. What would happen to the posterior width of $\tau$ if
                you instead used a *tight* prior like
                `tau ~ Normal(peak_guess, 0.1)` centred near the visible
                peak? Does that make the model better?

                **Solution.** A tight prior would of course produce a narrow
                *posterior* for $\tau$ — but that narrowness would come from
                the **prior**, not from information in the data. The
                likelihood is nearly flat across a range of knot positions
                (that is what made the wide posterior in the first place), so
                squeezing $\tau$ with a strong prior just hides the
                identifiability problem rather than solving it, and makes the
                result sensitive to where *you* guessed the peak was.
                Poor identifiability is a property of the model-plus-data, not
                a bug to be papered over with a confident prior. The honest
                fix is a model that does not need a single knot location at
                all.
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Exercise 3 — interpret a diagnostic readout": mo.md(
                r"""
                Imagine a rerun of the piecewise model reported: `r_hat` for
                `tau` = 1.06, `ess_bulk` for `tau` = 55, and 40 divergences,
                while `level`, `up`, and `sigma_pw` all looked fine. What
                would you conclude, and what would you try first?

                **Solution.** The trouble is localized to `tau`: an `r_hat`
                of 1.06 means the two chains have *not* agreed on its
                distribution, an `ess_bulk` of 55 means you effectively have
                only a few dozen independent draws of it, and 40 divergences
                point to the sampler struggling with the kinked geometry the
                hinge creates around the knot. You would *not* trust any
                statement about the peak time from this run. First moves:
                raise `target_accept` (e.g. to 0.95) to reduce divergences,
                run longer chains, and — most tellingly — recognize that a
                parameter this hard to sample is often a parameter the data
                barely identify. That recurring difficulty is another nudge
                toward the smoother, better-behaved GP formulation.
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Part B — From multivariate normals to Gaussian processes

        Part A left us wanting a model that assumes only *smoothness* and
        lets the data supply the shape. Remarkably, the tool for that is
        built entirely out of the **multivariate normal (MVN)** distribution
        you already know. This part assembles it piece by piece: two key
        properties of the MVN, a covariance function you code yourself,
        sample functions drawn from a GP prior, and finally conditioning a GP
        on data — the operation that *is* GP regression.

        ### The multivariate normal, and two properties that matter

        Recall that a $d$-dimensional MVN is fully specified by a mean vector
        $\boldsymbol\mu$ and a covariance matrix $\Sigma$:
        $\mathbf{x} \sim \mathcal N(\boldsymbol\mu, \Sigma)$. The diagonal of
        $\Sigma$ holds the variances; the off-diagonal entries encode how the
        components co-vary. Two properties of the MVN are the entire
        mathematical foundation of Gaussian processes.

        Partition the vector into two blocks $\mathbf x = (\mathbf a,
        \mathbf b)$ with

        $$\begin{bmatrix}\mathbf a\\ \mathbf b\end{bmatrix}\sim
        \mathcal N\!\left(
        \begin{bmatrix}\boldsymbol\mu_a\\ \boldsymbol\mu_b\end{bmatrix},
        \begin{bmatrix}\Sigma_{aa} & \Sigma_{ab}\\
        \Sigma_{ba} & \Sigma_{bb}\end{bmatrix}\right).$$

        **Marginalization.** The distribution of a sub-block on its own,
        ignoring the rest, is *again normal* — you simply read off the
        relevant sub-vector and sub-matrix:

        $$p(\mathbf a) = \mathcal N(\boldsymbol\mu_a,\ \Sigma_{aa}).$$

        **Conditioning.** The distribution of one block *given* the other is
        *also normal*, with a mean that shifts toward the observed values and
        a variance that shrinks:

        $$p(\mathbf a \mid \mathbf b) = \mathcal N\!\big(
        \boldsymbol\mu_a + \Sigma_{ab}\Sigma_{bb}^{-1}(\mathbf b -
        \boldsymbol\mu_b),\ \ \Sigma_{aa} - \Sigma_{ab}\Sigma_{bb}^{-1}
        \Sigma_{ba}\big).$$

        These are the two operations a GP lives on. **Marginalization** is
        what lets us ignore the "infinitely many" function values we did not
        ask about and work with only the finite set at hand.
        **Conditioning** is exactly how a GP turns a prior over functions
        into a posterior once data arrive. Let's make both concrete with a
        worked bivariate example.
        """
    )
    return


@app.cell
def _(np):
    # A concrete correlated bivariate normal to work marginalization/conditioning by hand.
    biv_mean = np.array([1.0, 2.0])
    biv_cov = np.array([[1.0, 0.8], [0.8, 1.5]])

    # Condition x1 on an observed x2 = 3.5 using the conditioning formula.
    x2_obs = 3.5
    cond_mean = biv_mean[0] + biv_cov[0, 1] / biv_cov[1, 1] * (x2_obs - biv_mean[1])
    cond_var = biv_cov[0, 0] - biv_cov[0, 1] ** 2 / biv_cov[1, 1]
    cond_sd = np.sqrt(cond_var)

    print(f"Marginal of x1:       mean = {biv_mean[0]:.3f}, var = {biv_cov[0, 0]:.3f}")
    print(f"Conditional x1|x2=3.5: mean = {cond_mean:.3f}, var = {cond_var:.3f}")
    return biv_cov, biv_mean, cond_mean, cond_sd, x2_obs


@app.cell(hide_code=True)
def _(biv_cov, biv_mean, cond_mean, cond_sd, mo, x2_obs):
    mo.md(
        f"""
        **Worked numbers.** Our joint has $\\boldsymbol\\mu = (1, 2)$ and

        $$\\Sigma = \\begin{{bmatrix}} 1.0 & 0.8\\\\ 0.8 & 1.5
        \\end{{bmatrix}}.$$

        *Marginalizing* to $x_1$ just reads off the top-left block:
        $x_1 \\sim \\mathcal N(1.0,\\ 1.0)$ — mean {biv_mean[0]:.1f}, variance
        {biv_cov[0, 0]:.1f}.

        *Conditioning* on observing $x_2 = {x2_obs}$ uses the formula above.
        The observed $x_2$ is {x2_obs - biv_mean[1]:.1f} above its own mean,
        and the positive covariance drags $x_1$ upward with it:

        $$\\mathbb E[x_1 \\mid x_2={x2_obs}] = 1.0 +
        \\tfrac{{0.8}}{{1.5}}({x2_obs}-2.0) = {cond_mean:.3f},$$

        while the conditional variance *shrinks* from 1.0 to
        {cond_sd**2:.3f} (sd {cond_sd:.3f}) — knowing $x_2$ has told us
        something about $x_1$, so we are less uncertain than the marginal.
        **That shrink-toward-the-data-with-reduced-uncertainty is the whole
        idea of GP regression**, applied to function values instead of two
        scalars.
        """
    )
    return


@app.cell
def _(biv_cov, biv_mean, cond_mean, cond_sd, go, multivariate_normal, norm, np, x2_obs):
    _grid = np.linspace(-3, 6, 160)
    _X1, _X2 = np.meshgrid(_grid, _grid)
    _pos = np.dstack((_X1, _X2))
    _dens = multivariate_normal(biv_mean, biv_cov).pdf(_pos)

    cond_fig = go.Figure()
    cond_fig.add_trace(
        go.Contour(
            x=_grid,
            y=_grid,
            z=_dens,
            colorscale="Blues",
            showscale=False,
            contours=dict(showlabels=False),
        )
    )
    # The horizontal slice x2 = x2_obs that defines the conditional p(x1 | x2).
    cond_fig.add_hline(y=x2_obs, line=dict(color="#40611F", width=2, dash="dash"))
    # Overlay the resulting 1D conditional density of x1 along that slice.
    _cond_curve = norm.pdf(_grid, cond_mean, cond_sd)
    cond_fig.add_trace(
        go.Scatter(
            x=_grid,
            y=x2_obs + _cond_curve,  # lift the curve up to the slice for display
            mode="lines",
            line=dict(color="#81C240", width=3),
            name="p(x1 | x2=3.5)",
        )
    )
    cond_fig.update_layout(
        title="Joint p(x1, x2) with the conditional slice at x2 = 3.5",
        xaxis_title="x1",
        yaxis_title="x2",
        template="plotly_white",
    )
    cond_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The contours are the joint density; the dashed line is the slice
        $x_2 = 3.5$. The green curve is the conditional $p(x_1 \mid x_2=3.5)$
        — a renormalized 1-D Gaussian along that slice, centred to the right
        of $x_1$'s marginal mean and narrower than it. Slide the slice up or
        down (mentally) and the conditional peak tracks with it: that
        tracking is the covariance doing its job.

        ### A Gaussian process is a distribution over functions

        Now take the leap. A **Gaussian process** generalizes the MVN to
        *infinitely many* variables. Formally, a GP is a collection of random
        variables, **any finite subset of which is jointly multivariate
        normal**. If we think of a function $f$ as an infinitely long vector
        — one entry $f(x)$ for every input $x$ — then a GP is a probability
        distribution over such functions:

        $$f(x) \sim \mathcal{GP}\big(m(x),\ k(x, x')\big).$$

        Just as an MVN needs a mean *vector* and covariance *matrix*, a GP
        needs a **mean function** $m(x)$ and a **covariance function**
        $k(x, x')$ (the *kernel*), which returns the covariance between the
        function's values at any two inputs. The **marginalization** property
        is what makes this usable: to work with a GP at a finite set of input
        points, we just evaluate $m$ and $k$ on that set to get an ordinary
        MVN and proceed — the infinitely many unqueried points marginalize
        away for free. And the **conditioning** property (next) is what turns
        the GP prior into a posterior given data. Everything reduces to the
        two MVN operations you just did by hand.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Exercise: implement the ExpQuad kernel from scratch

        The mean function is very often just zero; almost all of a GP's
        character comes from its **covariance function**. The most common one
        is the **exponential quadratic** (a.k.a. squared-exponential, RBF,
        Gaussian) kernel:

        $$k(x, x') = \eta^2 \exp\!\left(-\frac{(x - x')^2}{2\ell^2}\right).$$

        Two hyperparameters: $\eta$ (**amplitude**) scales how far function
        values swing, and $\ell$ (**lengthscale**) sets how quickly the
        correlation between two points decays as they move apart. Nearby
        points ($|x-x'|\ll\ell$) are strongly correlated (kernel near
        $\eta^2$); far-apart points ($|x-x'|\gg\ell$) are nearly independent
        (kernel near 0).

        Write a function `expquad(x1, x2, ls, eta)` that takes two 1-D arrays
        of input locations plus a lengthscale and amplitude and returns the
        `(len(x1), len(x2))` covariance matrix. Hint:
        `np.subtract.outer(x1, x2)` gives every pairwise difference at once.
        Try it, then expand the solution.
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
                    dist = np.subtract.outer(x1, x2)      # pairwise differences
                    return eta**2 * np.exp(-0.5 * dist**2 / ls**2)
                ```

                `np.subtract.outer` builds the full matrix of pairwise
                differences $x_i - x'_j$ in one vectorized call; squaring,
                scaling by the lengthscale, exponentiating, and multiplying by
                $\\eta^2$ then applies the formula elementwise. No Python loop
                needed.
                """
            )
        }
    )
    return (expquad,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        As a sanity check, PyMC ships this exact kernel as
        `pm.gp.cov.ExpQuad`. One important convention: PyMC's GP covariance
        functions always expect **2-D inputs of shape `(n, 1)`** (one row per
        input point), even in a single input dimension — unlike our
        from-scratch version, which takes plain 1-D arrays. We keep GP inputs
        2-D throughout the workshop.
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
        ### The Gram matrix and mean/covariance functions

        Evaluating the kernel on every pair of points from a grid produces
        the **Gram matrix** (or covariance matrix) $K$, with
        $K_{ij} = k(x_i, x_j)$. It is symmetric, has $\eta^2$ down its
        diagonal (a point's covariance with itself), and its off-diagonal
        entries fade smoothly to zero as points get farther apart. This
        matrix *is* the covariance of the MVN you get by evaluating the GP on
        that grid. Below, a heatmap of $K$ for an ExpQuad kernel over an
        evenly spaced grid.
        """
    )
    return


@app.cell
def _(expquad, go, np):
    gram_grid = np.linspace(0, 10, 40)
    gram_K = expquad(gram_grid, gram_grid, ls=1.5, eta=1.0)

    gram_fig = go.Figure(
        data=go.Heatmap(
            x=gram_grid,
            y=gram_grid,
            z=gram_K,
            colorscale="Blues",
            colorbar=dict(title="k(x, x')"),
        )
    )
    gram_fig.update_layout(
        title="Gram matrix of the ExpQuad kernel (ℓ=1.5, η=1.0)",
        xaxis_title="x",
        yaxis_title="x'",
        template="plotly_white",
        yaxis=dict(autorange="reversed"),
    )
    gram_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The bright diagonal band is the signature of a stationary smooth
        kernel: strong covariance for nearby points, decaying to near zero
        away from the diagonal, with the *width* of the band set by the
        lengthscale. A quick summary of what the pieces of a GP control:

        - **Mean function $m(x)$** — the function values' expected level
          *before* data. Often taken as $0$ (after standardizing the output),
          so the GP models departures from zero; a linear or other simple
          mean can be added when you expect a trend, as in Notebook 2.
        - **Lengthscale $\ell$** — the horizontal "wiggle scale". Small
          $\ell$ ⇒ correlation dies quickly ⇒ wiggly functions that can bend
          on a fine scale; large $\ell$ ⇒ correlation persists ⇒ smooth,
          slowly varying functions.
        - **Amplitude $\eta$** — the vertical scale. It sets how far function
          values wander from the mean, without changing their smoothness.
        - **Noise $\sigma$** (in regression) — scatter of *observations*
          around the latent function, added on the diagonal when we fit to
          data.

        ### Drawing sample functions from a GP prior

        With a kernel in hand we can *sample whole functions* from the GP
        prior: evaluate $K$ on a grid, add a tiny **jitter** to the diagonal
        for numerical stability (so the Cholesky factorization succeeds), and
        draw from $\mathcal N(\mathbf 0, K)$. Each draw is one plausible
        function under the prior.
        """
    )
    return


@app.cell
def _(expquad, go, np, rng):
    prior_grid = np.linspace(0, 10, 200)
    prior_K = expquad(prior_grid, prior_grid, ls=1.5, eta=1.0)
    prior_K = prior_K + 1e-8 * np.eye(len(prior_grid))  # jitter for stability

    prior_samples = rng.multivariate_normal(np.zeros(len(prior_grid)), prior_K, size=6)

    gp_prior_fig = go.Figure()
    for _i in range(prior_samples.shape[0]):
        gp_prior_fig.add_trace(
            go.Scatter(
                x=prior_grid,
                y=prior_samples[_i],
                mode="lines",
                line=dict(width=2),
                showlegend=False,
            )
        )
    gp_prior_fig.update_layout(
        title="Six sample functions from a GP prior (ExpQuad, ℓ=1.5, η=1.0)",
        xaxis_title="x",
        yaxis_title="f(x)",
        template="plotly_white",
    )
    gp_prior_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Each curve is a single draw from the *prior over functions*. They
        are all smooth (the ExpQuad kernel is infinitely differentiable),
        they wander over a similar vertical range (set by $\eta$), and they
        wiggle on a similar horizontal scale (set by $\ell$) — yet no two are
        alike. This cloud of curves is the GP's prior belief: "the function
        is some smooth wiggle of about this amplitude and this lengthscale,
        but I don't yet know which one." Data will narrow the cloud.

        ### GP regression *is* conditioning

        Here is the payoff. GP regression is nothing more than the MVN
        **conditioning** you did by hand earlier, applied to function values.
        Stack the (noisy) observed outputs $\mathbf y$ at training inputs $X$
        together with the unknown function values $\mathbf f_*$ at test inputs
        $X_*$; under the GP prior they are *jointly* MVN. Conditioning
        $\mathbf f_*$ on $\mathbf y$ gives the posterior

        $$\mathbf f_* \mid \mathbf y \sim \mathcal N\big(
        K_{*}(K + \sigma^2 I)^{-1}\mathbf y,\ \
        K_{**} - K_{*}(K + \sigma^2 I)^{-1}K_{*}^{\top}\big),$$

        where $K = k(X,X)$, $K_* = k(X_*,X)$, $K_{**} = k(X_*,X_*)$, and
        $\sigma^2$ is observation-noise variance added to the training block.
        This is *exactly* the conditioning formula from the bivariate example
        — same shape, just with matrices. Let's do it by hand on the
        theophylline subject that defeated the piecewise model, plugging in
        sensible hyperparameters (real inference over $\ell,\eta,\sigma$ is
        Notebook 2's job).
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
    expquad,
    go,
    np,
    time_grid,
    time_mean,
    time_std,
    time_vals,
    time_z,
    conc_z,
):
    # Condition a zero-mean GP prior on subject 1's standardized data by hand.
    cond_ls, cond_eta, cond_noise = 0.6, 1.0, 0.25
    Xtr = time_z  # 1-D standardized training inputs
    ytr = conc_z
    Xstar = (time_grid - time_mean) / time_std  # standardized test grid

    K_tr = expquad(Xtr, Xtr, cond_ls, cond_eta) + cond_noise**2 * np.eye(len(Xtr))
    K_s = expquad(Xstar, Xtr, cond_ls, cond_eta)
    K_ss = expquad(Xstar, Xstar, cond_ls, cond_eta)

    _solve = np.linalg.solve(K_tr, ytr)
    post_mean_z = K_s @ _solve
    post_cov = K_ss - K_s @ np.linalg.solve(K_tr, K_s.T)
    post_sd_z = np.sqrt(np.clip(np.diag(post_cov), 0, None))

    # Back to original mg/L units for plotting.
    gp_post_mean = post_mean_z * conc_std + conc_mean
    gp_post_lo = (post_mean_z - 2 * post_sd_z) * conc_std + conc_mean
    gp_post_hi = (post_mean_z + 2 * post_sd_z) * conc_std + conc_mean

    gp_reg_fig = go.Figure()
    gp_reg_fig.add_trace(
        go.Scatter(
            x=np.concatenate([time_grid, time_grid[::-1]]),
            y=np.concatenate([gp_post_hi, gp_post_lo[::-1]]),
            fill="toself",
            fillcolor="rgba(21,74,114,0.25)",
            line=dict(color="rgba(255,255,255,0)"),
            name="posterior mean ± 2 sd",
        )
    )
    gp_reg_fig.add_trace(
        go.Scatter(
            x=time_grid,
            y=gp_post_mean,
            mode="lines",
            name="GP posterior mean",
            line=dict(color=PYMC_GREEN, width=3),
        )
    )
    gp_reg_fig.add_trace(
        go.Scatter(
            x=time_vals,
            y=conc_vals,
            mode="markers",
            name="observed",
            marker=dict(color=PYMC_BLUE, size=9),
        )
    )
    gp_reg_fig.update_layout(
        title="GP conditioned on subject 1 — the smooth curve the piecewise model couldn't be",
        xaxis_title="Time since dose (hours)",
        yaxis_title="Concentration (mg/L)",
        template="plotly_white",
    )
    gp_reg_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Compare this directly with the piecewise-linear fit from Part A. The
        GP posterior mean glides **smoothly** through the rise, rounds the
        peak without any corner, and eases down the decay — no knot to place,
        no $\tau$ to argue over. The shaded ±2 sd band is *narrow where data
        are dense* (the early rise) and *widens where data are sparse* (the
        long tail and beyond the last point), which is precisely the honest
        uncertainty behavior we wanted and the piecewise model could not
        provide. We obtained all of it by conditioning a Gaussian — the same
        operation as the two-scalar example, scaled up to a whole function.
        **This is the flexible function the piecewise-linear model could not
        be.**

        We fixed $\ell$, $\eta$, and $\sigma$ by hand here to isolate the
        conditioning idea. In Notebook 2 we put priors on those
        hyperparameters and let PyMC infer them, with full posterior
        uncertainty — the natural next step now that the concept is in place.

        ### Widget: feel the hyperparameters

        Before moving on, build intuition for $\ell$ and $\eta$ by drawing
        prior samples reactively. **Predict before you move it:** think about
        what will happen to the sampled functions as you *shorten* the
        lengthscale, or *raise* the amplitude — then drag the sliders and
        check yourself.
        """
    )
    return


@app.cell
def _(mo):
    ls_slider = mo.ui.slider(0.1, 3.0, value=1.0, step=0.1, label="Lengthscale ℓ")
    eta_slider = mo.ui.slider(0.1, 3.0, value=1.0, step=0.1, label="Amplitude η")
    mo.hstack([ls_slider, eta_slider], gap=2)
    return eta_slider, ls_slider


@app.cell
def _(PYMC_DARK_GREEN, eta_slider, expquad, go, ls_slider, np, rng):
    widget_grid = np.linspace(0, 10, 200)
    widget_K = expquad(
        widget_grid, widget_grid, ls=ls_slider.value, eta=eta_slider.value
    )
    widget_K = widget_K + 1e-8 * np.eye(len(widget_grid))  # jitter

    widget_draws = rng.multivariate_normal(np.zeros(len(widget_grid)), widget_K, size=5)

    widget_fig = go.Figure()
    for _i in range(widget_draws.shape[0]):
        widget_fig.add_trace(
            go.Scatter(
                x=widget_grid,
                y=widget_draws[_i],
                mode="lines",
                line=dict(width=2),
                showlegend=False,
            )
        )
    widget_fig.update_layout(
        title=(
            f"GP prior draws — ℓ = {ls_slider.value:.1f}, η = {eta_slider.value:.1f}"
        ),
        xaxis_title="x",
        yaxis_title="f(x)",
        template="plotly_white",
        yaxis=dict(range=[-7, 7]),
    )
    widget_fig.update_traces(line_color=PYMC_DARK_GREEN, selector=dict(name="draw 1"))
    widget_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **What changed?** Dragging the **lengthscale** $\ell$ *down* toward
        0.1 makes the functions wigglier — correlation dies over short
        distances, so nearby values are freer to differ — while dragging it
        *up* toward 3.0 makes them smooth, gentle, slowly varying. Dragging
        the **amplitude** $\eta$ *up* stretches the functions vertically (note
        the fixed y-axis range: they run off the top and bottom) without
        changing how wiggly they are. Lengthscale controls *how fast* the
        function changes; amplitude controls *how far* it ranges. Those two
        knobs are most of what you tune (or infer) in practice.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Exercises — Part B

        These consolidate the conditioning-and-kernels intuition that the
        marginal and latent GP notebooks build on directly.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Exercise 4 — reason about smoothness from the lengthscale": mo.md(
                r"""
                In the hand-built GP regression above we set the lengthscale
                to `cond_ls = 0.6` on the *standardized* time axis. If you
                halved it to 0.3, would the posterior mean curve get smoother
                or wigglier, and what would happen to the fit near the sparse
                tail?

                **Solution.** Halving the lengthscale makes the GP *wigglier*:
                correlation now decays over a shorter distance, so the
                posterior mean is freer to bend between points and will chase
                the data more closely — potentially over-fitting the early,
                densely sampled rise. In the sparse tail the effect is
                different and instructive: with a short lengthscale, test
                points quickly lose correlation with any training point, so
                the posterior mean **reverts to the prior mean (zero)** faster
                and the uncertainty band balloons sooner past the data. A
                longer lengthscale extrapolates its trend further but risks
                oversmoothing the rise. Choosing $\ell$ is exactly this
                trade-off — which is why we *infer* it in Notebook 2 rather
                than guess.
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Exercise 5 — condition on an added point": mo.md(
                r"""
                Imagine appending one extra, very precise observation right at
                the peak of the curve (say a measurement at 1.5 hours). Using
                only the conditioning formula
                $\mathbf f_*\mid\mathbf y \sim \mathcal N(K_*(K+\sigma^2
                I)^{-1}\mathbf y,\ K_{**}-K_*(K+\sigma^2 I)^{-1}K_*^\top)$,
                what happens to the posterior *mean* and *variance* near that
                new point?

                **Solution.** Adding a point augments $X$ (and $\mathbf y$)
                with one more row/column, so $K_*$ gains a column that is
                large for test points near 1.5 hours. Through the mean term
                $K_*(K+\sigma^2 I)^{-1}\mathbf y$, the posterior mean is pulled
                toward the new observation in its neighbourhood; through the
                variance term $K_{**}-K_*(K+\sigma^2 I)^{-1}K_*^\top$, the
                subtracted quantity grows there, so the **posterior variance
                shrinks** near the new point. A precise ($\sigma$ small)
                measurement shrinks it more. Far from 1.5 hours — beyond a
                lengthscale away — the new point is nearly uncorrelated and
                barely changes anything. This is conditioning "tightening the
                cloud" exactly where information arrives, the same shrink you
                saw in the bivariate example.
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Exercise 6 — vary the amplitude and predict the band width": mo.md(
                r"""
                In the slider widget, suppose you doubled the amplitude $\eta$
                but left the lengthscale fixed. Would the *prior* uncertainty
                band around a future GP-regression fit get wider or narrower,
                and would the functions become any wigglier?

                **Solution.** Doubling $\eta$ doubles the prior standard
                deviation of the function values ($k(x,x)=\eta^2$), so the
                cloud of prior functions spans a **wider** vertical range and
                a prior predictive band would be correspondingly wider — but
                the functions are **not** any wigglier, because smoothness is
                governed solely by the lengthscale, which you held fixed. This
                separation of concerns — amplitude sets vertical scale,
                lengthscale sets horizontal wiggle — is why the two
                hyperparameters are usually given independent priors and
                interpreted separately, as we do in the notebooks that follow.
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Where we are, and what's next

        You built the Gaussian process from the ground up. You saw *why* a
        parametric model can fail on a smooth-but-formless curve (the
        piecewise-linear kink and its poorly identified peak time), you met
        the full PyMC workflow (priors, prior predictive checks, sampling,
        and the divergence + `r_hat` + ESS convergence triad), and you
        assembled a GP out of two familiar MVN operations — marginalization
        and conditioning — coding the ExpQuad kernel yourself and
        conditioning a GP on real data by hand.

        The one thing we *did not* do was infer the GP hyperparameters:
        $\ell$, $\eta$, and $\sigma$ were fixed by hand. **Notebook 2** closes
        that gap. It puts priors on the hyperparameters and lets PyMC sample
        the posterior — first in the analytically convenient *marginal*
        (Gaussian-likelihood) case with `pm.gp.Marginal`, then in the
        *latent* (non-Gaussian) case with `pm.gp.Latent` — turning the
        by-hand conditioning of this notebook into a full, uncertainty-aware
        Bayesian GP regression.
        """
    )
    return


if __name__ == "__main__":
    app.run()
