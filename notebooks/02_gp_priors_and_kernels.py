import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    import inspect

    import sys
    from pathlib import Path

    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from inference_contract import eti_bounds
    from exercises import exercise

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import polars as pl
    import pymc as pm
    import xarray as xr
    from scipy.stats import multivariate_normal, norm

    PYMC_BLUE = "#154A72"
    PYMC_GREEN = "#81C240"
    PYMC_LIGHT_BLUE = "#4A9EDE"
    PYMC_DARK_GREEN = "#40611F"

    RANDOM_SEED = 42

    data_dir = project_root / "data"

    def z(a):
        """Standardize an array: (a - mean) / population std."""
        return (a - a.mean()) / a.std(ddof=0)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Gaussian Process Priors and Covariance Functions

    Unlike splines or polynomial models, building Gaussian process models is about specifying a belief about *smoothness* and and then letting the data
    define the shape. We will help you try to develop an intuition about this approach here.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## From multivariate normals to Gaussian processes

    We want a model that assumes only *smoothness* and lets the data
    supply the shape, rather than committing to a fixed functional form
    up front. Remarkably, the tool for that is built entirely out of the
    **multivariate normal (MVN)** distribution you already know. This
    section assembles it piece by piece: two key properties of the MVN, a
    covariance function, sample functions drawn from a
    GP prior, and finally conditioning a GP on data, the operation that
    *is* GP regression.

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
    ignoring the rest, is *again normal*, you simply read off the
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
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Take two scalars $x_1$ and $x_2$ that are jointly normal, with

    $$\boldsymbol\mu = \begin{bmatrix} 1.0 \\ 2.0 \end{bmatrix},
    \qquad
    \Sigma = \begin{bmatrix} 1.0 & 0.8 \\ 0.8 & 1.5 \end{bmatrix}.$$

    Marginalizing to $x_1$ reads off the top-left entries, giving
    $x_1 \sim \mathcal N(1.0,\ 1.0)$. Conditioning on an observed
    $x_2 = 3.5$ puts the same numbers through the formula above.
    """)
    return


@app.cell
def _():
    biv_mean = np.array([1.0, 2.0])
    biv_cov = np.array([[1.0, 0.8], [0.8, 1.5]])
    x2_obs = 3.5

    cond_mean = biv_mean[0] + biv_cov[0, 1] / biv_cov[1, 1] * (x2_obs - biv_mean[1])
    cond_var = biv_cov[0, 0] - biv_cov[0, 1] ** 2 / biv_cov[1, 1]
    cond_sd = np.sqrt(cond_var)

    pl.DataFrame(
        {
            "distribution": ["marginal x1", "conditional x1 | x2 = 3.5"],
            "mean": [biv_mean[0], cond_mean],
            "variance": [biv_cov[0, 0], cond_var],
        }
    )
    return biv_cov, biv_mean, cond_mean, cond_sd, cond_var, x2_obs


@app.cell(hide_code=True)
def _(biv_cov, biv_mean, cond_mean, cond_var, x2_obs):
    mo.md(f"""
    The observed $x_2$ is {x2_obs - biv_mean[1]:.1f} above its own mean,
    and the positive covariance pulls $x_1$ up with it, from
    {biv_mean[0]:.1f} to {cond_mean:.2f}. The variance drops from
    {biv_cov[0, 0]:.1f} to {cond_var:.3f}, because knowing $x_2$ tells us
    something about $x_1$.

    That shift toward the data with reduced uncertainty is the whole idea
    of GP regression, applied to function values instead of two scalars.
    """)
    return


@app.cell(hide_code=True)
def _(biv_cov, biv_mean):
    cond_grid = np.linspace(-3, 6, 160)
    _X1, _X2 = np.meshgrid(cond_grid, cond_grid)
    _pos = np.dstack((_X1, _X2))
    cond_dens = multivariate_normal(biv_mean, biv_cov).pdf(_pos)
    return cond_dens, cond_grid


@app.cell(hide_code=True)
def _(cond_dens, cond_grid, cond_mean, cond_sd, x2_obs):
    cond_fig = go.Figure()
    cond_fig.add_trace(
        go.Contour(
            x=cond_grid,
            y=cond_grid,
            z=cond_dens,
            colorscale="Blues",
            showscale=False,
            contours=dict(showlabels=False),
        )
    )
    # The horizontal slice x2 = x2_obs that defines the conditional p(x1 | x2).
    cond_fig.add_hline(y=x2_obs, line=dict(color="#40611F", width=2, dash="dash"))
    # Overlay the resulting 1D conditional density of x1 along that slice.
    _cond_curve = norm.pdf(cond_grid, cond_mean, cond_sd)
    cond_fig.add_trace(
        go.Scatter(
            x=cond_grid,
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
def _():
    mo.md(r"""
    The contours are the joint density; the dashed line is the slice
    $x_2 = 3.5$. The green curve is the conditional $p(x_1 \mid x_2=3.5)$
       , a renormalized 1-D Gaussian along that slice, centred to the right
    of $x_1$'s marginal mean and narrower than it. Slide the slice up or
    down (mentally) and the conditional peak tracks with it: that
    tracking is the covariance doing its job.

    ### A Gaussian process is a distribution over functions

    A **Gaussian process** generalizes the MVN to
    *infinitely many* variables. Formally, a GP is a collection of random
    variables, **any finite subset of which is jointly multivariate
    normal**. If we think of a function $f$ as an infinitely long vector
       , one entry $f(x)$ for every input $x$, then a GP is a probability
    distribution over such functions:

    $$f(x) \sim \mathcal{GP}\big(m(x),\ k(x, x')\big).$$

    Just as an MVN needs a mean *vector* and covariance *matrix*, a GP
    needs a **mean function** $m(x)$ and a **covariance function**
    $k(x, x')$ (the *kernel*), which returns the covariance between the
    function's values at any two inputs.

    The **marginalization** property
    is what makes this usable: to work with a GP at a finite set of input
    points, we just evaluate $m$ and $k$ on that set to get an ordinary
    MVN and proceed, the infinitely many unqueried points marginalize
    away for free.

    The **conditioning** property is what turns
    the GP prior into a posterior given data, and allows us to make predictions from a fitted GP.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### The exponentiated quadratic kernel

    To go further we need a concrete kernel. The most common one is the
    **exponentiated quadratic**, also called the squared exponential or
    RBF:

    $$k(x, x') = \eta^2 \exp\!\left(-\frac{(x - x')^2}{2\ell^2}\right).$$

    The covariance is $\eta^2$ when two points coincide and decays
    smoothly to zero as they separate, at a rate set by the lengthscale
    $\ell$. PyMC provides it as `pm.gp.cov.ExpQuad(input_dim, ls)`.

    One convention to note: PyMC's covariance functions expect inputs of
    shape `(n, 1)` rather than plain 1-D arrays, even in a single input
    dimension. We keep GP inputs 2-D throughout the workshop.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### The Gram matrix and mean/covariance functions

    Evaluating the kernel on every pair of points from a grid produces
    the **Gram matrix** (or covariance matrix) $K$, with
    $K_{ij} = k(x_i, x_j)$. It is symmetric, has $\eta^2$ down its
    diagonal (a point's covariance with itself), and its off-diagonal
    entries fade smoothly to zero as points get farther apart. This
    matrix *is* the covariance of the MVN you get by evaluating the GP on
    that grid. Below, a heatmap of $K$ for an ExpQuad kernel over an
    evenly spaced grid.
    """)
    return


@app.cell
def _():
    gram_grid = np.linspace(0, 10, 40)
    gram_K = pm.gp.cov.ExpQuad(1, 1.5)(gram_grid[:, None]).eval()
    return gram_K, gram_grid


@app.cell(hide_code=True)
def _(gram_K, gram_grid):
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
def _():
    mo.md(r"""
    The bright diagonal band is the signature of a stationary smooth
    kernel: strong covariance for nearby points, decaying to near zero
    away from the diagonal, with the *width* of the band set by the
    lengthscale. A quick summary of what the pieces of a GP control:

    - **Mean function $m(x)$**, the function values' expected level
      *before* data. Often taken as $0$ (after standardizing the output),
      so the GP models departures from zero; a linear or other simple
      mean can be added when you expect a trend.
    - **Lengthscale $\ell$**, the horizontal "wiggle scale". Small
      $\ell$ ⇒ correlation dies quickly ⇒ wiggly functions that can bend
      on a fine scale; large $\ell$ ⇒ correlation persists ⇒ smooth,
      slowly varying functions.
    - **Amplitude $\eta$**, the vertical scale. It sets how far function
      values wander from the mean, without changing their smoothness.
    - **Noise $\sigma$** (in regression), scatter of *observations*
      around the latent function, added on the diagonal when we fit to
      data.

    ### Drawing sample functions from a GP prior

    With a kernel in hand we can *sample whole functions* from the GP
    prior. It is worth doing that slowly the first time, one point at a
    time, because it shows where the function comes from.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Start with no data at all. Under a zero-mean prior every function
    value is $\mathcal N(0, \eta^2)$ on its own, which is the flat band
    below. Now sample a value at a single input. Every other point on the
    grid is correlated with it, so conditioning on it, with the same
    formula used for $x_1 \mid x_2$, gives a new mean and a narrower
    variance everywhere nearby. Sample the next point from *that*
    conditional, condition again, and repeat.

    Drag the slider to add points one at a time and watch the band
    collapse onto a single curve. **New realization** reseeds and starts
    over.
    """)
    return


@app.cell(hide_code=True)
def _():
    seq_amplitude, seq_ls = 1.0, 0.5
    seq_x_locations = [
        1.0, -0.7, -2.1, -1.5, 0.3, 1.8, 2.5,
        -2.7, -2.4, -1.8, -1.2, -0.3, 0.8, 1.4, 2.2,
    ]


    def seq_covariance(a, b):
        """The ExpQuad kernel in numpy, so the demo can call it in a loop."""
        return seq_amplitude**2 * np.exp(
            -0.5 * np.subtract.outer(a, b) ** 2 / seq_ls**2
        )


    def seq_states(seed):
        """Draw each point from its conditional given the points already
        drawn, recording the conditional over the whole grid at each step."""
        rng = np.random.default_rng(seed)
        x_grid = np.linspace(-3, 3, 400)
        x, y = [], []
        prior_variance = seq_covariance(0.0, 0.0)
        states = [
            (
                np.array(x),
                np.array(y),
                np.zeros_like(x_grid),
                np.full_like(x_grid, prior_variance, dtype=float),
            )
        ]

        for x_new in seq_x_locations:
            if x:
                cross = seq_covariance([x_new], x)
                observed = seq_covariance(x, x)
                mean = cross @ np.linalg.solve(observed, y)
                variance = seq_covariance([x_new], [x_new]) - cross @ np.linalg.solve(
                    observed, cross.T
                )
                y_new = rng.normal(mean.item(), np.sqrt(variance.item()))
            else:
                y_new = rng.normal(scale=np.sqrt(prior_variance))
            x.append(x_new)
            y.append(y_new)

            observed = seq_covariance(x, x)
            cross = seq_covariance(x_grid, x)
            mean_grid = cross @ np.linalg.solve(observed, y)
            variance_grid = (
                np.diag(seq_covariance(x_grid, x_grid))
                - np.sum(cross * np.linalg.solve(observed, cross.T).T, axis=1)
                + 1e-8
            )
            states.append((np.array(x), np.array(y), mean_grid, variance_grid))

        return x_grid, states

    return seq_states, seq_x_locations


@app.cell(hide_code=True)
def _(seq_x_locations):
    seq_get_seed, seq_set_seed = mo.state(RANDOM_SEED)
    seq_get_point_count, seq_set_point_count = mo.state(0)

    seq_slider = mo.ui.slider(
        0,
        len(seq_x_locations),
        value=seq_get_point_count(),
        step=1,
        label="Conditional points",
        on_change=seq_set_point_count,
    )
    seq_reset = mo.ui.button(
        label="New realization",
        on_click=lambda _: (seq_set_seed(lambda s: s + 1), seq_set_point_count(0)),
    )
    mo.hstack([seq_slider, seq_reset])
    return seq_get_point_count, seq_get_seed


@app.cell(hide_code=True)
def _(seq_get_point_count, seq_get_seed, seq_states):
    seq_x_grid, seq_all_states = seq_states(seq_get_seed())
    seq_n_points = seq_get_point_count()
    seq_x_pts, seq_y_pts, seq_mean, seq_variance = seq_all_states[seq_n_points]
    seq_sd = np.sqrt(seq_variance)

    seq_fig = go.Figure()
    seq_fig.add_trace(
        go.Scatter(
            x=np.concatenate([seq_x_grid, seq_x_grid[::-1]]),
            y=np.concatenate([seq_mean + seq_sd, (seq_mean - seq_sd)[::-1]]),
            fill="toself",
            fillcolor="rgba(21,74,114,0.25)",
            line=dict(color="rgba(255,255,255,0)"),
            name="\u00b11 sd",
        )
    )
    seq_fig.add_trace(
        go.Scatter(
            x=seq_x_grid,
            y=seq_mean,
            mode="lines",
            line=dict(color=PYMC_BLUE, width=2),
            name="conditional mean",
        )
    )
    seq_fig.add_trace(
        go.Scatter(
            x=seq_x_pts,
            y=seq_y_pts,
            mode="markers",
            marker=dict(color=PYMC_GREEN, size=10),
            name="sampled points",
        )
    )
    seq_fig.update_layout(
        title=f"One realization of the GP prior, conditioned on {seq_n_points} points",
        xaxis_title="x",
        yaxis_title="f(x)",
        xaxis=dict(range=[-3, 3]),
        yaxis=dict(range=[-3, 3]),
        template="plotly_white",
        height=420,
    )
    seq_fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    By the last point the band has closed and what is the sillhouette of a
    smooth curve: one draw from the GP prior, built entirely out of
    conditioning.

    Nobody samples a GP this way in practice. Because *any* finite set of
    function values is jointly normal, we can skip the sequence and
    evaluate the kernel on the whole grid at once, then draw from
    $\mathcal N(\mathbf 0, K)$ in one step.

    In either case, each draw is one plausible
    function under the prior.
    """)
    return


@app.cell
def _():
    prior_grid = np.linspace(0, 10, 200)
    prior_K = pm.gp.cov.ExpQuad(1, 1.5)(prior_grid[:, None]).eval()
    prior_K = prior_K + 1e-8 * np.eye(len(prior_grid))  # jitter for stability

    _rng = np.random.default_rng(RANDOM_SEED)
    prior_samples = _rng.multivariate_normal(np.zeros(len(prior_grid)), prior_K, size=6)
    return prior_grid, prior_samples


@app.cell(hide_code=True)
def _(prior_grid, prior_samples):
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
def _():
    mo.md(r"""
    Each curve is a single draw from the *prior over functions*. They
    are all smooth (the ExpQuad kernel is infinitely differentiable),
    they wander over a similar vertical range (set by $\eta$), and they
    wiggle on a similar horizontal scale (set by $\ell$), yet no two are
    alike. This cloud of curves is the GP's prior belief: "the function
    is some smooth wiggle of about this amplitude and this lengthscale,
    but I don't yet know which one." Data will narrow the cloud, and the
    last section of this notebook shows how.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Widget: feel the hyperparameters

    Before moving on, build intuition for $\ell$ and $\eta$ by drawing
    prior samples reactively. **Predict before you move it:** think about
    what will happen to the sampled functions as you *shorten* the
    lengthscale, or *raise* the amplitude, then drag the sliders and
    check yourself.
    """)
    return


@app.cell(hide_code=True)
def _():
    ls_slider = mo.ui.slider(0.1, 3.0, value=1.0, step=0.1, label="Lengthscale ℓ")
    eta_slider = mo.ui.slider(0.1, 3.0, value=1.0, step=0.1, label="Amplitude η")
    return eta_slider, ls_slider


@app.cell
def _(eta_slider, ls_slider):
    widget_grid = np.linspace(0, 10, 200)
    widget_K = (
        eta_slider.value**2
        * pm.gp.cov.ExpQuad(1, ls_slider.value)(widget_grid[:, None]).eval()
    )
    widget_K = widget_K + 1e-8 * np.eye(len(widget_grid))  # jitter
    widget_rng = np.random.default_rng(
        RANDOM_SEED + round(100 * ls_slider.value) + round(1_000 * eta_slider.value)
    )
    widget_draws = widget_rng.multivariate_normal(
        np.zeros(len(widget_grid)), widget_K, size=5
    )
    return widget_draws, widget_grid


@app.cell(hide_code=True)
def _(eta_slider, ls_slider, widget_draws, widget_grid):
    widget_fig = go.Figure()
    for _i in range(widget_draws.shape[0]):
        widget_fig.add_trace(
            go.Scatter(
                x=widget_grid,
                y=widget_draws[_i],
                mode="lines",
                line=dict(width=2, color=PYMC_DARK_GREEN),
                showlegend=False,
            )
        )
    widget_fig.update_layout(
        title="Prior draws: move a slider, then explain what changed",
        xaxis_title="x",
        yaxis_title="f(x)",
        template="plotly_white",
        height=400,
    )
    mo.vstack([mo.hstack([ls_slider, eta_slider], gap=2), widget_fig])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **What changed?** Dragging the **lengthscale** $\ell$ *down* toward
    0.1 makes the functions wigglier, correlation dies over short
    distances, so nearby values are freer to differ, while dragging it
    *up* toward 3.0 makes them smooth, gentle, slowly varying. Dragging
    the **amplitude** $\eta$ *up* stretches the functions vertically (note
    the fixed y-axis range: they run off the top and bottom) without
    changing how wiggly they are. Lengthscale controls *how fast* the
    function changes; amplitude controls *how far* it ranges. Those two
    knobs are most of what you tune (or infer) in practice.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.vstack(
        [
            mo.md(
                r"""
                **Amplitude, prior versus posterior.** Doubling
                $\eta$ doubles the prior standard deviation of function values,
                since $k(x,x)=\eta^2$, the widget shows that directly. Now
                predict what it does to the *conditioned* fit further up: set
                `cond_eta = 2.0` and estimate by what factor the 89% interval
                widens (a) around 1–4 hours, where observations are dense, and
                (b) across the 13–23 hour gap. Write both numbers down before
                you re-run the cell, then explain why they differ.
                """),
            mo.accordion(
                {
                    "Discussion": mo.md(
                        r"""
                        The prior spread doubles, but the posterior barely
                        notices where data are dense: the interval widens about
                        **1.1×** over 1–4 hours (1.72 → 1.88 mg/L), because
                        there the likelihood, not the prior amplitude, sets the
                        scale, with `cond_noise` fixed, more prior freedom is
                        immediately spent back on fitting the same points. In
                        the gap the widening is about **1.3×** (2.30 → 2.91
                        mg/L): far from any observation the conditional
                        distribution relaxes toward the prior, so the amplitude
                        reasserts itself. Neither region doubles, which is the
                        point, a prior scale is an upper bound on what the
                        data are allowed to override, not a direct setting of
                        posterior width. Smoothness is untouched throughout:
                        horizontal wiggliness is $\ell$'s job, and $\ell$ did
                        not move.
                        """
                    )
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The kernel gallery

    Every covariance function below is evaluated on the same input grid.
    Each section shows one kernel: what it assumes about the function,
    its covariance matrix, and five prior draws produced by that same
    matrix, the shape of a kernel is much easier to see than to read off
    its formula. ExpQuad, the three Matérn kernels, Cosine and Periodic
    are all **stationary** (covariance depends only on $x - x'$); Linear
    is not, and is flagged as such in its section.
    """)
    return


@app.cell(hide_code=True)
def _():
    def plot_cov(X, K, stationary=True):
        """Two views of one covariance matrix: k(x, x') as a function of
        distance from the origin (or, for a non-stationary kernel, the
        diagonal k(x, x) — a function of distance alone is meaningless
        there), beside the full matrix as a heatmap."""
        x = X.flatten()
        left_label = "k(x, x')" if stationary else "diag k(x, x)"
        fig = make_subplots(rows=1, cols=2, subplot_titles=(left_label, "Covariance matrix"))
        left_y = K[:, 0] if stationary else np.diag(K)
        fig.add_trace(
            go.Scatter(x=x, y=left_y, mode="lines", line=dict(color=PYMC_BLUE, width=2), showlegend=False),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Heatmap(x=x, y=x, z=K, colorscale="Blues", showscale=False),
            row=1,
            col=2,
        )
        fig.update_xaxes(title_text="x", row=1, col=1)
        fig.update_yaxes(title_text=left_label, row=1, col=1)
        fig.update_xaxes(title_text="x", row=1, col=2)
        fig.update_yaxes(title_text="x'", autorange="reversed", row=1, col=2)
        fig.update_layout(template="plotly_white", height=220, margin=dict(t=40, b=30))
        return fig


    def kernel_draws_figure(X, K, title, n_draws=5, seed=RANDOM_SEED):
        """Draw n_draws sample functions from N(0, K), one plausible
        function per draw under this kernel's prior."""
        rng = np.random.default_rng(seed)
        draws = rng.multivariate_normal(np.zeros(K.shape[0]), K, size=n_draws)
        fig = go.Figure()
        for i in range(n_draws):
            fig.add_trace(
                go.Scatter(x=X.flatten(), y=draws[i], mode="lines", line=dict(width=2), showlegend=False)
            )
        fig.update_layout(
            title=title,
            xaxis_title="x",
            yaxis_title="f(x)",
            template="plotly_white",
            height=220,
            margin=dict(t=40, b=30),
        )
        return fig

    return kernel_draws_figure, plot_cov


@app.cell(hide_code=True)
def _():
    kernel_x_grid = np.linspace(0, 2, 200).reshape(-1, 1)  # shared grid for every kernel-gallery figure
    return (kernel_x_grid,)


@app.function(hide_code=True)
def kernel_covariance(kernel_name, amplitude, lengthscale, extra):
    """Map a kernel name and its hyperparameters to a scaled PyMC
    covariance function and whether that kernel is stationary."""
    if kernel_name == "ExpQuad":
        return amplitude**2 * pm.gp.cov.ExpQuad(1, lengthscale), True
    if kernel_name == "Matern 1/2":
        return amplitude**2 * pm.gp.cov.Matern12(1, lengthscale), True
    if kernel_name == "Matern 3/2":
        return amplitude**2 * pm.gp.cov.Matern32(1, lengthscale), True
    if kernel_name == "Matern 5/2":
        return amplitude**2 * pm.gp.cov.Matern52(1, lengthscale), True
    if kernel_name == "Cosine":
        return amplitude**2 * pm.gp.cov.Cosine(1, lengthscale), True
    if kernel_name == "Periodic":
        return amplitude**2 * pm.gp.cov.Periodic(1, period=extra, ls=lengthscale), True
    if kernel_name == "Linear":
        return amplitude**2 * pm.gp.cov.Linear(1, c=extra), False
    raise ValueError(f"Unknown kernel: {kernel_name}")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Exponential quadratic (ExpQuad)

    $$
    k(x, x') = \eta^2\exp\!\left[-\frac{(x - x')^2}{2\ell^2}\right]
    $$

    The default GP kernel used earlier in this notebook. Two
    hyperparameters, yet flexible enough to approximate almost any smooth
    function, every draw from its prior is infinitely differentiable, no
    matter how short the lengthscale.
    """)
    return


@app.cell(hide_code=True)
def _():
    expquad_ls_slider = mo.ui.slider(0.05, 2.0, value=0.30, step=0.05, label="Lengthscale ℓ")
    expquad_eta_slider = mo.ui.slider(0.1, 5.0, value=1.0, step=0.1, label="Amplitude η")
    return expquad_eta_slider, expquad_ls_slider


@app.cell(hide_code=True)
def _(expquad_eta_slider, expquad_ls_slider, kernel_x_grid):
    expquad_cov, expquad_stationary = kernel_covariance(
        "ExpQuad", expquad_eta_slider.value, expquad_ls_slider.value, 0.0
    )
    expquad_K = expquad_cov(kernel_x_grid).eval()
    expquad_K = expquad_K + 1e-8 * np.eye(len(kernel_x_grid))  # jitter for numerical stability
    return expquad_K, expquad_stationary


@app.cell(hide_code=True)
def _(
    expquad_K,
    expquad_eta_slider,
    expquad_ls_slider,
    expquad_stationary,
    kernel_draws_figure,
    kernel_x_grid,
    plot_cov,
):
    expquad_cov_fig = plot_cov(kernel_x_grid, expquad_K, stationary=expquad_stationary)
    expquad_draws_fig = kernel_draws_figure(
        kernel_x_grid, expquad_K, title="Five draws from the ExpQuad prior"
    )
    mo.vstack([
                mo.hstack([expquad_ls_slider, expquad_eta_slider], gap=2, justify="start"),
                expquad_cov_fig,
                expquad_draws_fig,
            ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Matérn ν = 1/2

    $$
    k(x, x') = \eta^2\exp\!\left[-\frac{|x - x'|}{\ell}\right]
    $$

    Also called the exponential or Ornstein–Uhlenbeck kernel. Draws are
    continuous but **nowhere differentiable**, jagged at every scale, no
    matter how far you zoom in. This is one end of a roughness dial the
    Matérn family turns: $\nu=1/2$ is nowhere differentiable, $\nu=3/2$ is
    once, $\nu=5/2$ is twice, and ExpQuad, effectively $\nu\to\infty$, is
    infinitely smooth. The next two sections turn that dial.
    """)
    return


@app.cell(hide_code=True)
def _():
    matern12_ls_slider = mo.ui.slider(0.05, 2.0, value=0.30, step=0.05, label="Lengthscale ℓ")
    matern12_eta_slider = mo.ui.slider(0.1, 5.0, value=1.0, step=0.1, label="Amplitude η")
    return matern12_eta_slider, matern12_ls_slider


@app.cell(hide_code=True)
def _(kernel_x_grid, matern12_eta_slider, matern12_ls_slider):
    matern12_cov, matern12_stationary = kernel_covariance(
        "Matern 1/2", matern12_eta_slider.value, matern12_ls_slider.value, 0.0
    )
    matern12_K = matern12_cov(kernel_x_grid).eval()
    matern12_K = matern12_K + 1e-8 * np.eye(len(kernel_x_grid))  # jitter for numerical stability
    return matern12_K, matern12_stationary


@app.cell(hide_code=True)
def _(
    kernel_draws_figure,
    kernel_x_grid,
    matern12_K,
    matern12_eta_slider,
    matern12_ls_slider,
    matern12_stationary,
    plot_cov,
):
    matern12_cov_fig = plot_cov(kernel_x_grid, matern12_K, stationary=matern12_stationary)
    matern12_draws_fig = kernel_draws_figure(
        kernel_x_grid, matern12_K, title="Five draws from the Matern 1/2 prior"
    )
    mo.vstack([
                mo.hstack([matern12_ls_slider, matern12_eta_slider], gap=2, justify="start"),
                matern12_cov_fig,
                matern12_draws_fig,
            ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Matérn ν = 3/2

    $$
    k(x, x') = \eta^2\left(1 + \frac{\sqrt3\,|x-x'|}{\ell}\right)
    \exp\!\left[-\frac{\sqrt3\,|x-x'|}{\ell}\right]
    $$

    One notch smoother than $\nu=1/2$: draws are differentiable **once**.
    Compare it against the 1/2 section above at the same lengthscale ,
    visibly less jagged, but still short of ExpQuad's polish.
    """)
    return


@app.cell(hide_code=True)
def _():
    matern32_ls_slider = mo.ui.slider(0.05, 2.0, value=0.30, step=0.05, label="Lengthscale ℓ")
    matern32_eta_slider = mo.ui.slider(0.1, 5.0, value=1.0, step=0.1, label="Amplitude η")
    return matern32_eta_slider, matern32_ls_slider


@app.cell(hide_code=True)
def _(kernel_x_grid, matern32_eta_slider, matern32_ls_slider):
    matern32_cov, matern32_stationary = kernel_covariance(
        "Matern 3/2", matern32_eta_slider.value, matern32_ls_slider.value, 0.0
    )
    matern32_K = matern32_cov(kernel_x_grid).eval()
    matern32_K = matern32_K + 1e-8 * np.eye(len(kernel_x_grid))  # jitter for numerical stability
    return matern32_K, matern32_stationary


@app.cell(hide_code=True)
def _(
    kernel_draws_figure,
    kernel_x_grid,
    matern32_K,
    matern32_eta_slider,
    matern32_ls_slider,
    matern32_stationary,
    plot_cov,
):
    matern32_cov_fig = plot_cov(kernel_x_grid, matern32_K, stationary=matern32_stationary)
    matern32_draws_fig = kernel_draws_figure(
        kernel_x_grid, matern32_K, title="Five draws from the Matern 3/2 prior"
    )
    mo.vstack([
                mo.hstack([matern32_ls_slider, matern32_eta_slider], gap=2, justify="start"),
                matern32_cov_fig,
                matern32_draws_fig,
            ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Matérn ν = 5/2

    $$
    k(x, x') = \eta^2\left(1 + \frac{\sqrt5\,|x-x'|}{\ell} +
    \frac{5(x-x')^2}{3\ell^2}\right)\exp\!\left[-\frac{\sqrt5\,|x-x'|}{\ell}\right]
    $$

    Twice differentiable, smoother again, and close enough to ExpQuad by
    eye that the two can be hard to tell apart from draws alone. That
    closeness is exactly why Matérn 5/2 is the usual default for physical
    processes: it buys most of ExpQuad's smoothness without ExpQuad's much
    stronger claim, infinite differentiability at every scale, which is
    rarely true of anything physical. Matérn 5/2 is the default kernel
    used everywhere else in this workshop.
    """)
    return


@app.cell(hide_code=True)
def _():
    matern52_ls_slider = mo.ui.slider(0.05, 2.0, value=0.30, step=0.05, label="Lengthscale ℓ")
    matern52_eta_slider = mo.ui.slider(0.1, 5.0, value=1.0, step=0.1, label="Amplitude η")
    return matern52_eta_slider, matern52_ls_slider


@app.cell(hide_code=True)
def _(kernel_x_grid, matern52_eta_slider, matern52_ls_slider):
    matern52_cov, matern52_stationary = kernel_covariance(
        "Matern 5/2", matern52_eta_slider.value, matern52_ls_slider.value, 0.0
    )
    matern52_K = matern52_cov(kernel_x_grid).eval()
    matern52_K = matern52_K + 1e-8 * np.eye(len(kernel_x_grid))  # jitter for numerical stability
    return matern52_K, matern52_stationary


@app.cell(hide_code=True)
def _(
    kernel_draws_figure,
    kernel_x_grid,
    matern52_K,
    matern52_eta_slider,
    matern52_ls_slider,
    matern52_stationary,
    plot_cov,
):
    matern52_cov_fig = plot_cov(kernel_x_grid, matern52_K, stationary=matern52_stationary)
    matern52_draws_fig = kernel_draws_figure(
        kernel_x_grid, matern52_K, title="Five draws from the Matern 5/2 prior"
    )
    mo.vstack([
                mo.hstack([matern52_ls_slider, matern52_eta_slider], gap=2, justify="start"),
                matern52_cov_fig,
                matern52_draws_fig,
            ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Cosine

    $$
    k(x, x') = \eta^2\cos\!\left(2\pi\frac{|x - x'|}{\ell}\right)
    $$

    Models periodic behavior without a dedicated periodic kernel: here
    $\ell$ sets the period directly. But it is a single pure frequency
    with no smoothness control and no decay, draws oscillate forever at
    exactly one rate. Periodic, next, separates those two knobs.
    """)
    return


@app.cell(hide_code=True)
def _():
    cosine_ls_slider = mo.ui.slider(0.05, 2.0, value=0.30, step=0.05, label="Period scale ℓ")
    cosine_eta_slider = mo.ui.slider(0.1, 5.0, value=1.0, step=0.1, label="Amplitude η")
    return cosine_eta_slider, cosine_ls_slider


@app.cell(hide_code=True)
def _(cosine_eta_slider, cosine_ls_slider, kernel_x_grid):
    cosine_cov, cosine_stationary = kernel_covariance(
        "Cosine", cosine_eta_slider.value, cosine_ls_slider.value, 0.0
    )
    cosine_K = cosine_cov(kernel_x_grid).eval()
    cosine_K = cosine_K + 1e-8 * np.eye(len(kernel_x_grid))  # jitter for numerical stability
    return cosine_K, cosine_stationary


@app.cell(hide_code=True)
def _(
    cosine_K,
    cosine_eta_slider,
    cosine_ls_slider,
    cosine_stationary,
    kernel_draws_figure,
    kernel_x_grid,
    plot_cov,
):
    cosine_cov_fig = plot_cov(kernel_x_grid, cosine_K, stationary=cosine_stationary)
    cosine_draws_fig = kernel_draws_figure(
        kernel_x_grid, cosine_K, title="Five draws from the Cosine prior"
    )
    mo.vstack([
                mo.hstack([cosine_ls_slider, cosine_eta_slider], gap=2, justify="start"),
                cosine_cov_fig,
                cosine_draws_fig,
            ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Periodic

    $$
    k(x, x') = \eta^2\exp\!\left[-\frac{\sin^2(\pi |x-x'| / p)}{2\ell^2}\right]
    $$

    Unlike every kernel above, Periodic repeats **exactly and forever**:
    shift $x$ by one period $p$ and $k$ returns to the same value, so
    every prior draw is a genuinely periodic function. $\ell$ still
    controls how smooth each cycle is, small $\ell$ gives sharp, spiky
    repeats; large $\ell$ gives gentle, near-sinusoidal ones.

    One thing to get right: $p$ is measured in the **units of the input
    axis**. If you standardize your inputs before fitting, as the tide
    model later in this notebook does, a physical period (say, a
    12.42-hour tidal cycle) has to be divided by that same standardization
    factor before it means anything to this kernel. Hand it the raw
    physical number on a standardized axis and it looks for cycles that
    are not there.
    """)
    return


@app.cell(hide_code=True)
def _():
    periodic_ls_slider = mo.ui.slider(0.05, 1.0, value=0.20, step=0.05, label="Lengthscale ℓ (within-cycle)")
    periodic_eta_slider = mo.ui.slider(0.1, 5.0, value=1.0, step=0.1, label="Amplitude η")
    periodic_period_slider = mo.ui.slider(0.1, 1.0, value=0.40, step=0.05, label="Period p")
    return periodic_eta_slider, periodic_ls_slider, periodic_period_slider


@app.cell(hide_code=True)
def _(
    kernel_x_grid,
    periodic_eta_slider,
    periodic_ls_slider,
    periodic_period_slider,
):
    periodic_cov, periodic_stationary = kernel_covariance(
        "Periodic", periodic_eta_slider.value, periodic_ls_slider.value, periodic_period_slider.value
    )
    periodic_K = periodic_cov(kernel_x_grid).eval()
    periodic_K = periodic_K + 1e-8 * np.eye(len(kernel_x_grid))  # jitter for numerical stability
    return periodic_K, periodic_stationary


@app.cell(hide_code=True)
def _(
    kernel_draws_figure,
    kernel_x_grid,
    periodic_K,
    periodic_eta_slider,
    periodic_ls_slider,
    periodic_period_slider,
    periodic_stationary,
    plot_cov,
):
    periodic_cov_fig = plot_cov(kernel_x_grid, periodic_K, stationary=periodic_stationary)
    periodic_draws_fig = kernel_draws_figure(
        kernel_x_grid, periodic_K, title="Five draws from the Periodic prior"
    )
    mo.vstack([
                mo.hstack(
                [periodic_ls_slider, periodic_eta_slider, periodic_period_slider], gap=2, justify="start"
            ),
                periodic_cov_fig,
                periodic_draws_fig,
            ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Linear

    $$
    k(x, x') = \eta^2 (x - c)(x' - c)
    $$

    The simplest kernel here: the (scaled, offset) inner product of the
    inputs. Unlike every kernel above, Linear is **non-stationary**, its
    value depends on the absolute location $x$, not just the distance
    $x - x'$, so the panel below shows $k(x, x)$ against $x$ instead of
    $k(x, x')$ against distance, and draws are simply straight lines
    pivoting near $c$. It has no lengthscale at all, and is usually
    combined multiplicatively with another kernel rather than used alone.
    """)
    return


@app.cell(hide_code=True)
def _():
    linear_offset_slider = mo.ui.slider(-2.0, 2.0, value=1.0, step=0.1, label="Pivot c")
    linear_eta_slider = mo.ui.slider(0.1, 5.0, value=1.0, step=0.1, label="Amplitude η")
    return linear_eta_slider, linear_offset_slider


@app.cell(hide_code=True)
def _(kernel_x_grid, linear_eta_slider, linear_offset_slider):
    linear_cov, linear_stationary = kernel_covariance(
        "Linear", linear_eta_slider.value, 1.0, linear_offset_slider.value
    )
    linear_K = linear_cov(kernel_x_grid).eval()
    linear_K = linear_K + 1e-8 * np.eye(len(kernel_x_grid))  # jitter for numerical stability
    return linear_K, linear_stationary


@app.cell(hide_code=True)
def _(
    kernel_draws_figure,
    kernel_x_grid,
    linear_K,
    linear_eta_slider,
    linear_offset_slider,
    linear_stationary,
    plot_cov,
):
    linear_cov_fig = plot_cov(kernel_x_grid, linear_K, stationary=linear_stationary)
    linear_draws_fig = kernel_draws_figure(
        kernel_x_grid, linear_K, title="Five draws from the Linear prior"
    )
    mo.vstack([
                mo.hstack([linear_offset_slider, linear_eta_slider], gap=2, justify="start"),
                linear_cov_fig,
                linear_draws_fig,
            ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Combining kernels

    Modeling with a single kernel works when the data carries one kind of
    structure, but real signals often layer several kinds together. Two
    operations combine covariance functions into richer ones:
    **multiplication** and **addition**.

    Multiplying two kernels acts like an `AND`:
    $k(x,x') = k_1(x,x')\,k_2(x,x')$ is large only where *both* constituent
    kernels are large, so a product encodes "this structure holds, modulated
    by that one." Pick a left and right kernel below and see what their
    product's covariance and draws look like.
    """)
    return


@app.cell(hide_code=True)
def _():
    def default_kernel_params(kernel_name):
        """Reasonable default (amplitude, lengthscale, extra) for a kernel,
        matching the slider defaults used in its gallery section above."""
        if kernel_name == "Linear":
            return 1.0, 1.0, 1.0
        if kernel_name == "Periodic":
            return 1.0, 0.20, 0.40
        return 1.0, 0.30, 0.0


    gp_kernel_options = [
        "ExpQuad",
        "Matern 1/2",
        "Matern 3/2",
        "Matern 5/2",
        "Cosine",
        "Periodic",
        "Linear",
    ]
    return default_kernel_params, gp_kernel_options


@app.cell(hide_code=True)
def _(gp_kernel_options):
    product_left_kernel = mo.ui.dropdown(
        gp_kernel_options, value="Periodic", label="Left kernel"
    )
    product_right_kernel = mo.ui.dropdown(
        gp_kernel_options, value="ExpQuad", label="Right kernel"
    )
    return product_left_kernel, product_right_kernel


@app.cell(hide_code=True)
def _(
    default_kernel_params,
    kernel_x_grid,
    product_left_kernel,
    product_right_kernel,
):
    product_left_cov, product_left_stationary = kernel_covariance(
        product_left_kernel.value, *default_kernel_params(product_left_kernel.value)
    )
    product_right_cov, product_right_stationary = kernel_covariance(
        product_right_kernel.value, *default_kernel_params(product_right_kernel.value)
    )
    product_cov = product_left_cov * product_right_cov
    product_stationary = product_left_stationary and product_right_stationary
    product_K = product_cov(kernel_x_grid).eval()
    product_K = product_K + 1e-8 * np.eye(len(kernel_x_grid))  # jitter for numerical stability
    return product_K, product_stationary


@app.cell(hide_code=True)
def _(
    kernel_draws_figure,
    kernel_x_grid,
    plot_cov,
    product_K,
    product_left_kernel,
    product_right_kernel,
    product_stationary,
):
    product_cov_fig = plot_cov(kernel_x_grid, product_K, stationary=product_stationary)
    product_draws_fig = kernel_draws_figure(
        kernel_x_grid,
        product_K,
        title=f"Five draws from {product_left_kernel.value} x {product_right_kernel.value}",
    )
    mo.vstack([
                mo.hstack([product_left_kernel, product_right_kernel], gap=2, justify="start"),
                product_cov_fig,
                product_draws_fig,
            ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Compare this to addition. Adding two kernels acts like an `OR`:
    $k(x,x') = k_1(x,x') + k_2(x,x')$ is large if *either* constituent kernel
    is large, and a draw from the sum is (statistically) a draw from $k_1$
    plus an independent draw from $k_2$, useful when the data is a
    **superposition** of separate structures rather than one modulating the
    other.
    """)
    return


@app.cell(hide_code=True)
def _(gp_kernel_options):
    sum_left_kernel = mo.ui.dropdown(
        gp_kernel_options, value="Matern 5/2", label="Left kernel"
    )
    sum_right_kernel = mo.ui.dropdown(
        gp_kernel_options, value="Periodic", label="Right kernel"
    )
    return sum_left_kernel, sum_right_kernel


@app.cell(hide_code=True)
def _(default_kernel_params, kernel_x_grid, sum_left_kernel, sum_right_kernel):
    sum_left_cov, sum_left_stationary = kernel_covariance(
        sum_left_kernel.value, *default_kernel_params(sum_left_kernel.value)
    )
    sum_right_cov, sum_right_stationary = kernel_covariance(
        sum_right_kernel.value, *default_kernel_params(sum_right_kernel.value)
    )
    sum_cov = sum_left_cov + sum_right_cov
    sum_stationary = sum_left_stationary and sum_right_stationary
    sum_K = sum_cov(kernel_x_grid).eval()
    sum_K = sum_K + 1e-8 * np.eye(len(kernel_x_grid))  # jitter for numerical stability
    return sum_K, sum_stationary


@app.cell(hide_code=True)
def _(
    kernel_draws_figure,
    kernel_x_grid,
    plot_cov,
    sum_K,
    sum_left_kernel,
    sum_right_kernel,
    sum_stationary,
):
    sum_cov_fig = plot_cov(kernel_x_grid, sum_K, stationary=sum_stationary)
    sum_draws_fig = kernel_draws_figure(
        kernel_x_grid,
        sum_K,
        title=f"Five draws from {sum_left_kernel.value} + {sum_right_kernel.value}",
    )
    mo.vstack([
                mo.hstack([sum_left_kernel, sum_right_kernel], gap=2, justify="start"),
                sum_cov_fig,
                sum_draws_fig,
            ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    This additive structure is exactly what the NOAA tide-gauge fit below
    uses: a slow Matérn 5/2 trend plus two Periodic components, a
    semidiurnal and a diurnal tidal cycle, summed together. That is
    precisely the sum-of-kernels idea the widget above just demonstrated,
    now doing real work on real data.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.vstack(
        [
            mo.md(
                r"""
                **Question, predict a kernel product.** Before touching the
                widget above, predict: what would draws from the *product* of
                a `Periodic` kernel and a long-lengthscale `ExpQuad` look
                like? What real phenomenon would that product model?
                """),
            mo.accordion(
                {
                    "Discussion": mo.md(
                        r"""
                        The product multiplies a repeating structure by a
                        slowly-varying envelope: draws still oscillate at the
                        `Periodic` kernel's fixed period, but the *amplitude*
                        of the oscillation now rises and falls smoothly,
                        following the `ExpQuad` term's long lengthscale,
                        rather than staying constant forever the way a pure
                        `Periodic` draw does. That is a **locally periodic**
                        kernel, a period that persists but whose strength
                        drifts, the kind of structure that shows up in, for
                        example, a seasonal cycle whose amplitude itself
                        varies from year to year. Try it in the widget above:
                        set the left kernel to `Periodic` and the right to
                        `ExpQuad` with a lengthscale much longer than the
                        periodic term's period, and watch the envelope appear.
                        """
                    )
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Kernel composition, NOAA tide gauge

    ### Background

    NOAA CO-OPS station 9414290 (San Francisco, CA) is a long-record
    **mixed semidiurnal** tide station: the water level shows two
    superimposed periodic components, a roughly 12.42-hour
    **semidiurnal** (twice-daily) tide driven mainly by the moon, and a
    roughly 23.93-hour **diurnal** (once-daily) tide, riding on top of
    a slower background trend. Values below are hourly water levels in
    meters relative to the MLLW (mean lower low water) datum for a
    slice of 2019.
    """)
    return


@app.cell(hide_code=True)
def _():
    N_EXACT = 200
    tides = pl.read_csv(data_dir / "noaa_tides_hourly.csv")
    tides = tides.with_columns(
        pl.col("time").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M")
    )
    tides_slice = tides.head(N_EXACT)
    tides_slice.head()
    return (tides_slice,)


@app.cell(hide_code=True)
def _(tides_slice):
    tide_t0 = tides_slice["time"][0]
    tide_hours = (tides_slice["time"] - tide_t0).dt.total_minutes().to_numpy() / 60.0
    tide_level = tides_slice["water_level"].to_numpy()

    tide_hours_std = tide_hours.std(ddof=0)

    X_tide = z(tide_hours).reshape(-1, 1)  # GP inputs are 2D: (n, 1)
    y_tide = z(tide_level)
    return X_tide, tide_hours, tide_hours_std, tide_level, y_tide


@app.cell(hide_code=True)
def _(tide_hours, tide_level):
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
def _():
    mo.md(r"""
    ### Additive vs. multiplicative kernel structure

    Two ways to combine covariance functions:

    - **Additive (OR)**: $k(x,x') = k_1(x,x') + k_2(x,x')$. A draw from
      the sum is a draw from $k_1$ *plus* a (statistically independent)
      draw from $k_2$, useful when the data is a **superposition** of
      distinct structures, e.g. a slow trend plus fast periodic
      wiggles. This is the right structure for tides: a slowly-drifting
      mean level, plus a semidiurnal cycle, plus a diurnal cycle, added
      together.
    - **Multiplicative (AND)**: $k(x,x') = k_1(x,x') \cdot k_2(x,x')$.
      This is how you build kernels whose behavior along one dimension
      is *modulated* by another (e.g. a periodic kernel times a slowly
      decaying ExpQuad gives a periodic pattern that fades in and out ,
      `pm.gp.cov.Periodic` combined this way is one route to a
      quasi-periodic kernel), or how an ARD kernel over multiple input
      dimensions is built.

    Tides call for the additive form. Below we build a long-lengthscale
        `Matern52` trend plus two `Periodic` components, one per known
        physical cycle. The two periods are not free parameters: 12.42h and
        23.93h are astronomical constants, so we fix them, and we fix each
        component's within-cycle lengthscale too, at a value that gives a
        smooth roughly sinusoidal cycle rather than a sharp spike.
    """)
    return


@app.cell
def _(X_tide, tide_hours_std):
    tide_semi_period_std = 12.42 / tide_hours_std
    tide_diurnal_period_std = 23.93 / tide_hours_std

    tide_cov = (
        0.5**2 * pm.gp.cov.Matern52(1, ls=1.0)
        + 1.0**2 * pm.gp.cov.Periodic(1, period=tide_semi_period_std, ls=0.5)
        + 0.5**2 * pm.gp.cov.Periodic(1, period=tide_diurnal_period_std, ls=0.5)
    )
    tide_prior_K = tide_cov(X_tide).eval() + 1e-8 * np.eye(len(X_tide))
    return (tide_prior_K,)


@app.cell(hide_code=True)
def _(X_tide, tide_hours, tide_prior_K, y_tide):
    tide_prior_fig = go.Figure()
    _tide_draws = np.random.default_rng(RANDOM_SEED).multivariate_normal(
        np.zeros(len(X_tide)), tide_prior_K, size=3
    )
    for _i in range(_tide_draws.shape[0]):
        tide_prior_fig.add_trace(
            go.Scatter(
                x=tide_hours,
                y=_tide_draws[_i],
                mode="lines",
                line=dict(width=2),
                name=f"prior draw {_i + 1}",
            )
        )
    tide_prior_fig.add_trace(
        go.Scatter(
            x=tide_hours,
            y=y_tide,
            mode="lines",
            line=dict(color="black", width=2, dash="dot"),
            name="observed (standardized)",
        )
    )
    tide_prior_fig.update_layout(
        title="Draws from the composed kernel, before seeing any data",
        xaxis_title="Hours since slice start",
        yaxis_title="Water level (standardized)",
        template="plotly_white",
        legend=dict(orientation="h", y=1.02, yanchor="bottom"),
    )
    tide_prior_fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    No data went into those curves. They come from the covariance function
    alone, and they already show the thing that makes this a mixed
    semidiurnal station: two high tides a day of unequal height, drifting
    slowly up and down. Composition is what bought that. A single kernel
    cannot produce it.

    The amplitudes above were set by hand to make the point. Which
    component actually dominates, and by how much, is a question for the
    data, and answering it means fitting the model. Notebook 3 does that
    with `pm.gp.Marginal`, using this same composed kernel.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Exercise: compose a kernel for a described pattern

    Suppose you have hourly foot-traffic counts at a retail store, and
    you're told: "traffic has a strong repeating **daily** pattern
    (open/close hours), a weaker repeating **weekly** pattern (weekends
    differ from weekdays), and a slow **seasonal** drift on top."

    Two questions, both worth writing out before you expand the solution:

    1. Using only `Matern52` and `Periodic`, write the additive kernel,
       one term per described structure.
    2. If the standardized input is
       $z=(\mathrm{hour}-\overline{\mathrm{hour}})/s_{\mathrm{hour}}$,
       what `period` values do the daily and weekly terms use? Express
       them in terms of $s_{\mathrm{hour}}$, the input standard deviation
       measured in hours. A literal `period=24` is wrong because the
       kernel uses standardized input units.
    """)
    return


@app.cell
def _():
    @exercise
    def exercise_retail_kernel():
        # Compose the seasonal, daily, and weekly covariance terms, then
        # return a concise useful result explaining standardized periods.
        ...


    exercise_retail_kernel()
    return


@app.cell(hide_code=True)
def _():
    def solution_retail_kernel(input_std_hours):
        eta_season, eta_daily, eta_weekly = 1.0, 2.0, 0.8
        ell_season, ell_daily, ell_weekly = 4.0, 0.5, 0.8
        daily_period = 24 / input_std_hours
        weekly_period = 24 * 7 / input_std_hours
        covariance = (
            eta_season**2 * pm.gp.cov.Matern52(1, ls=ell_season)
            + eta_daily**2
            * pm.gp.cov.Periodic(1, period=daily_period, ls=ell_daily)
            + eta_weekly**2
            * pm.gp.cov.Periodic(1, period=weekly_period, ls=ell_weekly)
        )
        return covariance, daily_period, weekly_period


    # Illustrative input scale: periods are expressed as 24 / s_hour and
    # 168 / s_hour for whichever standardized input is actually fitted.
    solution_covariance, daily_period, weekly_period = solution_retail_kernel(input_std_hours=24.0)
    solution_retail_kernel_table = mo.as_html(
        pl.DataFrame(
            {
                "component": ["seasonal drift", "daily cycle", "weekly cycle"],
                "kernel": ["Matern52", "Periodic", "Periodic"],
                "standardized period": [None, daily_period, weekly_period],
            }
        )
    )


    mo.accordion(
        {
            "Solution": mo.vstack(
                [
                    mo.md(f"```python\n{inspect.getsource(solution_retail_kernel)}\n```"),
                    mo.md(
                        "Add the slow Matern52 drift to daily and weekly Periodic "
                        "terms. If the input standard deviation is "
                        "$s_{\\mathrm{hour}}$ hours, use periods "
                        "$24/s_{\\mathrm{hour}}$ and $168/s_{\\mathrm{hour}}$."
                    ),
                    solution_retail_kernel_table,
                ]
            )
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## GP regression *is* conditioning

    Everything so far has been prior: kernels, the curves they imply, and
    how those curves change as you turn the hyperparameters. Now we
    condition on data, which is the whole of GP regression and the last
    thing to see before notebook 3 starts fitting models.

    It is the MVN **conditioning** from the start of this notebook,
    applied to function values instead of two scalars. Stack the (noisy)
    observed outputs $\mathbf y$ at training inputs $X$ together with the
    unknown function values $\mathbf f_*$ at test inputs $X_*$; under the
    GP prior they are *jointly* MVN. Conditioning $\mathbf f_*$ on
    $\mathbf y$ gives the posterior

    $$\mathbf f_* \mid \mathbf y \sim \mathcal N\big(
    K_{*}(K + \sigma^2 I)^{-1}\mathbf y,\ \
    K_{**} - K_{*}(K + \sigma^2 I)^{-1}K_{*}^{\top}\big),$$

    where $K = k(X,X)$, $K_* = k(X_*,X)$, $K_{**} = k(X_*,X_*)$, and
    $\sigma^2$ is observation-noise variance added to the training block.
    Set those matrices to $1 \times 1$ scalars and you get back the
    $x_1 \mid x_2$ formula from the bivariate example exactly.

    We do it by hand on the theophylline subject that defeated notebook
    1's piecewise model.

    One choice deserves stating before we make it. The ExpQuad kernel is
    **stationary**: a single $\ell$ sets the wiggle rate everywhere on the
    input axis. Theophylline does not oblige, it rises within an hour and
    then decays for a day, and it is sampled that way too, six points
    before the fourth hour and then a twelve-hour gap. Ask one lengthscale
    to serve both regimes on raw time and it must pick a side: long enough
    to bridge the gap and it flattens the absorption limb, short enough to
    catch the limb and it forgets everything mid-gap and snaps back to the
    prior mean. Conditioning on **log-time** puts the fast and slow parts
    of the curve on comparable footing, so one lengthscale suffices. The
    kernel's assumptions are claims about the axis you feed it, not just
    about the function.
    """)
    return


@app.cell(hide_code=True)
def _():
    theoph = pl.read_csv(data_dir / "theophylline.csv")
    theoph.head()
    return (theoph,)


@app.cell(hide_code=True)
def _(theoph):
    # One subject, sorted by time, reused for the GP conditioning example below.
    subject_id = 1
    subject_df = theoph.filter(pl.col("subject") == subject_id).sort("time")

    time_vals = subject_df["time"].to_numpy()
    conc_vals = subject_df["conc"].to_numpy()

    conc_mean, conc_std = conc_vals.mean(), conc_vals.std(ddof=0)

    conc_z = z(conc_vals)
    return conc_mean, conc_std, conc_vals, conc_z, time_vals


@app.cell(hide_code=True)
def _(time_vals):
    time_grid = np.linspace(time_vals.min(), time_vals.max(), 200)
    return (time_grid,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We can now do this on the theophylline subject from notebook 1, with
    no MCMC and no PyMC model: just the conditioning formula applied to
    kernel matrices. Nothing is being *fitted* here. The formula needs
    $\ell$, $\eta$ and $\sigma$ before it can be evaluated, and we simply
    choose them:

    - $\ell = 0.6$ on an input axis about 3.2 units wide, so the curve
      can bend a handful of times across the day but not between adjacent
      observations;
    - $\eta = 1.0$, matching the standardized concentrations, whose
      standard deviation is 1 by construction;
    - $\sigma = 0.25$, saying assay noise accounts for about a quarter of
      the variation.

    The input axis is standardized log-time rather than raw time, for the
    reason just given.
    """)
    return


@app.cell
def _(conc_mean, conc_std, conc_z, time_grid, time_vals):
    cond_ls, cond_eta, cond_noise = 0.6, 1.0, 0.25
    cond_log_time = np.log1p(time_vals)
    cond_log_time_mean, cond_log_time_std = cond_log_time.mean(), cond_log_time.std(ddof=0)
    Xtr = (cond_log_time - cond_log_time_mean) / cond_log_time_std
    ytr = conc_z
    Xstar = (np.log1p(time_grid) - cond_log_time_mean) / cond_log_time_std

    cond_cov = cond_eta**2 * pm.gp.cov.ExpQuad(1, cond_ls)
    K_tr = cond_cov(Xtr[:, None]).eval() + cond_noise**2 * np.eye(len(Xtr))
    K_s = cond_cov(Xstar[:, None], Xtr[:, None]).eval()
    K_ss = cond_cov(Xstar[:, None]).eval()

    cond_solve = np.linalg.solve(K_tr, ytr)
    post_mean_z = K_s @ cond_solve
    post_cov = K_ss - K_s @ np.linalg.solve(K_tr, K_s.T)
    posterior_draws_z = np.random.default_rng(RANDOM_SEED).multivariate_normal(
        post_mean_z, post_cov + 1e-8 * np.eye(len(post_mean_z)), size=2_000
    )
    posterior_draws = xr.DataArray(
        posterior_draws_z[None, :, :],
        dims=("chain", "draw", "time_grid"),
        coords={"chain": [0], "draw": np.arange(len(posterior_draws_z)), "time_grid": time_grid},
    )
    gp_post_mean = post_mean_z * conc_std + conc_mean
    gp_post_lo, gp_post_hi = (
        endpoint.values for endpoint in eti_bounds(posterior_draws * conc_std + conc_mean)
    )
    return gp_post_hi, gp_post_lo, gp_post_mean


@app.cell(hide_code=True)
def _(conc_vals, gp_post_hi, gp_post_lo, gp_post_mean, time_grid, time_vals):
    gp_reg_fig = go.Figure()
    gp_reg_fig.add_trace(
        go.Scatter(
            x=np.concatenate([time_grid, time_grid[::-1]]),
            y=np.concatenate([gp_post_hi, gp_post_lo[::-1]]),
            fill="toself",
            fillcolor="rgba(21,74,114,0.25)",
            line=dict(color="rgba(255,255,255,0)"),
            name="89% ETI",
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
def _():
    mo.vstack(
        [
            mo.md(
                r"""
                **Condition on an added point.** Imagine
                appending one extra, very precise observation at the peak (say,
                1.5 hours). Using the conditioning formula
                $\mathbf f_*\mid\mathbf y \sim \mathcal N(K_*(K+\sigma^2
                I)^{-1}\mathbf y,\ K_{**}-K_*(K+\sigma^2 I)^{-1}K_*^\top)$,
                predict the local posterior mean and variance changes before
                checking the discussion.
                """),
            mo.accordion(
                {
                    "Discussion": mo.md(
                        r"""
                        The added row and column make $K_*$ large for prediction
                        points near 1.5 hours. The posterior mean is pulled
                        toward the new observation, while the subtracted
                        covariance term grows locally, so posterior variance
                        shrinks. A smaller measurement-noise $\sigma$ produces
                        more shrinkage. Beyond roughly a lengthscale from the
                        added point, correlation is weak and the conditional
                        distribution changes little. This is the same local
                        information update seen in the bivariate-normal
                        conditioning example.
                        """
                    )
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Compare this directly with the piecewise-linear fit from notebook 1. The
    GP posterior mean glides **smoothly** through the rise, rounds the
    peak without any corner, and eases down the decay, no knot to place,
    no $\tau$ to argue over. The shaded 89% ETI is
    narrow where data are dense (the early rise) and widens where data are
    sparse (the long tail and beyond the last point), which is precisely
    uncertainty behavior we wanted and the piecewise model could not
    provide. We obtained all of it by conditioning a Gaussian, the same
    operation as the two-scalar example, scaled up to a whole function.
    **This is the flexible function the piecewise-linear model could not
    be.**

    So why sample anything, if a closed-form solution produces this?
    Because the band answers a narrower question than it appears to. It
    is $p(\mathbf f_* \mid \mathbf y)$ *for the three numbers we picked*,
    and we picked them by eye. Set $\ell$ to 0.3 instead and you get a
    different curve with an equally confident band, and nothing in the
    formula tells you which one to believe. An honest answer has to
    average over the hyperparameter values the data actually support, and
    that posterior has no closed form. Sampling is how we get it.

    The closed form also depends on the likelihood being Gaussian. Counts
    or binary outcomes break it outright, and then even the function
    values need sampling.

    Notebook 3 does both: priors on $\ell$, $\eta$ and $\sigma$ with
    `pm.gp.Marginal` for the Gaussian case, and `pm.gp.Latent` when the
    likelihood is not.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.vstack(
        [
            mo.md(
                r"""
                **Smoothness and prior implications.** In the
                hand-built GP regression above we set `cond_ls = 0.6` on the
                *standardized log-time* axis. Before changing it to 0.3, predict
                how the prior functions, the fit near the sparse tail, and the
                89% conditional interval should change.
                """),
            mo.accordion(
                {
                    "Discussion": mo.md(
                        r"""
                        Halving the lengthscale makes prior functions wigglier:
                        correlation now decays over a shorter distance. The
                        conditional mean can bend more between observations,
                        potentially chasing the densely sampled early rise. In
                        the sparse tail, prediction points lose correlation
                        with training points sooner, so the mean returns to the
                        zero prior mean faster and the 89% conditional interval
                        expands earlier. A longer lengthscale carries structure
                        farther but can oversmooth the rise. This prior
                        implication is why notebook 3 infers `ell` rather than
                        treating one hand-set value as established.
                        """
                    )
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Where we are, and what's next

    This notebook built the Gaussian process from the multivariate
    normal: marginalization and conditioning, the exponentiated quadratic
    kernel, a gallery of seven off-the-shelf kernels, composition by
    product and sum, and a composed kernel whose prior draws already look
    like tides. It closed by conditioning a GP on data, which is GP
    regression, done by hand with the hyperparameters set by eye.

    Nothing here was fitted. **Notebook 3** does the fitting: it puts
    priors on $\ell$, $\eta$ and $\sigma$ and samples them with
    `pm.gp.Marginal`, on the theophylline subject above and on the tide
    data, then moves to the non-conjugate case, Poisson counts, with
    `pm.gp.Latent`.
    """)
    return


if __name__ == "__main__":
    app.run()
