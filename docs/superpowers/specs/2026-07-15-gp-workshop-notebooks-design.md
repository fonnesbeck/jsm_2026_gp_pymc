# JSM 2026 GP Workshop Notebooks — Design

**Status:** Approved design, 2026-07-15
**Course:** *Nonparametric Bayesian Modeling: An Introduction to Gaussian Processes with PyMC* (JSM 2026 half-day continuing education, 4 hours)
**Audience:** Attendees comfortable with basic statistical modeling (linear regression) and the scientific Python stack (NumPy, pandas, Jupyter), but new to PyMC, Bayesian inference, and Gaussian processes.

## Goal

Produce four hour-aligned, interactive **marimo** notebooks that map 1:1 onto the `README.md` course outline. By the end, a learner can specify, fit, diagnose, and predict with current PyMC Gaussian-process models, and knows when to reach for exact, sparse, or HSGP inference. Every model is fit **live during the workshop**, so each is sized to finish within roughly two minutes on a 4-core laptop.

## Decisions and constraints

These were settled during brainstorming and are binding on the implementation plan.

- **Fresh start.** The prior 2026-07-09 design/spec/plan are discarded; this document supersedes them.
- **Format:** four marimo notebooks (`.py`), one per hour of the outline. No Jupyter deliverable is required.
- **Target stack:** latest PyMC 6.x with ArviZ 1.x DataTree conventions, `nuts_sampler="nutpie"`, **polars** for data, **plotly** for plots, marimo for notebooks and interactivity.
- **Live-execution budget.** Every sampled model must complete in ≤ ~2 minutes on a 4-core, 16 GB, GPU-free laptop. Datasets are subset/curated to respect this. Sampling is explicitly triggered and deterministically seeded; it is never hidden behind warning suppression or a result cache.
- **Vendored data, no runtime fetching.** Conference wifi is unreliable, so no notebook downloads anything at runtime. All datasets ship as frozen CSVs in `data/`, rebuilt by a committed `build_data.py` and documented in `data/README.md` (provenance, license, transformations).
- **Dataset background in every notebook.** Each notebook introduces its dataset with a short background cell: what it is, where it comes from, why it makes a particular GP modeling choice visible, its units, and any caveats.
- **PyMC 6 / ArviZ DataTree idioms are mandatory:** group access uses `idata["posterior"]`, `idata["sample_stats"]`, `idata["posterior_predictive"]`, never legacy attribute access.
- **Pedagogy:** short per-section "your turn" exercises with solutions hidden in `mo.accordion` blocks, plus `mo.ui` sliders driving reactive prior-draw plots (kernel hyperparameters → GP prior samples).
- **General-audience-first datasets** with one sports example (baseball spin rates) reserved for the grouped/hierarchical case.

## Non-goals

- Jupyter (`.ipynb`) deliverables, or cell-for-cell porting of the legacy PyMC3 `gp_regression` notebooks.
- Any model that cannot be fit live within the time budget.
- Runtime data downloading in learner notebooks.
- A general-purpose GP package, automated kernel search, multi-output/coregionalization curriculum, LOO model-comparison curriculum, or GPU/GPyTorch material.
- Assumed sports domain knowledge; baseball is used only as a compact grouped repeated-measurements dataset.

## Source materials

The notebooks curate and modernize existing teaching material (approach A from brainstorming):

- **`~/repos/instats_gp/sessions/`** — the backbone. Fonnesbeck's own current PyMC 5.26+ / Py 3.13 GP course (polars, plotly, nutpie, exercise-rich). Sessions 1–4 cover the PyMC primer, GP foundations, kernels/likelihoods, marginal/latent GPs, additive/multiplicative kernels, and HSGP/sparse scaling. `additive_hsgp_section.md` is a polished drop-in on additive HSGP covariance.
- **`~/Labs/london-workshop/notebooks/Session_7-Gaussian_Processes.py`** — already marimo; strongest HSGP *intuition* material (basis functions, `m`/`c` knobs, centered vs non-centered) and the computational-wall framing.
- **`~/repos/gp_regression/notebooks/`** — legacy PyMC3, not runnable as-is. Harvest narrative (the "parametric fits fail first" arc) and the small teaching datasets only; port no code.

## Dataset plan

Six datasets. Three replace the original instats choices the instructor found uninteresting (salmon, Mauna Loa, Walker Lake); coal disasters and baseball spin rates are retained.

| Role | Dataset | Source | Size (as used) | Why it teaches the concept |
| --- | --- | --- | --- | --- |
| Primer + marginal GP | **Theophylline pharmacokinetics** | R `Theoph` built-in | 132 obs / 12 subjects; single-subject fits use ~11 pts | Rise-peak-decay curve makes a naive stationary GP on raw time visibly struggle; flows naturally into hierarchical pooling. |
| Latent / count GP | **Coal-mining disasters** | Jarrett 1851–1962 annual counts (vendored) | ~110 annual counts | Classic non-Gaussian (Poisson) response that cannot be marginalized; motivates the latent parameterization. |
| Kernel composition + HSGP scaling | **NOAA tide-gauge water levels** | NOAA CO-OPS `datagetter` API | ~200–300-point slice for the exact fit; full ~1-year hourly series for HSGP | Two clear periodicities (semidiurnal + diurnal) plus a slow baseline — a richer additive/multiplicative kernel story than a single annual cycle. Gaussian, so it stays a `gp.Marginal` example. |
| 2D spatial GP | **CDC PLACES county diabetes prevalence** | data.cdc.gov (Socrata) | one state, ~100–254 counties | 2D spatial Matérn with ARD/anisotropy, covariates, posterior maps. Caveat that PLACES values are model-based small-area estimates is itself a useful discussion. |
| Hierarchical / grouped GP | **Baseball fastball spin rates** | instats_gp `fastball_spin_rates.csv` | 3 pitchers × ~10 games = ~30 rows | Grouped repeated measurements; shared trend + per-group latent GP shows partial pooling. |

### Data sourcing and provenance

- `data/build_data.py` fetches each dataset from its authoritative source **once** and writes a frozen CSV. It is run by the maintainer, not by learners.
- `data/README.md` records for each file: source URL, access date, license/terms, original units, and every filtering/transformation step applied.
- Standardization for modeling uses `z(a) = (a - a.mean()) / a.std(ddof=0)`; original-unit values are retained for plots.
- NOAA CO-OPS: `hourly_height` product, single named station, fixed datum (MLLW) and date window recorded in the build script.
- CDC PLACES: county-level diagnosed-diabetes measure, single release year, single state, joined to county lon/lat centroids.

## Repository architecture

```text
jsm_2026_gp_pymc/
├── README.md                        # course proposal (existing)
├── pixi.toml / pixi.lock            # pinned environment, 4 platforms
├── data/
│   ├── README.md                    # provenance, license, transformations
│   ├── build_data.py                # rebuilds every CSV from source
│   ├── theophylline.csv
│   ├── coal_disasters.csv
│   ├── noaa_tides_hourly.csv
│   ├── places_diabetes.csv
│   └── fastball_spin_rates.csv
└── notebooks/
    ├── 00_environment_check.py      # non-instructional: verify env + data
    ├── 01_foundations.py
    ├── 02_marginal_latent_gps.py
    ├── 03_kernels_and_hierarchy.py
    └── 04_scaling_and_workflow.py
```

**Environment:** a pixi workspace (matching instats_gp tooling) pinning latest PyMC 6.x, ArviZ 1.x, nutpie, polars, plotly, and marimo, solving for `win-64`, `linux-64`, `osx-64`, and `osx-arm64` so a single `pixi.lock` covers every attendee. `00_environment_check.py` verifies resolved versions, the presence of each vendored CSV, and a minimal model compilation (no sampling); attendees run it once before class.

The four instructional notebooks share the locked environment and the `data/` files but never share marimo reactive state. Each opens and teaches independently after the environment check passes.

## Curriculum

Approximately 220 minutes of material across the four hours, leaving room for breaks and Q&A. Model sizes noted where relevant to the time budget.

### `01_foundations.py` — Hour 1 (~50 min)

**Part A — Bayesian workflow & PyMC primer.** Introduce the Theophylline dataset (background cell). Teach the Bayesian paradigm through a simple regression in `pm.Model`: priors → likelihood → prior predictive check → `pm.sample` (nutpie) → ArviZ posterior summary. Adapts instats `01_introduction` §1.1–1.2 onto Theophylline so the dataset carries into Hour 2.

**Part B — GP concepts.** Bivariate-Gaussian visual for marginalization and conditioning; GP as an infinite-dimensional Gaussian / distribution over functions; mean and covariance functions; interpreting basic hyperparameters.

- **Exercise:** implement the ExpQuad kernel from scratch (from instats §1.3).
- **Widget:** `mo.ui` lengthscale/amplitude sliders that reactively redraw GP prior samples; learners predict the effect before moving the slider.

### `02_marginal_latent_gps.py` — Hour 2 (~55 min)

**Marginal GPs (conjugate case).** Continue with Theophylline. Show a naive stationary GP on raw time struggling with the rise-peak-decay shape (the teaching hook), motivate an input transform and/or mean function, then fit `pm.gp.Marginal(...).marginal_likelihood(...)` with a Matérn 5/2 kernel. Discuss conjugacy (Gaussian prior + Gaussian likelihood → closed-form Gaussian posterior); contrast MAP optimization with full MCMC; predict with `.conditional`, with and without `pred_noise`.

- **Exercise:** predict beyond the observed time range and diagnose extrapolation behavior.

**Latent GPs (non-conjugate case).** Coal-mining disaster counts (background cell). Explain why a Poisson response retains latent function values instead of marginalizing them; build `f = gp.prior("f", X=t)` + exp link + Poisson likelihood; note NUTS is required. Show the posterior rate trajectory and a posterior-predictive count check.

- **Exercise:** change the lengthscale prior and assess posterior sensitivity.

Both datasets are tiny (≤ ~110 rows), so live sampling is seconds to a minute.

### `03_kernels_and_hierarchy.py` — Hour 3 (~55 min)

**Covariance function deep dive.** Interactive prior-draw gallery comparing ExpQuad against Matérn 1/2, 3/2, 5/2 (roughness), plus Periodic / RatQuad / Linear.

**Kernel combinations.** NOAA tide data (background cell). Additive (trend + periodic) and multiplicative (locally-periodic) composition fit with `gp.Marginal` on a downsampled ~200–300-point slice (kept small so the exact fit meets the live-execution budget); the semidiurnal + diurnal structure motivates combining two periodic components. The full hourly series is deferred to the HSGP treatment in Hour 4.

- **Exercise:** compose a kernel for a described data pattern.

**Multi-dimensional inputs.** CDC PLACES county diabetes prevalence (background cell). 2D spatial GP with ARD (vector) lengthscales; posterior surface as a map/heatmap.

**Hierarchical GPs.** Baseball spin rates (background cell): shared trend + per-pitcher latent GP deviations showing partial pooling across 3 pitchers.

All four models are kept small enough to sample (or evaluate prior structure) live.

### `04_scaling_and_workflow.py` — Hour 4 (~60 min)

**GP limitations and scaling.** Timing demonstration of the O(n³) exact-inference cost as n grows.

**Approximation methods.** Brief inducing-points overview with one `gp.MarginalApprox` (FITC) fit. Then HSGP in depth: basis-function intuition (Session 7's strongest material), what `m` and `c` control (interactive), then `gp.HSGP` (+ `HSGPPeriodic`) on the **full ~1-year hourly NOAA tide series** — trend + periodicity at scale, compared against the Hour-3 subset fit. Discuss constraints (stationary kernels, input dimension ≤ 3).

- **Exercise:** change `m`/`c` and judge the approximation against diagnostics and boundary effects.

**Model workflow.** The full loop on the HSGP model — prior predictive check, divergence count, rank-normalized R-hat, bulk/tail ESS, posterior predictive check — plus a when-to-use-what decision guide (exact / sparse / HSGP).

## Model and code conventions

- One-dimensional GP inputs are shaped `(n, 1)`; two-dimensional inputs are `(n, 2)`.
- `pm.Data` holds input locations / index vectors whenever prediction swaps locations or changes a grid length.
- Default teaching priors: `Normal(0, 1)` for standardized location/effect terms, `HalfNormal(1)` for standardized positive amplitudes/noise scales, `LogNormal(0, 1)` for positive standardized lengthscales — each shown via a prior-predictive plot before fitting and narrowed only when the observed-domain check warrants.
- Where MCMC runs, learner-facing diagnostics display a prior-predictive check, a posterior-predictive check, divergence count, R-hat, and ESS.
- Inference is descriptive/predictive within the observed covariate range; no notebook interprets extrapolated curves as causal effects.

## Verification

- `00_environment_check.py` passes on a clean `pixi install` across the supported platforms.
- Each instructional notebook runs end-to-end (e.g., via `marimo` headless execution) and every sampled model completes within the ~2-minute live budget on the reference laptop.
- `data/build_data.py` reproduces every vendored CSV from source, and `data/README.md` documents provenance for each.
- Diagnostics reported in-notebook show acceptable convergence (no persistent divergences, R-hat ≈ 1.0, adequate ESS) for the taught models.
```
