# Gaussian Processes with PyMC Course Materials Design

**Status:** Approved design, 2026-07-09  
**Audience:** JSM 2026 continuing-education attendees familiar with basic regression and the scientific Python stack, but new to PyMC, Bayesian analysis, and Gaussian processes.

## Goal

Build four hour-aligned, interactive marimo notebooks for the four-hour course in `README.md`. Learners will specify, fit, diagnose, and predict with current PyMC Gaussian-process models using only observed real data.

## Decisions and constraints

- The learner-facing course consists of four instructional notebooks and one non-instructional pre-course environment check.
- Attendees complete a one-time `pixi install` before the course. Every notebook uses the same root, pinned CPython environment; no notebook creates or downloads its own environment.
- Core notebooks run PyMC/PyTensor locally on CPython. Browser-only WASM/Pyodide is not a supported path for fitting, sampling, prediction, or diagnostics.
- PEP 723/sandbox metadata is not part of the core course delivery. It may support a future server-backed fallback, but it does not make PyMC/PyTensor browser compatible.
- Every fitted model, prior-predictive check, posterior-predictive check, and diagnostic uses vendored observed real data. Simulated draws from a GP prior are permitted only to visualize kernel assumptions before fitting; they are never presented as data or evidence of model fit.
- All selected datasets ship in this repository with provenance and licence attribution. No learner notebook fetches data at runtime.
- PyMC 6 and ArviZ 1.0 `DataTree` conventions are mandatory: group access uses `idata["posterior"]`, `idata["sample_stats"]`, and `idata["posterior_predictive"]`, never legacy attribute access.
- Sampling is explicitly triggered, deterministically seeded, and never concealed behind warning suppression, a broad exception handler, or a fake result cache.
- Learner-facing diagnostics display a prior-predictive check, posterior-predictive check, divergence count, rank-normalized R-hat, and bulk/tail ESS where MCMC is performed.
- Course examples standardize continuous predictor and response variables before applying weakly informative teaching priors. Each prior is accepted only after a prior-predictive plot shows plausible scale and shape for the observed-data domain.

## Non-goals

- Browser-only execution of PyMC, PyTensor, NUTS, or HSGP.
- Synthetic fitted datasets, mock posterior results, or network-dependent data acquisition.
- Converting legacy Jupyter/PyMC3/PyMC 5 notebooks cell-for-cell.
- A general-purpose GP package, automated kernel search, LOO model-comparison curriculum, multi-output/coregionalization curriculum, or a full advanced hierarchical-GP fit.
- Assumed baseball knowledge. Baseball is used only as a compact grouped repeated-measurement dataset.

## Repository architecture

```text
jsm_2026_gp_pymc/
├── pixi.toml
├── pixi.lock
├── data/
│   ├── README.md                 # provenance, licence, schema, transformations
│   ├── salmon.txt
│   ├── coal_disasters.csv
│   ├── gb_electricity_demand.csv
│   ├── walker_lake.csv
│   └── fastball_spin_rates.csv
├── notebooks/
│   ├── 00_environment_check.py
│   ├── 01_foundations.py
│   ├── 02_exact_and_latent_gps.py
│   ├── 03_kernels_inputs_and_hierarchy.py
│   └── 04_workflow_and_hsgp.py
└── tests/
    ├── test_data_contracts.py
    ├── test_model_contracts.py
    └── test_notebook_smoke.py
```

`00_environment_check.py` verifies the resolved package versions, each vendored asset, and a minimal PyMC model compilation. It does not sample. A failure names the missing or incompatible package, version, or asset and stops before class begins.

The four instructional notebooks share files and the locked environment, but never share marimo reactive state. Each can be opened and taught independently after the environment check has passed.

## Curriculum sequence

| Notebook | Time budget | Real data | Learning contract | Primary exercise |
| --- | ---: | --- | --- | --- |
| `01_foundations.py` | 50 minutes | Fraser River salmon annual `spawners` and `recruits` | Specify a Normal regression in `pm.Model`; connect flexible residual structure to GP mean and covariance functions; interpret amplitude and lengthscale through prior draws. | Complete a covariance expression and predict its visible effect before checking the low-cost interactive prior plot. |
| `02_exact_and_latent_gps.py` | 55 minutes | Salmon continuous response; annual coal-disaster counts | Fit a Gaussian-response `pm.gp.Marginal` model and contrast it with a Poisson `pm.gp.Latent` model using a log link and `gp.prior(...)`. | Complete a likelihood/link expression and decide whether posterior-predictive count dispersion is credible. |
| `03_kernels_inputs_and_hierarchy.py` | 55 minutes | GB daily electricity demand; Walker Lake spatial concentration; a compact subset of pitcher game-level spin rates | Construct additive trend-plus-periodic covariance structure; use a two-column input matrix, `input_dim`, vector lengthscales, and `active_dims`; define a shared trajectory plus group deviations for partial pooling. | Select kernel algebra for an observed pattern; complete a two-dimensional input specification; complete a group coordinate/index line and inspect the resulting model structure. |
| `04_workflow_and_hsgp.py` | 60 minutes | Full GB daily electricity-demand series; a small observed slice of that same series | Explain exact-GP scaling, distinguish inducing points from HSGP, fit an HSGP with declared `m` and `c`, assess boundary/approximation effects, and complete the diagnostic/prediction workflow. | Choose an `m, c` configuration for a stated observed domain and judge whether diagnostics and approximation checks support its use. |

The selected daily electricity-demand file is an existing local candidate. Its original source and licence are a release gate. If that source cannot be established, replace it with a properly licensed observed daily demand series while preserving its 1-D continuous, long-run-plus-periodic contract.

## Model contracts

These contracts define the models learners construct. They are course examples, not a generalized data service.

### Common conventions

- Continuous predictors and continuous outcomes are centered and scaled with transformations documented beside each data loader. Plots also show values in original units.
- A one-dimensional input is always shaped `(n, 1)`; a two-dimensional input is `(n, 2)`.
- `pm.Data` holds input locations and index vectors whenever prediction swaps locations or a grid length changes.
- Priors use `Normal(0, 1)` for standardized location/effect terms, `HalfNormal(1)` for standardized positive amplitudes/noise scales, and `LogNormal(0, 1)` for positive standardized lengthscales unless an observed-domain prior-predictive check establishes a narrower scale. The notebook states the scale and shows that check before fitting.
- A curated analysis subset must contain at least 20 observed rows, and every displayed group must have at least 10 observed rows. A failure to meet either condition stops the fit and states that the exercise is not estimating a curve from insufficient observations.
- Inference is descriptive/predictive within the observed covariate range. No notebook interprets extrapolated curves as causal effects.

### 1. Salmon primer and exact Marginal GP

**Purpose:** show the transition from a finite-parameter Normal regression to a nonparametric mean function for annual salmon recruitment.

**Observation unit:** one year; observed input is annual spawners and observed outcome is annual recruits.

**Primer model:**

\[
y_i^* \sim \operatorname{Normal}(\alpha + \beta x_i^*, \sigma), \quad
\alpha, \beta \sim \operatorname{Normal}(0, 1), \quad
\sigma \sim \operatorname{HalfNormal}(1).
\]

**Exact GP model:**

\[
f \sim \operatorname{GP}\!\left(\alpha,\; \eta^2 k_{\mathrm{Matern52}}(x,x';\ell)\right),
\qquad
y_i^* \mid f_i,\sigma \sim \operatorname{Normal}(f_i,\sigma),
\]

with \(\alpha \sim \operatorname{Normal}(0,1)\), \(\eta,\sigma \sim \operatorname{HalfNormal}(1)\), and \(\ell \sim \operatorname{LogNormal}(0,1)\). The implementation uses `pm.gp.Marginal(...).marginal_likelihood(...)`, so the GP function values are analytically marginalized. The fitting input is deliberately small, making exact inference practical.

**Outputs:** observed-data scatterplot, posterior function draws at a prediction grid, an original-unit posterior summary, and diagnostic objects in a `DataTree`.

### 2. Coal-disaster latent Poisson GP

**Purpose:** show why a non-Gaussian response retains latent function values rather than marginalizing them away.

**Observation unit:** one calendar year; observed outcome is the historical number of reported coal disasters.

\[
f \sim \operatorname{GP}\!\left(\alpha,\; \eta^2 k_{\mathrm{Matern52}}(t,t';\ell)\right),
\qquad
y_i \sim \operatorname{Poisson}(\exp(f_i)),
\]

with \(\alpha \sim \operatorname{Normal}(0,1.5)\), \(\eta \sim \operatorname{HalfNormal}(1)\), and \(\ell \sim \operatorname{LogNormal}(0,1)\) after time is scaled. The implementation calls `f = gp.prior("f", X=t)` followed by the Poisson observation model. The model uses a deliberately short observed annual series because every input creates a latent value.

**Outputs:** posterior rate trajectories, posterior-predictive annual count distributions, a count-dispersion comparison, and MCMC diagnostics.

### 3. Kernel composition and two-dimensional GP

**Purpose:** connect covariance algebra to observed patterns rather than treating kernel menus as automatic model selection.

**Electricity observation unit:** one day; observed outcome is daily demand. A continuous-response GP decomposes the latent function into a long-timescale component plus a periodic component:

\[
f(t) = f_{\mathrm{long}}(t) + f_{\mathrm{periodic}}(t),
\qquad y_t^* \sim \operatorname{Normal}(f(t), \sigma).
\]

The model uses separately scaled amplitude and lengthscale parameters with the common standardized priors. Learners construct the covariance sum only after seeing component prior draws.

**Walker Lake observation unit:** one sampled geographic location; observed outcome is concentration. The input is \(X_i=(x_i,y_i)\), the likelihood is Normal after outcome standardization, and the covariance receives `input_dim=2` with a two-element lengthscale. The exercise makes the two input columns and their units explicit; it does not silently reuse a one-dimensional covariance.

### 4. Hierarchical grouped trajectories

**Purpose:** explain partial pooling through a compact observed grouped time series, without pretending that a multi-output coregionalization model is an introductory hierarchy.

**Observation unit:** one pitcher-game average spin rate. The model uses three named pitchers and game date as time:

\[
y_{gi}^* \sim \operatorname{Normal}(\mu(t_{gi}) + \delta_g(t_{gi}), \sigma),
\]

where \(\mu\) is a shared GP trajectory and \(\delta_g\) is a zero-mean group deviation process. The group-deviation amplitude is positively regularized with a `HalfNormal(1)` prior so groups are pulled toward the shared trajectory when their data are weak. The notebook uses named `group`, `obs`, and `time` coordinates plus an observed group index. It focuses on structure, model validation, and prior implications rather than a long full posterior fit.

**Outputs:** data by group, model graph/variable structure, prior trajectories that make pooling visible, and a plain-language partial-pooling explanation.

### 5. HSGP workflow

**Purpose:** show a scalable approximation on the full observed electricity series, after learners understand the exact one-dimensional case.

**Observation unit:** one day; observed outcome is standardized daily load. The model is a stationary Matérn latent function with Normal observation noise, represented by `pm.gp.HSGP` rather than an exact GP:

\[
f \approx \operatorname{HSGP}\!\left(k_{\mathrm{Matern52}}, m, c\right),
\qquad y_t^* \sim \operatorname{Normal}(f(t), \sigma).
\]

`m` is shown as the retained basis resolution and `c` as the domain extension; exactly one of `L` and `c` is supplied. The notebook explains the stationary-kernel and practical low-dimensional constraints, uses `c >= 1.2`, and requires learners to assess boundary behavior. An exact GP is fitted only to a small observed slice for a conceptual cost and prediction comparison. The full series is fit only through click-gated HSGP cells.

**Outputs:** declared domain and basis settings, HSGP posterior prediction in original units, prior/posterior predictive plots, divergence/R-hat/ESS display, and an explicit approximation decision.

## Interaction design

Each notebook follows the same loop:

1. **Observe** the observed data and name a modeling assumption.
2. **Predict** an effect with a low-cost widget or written prompt.
3. **Do** one isolated code-completion task.
4. **Inspect** a deterministic structural check or an explicitly triggered fit.

Kernel widgets update covariance and prior visualizations only; no widget automatically calls MCMC. Code-completion tasks are isolated from the canonical lesson DAG, so an incomplete learner answer cannot prevent later instructional cells from running. Each task has a collapsed adjacent solution and a deterministic checker for an observable contract such as input dimension, model variable, likelihood support, coordinate shape, or prediction-group presence.

## Error handling and compute behavior

- Data loaders validate documented column names, types, required non-missing values, and the curated subset size. They report the offending contract instead of replacing data or continuing with a different source.
- The environment check performs no automatic installation. It reports a missing package, incompatible version, or missing asset by name.
- MCMC cells appear only behind `mo.ui.run_button()` and `mo.stop(not run_button.value)`. They record the seed and expected runtime.
- Sampling warnings, divergences, and convergence failures are visible instructional output. There are no global warning filters, blanket `try/except` blocks, fake posterior results, or silent caches.
- Optional future remote hosting must use a pinned CPython/server runtime and a deployment smoke test. It must not publish the core notebooks as WASM or a Pyodide preview.

## Verification and release criteria

1. **Data contracts:** tests verify each vendored asset’s schema, documented transformations, expected observation count/range, and provenance record. They fail if a notebook includes a runtime HTTP data path.
2. **Model contracts:** fast tests build every model on a deterministic observed-data slice, call model validation, and verify coordinates, named variables, likelihood support, and prediction interfaces. `pymc.testing.mock_sample` is allowed only for downstream structure/rendering checks.
3. **Real-sampling smoke tests:** focused tests sample small fixed observed slices and assert the returned `DataTree` has the expected posterior, sample-stat, and posterior-predictive groups. Their low draw count proves execution only; it does not support a convergence claim.
4. **Notebook checks:** all notebooks pass `pixi run marimo check --strict`. Script-mode smoke tests cover the environment check and non-sampling paths.
5. **Locked-environment rehearsal:** before publication, the instructor runs every click-gated course fit in the exact locked target environment and records the measured runtime. A slow or failed fit blocks release until its observed-data subset or model budget is adjusted without changing the curriculum contract.
6. **Course acceptance:** a new attendee can complete the environment check, run all non-sampling material, execute each exercise’s checker, and obtain visible diagnostics from the core models without network access after the one-time bootstrap.

## Source adaptation decisions

| Source | Reuse | Exclusion or migration requirement |
| --- | --- | --- |
| `~/Labs/london-workshop/notebooks/Session_7-Gaussian_Processes.py` | Marimo kernel interactions; GP concepts; embedded coal-count story; HSGP narrative and parameter explanation. | Rebuild every PyMC model for PyMC 6/DataTree; retain no global warning suppression, forced sampler selection, matrix inversion demo helper, or attribute-style posterior access. |
| `../gp_regression/notebooks/spawning_salmon.ipynb` and `../gp_regression/data/salmon.txt` | Salmon data story and the parametric-to-GP teaching transition. | Do not convert PyMC3 code, IPython magics, `MultiTrace`, legacy posterior-predictive calls, or its legacy `noise=` argument. |
| `../instats_gp/sessions/Session_2.ipynb` and `Session_3.ipynb` | Kernel/HSGP theory and instructor reference. | Do not copy synthetic fitted exercises, LLM-dependent exercise placeholders, network `pm.get_data` paths, pre-DataTree handling, or legacy interval claims. |
| `../instats_gp/sessions/Session_4B.ipynb` and London spin-rate data | Hierarchical-HSGP structural reference and a compact real grouped data source. | Do not copy the advanced 27-player factor model or add PreliZ as a learner dependency. |

## Evidence

- Course contract: `README.md:24-49`.
- Local source-material audit: `docs/research/2026-07-09-pymc-gp-course-sources.md`.
- First-party API and platform evidence is cited in that research note, including PyMC 6 GP APIs, HSGP constraints, predictive/DataTree handling, marimo reactivity and validation, and the Pyodide package catalogue.
