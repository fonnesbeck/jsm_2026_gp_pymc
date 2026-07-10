# Gaussian Processes with PyMC Course Materials Design

**Status:** Approved design, 2026-07-09  
**Audience:** JSM 2026 continuing-education attendees familiar with basic regression and the scientific Python stack, but new to PyMC, Bayesian analysis, and Gaussian processes.

## Goal

Build four hour-aligned, interactive marimo notebooks for the four-hour course in `README.md`. Learners will specify, fit, diagnose, and predict with current PyMC Gaussian-process models using only observed real data.

## Decisions and constraints

- The learner-facing course consists of four instructional notebooks and one non-instructional pre-course environment check.
- Attendees complete a one-time `pixi install` before the course. Every notebook uses the same root, pinned CPython environment; no notebook creates or downloads its own environment.
- The supported attendee matrix is `win-64`, `linux-64`, `osx-64`, and `osx-arm64`. The Pixi workspace lists all four platforms so its single `pixi.lock` solves every supported target.
- The manifest pins Python `3.13.*`, PyMC `6.1.0`, ArviZ `1.2.0`, and marimo `0.23.13`; `pixi.lock` records the exact transitive package builds for every supported platform. The environment check reports the resolved versions rather than accepting a compatible but unlocked environment.
- The release baseline is a current Windows, macOS, or Linux laptop with four logical CPU cores and 16 GB RAM. No GPU, administrator privilege, compiler, or network access after the one-time bootstrap is required.
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
├── README.md                     # published course contract
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
| `03_kernels_inputs_and_hierarchy.py` | 55 minutes | GB daily electricity demand; Walker Lake spatial concentration; 30 pitcher game-level spin-rate observations | Build three observed-data likelihoods but run no posterior MCMC: an additive exact GP with `active_dims`, a two-dimensional exact GP with vector lengthscales, and a shared-trajectory hierarchical latent GP. | Select kernel algebra for an observed pattern; complete a two-dimensional input specification; complete a group coordinate/index line and inspect prior pooling. |
| `04_workflow_and_hsgp.py` | 60 minutes | Full GB daily electricity-demand series; the fixed 96-row observed comparison subset | Explain exact-GP scaling, distinguish inducing points from HSGP, compare an exact Marginal GP to full-series HSGP, assess boundary/approximation effects, and complete the diagnostic/prediction workflow. | Start from the baseline `m=[50], c=1.5` configuration, then judge a proposed change against visible approximation and diagnostic criteria. |

The four notebooks account for 220 minutes of instruction. The live schedule reserves 10 minutes for an opening/closing course orientation and 10 minutes for a break; `00_environment_check.py` is completed before class.
The selected daily electricity-demand file is an existing local candidate. Its original source and licence are a release gate. If that source cannot be established, replace it with a properly licensed observed daily demand series while preserving its 1-D continuous, long-run-plus-periodic contract.

### Deterministic data curation

All continuous transformations use `z(a) = (a - a.mean()) / a.std(ddof=0)` after the documented filtering and stable sort; original-unit values remain available for plots.

- **Salmon:** parse `year`, `recruits`, and `spawners` from the whitespace file; reject nulls or duplicate years; stable-sort by `year`; retain all 40 rows. Use `X = z(spawners)[:, None]` and `y = z(recruits)`.
- **Coal disasters:** vendor the 111 annual observations for 1851–1961; assert integer counts in `[0, 6]`; use `t = z(year)[:, None]`; leave counts unscaled for the Poisson likelihood.
- **Electricity:** parse daily `date` and `load_mw`; reject nulls, duplicate dates, or gaps; stable-sort by date; retain all 1,826 rows from 2015-01-01 through 2019-12-31. For every small exact-GP comparison, select the same 96 actual rows at `np.rint(np.linspace(0, n - 1, 96)).astype(int)` after validating `n >= 96`; standardize time and demand within that selected slice. The HSGP uses all validated rows with time and demand standardized across the full series.
- **Walker Lake:** skip the eight metadata rows; parse the six whitespace-delimited fields; treat `1E31` as missing in the unused `u_ppm` field; require finite `x_m`, `y_m`, and `v_ppm`; retain IDs 1–195 in stable ID order. Model only `v_ppm` with separately standardized `x_m` and `y_m`.
- **Spin rates:** reject missing pitcher/date/spin rate, duplicate pitcher-date pairs, and games with fewer than 10 pitches. Rank remaining pitchers by row count descending then name ascending; select `Rodriguez, Richard`, `Hearn, Taylor`, and `Buehler, Walker`; retain each pitcher’s first 10 rows after stable date sort. The resulting 30 rows use fixed alphabetical categories `[Buehler, Walker, Hearn, Taylor, Rodriguez, Richard]`, standardized days since 2021-04-01, and standardized spin rate.

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

**Electricity observation unit:** one day; observed outcome is daily demand. The fixed 96-row, time-spanning subset has two input columns: standardized elapsed day and elapsed year measured in annual cycles. The additive Normal-response model is

\[
f \sim \operatorname{GP}\!\left(
0,\;
\eta_{\mathrm{long}}^2 k_{\mathrm{Matern52}}(x_{\mathrm{long}};\ell_{\mathrm{long}})
+ \eta_{\mathrm{year}}^2 k_{\mathrm{Periodic}}(x_{\mathrm{year}};\ell_{\mathrm{year}},1)
\right),
\qquad y_t^* \sim \operatorname{Normal}(f(t), \sigma).
\]

Use `Matern52(input_dim=2, active_dims=[0])` for the long-timescale component and `Periodic(input_dim=2, active_dims=[1], period=1.0)` for the annual component. Both constructors receive the total two-column input dimension; `active_dims` selects their single relevant column. Each amplitude and `sigma` has a `HalfNormal(1)` prior; each lengthscale has a `LogNormal(0, 1)` prior. Implement it with `pm.Data("X", X, dims=("obs", "input"))`, `pm.gp.Marginal(cov_func=cov_long + cov_year)`, and `marginal_likelihood(..., sigma=sigma)`. The notebook visualizes the components and runs `model.debug()` plus a prior-predictive check; it does not sample a posterior in Hour 3.

**Walker Lake observation unit:** one retained sampled geographic location; observed outcome is `v_ppm`. With separately standardized coordinates \(X_i=(x_i^*,y_i^*)\), use

\[
f \sim \operatorname{GP}\!\left(0,\eta^2 k_{\mathrm{Matern52}}(X,X';[\ell_x,\ell_y])\right),
\qquad v_i^* \sim \operatorname{Normal}(f_i,\sigma).
\]

Implement `ell` as a two-element `LogNormal(0, 1)` variable with `spatial_dim=["x_m", "y_m"]`, `cov = eta**2 * pm.gp.cov.Matern52(input_dim=2, ls=ell, active_dims=[0, 1])`, and `pm.gp.Marginal(...).marginal_likelihood(..., sigma=sigma)`. Hour 3 displays the observed map, validates the exact model, and performs a prior-predictive check only.

### 4. Hierarchical grouped trajectories

**Purpose:** explain partial pooling through a compact observed grouped time series, without multi-output coregionalization.

**Observation unit:** one retained pitcher-game average spin rate. Let \(g_i\) be the observed pitcher index and \(t_i^*\) the standardized game date:

\[
m \sim \operatorname{GP}\!\left(0,\eta_m^2k_{\mathrm{Matern52}}(t,t';\ell_m)\right),
\qquad
\delta_g \overset{\mathrm{iid}}{\sim} \operatorname{GP}\!\left(0,\eta_\delta^2k_{\mathrm{Matern52}}(t,t';\ell_\delta)\right),
\qquad
y_{gi}^* \sim \operatorname{Normal}(m(t_{gi})+\delta_g(t_{gi}),\sigma).
\]

Use `HalfNormal(1)` priors for \(\eta_m\), \(\eta_\delta\), and \(\sigma\), plus `LogNormal(0, 1)` priors for \(\ell_m\) and \(\ell_\delta\). The shared \(\eta_\delta\) is the pooling parameter: smaller values pull all pitcher trajectories toward \(m\). The implementation uses `pm.Data` for `t` and `group_idx`, named `group`, `obs`, and `time` coordinates, one shared `pm.gp.Latent(...).prior("m", ...)`, and three independent `pm.gp.Latent` deviation priors that share `cov_delta` parameters. Stack deviations and index them by `group_idx`; do not use `Coregion`, factor loadings, or an advanced HSGP hierarchy.

Hour 3 constructs the real-data likelihood and runs prior predictive simulation only. It does not run the latent posterior, whose 120 function values would displace the required Hour 4 workflow.

**Outputs:** data by group, model graph/variable structure, prior trajectories that make pooling visible, and a plain-language partial-pooling explanation.

### 5. HSGP workflow

**Purpose:** show a scalable approximation on the full observed electricity series, after learners understand the exact one-dimensional case.

**Observation unit:** one day; observed outcome is standardized daily load. The full 1,826-row model is a stationary Matérn latent function with Normal observation noise, represented by `pm.gp.HSGP` rather than an exact GP:

\[
f \approx \operatorname{HSGP}\!\left(k_{\mathrm{Matern52}}, m=[50], c=1.5\right),
\qquad y_t^* \sim \operatorname{Normal}(f(t), \sigma).
\]

The baseline implementation uses `eta, sigma ~ HalfNormal(1)`, `ell ~ LogNormal(0, 1)`, `cov = eta**2 * Matern52(1, ls=ell)`, and `pm.gp.HSGP(m=[50], c=1.5, cov_func=cov)`. It supplies `c`, never `L`, and makes both the stationary-kernel requirement and `c >= 1.2` visible. The exact comparator is `pm.gp.Marginal` with the same Matérn-5/2 covariance family and Normal likelihood on the fixed 96-row observed subset. Both models use their respective standardized time/demand transformations and `pm.Data` for prediction grids.

The HSGP posterior fit is click-gated. Learners may alter only `m` or `c` from the baseline after first recording its prior/posterior predictive plots, boundary behavior, divergences, rank-R-hat, and ESS. An alternative setting is accepted only if it meets the release diagnostic criteria and does not visibly worsen the boundary or posterior-predictive assessment.

**Outputs:** declared domain and basis settings, exact-versus-HSGP posterior prediction in original units, prior/posterior predictive plots, divergence/R-hat/ESS display, and an explicit approximation decision.

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
- The only posterior-MCMC actions are the Hour 1 salmon primer, Hour 2 salmon Marginal GP and coal Latent GP, and Hour 4 exact-electricity comparator and full HSGP. Hour 3 runs model validation and prior predictive simulation only.
- Every course posterior fit uses four chains, 500 tuning draws, 500 posterior draws, and its notebook-specific fixed seed. Release rehearsal budgets are 60 seconds for the Hour 1 primer, 90 seconds for each Hour 2 fit and the Hour 4 exact comparator, and 180 seconds for the full HSGP fit on the supported laptop baseline.
- A fit exceeding its budget, failing to complete, or failing its diagnostic criteria blocks release. Its model or sampler budget may change only through a specification amendment; the observed-data selection and the learner-facing model contract remain fixed.
- Sampling warnings, divergences, and convergence failures are visible instructional output. There are no global warning filters, blanket `try/except` blocks, fake posterior results, or silent caches.
- Optional future remote hosting must use a pinned CPython/server runtime and a deployment smoke test. It must not publish the core notebooks as WASM or a Pyodide preview.

## Verification and release criteria

1. **Data contracts:** tests verify every vendored asset’s schema, documented transformations, exact curated count/range, and provenance record. They fail if a notebook includes a runtime HTTP data path. The coal and electricity source/licence records are mandatory before their assets are copied into `data/`.
2. **Model contracts:** fast tests build every model on its deterministic observed-data slice, call model validation, and verify coordinates, named variables, likelihood support, active dimensions, and prediction interfaces. `pymc.testing.mock_sample` is allowed only for downstream structure/rendering checks.
3. **Real-sampling smoke tests:** focused tests sample small fixed observed slices and assert the returned `DataTree` has the expected posterior, sample-stat, and posterior-predictive groups. Their low draw count proves execution only; it does not support a convergence claim.
4. **Notebook and platform checks:** all notebooks pass `pixi run marimo check --strict`; script-mode smoke tests cover the environment check and every non-sampling path; the lock must install and the same checks must pass on `win-64`, `linux-64`, `osx-64`, and `osx-arm64`.
5. **Locked-environment rehearsal:** before publication, run every click-gated posterior fit on the supported laptop baseline. Every released fit must have zero divergences, rank-R-hat below 1.01, and bulk and tail ESS above 400 for every reported scalar parameter. The posterior predictive display must show no persistent unmodeled pattern in the observed-data domain. The HSGP additionally must not show visibly worse boundary behavior than the exact 96-row comparator.
6. **Course acceptance:** a new attendee can complete the environment check, run all non-sampling material, execute each exercise’s checker, and obtain the released core-model diagnostics without network access after the one-time bootstrap.

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
- Cross-platform lock semantics: [Pixi manifest reference](https://prefix-dev.github.io/pixi/latest/reference/pixi_manifest/) documents that `[workspace].platforms` resolves all listed targets into one lockfile.
- Version floors and supported Python releases: [PyMC 6.1.0 PyPI metadata](https://pypi.org/pypi/pymc/6.1.0/json), [ArviZ 1.2.0 PyPI metadata](https://pypi.org/pypi/arviz/json), and [marimo 0.23.13 PyPI metadata](https://pypi.org/pypi/marimo/json), accessed 2026-07-09.
