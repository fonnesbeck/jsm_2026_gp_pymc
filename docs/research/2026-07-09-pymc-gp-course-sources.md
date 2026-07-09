# First-party sources for the PyMC Gaussian-process course

**Accessed:** 2026-07-09. **Scope:** implementation constraints for the four-hour
course described in `README.md`; no local teaching notebook was inspected. **Status
labels:** **Verified** means stated by the cited first-party documentation/source;
**Recommendation** is a course-design consequence.

## Data-evidence rule

**Course decision:** every fitted exercise—Marginal, Latent, hierarchical,
HSGP, HSGPPeriodic, posterior predictive, and diagnostic—**MUST use observed
real data**. Simulated GP **prior-function** draws are permitted only before
fitting to build kernel intuition; they are not a fitted exercise, a substitute
dataset, or evidence about model fit. Prior-predictive checks remain a required
diagnostic of a model fitted to observed data, and must be labelled as such.

This distinction applies to all recommendations below.

## Source inventory and course constraints

| Curriculum material | Instructional role and readiness | Verified implementation constraints | Dependencies / assets | Modern-API hazards and recommendation | Curriculum contract |
| --- | --- | --- | --- | --- | --- |
| PyMC and GP foundations | **Ready.** Use the official GP guide for the hour-one conceptual bridge: a GP is determined by a mean and covariance function, and PyMC uses GPs as composable model components. | **Verified:** covariance constructors require the total `input_dim`; `active_dims` selects columns. A covariance may be added to or multiplied by another covariance or a scalar. | One small observed real 1-D dataset for the primer; NumPy, PyMC, plotting; separate simulated prior-function draws for intuition only. | **Recommendation:** make every `X` a 2-D design matrix, even for 1-D examples, and introduce `active_dims` before multidimensional/kernel-combination exercises. Any fitted primer uses the observed data. | `README.md:26-28`; `README.md:32-35` |
| Exact conjugate GP: `pm.gp.Marginal` | **Ready.** The hour-two Gaussian-response model should use `gp = pm.gp.Marginal(cov_func=...)` followed by `gp.marginal_likelihood(name, X, y, sigma)`. | **Verified:** `Marginal` implements a GP plus additive noise, exposes `marginal_likelihood`, `conditional`, and `predict`, and is documented for normally distributed regression data. | Small observed real continuous-response dataset; covariance, noise prior, prediction grid. | **Recommendation:** teach `marginal_likelihood`, not an invented observation likelihood around a latent `f`; retain `conditional` only after fitting the observed data. The API’s numerical `jitter` option belongs in the numerical-stability discussion, not as an unexplained default. | `README.md:29-31` |
| Exact non-conjugate GP: `pm.gp.Latent` | **Ready.** Use it for a short Poisson or Binomial example: `f = gp.prior("f", X=...)`, then attach the appropriate likelihood/link. | **Verified:** `Latent` has no additive noise, represents function values as latent variables, and provides `prior` and `conditional`; PyMC documents it for non-Normal data. | Small observed real count/binary dataset, 1-D `X`, link and likelihood. | **Recommendation:** keep this observed-data fit deliberately small: a latent value is created for each input. Avoid calling the outline’s `gp.Prior`; the current public API is `gp.prior(...)`. | `README.md:29-31`; `README.md:44-48` |
| Covariance kernels and compositions | **Ready.** Use `ExpQuad`, `Matern32`, `Matern52`, and `Periodic`; demonstrate trend + seasonality as a sum/product only after learners can vary one kernel at a time. | **Verified:** the current module exposes those kernels; kernel algebra supports addition, multiplication, and scalar scaling. GP objects can be added, but must share the same implementation type; conditionals for a component of an additive GP require its `given={"X", "y", "sigma", "gp"}` inputs. | Simulated prior-function-draw plots for intuition only; one observed real 1-D seasonal dataset for fitting; 2-D observed data only for `active_dims`. | **Recommendation:** distinguish a kernel sum/product from adding GP *objects*. Never combine `Marginal` and `Latent` objects. Make amplitude explicit (for example, scalar-times-kernel) so learners can identify its prior separately from length scale. All kernel-combination fits use observed data. | `README.md:32-35`; `README.md:47-48` |
| Hierarchical GP | **Adapt, not copy.** The official advanced HSGP example demonstrates a hierarchical mean-plus-group-deviation structure and is a useful structural reference, but it is unsuitable as a four-hour beginner notebook verbatim. | **Verified:** the example implements a hierarchical HSGP and uses both group and deviation GPs; its published environment imports PreliZ in addition to PyMC, ArviZ, NumPy, Matplotlib, and PyTensor. | A compact observed real grouped 1-D dataset; do not require PreliZ merely to reproduce the example. | **Recommendation:** build a minimal latent-GP structural exercise on this observed data; do not transplant the advanced notebook or its external dependency stack. | `README.md:35`; `README.md:37-40` |
| Scalable `pm.gp.HSGP` | **Ready, with explicit approximation diagnostics.** This is the hour-four implementation focus. | **Verified:** HSGP is a reduced-rank approximation with a stationary covariance whose power spectral density is implemented. `m` is a sequence with one element per active dimension; specify exactly one of `L` or `c`; `L` bounds the domain and `c` constructs it. PyMC warns that `c >= 1.2` is recommended. Official examples give exact-GP per-step cost as $\mathcal{O}(n^3)$ and HSGP as $\mathcal{O}(mn + m)$. The basis count grows as the product of per-dimension `m`; the official example says HSGP is likely inefficient above three input dimensions. | One observed real 1-D time-series dataset; a small observed slice fitted with an exact GP; a larger observed slice fitted with HSGP; an explicit training-and-prediction domain for selecting `L`/`c`/`m`. | **Recommendation:** require learners to check that training *and prediction* inputs stay inside the intended boundary and to compare prior/posterior functions while increasing `m`. Do not present HSGP as exact or as generally preferable for small data. Do not use an ordinary `Periodic` kernel with HSGP. | `README.md:37-40`; `README.md:49` |
| `pm.gp.HSGPPeriodic` | **Optional short contrast.** Use only if seasonality is already established; it should not replace the general HSGP lesson. | **Verified:** it is a basis approximation for `Periodic`, follows the HSGP-like API, requires a `Periodic` covariance and positive integer `m`, and is implemented only for 1-D input. It is explicitly not the same Hilbert-space approximation as `HSGP`. | Observed real 1-D seasonal input and period prior; no multidimensional seasonal example. | **Recommendation:** label it a specialized periodic approximation and keep the general `HSGP` and `HSGPPeriodic` APIs separate in code and narration. | `README.md:34`; `README.md:37-40` |
| Sampling, predictive checks, diagnostics, and result handling | **Ready as a repeated workflow**, rather than a final add-on. | **Verified:** `pm.sample_prior_predictive(draws=...)` defaults to returning an `xarray.DataTree`; `pm.sample_posterior_predictive` can extend the input DataTree in place and places in-sample checks in `posterior_predictive` (`predictions=True` uses `predictions`). `pm.Data` supports replacing values/shapes (not dimensionality) for prediction. Current `pm.sample` chooses nutpie when installed and compatible, otherwise PyMC’s sampler; non-PyMC NUTS implementations require a fully continuous model. `arviz_stats.rhat` accepts DataTree input, defaults to rank-normalized R-hat, and requires at least two posterior chains; `arviz_stats.ess` defaults to bulk ESS; `arviz_stats.summary(..., kind="all")` reports bulk/tail ESS and R-hat. | A single `random_seed` convention; prior-function plots for intuition only; PPC plot, rank/trace diagnostic plot, and compact summary table from each observed-data fit. | **Recommendation:** run a labelled prior-predictive check before each observed-data fit and posterior predictive simulation after it. Pin the course runtime, then use group-safe DataTree access such as `idata["posterior"]` and `idata["sample_stats"]`, rather than attribute access. The current stable `pm.sample` page still labels its default return as `arviz.InferenceData`, while current predictive APIs and the current predictive tutorial display `xarray.DataTree`; the version-pinned notebook smoke test must settle the exact runtime object before learners see it. Use explicit `chains >= 2` for any R-hat lesson. | `README.md:26-31`; `README.md:37-40`; `README.md:44-49` |
| marimo validation and reactive interactivity | **Ready.** Apply these requirements to every eventual notebook; they are not optional deployment polish. | **Verified:** `marimo check [FILES]` checks/formats but changes files only with `--fix`; `--strict` makes warnings fail. marimo derives a DAG from cell definitions/references, requires unique globals, and does not track cross-cell mutation. `mo.ui.run_button()` sets `value=True` and runs dependent cells; pairing it with `mo.stop(not button.value)` gates computation. `mo.cache` is in-memory; `mo.persistent_cache` is disk-backed and defaults to `__marimo__/cache/`. | marimo; project-managed Python environment; cached sampling/result artifact strategy. | **Recommendation:** validate each generated notebook with `marimo check --strict <notebook.py>` **without** `--fix`. Give sampler cells an explicit run button and a stopped default; cache deterministic, parameter-keyed expensive work only. Do not cache an unseeded stochastic fit as though it were reproducible, and do not mutate an `idata` object across reactive cells—return a new output or keep the mutation in its defining cell. Add persistent-cache paths to ignore rules when the course implementation is created. | Entire interactive-notebook delivery; especially `README.md:37-40` |

## API decisions the course should lock before implementation

1. **Use the current public spellings:** `pm.gp.Marginal`,
   `gp.marginal_likelihood`, `pm.gp.Latent`, `gp.prior`, `pm.gp.HSGP`, and
   `pm.gp.HSGPPeriodic`. Do not inherit old snippets that call `gp.Prior`.
2. **Keep prediction data mutable:** use `pm.Data`/`pm.set_data` when an
   example swaps input locations; preserve named coordinates/dimensions when
   their length changes.
3. **Treat HSGP as an approximation with predeclared domain and resolution:**
   expose `m`, `L`/`c`, stationary-kernel eligibility, and its practical
   dimensional limit. The special 1-D periodic approximation is not a drop-in
   generalization.
4. **Make diagnosis observable:** prior predictive draws, posterior predictive
   draws, divergence count from `idata["sample_stats"]`, rank-R-hat, and
   bulk/tail ESS must be visible learner outputs—not prose-only advice.
5. **Treat MCMC as an expensive interactive action:** non-sampler cells may be
   reactive; sampling must be explicitly gated and deterministically seeded.

## Deployment disposition: remote/sandboxed execution vs browser-only WASM

### Verdict

**Browser-only marimo WASM/Pyodide: FAIL for the current PyMC/PyTensor GP
notebooks.** **Remote/server execution: supported in principle, but requires a
version-pinned environment smoke test.** This is a static compatibility
decision, not a browser-execution claim: no course notebook was run.

**Verified facts**

- marimo WASM runs Python *entirely in the browser*, without a backend, using
  Pyodide. Its own guidance directs heavy computations to a local/server
  marimo runner or molab instead.
- The current official Pyodide package catalog contains neither `pymc` nor
  `pytensor`. Pyodide can install only pure-Python PyPI wheels or
  wasm32/Emscripten wheels built for Pyodide. PyMC 6.1.0 declares
  `pytensor>=3.1.2,<3.2`; PyTensor’s current project manifest requires Cython
  at build time and depends on NumPy, SciPy, and Numba. No first-party
  Pyodide-built `pytensor`/PyMC wheel was found in the official catalog.
- PEP 723 records dependencies for a sandbox; it does not turn a native-only
  dependency into a WASM wheel. marimo does, however, honor PEP 508 markers,
  including `sys_platform == "emscripten"`, for sandbox and WASM installs.
- A marimo `--sandbox` is an isolated, `uv`-managed Python environment on the
  runner—not browser-only Python. molab separately documents normal server
  containers (default: four CPUs and 32 GB RAM) and distinguishes its
  ephemeral-server option from its browser/WASM option.

**Course recommendation**

1. The canonical Marginal, Latent, HSGP, HSGPPeriodic, sampling, and
   diagnostics notebooks **MUST run on CPython in a local/server sandbox or
   remote runner**. PEP 723 makes that fallback self-contained, but should pin
   the resolved PyMC/PyTensor/Python stack and be smoke-tested in the actual
   host image.
2. **Do not publish those fitting notebooks as `html-wasm` or a molab
   `/wasm` preview.** A PEP 508 exclusion such as
   `pymc; sys_platform != "emscripten"` prevents a misleading failed install;
   it does not make the `import pymc` cells executable in the browser.
3. If a browser-only companion is required, make it a separate, explicitly
   non-PyMC lesson: small embedded data, NumPy-only kernel/prior visualization,
   or static/precomputed results derived from observed real data. It must not claim to fit the GP or
   run MCMC. Keep it separate from the curriculum’s implementation exercises
   (`README.md:29-40`).
4. For a hosted teaching experience, use a normal server session (for example,
   local `marimo edit --sandbox` or molab’s ephemeral server), then reserve
   WASM only for lightweight previews. Server-side interactive controls and
   cache policy from the prior section still apply.

The explicit browser constraints are a release gate: a future custom
Emscripten build of the complete PyTensor/PyMC dependency graph would require
separate proof in the target browser. It is not an available course fallback
today.

## Primary sources

All links below are official documentation or first-party source, accessed
2026-07-09. PyMC stable pages reported **PyMC 6.1.0** at access time.

### PyMC

- [Gaussian Processes core guide](https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/Gaussian_Processes.html) — GP semantics, `input_dim`/`active_dims`, covariance algebra, and additive-GP conditionals.
- [`pymc.gp.Marginal` API](https://www.pymc.io/projects/docs/en/stable/api/gp/generated/pymc.gp.Marginal.html) — Normal-response marginal GP and public methods.
- [`pymc.gp.Latent` API](https://www.pymc.io/projects/docs/en/stable/api/gp/generated/pymc.gp.Latent.html) — latent-function GP and non-Normal use.
- [Covariance-function API](https://www.pymc.io/projects/docs/en/stable/api/gp/cov.html) — current kernel inventory.
- [`pymc.gp.HSGP` API](https://www.pymc.io/projects/docs/en/stable/api/gp/generated/pymc.gp.HSGP.html) — stationary-kernel, `m`, `L`/`c`, and parameterization constraints.
- [`pymc.gp.HSGPPeriodic` API](https://www.pymc.io/projects/docs/en/stable/api/gp/generated/pymc.gp.HSGPPeriodic.html) — specialized periodic approximation and 1-D limit.
- [HSGP basic example](https://www.pymc.io/projects/examples/en/latest/gaussian_processes/HSGP-Basic.html) and [advanced example](https://www.pymc.io/projects/examples/en/latest/gaussian_processes/HSGP-Advanced.html) — official complexity, dimensional, approximation-fidelity, hierarchy, and dependency evidence.
- [Current HSGP source](https://raw.githubusercontent.com/pymc-devs/pymc/main/pymc/gp/hsgp_approx.py) — constructor validation, `c >= 1.2` warning, and the `approx_hsgp_hyperparams` contract.
- [`pymc.sample` API](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.sample.html) — current sampler/backend selection and chain semantics.
- [`pymc.sample_prior_predictive` API](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.sample_prior_predictive.html) and [`pymc.sample_posterior_predictive` API](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.sample_posterior_predictive.html) — DataTree predictive outputs/groups and extension behavior.
- [Prior and posterior predictive checks](https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/posterior_predictive.html) — workflow rationale and current DataTree display.
- [`pymc.Data` API](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.Data.html) — prediction-time data replacement and dimensions.

### ArviZ / xarray

- [`arviz_stats.rhat`](https://python.arviz.org/projects/stats/en/stable/api/generated/arviz_stats.rhat.html), [`arviz_stats.ess`](https://python.arviz.org/projects/stats/en/stable/api/generated/arviz_stats.ess.html), and [`arviz_stats.summary`](https://python.arviz.org/projects/stats/en/stable/api/generated/arviz_stats.summary.html) — current official diagnostics APIs accepting DataTree.
- [`xarray.DataTree.__getitem__`](https://docs.xarray.dev/en/stable/generated/xarray.DataTree.__getitem__.html) — group/variable access used by the DataTree convention.

### marimo

- [CLI reference: `marimo check`](https://docs.marimo.io/cli/#marimo-check) — non-mutating validation and strict mode.
- [Reactivity guide](https://docs.marimo.io/guides/reactivity/) — DAG, unique globals, and mutation behavior.
- [Working with expensive notebooks](https://docs.marimo.io/guides/expensive_notebooks/) and [`run_button`](https://docs.marimo.io/api/inputs/run_button/) — run gating and autorun controls.
- [Caching API](https://docs.marimo.io/api/caching/) — cache scope, keys, persistence path, and limitations.
- [WebAssembly notebooks](https://docs.marimo.io/guides/wasm/) — browser-only Pyodide runtime, package limitations, PEP 508 markers, and 2-GB memory limit.
- [Inline dependencies / PEP 723](https://docs.marimo.io/guides/package_management/inlining_dependencies/) — `uv`-managed isolated sandboxes and script metadata.
- [molab cloud execution](https://docs.marimo.io/guides/molab/) — normal server containers and the distinction between ephemeral-server and WASM execution.
- [WebAssembly HTML export](https://docs.marimo.io/guides/exporting/webassembly_html/) — browser-only export constraints.

### Pyodide and package availability

- [Packages built in Pyodide](https://pyodide.org/en/stable/usage/packages-in-pyodide.html) — current official catalog; `pymc` and `pytensor` are absent.
- [Loading packages](https://pyodide.org/en/stable/usage/loading-packages.html) — only pure-Python wheels and Pyodide wasm32/Emscripten wheels are installable through `micropip`.
- [PyMC 6.1.0 PyPI metadata](https://pypi.org/pypi/pymc/6.1.0/json) — official dependency on PyTensor and Python-version constraint.
- [PyTensor project manifest](https://raw.githubusercontent.com/pymc-devs/pytensor/main/pyproject.toml) — current build and runtime dependency evidence.
