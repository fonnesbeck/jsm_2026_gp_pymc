# Gaussian Processes Course Materials Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a cross-platform, self-contained four-hour marimo course that teaches current PyMC Gaussian-process modeling with observed real data, interactive exercises, and verified diagnostics.

**Architecture:** A single Pixi workspace provides the locked CPython environment for Windows, macOS, and Linux. Four marimo notebooks load vendored, schema-validated datasets independently; no teaching notebook shares reactive state or fetches data. Each modeling notebook makes its Bayesian workflow explicit, and sampling is triggered only through a run button.

**Tech Stack:** Python 3.13; Pixi; PyMC 6.1.0; ArviZ 1.2.0; marimo 0.23.13; NumPy; Polars; Matplotlib; pytest; Ruff.

## Global Constraints

- Support exactly `win-64`, `linux-64`, `osx-64`, and `osx-arm64` through one `pixi.lock`.
- Pin Python `3.13.*`, PyMC `6.1.0`, ArviZ `1.2.0`, and marimo `0.23.13` in `pixi.toml`.
- All fitted models, predictive checks, and diagnostics use vendored observed real data. GP prior-function draws are permitted only as labelled kernel visualizations.
- Never fetch data from a notebook. The environment check and all notebook paths work offline after `pixi install`.
- Use PyMC 6 DataTree access: `idata["posterior"]`, `idata["sample_stats"]`, and `idata["posterior_predictive"]`. Do not use legacy attribute access, `MultiTrace`, `trace.extend`, `gp.Prior`, or explicit `nuts_sampler="nutpie"`.
- Use `pm.Data` for inputs and index arrays that support prediction; use named coordinates for observable dimensions.
- Every notebook exposes only pure data/model builders through single-function `@app.function` cells that reference imports from one `app.setup` cell. Tests dynamically import the numbered notebook files; they never test source text or execute `app.run()`.
- Every notebook ends with `if __name__ == "__main__": app.run()` so its marimo UI is available interactively while dynamic imports stay side-effect-free in tests.
- Every posterior fit is click-gated with `mo.ui.run_button()` and `mo.stop(not run_button.value)`, uses four chains, 500 tuning draws, 500 posterior draws, and a fixed notebook seed.
- Hour 3 constructs likelihoods and runs prior predictive checks only; it does not run posterior MCMC.
- Release limits on a 4-logical-core, 16-GB supported laptop: 60 seconds for the Hour 1 primer; 90 seconds for each Hour 2 fit and the exact-electricity comparator; 180 seconds for the full HSGP fit.
- Release posterior diagnostics: zero divergences, rank-R-hat < 1.01, and bulk/tail ESS > 400 for every reported scalar parameter. Posterior predictive plots may not show a persistent unexplained pattern; HSGP may not visibly worsen boundary behavior versus the exact 96-row comparator.
- No global warning filters, blanket exception handlers, fake posterior outputs, silent result caches, browser-only WASM execution, or runtime HTTP paths.

## File Structure

| File | Responsibility |
| --- | --- |
| `pixi.toml` / `pixi.lock` | Exact multi-platform course environment and standard test/lint/check tasks. |
| `data/README.md` | Public source URL, licence/redistribution decision, raw-source transform, schema, and deterministic selection contract for every distributed asset. |
| `data/salmon.txt` | 40 annual Fraser River salmon observations. |
| `data/coal_disasters.csv` | 111 annual historical coal-disaster counts for 1851–1961. |
| `data/gb_electricity_demand.csv` | 1,826 daily GB electricity-demand observations for 2015–2019, with verified provenance. |
| `data/walker_lake.csv` | Normalized Walker Lake table with `id`, `x_m`, `y_m`, and `v_ppm`; the raw `1E31` sentinel is not distributed as concentration data. |
| `data/fastball_spin_rates.csv` | Original game-level pitcher spin-rate file, used only through the fixed quality-filtered subset. |
| `notebooks/00_environment_check.py` | Pre-course version, data-asset, and minimal model-compilation check. |
| `notebooks/01_foundations.py` | Salmon Normal-regression primer, GP intuition, and prior-function kernel interaction. |
| `notebooks/02_exact_and_latent_gps.py` | Exact salmon `Marginal` GP and latent coal-count Poisson GP. |
| `notebooks/03_kernels_inputs_and_hierarchy.py` | Electricity kernel composition, Walker Lake 2-D inputs, and structural hierarchical latent GP. |
| `notebooks/04_workflow_and_hsgp.py` | Exact-versus-HSGP workflow, diagnostics, and approximation decision. |
| `tests/__init__.py` / `tests/notebook_loader.py` | Dynamic import utility for testable `@app.function` builders in numbered marimo files. |
| `tests/test_data_contracts.py` | Schema, provenance, deterministic curation, and no-network contracts. |
| `tests/test_model_contracts.py` | Fast model-build, coordinate, likelihood, prior-predictive, and prediction-interface contracts. |
| `tests/test_notebook_smoke.py` | Marimo/script-mode static behavior and notebook-content smoke contracts. |

---

### Task 1: Create the locked multi-platform course workspace

**Files:**
- Create: `pixi.toml`
- Create: `.gitignore`

**Interfaces:**
- Consumes: the platform and version decisions in `docs/superpowers/specs/2026-07-09-gp-course-materials-design.md`.
- Produces: `pixi run test`, `pixi run lint`, and `pixi run check` commands used by every later task.

- [ ] **Step 1: Create the manifest with the exact platform matrix and version pins**

```toml
[workspace]
name = "jsm-2026-gp-course"
channels = ["conda-forge"]
platforms = ["win-64", "linux-64", "osx-64", "osx-arm64"]

[dependencies]
python = "3.13.*"
pymc = "6.1.0"
arviz = "1.2.0"
marimo = "0.23.13"
numpy = "*"
polars = "*"
matplotlib = "*"
pytest = "*"
ruff = "*"

[tasks]
test = "pytest"
lint = "ruff check ."
format = "ruff format ."
check = "ruff check . && ruff format --check ."
marimo-check = "marimo check --strict notebooks"
```

Do not add `nutpie`: PyMC chooses an available compatible sampler without a course-level sampler override.

- [ ] **Step 2: Add only generated-environment and result artifacts to `.gitignore`**

```gitignore
.pixi/
__pycache__/
.pytest_cache/
.ruff_cache/
__marimo__/
*.pyc
```

Do not ignore `data/`; the course data are release artifacts.

- [ ] **Step 3: Solve every declared platform and verify the lock contains all four targets**

Run: `pixi lock`

Expected: `pixi.lock` is created or updated with `win-64`, `linux-64`, `osx-64`, and `osx-arm64` package records.

- [ ] **Step 4: Verify the local environment task surface**

Run: `pixi run check`

Expected: PASS before notebook files exist because the repository contains only valid TOML/ignore configuration.

- [ ] **Step 5: Commit**

```bash
git add pixi.toml pixi.lock .gitignore
git commit -m "build: add cross-platform course environment"
```

### Task 2: Establish licensed course datasets and deterministic data contracts

**Files:**
- Create: `data/README.md`
- Create: `data/salmon.txt`
- Create: `data/coal_disasters.csv`
- Create: `data/gb_electricity_demand.csv`
- Create: `data/walker_lake.csv`
- Create: `data/fastball_spin_rates.csv`
- Create: `tests/test_data_contracts.py`

**Interfaces:**
- Consumes: the audited local candidates and exact curation rules in the approved specification.
- Produces: offline data assets with auditable provenance and stable model inputs.

- [ ] **Step 1: Record source and redistribution evidence before copying each candidate asset**

`data/README.md` must contain one row per distributed file with these columns:

```markdown
| File | Primary source URL | Licence / redistribution basis | Raw-to-course transformation | Schema | Deterministic analysis subset |
| --- | --- | --- | --- | --- | --- |
```

Use the authoritative original source or its explicit data licence. For each candidate lacking redistribution permission—especially the locally derived electricity file and embedded coal vector—replace it before implementation with a source that preserves the approved model contract. Do not invent a citation or treat a source repository path as a licence.

- [ ] **Step 2: Write the failing data-contract tests**

```python
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

DATA = Path("data")


def assert_no_nulls(frame: pl.DataFrame) -> None:
    assert int(frame.null_count().to_numpy().sum()) == 0


def test_salmon_contract() -> None:
    salmon = pl.read_csv(DATA / "salmon.txt", separator=" ", has_header=True)
    assert salmon.columns == ["year", "recruits", "spawners"]
    assert salmon.height == 40
    assert salmon.get_column("year").n_unique() == 40
    assert_no_nulls(salmon)
    assert salmon.get_column("spawners").min() == 51
    assert salmon.get_column("spawners").max() == 490


def test_coal_contract() -> None:
    coal = pl.read_csv(DATA / "coal_disasters.csv")
    assert coal.columns == ["year", "count"]
    assert coal.height == 111
    assert coal.get_column("year").to_list() == list(range(1851, 1962))
    assert coal.schema["count"].is_integer()
    assert coal.get_column("count").min() == 0
    assert coal.get_column("count").max() == 6


def test_electricity_contract() -> None:
    demand = pl.read_csv(DATA / "gb_electricity_demand.csv", try_parse_dates=True)
    assert demand.columns == ["date", "load_mw"]
    assert demand.height == 1826
    assert_no_nulls(demand)
    expected_dates = pl.date_range(date(2015, 1, 1), date(2019, 12, 31), "1d", eager=True)
    assert demand.get_column("date").to_list() == expected_dates.to_list()
    positions = np.rint(np.linspace(0, demand.height - 1, 96)).astype(int)
    assert len(np.unique(positions)) == 96


def test_walker_contract() -> None:
    walker = pl.read_csv(DATA / "walker_lake.csv")
    assert walker.columns == ["id", "x_m", "y_m", "v_ppm"]
    assert walker.height == 195
    assert_no_nulls(walker)
    assert walker.get_column("id").to_list() == list(range(1, 196))


def test_spin_rate_contract() -> None:
    spin = pl.read_csv(DATA / "fastball_spin_rates.csv", try_parse_dates=True)
    curated = (
        spin.filter(pl.col("avg_spin_rate").is_not_null() & (pl.col("n_pitches") >= 10))
        .sort(["pitcher_name", "game_date"])
    )
    ranked = (
        curated.group_by("pitcher_name")
        .len()
        .sort(["len", "pitcher_name"], descending=[True, False])
        .head(3)
    )
    selected = ["Buehler, Walker", "Hearn, Taylor", "Rodriguez, Richard"]
    assert sorted(ranked.get_column("pitcher_name").to_list()) == selected
    per_pitcher = curated.filter(pl.col("pitcher_name").is_in(selected)).group_by("pitcher_name").head(10)
    assert per_pitcher.height == 30
    assert per_pitcher.group_by("pitcher_name").len().sort("pitcher_name").get_column("len").to_list() == [10, 10, 10]
```

- [ ] **Step 3: Run the tests to verify they fail before assets are normalized and copied**

Run: `pixi run pytest tests/test_data_contracts.py -v`

Expected: FAIL because course assets and provenance document do not yet exist.

- [ ] **Step 4: Vendor and normalize assets to the contract**

- Copy the salmon values with one-space separation and the three required columns.
- Write coal counts as the two-column CSV `[year,count]`; do not encode them in a notebook cell.
- Vendor a properly attributed electricity CSV with exactly `date,load_mw`, daily coverage from 2015-01-01 through 2019-12-31, and 1,826 rows.
- Parse the Walker Lake raw file, discard its eight metadata rows, retain `id <= 195`, and write only `id,x_m,y_m,v_ppm`. Never convert `1E31` into a valid observation.
- Vendor the original spin-rate CSV; retain source columns so the quality filter remains inspectable.
- Complete every provenance row in `data/README.md` with a real source URL and redistribution basis.

- [ ] **Step 5: Run the data tests**

Run: `pixi run pytest tests/test_data_contracts.py -v`

Expected: PASS; all curation rules are deterministic and every selected source is documented.

- [ ] **Step 6: Commit**

```bash
git add data tests/test_data_contracts.py
git commit -m "data: add verified GP course datasets"
```

### Task 3: Build the pre-course environment check notebook

**Files:**
- Create: `notebooks/00_environment_check.py`
- Create: `tests/__init__.py`
- Create: `tests/notebook_loader.py`
- Modify: `tests/test_notebook_smoke.py`

**Interfaces:**
- Consumes: the locked environment and all vendored data files.
- Produces: a script-mode-safe marimo notebook that proves installation, assets, and a minimal PyMC compilation without sampling; `environment_check() -> list[str]` is importable for tests.

- [ ] **Step 1: Add the dynamic numbered-notebook loader**

```python
# tests/notebook_loader.py
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType


def load_notebook(filename: str) -> ModuleType:
    path = Path("notebooks") / filename
    spec = spec_from_file_location(f"course_{path.stem.replace('-', '_')}", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
```

Create an empty `tests/__init__.py`.

- [ ] **Step 2: Write the failing behavioural environment-check test**

```python
from tests.notebook_loader import load_notebook


def test_environment_check_passes_in_the_locked_environment() -> None:
    notebook = load_notebook("00_environment_check.py")
    assert notebook.environment_check() == []
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `pixi run pytest tests/test_notebook_smoke.py::test_environment_check_passes_in_the_locked_environment -v`

Expected: FAIL because the notebook and its `environment_check` function do not exist.

- [ ] **Step 4: Implement the environment check**

Create a marimo `App(width="medium")` with one `app.setup` import cell and a single-function `@app.function` builder:

```python
@app.function
def environment_check() -> list[str]:
    expected_versions = {
        "pymc": "6.1.0",
        "arviz": "1.2.0",
        "marimo": "0.23.13",
    }
    required_assets = [
        "salmon.txt",
        "coal_disasters.csv",
        "gb_electricity_demand.csv",
        "walker_lake.csv",
        "fastball_spin_rates.csv",
    ]
    failures = []
    if sys.version_info[:2] != (3, 13):
        failures.append(f"python: expected 3.13, found {sys.version.split()[0]}")
    for package, expected in expected_versions.items():
        actual = importlib.metadata.version(package)
        if actual != expected:
            failures.append(f"{package}: expected {expected}, found {actual}")
    data_dir = Path(__file__).resolve().parents[1] / "data"
    failures.extend(
        f"missing data asset: {asset}"
        for asset in required_assets
        if not (data_dir / asset).is_file()
    )
    with pm.Model() as model:
        pm.Normal("environment_check_observed", 0, 1, observed=np.array([0.0]))
    model.compile_logp()
    return failures
```

Use `importlib.metadata.version`, `Path(__file__).resolve().parents[1] / "data"`, and an empty `pm.Model()` with one `pm.Normal` observed variable followed by `model.compile_logp()`. The display cell renders a success callout for `[]` and an error callout listing every returned failure. Do not sample, install packages, or catch unrelated exceptions.

- [ ] **Step 5: Verify structure and script behavior**

Run: `pixi run pytest tests/test_notebook_smoke.py::test_environment_check_passes_in_the_locked_environment -v`

Expected: PASS.

Run: `pixi run marimo check --strict notebooks/00_environment_check.py`

Expected: PASS.

Run: `pixi run python notebooks/00_environment_check.py`

Expected: exits successfully after compiling the minimal model and validating all assets.

- [ ] **Step 6: Commit**

```bash
git add notebooks/00_environment_check.py tests
git commit -m "feat: add course environment check"
```

### Task 4: Implement the real-data foundations notebook

**Files:**
- Create: `notebooks/01_foundations.py`
- Modify: `tests/test_model_contracts.py`
- Modify: `tests/test_notebook_smoke.py`

**Interfaces:**
- Consumes: `data/salmon.txt` with 40 validated rows.
- Produces: the standardized salmon Normal-regression primer, labelled GP prior draws, and a kernel-effect exercise.

- [ ] **Step 1: Write failing model-contract tests**

```python
import numpy as np
import pymc as pm

from tests.notebook_loader import load_notebook


def test_salmon_primer_uses_observed_data() -> None:
    foundations = load_notebook("01_foundations.py")
    model, x, y = foundations.build_salmon_linear_model()
    assert x.shape == (40, 1)
    assert y.shape == (40,)
    assert set(model.named_vars) >= {"alpha", "beta", "sigma", "recruits"}
    model.debug()


def test_salmon_gp_uses_marginal_likelihood() -> None:
    foundations = load_notebook("01_foundations.py")
    model, gp = foundations.build_salmon_marginal_model()
    assert isinstance(gp, pm.gp.Marginal)
    assert "recruits" in model.named_vars
    model.debug()
```

Expose each pure builder in its own `@app.function` cell; keep shared imports in the notebook's single `app.setup` cell. Marimo cells call those functions, but tests only dynamically import and invoke them.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run pytest tests/test_model_contracts.py -k salmon -v`

Expected: FAIL because `notebooks/01_foundations.py` and its builders do not exist.

- [ ] **Step 3: Implement the builders and notebook flow**

Implement a standardization helper with `ddof=0`, then these model forms:

```python
with pm.Model(coords={"obs": np.arange(len(y)), "input": ["spawners"]}) as model:
    x_data = pm.Data("x", x, dims=("obs", "input"))
    alpha = pm.Normal("alpha", 0, 1)
    beta = pm.Normal("beta", 0, 1)
    sigma = pm.HalfNormal("sigma", 1)
    pm.Normal("recruits", alpha + beta * x_data[:, 0], sigma, observed=y, dims="obs")
```

```python
with pm.Model(coords={"obs": np.arange(len(y)), "input": ["spawners"]}) as model:
    x_data = pm.Data("x", x, dims=("obs", "input"))
    alpha = pm.Normal("alpha", 0, 1)
    eta = pm.HalfNormal("eta", 1)
    ell = pm.LogNormal("ell", 0, 1)
    sigma = pm.HalfNormal("sigma", 1)
    cov = eta**2 * pm.gp.cov.Matern52(1, ls=ell)
    gp = pm.gp.Marginal(mean_func=pm.gp.mean.Constant(c=alpha), cov_func=cov)
    gp.marginal_likelihood("recruits", X=x_data, y=y, sigma=sigma)
```

Use a slider-only cell for `eta` and `ell` that regenerates labelled *prior-function* visualizations. Keep it independent from sampling cells. Add an isolated exercise that asks learners to select the covariance expression, validates a model-variable contract, and keeps a collapsed solution adjacent to the exercise.

- [ ] **Step 4: Add explicit prior, posterior, and prediction paths**

Before `pm.sample`, call `pm.sample_prior_predictive(draws=500, random_seed=SEED)`. Gate the four-chain posterior sampling cell with a run button. Add a `pm.Data` prediction grid and call `pm.sample_posterior_predictive(..., predictions=True, extend_inferencedata=True)` for out-of-sample predictions. Render observed data, original-unit predictions, and bracket-accessed diagnostics.

- [ ] **Step 5: Run focused checks**

Run: `pixi run pytest tests/test_model_contracts.py -k salmon -v`

Expected: PASS.

Run: `pixi run marimo check --strict notebooks/01_foundations.py`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add notebooks/01_foundations.py tests/test_model_contracts.py tests/test_notebook_smoke.py
git commit -m "feat: add GP foundations notebook"
```

### Task 5: Implement exact and latent GP instruction on salmon and coal data

**Files:**
- Create: `notebooks/02_exact_and_latent_gps.py`
- Modify: `tests/test_model_contracts.py`
- Modify: `tests/test_notebook_smoke.py`

**Interfaces:**
- Consumes: validated salmon and coal files.
- Produces: runnable `pm.gp.Marginal` and `pm.gp.Latent` examples with distinct likelihoods and posterior-predictive contracts.

- [ ] **Step 1: Write failing model tests**

```python
import pymc as pm

from tests.notebook_loader import load_notebook


def test_coal_model_is_latent_poisson_gp() -> None:
    lesson = load_notebook("02_exact_and_latent_gps.py")
    model, gp = lesson.build_coal_model()
    assert isinstance(gp, pm.gp.Latent)
    assert {"alpha", "eta", "ell", "f", "count"} <= set(model.named_vars)
    model.debug()


def test_coal_counts_stay_integer_observations() -> None:
    lesson = load_notebook("02_exact_and_latent_gps.py")
    years, counts = lesson.load_coal_data()
    assert years.shape == (111, 1)
    assert counts.dtype.kind in "iu"
    assert counts.min() == 0
    assert counts.max() == 6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run pytest tests/test_model_contracts.py -k coal -v`

Expected: FAIL because the second instructional notebook is absent.

- [ ] **Step 3: Implement `build_coal_model()` with the latent contract**

```python
with pm.Model(coords={"year": np.arange(len(counts)), "input": ["year_z"]}) as model:
    t_data = pm.Data("t", year_z, dims=("year", "input"))
    alpha = pm.Normal("alpha", 0, 1.5)
    eta = pm.HalfNormal("eta", 1)
    ell = pm.LogNormal("ell", 0, 1)
    cov = eta**2 * pm.gp.cov.Matern52(1, ls=ell)
    gp = pm.gp.Latent(mean_func=pm.gp.mean.Constant(c=alpha), cov_func=cov)
    f = gp.prior("f", X=t_data, dims="year", shape=len(counts))
    pm.Poisson("count", mu=pm.math.exp(f), observed=counts, dims="year")
```

Implement the salmon Marginal builder again inside Notebook 02 so the notebook remains standalone, but keep its current PyMC 6 covariance, `marginal_likelihood(..., sigma=sigma)`, prediction, and DataTree contracts identical to Task 4.

- [ ] **Step 4: Add lesson cells and a likelihood/link exercise**

Show why Gaussian observation noise permits `marginal_likelihood` while Poisson counts require the latent function. The exercise takes a learner-selected link/likelihood option, builds the chosen expression in an isolated model, and checks that positive rates and an integer observation distribution exist. The canonical solution is a collapsed adjacent cell.

- [ ] **Step 5: Add gated posterior workflow and count PPC**

For both models, add a prior-predictive cell before a run-button-gated posterior cell. Merge posterior predictive samples into the `DataTree`; use the coal posterior predictive count distribution to compare observed versus replicated dispersion. Display divergences, R-hat, ESS, and posterior rate trajectories with DataTree bracket access.

- [ ] **Step 6: Verify focused behavior**

Run: `pixi run pytest tests/test_model_contracts.py -k "salmon or coal" -v`

Expected: PASS.

Run: `pixi run marimo check --strict notebooks/02_exact_and_latent_gps.py`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add notebooks/02_exact_and_latent_gps.py tests/test_model_contracts.py tests/test_notebook_smoke.py
git commit -m "feat: add exact and latent GP notebook"
```

### Task 6: Implement kernels, two-dimensional inputs, and structural hierarchy

**Files:**
- Create: `notebooks/03_kernels_inputs_and_hierarchy.py`
- Modify: `tests/test_model_contracts.py`
- Modify: `tests/test_notebook_smoke.py`

**Interfaces:**
- Consumes: the deterministic 96-row electricity selection, 195-row Walker Lake table, and 30-row spin-rate subset.
- Produces: three model structures with real observed likelihoods and prior-predictive checks, but no posterior MCMC.

- [ ] **Step 1: Write failing structural tests**

```python
import pymc as pm

from tests.notebook_loader import load_notebook


def test_electricity_components_use_total_input_dimension() -> None:
    lesson = load_notebook("03_kernels_inputs_and_hierarchy.py")
    model, gp = lesson.build_electricity_model()
    assert isinstance(gp, pm.gp.Marginal)
    assert "demand" in model.named_vars
    model.debug()


def test_walker_model_has_two_anisotropic_lengthscales() -> None:
    lesson = load_notebook("03_kernels_inputs_and_hierarchy.py")
    model, gp = lesson.build_walker_model()
    assert isinstance(gp, pm.gp.Marginal)
    assert model.named_vars["ell"].ndim == 1
    assert int(model.dim_lengths["spatial_dim"].eval()) == 2
    model.debug()


def test_spin_model_has_shared_and_group_deviation_processes() -> None:
    lesson = load_notebook("03_kernels_inputs_and_hierarchy.py")
    model = lesson.build_spin_hierarchy_model()
    assert {"m", "delta", "spin_rate", "group_idx"} <= set(model.named_vars)
    model.debug()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run pytest tests/test_model_contracts.py -k "electricity or walker or spin" -v`

Expected: FAIL because the Hour 3 notebook is absent.

- [ ] **Step 3: Implement the additive electricity covariance exactly once**

```python
cov_long = eta_long**2 * pm.gp.cov.Matern52(
    input_dim=2, ls=ell_long, active_dims=[0]
)
cov_year = eta_year**2 * pm.gp.cov.Periodic(
    input_dim=2, ls=ell_year, period=1.0, active_dims=[1]
)
gp = pm.gp.Marginal(cov_func=cov_long + cov_year)
gp.marginal_likelihood("demand", X=X, y=demand_z, sigma=sigma)
```

Build `X` from standardized elapsed days and elapsed years divided by `365.2425`. Use exactly the fixed 96 data positions. Teach that both covariance constructors receive `input_dim=2`; `active_dims` selects their column.

- [ ] **Step 4: Implement the Walker Lake model and the hierarchy**

Build the Walker Lake model with named dimensions and the exact two-dimensional covariance:

```python
with pm.Model(coords={"obs": np.arange(len(v_z)), "spatial_dim": ["x_m", "y_m"]}) as model:
    x_data = pm.Data("X", x_xy, dims=("obs", "spatial_dim"))
    ell = pm.LogNormal("ell", 0, 1, dims="spatial_dim")
    eta = pm.HalfNormal("eta", 1)
    sigma = pm.HalfNormal("sigma", 1)
    cov = eta**2 * pm.gp.cov.Matern52(input_dim=2, ls=ell, active_dims=[0, 1])
    gp = pm.gp.Marginal(cov_func=cov)
    gp.marginal_likelihood("concentration", X=x_data, y=v_z, sigma=sigma)
```

For spin rate, use one shared latent GP `m` and three independent latent deviation GPs sharing `cov_delta`; stack them as `delta[group, obs]`, select with `group_idx`, and use a Normal observed likelihood:

```python
coords = {"group": pitcher_names, "obs": np.arange(n), "time": ["date_z"]}
with pm.Model(coords=coords) as model:
    t_data = pm.Data("t", time_z[:, None], dims=("obs", "time"))
    group_idx = pm.Data("group_idx", group_codes, dims="obs")
    eta_m = pm.HalfNormal("eta_m", 1)
    ell_m = pm.LogNormal("ell_m", 0, 1)
    eta_delta = pm.HalfNormal("eta_delta", 1)
    ell_delta = pm.LogNormal("ell_delta", 0, 1)
    sigma = pm.HalfNormal("sigma", 1)
    m = pm.gp.Latent(cov_func=eta_m**2 * pm.gp.cov.Matern52(1, ls=ell_m)).prior(
        "m", X=t_data, dims="obs", shape=n
    )
    delta_parts = [
        pm.gp.Latent(cov_func=eta_delta**2 * pm.gp.cov.Matern52(1, ls=ell_delta)).prior(
            f"delta_{group}", X=t_data, shape=n
        )
        for group in range(len(pitcher_names))
    ]
    delta = pm.Deterministic("delta", pt.stack(delta_parts), dims=("group", "obs"))
    mu = m + delta[group_idx, pt.arange(n)]
    pm.Normal("spin_rate", mu=mu, sigma=sigma, observed=spin_z, dims="obs")
```

Keep the three deviations conditionally independent; do not introduce `Coregion`, a factor-loading matrix, or a full posterior sampler.

- [ ] **Step 5: Build low-cost interactions only**

Use sliders/dropdowns only for component covariance/prior visualization. In the notebook, run `model.debug()` and `pm.sample_prior_predictive` for each model; do not call `pm.sample`. Add three isolated exercises: additive covariance selection, `active_dims`/two-dimensional lengthscale alignment, and group-index/partial-pooling interpretation.

- [ ] **Step 6: Verify the non-sampling lesson**

Run: `pixi run pytest tests/test_model_contracts.py -k "electricity or walker or spin" -v`

Expected: PASS.

Run: `pixi run marimo check --strict notebooks/03_kernels_inputs_and_hierarchy.py`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add notebooks/03_kernels_inputs_and_hierarchy.py tests/test_model_contracts.py tests/test_notebook_smoke.py
git commit -m "feat: add kernel and hierarchy notebook"
```

### Task 7: Implement the exact-versus-HSGP workflow notebook

**Files:**
- Create: `notebooks/04_workflow_and_hsgp.py`
- Modify: `tests/test_model_contracts.py`
- Modify: `tests/test_notebook_smoke.py`

**Interfaces:**
- Consumes: the 96-row fixed electricity comparator and all 1,826 validated electricity observations.
- Produces: a click-gated exact Marginal GP comparator and HSGP model with a fixed initial `m=[50], c=1.5`.

- [ ] **Step 1: Write failing HSGP model tests**

```python
import pymc as pm

from tests.notebook_loader import load_notebook


def test_exact_electricity_comparator_is_marginal() -> None:
    lesson = load_notebook("04_workflow_and_hsgp.py")
    model, gp = lesson.build_exact_electricity_model()
    assert isinstance(gp, pm.gp.Marginal)
    model.debug()


def test_hsgp_uses_course_baseline() -> None:
    lesson = load_notebook("04_workflow_and_hsgp.py")
    model, gp = lesson.build_hsgp_model(m=[50], c=1.5)
    assert isinstance(gp, pm.gp.HSGP)
    assert "demand" in model.named_vars
    model.debug()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run pytest tests/test_model_contracts.py -k hsgp -v`

Expected: FAIL because the Hour 4 notebook is absent.

- [ ] **Step 3: Implement the exact comparator and HSGP builders**

```python
with pm.Model(coords={"obs": np.arange(len(y_small)), "input": ["day_z"]}) as exact_model:
    x_data = pm.Data("x", x_small, dims=("obs", "input"))
    eta = pm.HalfNormal("eta", 1)
    ell = pm.LogNormal("ell", 0, 1)
    sigma = pm.HalfNormal("sigma", 1)
    cov = eta**2 * pm.gp.cov.Matern52(1, ls=ell)
    gp = pm.gp.Marginal(cov_func=cov)
    gp.marginal_likelihood("demand", X=x_data, y=y_small, sigma=sigma)
```

```python
with pm.Model(coords={"obs": np.arange(len(y_full)), "input": ["day_z"]}) as hsgp_model:
    x_data = pm.Data("x", x_full, dims=("obs", "input"))
    eta = pm.HalfNormal("eta", 1)
    ell = pm.LogNormal("ell", 0, 1)
    sigma = pm.HalfNormal("sigma", 1)
    cov = eta**2 * pm.gp.cov.Matern52(1, ls=ell)
    gp = pm.gp.HSGP(m=[50], c=1.5, cov_func=cov)
    f = gp.prior("f", X=x_data, dims="obs", shape=len(y_full))
    pm.Normal("demand", mu=f, sigma=sigma, observed=y_full, dims="obs")
```

Use `pm.Data` for both model inputs and declare exactly one of `c` or `L`. Expose `m` and `c` controls only after rendering the baseline prior implications. Prevent invalid values with a minimum `c` of `1.2`.

- [ ] **Step 4: Implement the required workflow and approximation exercise**

Present the exact cubic-cost explanation and an inducing-point overview without fitting a sparse approximation. For both exact and HSGP paths: generate prior predictive samples; gate 4-chain posterior sampling behind a run button; merge posterior predictions; then show DataTree diagnostics and original-unit predictive plots. The HSGP exercise asks learners to compare a proposed `m,c` choice against baseline boundary behavior, PPC, divergence, R-hat, and ESS—not against a hard-coded posterior value.

- [ ] **Step 5: Add a focused real-sampling smoke test**

```python
import pymc as pm

from tests.notebook_loader import load_notebook


def test_small_observed_hsgp_sampling_returns_required_groups() -> None:
    lesson = load_notebook("04_workflow_and_hsgp.py")
    model, _ = lesson.build_hsgp_model(m=[10], c=1.5, max_rows=40)
    with model:
        idata = pm.sample(
            chains=2,
            draws=20,
            tune=20,
            random_seed=20260709,
            progressbar=False,
        )
        idata.update(pm.sample_posterior_predictive(idata, progressbar=False))
    assert {"posterior", "sample_stats", "posterior_predictive"} <= set(idata.groups)
```

This is an execution test only. It must not assert R-hat, ESS, or posterior parameter values.

- [ ] **Step 6: Verify the notebook and focused tests**

Run: `pixi run pytest tests/test_model_contracts.py -k hsgp -v`

Expected: PASS.

Run: `pixi run marimo check --strict notebooks/04_workflow_and_hsgp.py`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add notebooks/04_workflow_and_hsgp.py tests/test_model_contracts.py tests/test_notebook_smoke.py
git commit -m "feat: add HSGP workflow notebook"
```

### Task 8: Validate the complete course and record release evidence

**Files:**
- Modify: `tests/test_data_contracts.py`
- Modify: `tests/test_model_contracts.py`
- Modify: `tests/test_notebook_smoke.py`
- Modify: `data/README.md`

**Interfaces:**
- Consumes: all notebooks, data assets, and the four-platform lock.
- Produces: repeatable release evidence and a final verified dataset manifest.

- [ ] **Step 1: Add the no-runtime-network test**

```python
from pathlib import Path


def test_notebooks_have_no_runtime_network_data_access() -> None:
    source = "\n".join(path.read_text() for path in Path("notebooks").glob("*.py"))
    forbidden = ["pm.get_data(", "requests.get(", "urllib.request", "http://", "https://"]
    assert not [token for token in forbidden if token in source]
```

- [ ] **Step 2: Run all fast checks**

Run: `pixi run check && pixi run marimo-check && pixi run pytest -m "not slow" -v`

Expected: PASS. The command must cover all data contracts, model-build contracts, notebook smoke tests, and the focused real-sampling smoke test.

- [ ] **Step 3: Rehearse all click-gated posterior fits on every supported platform**

For each of `win-64`, `linux-64`, `osx-64`, and `osx-arm64`:

1. Install from `pixi.lock`.
2. Run `00_environment_check.py`.
3. Run every click-gated posterior fit with the committed seed and 4 chains / 500 tune / 500 draws.
4. Record elapsed time and the divergence, R-hat, and ESS results in the corresponding `data/README.md` release-verification table.
5. Fail release on any budget or diagnostic breach; revise the specification before changing the model/sampler budget.

Use this table shape:

```markdown
| Platform | Notebook / fit | Elapsed seconds | Divergences | Max R-hat | Min bulk ESS | Min tail ESS | Pass |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
```

- [ ] **Step 4: Verify the attendee acceptance path**

On a clean supported platform, run only:

```bash
pixi install
pixi run python notebooks/00_environment_check.py
pixi run marimo check --strict notebooks
```

Open each notebook, execute all non-sampling cells, run each exercise checker, and verify that every posterior notebook exposes its run button instead of sampling automatically.

- [ ] **Step 5: Commit release artifacts**

```bash
git add data/README.md tests
git commit -m "test: verify GP course release contracts"
```
