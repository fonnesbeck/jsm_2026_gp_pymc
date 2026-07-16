# GP Workshop Notebooks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build four hour-aligned interactive marimo notebooks (plus a pre-course environment check) that teach Gaussian processes with PyMC, backed by five vendored real datasets, all runnable live within a ~2-minute-per-model budget.

**Architecture:** A pixi workspace pins the environment. A one-time `data/build_data.py` fetches each dataset from its authoritative source and writes frozen CSVs; pytest contract tests validate those CSVs. Five marimo notebooks read the frozen CSVs with polars, fit PyMC 6 GP models, and plot with plotly. Notebooks are self-contained (no shared runtime module) so learners see all code; the only shared assets are the `data/` CSVs.

**Tech Stack:** Python 3.13, PyMC 6.x, ArviZ 1.x, nutpie, PyTensor, polars, plotly, marimo, pytest; pixi for environment management.

## Global Constraints

- PyMC 6.x, ArviZ 1.x; ArviZ group access uses `idata["posterior"]` / `idata["sample_stats"]` / `idata["posterior_predictive"]` — never legacy attribute access.
- PyMC 6 selects the nutpie sampler automatically when installed — call `pm.sample(random_seed=42)` **without** a `nuts_sampler=` argument. Every `pm.sample` call passes an explicit integer seed (`random_seed=42`).
- Run a prior predictive check (`pm.sample_prior_predictive`) and plot it **before** every `pm.sample`; adjust priors if the range is implausible.
- Use non-centered parameterization for hierarchical group effects.
- GP teaching order is deliberately exact-first: HSGP is the production default (O(nm) vs O(n³)), but the course teaches `gp.Marginal`/`gp.Latent` (Hours 2–3) to build intuition, then motivates and applies HSGP at scale (Hour 4).
- Model comparison / LOO is a non-goal, so `pm.compute_log_likelihood` is not required. Persisting `idata` to NetCDF is taught once (Hour 4 workflow section).
- **Execution time is NOT a design constraint (revised 2026-07-16).** Do not subset data, thin sampling, or cut content to make cells run faster; choose model/data sizes for pedagogical value. A fit taking a few minutes is fine. Smoke-test timeouts are generous upper bounds (e.g. 900s), not budgets to optimize toward. Standard sampling is `draws=1000, tune=1000` (or more where a model needs it for ESS); do not trim draws for speed.
- **Content density is a first-class requirement (revised 2026-07-16).** Each teaching notebook must fill a ~55–60 min slot: target ≥ ~2,800 markdown words and ≥ ~55 cells, with several worked examples and multiple `mo.accordion` exercises per major section, and rich narrative/intuition/derivation. Thin notebooks are a defect. Reference density: instats 90-min sessions run 3,200–7,500 md words, 57–91 cells. Err rich.
- No learner notebook fetches data at runtime; all data is read from vendored CSVs in `data/`.
- Data/plotting stack is polars + plotly. ArviZ is used for diagnostics.
- Each notebook introduces its dataset with a background cell (what it is, source, why it is a good GP example, units, caveats).
- Standardization uses `z(a) = (a - a.mean()) / a.std(ddof=0)`; original-unit values are retained for plots.
- Exercises appear as short "your turn" cells with solutions hidden in `mo.accordion`; interactive prior-draw controls use `mo.ui` sliders.
- pixi platforms: `win-64`, `linux-64`, `osx-64`, `osx-arm64`.
- Notebooks are marimo apps generated with `marimo.App`; standard seed `RANDOM_SEED = 42`, `RNG = np.random.default_rng(RANDOM_SEED)`.

---

## File Structure

- `pixi.toml`, `pixi.lock` — pinned environment (4 platforms).
- `data/build_data.py` — rebuilds every vendored CSV from source (maintainer-run).
- `data/theophylline.csv`, `data/coal_disasters.csv`, `data/noaa_tides_hourly.csv`, `data/places_diabetes.csv`, `data/fastball_spin_rates.csv` — frozen data.
- `data/README.md` — provenance, license, transformations per file.
- `tests/test_data_contracts.py` — schema / row-count / null / range assertions on the frozen CSVs.
- `tests/test_notebook_smoke.py` — headless execution + wall-clock budget check for each notebook.
- `notebooks/00_environment_check.py` — verify env + data, compile a trivial model (no sampling).
- `notebooks/01_foundations.py` — PyMC primer + GP concepts (Theophylline).
- `notebooks/02_marginal_latent_gps.py` — marginal (Theophylline) + latent Poisson (coal).
- `notebooks/03_kernels_and_hierarchy.py` — kernel zoo, additive/mult (NOAA), 2D spatial (PLACES), hierarchical (spin rates).
- `notebooks/04_scaling_and_workflow.py` — O(n³) demo, sparse, HSGP (full NOAA), full workflow.

---

## Task 1: Pixi environment and repository scaffolding

**Files:**
- Create: `pixi.toml`
- Create: `data/README.md` (stub, filled per dataset in later tasks)
- Create: `tests/__init__.py` (empty)
- Create: `.gitignore` additions if needed (already ignores `.worktrees`)

**Interfaces:**
- Produces: a resolved pixi environment named `default` exposing `python`, `pymc`, `arviz`, `nutpie`, `polars`, `plotly`, `marimo`, `pytest`, `requests`; a `pixi.lock` covering all four platforms.

- [ ] **Step 1: Write `pixi.toml`**

```toml
[workspace]
authors = ["Chris Fonnesbeck <fonnesbeck@gmail.com>"]
channels = ["conda-forge"]
name = "jsm_2026_gp_pymc"
platforms = ["win-64", "linux-64", "osx-64", "osx-arm64"]
version = "0.1.0"

[tasks]
check = "python -m pytest tests/ -v"

[dependencies]
python = "3.13.*"
pymc = ">=6,<7"
arviz = ">=1,<2"
nutpie = "*"
pytensor = "*"
numba = "*"
polars = ">=1,<2"
plotly = ">=6,<7"
marimo = ">=0.17,<1"
ipywidgets = "*"
watchdog = "*"
scikit-learn = "*"
pytest = "*"
requests = "*"
pip = "*"
```

- [ ] **Step 2: Solve and install the environment**

Run: `cd /var/home/fonnesbeck/repos/jsm_2026_gp_pymc && pixi install`
Expected: solve succeeds, `pixi.lock` is written, exit code 0.

If PyMC 6 / ArviZ 1 are not yet on conda-forge for all four platforms, relax the ArviZ/PyMC floors to the latest available `>=5.26` line and record the substitution in `data/README.md` under an "Environment notes" heading. Do not proceed with a partial solve.

- [ ] **Step 3: Verify the GP API names against the installed PyMC**

Run:
```bash
pixi run python -c "import pymc as pm; print(pm.__version__); [print(n) for n in ['Marginal','Latent','HSGP','HSGPPeriodic','MarginalApprox'] if hasattr(pm.gp, n)]"
```
Expected: prints the version and each of `Marginal`, `Latent`, `HSGP`, `HSGPPeriodic`, `MarginalApprox` that exists. Record any missing name in `data/README.md` "Environment notes"; later tasks that reference a missing name must use the installed equivalent.

- [ ] **Step 4: Create `data/README.md` stub**

```markdown
# Datasets

All files here are frozen snapshots rebuilt by `build_data.py`. No workshop
notebook fetches data at runtime. Each entry records source, access date,
license, units, and transformations.

## Environment notes

(record any dependency-version substitutions here)

<!-- per-dataset sections added by later tasks -->
```

- [ ] **Step 5: Create empty `tests/__init__.py`**

Run: `touch tests/__init__.py`

- [ ] **Step 6: Commit**

```bash
git add pixi.toml pixi.lock data/README.md tests/__init__.py
git commit -m "chore: add pixi environment and repo scaffolding"
```

---

## Task 2: Data infrastructure + Theophylline + coal disasters

**Files:**
- Create: `data/build_data.py`
- Create: `data/theophylline.csv`
- Create: `data/coal_disasters.csv`
- Create: `tests/test_data_contracts.py`
- Modify: `data/README.md` (add Theophylline and coal sections)

**Interfaces:**
- Produces: `data/theophylline.csv` with columns `subject` (int), `time` (float, hours), `conc` (float, mg/L), `dose` (float), `weight` (float); 132 rows. `data/coal_disasters.csv` with columns `year` (int), `disasters` (int); one row per year 1851–1962 (112 rows).
- Produces: `build_data.py` exposing `build_theophylline() -> None` and `build_coal_disasters() -> None` writing the CSVs; a `main()` that runs every builder.

- [ ] **Step 1: Write the failing contract tests**

```python
# tests/test_data_contracts.py
from pathlib import Path
import polars as pl

DATA = Path(__file__).resolve().parents[1] / "data"

def test_theophylline_contract():
    df = pl.read_csv(DATA / "theophylline.csv")
    assert df.columns == ["subject", "time", "conc", "dose", "weight"]
    assert df.height == 132
    assert df["subject"].n_unique() == 12
    assert df.null_count().sum_horizontal().item() == 0
    assert df["conc"].min() >= 0.0
    assert df["time"].min() >= 0.0

def test_coal_disasters_contract():
    df = pl.read_csv(DATA / "coal_disasters.csv")
    assert df.columns == ["year", "disasters"]
    assert df.height == 112
    assert df["year"].min() == 1851
    assert df["year"].max() == 1962
    assert df["disasters"].min() >= 0
    assert df.null_count().sum_horizontal().item() == 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run pytest tests/test_data_contracts.py -v`
Expected: FAIL (files do not exist).

- [ ] **Step 3: Write `build_data.py` for these two datasets**

Theophylline is R's built-in `Theoph`. Rather than depend on R, vendor the values. The `Theoph` dataset is small and public; fetch it from a stable mirror and normalize columns.

```python
# data/build_data.py
"""Rebuild every vendored dataset from its authoritative source.

Run once by the maintainer:  pixi run python data/build_data.py
Learner notebooks never call this; they read the frozen CSVs.
"""
from __future__ import annotations

import io
from pathlib import Path

import polars as pl
import requests

DATA = Path(__file__).resolve().parent
TIMEOUT = 60


def build_theophylline() -> None:
    # R's Theoph dataset, mirrored as CSV by the vincentarelbundock Rdatasets project.
    url = "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/Theoph.csv"
    raw = requests.get(url, timeout=TIMEOUT).text
    df = pl.read_csv(io.StringIO(raw))
    # Columns: rownames, Subject, Wt, Dose, Time, conc
    out = (
        df.select(
            pl.col("Subject").cast(pl.Int64).alias("subject"),
            pl.col("Time").cast(pl.Float64).alias("time"),
            pl.col("conc").cast(pl.Float64),
            pl.col("Dose").cast(pl.Float64).alias("dose"),
            pl.col("Wt").cast(pl.Float64).alias("weight"),
        )
        .sort(["subject", "time"])
    )
    assert out.height == 132, out.height
    out.write_csv(DATA / "theophylline.csv")


# Classic Jarrett (1979) British coal-mining disaster counts, 1851-1962.
_COAL = [
    4, 5, 4, 1, 0, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6, 3, 3, 5, 4, 5, 3, 1, 4, 4,
    1, 5, 5, 3, 4, 2, 5, 2, 2, 3, 4, 2, 1, 3, 2, 2, 1, 1, 1, 1, 3, 0, 0, 1, 0,
    1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0,
    0, 1, 1, 0, 2, 3, 3, 1, 1, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4, 0, 0, 0, 1,
    0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0,
]


def build_coal_disasters() -> None:
    years = list(range(1851, 1851 + len(_COAL)))
    df = pl.DataFrame({"year": years, "disasters": _COAL})
    assert df.height == 112, df.height
    df.write_csv(DATA / "coal_disasters.csv")


def main() -> None:
    build_theophylline()
    build_coal_disasters()
    build_noaa_tides()      # Task 3
    build_places_diabetes()  # Task 4
    build_spin_rates()       # Task 5


if __name__ == "__main__":
    main()
```

Note: leave the Task 3–5 calls in `main()` commented out until those builders exist, or define them as stubs raising `NotImplementedError`. For this task, temporarily comment the three later calls so `main()` runs.

- [ ] **Step 4: Run the two builders**

Run: `pixi run python -c "from data.build_data import build_theophylline, build_coal_disasters; build_theophylline(); build_coal_disasters()"`
Expected: writes `data/theophylline.csv` (132 rows) and `data/coal_disasters.csv` (112 rows), exit code 0.

If the Rdatasets URL is unreachable, fall back to any stable `Theoph` CSV mirror (nlme package data) and record the URL used in `data/README.md`; the column normalization is unchanged.

- [ ] **Step 5: Run the contract tests to verify they pass**

Run: `pixi run pytest tests/test_data_contracts.py -v`
Expected: both tests PASS.

- [ ] **Step 6: Document provenance in `data/README.md`**

Append sections for Theophylline (source URL, access date, R `datasets::Theoph`, units: time in hours, conc in mg/L; one oral dose per subject) and coal disasters (Jarrett 1979 counts of British coal-mining disasters ≥10 deaths, 1851–1962, annual).

- [ ] **Step 7: Commit**

```bash
git add data/build_data.py data/theophylline.csv data/coal_disasters.csv tests/test_data_contracts.py data/README.md
git commit -m "feat: add Theophylline and coal-disaster datasets with contract tests"
```

---

## Task 3: NOAA tide-gauge dataset

**Files:**
- Modify: `data/build_data.py` (add `build_noaa_tides`)
- Create: `data/noaa_tides_hourly.csv`
- Modify: `tests/test_data_contracts.py` (add NOAA test)
- Modify: `data/README.md` (add NOAA section)

**Interfaces:**
- Consumes: `requests`, `DATA`, `TIMEOUT` from `build_data.py`.
- Produces: `data/noaa_tides_hourly.csv` with columns `time` (ISO 8601 string), `water_level` (float, meters, MLLW); one full year of hourly `hourly_height` observations (~8,760 rows) from a single mixed-tide station.

- [ ] **Step 1: Write the failing contract test**

```python
def test_noaa_tides_contract():
    df = pl.read_csv(DATA / "noaa_tides_hourly.csv")
    assert df.columns == ["time", "water_level"]
    assert df.height >= 8000        # ~1 year of hourly obs, allowing small gaps
    assert df.height <= 9000
    assert df["water_level"].is_finite().all()
    # two-week exact-fit slice used in Hour 3 must be small enough to fit live
    assert df.head(300).height == 300
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run pytest tests/test_data_contracts.py::test_noaa_tides_contract -v`
Expected: FAIL (file missing).

- [ ] **Step 3: Implement `build_noaa_tides`**

Use NOAA CO-OPS `datagetter`. Default station `9414290` (San Francisco, CA — long record, mixed semidiurnal showing both diurnal and semidiurnal components). Fetch one calendar year of `hourly_height`, datum MLLW, metric units, GMT.

```python
def build_noaa_tides(station: str = "9414290", year: int = 2019) -> None:
    url = (
        "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
        f"?begin_date={year}0101&end_date={year}1231"
        f"&station={station}&product=hourly_height&datum=MLLW"
        "&time_zone=gmt&units=metric&format=json&application=jsm2026_gp"
    )
    payload = requests.get(url, timeout=TIMEOUT).json()
    rows = payload["data"]
    df = (
        pl.DataFrame({"time": [r["t"] for r in rows],
                      "water_level": [r["v"] for r in rows]})
        .with_columns(pl.col("water_level").cast(pl.Float64, strict=False))
        .drop_nulls("water_level")
        .sort("time")
    )
    assert df.height >= 8000, df.height
    df.write_csv(DATA / "noaa_tides_hourly.csv")
```

- [ ] **Step 4: Run the builder**

Run: `pixi run python -c "from data.build_data import build_noaa_tides; build_noaa_tides()"`
Expected: writes `data/noaa_tides_hourly.csv` (~8,760 rows), exit code 0.

If station `9414290` returns sparse/empty data for 2019, try `9447130` (Seattle) or another year and record the choice in `data/README.md`.

- [ ] **Step 5: Verify the exact-fit slice samples within budget**

Run:
```bash
pixi run python -c "
import polars as pl, numpy as np, pymc as pm, time
from pathlib import Path
df = pl.read_csv('data/noaa_tides_hourly.csv').head(300)
t = ((np.arange(df.height) - np.arange(df.height).mean())/np.arange(df.height).std())[:,None]
y = ((df['water_level'] - df['water_level'].mean())/df['water_level'].std()).to_numpy()
with pm.Model() as m:
    l = pm.LogNormal('l',0,1); eta = pm.HalfNormal('eta',1)
    cov = eta**2 * pm.gp.cov.Matern52(1, ls=l)
    gp = pm.gp.Marginal(cov_func=cov)
    sig = pm.HalfNormal('sig',1)
    gp.marginal_likelihood('y', X=t, y=y, sigma=sig)
    s=time.time(); idata=pm.sample(500, tune=500, chains=2, random_seed=42, progressbar=False)
print('elapsed', time.time()-s)
"
```
Expected: completes in well under 2 minutes; prints elapsed seconds. If it exceeds ~90s, reduce the slice size constant used in Hour 3 (Task 9) accordingly and update the contract test's `head(300)` bound to match.

- [ ] **Step 6: Uncomment the `build_noaa_tides()` call in `main()`.**

- [ ] **Step 7: Run the contract test to verify it passes**

Run: `pixi run pytest tests/test_data_contracts.py::test_noaa_tides_contract -v`
Expected: PASS.

- [ ] **Step 8: Document provenance and commit**

Add a NOAA section to `data/README.md` (station id + name, product `hourly_height`, datum MLLW, year, GMT, units meters, NOAA CO-OPS terms of use, access date, note that hourly data may contain small gaps).

```bash
git add data/build_data.py data/noaa_tides_hourly.csv tests/test_data_contracts.py data/README.md
git commit -m "feat: add NOAA tide-gauge dataset with contract test"
```

---

## Task 4: CDC PLACES county diabetes dataset

**Files:**
- Modify: `data/build_data.py` (add `build_places_diabetes`)
- Create: `data/places_diabetes.csv`
- Modify: `tests/test_data_contracts.py` (add PLACES test)
- Modify: `data/README.md` (add PLACES section)

**Interfaces:**
- Produces: `data/places_diabetes.csv` with columns `county` (str), `lon` (float), `lat` (float), `diabetes_pct` (float), `obesity_pct` (float); one row per county in a single state (default North Carolina, 100 counties). `diabetes_pct`/`obesity_pct` are crude prevalence percentages; `lon`/`lat` are county centroids.

- [ ] **Step 1: Write the failing contract test**

```python
def test_places_diabetes_contract():
    df = pl.read_csv(DATA / "places_diabetes.csv")
    assert df.columns == ["county", "lon", "lat", "diabetes_pct", "obesity_pct"]
    assert 90 <= df.height <= 260          # one state's counties
    assert df["diabetes_pct"].is_finite().all()
    assert df["diabetes_pct"].min() > 0
    assert df["diabetes_pct"].max() < 40
    assert df["lat"].min() > 20 and df["lat"].max() < 55
    assert df.null_count().sum_horizontal().item() == 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run pytest tests/test_data_contracts.py::test_places_diabetes_contract -v`
Expected: FAIL.

- [ ] **Step 3: Implement `build_places_diabetes`**

PLACES county data is served via the Socrata API on data.cdc.gov. Pull the county-level release, filter to one state and the Diabetes + Obesity measures (crude prevalence), pivot to one row per county, and parse the `geolocation` centroid.

```python
def build_places_diabetes(state: str = "North Carolina") -> None:
    # PLACES county-level current release (Socrata dataset id "swc5-untb").
    base = "https://data.cdc.gov/resource/swc5-untb.json"
    params = (
        f"?$where=statedesc='{state}'"
        "&$select=locationname,latitude,longitude,measureid,data_value,data_value_type"
        "&$limit=50000"
    )
    rows = requests.get(base + params.replace(" ", "%20"), timeout=TIMEOUT).json()
    raw = pl.DataFrame(rows)
    crude = raw.filter(pl.col("data_value_type") == "Crude prevalence")
    wide = (
        crude.filter(pl.col("measureid").is_in(["DIABETES", "OBESITY"]))
        .with_columns(pl.col("data_value").cast(pl.Float64, strict=False))
        .pivot(values="data_value", index=["locationname", "latitude", "longitude"],
               on="measureid", aggregate_function="first")
        .rename({"locationname": "county", "latitude": "lat",
                 "longitude": "lon", "DIABETES": "diabetes_pct",
                 "OBESITY": "obesity_pct"})
        .with_columns(pl.col("lat").cast(pl.Float64), pl.col("lon").cast(pl.Float64))
        .select("county", "lon", "lat", "diabetes_pct", "obesity_pct")
        .drop_nulls()
        .sort("county")
    )
    assert 90 <= wide.height <= 260, wide.height
    wide.write_csv(DATA / "places_diabetes.csv")
```

- [ ] **Step 4: Run the builder**

Run: `pixi run python -c "from data.build_data import build_places_diabetes; build_places_diabetes()"`
Expected: writes `data/places_diabetes.csv` (~100 rows for NC), exit code 0.

If the Socrata dataset id or field names differ in the current release, inspect one record via `requests.get(base + "?$limit=1").json()` and adjust field names; record the release id and access date in `data/README.md`. The measure ids `DIABETES`/`OBESITY` and type `Crude prevalence` are the selection contract.

- [ ] **Step 5: Uncomment the `build_places_diabetes()` call in `main()`.**

- [ ] **Step 6: Run the contract test to verify it passes**

Run: `pixi run pytest tests/test_data_contracts.py::test_places_diabetes_contract -v`
Expected: PASS.

- [ ] **Step 7: Document provenance and commit**

Add a PLACES section to `data/README.md`: source (CDC PLACES, data.cdc.gov, Socrata id, release year, access date), the state used, measures (diagnosed diabetes and obesity, crude prevalence, %), centroids are `latitude`/`longitude` fields, and the caveat that PLACES values are **model-based small-area estimates**, not raw county observations.

```bash
git add data/build_data.py data/places_diabetes.csv tests/test_data_contracts.py data/README.md
git commit -m "feat: add CDC PLACES county diabetes dataset with contract test"
```

---

## Task 5: Baseball spin-rate dataset

**Files:**
- Modify: `data/build_data.py` (add `build_spin_rates`)
- Create: `data/fastball_spin_rates.csv`
- Modify: `tests/test_data_contracts.py` (add spin-rate test)
- Modify: `data/README.md` (add spin-rate section)

**Interfaces:**
- Produces: `data/fastball_spin_rates.csv` with columns `pitcher` (str), `game_date` (str, ISO date), `spin_rate` (float, rpm), `n_pitches` (int); exactly 3 pitchers × 10 games = 30 rows, alphabetical pitcher order, date-sorted within pitcher.

- [ ] **Step 1: Write the failing contract test**

```python
def test_spin_rates_contract():
    df = pl.read_csv(DATA / "fastball_spin_rates.csv")
    assert df.columns == ["pitcher", "game_date", "spin_rate", "n_pitches"]
    assert df.height == 30
    assert df["pitcher"].n_unique() == 3
    counts = df.group_by("pitcher").len()["len"].to_list()
    assert all(c == 10 for c in counts)
    assert df["n_pitches"].min() >= 10
    assert df.null_count().sum_horizontal().item() == 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run pytest tests/test_data_contracts.py::test_spin_rates_contract -v`
Expected: FAIL.

- [ ] **Step 3: Implement `build_spin_rates`**

Source is the existing instats file. Curate deterministically to 3 pitchers × 10 games.

```python
def build_spin_rates() -> None:
    src = Path("/var/home/fonnesbeck/repos/instats_gp/data/fastball_spin_rates.csv")
    df = (
        pl.read_csv(src)
        .rename({"pitcher_name": "pitcher", "avg_spin_rate": "spin_rate"})
        .drop_nulls(["pitcher", "game_date", "spin_rate"])
        .filter(pl.col("n_pitches") >= 10)
        .unique(["pitcher", "game_date"])
    )
    # rank pitchers by count desc then name asc; keep 3 with >=10 games
    counts = df.group_by("pitcher").len().sort(["len", "pitcher"], descending=[True, False])
    keep = [p for p in counts["pitcher"].to_list()
            if df.filter(pl.col("pitcher") == p).height >= 10][:3]
    keep = sorted(keep)
    out = (
        df.filter(pl.col("pitcher").is_in(keep))
        .sort(["pitcher", "game_date"])
        .group_by("pitcher", maintain_order=True)
        .head(10)
        .select("pitcher", "game_date", "spin_rate", "n_pitches")
    )
    assert out.height == 30, out.height
    out.write_csv(DATA / "fastball_spin_rates.csv")
```

- [ ] **Step 4: Run the builder**

Run: `pixi run python -c "from data.build_data import build_spin_rates; build_spin_rates()"`
Expected: writes `data/fastball_spin_rates.csv` (30 rows), exit code 0.

- [ ] **Step 5: Uncomment the `build_spin_rates()` call in `main()`; run the full build to confirm reproducibility**

Run: `pixi run python data/build_data.py`
Expected: rebuilds all five CSVs, exit code 0.

- [ ] **Step 6: Run the full contract suite**

Run: `pixi run pytest tests/test_data_contracts.py -v`
Expected: all five tests PASS.

- [ ] **Step 7: Document provenance and commit**

Add a spin-rate section to `data/README.md` (derived from instats_gp fastball spin rates; MLB Statcast 2021 fastball average spin rate per pitcher per game; curated to 3 pitchers × 10 games; units rpm).

```bash
git add data/build_data.py data/fastball_spin_rates.csv tests/test_data_contracts.py data/README.md
git commit -m "feat: add curated baseball spin-rate dataset with contract test"
```

---

## Task 6: Environment-check notebook + notebook smoke-test harness

**Files:**
- Create: `notebooks/00_environment_check.py`
- Create: `tests/test_notebook_smoke.py`

**Interfaces:**
- Consumes: all five `data/*.csv` files; the installed PyMC/ArviZ/marimo/polars/plotly.
- Produces: a headless-executable marimo notebook that verifies versions, loads each CSV, and compiles (does not sample) a trivial GP model. A reusable smoke-test helper `run_notebook(path, timeout_s)` that executes a marimo notebook headlessly and returns `(elapsed_seconds, ok)`.

- [ ] **Step 1: Write the smoke-test harness with the env-check test**

```python
# tests/test_notebook_smoke.py
import subprocess, time
from pathlib import Path
import pytest

NB = Path(__file__).resolve().parents[1] / "notebooks"

def run_notebook(path: Path, timeout_s: int) -> float:
    """Execute a marimo notebook headlessly; return elapsed seconds. Raise on error."""
    start = time.time()
    proc = subprocess.run(
        ["marimo", "export", "html", str(path), "--no-include-code", "-o", "/dev/null"],
        capture_output=True, text=True, timeout=timeout_s,
    )
    if proc.returncode != 0:
        raise AssertionError(f"{path.name} failed:\n{proc.stderr[-2000:]}")
    return time.time() - start

def test_00_environment_check():
    run_notebook(NB / "00_environment_check.py", timeout_s=120)
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run pytest tests/test_notebook_smoke.py::test_00_environment_check -v`
Expected: FAIL (notebook missing). Confirm `marimo export html --help` lists the flags used; if `--no-include-code` differs in the installed marimo, adjust the command and re-run.

- [ ] **Step 3: Write `notebooks/00_environment_check.py`**

A marimo app: a markdown intro cell; a cell importing pymc/arviz/marimo/polars/plotly and printing versions; a cell loading each of the five CSVs and asserting non-empty; a cell that builds a tiny `pm.gp.Marginal` model and calls `pm.model_to_graphviz` or compiles the logp **without sampling**; a final cell printing "Environment OK". Use the standard marimo skeleton (see Session 7 header) with `mo`, and read CSVs via `pl.read_csv(Path("data")/name)` resolved relative to the notebook.

- [ ] **Step 4: Run the notebook test to verify it passes**

Run: `pixi run pytest tests/test_notebook_smoke.py::test_00_environment_check -v`
Expected: PASS in under 120s.

- [ ] **Step 5: Commit**

```bash
git add notebooks/00_environment_check.py tests/test_notebook_smoke.py
git commit -m "feat: add environment-check notebook and smoke-test harness"
```

---

## Task 7: `01_foundations.py` — PyMC primer + GP concepts

**Files:**
- Create: `notebooks/01_foundations.py`
- Modify: `tests/test_notebook_smoke.py` (add test)

**Interfaces:**
- Consumes: `data/theophylline.csv`; `run_notebook` from Task 6.
- Produces: a taught notebook that runs headlessly without error (no time budget). **Density target: ≥ ~2,800 markdown words, ≥ ~55 cells**, with multiple worked examples and several `mo.accordion` exercises per part. This is the flagship first hour — build it out richly, not as a terse script.

> **This task was expanded on 2026-07-16.** The prior version used a single straight-line baseline and was far too thin (~1,266 md words, 30 cells). The rebuild uses a **piecewise-linear** baseline and roughly doubles the content. Execution time is NOT a constraint — do not thin sampling or content for speed. Standard sampling is `draws=1000, tune=1000, chains=2` (or more if a fit needs it for ESS).

- [ ] **Step 1: Add the failing smoke test** (generous timeout — execution time is not gated)

```python
def test_01_foundations():
    run_notebook(NB / "01_foundations.py", timeout_s=900)
```

Run: `pixi run pytest tests/test_notebook_smoke.py::test_01_foundations -v` → Expected: FAIL (missing).

- [ ] **Step 2: Part A — Bayesian workflow & PyMC primer (build richly)**

Cells (each markdown cell substantial, not one-liners):
- Title + an "in this notebook" overview markdown.
- Imports + seed + PYMC brand colors + `z()` helper (match `00`/house style).
- **Motivation markdown**: the modeling problem (unknown functional form), parametric vs nonparametric thinking, why an hour of foundations precedes GPs.
- **The Bayesian paradigm markdown**: prior, likelihood, posterior with real intuition, Bayes' rule stated, and a small explanatory figure.
- **Warm-up PyMC model**: estimate a single quantity (e.g. the mean & scale of one subject's concentrations) in `pm.Model` — the smallest model — to introduce the context manager, priors, likelihood, `pm.sample(random_seed=42)`, the returned DataTree, `az.summary` via `idata["posterior"]`, and a trace/rank plot. Prior predictive check plotted and explained.
- **Theophylline background cell** (R `Theoph`, one oral dose per subject, rise-peak-decay, units) + EDA plotting several subjects with plotly.
- **Piecewise-linear baseline**: fit a breakpoint/hinge model to one subject — rising slope then falling slope meeting at an estimated peak time `tau`. Priors on both slopes, the level, `tau` (e.g. a Uniform/Normal over the observed time range), and noise; build `mu` with `pm.math.where`/`pytensor` switch or a hinge basis. Prior predictive plotted; `pm.sample(random_seed=42)`; diagnostics (divergences, r_hat, ess); posterior fit plotted with an HDI band via `az.hdi`.
- **Diagnosis-of-inadequacy markdown + plot**: the kink is unphysical for a smooth PK curve, `tau` is poorly identified (show its posterior width), and straight segments miss the curvature — motivating a flexible function with no hand-specified form.
- 2–3 `mo.accordion` exercises (e.g. change a prior and re-read the prior predictive; widen/narrow the `tau` prior and inspect identifiability; interpret an ESS/r_hat readout).

- [ ] **Step 3: Part B — GP concepts (build richly)**

Cells:
- **From MVN to GP markdown**: the multivariate normal; then marginalization and conditioning — each stated with the formula AND a worked bivariate numeric example AND a plotly visual (e.g. conditioning a 2D Gaussian, showing the conditional slice). Build to the GP definition (distribution over functions / infinite-dimensional Gaussian).
- **ExpQuad from scratch exercise** (`mo.accordion` solution: `eta**2 * exp(-0.5*dist**2/ls**2)`), then show the Gram matrix as a heatmap and draw several sample functions from the GP prior on a grid (with jitter), plotted.
- **Mean & covariance functions** markdown; what each hyperparameter controls.
- **GP regression = conditioning**: a small worked example conditioning the GP prior on a handful of points to get the posterior over functions; plot posterior mean + band; connect explicitly back to the piecewise-linear failure ("the GP is the flexible function that model couldn't be").
- **Widget**: `mo.ui.slider` for lengthscale and one for amplitude (with defaults) driving a reactive cell that redraws GP prior samples; a "predict before you move it" prompt.
- 2–3 more `mo.accordion` exercises (vary hyperparameters; condition on an added point; reason about smoothness).

- [ ] **Step 4: Check density, then run the smoke test**

Count: confirm ≥ ~2,800 markdown words and ≥ ~55 `@app.cell`s (the density target). If short, add more worked examples/intuition/exercises — do not pad with filler.
Run: `pixi run pytest tests/test_notebook_smoke.py::test_01_foundations -v` → Expected: PASS (any runtime under the 900s ceiling is fine).

- [ ] **Step 5: Convention self-check**

`grep -n nuts_sampler notebooks/01_foundations.py` → only prose matches. Confirm: a prior predictive precedes EACH `pm.sample`; each fit shows divergences + `az.summary` r_hat/ess; DataTree bracket access; GP inputs 2D; HDI bands via `az.hdi`; sliders have defaults; exercises in `mo.accordion`.

- [ ] **Step 6: Commit**

```bash
git add notebooks/01_foundations.py tests/test_notebook_smoke.py
git commit -m "feat: expand foundations notebook (piecewise-linear baseline, fuller GP concepts)"
```

---

## Task 8: `02_marginal_latent_gps.py` — marginal + latent GPs

**Files:**
- Create: `notebooks/02_marginal_latent_gps.py`
- Modify: `tests/test_notebook_smoke.py` (add test)

**Interfaces:**
- Consumes: `data/theophylline.csv`, `data/coal_disasters.csv`; `run_notebook`.
- Produces: a taught notebook that runs headlessly without error (no time budget). **Density target: ≥ ~2,800 md words, ≥ ~55 cells.**

> **Revised 2026-07-16 — expand to the density target and remove the execution-time budget.** The section outline below is the minimum skeleton; build each part out richly (motivation → intuition → worked example → multiple `mo.accordion` exercises), matching notebook 01's depth. Do NOT subset data or thin sampling for speed; standard sampling `draws=1000, tune=1000`. The full content spec will be finalized when this notebook is rebuilt (after notebook 01 is approved as the template).

- [ ] **Step 1: Add the failing smoke test**

```python
def test_02_marginal_latent():
    run_notebook(NB / "02_marginal_latent_gps.py", timeout_s=900)
```

Run it → Expected: FAIL (missing).

- [ ] **Step 2: Build Part A — Marginal GP on Theophylline**

Cells: recap markdown; load Theophylline, select one subject, standardize `time`→X `(n,1)` and `conc`→y; plot; markdown + a naive `pm.gp.Marginal` with `Matern52` on **raw standardized time** showing the fit struggling with the rise-peak-decay; markdown motivating an input transform (`log1p(time)`) and/or a mean function; refit `pm.gp.Marginal(mean_func=..., cov_func=eta**2*Matern52(1, ls=l))` with `.marginal_likelihood("y", X=X, y=y, sigma=sigma)`; MAP via `pm.find_MAP` contrasted with `pm.sample(random_seed=42)`; `.conditional("f_pred", Xnew)` with and without `pred_noise`, plotted with HDI; **exercise**: extrapolate beyond observed time and diagnose (accordion solution).

- [ ] **Step 3: Build Part B — Latent Poisson GP on coal disasters**

Cells: **background cell** (coal disasters, annual counts 1851–1962, non-Gaussian); load CSV, standardize `year`→t `(n,1)`, keep integer counts; markdown on why counts can't be marginalized; `gp = pm.gp.Latent(cov_func=eta**2*Matern52(1, ls=l))`, `f = gp.prior("f", X=t)`, `pm.Poisson("y", mu=pm.math.exp(f), observed=counts)`; prior predictive check first; `pm.sample(random_seed=42, target_accept=0.9)`; plot posterior rate `exp(f)` trajectory with HDI; posterior-predictive count check via `pm.sample_posterior_predictive`; **exercise**: change the lengthscale prior and compare (accordion). Use standard `draws=1000, tune=1000, chains=2` — 112 latent values sample in ~1–2 min; confirm `ess_bulk`/`ess_tail` > 400 and zero divergences before interpreting.

- [ ] **Step 4: Run the smoke test to verify it passes**

Run it → Expected: PASS ≤ 180s. If sampling exceeds budget, first prefer a smaller Theophylline subject / fewer coal years or a coarser prediction grid; trim draws only as a last resort and re-verify `ess_bulk`/`ess_tail` > 400 and zero divergences.

- [ ] **Step 5: Interactive eyeball, then commit**

```bash
git add notebooks/02_marginal_latent_gps.py tests/test_notebook_smoke.py
git commit -m "feat: add marginal and latent GP notebook"
```

---

## Task 9: `03_kernels_and_hierarchy.py` — kernels, 2D inputs, hierarchy

**Files:**
- Create: `notebooks/03_kernels_and_hierarchy.py`
- Modify: `tests/test_notebook_smoke.py` (add test)

**Interfaces:**
- Consumes: `data/noaa_tides_hourly.csv`, `data/places_diabetes.csv`, `data/fastball_spin_rates.csv`; `run_notebook`.
- Produces: a taught notebook that runs headlessly without error (no time budget). **Density target: ≥ ~2,800 md words, ≥ ~55 cells.**

> **Revised 2026-07-16 — expand to the density target and remove the execution-time budget.** Build each section out richly to notebook 01's depth; do NOT subset data (drop the `N_EXACT` speed cap — size the NOAA slice for teaching value) or thin sampling for speed. Full content spec finalized when rebuilt (after notebook 01 is approved).

- [ ] **Step 1: Add the failing smoke test**

```python
def test_03_kernels_hierarchy():
    run_notebook(NB / "03_kernels_and_hierarchy.py", timeout_s=900)
```

Run it → Expected: FAIL (missing).

- [ ] **Step 2: Build the kernel zoo section**

Cells: markdown; a reactive `mo.ui.dropdown` selecting kernel (ExpQuad, Matern12, Matern32, Matern52, Periodic, RatQuad, Linear) plus lengthscale/amplitude sliders; a cell drawing prior samples for the chosen kernel and plotting; markdown comparing smoothness/roughness.

- [ ] **Step 3: Build the kernel-combinations section (NOAA)**

Cells: **background cell** (NOAA station, mixed semidiurnal tide, meters MLLW); load the tides CSV, take the first **~250-point** slice (constant `N_EXACT = 250`; adjust down if Task 3 Step 5 showed >90s), standardize time and level; markdown on additive (OR) vs multiplicative (AND) structure; fit `gp.Marginal` with an **additive** kernel (long `Matern52` trend + two `Periodic` components for semidiurnal/diurnal) via `.marginal_likelihood`; `pm.sample(random_seed=42)`; plot the fit and, optionally, the additive component contributions; **exercise**: compose a kernel for a described pattern (accordion). Note in markdown that the full series is deferred to Hour 4 HSGP.

- [ ] **Step 4: Build the 2D spatial section (PLACES)**

Cells: **background cell** (CDC PLACES county diabetes, model-based small-area estimates caveat, %); load CSV, standardize `lon`/`lat` into X `(n,2)` and `diabetes_pct`→y; markdown on ARD/vector lengthscales; fit `gp.Marginal` with `Matern52(input_dim=2, ls=[l1, l2])` and `.marginal_likelihood`; `pm.sample(random_seed=42)`; predict on a lon/lat grid via `.conditional` and plot a plotly heatmap/scatter map with county points. ~100 counties → 100³ Cholesky is trivial.

- [ ] **Step 5: Build the hierarchical section (spin rates)**

Cells: **background cell** (baseball fastball spin rate, 3 pitchers × 10 games, rpm); load CSV; set up `coords={"pitcher": [...], "obs": ...}`, hold the pitcher index and standardized day-of-season in `pm.Data(..., dims="obs")`; markdown on partial pooling; a hierarchical latent GP — a shared population `gp.Latent` trend plus **non-centered** per-pitcher deviations (`offset = pm.Normal("offset", 0, 1, dims="pitcher")`, `dev = sigma_dev * offset`, small `sigma_dev` amplitude), Normal likelihood on standardized spin indexed by pitcher; prior predictive check first; `pm.sample(random_seed=42, target_accept=0.9)`; plot per-pitcher posterior trajectories with the shared trend using `az.plot_forest`/plotly; **exercise**: inspect pooling by shrinking/growing the `sigma_dev` prior (accordion). 30 observations → fast.

- [ ] **Step 6: Run the smoke test to verify it passes**

Run it → Expected: PASS ≤ 240s. If total runtime is tight, first prefer smaller `N_EXACT` / coarser grids; trim draws only as a last resort and re-verify `ess_bulk`/`ess_tail` > 400 and zero divergences.

- [ ] **Step 7: Interactive eyeball, then commit**

```bash
git add notebooks/03_kernels_and_hierarchy.py tests/test_notebook_smoke.py
git commit -m "feat: add kernels, 2D inputs, and hierarchy notebook"
```

---

## Task 10: `04_scaling_and_workflow.py` — scaling, HSGP, workflow

**Files:**
- Create: `notebooks/04_scaling_and_workflow.py`
- Modify: `tests/test_notebook_smoke.py` (add test)

**Interfaces:**
- Consumes: `data/noaa_tides_hourly.csv`; `run_notebook`.
- Produces: a taught notebook that runs headlessly without error (no time budget). **Density target: ≥ ~2,800 md words, ≥ ~55 cells.**

> **Revised 2026-07-16 — expand to the density target and remove the execution-time budget.** Build each section out richly to notebook 01's depth; do NOT thin sampling or shrink `m` for speed. Full content spec finalized when rebuilt (after notebook 01 is approved).

- [ ] **Step 1: Add the failing smoke test**

```python
def test_04_scaling_workflow():
    run_notebook(NB / "04_scaling_and_workflow.py", timeout_s=900)
```

Run it → Expected: FAIL (missing).

- [ ] **Step 2: Build the scaling-cost section**

Cells: markdown on O(n³) exact-inference cost; a timing demo that builds and factorizes covariance matrices at growing n (e.g., 100/300/900/2700) with `time` + a plotly line of elapsed vs n on a log scale. No sampling here.

- [ ] **Step 3: Build the sparse-approximation section**

Cells: markdown on inducing points / pseudo-inputs; one `pm.gp.MarginalApprox(approx="FITC")` fit on a moderate NOAA slice (~500 points) with a small set of inducing points; `pm.sample(random_seed=42)`; plot the fit. Keep it brief.

- [ ] **Step 4: Build the HSGP section (full NOAA series)**

Cells: markdown + basis-function intuition (plot a few HSGP basis functions); `mo.ui` sliders for `m` and `c` with a reactive cell explaining their effect on the basis; fit `pm.gp.HSGP(m=[...], c=..., cov_func=...)` (with `HSGPPeriodic` for the tidal cycles if available) on the **full ~8,760-point** series, `.prior("f", X=Xfull)`, Normal likelihood; `pm.sample(random_seed=42)`; plot the fit over the full year and overlay against the Hour-3 subset behavior; markdown on constraints (stationary kernels, input dim ≤ 3, boundary factor `c`); **exercise**: change `m`/`c` and judge approximation vs diagnostics (accordion). HSGP is linear in n → the full series fits in ~1 min.

- [ ] **Step 5: Build the workflow section**

Cells: teach the full diagnostic loop on the HSGP `idata` in order — prior predictive plot (shown before the fit above), **save-early** `idata.to_netcdf("results/hsgp.nc")` immediately after sampling as a demonstrated best practice, divergence count from `idata["sample_stats"]["diverging"].sum().item()`, `az.summary` reporting `r_hat`/`ess_bulk`/`ess_tail`, `az.plot_energy`, posterior predictive check via `pm.sample_posterior_predictive` + `az.plot_ppc_dist`; a markdown decision guide table (exact vs sparse vs HSGP: when each applies, cost, constraints). Note in markdown that `compute_log_likelihood`/LOO is out of scope for this intro course.

- [ ] **Step 6: Run the smoke test to verify it passes**

Run it → Expected: PASS ≤ 240s. If the HSGP fit is slow, reduce `m` (fewer basis functions) and re-verify the fit quality and diagnostics.

- [ ] **Step 7: Interactive eyeball, then commit**

```bash
git add notebooks/04_scaling_and_workflow.py tests/test_notebook_smoke.py
git commit -m "feat: add scaling, HSGP, and workflow notebook"
```

---

## Task 11: Full-suite verification and README wiring

**Files:**
- Modify: `README.md` (add a short "Running the notebooks" section)
- Modify: `data/README.md` (final proofread)

**Interfaces:**
- Consumes: everything built above.

- [ ] **Step 1: Run the entire test suite**

Run: `pixi run pytest tests/ -v`
Expected: all data-contract and notebook-smoke tests PASS. Note total notebook runtime; if any single notebook exceeds its budget on the reference laptop, first prefer smaller data slices / fewer HSGP basis functions (`m`), and trim draws only as a last resort while keeping `ess_bulk`/`ess_tail` > 400.

- [ ] **Step 2: Rebuild all data from scratch to confirm reproducibility**

Run: `pixi run python data/build_data.py && pixi run pytest tests/test_data_contracts.py -v`
Expected: CSVs regenerate and contracts still PASS.

- [ ] **Step 3: Add a "Running the notebooks" section to `README.md`**

Document: `pixi install`, run `notebooks/00_environment_check.py` first (`pixi run marimo edit notebooks/00_environment_check.py`), then the four hour notebooks in order; note that data is pre-vendored and `data/build_data.py` is only for maintainers.

- [ ] **Step 4: Final commit**

```bash
git add README.md data/README.md
git commit -m "docs: document running the workshop notebooks"
```

---

## Self-Review Notes

- **Spec coverage:** Environment/pixi (Task 1) ✓; five datasets + provenance + contracts (Tasks 2–5) ✓; env-check notebook (Task 6) ✓; four hour notebooks mapped to outline (Tasks 7–10) ✓; background cells per dataset (each notebook task) ✓; interactive widgets + accordion exercises (Tasks 7–10) ✓; live-runtime budget enforced via smoke tests + explicit sizing (Tasks 3, 9, 10) ✓; DataTree idioms and nutpie seeding (Global Constraints, referenced in fits) ✓; decision guide + full workflow diagnostics (Task 10) ✓.
- **Live-budget risk:** the NOAA exact fit is the main risk; Task 3 Step 5 measures it before any notebook depends on it, and Task 9 Step 3 ties the notebook's `N_EXACT` to that measurement.
- **API-name risk:** Task 1 Step 3 verifies `Marginal`/`Latent`/`HSGP`/`HSGPPeriodic`/`MarginalApprox` against the installed PyMC before any notebook uses them.
- **Type consistency:** dataset column names are fixed in each contract test and reused verbatim by the consuming notebook task; `run_notebook(path, timeout_s)` signature is defined once (Task 6) and reused unchanged.
