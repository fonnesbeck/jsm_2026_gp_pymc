# Datasets

All files here are frozen snapshots rebuilt by `build_data.py`. No workshop
notebook fetches data at runtime. Each entry records source, access date,
license, units, and transformations.

## Environment notes

No dependency-version substitutions were needed. `pymc>=6,<7` and `arviz>=1,<2`
solved as specified on conda-forge for all four platforms (win-64, linux-64,
osx-64, osx-arm64), resolving to pymc 6.1.0 and arviz 1.2.0. All five GP API
names (`Marginal`, `Latent`, `HSGP`, `HSGPPeriodic`, `MarginalApprox`) are
present on `pymc.gp` in this version.

## Theophylline (`theophylline.csv`)

- **Source:** vincentarelbundock Rdatasets mirror of R's built-in
  `datasets::Theoph` — https://vincentarelbundock.github.io/Rdatasets/csv/datasets/Theoph.csv
- **Access date:** 2026-07-15
- **License:** Rdatasets redistributes R's bundled `datasets` package data;
  `Theoph` is public-domain pharmacokinetic teaching data (Boeckmann, Sheiner
  & Beal 1994, *NONMEM Users Guide*, via R's `nlme`/`datasets` packages).
- **Transformations:** columns renamed/selected from the source
  (`Subject`, `Time`, `conc`, `Dose`, `Wt`) to `subject`, `time`, `conc`,
  `dose`, `weight`; all numeric columns cast to their target dtype; rows
  sorted by `(subject, time)`. No values were altered.
- **Units:** `time` in hours since dosing; `conc` in mg/L (theophylline
  serum concentration); `dose` in mg/kg (single oral dose); `weight` in kg.
  One oral dose per subject, 12 subjects, 11 concentration measurements
  each (132 rows).

## Coal-mining disasters (`coal_disasters.csv`)

- **Source:** classic counts of British coal-mining disasters (≥10 deaths)
  from Jarrett, R.G. (1979), "A note on the intervals between coal-mining
  disasters," *Biometrika* 66(1), 191–193. Values vendored verbatim as a
  literal list in `build_data.py` (no network fetch) — this is the
  standard annual aggregation of Jarrett's data widely used as a
  changepoint/Poisson-process teaching example.
- **Access date:** 2026-07-15 (values embedded at authoring time)
- **License:** public-domain historical count data; no redistribution
  restrictions.
- **Transformations:** none beyond pairing each annual count with its
  calendar year.
- **Units:** `year` is the calendar year (1851–1962 inclusive, 112 years);
  `disasters` is the annual count of coal-mining disasters with 10 or more
  deaths.

## NOAA tide gauge (`noaa_tides_hourly.csv`)

- **Source:** NOAA CO-OPS `datagetter` API —
  https://api.tidesandcurrents.noaa.gov/api/prod/datagetter — station
  `9414290` (San Francisco, CA), a long-record mixed semidiurnal station
  showing both diurnal and semidiurnal tidal components. Product
  `hourly_height` (verified hourly water levels), datum MLLW (mean lower
  low water), time zone GMT, units metric. Fetched the full calendar year
  2019 (`begin_date=20190101&end_date=20191231`); 2019 returned a complete
  hourly series at this station, so no fallback station/year was needed.
- **Access date:** 2026-07-15
- **License:** NOAA CO-OPS data are public and freely available for reuse;
  see NOAA's data policy and disclaimer at
  https://tidesandcurrents.noaa.gov/disclaimers.html. Data are provided
  "as is" without warranty; NOAA requests attribution when redistributed.
- **Transformations:** raw API response's `data` array reduced to two
  columns (`t` -> `time`, `v` -> `water_level`); `water_level` cast to
  Float64; rows with null/unparseable water levels dropped; rows sorted by
  `time`. No values were altered.
- **Units:** `time` is an ISO-8601-like GMT timestamp string
  (`YYYY-MM-DD HH:MM`); `water_level` is meters relative to the MLLW
  datum. One full year of hourly observations, 8,760 rows (2019 is not a
  leap year). Hourly NOAA CO-OPS series can occasionally contain small
  gaps (sensor outages, QC removals); the contract test allows a row
  count between 8,000 and 9,000 to tolerate this, though this particular
  fetch returned the complete 8,760-row series.

## CDC PLACES county diabetes & obesity (`places_diabetes.csv`)

- **Source:** CDC PLACES: Local Data for Better Health, County Data, 2025
  release — served via the Socrata Open Data API on data.cdc.gov, dataset
  id `swc5-untb` (`https://data.cdc.gov/resource/swc5-untb.json`,
  metadata at `https://data.cdc.gov/api/views/swc5-untb.json`). Underlying
  survey/model year for the pulled records is 2023.
- **Access date:** 2026-07-15
- **State used:** North Carolina (100 counties; 200 raw records — one
  DIABETES row and one OBESITY row per county).
- **Measures:** `DIABETES` ("Diagnosed diabetes among adults") and
  `OBESITY` ("Obesity among adults"), both restricted to
  `data_value_type == "Crude prevalence"` (percent, unadjusted for age).
- **License:** CDC PLACES data are public domain U.S. government data,
  freely available for reuse; see
  https://www.cdc.gov/places/about/index.html.
- **Field-name adaptation:** the brief assumed top-level `latitude`/
  `longitude` fields, but the live 2025-release schema has no such
  fields — the county centroid is instead nested in a `geolocation`
  GeoJSON `Point` field (`geolocation.coordinates` = `[lon, lat]`). The
  builder selects `geolocation` instead of `latitude`/`longitude` and
  unpacks `coordinates[0]` -> `lon`, `coordinates[1]` -> `lat` after
  fetch. The measure ids (`DIABETES`, `OBESITY`), `data_value_type`
  filter (`"Crude prevalence"`), and the rest of the selection contract
  from the brief were unchanged and confirmed against a live sample
  record before building.
- **Transformations:** raw long-format rows (one row per
  county x measure) filtered to `statedesc == state` and
  `data_value_type == "Crude prevalence"`, `data_value` cast to
  Float64, `geolocation.coordinates` unpacked into `lon`/`lat` Float64
  columns, then pivoted to one row per county with `DIABETES` ->
  `diabetes_pct` and `OBESITY` -> `obesity_pct`; rows with nulls
  dropped; sorted by `county`.
- **Units:** `diabetes_pct`/`obesity_pct` are crude (unadjusted)
  prevalence percentages among adults; `lon`/`lat` are the county
  centroid in decimal degrees (WGS84).
- **Caveat:** PLACES values are **model-based small-area estimates**
  produced by CDC from BRFSS survey data plus census/population
  covariates via multilevel regression and poststratification — they
  are not raw county-level observations or a county census, and carry
  associated modeling uncertainty (see the `low_confidence_limit`/
  `high_confidence_limit` fields in the source data, not vendored here).

## Fastball spin rates (`fastball_spin_rates.csv`)

- **Source:** derived from the `instats_gp` project's vendored fastball
  spin-rate file
  (`/var/home/fonnesbeck/repos/instats_gp/data/fastball_spin_rates.csv`),
  itself sourced from MLB Statcast — 2021-season fastball average spin
  rate per pitcher per game. No network fetch: the source file already
  exists on disk and is read directly.
- **Access date:** 2026-07-15
- **License:** MLB Statcast data as redistributed for teaching/analysis
  use in the `instats_gp` project; no additional restrictions applied
  here.
- **Curation:** the source file covers many pitcher-games. It is
  deterministically curated down to a small teaching-sized panel:
  rows with a null `pitcher`/`game_date`/`spin_rate` are dropped; rows
  with `n_pitches < 10` are dropped; duplicate `(pitcher, game_date)`
  rows are collapsed to one; pitchers are ranked by number of
  remaining games (descending), ties broken alphabetically, and the
  top 3 pitchers with at least 10 qualifying games are kept (sorted
  alphabetically by name); for each of those 3 pitchers, the first 10
  games in date order are kept. Columns renamed from the source
  (`pitcher_name` -> `pitcher`, `avg_spin_rate` -> `spin_rate`); rows
  sorted by `(pitcher, game_date)`.
- **Selected pitchers (deterministic given the source file):**
  Buehler, Walker; Hearn, Taylor; Rodriguez, Richard.
- **Units:** `game_date` is an ISO-8601 date string; `spin_rate` is the
  average fastball spin rate for that pitcher-game in revolutions per
  minute (rpm); `n_pitches` is the number of fastballs thrown by that
  pitcher in that game (all rows have `n_pitches >= 10`). 3 pitchers x
  10 games = 30 rows, no nulls.

<!-- per-dataset sections added by later tasks -->
