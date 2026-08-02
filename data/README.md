# Datasets

All files here are frozen snapshots. No workshop notebook fetches data at
runtime. Most are rebuilt by `build_data.py`; the two baseball grading and
pitch-location files are vendored from the CQS PyMC course, which has no build
script for them. Each entry records what is known about source, units, and
transformations.

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
- **Curation:** only the six pitchers the workshop models are kept, not
  the full 2021 season. The builder (`data/build_data.py` lines 237-264)
  reads the source file, renames columns (`pitcher_name` -> `pitcher`,
  `avg_spin_rate` -> `spin_rate`), drops rows with a null
  `pitcher`/`game_date`/`spin_rate`, collapses duplicate
  `(pitcher, game_date)` rows to one, then ranks pitchers by game count
  (ties broken alphabetically for determinism) and keeps the five with
  the most games — Rodriguez, Richard (64); Taylor, Josh (59); Kopech,
  Michael (43); Wells, Tyler (43); Hearn, Taylor (42) — plus Buehler,
  Walker (33). The top five are exactly the set the ICM multi-output
  example selects when it takes the pitchers with the most games, so
  nothing is silently excluded from that example. Buehler is not part of
  the new material; he is kept only because
  `notebooks/02_gp_priors_and_kernels.py` still fits the legacy
  three-pitcher hierarchical model on a pinned subset that includes him.
  Once that notebook is rebuilt, Buehler can be dropped and the file
  becomes 251 rows across five pitchers. No filter is applied on
  `n_pitches`; the observed range is 1-66.
- **Rebuild caveat:** the builder reads its source from an absolute path
  into a sibling repository,
  `/var/home/fonnesbeck/repos/instats_gp/data/fastball_spin_rates.csv`,
  which is not part of this repository and is not fetched over the
  network. `build_spin_rates()` therefore only runs successfully on the
  maintainer's machine; anyone else must re-vendor `fastball_spin_rates.csv`
  by hand if it is ever lost.
- **Units:** `game_date` is an ISO-8601 date string; `spin_rate` is the
  average fastball spin rate for that pitcher-game in revolutions per
  minute (rpm); `n_pitches` is the number of fastballs thrown by that
  pitcher in that game (observed range 1-66). 284 rows, 6 pitchers,
  no nulls.

## Batter swing-decision grades (`batter_grades_2023.csv`)

- **Source:** vendored verbatim from
  `cqs-pymc-course/notebooks/data/batter_grades_2023.csv` (the CQS PyMC
  course, a sibling workshop repository). No upstream build script
  produces this file in that repository either — it is itself a frozen
  snapshot there, so there is no rebuild path to reproduce it from raw
  Statcast data. If it is ever lost, it must be re-copied from that
  repository rather than regenerated.
- **Access date:** 2026-07-26 (copied into this repository).
- **License:** derives from MLB Statcast data for the 2023 season, as
  redistributed for teaching/analysis use in the CQS PyMC course; no
  additional restrictions applied here.
- **Transformations:** none — copied byte-for-byte from the source file.
- **Shape:** 9,971 rows, 13 columns. Other columns beyond those listed
  under Units (`batter_id`, `batter`, `season`, `level`, `bats`,
  `throws`, `bat_speed`, `bat_to_ball`, `attack_angle`) are present in
  the file but not used by the workshop notebooks.
- **Units:** `age` is batter age in years (observed range 17-43);
  `swing_decision` is a standardized swing-decision grade (float, 2 rows
  are null because that batter has no batted-ball data to grade); `n_pa`
  is plate appearances that season (integer, minimum observed value is 0
  for partial-season callups with no qualifying PAs).

## Called-strike locations (`taken_pitches_walker.csv`)

- **Source:** vendored verbatim from
  `cqs-pymc-course/notebooks/data/taken_pitches_walker.csv` (the CQS
  PyMC course). No upstream build script produces this file in that
  repository either — it is itself a frozen snapshot there, so there is
  no rebuild path. If it is ever lost, it must be re-copied from that
  repository rather than regenerated.
- **Access date:** 2026-07-26 (copied into this repository).
- **License:** derives from MLB Statcast data for the 2023 season
  (taken/called pitches thrown by a single pitcher, Walker), as
  redistributed for teaching/analysis use in the CQS PyMC course; no
  additional restrictions applied here.
- **Transformations:** none — copied byte-for-byte from the source file.
- **Shape:** 1,568 rows, 9 columns. Other columns beyond those listed
  under Units (`play_id`, `game_pk`, `pitcher_id`, `pitcher_name`,
  `bats`, `throws`) are present in the file but not used by the
  workshop notebooks.
- **Units:** `location_x` and `location_z` are pitch location in feet
  relative to the plate, at the front edge of home plate (observed
  ranges are approximately [-3.03, 3.15] for `location_x` and
  [-1.34, 5.92] for `location_z` — wider than the nominal strike-zone
  plotting grid of [-3, 3] x [0, 6] because a few tracked pitches fall
  well outside the zone); `is_strike` is a called-strike indicator
  (integer 0/1, no other encodings present).

<!-- per-dataset sections added by later tasks -->
