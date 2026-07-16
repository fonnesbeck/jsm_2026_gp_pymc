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

<!-- per-dataset sections added by later tasks -->
