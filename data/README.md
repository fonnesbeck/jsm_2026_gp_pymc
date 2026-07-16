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

<!-- per-dataset sections added by later tasks -->
