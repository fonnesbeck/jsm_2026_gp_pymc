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
    out = df.select(
        pl.col("Subject").cast(pl.Int64).alias("subject"),
        pl.col("Time").cast(pl.Float64).alias("time"),
        pl.col("conc").cast(pl.Float64),
        pl.col("Dose").cast(pl.Float64).alias("dose"),
        pl.col("Wt").cast(pl.Float64).alias("weight"),
    ).sort(["subject", "time"])
    assert out.height == 132, out.height
    out.write_csv(DATA / "theophylline.csv")


# Classic Jarrett (1979) British coal-mining disaster counts, 1851-1962.
_COAL = [
    4,
    5,
    4,
    1,
    0,
    4,
    3,
    4,
    0,
    6,
    3,
    3,
    4,
    0,
    2,
    6,
    3,
    3,
    5,
    4,
    5,
    3,
    1,
    4,
    4,
    1,
    5,
    5,
    3,
    4,
    2,
    5,
    2,
    2,
    3,
    4,
    2,
    1,
    3,
    2,
    2,
    1,
    1,
    1,
    1,
    3,
    0,
    0,
    1,
    0,
    1,
    1,
    0,
    0,
    3,
    1,
    0,
    3,
    2,
    2,
    0,
    1,
    1,
    1,
    0,
    1,
    0,
    1,
    0,
    0,
    0,
    2,
    1,
    0,
    0,
    0,
    1,
    1,
    0,
    2,
    3,
    3,
    1,
    1,
    2,
    1,
    1,
    1,
    1,
    2,
    4,
    2,
    0,
    0,
    1,
    4,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    1,
    0,
    1,
    0,
]


def build_coal_disasters() -> None:
    years = list(range(1851, 1851 + len(_COAL)))
    df = pl.DataFrame({"year": years, "disasters": _COAL})
    assert df.height == 112, df.height
    df.write_csv(DATA / "coal_disasters.csv")


def main() -> None:
    build_theophylline()
    build_coal_disasters()
    # build_noaa_tides()      # Task 3
    # build_places_diabetes()  # Task 4
    # build_spin_rates()       # Task 5


if __name__ == "__main__":
    main()
