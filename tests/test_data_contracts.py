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


def test_noaa_tides_contract():
    df = pl.read_csv(DATA / "noaa_tides_hourly.csv")
    assert df.columns == ["time", "water_level"]
    assert df.height >= 8000  # ~1 year of hourly obs, allowing small gaps
    assert df.height <= 9000
    assert df["water_level"].is_finite().all()
    # exact-fit slice used in Hour 3 (N_EXACT=200, ~8 days) must be small enough to fit live
    assert df.head(300).height == 300


def test_places_diabetes_contract():
    df = pl.read_csv(DATA / "places_diabetes.csv")
    assert df.columns == ["county", "lon", "lat", "diabetes_pct", "obesity_pct"]
    assert 90 <= df.height <= 260  # one state's counties
    assert df["diabetes_pct"].is_finite().all()
    assert df["diabetes_pct"].min() > 0
    assert df["diabetes_pct"].max() < 40
    assert df["lat"].min() > 20 and df["lat"].max() < 55
    assert df.null_count().sum_horizontal().item() == 0


def test_spin_rates_contract():
    df = pl.read_csv(DATA / "fastball_spin_rates.csv")
    assert df.columns == ["pitcher", "game_date", "spin_rate", "n_pitches"]
    assert df.height == 30
    assert df["pitcher"].n_unique() == 3
    counts = df.group_by("pitcher").len()["len"].to_list()
    assert all(c == 10 for c in counts)
    assert df["n_pitches"].min() >= 10
    assert df.null_count().sum_horizontal().item() == 0
