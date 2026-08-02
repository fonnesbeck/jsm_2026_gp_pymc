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
    assert df.height == 284
    assert df.null_count().sum_horizontal().item() == 0
    assert df["n_pitches"].min() >= 1
    assert 1000 <= df["spin_rate"].min() <= df["spin_rate"].max() <= 3500

    games = (
        df.group_by("pitcher").len().sort(["len", "pitcher"], descending=[True, False])
    )
    assert games.height == 6

    # The multi-output example models the five pitchers with the most games;
    # the single-pitcher example models Kopech. Both must be present with
    # enough games to fit a season-long curve.
    top_five = games.head(5)["pitcher"].to_list()
    assert "Kopech, Michael" in top_five
    assert games.head(5)["len"].min() >= 40

    # Notebook 2's legacy pinned model still fits Buehler.
    assert "Buehler, Walker" in games["pitcher"].to_list()


def test_batter_grades_contract():
    df = pl.read_csv(DATA / "batter_grades_2023.csv")
    assert {"age", "swing_decision", "n_pa"} <= set(df.columns)
    assert df.height == 9971
    assert df["age"].min() >= 16
    assert df["age"].max() <= 50
    # One batter (2 rows) has a null swing_decision (no batted-ball data to
    # grade); every non-null value is finite.
    assert df["swing_decision"].null_count() == 2
    assert df["swing_decision"].drop_nulls().is_finite().all()
    # n_pa includes rows with zero plate appearances (partial-season callups
    # with no qualifying PAs), so the floor is 0, not 1.
    assert df["n_pa"].min() >= 0
    # The spline and HSGP examples model swing decision against age, so age
    # must span enough distinct values for a basis to be worth fitting.
    assert df["age"].n_unique() >= 15


def test_taken_pitches_contract():
    df = pl.read_csv(DATA / "taken_pitches_walker.csv")
    assert {"location_x", "location_z", "is_strike"} <= set(df.columns)
    assert df.height == 1568
    assert set(df["is_strike"].unique().to_list()) <= {0, 1, True, False}
    # The 2-D HSGP predicts over a fixed grid spanning the strike zone, but
    # the raw Statcast locations include some pitches tracked outside the
    # nominal zone (actual range is x in [-3.027, 3.147], z in
    # [-1.337, 5.923]), so the bounds below are widened accordingly rather
    # than clipped to [-3, 3] x [0, 6].
    assert df["location_x"].min() >= -3.5
    assert df["location_x"].max() <= 3.5
    assert df["location_z"].min() >= -1.5
    assert df["location_z"].max() <= 6.0
