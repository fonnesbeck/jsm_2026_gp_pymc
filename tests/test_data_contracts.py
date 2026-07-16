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
