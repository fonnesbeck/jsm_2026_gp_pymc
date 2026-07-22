import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Environment Check

    This notebook verifies that the workshop environment is set up correctly:
    the required libraries import and report sane versions, the vendored
    datasets load and are non-empty, and a small `pm.gp.Marginal` model
    compiles (without sampling).
    """)
    return


@app.cell
def _():
    import arviz as az
    import marimo
    import plotly
    import polars as pl
    import pymc as pm

    print(f"pymc:    {pm.__version__}")
    print(f"arviz:   {az.__version__}")
    print(f"marimo:  {marimo.__version__}")
    print(f"polars:  {pl.__version__}")
    print(f"plotly:  {plotly.__version__}")
    return pl, pm


@app.cell
def _(pl):
    from pathlib import Path

    data_dir = Path(__file__).parent.parent / "data"

    csv_names = [
        "theophylline.csv",
        "coal_disasters.csv",
        "noaa_tides_hourly.csv",
        "places_diabetes.csv",
        "fastball_spin_rates.csv",
    ]

    dataframes = {}
    for name in csv_names:
        df = pl.read_csv(data_dir / name)
        assert df.height > 0, f"{name} loaded but is empty"
        dataframes[name] = df
        print(f"{name}: {df.height} rows, {len(df.columns)} columns")
    return (dataframes,)


@app.cell
def _(dataframes, pm):
    import numpy as np

    x = np.linspace(0, 10, 20)
    y = np.sin(x) + 0.1 * np.random.default_rng(0).standard_normal(x.size)

    with pm.Model() as env_check_model:
        ell = pm.Gamma("ell", alpha=2, beta=1)
        eta = pm.HalfCauchy("eta", beta=5)
        cov_func = eta**2 * pm.gp.cov.Matern52(1, ell)
        mean_func = pm.gp.mean.Zero()
        gp = pm.gp.Marginal(mean_func=mean_func, cov_func=cov_func)

        sigma = pm.HalfCauchy("sigma", beta=5)
        gp.marginal_likelihood("obs", X=x.reshape(-1, 1), y=y, sigma=sigma)

    # Compile the logp function without sampling to confirm the model builds.
    env_check_model.compile_logp()(env_check_model.initial_point())
    graph = pm.model_to_graphviz(env_check_model)

    assert len(dataframes) == 5
    print("GP model compiled successfully (no sampling performed).")
    return (graph,)


@app.cell
def _(graph):
    graph
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Environment OK
    """)
    return


if __name__ == "__main__":
    app.run()
