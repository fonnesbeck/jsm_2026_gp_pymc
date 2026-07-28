# Nonparametric Bayesian Modeling: An Introduction to Gaussian Processes with PyMC

*JSM 2026 Continuing Education Course*

Gaussian processes (GPs) are Bayesian models for unknown functions. In this
hands-on course, you will build, fit, check, and scale GP models with
[PyMC](https://www.pymc.io/) through interactive
[marimo](https://marimo.io/) notebooks.

## Course details

| | |
|---|---|
| **Date** | Monday, August 3, 2026 |
| **Time** | 8:30 AM–12:30 PM |
| **Room** | CC-153B |
| **Venue** | Thomas M. Menino Convention & Exhibition Center, Boston |
| **Instructor** | Chris Fonnesbeck, PyMC Labs |

## Before the course

### Prerequisites

Bring a laptop. You should be comfortable with basic statistical modeling,
such as linear regression and estimation. Some familiarity with Python is
helpful, but no previous experience with PyMC, Bayesian statistics, or Python
package management is required.

### Install Pixi

This course uses [Pixi](https://pixi.sh/) to install the correct version of
Python and every course package. You do **not** need to install Python
separately.

Open a terminal application and run the command for your operating system:

**macOS or Linux**

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

**Windows PowerShell**

```powershell
powershell -ExecutionPolicy ByPass -c "irm -Uri https://pixi.sh/install.ps1 | iex"
```

Close and reopen the terminal after installation. If `pixi` is still not
recognized, see the [official Pixi installation instructions](https://pixi.sh/latest/).

### Get the course materials

Choose one option:

**Clone with Git**

```bash
git clone https://github.com/fonnesbeck/jsm_2026_gp_pymc.git
cd jsm_2026_gp_pymc
```

**Download a ZIP file**

Download [the course materials as a ZIP file](https://github.com/fonnesbeck/jsm_2026_gp_pymc/archive/refs/heads/main.zip),
extract it, then open a terminal in the extracted `jsm_2026_gp_pymc` folder.

### Install the course environment

From the course-materials folder, run:

```bash
pixi install
```

Pixi downloads the pinned Python environment and all packages used in the
notebooks. The first installation may take several minutes.

### Verify your setup

Run the environment-check notebook before the course:

```bash
pixi run marimo edit notebooks/00_environment_check.py
```

`pixi run` starts marimo inside the course environment that Pixi installed.
Marimo opens the notebook in your web browser. Run each cell; the notebook
checks package imports, verifies the course data, and compiles a small GP model
without sampling. You are ready when it displays **Environment OK**.

### Get help

Ask setup and course-material questions in
[GitHub Discussions](https://github.com/fonnesbeck/jsm_2026_gp_pymc/discussions).
Include your operating system, the command you ran, and the complete error
message.

## During the course

Open the notebooks in order. To work interactively, use `marimo edit`:

```bash
pixi run marimo edit notebooks/01_foundations.py
```

To view a notebook as a read-only app with code hidden by default, replace
`edit` with `run`:

```bash
pixi run marimo run notebooks/01_foundations.py
```

| Notebook | Topic |
|---|---|
| `00_environment_check.py` | Verify the environment and course data |
| `01_foundations.py` | Bayesian workflow, the PyMC API, prior/posterior predictive checks, and flexible regression with splines |
| `02_gp_priors_and_kernels.py` | From multivariate normals to GP priors, covariance functions, conditioning, and kernel composition |
| `03_marginal_and_latent_gps.py` | Exact marginal GPs, multi-output and hierarchical GPs, robust latent GPs, and a latent Poisson GP |
| `04_scaling_and_workflow.py` | Exact-GP computational cost, sparse FITC and HSGP approximations, and model-development workflow |

```bash
pixi run marimo edit notebooks/01_foundations.py
pixi run marimo edit notebooks/02_gp_priors_and_kernels.py
pixi run marimo edit notebooks/03_marginal_and_latent_gps.py
pixi run marimo edit notebooks/04_scaling_and_workflow.py
```

## What you will learn

By the end of the course, you will be able to:

1. Specify basic Bayesian models in PyMC and use Markov chain Monte Carlo
   (MCMC) to draw posterior samples.
2. Explain a GP as a probability distribution over functions and describe the
   role of its mean and covariance functions.
3. Fit exact marginal GP models, multi-output and hierarchical GPs, and
   latent GP models for robust and Poisson observations.
4. Choose and combine covariance functions, including exponential-quadratic,
   Matérn, and periodic kernels, to express assumptions about smoothness,
   trend, and periodicity.
5. Check model fit with prior and posterior predictive checks and diagnose
   sampling problems with ArviZ.
6. Compare exact, sparse FITC, and HSGP GP approaches and select an
   approximation appropriate for a larger dataset.

## Course topics

The four-hour course covers:

1. **Foundations and PyMC** — Bayesian inference, model specification, priors,
   likelihoods, optimization, MCMC, predictive checks, and the limits of
   hand-chosen functional forms.
2. **GP priors and kernels** — multivariate-normal conditioning, GP priors,
   covariance functions, sample functions, and additive, multiplicative, and
   periodic kernels.
3. **Exact GP models** — marginal GPs for Gaussian observations, multi-output
   GPs, hierarchical GPs, robust latent GPs, and a latent Poisson GP.
4. **Workflow and scaling** — prior and posterior predictive checks,
   diagnostics, exact-GP computational cost, sparse FITC approximations, HSGP
   approximations, and choosing among these approaches.

## Instructor

Chris Fonnesbeck is a Principal Quantitative Analyst at PyMC Labs and an
Adjoint Associate Professor at the Vanderbilt University Medical Center. He has
20 years of data-science experience in academia, industry, and government,
including seven years of baseball research with the Philadelphia Phillies, New
York Yankees, and Milwaukee Brewers. His work spans computational statistics,
machine learning, Bayesian methods, and applied decision analysis.

## Data and reproducibility

All notebook datasets are included as CSV files under `data/`; the notebooks
do not download data from the network. After you have installed the course
environment, you can use the materials without an internet connection.

`data/README.md` records dataset provenance and access dates.
`data/build_data.py` re-fetches and regenerates the data and is for course
maintainers only; attendees do not need to run it.
