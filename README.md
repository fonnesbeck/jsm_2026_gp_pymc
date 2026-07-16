# JSM 2026 Continuing Education Course Proposal Draft

## COURSE TITLE

Nonparametric Bayesian Modeling: An Introduction to Gaussian Processes with PyMC

## INSTRUCTOR

1. Christopher Fonnesbeck (PyMC Labs; Email: fonnesbeck@gmail.com)

## COURSE LENGTH

Half-Day (4 hours)

## ABSTRACT

Gaussian Processes (GPs) are flexible tools for Bayesian nonparametric modeling, providing an effective solution for nonlinear regression and classification problems where traditional parametric models often fail due to assumptions about functional form or error distribution. GPs model the underlying function directly, defining a distribution over functions that inherently yields robust, probabilistic predictions and uncertainty quantification.
This course offers a practical introduction, starting with core Bayesian concepts and a concise PyMC tutorial. We will explore how to specify models using covariance functions (kernels) and apply both Marginal (conjugate) and Latent (non-conjugate) GPs to real-world data. The course culminates in building complex structures, such as additive and multiplicative kernels and hierarchical models, and addresses the critical issue of scalability. Attendees will learn when and how to deploy efficient modern approximations, such as the Hilbert Space Gaussian Process (HSGP), necessary for applying GPs to larger datasets.

**Prerequisite Knowledge**: Learners should have familiarity with basic statistical modeling (e.g., linear regression, estimation) and core components of the scientific Python stack (NumPy, pandas, Jupyter). No direct experience with PyMC or Bayesian statistics is expected.

## OUTLINE

This half-day course is structured to provide new users with a complete conceptual and practical foundation in GP modeling, ensuring they can specify, fit, and evaluate models of varying complexity using PyMC (version 5.22 or later).

1.  **Hour 1: Foundations and PyMC Primer**
    *   **Part A: Bayesian Workflow & PyMC Primer.** Introduction to the Bayesian paradigm (Prior, Likelihood, Posterior). A three-step PyMC workflow: model specification using the `pm.Model` context manager, defining priors for unknown parameters, and specifying the likelihood. Using optimization and sampling for inference.
    *   **Part B: GP Concepts.** Defining a Gaussian Process as a distribution over functions—an infinitely large Gaussian. The key properties of Gaussians: marginalization and conditioning. Key components: Mean function and Covariance function (kernel). Interpreting basic hyperparameters.
2.  **Hour 2: Core GP Modeling: Marginal and Latent GPs**
    *   **Marginal GPs (Conjugate Case):** Applying GPs where the data is Gaussian (e.g., log-transformed counts). Discussing conjugacy: Gaussian prior + Gaussian likelihood yields a Gaussian posterior in closed form. PyMC implementation using `gp.Marginal` and optimizing hyperparameters using MAP optimization or MCMC.
    *   **Latent GPs (Non-Conjugate Case):** Modeling non-Gaussian data (e.g., Binomial/Poisson counts). Using a transformation to link the Gaussian latent process to the likelihood. PyMC implementation using `gp.Prior` and the necessity of MCMC sampling (NUTS).
3.  **Hour 3: Building Flexible Models: Kernels and Hierarchy**
    *   **Covariance Function Deep Dive:** Comparing standard stationary kernels like the Exponential Quadratic (smooth) and the Matérn family (roughness).
    *   **Kernel Combinations:** Using additive (OR) and multiplicative (AND) structures to construct complex covariance functions (e.g., modeling long-term trend + seasonality).
    *   **Multi-Dimensional Inputs:** Extending kernels to handle multiple input dimensions (e.g., spatial or space-time data).
    *   **Hierarchical GPs:** Implementing partial pooling when data is grouped using latent GPs, focusing on structural definition.
4.  **Hour 4: Practical Implementation, Workflow, and Scalability**
    *   **GP Limitations and Scaling:** Identifying the computational bottleneck: $\mathcal{O}(n^3)$ cost for exact inference due to matrix inversion.
    *   **Approximation Methods:** Brief overview of sparse approximations using inducing points (pseudo inputs) to reduce matrix size. In-depth look at Hilbert Space Gaussian Processes (HSGP): a parametric approximation offering $\mathcal{O}(mn + m)$ speed. Discussing HSGP constraints (stationary kernels, input dimension $\le 3$). Implementing HSGP in PyMC using the streamlined API.
    *   **Model Workflow:** Practical topics, including prior predictive checks, posterior processing, and using diagnostics (ArviZ, divergences).

## LEARNING OUTCOMES

Upon completion of this half-day course, attendees will be able to:
1. Use PyMC Fundamentals: Specify basic Bayesian models in PyMC using the context manager, define priors and likelihoods, and run Markov Chain Monte Carlo (MCMC) inference.
2. Understand GP Fundamentals: Explain how Gaussian Processes define a distribution over functions, differentiate this approach from standard parametric regression, and describe the role of the covariance function.
3. Implement Core GP Models: Construct and fit GP models in PyMC using both the marginal likelihood (for Gaussian data) and the latent variable parameterization (for non-Gaussian data like Binomial or Poisson).
4. Design Flexible Kernels: Select appropriate covariance kernels (e.g., ExpQuad, Matérn) and combine them additively or multiplicatively to encode complex prior assumptions about the latent function (e.g., trend and periodicity).
5. Address Scalability: Recognize the computational challenges of exact GPs and implement scalable alternatives, specifically the Hilbert Space Gaussian Process (HSGP), for fitting models to larger datasets.

## INSTRUCTOR BIO

Chris is a Principal Quantitative Analyst at PyMC Labs and an Adjoint Associate Professor at the Vanderbilt University Medical Center, with 20 years of experience as a data scientist in academia, industry, and government, including 7 years in pro baseball research with the Philadelphia Phillies, New York Yankees, and Milwaukee Brewers. He is interested in computational statistics, machine learning, Bayesian methods, and applied decision analysis. He hails from Vancouver, Canada and received his Ph.D. from the University of Georgia.​​

## Running the notebooks

Course materials are marimo notebooks under `notebooks/`, backed by pinned
dependencies managed with [pixi](https://pixi.sh).

1. **Install the environment:**

   ```bash
   pixi install
   ```

2. **Check your setup first.** Run the environment-check notebook and confirm
   every cell reports OK before starting the course:

   ```bash
   pixi run marimo edit notebooks/00_environment_check.py
   ```

3. **Work through the four hour notebooks in order:**

   ```bash
   pixi run marimo edit notebooks/01_foundations.py
   pixi run marimo edit notebooks/02_marginal_latent_gps.py
   pixi run marimo edit notebooks/03_kernels_and_hierarchy.py
   pixi run marimo edit notebooks/04_scaling_and_workflow.py
   ```

All datasets used by the notebooks are pre-vendored as CSVs in `data/`, with
provenance and access dates recorded in `data/README.md` — no notebook fetches
data over the network, so attendees don't need internet access during the
course. `data/build_data.py` is a maintainer-only script that re-fetches and
regenerates those CSVs from their original sources; attendees never need to
run it.

The test suite (`pixi run pytest tests/`) exercises the data contracts and
runs each notebook end-to-end as a smoke test.
