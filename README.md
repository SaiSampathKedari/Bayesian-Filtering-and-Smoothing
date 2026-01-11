# Bayesian Filtering and Smoothing

![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange.svg)
![Last Commit](https://img.shields.io/github/last-commit/SaiSampathKedari/Bayesian-Filtering-and-Smoothing)
![Stars](https://img.shields.io/github/stars/SaiSampathKedari/Bayesian-Filtering-and-Smoothing?style=social)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A practical and conceptual study of **Bayesian filtering and smoothing** for systems whose internal states are not directly observable and evolve over time under uncertainty.  
The emphasis is on **inference**, not simulation: how noisy measurements and imperfect models are combined to reason about hidden states, and how uncertainty propagates through time.

---

## Overview

In robotics and dynamical systems, the true state of a system is rarely known exactly. Instead, we observe noisy sensor data and rely on approximate models of system dynamics. Bayesian filtering and smoothing provide a principled framework for **state estimation under uncertainty**, where the system state is treated as a random variable rather than a deterministic quantity.

This repository builds these ideas step by step, starting from **linear Gaussian models** and Bayesian regression, and progressing toward recursive Bayesian filters and particle-based methods. Emphasis is placed on understanding:
- what is uncertain,
- where that uncertainty comes from,
- how it is propagated through time,
- and why filtering methods succeed or fail in practice.

The material is developed through **formal derivations (PDFs), executable notebooks, and visualizations**, closely tied to probabilistic modeling foundations.

---

## Topics Covered (and in progress)

- Linear Gaussian estimation and covariance inference
- Batch vs recursive Bayesian regression
- Kalman Filter (KF)
- Extended Kalman Filter (EKF)
- Unscented Kalman Filter (UKF)
- Gauss–Hermite Kalman Filter
- Particle Filtering (bootstrap and EKF/UKF-based proposals)
- Importance sampling, weight degeneracy, ESS, and resampling
- Practical diagnostics and failure modes

---

## Visualizations

### Bayesian Filtering Flow

<p align="center">
  <img src="images/bayesian_filtering/filtering_flow.png" height="260">
</p>

<p align="center"><i>Recursive Bayesian filtering as probabilistic inference over time.</i></p>

---

## Notebook Gallery

### Chapter 1: Linear Gaussian Regression
- [Batch vs Recursive Linear Gaussian Regression](notebooks/ch01_regression/ch01_linear_gaussian_regression_batch_vs_recursive.ipynb)

### Chapter 2: Bayesian Filtering
- [Kalman Filter](notebooks/ch02_filtering/01_kalman_filter.ipynb)
- [Extended Kalman Filter](notebooks/ch02_filtering/02_extended_kalman_filter.ipynb)
- [Gauss–Hermite Kalman Filter](notebooks/ch02_filtering/03_gauss_hermite_kalman_filter.ipynb)
- [Unscented Kalman Filter](notebooks/ch02_filtering/04_unscented_kalman_filter.ipynb)
- [Particle Filter](notebooks/ch02_filtering/05_particle_filter.ipynb)

---

## PDF Notes and Derivations

Formal derivations and conceptual write-ups supporting the implementations:

- `01_linear_gaussian_estimation_covariance_inference.pdf`
- `02_Linear_Gaussian_Regression_Batch_Recursive.pdf`
- `03_Bayesian_modeling_for_Dynamical_Systems.pdf`
- `04_Bayesian_Filtering_Equations.pdf`
- `05_Gaussian_Filtering_Equations.pdf`
- `06_Kalman_Filtering_Equations.pdf`
- `07_Extended_Kalman_Filtering_Equations.pdf`
- `10_Gaussian_Filtering_Summary_and_motivation_for_ParticleFiltering.pdf`
- `11_PF_Empirical_Distributions_and_Importance_Sampling_Foundations.pdf`
- `12_Sequential_Self-Normalized_Importance_Sampling.pdf`
- `13_WeightDegeneracy_ESS_Resampling.pdf`
- `15_Optimal_Importance_Proposal_for_ParticleFiltering.pdf`
- `16_GaussianApproximations_of_the_OptimalProposal_in_ParticleFiltering.pdf`
- `17_OptimalProposal_for_LinearMeasurementModel_in_ParticleFiltering.pdf`

---

## Project Structure

```text
Bayesian-Filtering-and-Smoothing/
│
├── images/
│   ├── bayesian_filtering/
│   └── regression/
│
├── notebooks/
│   ├── ch01_regression/
│   └── ch02_filtering/
│
├── reports/
│
├── src/
│   ├── regression/
│   │   └── linear_regression.py
│   │
│   ├── filters/
│   │   ├── kalman_filter.py
│   │   ├── extended_kalman_filter.py
│   │   ├── unscented_kalman_filter.py
│   │   ├── gauss_hermite_kalman_filter.py
│   │   ├── particle_filter.py
│   │   ├── bootstrap_particle_filter.py
│   │   ├── ekf_particle_filter.py
│   │   └── ukf_particle_filter.py
│   │
│   └── models/
│       └── pendulum.py
│
├── README.md
└── pyproject.toml
````
---

# Status

🚧 Under active development

 Further extensions include smoother implementations, deeper diagnostic studies, and tighter integration between Gaussian and particle-based methods.
 
---

# About Me

I study and implement methods in **optimization, control, robotics, Bayesian inference, and probabilistic reasoning**.

* GitHub: [https://github.com/SaiSampathKedari](https://github.com/SaiSampathKedari)
* LinkedIn: [https://linkedin.com/in/sai-sampath-kedari](https://linkedin.com/in/sai-sampath-kedari)
* X: [https://x.com/SSampathKedari](https://x.com/SSampathKedari)
* Email: [sampath@umich.edu](mailto:sampath@umich.edu)
