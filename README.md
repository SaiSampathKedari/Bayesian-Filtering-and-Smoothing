# Bayesian Filtering and Smoothing

![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange.svg)
![Last Commit](https://img.shields.io/github/last-commit/SaiSampathKedari/Bayesian-Filtering-and-Smoothing)
![Stars](https://img.shields.io/github/stars/SaiSampathKedari/Bayesian-Filtering-and-Smoothing?style=social)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A practical and conceptual study of **Bayesian filtering and smoothing** for systems whose internal states are not directly observable and evolve over time under uncertainty.  
The focus is on inference, not simulation: how noisy measurements and imperfect models are combined to reason about hidden states, and how uncertainty propagates through time.

---

## Overview

In robotics and dynamical systems, the true state of a system is rarely known exactly. Instead, we observe noisy sensor data and rely on approximate models of system dynamics. Bayesian filtering and smoothing provide a principled framework for **state estimation under uncertainty**, treating the system state as a random variable rather than a deterministic quantity.

This repository builds these ideas step by step, starting from **linear Gaussian models** and progressing toward recursive Bayesian filters. Emphasis is placed on understanding:
- what is uncertain,
- where that uncertainty comes from,
- how it is propagated,
- and why filtering methods succeed or fail in practice.

The material is developed through **derivations, visualizations, and executable notebooks**, closely tied to formal probabilistic modeling.

---

## Topics (in progress)

- Linear Gaussian estimation
- Batch vs recursive Bayesian regression
- Kalman Filter (KF)
- Bayesian filtering equations
- Gaussian filtering foundations
- Diagnostics and failure modes

---

## Visualizations (early)

<p align="center">
  <img src="images/bayesian_filtering/filtering_flow.png" height="260">
</p>

<p align="center"><i>Bayesian filtering as recursive probabilistic inference over time.</i></p>

---

## Notebook Gallery

### Chapter 1: Linear Gaussian Models
- [Batch vs Recursive Linear Gaussian Regression](notebooks/ch01_regression/ch01_linear_gaussian_regression_batch_vs_recursive.ipynb)

### Chapter 2: Filtering
- [Kalman Filter Foundations](notebooks/ch02_filtering/ch02_kalman_filter.ipynb)

---

## PDF Notes and Derivations

Formal derivations and conceptual write-ups:

- `01_linear_gaussian_estimation_covariance_inference.pdf`
- `02_Linear_Gaussian_Regression_Batch_Recursive.pdf`
- `03_Bayesian_modeling_for_Dynamical_Systems.pdf`
- `04_Bayesian_Filtering_Equations.pdf`
- `05_Gaussian_Filtering_Equations.pdf`
- `06_Kalman_Filtering_Equations.pdf`
- `07_Extended_Kalman_Filtering_Equations.pdf`

---

## Project Structure

```text
Bayesian-Filtering-and-Smoothing/
│
├── images/
│   ├── bayesian_filtering/
│   └── regression/
├── notebooks/
│   ├── ch01_regression/
│   └── ch02_filtering/
├── reports/
├── src/
│   ├── regression/
│   └── filters/
├── README.md
└── pyproject.toml
````
---

# Status

🚧 Under active development

Additional filters (EKF, UKF, PF), visualizations, and diagnostic studies will be added incrementally as the theoretical framework is extended.

---

# About Me

I study and implement methods in **optimization, control, robotics, Bayesian inference, and probabilistic reasoning**.

* GitHub: [https://github.com/SaiSampathKedari](https://github.com/SaiSampathKedari)
* LinkedIn: [https://linkedin.com/in/sai-sampath-kedari](https://linkedin.com/in/sai-sampath-kedari)
* X: [https://x.com/SSampathKedari](https://x.com/SSampathKedari)
* Email: [sampath@umich.edu](mailto:sampath@umich.edu)
