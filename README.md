# Bayesian Filtering and Smoothing

This repository studies **Bayesian filtering and smoothing for systems whose internal states are not directly observable and evolve over time under uncertainty**.

In robotics and related domains, system behavior is governed by dynamics that are only approximately known, while sensors provide indirect and noisy measurements. As a result, the central problem is not simulation, but **inference**: how to reason about hidden states when both the model and observations are uncertain.

This repository focuses on how Bayesian filtering and smoothing methods combine:
- system dynamics,
- sensor models,
- and uncertainty in states and noise,

to produce probabilistic estimates of system behavior over time. Emphasis is placed not only on how these methods work, but also on **why they fail**, including the effects of model mismatch, noise mis-specification, and accumulated uncertainty.

**Topics (in progress):**
- Kalman Filter (KF)
- Extended Kalman Filter (EKF)
- Unscented Kalman Filter (UKF)
- Particle Filter (PF)
- RTS and fixed-lag smoothing
- Practical diagnostics and failure cases

🚧 This repository is under active development. Code, experiments, and explanations are added incrementally as the theoretical framework is instantiated for concrete systems.

**Reference:**  
Simo Särkkä, *Bayesian Filtering and Smoothing*

## About Me

I am interested in **probabilistic methods for robots**, with an emphasis on **state estimation, control, and uncertainty quantification**.

Feel free to reach out:
- **Email:** sampath@umich.edu  
- **LinkedIn:** https://www.linkedin.com/in/sai-sampath-kedari
