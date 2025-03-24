# Roadmap: Hyperparameter Optimization for Neural Networks

Following the author's supervisor review feedback, this document outlines the progression of experiments and future plans for benchmarking hyperparameter optimization methods for deep learning.

---

## Completed Stages

- [x] Implemented Standard DE and Adaptive DE for MLPs
- [x] Added Bayesian-assisted DE with Random Forest surrogate
- [x] Integrated Optuna for Bayesian optimization
- [x] Developed Hybrid DE + Optuna approach
- [x] Enabled automated multi-run evaluation with statistical analysis
- [x] Evaluated on Fashion-MNIST and CIFAR-10
- [x] Analyzed convergence, runtime, and mutation strategies
- [x] Parallelized optimization runs

---

## Current Focus

- [x] Run Optuna on **TinyImageNet**
  - CNN-based model
  - TensorBoard integration
  - Logging and saving best models
- [ ] Add DE and Hybrid methods for TinyImageNet
- [ ] Enhance search space and callback metrics

---

## Next Milestones

### Extended Dataset Benchmarks
- [ ] Add CIFAR-100 (higher complexity, 100 classes)
- [ ] (Optional) Add **EuroSAT** (remote sensing dataset)

### Methodological Improvements
- [ ] Compare with **Hyperband**, **BOHB**, and **Population-Based Training (PBT)**
- [ ] Add **multi-objective optimization**: accuracy + training time
- [ ] Try alternative surrogate models (e.g., XGBoost, MLP)
- [ ] Explore more advanced DE variants
- [ ] Analyze convergence behavior in detail

### Reproducibility and Logging
- [ ] Improve experiment reproducibility via config files
- [ ] Organize results in structured logs (JSON, CSV)
- [ ] Export all best trials and metrics for visualization

---

## Paper Writing Support

This roadmap supports the **experimental design**, **scaling rationale**, and **future work** sections of the paper or thesis.

