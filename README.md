Hyperparameter Optimization: Benchmarking DE, Bayesian-assisted DE, Optuna, and Hybrid DE + Optuna

This repository contains the code, datasets, and results for the research study Benchmarking Differential Evolution, Bayesian-assisted DE, and Optuna for Hyperparameter Optimization of MLP Classifiers.
The study compares the performance of Standard DE, Bayesian-assisted DE, Optuna, and Hybrid DE + Optuna for tuning Multi-Layer Perceptron (MLP) classifiers on Fashion-MNIST and CIFAR-10.  

---

Overview

Implemented Optimization Methods:  
Standard Differential Evolution (DE) – Evolutionary algorithm for global search.  
Bayesian-assisted DE – Uses a surrogate model (Random Forest) to reduce unnecessary function evaluations.  
Optuna (Bayesian Optimization with TPE) – Efficient hyperparameter tuning with trial pruning.  
Hybrid DE + Optuna – Combines DE’s global exploration with Optuna’s local refinement.  

Datasets Used:  
Fashion-MNIST – 28×28 grayscale images of clothing items (10 classes).  
CIFAR-10 – 32×32 RGB images of objects (10 classes, reduced to 100 PCA components for efficiency).  

---
Installation & Setup  

 Clone the Repository  
```bash
git clone https://github.com/https://github.com/igiraneza26/DE-Optuna-benchmarking-neural-nextwork-hyperparameter-otpimization.git
cd your-repo-link
```
Create a Virtual Environment (Optional but Recommended)  
```bash
python -m venv venv
source venv/bin/activate   For Linux/macOS
venv\Scripts\activate      For Windows
```
Run the Main Experiment  
```bash
python main.py
```
---

Experimental Settings  

| Hyperparameter  | Search Space         |
|--------------------|------------------------|
| Learning Rate     | 0.0001 – 0.1            |
| Hidden Layer Size | 10 – 200 neurons        |
| Dropout Rate     | 0.1 – 0.5                |

DE Settings: Population = 10, Generations = 20, Mutation Strategies (`rand1bin`, `rand2bin`, `currenttobest1bin`).  
Optuna Settings: 50 trials per run, Pruning enabled.  
Hybrid DE + Optuna: DE for exploration, Optuna for refinement (20 trials).  

---

 Results Summary  

| Method            | Fashion-MNIST Accuracy | CIFAR-10 Accuracy | Runtime (sec) |
|----------------------|-------------------------|----------------------|------------------|
| Standard DE         | 83.96% ± 0.48%           | 40.87% ± 0.35%       | 1006.98 sec      |
| Bayesian-assisted DE | 84.31% ± 0.33%           | 40.98% ± 0.22%       | 958.67 sec       |
| Optuna (TPE)        | 84.99% ± 0.09%           | 41.32% ± 0.19%       | 119.06 sec       |
| Hybrid DE + Optuna  | 84.74% ± 0.20%           | 40.79% ± 0.23%       | 569.31 sec       |

Key Findings:  
- Optuna achieved the highest accuracy and fastest convergence due to its adaptive Bayesian search and trial pruning.  
- Bayesian-assisted DE improved efficiency compared to Standard DE, but the surrogate model introduced overhead.  
- Hybrid DE + Optuna did not outperform Optuna alone, suggesting DE-based global exploration was unnecessary for deep learning tasks.  

---
How to Cite  

If you use this repository for your research, please cite:  

```
@article{YourPaper2024,
  title={Benchmarking Differential Evolution and Optuna for Neural Network Hyperparameter Optimization},
  author={Berthille Igiraneza},
  journal={Your Journal / Conference Name},
  year={},
  url={https://github.com/your-repo-link}
}
```
