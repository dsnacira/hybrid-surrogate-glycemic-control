# Title
**Simulation-Efficient Surrogate-Assisted Evolutionary Optimization
for Glycemic Control Using the UVA/Padova Simulator**

## Overview
The proposed framework combines:

- Genetic Algorithm (GA)-based optimization
- Random Forest surrogate modeling
- Selective high-fidelity physiological simulation
- Safety-aware fitness evaluation
- 
The proposed framework combines:

- Genetic Algorithm (GA)-based optimization
- Random Forest surrogate modeling
- Selective high-fidelity physiological simulation
- Safety-aware fitness evaluation

## Repository Structure
text
notebooks/
│
├── 01_experiments.ipynb
├── 02_results_summaryfixed.ipynb
└── 06_ablation_study_updated.ipynb
outputs/
│
├── figures/
└── results/

requirements.txt
README.md

## Main Notebooks
01_experiments.ipynb
Runs the optimization experiments and generates raw results.

02_results_summaryfixed.ipynb
Generates the final tables and figures reported in the manuscript.

06_ablation_study_updated.ipynb
Performs the component-wise ablation analysis of:
surrogate-assisted evaluation,
safety-aware fitness evaluation,
selective high-fidelity validation.

## Reproducibility
The experiments were tested using:

Python 3.10
Google Colab
Jupyter Notebook

Random seeds are fixed for reproducibility.

## Installation
Install dependencies using:

pip install -r requirements.txt

## Notes
The current implementation uses a simplified experimental setup
designed for reproducibility and computational efficiency.

The UVA/Padova simulator itself is not redistributed in this repository.
## Citation

## License
This project is provided for academic and research purposes