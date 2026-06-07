# Reproduction: Deep Ensembles for Predictive Uncertainty

Reproduction of:

> B. Lakshminarayanan, A. Pritzel, C. Blundell.
> **Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.**
> *NeurIPS 2017.*

Course assignment for **ASI** (Advanced Statistical Inference), paper
reproduction project.

**Authors:** Theodoris Donval & Lucas Aliome (EURECOM). See the *Contributions*
section of the report for the breakdown of the work.

---

## What this repository reproduces

The paper proposes a simple, non-Bayesian recipe for predictive uncertainty:
(1) train a *probabilistic* network with a **proper scoring rule**, (2) train an
**ensemble** of them, and (3) optionally add **adversarial training**.

This repository reproduces the two regression experiments.

| Notebook | Paper section | What it reproduces |
|----------|---------------|--------------------|
| `notebooks/01_toy_regression.ipynb` | Section 3.2, **Figure 1** | The 1-D toy task `y = x³ + N(0,3²)`: MSE baseline, NLL, NLL+adversarial, and ensemble. Shows uncertainty growing away from the data. |
| `notebooks/02_uci_regression.ipynb` | Section 3.3, **Table 1** | The k-fold RMSE / NLL evaluation protocol on UCI regression benchmarks, plus an ablation (single network, ensemble, ensemble + adversarial training). |

The ImageNet and large-scale classification experiments (Section 3.4) are out
of scope for a course project and are not reproduced here.

---

## Repository structure

```
deep-ensembles-reproduction/
├── README.md
├── requirements.txt
├── src/                          # the reproduction code (importable package)
│   ├── models.py                 # GaussianMLP (outputs mean + variance), PlainMLP
│   ├── losses.py                 # Gaussian NLL, the proper scoring rule (Eq. 1)
│   ├── adversarial.py            # FGSM adversarial training (Section 2.3)
│   ├── ensemble.py               # DeepEnsemble: trains M nets, mixture prediction
│   ├── baselines.py              # MSEEnsemble, the MSE empirical-variance baseline
│   ├── metrics.py                # RMSE and NLL on the original target scale
│   ├── evaluate.py               # k-fold evaluation protocol (Table 1)
│   └── data.py                   # toy dataset + UCI dataset loaders
├── notebooks/
│   ├── 01_toy_regression.ipynb   # reproduces Figure 1  (executed)
│   └── 02_uci_regression.ipynb   # reproduces Table 1   (executed on a demo dataset)
├── results/
│   └── figure1_toy_regression.png
└── report/
    └── report_outline.md         # suggested structure for the NeurIPS report
```

The code lives in `src/` as a small package, and the notebooks only orchestrate
and visualise. This keeps the logic testable and avoids duplicating code across
notebooks.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

A GPU is **not** required. Every experiment here runs on CPU in minutes, since
the networks are small MLPs.

## Running

```bash
jupyter notebook
```

then open the notebooks in `notebooks/` and run all cells.

* **Notebook 1** is fully self-contained and runs offline.
* **Notebook 2** is executed here on the `diabetes` dataset (ships with
  scikit-learn, no download). To reproduce the paper's **Table 1**, edit the
  configuration cell: set `DATASETS` to the UCI dataset names and `N_FOLDS = 20`.
  The UCI files are downloaded automatically on first use (internet required).

---

## Notes and reproduction findings

* The paper's stated Adam learning rate of `0.1` (Section 3.1) was unstable on
  the UCI regression benchmarks in our runs. A rate of `0.01` reproduces the
  paper's qualitative behaviour. See the discussion in notebook 2.
* Exact numbers differ slightly from the paper because of framework differences
  (PyTorch vs. the original Torch implementation), random initialisation, and,
  in the executed demo, a reduced number of folds.

---

## Report

The report (NeurIPS format, 5 pages) is in `report/`:

* `report/report.tex` is the LaTeX source (NeurIPS 2023 style).
* `report/report.pdf` is the compiled report.
* `report/neurips_2023.sty` is the style file (so it compiles standalone).
* `report/report_outline.md` is the original scaffold with structure notes.

Compile with `pdflatex report.tex` (run it twice for cross-references), or
upload `report.tex` + `neurips_2023.sty` + `figure1_toy_regression.png` to
Overleaf.
