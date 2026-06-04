# Report outline (NeurIPS format, 5 pages max)

This is a **suggested structure** for the report, mapped to the assignment
requirements. The report is **individual** (one per person, even when the code
was done in a pair). This file is only a scaffold; the actual report is in
`report.tex` / `report.pdf`.

> **Important:** the GitHub/GitLab repository link must appear **in the
> abstract** of the report.

---

## Abstract (~5 to 8 lines)
- One sentence on the paper and what it proposes.
- One sentence on what you reproduced (Figure 1, Table 1 protocol, ablation).
- The link to this repository.

## 1. Summary of the paper (1 to 2 pages)
- **Problem.** NNs are overconfident; quantifying predictive uncertainty
  matters for real applications. Two evaluation angles: calibration, and
  generalisation of uncertainty to out-of-distribution / dataset shift.
- **Method (the recipe).**
  - Proper scoring rules as the training criterion. For regression, the
    Gaussian NLL (Eq. 1), with a network that outputs both mean *and* variance.
  - Ensembles: M independently trained networks, combined as a uniformly
    weighted mixture (approximated by a single Gaussian).
  - Adversarial training (FGSM) as an optional smoothing step.
- **Why it is interesting.** A non-Bayesian, simple, parallelisable alternative
  to Bayesian NNs that matches or beats them.
- **Key results in the paper.** Toy regression (Fig. 1), UCI regression
  (Table 1), classification + out-of-distribution detection, ImageNet.

## 2. Reproduced results (1 to 2 pages)
- **Experimental setup.** Architectures, hyper-parameters, what you changed.
- **Experiment 1, toy regression (Figure 1).** Show your reproduced figure,
  comment panel by panel, confirm the uncertainty grows away from the data.
- **Experiment 2, UCI regression (Table 1).** Report your RMSE / NLL table next
  to the paper's values, show the ensemble-vs-single ablation.
- Be explicit about what matches and what does not.

## 3. Discussion (1 page)
- **What worked well.** The method really is simple; the toy result is clean.
- **What was challenging.** For example the learning-rate discrepancy (paper
  says 0.1, we needed 0.01 on UCI), computing the NLL on the original target
  scale, standardisation done per fold.
- **What could be improved.** More datasets, 20 folds, the MNIST/NotMNIST
  out-of-distribution experiment, calibration curves.

## Appendix (outside the 5 pages, optional)
- Extra figures, full hyper-parameter tables, per-dataset results.

## Required section (outside the 5 pages)
- **AI use.** Disclose any AI tools used and *how* (for example "used to
  scaffold the repository structure and draft docstrings; all code was
  reviewed, tested, and modified by me; the learning-rate finding was
  discovered and verified by me"). Be specific and honest. This is the section
  that keeps you compliant with the course policy on LLMs.

> Note: a work-distribution / contributions statement is **not** required by the
> assignment, and the report is individual, so it is omitted.
