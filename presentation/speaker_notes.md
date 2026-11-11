# Six-Member Presentation & Speaker Distribution — Project 9
## Neural-Network Capacity and Generalization (20 Slides, ~20 Minutes)

---

## Slide Allocation & Speaker Roster

| Member | Assigned Slides | Section Theme | Target Time |
| :--- | :--- | :--- | :---: |
| **Member 1** | Slides 1, 4, 5, 6 | Title, Architecture, Math Derivations, Backprop & Gradient Verification | ~3.5 min |
| **Member 2** | Slides 2, 3, 7, 8 | Motivation, Research Questions, Datasets & Leakage Control, Linear Baseline | ~3.0 min |
| **Member 3** | Slides 9, 10, 18 | Hidden-Width, Training-Size, 5-Fold CV & Double-Descent Depth Sweeps | ~3.5 min |
| **Member 4** | Slides 11, 12, 17 | Regularization, Randomized-Label Memorization, Warmup & Mini-Batch Extensions | ~3.5 min |
| **Member 5** | Slides 13, 14 | Computational Budgets (Protocols A & B) & Diagnostic Interventions | ~3.0 min |
| **Member 6** | Slides 15, 16, 19, 20 | Concrete Results, Consolidated Findings, Limitations & Conclusion | ~3.5 min |

---

## Slide-by-Slide Guide, Key Equations & Expected Questions

### Member 1 — Mathematical Foundations & Backpropagation
- **Slide 1 — Title & Team Overview:** Introduce Project 9 title, core research goal (separating memorization, representation capacity, overfitting, and optimization failures), and six-member workstream structure.
- **Slide 4 — One-Hidden-Layer Architecture:** Present $z_1 = W_1 x + b_1$, $h = \phi(z_1)$, $\hat{y} = W_2 h + b_2$, matrix shapes ($W_1 \in \mathbb{R}^{H\times d}$, $W_2 \in \mathbb{R}^{1\times H}$), parameter count $P = H(d+2)+1$, and `tanh` vs `ReLU` comparison.
- **Slide 5 — Gaussian Observation Model & Loss:** Explain $y \mid x \sim \mathcal{N}(\hat{y}(x;\theta), \sigma_y^2)$ leading directly to MLE $\equiv$ MSE minimization, plus coupled L2 regularization excluding biases.
- **Slide 6 — Backpropagation & Gradient Verification (`fig02`):** Walk through $\delta_{\text{out}} = \frac{2}{B}(\hat{y}-y)$ and $\delta_1 = (\delta_{\text{out}} W_2) \odot \phi'(Z_1)$; highlight central finite-difference verification ($O(10^{-11}\text{–}10^{-10})$ relative error) and PyTorch autograd equivalence ($O(10^{-17})$).
- **Possible Faculty Questions:** *Why central vs forward finite differences? Why avoid ReLU kinks during gradient checks? Why exclude biases from L2?*

### Member 2 — Motivation, Datasets & Linear Baseline
- **Slide 2 — Motivation: Capacity vs Generalization:** Explain why low training error does not guarantee unseen generalization and contrast classical U-shaped bias-variance curves with overparameterized behavior.
- **Slide 3 — Central Research Questions:** State the 10 research questions structured around capacity, sample size, regularization, memorization, compute fairness, and diagnostics.
- **Slide 7 — Synthetic & UCI Concrete Datasets (`fig01`):** Show $f^*(x) = 2\sin(2x) + 0.5x^2$ ($\sigma=0.35$, Bayes MSE $=0.1225$) and UCI Concrete ($N=1030, d=8$, MPa), emphasizing strict train-only scaling and locked test split.
- **Slide 8 — Linear Baseline vs 1-HL MLP (`fig03`):** Show OLS failing completely on synthetic nonlinear structure ($R^2 = 0.0238$) and plateauing at $R^2 = 0.5752$ ($\text{RMSE} = 10.59\text{ MPa}$) on Concrete, while $H=32$ `tanh` MLP achieves $R^2 = 0.9580$ (synthetic) and $R^2 = 0.8575$ (Concrete).
- **Possible Faculty Questions:** *Why choose Concrete over Energy Efficiency? How did you prevent scaling leakage across sample-size sweeps?*

### Member 3 — Capacity & Sample-Size Experiments
- **Slide 9 — Hidden-Layer Width Experiments (`fig04`, `fig05`, `fig07`):** Present width sweep $H \in \{4,\dots,256\}$ ($\mu \pm \sigma$ across seeds), generalization gap vs $P/N$, and contrast 1D smooth behavior vs 8D Concrete overfitting.
- **Slide 10 — Training-Set Size Experiments (`fig06`):** Present learning curves across $N \in \{25,\dots,800\}$ (synthetic) and $\{25,\dots,618\}$ (Concrete) at fixed $H=32$, showing how increasing $N$ collapses variance and shrinks the generalization gap.
- **Possible Faculty Questions:** *Why does test error not always diverge sharply at $H=256$ on 1D synthetic? Why hold $H$ fixed during $N$-sweeps?*

### Member 4 — Regularization & Randomized-Label Memorization
- **Slide 11 — Unregularized vs Weight Decay vs Early Stopping (`fig08`, `fig09`):** Contrast unregularized $H=128$ overfitting against coupled L2 ($\lambda \in [10^{-4}, 10^{-1}]$), decoupled AdamW, bias-exclusion ablation, and validation early stopping with best-checkpoint restoration.
- **Slide 12 — Randomized-Label Memorization Experiment (`fig10`):** Contrast normal vs independently permuted training labels (with unpermuted val/test), highlighting 8D Concrete exact memorization vs 1D spectral-bias resistance, and clearly defining memorization vs generalization.
- **Possible Faculty Questions:** *How do Adam+L2 and AdamW differ mathematically? Why did Concrete memorize to ~0 training MSE while 1D synthetic resisted?*

### Member 5 — Computational Budgets & Diagnostics
- **Slide 13 — Computational Budget Protocols (`fig11`):** Explain **Protocol A** (fixed 2000 steps vs matched total FLOPs $6BP$) and **Protocol B** (validation convergence criterion + max budget), showing how accounting for parameter count changes model rankings.
- **Slide 14 — Controlled Failure-Mode Diagnostics (`fig12` & Table):** Present the three-way controlled intervention matrix separating **Optimization Failure** (LR/optimizer/init), **Overfitting** (L2/early stopping/data), and **Inadequate Representation** (linear/narrow width).
- **Possible Faculty Questions:** *Why is fixed-step comparison biased toward wide models? How do you experimentally prove a high error is optimization failure rather than underfitting?*

### Member 6 — Real-World Synthesis, Extensions & Conclusion
- **Slide 15 — Concrete Test Predictions & Residuals (`fig13`):** Show predicted vs actual MPa scatter and residual plots comparing Linear OLS against the selected regularized MLP on the untouched test set.
- **Slide 16 — Consolidated Test Comparison & Key Findings (`fig14`):** Compare Linear, Best-Width, Unregularized $H=128$, Early-Stopped, and Regularized MLPs across both datasets, synthesizing all 10 research questions.
- **Slide 17 (Member 4) — Extensions I: Warmup ES & Mini-Batch SGD (`fig15`, `fig16`):** Explain how `warmup=100` eliminates saddle-plateau stopping (`0.74 -> 0.196`) and `Momentum B=16` reaches Concrete Test $R^2=0.9008$.
- **Slide 18 (Member 3) — Extensions II: 5-Fold CV & Double Descent (`fig17`, `fig18`):** Show 5-fold RMSE `5.56 ± 0.56 MPa` (`R²=0.887`) and $P/N=16.57$ double-descent easing + 2-HL comparison.
- **Slide 19 — Limitations & Future Scope:** Discuss depth bounds beyond 2-HL, homoscedastic variance assumptions, and unregularized 2-HL extensions.
- **Slide 20 — Conclusion & Reproducibility Audit:** Summarize core takeaways and confirm zero leakage, gradient verification, and full pipeline reproducibility.

