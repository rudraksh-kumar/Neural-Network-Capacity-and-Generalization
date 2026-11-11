# Viva Examination Question Bank — Project 9
## Neural-Network Capacity and Generalization

Concise, technically accurate answers covering all 13 core viva examination topics.

---

### 1. Backpropagation
**Q1.1: Why do we cache pre-activations $Z_1$ and hidden activations $A_1$ during forward propagation?**  
Because the local derivative $\frac{\partial \mathcal{L}}{\partial W_2} = \delta_{\text{out}}^\top A_1$ depends directly on $A_1$, and the hidden delta $\delta_1 = (\delta_{\text{out}} W_2) \odot \phi'(Z_1)$ requires $\phi'(Z_1)$. Caching them converts what would be exponential redundant recomputation into linear-time reverse-mode automatic differentiation ($O(BP)$ per step).

**Q1.2: How did you verify that your manual NumPy backpropagation equations contain no transpose, sign, or scaling bugs?**  
We evaluated central finite differences $\frac{J(\theta + \epsilon e_k) - J(\theta - \epsilon e_k)}{2\epsilon}$ ($\epsilon = 10^{-6}$ in `float64`) on every single scalar entry of $W_1, b_1, W_2, b_2$ (with and without $\lambda$) and cross-checked against PyTorch autograd. Relative errors against finite differences were $O(10^{-11}\text{–}10^{-10})$ and against PyTorch autograd $O(10^{-17}\text{–}10^{-16})$, plus a 50-step Adam optimization trajectory matched PyTorch to `rtol < 1e-8`.

**Q1.3: Why use central finite differences instead of forward finite differences?**  
Forward differences $\frac{J(\theta+\epsilon)-J(\theta)}{\epsilon}$ have $O(\epsilon)$ truncation error, whereas central differences cancel the quadratic Taylor term and achieve $O(\epsilon^2)$ truncation error ($\sim 10^{-12}$ at $\epsilon=10^{-6}$), safely above float64 roundoff noise ($\sim 10^{-16}/\epsilon \approx 10^{-10}$).

---

### 2. Activation Functions
**Q2.1: Compare `tanh` and `ReLU` in a one-hidden-layer regression network, and justify why `tanh` was chosen as primary.**  
`tanh` is $C^\infty$ smooth, bounded in $(-1, 1)$, zero-centered (reducing zig-zagging gradients), and naturally produces smooth interpolants suited for continuous regression. `ReLU` is piecewise linear, avoids saturation for $z>0$, and excels in deep nets, but suffers from dead units ($z<0$) and non-differentiable kinks ($z=0$) that create piecewise-linear corners on smooth 1D targets and require kink avoidance during finite-difference checks. In Experiment E2, `tanh` outperformed `ReLU` on both Synthetic ($R^2 = 0.9580$ vs. $0.9415$) and Concrete ($R^2 = 0.8575$ vs. $0.8410$).

**Q2.2: What happens in finite-difference gradient checking if a ReLU unit has $|z_{1,ij}| < \epsilon$?**  
Perturbing a parameter by $\pm \epsilon$ crosses the nondifferentiable kink $z_{1,ij}=0$, where the left and right derivatives differ ($0$ vs. $1$). Central differences return a blended slope while analytical subgradient picks one side, causing spurious relative errors. Our verifier explicitly checks $\min |Z_1| > 10^{-3}$ before testing ReLU.

---

### 3. Loss Functions
**Q3.1: Derive the relationship between a Gaussian observation model and Mean Squared Error.**  
Assuming $y \mid x \sim \mathcal{N}(\hat{y}(x; \theta), \sigma_y^2)$ with fixed variance $\sigma_y^2$, the negative log-likelihood over $B$ i.i.d. samples is:
$$-\log p(Y \mid X; \theta) = \frac{B}{2}\log(2\pi \sigma_y^2) + \frac{1}{2\sigma_y^2} \sum_{i=1}^B (\hat{y}_i - y_i)^2 = \text{const} + \frac{B}{2\sigma_y^2}\,\mathcal{L}_{\text{MSE}}(\theta).$$
Because $\frac{B}{2\sigma_y^2} > 0$ is independent of $\theta$, Maximum Likelihood Estimation (MLE) is mathematically equivalent to minimizing MSE.

**Q3.2: Why did you evaluate MSE, RMSE, MAE, and $R^2$ rather than MSE alone?**  
MSE matches the training objective and exposes the variance/irreducible-noise floor ($\sigma^2 = 0.1225$ on synthetic data), RMSE shares the exact physical units of the target (MPa on Concrete), MAE is less sensitive to outlier residuals, and $R^2$ measures normalized fraction of variance explained across datasets of vastly different target scales.

---

### 4. Model Capacity
**Q4.1: How do you count the exact capacity (parameter count) of a one-hidden-layer MLP?**  
$P(d, H) = \underbrace{H d}_{W_1} + \underbrace{H}_{b_1} + \underbrace{H}_{W_2} + \underbrace{1}_{b_2} = H(d + 2) + 1$. Note that parameter count alone is not the full measure of effective capacity: weight norms, activation smoothness, and implicit optimization bias also constrain effective capacity.

**Q4.2: Why does increasing hidden units $H$ not automatically improve test performance?**  
Once $H$ suffices to approximate $f^*(x)$, additional units either fit finite-sample noise (increasing variance and the generalization gap, as seen sharply on Concrete) or add computational cost without reducing structural bias. Effective generalization depends on balancing representation power against sample size $N$ and regularization.

---

### 5. Bias–Variance Tradeoff
**Q5.1: How can you recognize high bias vs. high variance from train and validation MSE curves?**  
- **High Bias (Underfitting):** Both $\text{MSE}_{\text{train}}$ and $\text{MSE}_{\text{val}}$ remain high and nearly equal (small generalization gap), e.g., Linear OLS on synthetic ($R^2 = 0.0238$).  
- **High Variance (Overfitting):** $\text{MSE}_{\text{train}}$ drops very low (even below irreducible noise $\sigma^2$ or near $0$) while $\text{MSE}_{\text{val}}$ diverges upward, producing a large generalization gap.

**Q5.2: What is irreducible error ($\sigma^2$), and what does it tell us when $\text{MSE}_{\text{train}} < \sigma^2$?**  
Irreducible error is $\operatorname{Var}(y \mid x) = \sigma^2$ ($0.1225$ on our synthetic task), the theoretical lower bound on expected test MSE achievable by the true population function $f^*(x)$. Whenever $\text{MSE}_{\text{train}} < \sigma^2$, the model has begun fitting observation noise rather than signal.

---

### 6. Generalization
**Q6.1: What is the generalization gap, and is a zero generalization gap always desirable?**  
The generalization gap is $\text{MSE}_{\text{val/test}} - \text{MSE}_{\text{train}}$. A zero gap is *not* desirable by itself: an underfitting Linear baseline has a tiny gap while having terrible test MSE! Good generalization requires **both** low training error and a small gap.

**Q6.2: Why did the 1D synthetic network tolerate large widths ($H=128, 256$) much more gracefully than Concrete ($d=8$)?**  
In 1D with standard Xavier initialization and zero initial biases ($b_1=0$), gradient descent exhibits strong **implicit spectral bias** toward smooth low-frequency functions, acting as implicit regularization. In 8D Concrete ($P = 10H+1$), $H$ hyperplanes in $\mathbb{R}^8$ provide far higher geometric degrees of freedom to interpolate noisy sample fluctuations, producing severe overfitting unless regularized.

---

### 7. Weight Decay
**Q7.1: Why do we typically exclude biases $b_1, b_2$ from L2 regularization?**  
Penalizing $b_2$ shrinks the output offset toward zero rather than the sample mean of targets, introducing unnecessary bias. Penalizing $b_1$ forces hidden transitions toward the origin $x=0$. Penalizing only weights $W_1, W_2$ controls function slope/curvature ($\|W\|_F^2$) while preserving translation invariance.

**Q7.2: Explain the mathematical difference between coupled L2 regularization and decoupled weight decay (AdamW).**  
With coupled L2, the gradient becomes $g_t^{\text{reg}} = \nabla \mathcal{L}_t + \lambda W_t$. Under Adam, this gradient is divided by the adaptive second-moment estimate $\sqrt{\hat{v}_t} + \epsilon$, so parameters with large historical gradients experience *weaker* effective weight decay. AdamW decouples the decay: $W_{t+1} = (1 - \eta \lambda_{\text{dec}}) W_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}$, applying uniform relative shrinkage independent of gradient variance.

---

### 8. Early Stopping
**Q8.1: Why is early stopping conceptually related to L2 weight decay?**  
Starting from small weights near zero, each gradient step explores a slightly larger effective radius in weight space. Stopping training at the minimum validation epoch implicitly constrains $\|W - W_0\|_2$ (effectively regularizing low-curvature directions slowly, acting similarly to Tikhonov/L2 regularization under GD without adding a penalty hyperparameter $\lambda$).

**Q8.2: Why must best-checkpoint restoration accompany patience-based early stopping?**  
During the `patience` window (e.g., 60 epochs) after the validation minimum, validation error is already degrading or oscillating upward. Stopping without restoring `best_params` returns the overfitted weights from `stopped_epoch` rather than the optimal checkpoint from `best_val_epoch`.

---

### 9. Randomized Labels
**Q9.1: What does the randomized-label experiment prove mathematically about deep/wide neural networks?**  
By keeping inputs $X_{\text{train}}$ identical while destroying all input-target mutual information via random permutation $y_{\text{train}}^\pi$, any training loss reduction reflects pure **memorization** (finite-sample Rademacher complexity / interpolation capacity). Simultaneously evaluating on unpermuted $(X_{\text{val}}, y_{\text{val}})$ yields catastrophic error, proving that low training loss alone cannot certify generalization.

**Q9.2: Why did the 1D synthetic MLP ($b_1=0$) resist memorizing random labels compared to 8D Concrete?**  
In 1D, interpolating $N$ uncorrelated random points requires steep, localized high-frequency oscillations with total variation $\mathcal{O}(N)$, whereas `b1=0` Xavier init puts all hidden thresholds at $x=0$ with small slopes—an optimization/spectral-bias barrier. In 8D Concrete, $H=256$ hyperplanes easily separate $N=25\text{–}100$ points in $\mathbb{R}^8$ and achieve near-zero training MSE (`0.0000`–`0.0017`) while validation MSE explodes.

---

### 10. Optimization
**Q10.1: How can you distinguish an optimization failure from inadequate model capacity?**  
An **optimization failure** occurs when a model *possesses* enough theoretical capacity to fit the training set, yet fails to reduce training error due to poor learning rate, ill-conditioned curvature, bad initialization, or insufficient budget—proven when changing LR, optimizer (e.g. GD $\to$ Adam), or initialization makes training MSE plummet. **Inadequate representation** cannot be cured by better optimization: training MSE stays high across LRs/optimizers until capacity ($H$ or nonlinearity) increases.

---

### 11. Computational Budgets
**Q11.1: Why is comparing models at fixed epochs/steps unfair across different widths $H$?**  
Each forward+backward step scales approximately as $6 B P(d, H)$ FLOPs. A width $H=256$ model performs $\sim 60\times$ more arithmetic operations per step than $H=4$. Therefore, **Protocol A** must examine both fixed optimization steps and **matched total FLOPs** (giving smaller models proportionally more updates), alongside **Protocol B** (convergence criterion + max budget).

---

### 12. Dataset Leakage
**Q12.1: How could scaling before train/val/test splitting silently corrupt an experiment?**  
Fitting `StandardScaler` on the full dataset leaks validation/test mean and variance into training preprocessing and evaluation. In our Concrete pipeline, splitting occurs first by index; then BOTH feature scaler and target scaler are fitted **exclusively** on the active training split (even inside subsampled $N$-sweeps), and test samples are completely locked out of width/regularization selection.

---

### 13. Experimental Reproducibility
**Q13.1: What safeguards guarantee that every table and plot in this project is strictly reproducible?**  
All configurations, grids, and seeds live in `config/experiment_config.py`; synthetic train/val/test splits use separate `np.random.default_rng` seeds; Concrete raw CSV carries a verified SHA-256 checksum; deterministic seed loops record exact JSON logs in `results/experiment_logs/`; and `scripts/run_all_experiments.py` regenerates every log, table, and figure from scratch on a single command.

