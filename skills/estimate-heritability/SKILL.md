---
name: estimate-heritability
description: Estimate SNP heritability (h2, the variance explained by a kinship matrix) with pylmm's null LMM fit - REML vs ML, reading h2 and the genetic and residual variances, rescaling the kinship so h2 is comparable with published values, boundary estimates at 0 and the 0.99 cap, a likelihood-ratio test of h2 = 0, and standard errors and confidence intervals from the profile likelihood over the h grid or a parametric bootstrap. Use when asked for heritability, variance components, h2 or PVE with pylmm, or to compare pylmm's h2 with another tool or a paper.
---

# Estimate heritability with pylmm

pylmm's null model is `Y = X0 b + u + e`, `u ~ N(0, h s2 K)`, `e ~ N(0, (1 - h) s2 I)`. `LMM(Y, K,
X0=...).fit()` returns `h` (the heritability on the scale of K), the coefficients, `s2` and the
log-likelihood. The blocks below run in CI in order (paths relative to the repository root).

## 1. Fit

A simulated example with known h² = 0.5 (300 individuals in 100 families of 3, plus a covariate):

```python
import numpy as np
from scipy import linalg, stats
from pylmm.lmm import LMM, calculate_kinship

rng = np.random.default_rng(3)
n_fam, n_sib, m = 100, 3, 3000
p = rng.uniform(0.05, 0.5, m)
parents = rng.binomial(1, p, size=(n_fam, 4, m))          # 4 haplotypes per family
kids = [parents[:, rng.integers(0, 2), :] + parents[:, 2 + rng.integers(0, 2), :] for _ in range(n_sib)]
G = np.concatenate(kids).astype(float)                    # (300, m) allele counts
n = G.shape[0]
K = calculate_kinship(G)
kva, kve = linalg.eigh(K)
age = rng.normal(50, 10, n)
u = kve @ (np.sqrt(np.clip(kva, 0, None)) * rng.standard_normal(n)) * np.sqrt(0.5)
Y = 2.0 + 0.03 * age + u + rng.normal(scale=np.sqrt(0.5), size=n)

X0 = np.column_stack([np.ones(n), age])                   # ALWAYS include the intercept column
model = LMM(Y, K, kva, kve, X0=X0)
h, beta, s2, ll = model.fit()                             # REML (the default)
print("h2=%.3f  s2=%.3f  sigma_g2=%.3f  sigma_e2=%.3f" % (h, s2, h * s2, (1 - h) * s2))
assert 0.25 < h < 0.75
```

* `h` is a numpy float in [0, 0.99]; `s2` (= `model.opt_sigma`) is the total variance, so the genetic
  variance is `h * s2` and the residual `(1 - h) * s2`. `beta` is (q, 1), in the column order of `X0`.
* Missing phenotypes (`nan`) are dropped by `LMM`; `X0` must be complete on the others.
* `ll` is the REML log-likelihood including constants; compare log-likelihoods only between fits of the
  same data, `X0` and method.

## 2. REML or ML

```python
h_ml = LMM(Y, K, kva, kve, X0=X0).fit(reml=False)[0]
print("REML %.3f   ML %.3f" % (h, h_ml))
```

Use REML (the default; it accounts for the fitted fixed effects, and published LMM h² are almost always
REML). ML is biased downward in small samples and, for a centred K built from more SNPs than individuals
(the usual outbred GRM), its optimum can run to the boundary: there is no MLE, and pylmm returns ~0.99
(Phase 4, bug 17).

## 3. Making h2 comparable

`h` is a variance ratio on the scale of K. It equals the proportion of phenotypic variance explained only
when K is scaled so that `trace(P K P) / (n - 1) = 1` (`P = I - 11'/n`, Gower centring; a mean diagonal of
1 is the usual approximation). Rescaling K by `c` changes `h` exactly as `h_c = h c / (h c + 1 - h)` and
leaves every test statistic unchanged:

```python
P = np.eye(n) - 1.0 / n
c = np.trace(P @ K @ P) / (n - 1)
h_gower = LMM(Y, K / c, X0=X0).fit()[0]
print("c=%.4f  h(K)=%.4f  h(K/c)=%.4f  formula=%.4f" % (c, h, h_gower, h * c / (h * c + 1 - h)))
assert abs(h_gower - h * c / (h * c + 1 - h)) < 1e-4
```

When comparing with another tool or a paper, check:

* **The kinship.** pylmm's `calculate_kinship` standardises every SNP (mean diagonal 1 when no SNP is
  monomorphic and nothing is missing). `pylmm kinship` gives a slightly smaller K with missing calls, and
  both shrink K when SNPs are monomorphic (bug 8). GCTA's GRM, GEMMA's `-gk 1` (centred) and `-gk 2`
  (standardised) and plink2's `--make-rel` use other scalings; in inbred lines a GCTA-style GRM has
  diagonal ≈ 1 + F ≈ 2. Rescale to mean diagonal 1 (or Gower) before comparing, or report both.
* **The unit of observation.** In panels of inbred strains with replicate animals, h² of individual
  animals (what you get from individual-level data) is lower than h² of strain means (broad-sense, from
  averaging replicates). Fit the same unit as the paper.
* **Covariates.** h² is the share of the variance left after `X0`; adding covariates changes it.
* **Method.** REML vs ML, and whether the paper reports PVE (GEMMA) or h² on the observed scale.

## 4. Boundaries and the h = 0 test

The h search is a grid 0, 0.01, ..., 0.99 plus a bounded refinement (Phase 6.3 will extend it to 1):

* `h == 0`: no detectable genetic variance (common for small n or weak K).
* `h == 0.99`: "0.98 or more", capped; true values ≥ 0.98 are all reported as 0.99.

Test h² = 0 by a likelihood ratio; under the null the statistic is a 50:50 mixture of χ²₀ and χ²₁:

```python
ll0 = model.log_likelihood(0.0).ll                       # REML log-likelihood at h = 0
lrt = 2 * (model.opt_ll - ll0)
p_h0 = 0.5 * stats.chi2.sf(lrt, 1) if lrt > 0 else 1.0
print("LRT=%.1f  p(h2=0)=%.2e" % (lrt, p_h0))
```

## 5. Standard errors and intervals

**Profile likelihood** (preferred near a boundary): evaluate the REML log-likelihood on a fine grid of
`h` and keep the values within `chi2(1, 0.95) / 2 = 1.92` of the maximum.

```python
grid = np.linspace(0, 0.999, 1000)
prof = np.array([model.log_likelihood(g).ll for g in grid])
inside = grid[prof >= prof.max() - stats.chi2.ppf(0.95, 1) / 2]
print("95%% profile CI: [%.3f, %.3f]" % (inside.min(), inside.max()))
assert inside.min() < h < inside.max()
```

**Curvature SE** (interior estimates only): `se = 1 / sqrt(-d²ll/dh²)` at `h`:

```python
d = 1e-3
d2 = (model.log_likelihood(h + d).ll - 2 * model.opt_ll + model.log_likelihood(h - d).ll) / d**2
print("SE(h2) = %.3f" % (1 / np.sqrt(-d2)))
```

**Grid posterior**: `model.mean_and_var()` returns the mean and variance of `h` under the normalised grid
likelihood (a flat prior on the 100-point grid); `model.plot_fit()` draws it (needs matplotlib). Its SD is
a quick summary, not a frequentist SE.

**Parametric bootstrap** (slow but honest, also at the boundary): simulate from the fitted model, refit,
and take the SD and quantiles of the refitted `h`. Reuse the eigendecomposition:

```python
b_hat = model.opt_beta[:, 0]
boot = []
for _ in range(30):
    u_b = kve @ (np.sqrt(np.clip(kva, 0, None)) * rng.standard_normal(n)) * np.sqrt(h * s2)
    y_b = X0 @ b_hat + u_b + rng.normal(scale=np.sqrt((1 - h) * s2), size=n)
    boot.append(LMM(y_b, K, kva, kve, X0=X0).fit()[0])
print("bootstrap SD %.3f (30 replicates; use 200+)" % np.std(boot, ddof=1))
```

## 6. More

* Two kinships (e.g. two marker sets, or K plus a shared-environment matrix): `pylmm.misc.fit_two(Y, K1,
  K2, X0=X0)` returns the h² of each (`h1, h2, e, beta, sigma, ll`); it costs one eigendecomposition per
  weight grid point.
* Many phenotypes on the same individuals: compute `kva, kve` once and pass them to every `LMM`. If a
  phenotype has missing values, `LMM` subsets K and recomputes them for that phenotype.
* From files: read `Y` and `X0` with `PlinkReader` (`read_phenos`, `read_covariates`) and K with
  `PlinkReader(prefix, k_file=...)`, which aligns K by its `.id` file (see `run-gwas`).
* `pylmm gwas -v` logs the null fit as `null model: heritability=..., sigma=...` (3 decimals).
