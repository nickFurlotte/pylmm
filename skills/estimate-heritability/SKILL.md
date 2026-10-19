---
name: estimate-heritability
description: Estimate SNP heritability (h2, the variance explained by a kinship matrix) with pylmm's null LMM fit - REML vs ML, reading h2 and the genetic and residual variances, putting h2 on the kinship scale of another tool or paper (GCTA-style GRMs, PVE), individual-level vs strain-mean heritability in inbred panels with replicates, boundary estimates at 0 and the 0.99 cap, a likelihood-ratio test of h2 = 0, and standard errors and confidence intervals from the profile likelihood or a parametric bootstrap. Use when asked for heritability, variance components, h2 or PVE with pylmm, or to compare pylmm's h2 with another tool or a paper.
---

# Estimate heritability with pylmm

pylmm's null model is `Y = X0 b + u + e`, `u ~ N(0, h s2 K)`, `e ~ N(0, (1 - h) s2 I)`. `LMM(Y, K,
X0=...).fit()` returns `h` (the heritability on the scale of K), the coefficients, `s2` and the
log-likelihood. The blocks below run in CI in order (paths relative to the repository root).

## 1. Fit

A simulated example with known h² = 0.5 (300 individuals in 100 families of 3, plus a covariate):

```python
import itertools
import numpy as np
from scipy import linalg, stats
from pylmm.lmm import LMM, calculate_kinship
from pylmm.plink_io import PlinkReader

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
REML). ML ignores the degrees of freedom used by the fixed effects, so its variance components are biased (in 10
simulations of 300 unrelated people with true h² of 0.3 and 0.8, ML gave a higher h than REML every time, by
0.03 to 0.2) and, for a centred K built from more SNPs than individuals
(the usual outbred GRM), its optimum can run to the boundary: there is no MLE, and pylmm returns ~0.99
(Phase 4, bug 17).

## 3. Making h2 comparable

`h` is a variance ratio on the scale of K: `h / (1 - h)` is the genetic variance per unit of K over the
residual variance. Dividing K by a constant `c` changes `h` exactly as `h_c = h c / (h c + 1 - h)` and
leaves every test statistic unchanged:

```python
c = 2.0
h_c = LMM(Y, K / c, kva / c, kve, X0=X0).fit()[0]
print("h(K)=%.4f  h(K/%g)=%.4f  formula=%.4f" % (h, c, h_c, h * c / (h * c + 1 - h)))
assert abs(h_c - h * c / (h * c + 1 - h)) < 1e-3
```

So two tools report the same h² only if their kinships have the same scale. To compare pylmm's `h` with
the h² of another tool or a paper, **put both on the same K scale**: if their K ≈ pylmm's K / c, their
h² corresponds to `h' = h c / (h c + 1 - h)` with

```text
c = mean diag(pylmm's K) / mean diag(their K)
```

(for a K that is not proportional to pylmm's, the ratio of mean diagonals is the usual approximation).
Getting their K's mean diagonal:

* **Compute their K** from the same genotypes when you can (the tool itself, or the formula). GCTA's GRM
  (also plink2 `--make-grm-bin`) has diagonal `1 + F`: **≈ 1 in outbred samples, ≈ 2 in inbred lines**
  (F ≈ 1; 1.99 on the bundled mice). GEMMA `-gk 2` (standardised) is ≈ 1 when outbred; GEMMA `-gk 1`
  (centred, not standardised) has diagonal ≈ the mean of `2p(1-p)(1+F)` over SNPs, well below 1:
  compute it.
* pylmm's `calculate_kinship` / `pylmm kinship` have mean diagonal ≈ 1 whatever the inbreeding (every SNP
  is standardised over the sample), a little less with missing calls or monomorphic SNPs (bug 8).

```python
def gcta_grm(G):
    """GCTA's GRM (Yang et al. 2011) of complete 0/1/2 allele counts."""
    p = G.mean(axis=0) / 2
    keep = (p > 0) & (p < 1)
    G, p = G[:, keep], p[keep]
    Z = (G - 2 * p) / np.sqrt(2 * p * (1 - p))
    A = Z @ Z.T / G.shape[1]
    np.fill_diagonal(A, 1 + np.mean((G**2 - (1 + 2 * p) * G + 2 * p**2) / (2 * p * (1 - p)), axis=1))
    return A

# the bundled inbred mice: 2000 SNPs, those without missing calls
with PlinkReader("data/snps.132k.clean.noX", norm_genotype=False) as r:
    G_mice = 2 * np.column_stack([g for g, _ in itertools.islice(r, 2000)])   # A2 fraction -> count
G_mice = G_mice[:, ~np.isnan(G_mice).any(axis=0)]
K_mice = calculate_kinship(G_mice)                        # pylmm's K: mean diagonal ~1
K_gcta = gcta_grm(G_mice)                                 # "their" K: mean diagonal ~2 (inbred)
kva_m, kve_m = linalg.eigh(K_mice)
u_m = kve_m @ (np.sqrt(np.clip(kva_m, 0, None)) * rng.standard_normal(len(K_mice)))
y_mice = np.sqrt(0.5) * u_m + rng.normal(scale=np.sqrt(0.5), size=len(K_mice))

c = np.mean(np.diag(K_mice)) / np.mean(np.diag(K_gcta))
h_m = LMM(y_mice, K_mice, kva_m, kve_m).fit()[0]
h_theirs = h_m * c / (h_m * c + 1 - h_m)
h_fit = LMM(y_mice, K_gcta).fit()[0]                      # what the GRM-based tool would report
print("mean diag: pylmm %.2f, GCTA %.2f; c=%.3f" % (np.mean(np.diag(K_mice)), np.mean(np.diag(K_gcta)), c))
print("pylmm h=%.3f -> on the GCTA scale %.3f; fitted on the GCTA GRM %.3f" % (h_m, h_theirs, h_fit))
assert abs(h_fit - h_theirs) < 0.02
```

With an inbred GCTA-style GRM, `c ≈ 1/2`: pylmm's `h = 0.5` corresponds to `0.33` on that GRM. (This
`gcta_grm` needs complete genotypes; with missing calls use the tool's own output.)

**PVE** (proportion of phenotypic variance explained; GEMMA's `pve estimate`) does not depend on the
scale of K: `PVE = h d / (h d + 1 - h)` with `d` = the mean diagonal of the K that was fitted (GEMMA
uses `trace(K)/n`; the centred version uses `d = trace(P K P) / (n - 1)`, `P = I - 11'/n`). Report PVE
when the paper reports PVE, when you cannot reconstruct their K, or when you want one number that means
the same with any kinship scaling:

```python
def pve(h, K):
    d = np.mean(np.diag(K))
    return h * d / (h * d + 1 - h)

print("PVE: pylmm K %.3f, GCTA GRM %.3f" % (pve(h_m, K_mice), pve(h_fit, K_gcta)))
assert abs(pve(h_m, K_mice) - pve(h_fit, K_gcta)) < 0.02
```

When comparing with another tool or a paper, also check:

* **The unit of observation.** Individual-level h² and strain-mean (entry-mean) heritability differ; in
  inbred panels with replicates, see section 7.
* **Broad vs narrow sense** is a separate distinction: a kinship from SNPs gives SNP (narrow-sense-like)
  heritability; a model with a strain-identity random effect (or an ANOVA of strains) gives broad-sense
  heritability, all between-strain genetic variance.
* **Covariates.** h² is the share of the variance left after `X0`; adding covariates changes it.
* **Method.** REML vs ML, observed scale vs liability scale for binary traits.

## 4. Boundaries and the h = 0 test

The h search is a grid 0, 0.01, ..., 0.99 plus a bounded refinement (Phase 6.3 will extend it to 1):

* `h == 0`: no detectable genetic variance (common for small n or weak K).
* `h == 0.99`: "0.98 or more", capped; true values ≥ 0.98 are all reported as 0.99.

Test h² = 0 by a likelihood ratio; under the null the statistic is (asymptotically) a 50:50 mixture of
χ²₀ and χ²₁:

```python
ll0 = model.log_likelihood(0.0).ll                       # REML log-likelihood at h = 0
lrt = 2 * (model.opt_ll - ll0)
p_h0 = 0.5 * stats.chi2.sf(lrt, 1) if lrt > 0 else 1.0
print("LRT=%.1f  p(h2=0)=%.2e" % (lrt, p_h0))
```

The 50:50 mixture assumes independent observations; with correlated data it is slightly conservative
(more than half of the null mass sits at 0; Crainiceanu & Ruppert 2004). For an exact p-value near the
threshold, simulate the LRT under `h = 0`.

## 5. Standard errors and intervals

**Profile likelihood** (preferred, and the one to use near the boundaries 0 and 0.99): evaluate the REML
log-likelihood on a fine grid of `h` and keep the values within `chi2(1, 0.95) / 2 = 1.92` of the
maximum.

```python
def profile_ci(model, level=0.95):
    """Profile-likelihood interval for h (a fitted LMM), on a 1000-point grid of [0, 0.999]."""
    grid = np.linspace(0, 0.999, 1000)
    prof = np.array([model.log_likelihood(g).ll for g in grid])
    inside = grid[prof >= prof.max() - stats.chi2.ppf(level, 1) / 2]
    return inside.min(), inside.max()

lo, hi = profile_ci(model)
print("95%% profile CI: [%.3f, %.3f]" % (lo, hi))
assert lo < h < hi
```

An interval that reaches 0 or 0.999 is open on that side ("h² ≤ hi" or "h² ≥ lo").

**Curvature SE** (interior estimates only): `se = 1 / sqrt(-d²ll/dh²)` at `h`:

```python
d = 1e-3
d2 = (model.log_likelihood(h + d).ll - 2 * model.opt_ll + model.log_likelihood(h - d).ll) / d**2
print("SE(h2) = %.3f" % (1 / np.sqrt(-d2)))
```

**Grid posterior**: `model.mean_and_var()` returns the mean and variance of `h` under the normalised grid
likelihood (a flat prior on the 100-point grid); `model.plot_fit()` draws it (needs matplotlib). Its SD is
a quick summary, not a frequentist SE.

**Parametric bootstrap SE** (slow): simulate from the fitted model, refit, and take the SD of the refitted
`h`. It is an SE for interior estimates; it does not give intervals with honest coverage near 0 or the
0.99 cap (use the profile interval there). Reuse the eigendecomposition:

```python
b_hat = model.opt_beta[:, 0]
boot = []
for _ in range(30):
    u_b = kve @ (np.sqrt(np.clip(kva, 0, None)) * rng.standard_normal(n)) * np.sqrt(h * s2)
    y_b = X0 @ b_hat + u_b + rng.normal(scale=np.sqrt((1 - h) * s2), size=n)
    boot.append(LMM(y_b, K, kva, kve, X0=X0).fit()[0])
print("bootstrap SE %.3f (30 replicates; use 200+)" % np.std(boot, ddof=1))
```

## 6. From files: kinship, phenotype and covariates, then a CI

The usual file-based run: K from `pylmm kinship` (with its `.id` file), phenotypes and covariates from
`FID IID ...` tables, all aligned by ID by `PlinkReader`. Here the files are written from the simulated
data above (one individual lacks a covariate):

```python
import sys
sys.path.insert(0, "skills/convert-genotype-formats/scripts")
import plink_tools

ids = [("f%d" % (i % n_fam), "i%d" % i) for i in range(n)]
plink_tools.write_bed("h2", G, iids=[iid for _, iid in ids], fids=[fid for fid, _ in ids])
plink_tools.write_kinship("h2.kin", K, ids)              # as `pylmm kinship` writes it, + h2.kin.id
plink_tools.write_pheno("h2.pheno", ids, Y, ["trait"])
age_file = age.copy()
age_file[7] = np.nan
plink_tools.write_pheno("h2.covar", ids, age_file, ["age"])

with PlinkReader("h2", k_file="h2.kin", pheno_file="h2.pheno") as r:   # K aligned by h2.kin.id
    y_f, K_f = r.phenos[:, 0], r.K
    cov = r.read_covariates("h2.covar", columns=["age"])
keep = ~np.isnan(y_f) & ~np.isnan(cov).any(axis=1)      # complete cases
X0_f = np.column_stack([np.ones(keep.sum()), cov[keep]])  # add the intercept yourself
model_f = LMM(y_f[keep], K_f[np.ix_(keep, keep)], X0=X0_f)
h_f = model_f.fit()[0]
lo_f, hi_f = profile_ci(model_f)
print("n=%d  h2=%.3f  95%% CI [%.3f, %.3f]" % (keep.sum(), h_f, lo_f, hi_f))
assert keep.sum() == n - 1 and lo_f < h_f < hi_f
```

`pylmm gwas -v` logs the same null fit as `null model: heritability=..., sigma=...` (3 decimals), after
the same alignment and complete-case filtering.

## 7. Inbred strains with replicate animals

This is the canonical guidance for inbred panels (the other skills point here). Animals of one strain
share a genotype, so there are two ways to set up the data:

* **Individual level** (EMMA-style): one row per animal; expand the strain kinship with the incidence
  matrix `Z` (animal × strain): `K_animals = Z K_strains Z'`. K is singular (rank = number of strains);
  pylmm handles that (eigenvalues below 1e-6 are floored). For a scan, expand the genotypes the same way,
  by row indexing (`W_animals = W_strains[strain_of_animal]`: `Z @ W_strains` spreads a missing call to
  every animal), and write the IDs of the animals, not the strains. Data preparation (aliases, duplicate
  strain columns, imputing over strains): the `inbred-strain-panels` skill.
* **Strain means**: average the phenotype per strain and fit one row per strain on `K_strains`.

```python
K_strains = K[:10, :10]                    # pretend the first 10 individuals are 10 strains
strain_of_animal = np.repeat(np.arange(10), 4)                     # 4 animals per strain
Z = np.eye(10)[strain_of_animal]
K_animals = Z @ K_strains @ Z.T
print(K_animals.shape, "rank", np.linalg.matrix_rank(K_animals))
```

The two fits estimate different quantities. With `Vg` and `Ve` the genetic and residual variances of
individual animals and `r` replicates per strain, the individual-level fit estimates `h² = Vg / (Vg +
Ve)`, while a fit to strain means estimates the **strain-mean (entry-mean) heritability**

```text
H²_mean = Vg / (Vg + Ve / r) = r h² / (r h² + 1 - h²)      and back:   h² = H²_mean / (r - (r - 1) H²_mean)
```

(the same algebra as dividing K by `c = r`). With unequal numbers of replicates `r_i`, use the harmonic
mean of the `r_i` as an approximation; the strain means then have unequal residual variances `Ve / r_i`,
which pylmm's model (one residual variance) does not represent, so prefer the individual-level fit.

```python
def h2_mean(h2, r):
    """Strain-mean heritability from individual-level h2, r replicates per strain."""
    return r * h2 / (r * h2 + 1 - h2)

def h2_individual(H2_mean, r):
    return H2_mean / (r - (r - 1) * H2_mean)

r_i = np.array([3, 4, 4, 6, 8])                            # replicates per strain
r_h = len(r_i) / np.sum(1.0 / r_i)                         # harmonic mean
print("h2 = 0.3 -> H2_mean = %.3f (r = %.1f)" % (h2_mean(0.3, r_h), r_h))
assert abs(h2_individual(h2_mean(0.3, r_h), r_h) - 0.3) < 1e-12
```

Fit the same unit as the paper you compare with, or convert with the formulas above; and note whether
its Vg is SNP-based or broad-sense (section 3).

**What to expect in small panels.** The information about h² comes from the number of strains and the
replicates within them, not from the number of animals alone. In a simulation with 75 strains (an
8-founder mosaic, 5000 SNPs, true h² = 0.5, individual-level REML), the 95% profile interval had a total
width of about 0.34 with 2 animals per strain, 0.23 with 4 and 0.19 with 8: about ±0.1 to ±0.17. A fit to
the 75 strain means alone gave intervals spanning most of [0, 1]. Report the interval, not just `h`.

**GWAS in such panels.** K built from the tested SNPs absorbs causal signal when there are few distinct
genotypes, so power is low and λ_GC from one scan is noisy (see `run-gwas` and `interpret-gwas-results`);
`simulate-lmm-data` section 5 shows the disjoint-K and LOCO checks.

## 8. More

* Two kinships (e.g. two marker sets, or K plus a shared-environment matrix): `pylmm.misc.fit_two(Y, K1,
  K2, X0=X0)` returns the h² of each (`h1, h2, e, beta, sigma, ll`); it costs one eigendecomposition per
  weight grid point.
* Many phenotypes on the same individuals: compute `kva, kve` once and pass them to every `LMM`. If a
  phenotype has missing values, `LMM` subsets K and recomputes them for that phenotype.
