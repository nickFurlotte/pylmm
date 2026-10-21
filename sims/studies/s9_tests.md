# S9: Wald, Wald with the exact refit, LRT and score tests (Phase 6.5)

Study code: `sims/studies/s9_tests.py`. Full tables: `sims/results/s9_tests/summary.md` / `summary.csv`.
Reproduce: `just sim s9_tests --jobs 6 --report` (760 replicates, about 8 min at 6 jobs on an 8-core
laptop); `just sim s9_tests --quick` runs in seconds. Speed of the refit against the per-SNP search:
`just bench refit` (`sims/bench/refit.py`).

**Questions.** Are the tests of `gwas(test=..., refit=...)` calibrated under the null, from λ_GC down to
type-I error 1e-5? How much power has each? Does the exact per-SNP refit (`LMM.fit_snps`) reach the
maximum of the per-SNP grid search (`LMM.fit(X=x)`) and of the dense oracle? What does each cost?

**Design.**
- Cohorts: `unrelated300` and `unrelated100` (n = 300 / 100 unrelated individuals) and `family300`
  (100 families of 3 sibs). K is pylmm's standardised (centred) kinship of 1000 SNPs, so `K 1 = 0`, and
  with an intercept the plain ML likelihood has no interior maximum along that direction (bug 17).
- True h² ∈ {0, 0.3, 0.8}.
- `null` cells: 5000 null test SNPs × 40 replicates (200,000 null tests per cell).
- `power` cells: 1000 test SNPs, 5 of them causal, each explaining 3% of the variance (9% at n = 100),
  × 40 replicates.
- One `tail` cell (unrelated100, h² = 0.3): 50,000 null SNPs × 40 replicates (2 million null tests), for
  the 1e-5 tail.
- Tests:
  - `wald`: REML null fit, the default scan.
  - `wald_refit`: REML, `fit_snps`.
  - `lrt`: `gwas(test="lrt")`, reduced ML (below), χ²₁.
  - `lrt_plain`: study only. The same with the plain ML likelihood: the LRT of this PR's first version,
    and what GEMMA's `-lmm 2` computes after centring K.
  - `score`: `LMM.score_batch`, library only; REML null fit, t(n − q − 1).
- Each replicate also compares `fit_snps` with `fit(X=x)` on 20 SNPs (REML and ML) and with the dense
  oracle on 2.
- **Scope of the intervals.** Every cell uses one genotype cohort (one `geno_seed`), so the results are
  conditional on it. The Wilson intervals treat the pooled tests as independent; that holds in the tail
  (1e-4, 1e-5) but not at 0.05. There the replicate counts are overdispersed: about 1 for Wald and score
  on unrelated cohorts, 2.4–3.6 for every test on `family300`, and 3.0 (LRT) / 6.8 (refit) / 35 (plain
  LRT) in the tail cell. So the pooled 0.05 and 1e-3 rates are less precise than their counts suggest.

## The LRT: plain ML vs reduced ML

With a centred K and an intercept, the plain ML likelihood grows without bound as h → 1. The
intercept fits y exactly along K's null direction, whose variance goes to 0 there (bug 17). In S9
the plain ML null fit sat at h = 1 in:
- 90–100% of replicates at n = 100, **h² = 0 included**;
- 88% at n = 300, h² = 0.8;
- 60% on `family300` at h² = 0.8.

The LRT then compares two models with no residual variance, whatever the true h², and it is
anti-conservative: λ_GC up to 1.136, 2.4× the nominal rate at 1e-5. That boundary, not small n, is
the main cause. `gwas(test="lrt")` therefore uses the **reduced ML likelihood** (`LMM.lrt_batch`):
- It drops the null directions of K that the covariates span (here one, the intercept's).
- What remains is the exact likelihood of the other n − 1 contrasts, the same for the model with and
  without the SNP, so the LRT stays nested.
- Its null fit sits at h = 1 in 0–3% of replicates (10–25% at n = 100 with h² ≥ 0.3), with mean ĥ close
  to the true h².

| null fit at h = 1 / mean ĥ | n = 100, h² 0 | n = 100, h² 0.3 | n = 300, h² 0.8 | family, h² 0.8 |
|---|---|---|---|---|
| plain ML | 0.90 / 0.90 | 1.00 / 1.00 | 0.88 / 0.95 | 0.60 / 0.90 |
| reduced ML | 0.03 / 0.15 | 0.10 / 0.37 | 0.03 / 0.77 | 0.00 / 0.80 |

## Null calibration

Mean λ_GC; type-I rate ÷ α, pooled, with the 95% interval:

| | Wald | Wald + refit | LRT (reduced ML) | LRT, plain ML | score |
|---|---|---|---|---|---|
| λ_GC, n = 300 (6 cells) | 0.985–1.007 | 0.993–1.021 | 0.994–1.019 | 0.997–1.129 | 0.987–1.009 |
| λ_GC, n = 100 (3 cells) | 0.992–1.001 | 1.016–1.025 | 1.022–1.026 | 1.059–1.136 | 0.997–1.006 |
| α = 0.05, n = 100, h² = 0 | 0.98 | 1.02 | 1.04 | 1.30 | 0.91 |
| α = 1e-4, tail cell (2 M tests) | 0.92 (0.80–1.07) | 1.23 (1.09–1.39) | 1.23 (1.09–1.39) | 1.96 (1.77–2.16) | 0.25 (0.19–0.33) |
| α = 1e-5, tail cell (2 M tests) | 0.95 (0.61–1.48) | 1.40 (0.97–2.02) | 1.35 (0.93–1.96) | 2.40 (1.81–3.18) | 0.10 (0.03–0.36) |

* **Wald (default)** is calibrated throughout, into the 1e-5 tail.
* **Wald with the refit** and **the LRT (reduced ML)** behave alike.
  - At n = 300 they are within 2% (λ_GC).
  - At n = 100 they are mildly anti-conservative: λ_GC ~1.02, 1.2–1.4× the nominal rate at 1e-4 / 1e-5.
    The 1e-5 intervals include 1. The variance component is fitted together with the SNP, and neither
    t(n − q − 1) nor χ²₁ accounts for that.
  - This small-n excess remains after the fix.
* **The plain ML LRT** is anti-conservative wherever its null fit sits at h = 1: up to λ_GC 1.136 and
  2.4× at 1e-5. Where it does not (n = 300 at h² ≤ 0.3, `family300` at h² ≤ 0.3), it equals the reduced
  one.
* **Score** is calibrated at the median (λ_GC ~1.00) but conservative in the tail, more so at small n
  (n = 100: 0.25× at 1e-4, 0.1× at 1e-5).
  - Its statistic is bounded: t² ≤ n − q, because the null variance it uses still contains the SNP's
    effect. Its p-value cannot fall below about 9.7e-6 at n = 30, 7.4e-9 at n = 50 and 1.3e-15 at
    n = 94.
  - It is a monotone function of the default Wald statistic (identical rankings).
  - These are why it is `LMM.score_batch` only, not a `gwas` / `--test` option.

## Power

Share of causal SNPs with p < 1e-3 / 1e-5. These are not calibration-adjusted.

| cohort, h² | Wald | Wald + refit | LRT | LRT, plain | score |
|---|---|---|---|---|---|
| unrelated300, 0.3 | 0.455 / 0.105 | 0.470 / 0.105 | 0.460 / 0.105 | 0.470 / 0.105 | 0.430 / 0.080 |
| unrelated300, 0.8 | 0.495 / 0.110 | 0.500 / 0.120 | 0.495 / 0.120 | 0.525 / 0.165 | 0.470 / 0.085 |
| unrelated100, 0.3 | 0.425 / 0.065 | 0.430 / 0.085 | 0.435 / 0.080 | 0.445 / 0.100 | 0.320 / 0.015 |
| family300, 0.8 | 0.465 / 0.075 | 0.470 / 0.090 | 0.470 / 0.085 | 0.485 / 0.080 | 0.420 / 0.045 |

- The refit and the (reduced) LRT gain a little over the null-fit Wald test at every h² (mean −log10 p
  of causal SNPs +0.02 to +0.08). The null fit absorbs part of a causal SNP's effect into h, which makes
  P3D conservative at real loci.
- The plain LRT's extra power comes with its excess type-I error.
- The score test loses power in the tail, as its calibration predicts.

## The exact refit

- Over 28,800 per-SNP fits (720 replicates × 20 SNPs × REML and ML), `fit_snps` and `fit(X=x)` differ
  by at most 2.9e-12 in log-likelihood, in either direction.
- |Δh| ≤ 3.6e-7, on flat profiles: the grid search's Brent stops at ~1.5e-8·h.
- Against the dense oracle the shortfall is ≤ 3.3e-10 at n = 100 and ≤ 9.2e-12 at n = 300.
- No bracket fell back to the scalar search.

## Cost

Median µs per SNP of one scan, null fit included, 6 parallel jobs:

| test | µs per SNP | × the default Wald scan |
|---|---|---|
| Wald | 3.1–13 | 1 |
| score | 3.0–9.8 | about 1 |
| Wald + refit | 48–233 | 12–24× |
| LRT | 30–169 | 7.5–15× |

In a single process (`sims/bench/refit.py`) the refit is 11–61× faster than the per-SNP grid search, and costs 6.5–10× a default scan (the LRT 8–8.5×).

## Which test when

* **Genome-wide scans:** the default Wald test (fastest, calibrated).
* **Strong loci, or to match EMMA / GEMMA `-lmm 1`:** `refit=True` / `--refit`. Mildly anti-conservative
  at n ≈ 100.
* **A likelihood-ratio statistic:** `test="lrt"` / `--test lrt`. About as calibrated and as powerful as
  the refitted Wald test. It differs from GEMMA's `-lmm 2` when K has a null direction the covariates
  span (GEMMA centres every K, so always with an intercept), because GEMMA keeps the plain ML (the
  `lrt_plain` column).
* **Score:** `LMM.score_batch`, to reproduce GEMMA `-lmm 3` only.

## Agreement with GEMMA (PR evidence, not part of the study)

The comparison used GEMMA 0.98.5, the static Linux binary, in a container: n = 200, 60 SNPs, an
intercept and a covariate.
- **Positive definite uncentred K:** pylmm's `--refit` (Wald), the per-SNP REML λ, β and its SE agree
  with GEMMA's `-lmm 4` to its 6 printed digits (p ≤ 4.8e-7, λ ≤ 9e-6 relative).
- **GEMMA centres every K.** That changes nothing under REML with an intercept, but its ML fits then sit
  at its bound λ = 1e5.
- **Given the same centred K, pylmm's plain ML sits at h = 1 too.** Its plain ML LRT and score
  (`reml=False`) p-values then agreed with GEMMA's to 2e-5 relative.
- **The reduced-ML LRT departs from GEMMA's there, by design.**
- `tests/test_association_tests.py` checks:
  - the three statistics against GEMMA's formulas, on dense matrices, on a positive definite K (nothing
    dropped);
  - the reduced ML against a dense reference that finds the dropped directions by a different route.
