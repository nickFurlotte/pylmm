# S9: Wald, Wald with the exact refit, LRT and score tests (Phase 6.5)

Study code: `sims/studies/s9_tests.py`. Full tables: `sims/results/s9_tests/summary.md` / `summary.csv`.
Reproduce: `just sim s9_tests --jobs 6 --report` (760 replicates, about 10 min at 6 jobs on an 8-core
laptop); `just sim s9_tests --quick` runs in seconds. Speed of the refit against the per-SNP search:
`just bench refit` (`sims/bench/refit.py`).

**Questions.** Are the four tests of `gwas(test=..., refit=...)` calibrated under the null, from
λ_GC down to type-I error 1e-5? How much power has each? Does the exact per-SNP refit (`LMM.fit_snps`)
reach the maximum of the per-SNP grid search (`LMM.fit(X=x)`) and of the dense oracle? What does each
cost?

**Design.** Cohorts: `unrelated300` and `unrelated100` (n = 300 / 100 unrelated individuals) and
`family300` (100 families of 3 sibs); K is pylmm's standardised (centred) kinship of 1000 SNPs, so ML has
no interior maximum along K's null direction (bug 17). True h² in {0, 0.3, 0.8}. `null` cells: 5000 null
test SNPs × 40 replicates (200,000 null tests per cell); `power` cells: 1000 test SNPs of which 5 are
causal, each explaining 3% of the variance (9% at n = 100), × 40 replicates; one `tail` cell
(unrelated100, h² = 0.3) with 50,000 null SNPs × 40 replicates (2 million null tests) for the 1e-5 tail.
Tests: `wald` (REML null fit, the default scan), `wald_refit` (REML, `fit_snps`), `lrt` (ML null and
per-SNP ML fits, χ²₁) and `score` (REML null fit, t(n − q − 1)). Each replicate also compares `fit_snps`
with `fit(X=x)` on 20 SNPs (REML and ML) and with the dense oracle on 2.

## Results (full run)

**The exact refit reaches the same maximum.** Over 28,800 per-SNP fits (720 replicates × 20 SNPs × REML
and ML) the largest log-likelihood difference between `fit_snps` and `fit(X=x)` is 2.9e-12 in either
direction, and |Δh| ≤ 3.6e-7 (on flat profiles: the grid search's Brent stops at ~1.5e-8·h). Against the
dense oracle the shortfall is ≤ 3.3e-10 (n = 100) and ≤ 9.2e-12 (n = 300). No bracket fell back to the
scalar search. Up to 25% of the REML fits (unrelated100, h² = 0.8) and nearly all ML fits at n = 100
(and 55-89% at h² = 0.8 with n = 300) are at h = 1, where both searches agree.

**Null calibration** (mean λ_GC; type-I error rate ÷ α, pooled, 95% interval):

| | Wald | Wald + refit | LRT | score |
|---|---|---|---|---|
| λ_GC, n = 300 (6 cells) | 0.985–1.007 | 0.993–1.021 | 0.997–1.129 | 0.987–1.009 |
| λ_GC, n = 100 (3 cells) | 0.992–1.001 | 1.016–1.025 | 1.059–1.136 | 0.997–1.006 |
| α = 0.05, n = 100, h² = 0 | 0.98 | 1.02 | 1.30 | 0.91 |
| α = 1e-4, tail cell (2 M tests) | 0.92 (0.80–1.07) | 1.23 (1.09–1.39) | 1.96 (1.77–2.16) | 0.25 (0.19–0.33) |
| α = 1e-5, tail cell (2 M tests) | 0.95 (0.61–1.48) | 1.40 (0.97–2.02) | 2.40 (1.81–3.18) | 0.10 (0.03–0.36) |

* **Wald (default)** is calibrated throughout, into the 1e-5 tail.
* **Wald with the refit** is slightly anti-conservative at n = 100 (λ_GC ~1.02, 1.2–1.4× at 1e-4 / 1e-5):
  the per-SNP h is fitted together with the SNP, and the t(n − q − 1) reference does not account for it.
  At n = 300 it is within 1–2%.
* **LRT** is anti-conservative at small n (n = 100: λ_GC 1.06–1.14, 2–2.4× in the tail) — χ²₁ with no
  degrees-of-freedom correction — and where ML fits sit at h = 1 (unrelated300 and family300 at h² = 0.8:
  λ_GC 1.10–1.13, type-I 0.063 at 0.05; there 61–89% of the ML fits are at the boundary, the bug-17
  situation of a centred K with an intercept). At n = 300 and h² ≤ 0.3 it is within 2.5%.
* **Score** is calibrated at the median (λ_GC ~1.00) but conservative in the tail, increasingly so at
  small n (n = 100: 0.25× at 1e-4, 0.1× at 1e-5): its statistic is bounded (t² ≤ n − q), since the null
  variance it uses still contains the SNP's effect.

**Power** (share of causal SNPs with p < 1e-3 / 1e-5; not calibration-adjusted):

| cohort, h² | Wald | Wald + refit | LRT | score |
|---|---|---|---|---|
| unrelated300, 0.3 | 0.455 / 0.105 | 0.470 / 0.105 | 0.470 / 0.105 | 0.430 / 0.080 |
| unrelated300, 0.8 | 0.495 / 0.110 | 0.500 / 0.120 | 0.525 / 0.165 | 0.470 / 0.085 |
| unrelated100, 0.3 | 0.425 / 0.065 | 0.430 / 0.085 | 0.445 / 0.100 | 0.320 / 0.015 |
| family300, 0.8 | 0.465 / 0.075 | 0.470 / 0.090 | 0.485 / 0.080 | 0.420 / 0.045 |

The refit gains a little over the null-fit Wald test at every h² (mean −log10 p of causal SNPs +0.02 to
+0.08): the null fit absorbs part of a causal SNP's effect into h (P3D is conservative at real loci). The
LRT's extra power at n = 100 and h² = 0.8 comes with its excess type-I error above; the score test loses
power in the tail, as its calibration predicts.

**Cost** (median µs per SNP of one scan, null fit included, 6 parallel jobs): Wald 4.5–15, score 3.3–12,
Wald + refit 60–440, LRT 52–525. The refit and the LRT cost 13–38× the default scan; per SNP that is
in a single process 9.5–12× (`sims/bench/refit.py`), and 8–46× less than the per-SNP grid search.

## Which test when

* Genome-wide scans: the default Wald test (fastest, calibrated).
* Strong loci, or to match EMMA / GEMMA `-lmm 1`: `refit=True` / `--refit`, at n ≳ 200 (slightly
  anti-conservative below).
* LRT (GEMMA `-lmm 2`): for comparison with tools that report it; avoid it at small n or when the ML fit
  is at h = 1 (centred K, high h²).
* Score (GEMMA `-lmm 3`): the default scan's ranking with conservative tail p-values; no advantage in
  cost over the default here, since the default Wald scan is already batched.

## Agreement with GEMMA (not part of the study; PR evidence)

GEMMA 0.98.5 (static Linux binary in a container), n = 200, 60 SNPs, an intercept and a covariate: with
a positive definite uncentred K, pylmm's `--refit` (Wald), the per-SNP REML λ, β and its SE agree with
GEMMA's `-lmm 4` to its 6 printed digits (p ≤ 4.8e-7, λ ≤ 9e-6 relative). GEMMA centres every K; that
changes nothing under REML with an intercept, but its ML fits then sit at its bound λ = 1e5 (h → 1). Given
the same centred K, pylmm's ML fits sit at h = 1 too, and the LRT and score (`--ml`) p-values agree with
GEMMA's to 2e-5 relative. `tests/test_association_tests.py` checks the three tests against GEMMA's
formulas implemented on the dense matrices.
