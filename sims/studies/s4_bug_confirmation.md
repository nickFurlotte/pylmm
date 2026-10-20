# S4: bug confirmation and bug ledger

> **Note:** apart from the bug-4 row, the impact numbers in this file come from the
> enlarged quick run. The full-run figures, which are authoritative, are in
> `docs/phase4-simulations.md` §5 and `sims/results/synthesis/bug_ledger.csv`.
> For example, bug 17's worst-case ML bias is +0.16 in the full run, not +0.14.

Question: for each suspected bug in `docs/phase1-baseline.md`, is it real, can a
user reach it, and how much does it change the results (ĥ, t, p, −log10 p)?
Each bug was scheduled either for the robustness and behaviour fixes (Phase 5,
in 1.0.0) or for the numerics work (Phase 6.3, still open; see "Known caveats"
in `AGENTS.md`).

* Code: `sims/studies/s4_bug_confirmation.py`, with five experiments:
  `remlml`, `hgrid`, `kinship`, `mlcent` and `bednull`.
* Reproducers: `tests/sims/test_s4_reproducers.py`, 46 tests that run in
  about 3 s (after Phase 5.2: bugs 1, 6, 7, 9 (ML test), 10–14 and part of
  15 are fixed and their reproducers are regression tests; the remaining
  `xfail(strict=True)` tests are the Phase 6.3 bugs 2, 3, 5 and 8). `xfail(strict=True)` tests assert the correct behaviour, so when
  a fix lands they XPASS and fail the suite, and the marker must then be
  removed. `test_bugN_current_*` tests assert today's behaviour and must be
  inverted or deleted together with the fix.
* Machine-readable ledger: `LEDGER` in the study module, written to
  `sims/results/s4_bug_confirmation/ledger.csv` by the report.
* Bug numbers 1–16 follow `docs/phase1-baseline.md`. Bug 17 is new, found by the
  simulation framework. Item 18 is the λ_GC investigation.
* Software used: numpy 2.5.4, scipy 1.18.1, Python 3.13 (uv.lock). Bugs 4 and
  16 depend on the version (see below).

## Where the numbers come from

* **Item 18:** the item-18 cells of the **full** grid. 24 cells give 80
  replicates at n = 300 and at n = 600 with h² = 0, and 40 replicates in the
  other n = 600 cells. 5000 test SNPs per replicate. The run took about 10 min
  of wall time at `--jobs 2`. Results are committed in
  `sims/results/s4_bug_confirmation/` (`summary.md`, `table_item18.csv`,
  `bednull_lambda.png`).
* **Everything else:** the `--quick` grid, which here is reduced but
  informative: 11 cells × 30 replicates at full per-cell size (n = 200–300,
  200 test SNPs). It takes about 100 s at `--jobs 2`. The quick output is
  gitignored, so the numbers are copied into this file. The full grid adds
  regimes (h² sweep, more structures) but uses the same code.
* **Full grid:** 91 cells, 1820 tasks. The `bednull` part already exists in
  `raw/full.jsonl` and will be skipped. From the measured per-task times
  (`remlml` ≈ 1.5 s, `hgrid` ≈ 0.7 s, `kinship` ≈ 0.2 s, `mlcent` ≈ 0.25 s,
  `bednull` 1.4 s at n = 300 and 3.9 s at n = 600), the whole grid is about
  2300 core-seconds:
  * about 8–10 min at `--jobs 6` from scratch;
  * about 3–4 min for the remainder (`just sim s4_bug_confirmation --jobs 6`).

## Ledger

Severity scale:

* **high:** a silent wrong result, or a crash in a realistic default setting.
* **medium:** wrong or confusing behaviour on a documented path, or a crash on a common input.
* **low:** a crash or annoyance on a rarely used flag.
* **cosmetic:** no effect on results.

| # | bug | verdict | reproducer (tests/sims/test_s4_reproducers.py) | measured impact | severity | affected | fix sketch | phase |
|---|---|---|---|---|---|---|---|---|
| 1 | Null fit always REML; `REML=` only reaches the Wald step | **CONFIRMED** | **Fixed in Phase 5.2**; now regression tests: `test_bug1_gwas_reml_flag_changes_t`, `test_bug1_cli_refit_follows_reml_choice`, `test_bug1_cli_ml_flag`, `test_bug1_remove_missing_follows_reml_choice`, `test_bug1_gwas_ml_equals_textbook_ml_wald` | **Default path:** `GWAS(REML=False)` t equals `REML=True` exactly (Δt = 0), so the CLI's documented "ML default" is really a REML Wald test. Against a textbook ML Wald, mean \|Δt\| is 0.003–0.004 and max \|Δt\| 0.013–0.026; at causal SNPs Δ−log10 p ≈ −0.01 to −0.03. Against an ML LRT, max \|Δ−log10 p\| is 0.22–0.38, which is the Wald-vs-LRT difference, not this bug. **`--refit`:** the CLI fits h by ML for each SNP while `lmm.GWAS(refit=True, REML=False)` fits REML. The two agree to \|Δt\| ≤ 0.015 when ML is bounded, but differ by up to \|Δt\| = 0.35 with a centred GRM built from m > n SNPs (bug 17); 7.8% of those per-SNP ML fits stop at the 0.99 cap. **`--removeMissingGenotypes`:** SNPs with a missing call get an ML refit and the others the REML null, inside one run. | medium (docs/behaviour mismatch; tiny numeric effect on the default path) | CLI default and `lmm.GWAS`; CLI `--refit` / `--removeMissingGenotypes` | Thread `REML` through every `fit()` call. To keep default output unchanged, make the default explicitly REML (`--REML` becomes a no-op alias) and add `--ML`, which is ML for both the null fit and the test. | 5 |
| 2 | `getMax` loops `range(1, n−2)`, so grid index 98 (h = 0.98) is never tested | **CONFIRMED** | `test_bug2_current_getmax_skips_grid_index_98`, xfail `test_bug2_getmax_finds_peak_at_098` | Family data, n = 200, h² = 0.95: 3.3% of null fits have their only local maximum at 0.98, and pylmm then returns the 0.99 endpoint. The effect is \|Δh\| ≤ 0.01. | low | every null fit and refit at high h² | `range(1, n-1)` | 6.3 |
| 3 | h grid stops at 0.99; `fitTwo` w grid also stops at 0.99 | **CONFIRMED** | `test_bug3_current_h_grid_stops_at_099`, xfail `test_bug3_h_search_reaches_one`, xfail `test_bug3_fittwo_can_select_k1_alone` | Family data, n = 200, h² = 0.95: 30% of fits return h = 0.99 and the oracle's h is above 0.99 in 27%. Max \|Δh\| = 0.015, max \|Δt\| = 0.042, max \|Δ−log10 p\| = 0.044; at causal SNPs Δ−log10 p is about 0.001 (5th percentile −0.007). The BED data (rank ≈ 96) is unaffected: oracle h ≤ 0.99 there. `fitTwo` can never choose K1 alone (in the hmdp golden, w = 0.99 has LL 23.433 against 23.501 for K1 only). | medium for h (the reported ĥ is capped and \|Δh\| ≤ 0.015; p barely moves); low for `fitTwo` | every null fit at high h²; `misc.fitTwo` | Search h on the closed interval [0, 1] with a bounded optimiser and compare both endpoints. At h = 1 use the clamped Kva or return −inf when K is singular. Use `np.linspace(0, 1, wgrids + 1)` for `fitTwo`. | 6.3 |
| 4 | Brent search unbounded above 1; nan replaced by H[i−1] | **CONFIRMED, worse than logged (on scipy ≥ 1.11)** | **Fixed in Phase 5.1**; now regression tests: `test_bug4_ll_is_nan_above_one_but_ll_brent_guards_both_ends`, `test_bug4_old_unbounded_brent_crashed_with_bracketerror`, `test_bug4_fit_with_peak_near_one`, `test_bug4_high_h2_case_matches_oracle` | When the grid maximum is at 0.96 or 0.97, scipy's bracket search steps to 0.98 + 1.618·0.02 = 1.012, where pylmm's LL is nan. **On scipy ≥ 1.11 `LMM.fit` raises `BracketError`**, and the whole GWAS (or CLI run) aborts. Crash rates for null fits: **8.75% on the real BED genotypes, n = 300, h² = 0.95** (80 replicates); 3.3% in the quick `hgrid` run on BED n = 300 (12.5% of 40 in the full run); about 2% on family data, n = 100. For per-SNP refits (`--refit`, `GWAS(refit=True)`), 1.5–2.7% of SNPs crash in the quick run and 0.25–2.8% in the full `hgrid` run (h² 0.8–0.99), so a refit scan of more than about 200 SNPs on a high-h² trait almost surely aborts. **On scipy < 1.11 (including the supported lower bound 1.9) the same code does not crash and is correct:** replaying 95 crashed null fits from S1–S3 on numpy 1.22 / scipy 1.9 (Phase 4 synthesis, §4 of `docs/phase4-simulations.md`), 90 return the bounded optimum to ≤ 7e-8 (the textbook oracle with pylmm's clamp); the 5 ML fits with an unbounded ML profile (m > n, bug 17) return h = 1.002–1.035, outside [0, 1], with LL = nan and finite t. The nan → H[i−1] fallback never fired on either scipy. The crash is therefore a regression of the Python 3 port running on newer scipy, not of the original code. | **high** (crash on realistic high-h² traits, e.g. inbred-strain expression) | every null fit and refit; CLI default | Bound the search: `minimize_scalar(method='bounded', bounds=(H[i-1], H[i+1]))`, or clip h to [0, 1] in `LL_brent`. Implemented: a bounded `minimize_scalar` search inside the grid bracket (the nan fallback became unreachable and was removed). | **5** |
| 5 | `tstat(log=True)` returns `2.0 + logsf` instead of `log 2 + logsf` | **UNREACHABLE** (the code is wrong) | `test_bug5_current_log_p_is_off_by_2_minus_log2_and_unreachable`, xfail `test_bug5_log_p` | log p is overstated by 2 − log 2 = 1.307. No caller in pylmm, the CLI or `misc` passes `log=True`. | cosmetic | none today (needed for extreme p-values later) | `np.log(2.0) + logsf` | 6.3 |
| 6 | `--kfile2` uses the undefined `LMM_withK2` and `optW` | **CONFIRMED** | **Fixed in Phase 5.2** (flag removed); `test_bug6_kfile2_removed` | Crash every time the flag is used. | low | CLI `--kfile2` | Remove the flag, or implement it with `misc.fitTwo` | 5 |
| 7 | `--eigen` cannot read what `pylmmKinship -e` writes | **CONFIRMED** | **Fixed in Phase 5.2**; `test_bug7_eigen_reads_pylmmkinship_output` (text and `.npz`), `test_bug7_eigen_of_another_kinship_rejected` | Crash every time. On macOS's case-insensitive filesystem `np.load` finds the text `.kva` and raises "pickled data"; on Linux it raises FileNotFoundError. The value given to `-e` is also ignored. | low | CLI `--eigen` | Write `np.save(<out>.Kva/.Kve)` from pylmmKinship, or have `--eigen` read both text and .npy and check the shape. | 5 |
| 8 | Two kinship normalisations; the denominator counts dropped monomorphic SNPs | **CONFIRMED** | `test_bug8_current_monomorphic_snps_shrink_k`, `test_bug8_current_kinship_cli_counts_monomorphic_snps`, xfails `..._do_not_change_k` and `..._two_kinships_agree_with_missing_data` | **Denominator:** K shrinks by a factor c = m_kept/m: 0.927 on hmdp (7.3% monomorphic) and 0.75 with 25% monomorphic; the BED subsets have none. A pure rescaling changes ĥ (+0.017 on hmdp, h² = 0.5; about +0.06 at c = 0.75) but leaves t and p **exactly** unchanged (Δt = 0, because the covariance family {σ_g²K + σ_e²I} is the same). So the bug affects the *reported heritability* only; it can push ĥ into the 0.99 cap sooner. **Missing-data normalisation:** max \|K_lib − K_cli\| is 0.09 at 2% (hmdp) and 10% missing; ĥ differs by 0.005–0.02; max \|Δt\| is 0.03 on hmdp and Δ−log10 p ≤ 0.024. With 10% synthetic missingness both kinships are about equally far from the complete-data K (max \|Δt\| ≈ 0.38 for both), so that difference comes from missingness itself, not from choosing one implementation. | low (ĥ interpretation; p essentially unchanged) | `lmm.calculateKinship` (library, and the sims framework), `pylmmKinship` | Divide by the number of SNPs kept. Use one shared normalisation function for both entry points, and choose mean-impute-then-standardise or observed-variance on purpose (document the choice). | 6.3 (changes golden ĥ) |
| 9 | Under ML the reported/used σ² is Q/(n−q), not Q/n | **CONVENTION** | **Fixed in Phase 5.2 for the ML test**; `test_bug9_ml_wald_uses_q_over_n` | The t ratio is exactly √(n/(n−p)): a factor of 1.0034 at n = 300. Δ−log10 p at causal SNPs is −0.012 to −0.023. The t-distribution reference with n−p df is the usual small-sample choice. | cosmetic | `association(REML=False)`, reported sigma | Document it. If an ML LRT is added (D5), report σ²_ML separately. | 6.3 |
| 10 | `getPhenos` skips fam individuals missing from the pheno file | **CONFIRMED** (shows as a crash in the CLI) | **Fixed in Phase 5.2**; `test_bug10_getphenos_aligns_by_id`, `test_bug10_absent_individuals_treated_as_missing`, `test_bug10_pheno_file_reordered_or_with_strangers`, `test_bug10_unalignable_pheno_file_is_a_usage_error`, `test_bug10_covariate_file_absent_individuals_dropped` | `IN.phenos` has n − k rows, shifted by position (individual I5 gets I6's value). On every CLI path tried (BED, with or without NA values, extra unknown IDs) this **does not give silently wrong output**. It crashes: `ValueError: cannot reshape array of size 38 into shape (40,1)`, or `IndexError` when NA values are present. `getCovariates` behaves the same way. Wrong results without an error need a library caller who pairs `IN.phenos` with `IN.indivs` by position. A pheno file covering a subset of the genotyped animals is a very common real input. | medium (common input → cryptic crash; silent only for library users) | CLI (pheno and cov files); `input.plink` API | Reindex by ID and fill absent individuals with NaN; do the same for covariates (then drop those individuals); log how many were missing. | 5 |
| 11 | All-missing SNP → `normalizeGenotype` returns an empty array | **CONFIRMED** | **Fixed in Phase 5.2**; `test_bug11_normalize_all_missing_keeps_length`, `test_bug11_cli_gwas_all_missing_snp`, `test_bug11_library_all_missing_snp`, `test_bug11_cli_kinship_all_missing_snp`, `test_bug11_calculatekinship_skips_all_missing_snp` | Both CLIs crash on the first all-missing SNP. That is common after QC subsetting or with sex-specific markers. | medium | CLI GWAS default path; pylmmKinship | Return an n-vector (all NaN, or zeros flagged as monomorphic) and skip the SNP with a NaN row. | 5 |
| 12 | `LMM.__init__` clamps the caller's `Kva` in place; `calculateKinship` imputes and standardises its argument in place | **CONFIRMED** | **Fixed in Phase 5.2**; `test_bug12_lmm_does_not_mutate_kva`, `test_bug12_calculatekinship_does_not_mutate_w` | Results are correct for the call itself, but the caller's arrays change. A reused Kva carries the clamp silently, and `W` is overwritten with standardised values (`example.py` already hits this). | low | library API | `Kva = np.maximum(Kva, 1e-6)` (a new array); `W = np.array(W, float)` | 5 |
| 13 | Zero-variance filters differ: `xs.var() <= 1e-6` in the missing branch, `x.var() == 0` in the complete branch | **CONFIRMED** (narrow) | **Fixed in Phase 5.2**; `test_bug13_one_filter_for_both_branches` | It is reachable only with `--removeMissingGenotypes` and genotype values whose variance is below 1e-6, e.g. EMMA dosages. Identical SNPs are then tested or skipped depending on whether one call is missing. With BED 0/0.5/1 coding, var ≤ 1e-6 needs n > 250,000. `lmm.GWAS` uses `== 0` in both branches. | cosmetic | CLI `--removeMissingGenotypes` | One `is_testable(x)` helper with a relative tolerance, used everywhere | 5 |
| 14 | No pheno file → `os.path.isfile(None)` TypeError | **CONFIRMED** | **Fixed in Phase 5.2**; `test_bug14_no_pheno_file_is_a_usage_error` | A traceback instead of the intended usage message. | low | CLI | `options.emmaPheno and os.path.isfile(...)` | 5 |
| 15 | Inconsistent return types | **CONFIRMED** | **Fixed in Phase 5.2** (GWAS return type) **and 5b** (scalars: `sigma`, `Q`, `beta_var` are floats); `test_bug15_gwas_return_type_consistent`, `test_bug15_fit_and_association_return_scalars` | `GWAS` returns lists normally but ndarrays when all Y are missing. `fit` returns β and σ as (q×1) and (1×1) arrays, and `association(returnBeta=True)` returns betaVar as a 1×1 array. Numbers are unaffected. | cosmetic | library API | Always return 1-D float arrays, and scalars via `.item()` | 5 |
| 16 | `--kfile` read unchecked (`np.fromfile(sep=' ')` + `resize`) | **CONFIRMED** (partly version-dependent) | **Fixed in Phase 5.1**; now regression tests: `test_bug16_headered_kinship_rejected`, `test_bug16_wrong_size_kinship_rejected` (and `tests/test_kinship_file.py`) | **Headered file:** on numpy ≥ 2.3 (the locked 2.5.4) `fromfile` now raises ValueError, as expected. On numpy 1.22–2.2, which `pyproject` allows (checked with 1.26.4 and 2.2.6), it only gives a DeprecationWarning that is hidden by default, and K is **all zeros**, so the run is silently OLS. **Wrong-size file:** on every numpy version, a file with fewer or more than n² numbers is zero-filled or truncated without warning, which scrambles K. | **high** (silent wrong results in the supported dependency range) | CLI `--kfile` | Implemented: a strict reader (`readKinshipFile`) that rejects `#` and header lines, requires exactly n rows of n finite numbers (warns if asymmetric), and raises a ValueError naming the file | **5** |
| 17 | ML likelihood unbounded as h → 1 with an intercept when K is row-centred and built from m > n SNPs | **NOT A BUG** in pylmm code; a property of ML + centred GRM | `test_bug17_ml_profile_unbounded_with_centred_grm_m_gt_n` | K·1 = 0, and when rank(K) = n − 1 the null space is exactly span(1). The intercept absorbs y along 1, and the variance in that direction, σ²(1−h), goes to 0, so l_ML gains −½log(1−h) → +∞. (REML's log\|X'H⁻¹X\| term cancels it.) Measured: textbook ML is unbounded in **100%** of replicates with m > n (unrelated, n = 200) and 0% with m < n or with the BED kinship (rank ≈ 93; its larger null space carries residual mass). pylmm hides the divergence with the 1e-6 clamp and the 0.99 cap, but the term still biases its **ML ĥ upward: +0.14 vs REML** (0.61 vs 0.47 at h² = 0.5), and **40% of ML null fits stop at 0.99**. With m < n the ML−REML gap is +0.004. | medium on ML paths (`fit(REML=False)`, CLI `--refit` / `--removeMissingGenotypes`, which use ML by default); none on the default path (REML) | ML users | It is not fixable as a "bug". Recommended: REML by default (see bug 1). For ML/LRT, detect `K·1 ≈ 0` and rank n−1, then warn, or fit ML in the n−1 dimensional space orthogonal to 1, or report the interior local maximum. Document it. | 6.3 |
| 18 | λ_GC ≈ 1.3–4.9 on the BED genotypes in the S1/S3 quick runs | **NOT A BUG** (single-replicate noise from extreme LD; in S3, also LD tagging of the causal SNPs) | study `bednull` (full cells) | See below. On the pure null, pooled type-I error and mean χ² match the **exact** test. | none | study design | Report pooled type-I error and mean χ² over many replicates, not single-replicate λ. In S3, compute "null" λ conditional on the causal SNPs, or only on SNPs not in LD with them. | – |

### Phase recommendation, in short

* **Phase 5 (robustness, behaviour):** 4 (crash, do first), 16, 10, 11, 1, 6,
  7, 12, 14, 13, 15. Status: 4 and 16 fixed in Phase 5.1; 1, 6, 7, 10–14
  fixed and 15 partly fixed in Phase 5.2.
  * Bug 1 is phase 5 because it is a behaviour and documentation fix.
    Defaulting explicitly to REML keeps today's default goldens unchanged.
* **Phase 6.3 (numerics):** 3, 2, 8, 17, 9, 5.
  * 3 and 2 belong together: search the closed interval [0, 1] with a bounded
    optimiser. That also covers bug 4 if Phase 5 only puts in the bounded
    Brent.
  * Bug 8 changes K's scale and so every golden ĥ, but not any t or p.

## Item 18: is pylmm's λ_GC on the BED genotypes real inflation?

**No.** On the bundled BED genotypes, pylmm is as well calibrated as the exact
test. The 1.3–4.9 values were single-replicate λ's, which are very noisy here
because the data contain only about 100 independent dimensions. The higher
values in S3 also come from true LD tagging of the simulated causal SNPs.

Design: as in S1. n random mice; K from 6000 random SNPs, built by pylmm's
`calculateKinship`; 5000 disjoint test SNPs (MAF ≥ 0.05, mean-imputed);
y = μ + u + e with u ~ N(0, h²K) drawn exactly from K. Each replicate fits:

* pylmm `GWAS(REML=True)`;
* the oracle, with textbook settings (fine grid over [0, 1], no clamp, dense
  REML);
* the **exact test**: GLS at the *true* h. Because the model is exactly
  right, its t is exactly t-distributed, so any spread it shows is pure
  sampling noise;
* OLS.

| cell (reps) | unique mice / K rank / M_eff | pylmm λ mean [5%, 95%] (min–max) | exact-test λ mean [5%, 95%] (min–max) | P(λ>1.3) pylmm / exact | pooled t1e at 0.05: pylmm / exact / OLS | mean χ²: pylmm / exact | pylmm crash |
|---|---|---|---|---|---|---|---|
| BED n=300, h²=0 (80) | 97 / 96 / 17.7 | 0.97 [0.47, 1.56] (0.43–1.86) | 1.10 [0.47, 1.96] (0.43–3.65) | 0.11 / 0.24 | 0.041 / 0.052 / 0.052 | 0.92 / 1.02 | 0 |
| BED n=300, h²=0.5 (80) | 97 / 96 / 17.7 | 1.04 [0.71, 1.48] (0.62–1.81) | 1.05 [0.67, 1.56] (0.59–2.19) | 0.13 / 0.18 | 0.050 / 0.052 / 0.45 | 1.01 / 1.02 | 0 |
| BED n=300, h²=0.95 (80) | 97 / 96 / 17.7 | 1.10 [0.72, 1.57] | 1.12 [0.71, 1.74] (0.52–2.07) | 0.15 / 0.25 | 0.055 / 0.058 / 0.56 | 1.05 / 1.07 | **8.75%** (bug 4) |
| BED n=600, h²=0 (80) | 108 / 107 / 18.6 | 0.92 [0.53, 1.38] (0.47–1.85) | 1.05 [0.53, 1.87] (0.47–2.41) | 0.09 / 0.21 | 0.038 / 0.051 / 0.051 | 0.88 / 0.98 | 0 |
| BED n=600, h²=0.5 (40) | 108 / 107 / 18.6 | 1.04 [0.69, 1.47] (0.63–1.67) | 1.04 [0.69, 1.53] (0.60–1.78) | 0.15 / 0.18 | 0.049 / 0.049 / 0.57 | 1.00 / 1.00 | 0 |
| BED n=600, h²=0.95 (40) | 108 / 107 / 18.6 | 1.05 [0.77, 1.52] (0.57–1.76) | 1.05 [0.66, 1.76] (0.59–1.83) | 0.10 / 0.13 | 0.050 / 0.049 / 0.68 | 1.01 / 1.01 | 0 |
| unrelated n=600, h²=0.5 (40), control | 600 / 599 / 535 | 1.00 [0.96, 1.05] | 1.00 [0.94, 1.04] | 0 / 0 | 0.050 / 0.050 / 0.050 | 1.00 / 1.00 | 0 |
| BED n=600, h²=0.5, **5 causal × 2%** (40), S3-like | 108 / 107 / 18.6 | **1.45** [0.84, 1.97] (0.71–**3.42**) | 1.67 (marginal) / **1.17 conditional on the causal SNPs** | 0.65 / 0.75 (marginal), 0.28 (conditional) | 0.101 / 0.124 (marginal), **0.063** (conditional) | 1.44 / 1.67 / 1.11 | 0 |

Findings:

1. **pylmm equals the textbook oracle.** Max \|Δĥ\| ≤ 1e-6 and
   max \|Δ−log10 p\| ≤ 8e-6 in every BED cell. So the eigenvalue clamp, the
   0.99 cap and the grid search have no effect at these h² values.
   Mean-imputation and the MAF filter cannot create miscalibration either,
   because x is fixed and the exact test, which uses the same imputed x, is
   calibrated.
2. **No inflation.** Under the pure null, the replicate-mean λ is 0.92–1.10
   and the pooled type-I error at 0.05 is 0.038–0.055. The mean χ² is
   0.88–1.05 (SE ≈ 0.025), the same as the exact test within noise. At 1e-3
   and 1e-4 pylmm's rates are also at or below nominal (`table_item18.csv`).
3. **The replicate λ is extremely noisy because there are only about 18
   effective tests.**
   * The 300 or 600 mice are 97 or 108 distinct genotypes (inbred-strain
     replicates); K has rank 96 or 107.
   * The participation-ratio M_eff of the 5000 test SNPs is about 18, against
     535 for unrelated data.
   * So one λ is roughly the median of about 18 correlated χ²'s. Its 5–95%
     range is about [0.5, 1.6] even for the exact test, the exact test
     reaches λ = 3.65 in one null replicate, and P(λ > 1.3) is 0.13–0.25 for
     the exact test.
   * The unrelated control (M_eff = 535) has λ in [0.95, 1.05].
4. **h² = 0 is slightly conservative, not inflated.**
   * REML puts ĥ at the 0 boundary in about 70% of replicates.
   * In the rest, ĥ > 0 down-weights the strain direction, giving
     λ_mean = 0.92–0.97 and t1e = 0.038–0.041, against 0.051 for the exact
     test. This is a known property of plug-in REML and not a bug.
5. **The S3 numbers come from LD tagging.**
   * With 5 causal SNPs at 2% each, the "non-causal" SNPs are not null on
     these data: they share strain patterns with the causal SNPs.
   * Even the exact test gives a marginal λ of 1.67 (max 4.4) and t1e of
     0.12.
   * Conditioning on the causal SNPs brings the exact test back to λ ≈ 1.17
     and t1e 0.063. The remainder is within the LD-inflated noise, since that
     test is exact by construction.
   * pylmm's 1.45 is therefore the expected association signal, not a
     calibration failure.
6. A real issue turned up on the way. At h² = 0.95, n = 300 on these real
   genotypes, **8.75% of pylmm null fits crash** with scipy's BracketError
   (bug 4). S1's `bed`, h² = 0.95 cells will hit this.

Recommendation for S1 and S3: on the BED source, quote pooled type-I error
(Wilson CI over replicates × SNPs; they are correlated, so treat it as
approximate) and the mean χ² with its SE across replicates, not single-
replicate λ_GC. In S3, measure null calibration on SNPs conditioned on the
causal SNPs, or on a separate null-only phenotype. Catch BracketError in
h² ≥ 0.95 BED cells and count it.

## Detailed numbers (quick grid, 30 replicates per cell)

Bugs 1, 9, 17 (`remlml`, n = 300, h² = 0.5, 2 causal SNPs at 4%):

| pop, m_kin | ĥ REML | ĥ ML oracle | ĥ ML pylmm | ML unbounded | \|Δt\| CLI vs ML Wald mean / max | Δ−log10p causal: total / bug 1 alone / bug 9 alone | max \|Δ−log10 p\| CLI vs ML LRT | refit ĥ CLI (ML) / lib (REML), max \|Δt\| |
|---|---|---|---|---|---|---|---|---|
| BED, 2000 | 0.528 | 0.529 | 0.529 | 0% | 0.0026 / 0.013 | −0.007 / +0.005 / −0.012 | 0.38 | 0.517 / 0.519, 0.011 |
| unrelated, 200 (< n) | 0.480 | 0.483 | 0.483 | 0% | 0.0042 / 0.026 | −0.028 / −0.005 / −0.023 | 0.22 | 0.490 / 0.486, 0.015 |
| unrelated, 2000 (> n) | 0.445 | (0.94, unbounded) | 0.498 (6.7% at 0.99) | 100% | n/a | n/a | n/a | 0.511 / 0.453, **0.35** (7.8% of SNP fits at 0.99) |

On the BED data, "null" SNPs in this cell have a pooled t1e of 0.144 for every
method alike (the LD tagging of item 18), so it is not a bug-1 effect.

Bugs 2, 3, 4 (`hgrid`, h² = 0.95, 2 causal SNPs at 3%):

| pop | crash | ĥ = 0.99 (cap) | skip98 | oracle ĥ > 0.99 | \|Δh\| > 1e-3 | max \|Δh\| | max \|Δt\| | max \|Δ−log10 p\| | refit crash per SNP |
|---|---|---|---|---|---|---|---|---|---|
| family n=200 | 0% | 30% | 3.3% | 27% | 30% | 0.015 | 0.042 | 0.044 | 2.7% |
| BED n=300 | 3.3% | 0% | 0% | 0% | 0% | 1e-6 | 7e-6 | 1e-5 | 1.6% |

The clamp alone (oracle with `eig_clamp=1e-6`) moves ĥ by ≤ 1e-6, so the
optimiser is the cause.

Bug 8 (`kinship`, h² = 0.5): see the ledger row. Full columns are in
`table_bug8.csv` of the quick report.

Bug 17 (`mlcent`, n = 200, h² = 0.5):

| pop, m_kin | rank K | ML unbounded | ĥ ML pylmm − REML | ML fits at 0.99 | l_ML(0.999) − l_ML(0.99) |
|---|---|---|---|---|---|
| unrelated, 100 | 100 | 0% | +0.004 | 0% | −113 |
| unrelated, 2000 | 199 | 100% | **+0.14** | **40%** | +1.09 |
| BED, 2000 | 93 | 0% | +0.003 | 0% | −105 |

## Reproduce

```
uv run --group sims pytest tests/sims/test_s4_reproducers.py        # ~3 s
just sim s4_bug_confirmation --quick --jobs 2                       # ~100 s
just sim-report s4_bug_confirmation --quick
S4_ONLY=bednull just sim s4_bug_confirmation --jobs 2               # item 18, ~10 min
just sim s4_bug_confirmation --jobs 6                               # full, ~8-10 min
just sim-report s4_bug_confirmation
```

The `S4_ONLY=<exp,...>` filter keeps the full grid's cell ids, so a later full
run skips cells that are already done.

Two things the framework does differently from pylmm, as declared in the code:

* Item 18 uses `pylmm_fit_instrumented`, which is pylmm's `LMM` subclassed
  only to record Brent evaluations, so that bug-4 crashes are counted instead
  of aborting.
* The CLI-only `--refit` ML path is reproduced by calling
  `L.fit(X=x, REML=False)`, `L.association(x, REML=False)` exactly as
  `pylmm/cli/gwas.py` does.
