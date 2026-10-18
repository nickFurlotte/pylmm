# pylmm: literature review and prioritized improvement directions (2014–2026)

*Prepared 2026-10-18 for the pylmm revamp (TODO.md Phase 6). Read-only review; no code was changed.*

## 0. TL;DR

pylmm's default mode (`refit=False`) is already the **EMMAX / P3D** approximation on top of a
**FaST-LMM-style spectral rotation**. That core idea is still the right one for a small library.
Most of what changed since 2014 either (a) makes the same model faster through vectorization and
better I/O, (b) fixes known modelling problems (proximal contamination, LOCO, GRM definition), or
(c) moves to biobank-scale approximations (BOLT-LMM, REGENIE, fastGWA, SAIGE). Those need a different
architecture and are mostly out of scope, though a few of their ideas carry over.

Recommended order (details in §4):

1. **Batched rotation + vectorized closed-form per-SNP Wald tests.** Exact under P3D, 10–100x faster. Effort S.
2. **Vectorized BED decoding** (numpy lookup table; `bed-reader` as an optional backend). Effort S.
3. **Optimizer and numerics hardening.** Log-δ grid, best-of-optima, `slogdet`/`solve`, a correct log-p path. Effort S.
4. **LOCO kinship**, plus a GCTA-style standardized GRM option and proper missing-data normalization. Effort M.
5. **Exact per-SNP refit (GEMMA-style).** Analytic derivatives with Newton/Brent, plus LRT and score tests. Effort M.

There is also a short list of **probable bugs** found while reading the code (§1.3). Per the TODO
ground rules, these are flagged for Nick and not fixed here.

---

## 1. What pylmm does today

### 1.1 Algorithm (pylmm/lmm.py)

| Step | Code | Cost |
|---|---|---|
| Eigendecomposition of K | `LMM.__init__` → `linalg.eigh(K)` | O(n³), once |
| Rotate y, covariates | `LMM.transform` | O(n²q) |
| Null fit of h | `LMM.fit`: 100-point grid on h∈[0,0.99] → `getMax` → `optimize.brent` on each local max | ~100–130 evaluations of `LL`, each O(nq²) |
| Per-SNP rotation | `association` → `matrixMult(Kve.T, X)` for **one SNP at a time** (BLAS-2 mat-vec) | **O(n²) per SNP → O(mn²) total, the dominant cost** |
| Per-SNP Wald test | `LL` → `getMLSoln` (builds X'SX, `linalg.inv`, β, Q) → `tstat` (t with N−q df) | O(nq²) per SNP plus Python overhead |
| `refit=True` (exact per-SNP) | `fit(X=x)` per SNP: full grid + Brent | ~100x the per-SNP cost |
| SNP with missing genotypes, **only under `--removeMissingGenotypes`** (the CLI default mean-imputes, so no NaNs reach this branch) | `GWAS()` / `pylmmGWAS.py` (~L244-267) builds a **new `LMM` on the subset**, which means a **fresh O(n³) `eigh` per SNP** | catastrophic when missingness is common and this option is set |

Other notes:

- Eigenvalues below 1e-6 are clamped to 1e-6 (`LMM.__init__`). This acts as a silent regularizer and
  matters for low-rank K.
- The REML term uses `np.log(linalg.det(...))`, which can overflow or underflow when there are many
  covariates or the scaling is large.
- `matrixMult` probes `linalg.fblas`. That module is gone in modern SciPy, so the code always falls
  back to `np.dot`.

### 1.2 Kinship and I/O

- `calculateKinship` (lmm.py) and `scripts/pylmmKinship.py` both build K = WWᵀ/m from SNPs that were
  standardized with the sample mean and variance and mean-imputed (`input.plink.normalizeGenotype`).
  This is the "realized relationship matrix" (RRM). Only `calculateKinship` has the optional trace
  normalization.
- BED decoding (`input.plink.next` / `formatBinaryGenotypes`) runs `bin(ord(byte))`, string-slices
  each byte and looks the pieces up in a dict. That is pure-Python per byte, likely 100–1000x slower
  than a vectorized decoder.
- `next()` returns only the rsid. The **chromosome and position from .bim are thrown away**, so LOCO
  is impossible today without changing the I/O layer.
- Two-component model: `misc.fitTwo` runs a grid over w with a **full eigh per grid point** (100
  eighs).

### 1.3 Probable bugs and inconsistencies (flag to Nick; do not fix silently)

1. **The REML flag is inconsistent in `scripts/pylmmGWAS.py`.**
   - The null model is fit with `L.fit()`. Its default is `REML=True`, but the CLI default is
     `--REML` off (ML). Association then uses `REML=options.REML`.
   - So on every run, including the default path, the null fit is REML while `association` uses ML
     (unless `--REML` is given): a REML/ML mismatch.
   - Only under `--removeMissingGenotypes` (the default mean-imputes, so this branch, ~L244-267, is
     never reached), SNPs that have missing genotypes are refit with `Ls.fit(REML=options.REML)`,
     i.e. ML by default.
   - Result: in that mode, SNPs with and without missingness are also tested under different
     variance-component estimators within one run.
2. **The log p-value branch in `LMM.tstat(log=True)` is wrong.** It computes `2.0 + logsf(...)`; it
   should be `log(2) + logsf(...)`. No caller uses it today, but it is the path needed for extreme
   p-values.
3. **`getMax` never checks the last interior grid point.** It loops `range(1, n-2)`. With multiple
   optima it returns the *first* one, not the one with the highest LL. The grid also excludes h=1.
4. **`--kfile2` is broken.** `pylmmGWAS.py` references `LMM_withK2`, which is not defined or imported
   anywhere, so using the option raises a NameError.
5. **The kinship denominator is off in two places.** `pylmmKinship.py` divides by `IN.numSNPs`, which
   includes the monomorphic SNPs it skipped. `lmm.calculateKinship` has the same bug: it drops
   monomorphic SNPs via `W[:,keep]` (~L51) but divides by the earlier `m` (~L63). K is scaled by a
   constant, which changes h² interpretation but not p-values.
6. **Python 2 idioms that break on modern Python/numpy/scipy:**
   - `True - mask` (numpy now raises for boolean subtract)
   - `X0 == None` on arrays
   - `linalg.fblas`
   - `D.has_key` (a Python 3 break, not a numpy/scipy one)
   - `'\x01'` string compare on bytes

   Phase 3 should already cover these.

---

## 2. Literature and ecosystem, 2014–2026, mapped to pylmm

### 2.1 Exact and fast single-kinship solvers

- **EMMA** (Kang et al. 2008, *Genetics* 178:1709) is the spectral trick for optimizing δ=σe²/σg² in
  1-D. pylmm's `LMM` is this algorithm, written in terms of h.
- **FaST-LMM** (Lippert et al. 2011, *Nat Methods* 8:833):
  - Rotate once by the eigenvectors of K, then optimize log δ on a grid followed by Brent.
  - When K = WWᵀ has rank k < n, use the thin SVD of W, which costs O(nk²), and handle the null
    space analytically.
  - The Python implementation is maintained as `fastlmm` (0.6.x; https://fastlmm.github.io/,
    https://pypi.org/project/fastlmm/).
  - The cloud/cluster variant, `single_snp_scale`, comes from Kadie & Heckerman, "Ludicrous Speed
    LMMs", bioRxiv 10.1101/154682 (2017/2019).
- **GEMMA** (Zhou & Stephens 2012, *Nat Genet* 44:821; https://github.com/genetics-statistics/GEMMA):
  - Gives *exact* per-SNP ML or REML. After one rotation, the likelihood and its first and second
    derivatives in λ come from O(n) recursions, with a grid scan over log λ first and then Newton-Raphson within the intervals where the derivative changes sign.
  - Reports Wald, LRT and score tests.
  - This is the principled replacement for pylmm's `refit=True` grid search.
- **EMMAX** (Kang et al. 2010, *Nat Genet* 42:348) and **P3D** (Zhang et al. 2010, *Nat Genet*
  42:355):
  - Estimate the variance components once under the null and reuse them for every SNP.
  - This is pylmm's default, and it is accurate when single SNPs explain little variance.
  - Exact refit matters mainly for large-effect loci (e.g. in mouse HMDP data, which ships in
    `data/`).
- **GRAMMAR-Gamma** (Svishcheva et al. 2012, *Nat Genet* 44:1166):
  - Uses residuals from the null LMM with a simple per-SNP score statistic, then corrects with one
    calibration factor γ estimated on a random subset of SNPs.
  - BOLT-LMM-inf and fastGWA use the same "approximate statistic + calibration" idea.

**Per-SNP rotation tricks.** The essential fact is that with h fixed, every per-SNP quantity is a
weighted inner product in the rotated space. So:

1. Rotate a block of B SNPs at once with one GEMM (`Kve.T @ X_block`, BLAS-3).
2. With w = 1/(h·λ + 1−h), the q×q covariate block A = X0ᵗᵀWX0ᵗ is fixed.
3. Per SNP you only need b = X0ᵗᵀWxᵗ (q-vector), c = xᵗᵀWxᵗ and d = xᵗᵀWyᵗ.
4. A Schur-complement / Frisch–Waugh step gives, in closed form and vectorized over SNPs:
   - β̂ = (d − bᵀA⁻¹X0ᵗᵀWyᵗ)/(c − bᵀA⁻¹b)
   - Var(β̂) = σ²/(c − bᵀA⁻¹b)
   - Q_snp = Q₀ − (numerator)²/(c − bᵀA⁻¹b)

FaST-LMM, GEMMA, and every biobank tool use this. pylmm does it one SNP at a time through
`np.hstack`-free buffers (`X0t_stack`) but still in Python and BLAS-2.

### 2.2 LOCO and proximal contamination

- **Proximal contamination** (Listgarten et al. 2012, *Nat Methods* 9:525):
  - If the test SNP, or SNPs in LD with it, are in K, the SNP is "double fit". This deflates its
    statistic and costs power.
  - The fix is to exclude the test SNP and nearby SNPs from K. For low-rank K, FaST-LMM does this
    with a rank-k Woodbury downdate.
- **Yang et al. 2014** (*Nat Genet* 46:100, "Advantages and pitfalls in the application of
  mixed-model association methods"):
  - Analyzes the power loss and popularized **LOCO** (leave-one-chromosome-out): test chromosome c
    using K built from all other chromosomes.
  - LOCO is now the default in GCTA `--mlma-loco`, BOLT-LMM, REGENIE, SAIGE, and fastlmm
    `single_snp`.
- **Cost for pylmm:** 22 eighs (one per autosome) instead of 1.
  - K_{−c} = (K_counts_total − K_counts_c)/(m − m_c) is cheap if per-chromosome *unnormalized*
    WᵀW blocks are kept.
  - For n ≤ ~10k, 22 eighs of n×n take tens of minutes, which is acceptable.

### 2.3 Large-scale methods: what is portable?

| Method | Key idea | Portable to pylmm? |
|---|---|---|
| **BOLT-LMM**: Loh et al. 2015, *Nat Genet* 47:284; Loh et al. 2018, *Nat Genet* 50:906 | No eigh. Conjugate-gradient solves with V = σg²WWᵀ/m + σe²I, which are matrix-free and O(nm) per iteration. Monte-Carlo REML. Bayesian mixture prior. LOCO. Retrospective statistic calibrated with LD-score-style factor | **Partially.** A matrix-free CG solve (`scipy.sparse.linalg.cg` with a `LinearOperator` over W) plus a GRAMMAR-γ / BOLT-inf calibrated score test would let pylmm handle n ≫ 20k, but it is a separate code path (L). The mixture prior and MC-REML are out of scope |
| **REGENIE**: Mbatchou et al. 2021, *Nat Genet* 53:1097; https://rgcgithub.github.io/regenie/ (v4.1.x, 2025) | Step 1: block-wise ridge regression stacked into LOCO polygenic predictions, with no kinship at all. Step 2: per-SNP linear or Firth regression with the LOCO prediction as an offset | **Step 2 is trivial** once pylmm has vectorized per-SNP regression (§2.1). Step 1 is M–L and changes the model. Not recommended for this library except perhaps as an experimental mode later. (sgkit, https://github.com/sgkit-dev/sgkit, has a Python REGENIE implementation to reference) |
| **fastGWA**: Jiang et al. 2019, *Nat Genet* 51:1749; GCTA | Sparse GRM (entries below a threshold, e.g. 0.05, set to 0), so only family structure is modeled and PCs handle population structure. P3D-style one-time variance estimate. GRAMMAR-γ-like test | **Low priority.** Needs sparse Cholesky (CHOLMOD via scikit-sparse) to pay off. Mostly relevant for biobank family data, not pylmm's typical inbred or model-organism use |
| **SAIGE**: Zhou et al. 2018, *Nat Genet* 50:1335; fastGWA-GLMM (Jiang et al. 2021, *Nat Genet* 53:1616) | Logistic GLMM for binary traits plus saddlepoint approximation (SPA) of the score test for case-control imbalance | **Out of scope** (pylmm is Gaussian-only). If binary traits ever matter, SPA on a score statistic is the piece worth porting |

### 2.4 Kinship estimation variants

- **What pylmm uses now:** an RRM with *sample* allele variance and mean imputation. Normalization is
  by number of SNPs, with an optional trace-based rescale in `calculateKinship`.
- **GCTA standardized GRM** (Yang et al. 2011, *AJHG* 88:76):
  - Uses the 2p(1−p) variance computed from allele frequency rather than the sample variance.
  - Divides each entry (j,k) by the number of SNPs non-missing in *both* individuals. This
    count-matrix normalization costs one more GEMM: (1−M)(1−M)ᵀ, where M is the missingness mask.
  - Output is directly comparable with `gcta --make-grm` and `plink2 --make-rel`, which makes it a
    natural external validation target.
- **LDAK** (Speed et al. 2012, *AJHG* 91:1011; Speed et al. 2017, *Nat Genet* 49:986): LD-weighted
  SNPs and an α-parameterized MAF scaling. This mainly affects h² estimation and is lower priority
  for association.
- **Missing data:** mean imputation shrinks relatedness toward 0 for high-missing individuals.
  Per-pair counts fix most of the bias. Either way, report the per-individual missingness rate.
- **Selection** (FaST-LMM-Select; Listgarten et al. 2012, *Nat Methods*): build K from a selected SNP subset.
  It can raise power but is fragile and adds tuning. Not recommended.

### 2.5 Multiple variance components

- Today, `misc.fitTwo` does 100 eighs, and `--kfile2` is broken (§1.3).
- **Better: AI-REML** (Gilmour, Thompson & Cullis 1995, *Biometrics* 51:1440; used in GCTA).
  - Newton-type iterations using the average-information matrix.
  - Each iteration costs a Cholesky factorization of V (O(n³)), and convergence typically takes
    5–15 iterations. That is ~10x fewer O(n³) operations than the grid.
- **Special case:** if K2 is low-rank (e.g. a few PCs or a local kinship), the main eigenbasis plus
  Woodbury gives a fast exact solution.
- **Reference implementations:** limix / glimix-core (https://github.com/limix). Their maintenance
  status was not confirmed as of 2026; use them as math references, not dependencies.
- **Multivariate LMM** (GEMMA mvLMM; Zhou & Stephens 2014, *Nat Methods* 11:407) is out of scope.

### 2.6 Numerical and linear-algebra ecosystem

- **`scipy.linalg.eigh`:**
  - Takes `driver=` ('ev', 'evd', 'evr', 'evx') and `subset_by_index=`
    (https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigh.html).
  - For a full decomposition, the divide-and-conquer driver `evd` is often fastest; `evr` (the
    default) uses less memory.
  - `subset_by_index` returns only the top-k eigenpairs, which can be useful for a low-rank
    approximate mode.
  - Also set `check_finite=False` and `overwrite_a=True` to skip copies.
  - Benchmark on the target BLAS. NumPy ≥2.0 wheels for macOS 14+ arm64 use Apple Accelerate, while Linux
    wheels use OpenBLAS.
- **Thin SVD of W instead of eigh(WWᵀ):** when m < n (or for a LOCO block), this costs O(nm²) and is
  more accurate than forming K.
- **Randomized SVD** (Halko, Martinsson & Tropp 2011, *SIAM Rev* 53:217; `sklearn.utils.extmath.randomized_svd`)
  gives an approximate top-k basis. It is only appropriate for an explicitly *approximate* mode
  because truncation turns the LMM into PC regression plus a residual, which can hurt calibration.
- **Stable formulas:**
  - Replace `inv` and `det` with `cho_factor`/`cho_solve` and `slogdet`.
  - Optimize over log δ ∈ [−5, 5] like FaST-LMM, which covers h near 0 and 1 symmetrically and
    avoids the h≥1 singularity.
- **Optional backends (numba, JAX, torch):** once the per-SNP work is batched GEMM plus elementwise
  numpy, plain numpy already runs at BLAS speed, so numba buys little. A GPU (JAX/CuPy/torch) helps
  only for eigh and GEMM when n ≳ 20k, and memory (n² × 8 bytes) becomes the limit first.
  Recommendation: none for now; keep the code array-API-friendly so a backend can be swapped in
  later.

### 2.7 Modern genotype I/O

| Library | Notes | Fit for pylmm |
|---|---|---|
| **bed-reader** (https://github.com/fastlmm/bed-reader, PyPI 1.1.0, June 2026; Apache-2.0 (per its LICENSE.md and Cargo.toml; not dual-licensed); Rust core, from the FaST-LMM team) | Multithreaded, reads arbitrary SNP or sample subsets, returns float32/64 with NaN for missing, supports cloud URLs | **Best optional backend.** License-compatible with MIT |
| **pandas-plink** (https://github.com/limix/pandas-plink, 2.3.2) | Dask-lazy BED plus bim/fam DataFrames | Fine, but adds dask. bed-reader is lighter |
| **Pgenlib** (https://pypi.org/project/Pgenlib/, 0.95.1, Oct 2026; LGPL-3.0; from plink-ng) | Reads and writes PLINK 2 .pgen, including dosages | Optional extra only (LGPL as an optional dependency is fine; do not vendor it) |
| **Pure numpy** | `np.fromfile` the bytes, then a 256×4 lookup table maps each byte to 4 genotypes: `lut[bytes].reshape(-1)[:n]`, vectorized over SNP blocks | **Default.** Zero new dependencies and ~100x+ faster than the current decoder |

### 2.8 P-value accuracy and calibration

- **Test statistic:**
  - pylmm's Wald t with N−q degrees of freedom is the same as GEMMA's Wald F(1, n−c−1).
  - `stats.t.sf` is accurate down to ~1e-300.
  - For extreme signals (common in inbred-mouse eQTL data), return `log10 p` from `stats.t.logsf`
    after fixing the bug in §1.3.2.
- **Calibration:**
  - Report λ_GC, the median χ² divided by 0.4549 (Devlin & Roeder 1999, *Biometrics* 55:997).
  - Offer LRT and score tests alongside Wald, as GEMMA does. P3D is generally conservative for
    large-effect SNPs, because the SNP's effect is absorbed into the null σg².
  - LOCO (§2.2), or excluding nearby SNPs from K, is the principled fix for deflation from proximal
    contamination. Exact refit (§2.1) does not fix it; it addresses the P3D conservatism above.
  - GRAMMAR-γ-style calibration is only needed if an approximate fast statistic is added.

---

## 3. Candidate directions (each tied to code)

### D1. Batched rotation and vectorized per-SNP association (P3D, exact)

- **What:** New `LMM.association_batch(X_block, h=None)`, which:
  1. computes `Xt = Kve.T @ X_block` (one GEMM);
  2. applies the closed-form Schur-complement formulas from §2.1;
  3. returns arrays of β, SE, t and p.
- **Why for pylmm:** replaces the per-SNP loop in `GWAS()` and `pylmmGWAS.py`, along with the
  `association` → `LL` → `getMLSoln` → `linalg.inv` chain. BLAS-2 becomes BLAS-3 and the Python
  overhead disappears. Expect 10–100x on the example data. `pylmmKinship.py` already does block
  processing via `computeSize` (`-n`).
- **Effort:** S.
- **Validate:** t and p from `association(x)` must match to ~1e-10 relative on every SNP of the
  golden data (non-refit, non-missing path). Also check edge cases: monomorphic SNPs, q>1
  covariates, ML vs REML h.
- **Risks:** minimal. Numerical differences at near-collinear SNPs; guard c − bᵀA⁻¹b ≤ ε by
  returning NaN, matching the current var==0 handling.

### D2. Vectorized BED decoding plus optional bed-reader / Pgenlib backends

- **What:** Rewrite `input.plink.getSNPIterator_bed`/`next` to read blocks of SNPs with a lookup
  table. Add `read_block(start, count)` returning an (n × B) float array with NaN for missing, and
  also return chromosome and position from .bim (needed by D4). Optional `backend="bed-reader"`.
- **Why:** the `bin()`/string decoder is likely the wall-clock bottleneck after D1. Exposing
  chromosome is the prerequisite for LOCO.
- **Effort:** S (numpy); S for the bed-reader adapter; M for pgen.
- **Validate:** bit-identical arrays against the current `formatBinaryGenotypes` (before and after
  `normalizeGenotype`) on `data/snps.132k.clean.noX.bed`, and against `bed_reader.open_bed(...).read()`.
  Note that bed-reader counts A1 alleles by default, so check allele orientation explicitly.
- **Risks:** allele-orientation and coding conventions (pylmm uses 0/0.5/1); the trailing-byte
  padding when n%4≠0.

### D3. Optimizer and numerics hardening (null fit and per-SNP fit)

- **What:**
  - Grid on log δ (FaST-LMM style), or on h∈[0,1) including the last interior point.
  - Evaluate all local maxima and keep the highest LL.
  - Use bounded Brent (`optimize.minimize_scalar(method="bounded")`).
  - `slogdet` and Cholesky instead of `det`/`inv` in `LL` and `getMLSoln`.
  - Correct `tstat(log=True)`.
  - Remove the `linalg.fblas` probe.
- **Why:** fixes the `getMax` issues in §1.3, avoids h=1 singularity and overflow, and makes the
  REML term stable for many covariates. Also the right moment to settle the REML/ML inconsistency
  in `pylmmGWAS.py` (§1.3.1), which is a decision for Nick.
- **Effort:** S.
- **Validate:** on golden data, ĥ and LL agree with current outputs to ~1e-6 except where the old
  code hit a documented edge case. Add simulation cases with h²∈{0, 0.02, 0.98} and check recovery.
  Cross-check ĥ against GEMMA (`-lmm`) or `fastlmm` on the example data.
- **Risks:** intentionally changes outputs at edge cases, so golden tests need an explicit,
  reviewed tolerance update.

### D4. LOCO kinship plus a GRM-variant option

- **What:**
  - `pylmmKinship.py --loco` writes per-chromosome *unnormalized* WᵀW blocks plus SNP counts.
  - The GWAS uses K_{−c} for chromosome c: 22 eighs, or thin SVDs when m_{−c} < n.
  - Add `--grm {rrm,gcta}`, where gcta means 2p(1−p) scaling and per-pair non-missing counts.
  - Fix the denominator in §1.3.5 (both `pylmmKinship.py` and `lmm.calculateKinship`).
- **Why:** removes proximal contamination (Listgarten 2012; Yang 2014) and is the community-standard
  default. The GCTA option makes K comparable with external tools.
- **Effort:** M (depends on D2's chromosome plumbing).
- **Validate:**
  - Simulation: polygenic background plus a few causal SNPs, on real genotypes from `data/`.
    Expect LOCO to give higher power at the causal SNPs, with λ_GC under the null still ≈1.
  - External: compare against `gcta --mlma-loco` or `fastlmm.association.single_snp`; p-values
    should correlate >0.99 in −log10.
  - GRM: `--grm gcta` vs `gcta --make-grm` / `plink2 --make-rel` entries to ~1e-6.
- **Risks:**
  - 22x eigh cost. Memory: holding one n×n at a time is fine, holding all 22 is not.
  - In inbred mouse panels (HMDP), LOCO can change calibration because relatedness is strongly
    genome-wide. Treat it as opt-in at first.

### D5. Exact per-SNP refit (GEMMA-style) with LRT and score tests

- **What:** For `refit=True`:
  - Analytic first and second derivatives of the (RE)ML log-likelihood in δ in the rotated space.
  - Newton from the null ĥ, with a Brent fallback, giving ~5–10 evaluations per SNP instead of ~120.
  - Vectorize across a SNP block, since each SNP's 1-D optimization is independent.
  - Add `test={"wald","lrt","score"}`.
- **Why:** the current `fit(X=x)` per SNP is ~100x the P3D cost. This makes exact mode practical for
  the large-effect loci typical in model-organism data.
- **Effort:** M.
- **Validate:**
  - Current `--refit` ĥ and p on golden data, with p agreeing to ~1e-4 relative.
  - GEMMA `-lmm 4` outputs on the same data.
  - Simulation: under the null, the score test and LRT are calibrated; when a SNP has large effect,
    exact refit differs from P3D in the expected direction.
- **Risks:**
  - Multiple optima or boundary maxima. Keep a grid fallback when Newton fails.
  - The REML LRT is not valid for fixed effects; use the ML LRT, as GEMMA does.

### D6. Missing-genotype path without per-SNP eigh

- **What:** In `GWAS()`/`pylmmGWAS.py`, when a SNP has missing genotypes:
  - (a) default to mean imputation, which is already the default because `normalizeGenotype=True`;
    or
  - (b) cache the eigendecomposition per missingness pattern (hash of the mask); or
  - (c) for small missingness, use an exact conditional approach with a low-rank correction.
- **Why:** only with `--removeMissingGenotypes` (not the default) does each such SNP currently cost a fresh O(n³) eigh.
- **Effort:** S for (b); M for (c).
- **Validate:** identical outputs to the current path for (b). Report timing.
- **Risks:** in sequencing-like data, every SNP can have a unique pattern, so the cache never hits.
  Document mean imputation as the recommended mode.

### D7. Multiple variance components via AI-REML

- **What:** Replace the `misc.fitTwo` grid with AI-REML for K1 + K2 + I. Fix or remove `--kfile2`
  (LMM_withK2).
- **Effort:** M.
- **Validate:**
  - Recovers `fitTwo`'s (h1, h2) on example data to grid resolution.
  - Matches GCTA `--reml --mgrm`.
  - Simulation with known components.
- **Risks:** convergence at boundaries (a component going to 0) needs constrained steps.
  Lower value for association than D1–D5.

### D8. Approximate scalable mode (BOLT-inf / GRAMMAR-γ style). Defer.

- **What:**
  - Matrix-free CG to compute V⁻¹y with K = WWᵀ/m (never forming K).
  - A residual-based score statistic for all SNPs.
  - A calibration factor estimated from exact tests on ~100–1000 random SNPs.
- **Effort:** L.
- **Validate:** χ² against exact pylmm on subsets, with λ ratio ≈1.
- **Risks:** a second code path to maintain, and it is outside the "lightweight" mission. Revisit
  only if n > ~20k becomes a goal.

---

## 4. Recommended top 5 (ordered)

| # | Direction | Effort | Why first |
|---|---|---|---|
| 1 | **D1 Vectorized per-SNP association** | S | Biggest speedup. Exactly testable against golden outputs. Foundation for D4/D5 |
| 2 | **D2 Vectorized BED reader** (+ chromosome/position, optional bed-reader) | S | The next bottleneck after D1. Bit-identical test. Unblocks LOCO |
| 3 | **D3 Optimizer and numerics hardening** (incl. log-p fix, REML consistency decision) | S | Fixes the real edge-case bugs in §1.3. Small, reviewable output changes |
| 4 | **D4 LOCO plus GCTA-style GRM option** | M | Biggest *statistical* improvement in the field since 2014. Externally validatable against GCTA/fastlmm |
| 5 | **D5 GEMMA-style exact refit plus LRT/score** | M | Makes `--refit` practical. Validates against GEMMA |

D6 (eigh cache for missing patterns) is a cheap add-on during D1. D7 and D8 are deferred.

**Testing scaffolding shared by all of these:** golden outputs from Phase 1, plus the Phase 4
simulation harness (null QQ/λ_GC, power at causal SNPs, h² recovery). The external cross-checks
(GEMMA, GCTA, fastlmm, plink2) need only be run once to produce stored reference files. They should
not become CI dependencies.

---

## 5. References

- Kang HM et al. (2008) Efficient control of population structure in model organism association mapping. *Genetics* 178:1709–1723. (EMMA)
- Kang HM et al. (2010) Variance component model to account for sample structure in genome-wide association studies. *Nat Genet* 42:348–354. (EMMAX)
- Zhang Z et al. (2010) Mixed linear model approach adapted for genome-wide association studies. *Nat Genet* 42:355–360. (P3D)
- Lippert C, Listgarten J, Liu Y, Kadie CM, Davidson RI, Heckerman D (2011) FaST linear mixed models for genome-wide association studies. *Nat Methods* 8:833–835.
- Yang J, Lee SH, Goddard ME, Visscher PM (2011) GCTA: a tool for genome-wide complex trait analysis. *AJHG* 88:76–82.
- Zhou X, Stephens M (2012) Genome-wide efficient mixed-model analysis for association studies. *Nat Genet* 44:821–824. (GEMMA)
- Listgarten J, Lippert C, Kadie CM, Davidson RI, Eskin E, Heckerman D (2012) Improved linear mixed models for genome-wide association studies. *Nat Methods* 9:525–526. (proximal contamination, FaST-LMM-Select)
- Svishcheva GR et al. (2012) Rapid variance components–based method for whole-genome association analysis. *Nat Genet* 44:1166–1170. (GRAMMAR-Gamma)
- Speed D, Hemani G, Johnson MR, Balding DJ (2012) Improved heritability estimation from genome-wide SNPs. *AJHG* 91:1011–1021. (LDAK)
- Yang J, Zaitlen NA, Goddard ME, Visscher PM, Price AL (2014) Advantages and pitfalls in the application of mixed-model association methods. *Nat Genet* 46:100–106. (LOCO)
- Zhou X, Stephens M (2014) Efficient multivariate linear mixed model algorithms for genome-wide association studies. *Nat Methods* 11:407–409.
- Loh P-R et al. (2015) Efficient Bayesian mixed-model analysis increases association power in large cohorts. *Nat Genet* 47:284–290. (BOLT-LMM)
- Speed D et al. (2017) Reevaluation of SNP heritability in complex human traits. *Nat Genet* 49:986–992.
- Loh P-R, Kichaev G, Gazal S, Schoech AP, Price AL (2018) Mixed-model association for biobank-scale datasets. *Nat Genet* 50:906–908.
- Zhou W et al. (2018) Efficiently controlling for case-control imbalance and sample relatedness in large-scale genetic association studies. *Nat Genet* 50:1335–1341. (SAIGE)
- Jiang L et al. (2019) A resource-efficient tool for mixed model association analysis of large-scale data. *Nat Genet* 51:1749–1755. (fastGWA)
- Kadie CM, Heckerman D (2017/2019) Ludicrous speed linear mixed models for genome-wide association studies. *bioRxiv* 10.1101/154682.
- Mbatchou J et al. (2021) Computationally efficient whole-genome regression for quantitative and binary traits. *Nat Genet* 53:1097–1103. (REGENIE)
- Jiang L, Zheng Z, Fang H, Yang J (2021) A generalized linear mixed model association tool for biobank-scale data. *Nat Genet* 53:1616–1621. (fastGWA-GLMM)
- Gilmour AR, Thompson R, Cullis BR (1995) Average information REML. *Biometrics* 51:1440–1450.
- Halko N, Martinsson PG, Tropp JA (2011) Finding structure with randomness. *SIAM Rev* 53:217–288.
- Devlin B, Roeder K (1999) Genomic control for association studies. *Biometrics* 55:997–1004.

Software (versions as seen on package indices, late 2026; verify before pinning):

- fastlmm: https://fastlmm.github.io/ and https://pypi.org/project/fastlmm/ (0.6.x)
- bed-reader: https://github.com/fastlmm/bed-reader (1.1.0)
- pandas-plink: https://github.com/limix/pandas-plink (2.3.2)
- Pgenlib: https://pypi.org/project/Pgenlib/ (0.95.1, LGPL-3.0)
- PLINK 2: https://www.cog-genomics.org/plink/2.0/
- GEMMA: https://github.com/genetics-statistics/GEMMA
- REGENIE: https://rgcgithub.github.io/regenie/ (4.1.x)
- GCTA/fastGWA: https://yanglab.westlake.edu.cn/software/gcta/
- sgkit: https://github.com/sgkit-dev/sgkit
- limix / glimix-core: https://github.com/limix
- SciPy eigh: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigh.html

Caveats:

- Exact current version numbers came from search-engine and package-index snapshots and were not
  verified on each project's release page.
- The maintenance status of limix and glimix-core was not confirmed.
- Journal page numbers are from memory and standard citations; the DOIs for Yang 2014
  (10.1038/ng.2876) and Listgarten 2012 (10.1038/nmeth.2037) were verified.
