---
name: simulate-lmm-data
description: Simulate genotypes and phenotypes with a known heritability, known causal SNPs and covariates using pylmm's sims/ framework (unrelated, Balding-Nichols structured, family and real-genotype sources; kinship with cached eigendecomposition; the LMM phenotype generator; metrics; an independent REML oracle), write them as PLINK files, and sanity-check a GWAS or heritability pipeline end to end - null calibration (lambda_GC, type-I error), causal SNP recovery and h2 recovery. Use when asked to simulate GWAS or mixed-model data, to test or validate a pylmm pipeline on data with a known answer, or to run or write a sims/ study.
---

# Simulate LMM data and sanity-check a pipeline

`sims/` (repository root, not part of the installed package; docs in [sims/README.md](../../sims/README.md))
generates data exactly under pylmm's model. Its generator modules need only numpy and scipy; the study
runner's reports need the `sims` dependency group (`uv sync --group sims`: pandas, matplotlib,
statsmodels). Run from the repository root (the blocks below run in CI in order).

## 1. Genotypes

```python
import numpy as np
from sims import genotypes, kinship, metrics, oracle, phenotypes

spec = {"source": "balding_nichols", "n": 400, "m": 4000, "k": 3, "fst": 0.1, "seed": 1}
geno = genotypes.get(spec)                 # cached, READ-ONLY arrays: geno.G (n x m, 0/1/2), geno.pops
print(geno.G.shape, np.bincount(geno.pops))
```

| source | parameters | structure |
|---|---|---|
| `unrelated` | `n, m, maf, seed` | independent HWE SNPs, K ≈ I |
| `balding_nichols` | `n, m, k, fst, maf, admixture, seed` | k populations (or Dirichlet admixture) |
| `family` | `n_families, n_sibs, m, include_parents, maf, seed` | nuclear sibships, relatedness 0.5 |
| `bed` | `n_snps, snps, n_indivs, indivs, select, seed` | the bundled 1219 mice (0/0.5/1 coding, real LD, ~110 distinct genotypes, ~1% missing calls) |
| `hmdp` | same | the bundled 94-strain EMMA text data |

For `bed` / `hmdp`: `n_snps` takes that many SNPs, chosen by `select`: `"first"` (the default: the first
`n_snps` in `.bim` order, i.e. the start of chromosome 1 only), `"even"` (evenly spaced over the whole
file, the usual choice) or `"random"` (with `seed`); `snps` gives explicit 0-based indices instead.
`n_indivs` / `indivs` choose individuals the same way. The chosen indices are in
`geno.meta["snp_index"]` / `geno.meta["indiv_index"]`, and `geno.indiv_ids` are `"FID:IID"` strings
(section 6 turns them into `(FID, IID)` pairs).

`genotypes.with_missing(G, rate, rng)` adds missing calls. Copy before changing (`geno.copy()`).

**Coding.** The synthetic sources give allele counts 0/1/2. The `bed` and `hmdp` sources give what
`PlinkReader(..., norm_genotype=False)` returns: the **fraction** of A2, 0/0.5/1 (nan = missing). Before
writing such a `G` with `plink_tools.write_bed`, double it (`2 * G`) or pass `coding="fraction"`;
`write_bed` raises `ValueError` on 0.5 rather than rounding it.

## 2. Kinship and a test-SNP split

Use disjoint SNP sets for K and for the null tests (no proximal contamination), or `overlap=True` for the
usual "K from all SNPs":

```python
kin_idx, test_idx = kinship.split_snps(geno.m, n_kinship=3000, n_test=1000)
kin = kinship.get(spec, snps=kin_idx)      # K = pylmm's calculate_kinship, plus cached kva, kve
print(kin.K.shape, round(float(np.mean(np.diag(kin.K))), 3))
```

## 3. Phenotypes

`phenotypes.simulate(kin, h2, rng, G=, causal_idx=, causal_var=, n_covariates=, missing_rate=)` draws
`y = X0 alpha + sum_j beta_j g_j + u + e` with `u ~ N(0, sigma_g2 K)`. The causal SNPs explain
`causal_var` of the variance (in-sample), and the rest is split `h2 : 1 - h2` between `u` and `e`:

```python
rng = np.random.default_rng(42)
G_test = geno.G[:, test_idx]
causal = [10, 500]                         # columns of G_test
ph = phenotypes.simulate(kin, h2=0.5, rng=rng, G=G_test, causal_idx=causal, causal_var=0.05,
                         n_covariates=1)
print("true h2=%.2f, beta=%s, X0 columns=%d" % (ph.h2, np.round(ph.beta, 3), ph.X0.shape[1]))
```

`ph.X0` holds the intercept and the simulated covariate. `ph.u`, `ph.e` and `ph.missing` are kept for
checks. `h2` is the parameter an LMM estimates when the causal SNPs are in `X0`; left out, they add to
the apparent h². With `pylmm`'s kinship the scale of K is ~1 for outbred synthetic data; for the `bed`
source see the kinship-scaling notes in `estimate-heritability`.

## 4. Check the library on it

```python
from pylmm.lmm import LMM, gwas

kva, kve = kin.eig_for_pylmm()             # a copy of kva, safe to pass on
h_hat = LMM(ph.y, kin.K, kva, kve, X0=ph.X0).fit()[0]
ts, ps = gwas(ph.y, G_test, kin.K, kva, kve, X0=ph.X0)
ps = np.asarray(ps)
null = np.setdiff1d(np.arange(len(ps)), causal)
print("h_hat=%.3f  lambda_GC(null SNPs)=%.3f  causal p=%s" % (h_hat, metrics.lambda_gc(ps[null]),
                                                           ["%.1e" % ps[j] for j in causal]))
print("type-I error:", metrics.type1_error(ps[null], (0.05, 0.01)))
assert 0.2 < h_hat < 0.8 and 0.8 < metrics.lambda_gc(ps[null]) < 1.25
```

`oracle.fit(y, K, X0)` is an independent dense REML/ML fit (O(n³) per evaluation: n ≤ 500) to compare
`h` with; `oracle.wald(y, K, x, X0, h=...)` the matching Wald test:

```python
ofit = oracle.fit(ph.y, np.asarray(kin.K), ph.X0, reml=True, eig_clamp=1e-6, h_grid="pylmm")
print("pylmm h=%.6f  oracle h=%.6f" % (h_hat, ofit.h))
assert abs(h_hat - ofit.h) < 1e-4
```

## 5. Check a file-based pipeline end to end

Write the simulation as PLINK files, run the command line, and check that the known answer comes back.
This is the test to run before trusting a new pipeline (format conversion, ID handling, covariates):

```python
import sys
sys.path.insert(0, "skills/convert-genotype-formats/scripts")
import plink_tools

iids = ["ind%d" % i for i in range(geno.n)]
fid_iid = [(i, i) for i in iids]
plink_tools.write_bed("sim", G_test, iids=iids)        # counts of A2; for a `bed` source G: 2 * G
plink_tools.write_pheno("sim.pheno", fid_iid, ph.y, ["y"])
plink_tools.write_pheno("sim.covar", fid_iid, ph.X0[:, 1:], ["c1"])          # no intercept column
plink_tools.write_kinship("sim.kin", kin.K, fid_iid)              # K from the kinship SNPs, + .id
```

```bash
pylmm gwas --bfile sim --kinship sim.kin --pheno sim.pheno --covar sim.covar -o sim.tsv -v
```

Keep `-v` in such checks: it logs how many individuals were read, matched and dropped, and the null fit;
`-q` would hide the warnings that reveal an ID mismatch.

```python
res = np.genfromtxt("sim.tsv", names=True, dtype=None, encoding="utf-8")
assert np.allclose(res["P_VALUE"], ps, rtol=1e-5)                 # CLI == library
assert np.all(np.sign(res["BETA"][causal]) == np.sign(ph.beta))  # write_bed counts A2 = the sim allele
print("pipeline OK: causal p =", ["%.1e" % p for p in res["P_VALUE"][causal]])
```

Things such a check catches: an ID mismatch (individuals dropped or misaligned), a missing intercept, a
flipped allele (BETA signs flip), a kinship for other samples (λ off, h wrong).

## 6. Real genotypes (the `bed` source) and inbred panels

The bundled mice are inbred strains with replicate animals: real LD, ~110 distinct genotypes among 1219
animals, and missing calls. Three things differ from the synthetic sources:

* **Missing calls.** `phenotypes.simulate` needs complete causal genotypes (`ValueError: causal
  genotypes must be complete`). Pick causal SNPs among the complete ones (below), or impute first,
  e.g. `G_imp = np.where(np.isnan(G), np.nanmean(G, axis=0), G)` (then the causal "genotypes" include
  imputed means).
* **Coding.** `G` is the fraction of A2 (0/0.5/1). `ph.beta` is the effect per unit of the `G` you pass,
  so the effect per copy of A2 is `ph.beta / 2`; double `G` (or `coding="fraction"`) before `write_bed`.
* **IDs.** `geno.indiv_ids` are `"FID:IID"` strings; split them, or take the pairs from
  `PlinkReader(...).indivs` with `geno.meta["indiv_index"]`.

```python
from pylmm.plink_io import PlinkReader

spec_bed = {"source": "bed", "n_snps": 3000, "select": "even"}   # every ~33rd SNP of the 98,740
mice = genotypes.get(spec_bed)
complete = ~np.isnan(mice.G).any(axis=0)
print(mice.G.shape, "missing calls %.1f%%, complete SNPs %d" % (100 * np.isnan(mice.G).mean(), complete.sum()))
fid_iid_m = [tuple(s.split(":", 1)) for s in mice.indiv_ids]          # "FID:IID" -> (FID, IID)
with PlinkReader("data/snps.132k.clean.noX") as r:                  # or from the reader
    assert fid_iid_m == [r.indivs[i] for i in mice.meta["indiv_index"]]
chrom = plink_tools.read_bim("data/snps.132k.clean.noX")["chrom"][mice.meta["snp_index"]]

test = np.flatnonzero((chrom == "1") & complete)      # test chromosome 1's complete SNPs
causal_m = test[[20, 100]]                            # columns of mice.G, complete
kin_all = kinship.get(spec_bed)                       # K from all 3000 SNPs, tested ones included
ph_m = phenotypes.simulate(kin_all, h2=0.5, rng=np.random.default_rng(1), G=mice.G,
                           causal_idx=causal_m, causal_var=0.1)
print("beta per unit of G (fraction) %s, per copy of A2 %s" % (np.round(ph_m.beta, 3), np.round(ph_m.beta / 2, 3)))
```

**Expect low power with K from the tested SNPs.** With few distinct strains, a K that includes the
tested SNPs (and the SNPs in LD with them, which in inbred panels span megabases) absorbs much of a causal
SNP's signal: its p-value is far weaker than the effect deserves (proximal contamination). For a pipeline
check, build K from SNPs that are not tested (a **disjoint K**); for a real analysis use **LOCO** (K
without the tested chromosome; `custom-association-analysis`):

```python
def causal_p(kin):
    kva_k, kve_k = kin.eig_for_pylmm()
    p_k = np.asarray(gwas(ph_m.y, mice.G[:, test], kin.K, kva_k, kve_k)[1])
    return p_k[np.searchsorted(test, causal_m)]

kin_disjoint = kinship.get(spec_bed, snps=np.setdiff1d(np.arange(mice.m), test))   # all but the tested
kin_loco = kinship.get(spec_bed, snps=chrom != "1")                                # all but chromosome 1
p_all, p_dis, p_loco = causal_p(kin_all), causal_p(kin_disjoint), causal_p(kin_loco)
for name, pv in (("K from all SNPs", p_all), ("disjoint K", p_dis), ("LOCO K", p_loco)):
    print("%-16s causal p = %s" % (name, ["%.1e" % x for x in pv]))
assert np.all(p_loco < p_all)
```

Each causal SNP explains 5% of the variance of 1219 animals, yet even with LOCO p is only ~1e-6: the
effective sample size is closer to the ~110 distinct genotypes than to 1219, and with two causal SNPs
on one chromosome the ranking of the three kinships varies between draws. The disjoint K still contains
chromosome-1 SNPs in LD with the causal ones, so it helps less than
LOCO. Under LOCO, λ_GC of one chromosome's scan can be well above 1 when that chromosome carries a
causal SNP: in inbred panels LD makes much of the chromosome associated. For what h² recovery to expect
in panels of this kind (wide intervals), see `estimate-heritability` section 7.

## 7. Calibration across replicates, and studies

One replicate is noisy. For calibration, pool null p-values over replicates (new phenotype draws on the
same genotypes) and look at the pooled type-I error and mean χ², not one λ_GC (`metrics.type1_error`,
`metrics.chi2_from_p`, `metrics.qq_data`, `metrics.rate_ci` for a Wilson CI).

For a grid of settings × replicates with seeds, resumable runs and reports, write a study
(`sims/studies/<name>.py` with `GRID`, `QUICK_GRID`, `REPS` and `run_cell(params, seed)`; template
`sims/studies/example_null.py`) and run it:

```bash
uv run --group sims python -m sims list
uv run --group sims python -m sims run example_null --quick
```

`--report` (or `python -m sims report <study>`) writes `summary.csv/md` and PNGs (needs the `sims`
group). Results go to `sims/results/<study>/` (set `SIMS_RESULTS_DIR` to redirect); commit summaries,
never `raw/`.

Phase 4's studies (S1–S5, [docs/phase4-simulations.md](../../docs/phase4-simulations.md)) are the
reference for what "calibrated" looks like: pooled type-I error 0.0495–0.0500 at α = 0.05 for h² > 0,
mildly conservative at h² = 0.
