# pylmm user guide: ad-hoc analysis with the library

This guide shows how to use pylmm from Python for analyses the `pylmm` command line does not cover:
subsets, custom loops, your own covariates, heritability, and checking results. The command line is
covered too (section 9), because most analyses mix both.

Every code block in this file runs in CI (`tests/test_docs.py`): the blocks of this file run in order, in
one Python namespace, from a scratch directory where `data/`, `docs/`, `sims/` and `skills/` are links to
the repository's folders. So the paths below are relative to the repository root, and a block can use the
variables of the blocks above it. Blocks marked `skip-ci` need something CI does not have (a network
install, plink2).

Contents: 1 install · 2 the model · 3 loading data · 4 kinship · 5 the null fit · 6 one SNP ·
7 scans · 8 missing data · 9 the command line · 10 reading the results · 11 performance ·
12 pitfalls · 13 validated behaviour

## 1. Install

pylmm needs Python 3.10+, numpy and scipy. Install the Python 3 branch, `main` (the repository's default
branch still holds the old Python 2 code):

```bash skip-ci: network install
pip install "git+https://github.com/nickFurlotte/pylmm@main"
# or, in a uv project
uv add "pylmm @ git+https://github.com/nickFurlotte/pylmm@main"
```

A git install clones the whole repository, including ~215 MB of example data, so it takes a while. From a
clone, `uv sync` (add `--group sims` for the simulation framework) and prefix commands with `uv run`:

```bash
uv run pylmm --version
uv run python -c "import pylmm, pylmm.lmm; print(pylmm.__version__)"
```

## 2. The model

For `n` individuals with phenotype `Y`, covariates `X0` (n × q) and kinship `K` (n × n):

```text
Y = X0 b + x a + u + e,    u ~ N(0, h s2 K),    e ~ N(0, (1 - h) s2 I)
```

* `h` is the heritability (the share of the total variance `s2` carried by `K`; see section 5 for when
  that equals the h² of a paper), `s2` is called `sigma` in the code.
* `x` is one SNP. `association` and `gwas` test `a = 0` with a Wald t test, `t = a_hat / se`, against a
  t distribution with `n - q - 1` degrees of freedom.
* By default `h` and `s2` are fitted once without a SNP (the "null fit") and reused for every SNP (EMMAX /
  P3D). `refit=True` (`--refit`) refits them per SNP.
* REML is the default everywhere (`reml=True`, `--reml`); `reml=False` / `--ml` is maximum likelihood.

The modules: `pylmm.lmm` (`calculate_kinship`, `LMM`, `gwas`), `pylmm.plink_io` (`PlinkReader`),
`pylmm.samples` (sample IDs and alignment), `pylmm.misc` (`fit_two`, two kinships). Their docstrings are
complete (numpydoc, with examples): `help(pylmm.lmm.LMM)`.

## 3. Loading data

### PLINK BED/BIM/FAM

`PlinkReader` reads one SNP at a time. Take the first `m` SNPs with `itertools.islice`; reading all 98,740
SNPs of the example as float64 needs about 1 GB.

```python
import itertools
import numpy as np
from pylmm.plink_io import PlinkReader

BED = "data/snps.132k.clean.noX"          # .bed/.bim/.fam: 1219 mice, 98,740 SNPs
with PlinkReader(BED, pheno_file=BED + ".fake.phenos") as reader:
    snps = [(g, name) for g, name in itertools.islice(reader.iter_snps(), 2000)]
    ids = reader.indivs                   # [(FID, IID), ...] in .fam order
    Y_fake = reader.phenos[:, 0]          # phenos is (n, n_phenotypes)

W = np.column_stack([g for g, _ in snps])  # (n, m) genotype matrix
snp_names = [name for _, name in snps]
n = len(ids)
print(W.shape, ids[:2], snp_names[0])
assert W.shape == (1219, 2000) and ids[0] == ("1", "1")
```

**Genotype coding.** By default (`norm_genotype=True`) each SNP is standardised: missing calls are replaced
by the mean, then the SNP is centred and scaled to variance 1. With `norm_genotype=False` you get the raw
value, the **fraction of copies of A2**, the second allele of the `.bim` line (column 6): 0.0, 0.5, 1.0, or
nan for a missing call. It is not a 0/1/2 count. A plink2-written `.bim` has A1 = ALT and A2 = REF, so the
counted allele is REF.

```python
with PlinkReader(BED, norm_genotype=False) as reader:
    g_raw, name = next(reader.iter_snps())
print(name, np.unique(g_raw[~np.isnan(g_raw)]), int(np.isnan(g_raw).sum()), "missing")
assert set(np.unique(g_raw[~np.isnan(g_raw)])) <= {0.0, 0.5, 1.0}
```

To keep the chromosome and position (PlinkReader keeps only the SNP ID), read the `.bim` yourself; its
lines are in the order the reader iterates. `skills/convert-genotype-formats/scripts/plink_tools.py` has
`read_bim`, plus BED writing and subsetting helpers that pylmm itself does not have:

```python
import sys
sys.path.insert(0, "skills/convert-genotype-formats/scripts")
import plink_tools

bim = plink_tools.read_bim(BED)            # chrom, snp, cm, pos, a1, a2 arrays
assert list(bim["snp"][:2000]) == snp_names
print(bim["chrom"][0], bim["pos"][0], bim["a1"][0], bim["a2"][0])
```

### Other genotype formats

* `PlinkReader(prefix, type="t")`: PLINK TPED/TFAM, alleles coded `1`/`2` (`0` missing; make one with
  `plink --bfile X --recode 12 transpose --out Y`).
* `PlinkReader(path, type="emma")`: an EMMA matrix, one SNP per row, one individual per column, no IDs
  (individuals are positions `0..n-1`, and every other file is matched by position).
* VCF, BCF, BGEN, PGEN, VCF Zarr, Hail: convert to BED first, see the `convert-genotype-formats` skill.
* Any genotype matrix you already have in numpy works directly with `calculate_kinship`, `LMM` and `gwas`.

```python
with PlinkReader("data/hmdp.liver.snps", type="emma") as emma:
    G_emma = np.column_stack([g for g, _ in itertools.islice(emma.iter_snps(), 100)])
print(G_emma.shape, emma.indivs[:3])       # (94, 100) [0, 1, 2]
```

### Phenotypes and covariates

Phenotype and covariate files are whitespace-separated `FID IID value ...` tables, optionally with a
header line whose first two fields are `FID IID` (any case; `#FID` is not recognised), naming the
columns. Missing values: `NA` and `-9` in phenotypes, only `NA` in covariates.

```python
from pylmm import samples

# a phenotype file with a header, rows in a different order than the .fam, one individual missing
rng = np.random.default_rng(7)
with open("pheno.txt", "w") as f:
    f.write("FID IID weight glucose\n")
    for fid, iid in ids[::-1][:-1]:        # reversed, and the first .fam individual left out
        f.write("%s %s %.3f %.3f\n" % (fid, iid, rng.normal(30, 3), rng.normal(150, 20)))
with open("covar.txt", "w") as f:
    f.write("FID IID sex age\n")
    for i, (fid, iid) in enumerate(ids):
        f.write("%s %s %d %s\n" % (fid, iid, i % 2, "NA" if i == 5 else 8 + i % 5))

import warnings
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    reader = PlinkReader(BED, pheno_file="pheno.txt")
    glucose = reader.read_phenos("pheno.txt", columns=["glucose"])[:, 0]
    cov = reader.read_covariates("covar.txt", columns=["sex", "age"])
print(reader.pheno_names, reader.covariate_names, cov.shape)
print(caught[0].message)                   # 1 of 1219 individuals are not in the phenotype file ...
assert np.isnan(glucose[0]) and np.isnan(cov[5, 1])
```

Rows are matched to the genotype file by `(FID, IID)`, never by row position: the file's order does not
matter, its extra individuals are ignored, and individuals it lacks get `nan` (with a warning).
`PlinkReader(..., id_match="iid")` matches on the IID alone, for files whose FIDs disagree (one file uses
`0` as FID, another repeats the IID). Duplicate IDs are an error. `pylmm.samples.align_samples` does the
same alignment for any lists of IDs, for example when you load data with other tools:

```python
al = samples.align_samples({"genotypes": ids, "phenotype": samples.read_sample_table("pheno.txt").ids})
print(al.n, "shared;", al.report()[0])
assert al.n == n - 1
```

`reader.subset(ids)` keeps, and reorders, individuals in `indivs`, `phenos`, `K` and every later genotype
vector together.

## 4. Kinship

Two implementations of the realized relationship matrix `K = Z Z' / m` (`Z` the standardised SNPs):

* `pylmm kinship` (the command line) streams the SNPs of a file through `PlinkReader`'s standardisation
  (centre and scale over the called values, then put 0 for missing calls) and writes `K` plus `K.id`, the
  `FID IID` of each row. In Python that is `W @ W.T / m` for `W` read with the default
  `norm_genotype=True`.
* `calculate_kinship(W)` mean-imputes, then standardises each column of whatever `W` you give it (0/1/2,
  dosages, fractions, already standardised).

They agree when no call is missing. With missing calls the reader's SNPs have variance below 1, so the
command-line K is a little smaller (bug 8, deferred to Phase 6.3). Both skip monomorphic SNPs but count
them in `m`.

```python
from pylmm.lmm import calculate_kinship

K_2000 = calculate_kinship(W)              # from the 2000 SNPs above (67 missing calls in SNP 1 alone)
print(K_2000.shape, round(float(np.mean(np.diag(K_2000))), 4))
```

To try the command line on the same 2000 SNPs, copy them to a small BED file set (byte for byte, with
`plink_tools.subset_bed`; `plink2 --bfile X --extract snps.txt --make-bed` does the same):

```python
plink_tools.subset_bed(BED, "mini", np.arange(2000))
```

```bash
uv run pylmm kinship --bfile mini -o mini.kin -q
head -n 2 mini.kin.id
```

```python
K_cli = np.loadtxt("mini.kin")
assert np.allclose(K_cli, W @ W.T / W.shape[1], atol=1e-12)      # the reader's standardisation
print("mean diagonal: CLI %.4f, calculate_kinship %.4f; max |difference| %.3f"
      % (np.diag(K_cli).mean(), np.diag(K_2000).mean(), np.abs(K_cli - K_2000).max()))
assert samples.read_id_file("mini.kin.id") == ids

# The rest of this guide uses the kinship of all 98,740 SNPs, shipped with the example data
# (`pylmm kinship --bfile data/snps.132k.clean.noX -o full.kin` recomputes it, in ~10 s).
K_FILE = BED + ".pylmm.kin"                 # it has its ID file, K_FILE + ".id"
K = np.loadtxt(K_FILE)
```

* The `.id` sidecar is what lets `pylmm gwas` (and `PlinkReader(k_file=...)`) align K to the genotype
  file by ID. Without it K's rows are *assumed* to be in genotype order (with a warning). For a kinship
  computed elsewhere, write `K.id` yourself: `pylmm.samples.write_id_file("K.id", ids)` or
  `plink_tools.write_kinship("K", K, ids)`.
* SNPs that are monomorphic or all missing are skipped but still counted in `m`, so K shrinks if many SNPs
  are monomorphic (a known issue, bug 8). This changes `h` (section 5) but not t or p.
* `calculate_kinship(W, center=True)` returns a centred K scaled so that `trace(P K P) = n - 1`.
* With IDs, `PlinkReader(prefix, k_file="K")` reads K, reorders it to the genotype file, and drops
  genotype individuals that are not in it from `indivs`, `phenos` and the genotype vectors.

## 5. The null fit and heritability

A simulated phenotype with a known heritability makes the numbers checkable: `u ~ N(0, 0.6 K)` plus one
causal SNP. (The `simulate-lmm-data` skill does this with the `sims/` framework.)

```python
from scipy import linalg
from pylmm.lmm import LMM

kva, kve = linalg.eigh(K)                  # reuse it: every LMM(Y, K) would recompute it
rng = np.random.default_rng(1)
u = kve @ (np.sqrt(np.clip(kva, 0, None)) * rng.standard_normal(n)) * np.sqrt(0.6)
x_causal = W[:, 100]                       # a standardised SNP
Y = 10 + 0.8 * x_causal + u + rng.normal(scale=np.sqrt(0.4), size=n)

model = LMM(Y, K, kva, kve)                # X0 defaults to an intercept
h, beta, sigma, ll = model.fit()           # REML
print("h=%.3f  s2=%.3f  intercept=%.2f  REML LL=%.1f" % (h, sigma, beta[0, 0], ll))
assert 0.4 < h < 0.8
```

* `h` is a numpy float; `sigma` (the total variance `s2`) and `ll` are floats; `beta` is a (q, 1) array.
  The genetic and residual variances are `h * sigma` and `(1 - h) * sigma`.
* After `fit`, the model keeps `opt_h`, `opt_sigma`, `opt_beta`, `opt_ll`, and the log-likelihood on the
  search grid, `ll_grid` over `h_grid` (0, 0.01, ..., 0.99).
* `fit(reml=False)` is ML. Prefer REML: with a centred kinship built from more SNPs than individuals the
  ML optimum can run to the 0.99 cap (no MLE exists; Phase 4 bug 17).
* `h` is searched on [0, 0.99]: `h = 1` is never returned and a true h² ≥ 0.98 is reported as 0.99
  (deferred to Phase 6.3). Treat 0 and 0.99 as boundary estimates.

**Is `h` comparable with a published h²?** `h` is the share of `s2` on the scale of K. It equals the
proportion of phenotypic variance explained only if K is scaled so that the average variance it implies is
1 (Gower's centring: `trace(P K P) / (n - 1) = 1`, `P = I - 11'/n`); a mean diagonal of 1 is the usual
approximation. A kinship from another tool, or one shrunk by monomorphic SNPs, may not be. Dividing K by
`c` changes `h` exactly as `h' = h c / (h c + 1 - h)`, and changes no test statistic:

```python
P = np.eye(n) - 1.0 / n
c = np.trace(P @ K @ P) / (n - 1)          # 1 for a Gower-scaled K
h_gower = LMM(Y, K / c).fit()[0]
print("c=%.4f  h=%.4f  h(K/c)=%.4f  predicted=%.4f" % (c, h, h_gower, h * c / (h * c + 1 - h)))
assert abs(h_gower - h * c / (h * c + 1 - h)) < 1e-4
```

The `estimate-heritability` skill covers putting `h` on another tool's kinship scale (e.g. a GCTA GRM with
diagonal ≈ 2 in inbred lines), PVE, standard errors and profile-likelihood intervals for `h`.

## 6. Testing one SNP

```python
t, p, b, b_var = model.association(x_causal, return_beta=True)
print("t=%.2f  p=%.2e  beta=%.3f  se=%.3f" % (t, p, b, np.sqrt(b_var)))
assert p < 1e-6
t_null, p_null = model.association(W[:, 1500])   # t and p only
```

* `association` uses the `h` of the last `fit` (or `h=`). It returns numpy floats; `b_var` is a float.
* The SNP may be `(n,)` or `(n, 1)`, over the same individuals as `Y`, and must be complete: a `nan` makes
  scipy raise `ValueError: array must not contain infs or NaNs` (use `gwas`, section 8).
* `beta` is per unit of the column you pass: here per SD of the standardised genotype.
* `association` does not check that the SNP is estimable (not constant, not collinear with `X0`):
  `pylmm.lmm.snp_estimable(x, X0)` does, as `gwas` does for you.
* `fit(X=x)` followed by `association(x)` is the per-SNP refit.

## 7. Scans with `gwas`

`gwas(Y, X, K)` tests every column of `X` and returns two lists, t statistics and p-values (`nan` for a SNP
that cannot be tested: constant, collinear with the covariates, all missing).

```python
from pylmm.lmm import gwas

ts, ps = gwas(Y, W, K, kva, kve)           # null fit once, REML; ~1 s for 2000 SNPs
ps = np.asarray(ps)
top = np.argsort(ps)[:3]
print([(snp_names[j], "%.1e" % ps[j]) for j in top])
assert ps[100] < 1e-10                   # the causal SNP; in this inbred panel SNPs in LD with it
                                         # (and with the polygenic effect) can rank even higher
```

**Covariates and the intercept.** `X0` is used exactly as given. `read_covariates` returns the covariate
columns only, so add the intercept yourself, or the model has none and the test statistics change, often
a lot. (`pylmm gwas --covar` adds it for you; `--no-intercept` turns that off.)

```python
X0 = np.column_stack([np.ones(n), cov])    # intercept + sex + age
keep = ~np.isnan(cov).any(axis=1)          # gwas drops missing Y, not missing covariates
ts_cov, ps_cov = gwas(Y[keep], W[keep][:, :200], K[np.ix_(keep, keep)], X0=X0[keep])
ts_noint, _ = gwas(Y[keep], W[keep][:, :200], K[np.ix_(keep, keep)], X0=cov[keep])  # WRONG: no intercept
print("max |t| change without the intercept: %.2f" % np.nanmax(np.abs(np.subtract(ts_cov, ts_noint))))
```

`gwas` and `LMM` drop individuals with a missing `Y` themselves (and recompute the eigendecomposition),
but `X0` must be complete on the individuals with a phenotype, so drop or impute missing covariates first,
as above. Other options: `reml=False` (ML everywhere), `refit=True`, and `ids=` / `kinship_ids=` to have
`gwas` check that Y and K list the same individuals in the same order.

## 8. Missing genotypes

* `gwas` handles a SNP with missing calls (`nan`) by testing it on the called individuals only (a new LMM
  on that subset, with its own eigendecomposition: slower).
* `LMM.association` does not; impute first, or use `gwas`.
* Genotypes read with `norm_genotype=True` (the default) are already mean-imputed, so they have no `nan`
  (except an all-missing SNP, returned as all `nan`, and a constant SNP, returned as zeros: both untestable).
* `pylmm gwas` mean-imputes by default; `--remove-missing-genotypes` tests each SNP on its called
  individuals instead.

```python
X = W[:, :5].copy()
X[:50, 2] = np.nan                         # 50 missing calls in SNP 2
ts_m, ps_m = gwas(Y, X, K, kva, kve)
print(["%.3f" % t for t in ts_m])
assert np.all(np.isfinite(ts_m))
```

## 9. The command line

`pylmm gwas` is the same model as `gwas`, reading files. Write the simulated phenotype (and the fake one)
to a phenotype file with a header, and the covariates, then scan the `mini` file set of section 4:

```python
plink_tools.write_pheno("sim.pheno", ids, np.column_stack([Y, Y_fake]), ["trait", "noise"])
plink_tools.write_pheno("sim.covar", ids, cov, ["sex", "age"])
```

```bash
pylmm gwas --bfile mini --kinship data/snps.132k.clean.noX.pylmm.kin \
    --pheno sim.pheno --pheno-name trait --covar sim.covar --covar-name sex age \
    -o trait.tsv -v 2> trait.log
grep "null model" trait.log
head -3 trait.tsv
```

* `--kinship K` uses `K.id`, if it exists, to align K by ID; without it, it warns and assumes genotype
  order.
* The intercept is added unless `--no-intercept`; individuals with a missing phenotype or covariate are
  dropped (with a warning).
* `--pheno-col N` is 1-based (like plink2 `--pheno-col-nums`); `--pheno-name` picks by header name.
* `--id-match iid` matches on IID alone; `--ml`, `--refit`, `--remove-missing-genotypes` as in the
  library; `--eigen` reuses a decomposition (section 11).
* Exit code 0 on success, 2 for a usage or input error (one `pylmm: error: ...` line), 1 for an unexpected
  error (rerun with `--debug` for the traceback).
* `pylmm gwas --help` lists every option, with examples.

## 10. Reading the results

The output is tab-separated, one line per SNP, in `.bim` order:

| column | meaning |
|---|---|
| `SNP_ID` | `.bim` column 2 |
| `BETA` | the SNP effect, **per SD of the standardised genotype** of the A2 allele (`.bim` column 6) |
| `BETA_SD` | the **standard error** of `BETA` (despite the name) |
| `F_STAT` | the signed **Wald t statistic**, `BETA / BETA_SD` (despite the name; t² is the F) |
| `P_VALUE` | two-sided, from t(n - q - 1) |

`nan` in all four value columns marks a SNP that cannot be tested.

```python
res = np.genfromtxt("trait.tsv", names=True, dtype=None, encoding="utf-8")
assert np.allclose(res["F_STAT"], res["BETA"] / res["BETA_SD"], equal_nan=True)
best = res[np.nanargmin(res["P_VALUE"])]
print(best["SNP_ID"], "%.2e" % best["P_VALUE"])
assert res["P_VALUE"][100] < 1e-10
```

**Per-allele effects.** To compare `BETA` with tools that report the effect per copy of A1 (plink2 `--glm`,
GEMMA, GCTA and REGENIE count A1 = ALT), divide by the SD of the 0/1/2 dosage *over the called
individuals* (the SD the reader standardised by) and flip the sign: `beta_A1 = -BETA / np.nanstd(2 *
g_raw)`, with `g_raw` the `norm_genotype=False` values over the individuals of the genotype file (after
any dropped by the kinship). `BETA_SD` converts the same way; t and p need no conversion (only the sign of
t flips with the allele). Do not compare raw `BETA` values between runs with and without
`--remove-missing-genotypes` (section 12).

The same conversion, checked with the library on the causal SNP (`x_causal` is the standardised SNP 100,
`model` the fit of section 5):

```python
with PlinkReader(BED, norm_genotype=False) as reader:
    g_raw = next(itertools.islice(reader.iter_snps(), 100, None))[0]
sd_called = np.nanstd(2 * g_raw)           # SD of the dosage over the called individuals
dosage_a1 = 2 - 2 * np.where(np.isnan(g_raw), np.nanmean(g_raw), g_raw)   # copies of A1, mean-imputed
_, _, b_std, _ = model.association(x_causal, return_beta=True)
_, _, b_a1, _ = model.association(dosage_a1, return_beta=True)
print("per SD of A2: %.4f   per copy of A1: %.4f   -b_std / sd: %.4f" % (b_std, b_a1, -b_std / sd_called))
assert np.isclose(b_a1, -b_std / sd_called)
```

The `interpret-gwas-results` skill covers λ_GC, QQ plots, thresholds and when to distrust a scan.

## 11. Performance

* The cost is one O(n³) eigendecomposition of K plus O(n²) per SNP (a rotation). n of a few thousand is
  quick; K itself is 8 n² bytes (3.2 GB at n = 20,000), the practical limit.
* Compute the eigendecomposition once and pass it: `LMM(Y, K, kva, kve)`, `gwas(Y, X, K, kva, kve)`. It is
  ignored (recomputed) when `Y` has missing values, because K is then subset.
* `pylmm kinship --eigen-npz eig.npz` writes it for `pylmm gwas --eigen eig.npz`; it is used only when no
  individual is dropped or reordered (otherwise it is ignored, with a warning).
* SNPs with missing calls under `--remove-missing-genotypes` (or library `gwas`) each cost an
  eigendecomposition; `--refit` costs a likelihood search per SNP.
* Process SNPs in blocks with `islice` rather than building a full genotype matrix.

```bash
pylmm kinship --bfile mini -o mini.kin --eigen-npz mini.eig.npz -q
pylmm gwas --bfile mini --kinship mini.kin --eigen mini.eig.npz --pheno sim.pheno -o fast.tsv -q
```

## 12. Pitfalls

1. **No intercept with library covariates.** `X0` is used as given; prepend `np.ones(n)`.
2. **Kinship order.** A kinship without a `.id` file is assumed to be in genotype order. Write the `.id`
   file whenever K comes from elsewhere, and check `samples.read_id_file(...) == reader.indivs`.
3. **FID mismatches.** Phenotype rows are matched on `(FID, IID)`. If your phenotype file uses other FIDs
   than the `.fam` (e.g. `0`, or plink2's `--double-id`), every individual is "missing"; use
   `--id-match iid` / `id_match="iid"`.
4. **`#FID` headers** (plink2 style) are not recognised as headers; write `FID IID ...`.
5. **`-9` in covariates** is a value, not missing (only `NA` is). In phenotypes `-9` is missing.
6. **Effect allele and scale.** BETA is per SD of the A2 allele (REF for plink2-written files, the
   *major* allele for plink 1.9 without `--keep-allele-order`), opposite in sign to plink2 `--glm`.
   With `--remove-missing-genotypes`, SNPs without missing calls are tested on the raw 0–1 scale and SNPs
   with missing calls on the standardised scale, so BETA is not comparable across SNPs (t and p are fine).
7. **`h` at a boundary.** 0.99 means "≥ 0.98, capped", 0 means none detected; neither is a precise
   estimate. ML can sit at 0.99 for a centred K from more SNPs than individuals; use REML.
8. **K scale.** `h` depends on how K is scaled (section 5); test statistics do not.
9. **Inbred panels.** Replicate animals of one strain share a genotype: K is rank-deficient (the example
   K has rank about 110 for 1219 mice), the effective number of independent tests is small, and λ_GC from
   one scan is noisy.
10. **`association` with `nan`** fails; `gwas` handles missing calls.
11. **Imputation set.** When K drops individuals, genotypes are standardised over the kept individuals, but
    individuals dropped for a missing phenotype or covariate still count in the mean and SD (|Δt| ≤ 0.024
    in our check; Phase 6.3b).

## 13. Validated behaviour

pylmm's default path (REML null fit, Wald test) was checked by simulation in Phase 4
([docs/phase4-simulations.md](phase4-simulations.md), framework in [sims/README.md](../sims/README.md)):

* **Calibrated null p-values** on unrelated, structured (Balding–Nichols), family and the bundled mouse
  data, for h² > 0: pooled type-I error 0.0495–0.0500 at α = 0.05 and mean χ² 0.99–1.01. At h² = 0 the
  test is mildly conservative (REML puts ĥ at 0).
* **Matches an independent dense REML oracle**: Wald t to 2e-7, ĥ RMS within 0.0033 of the textbook
  optimum for h² ≤ 0.95.
* **Power ≥ OLS**, much higher under population structure (e.g. 0.94 vs 0.15 at h² = 0.5, 5% variance
  explained, Balding–Nichols), and OLS is badly inflated there.
* **Known limits** (deferred to Phase 6.3): ĥ capped at 0.99; ML unidentified with a centred K from m > n
  SNPs; kinship normalisation (bug 8).
* The Phase 1 golden tests (`tests/golden/`) pin every number of the original 2013 code: the CLI output is
  byte-identical for the default path.
