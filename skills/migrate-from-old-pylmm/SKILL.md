---
name: migrate-from-old-pylmm
description: Port code and scripts written for the old pylmm (Python 2 era, camelCase API such as lmm.GWAS, calculateKinship, getMLSoln, optH, the pylmm.input module and input.plink, and the pylmmGWAS / pylmmKinship command lines) to the current pylmm API and the single pylmm CLI (pylmm gwas, pylmm kinship), using the repository's name and flag maps and a rewriting script. Covers the traps a plain rename misses - -p N becomes --pheno-col N+1, reader.next() becomes next(reader), kva=[] becomes None, scalar returns instead of 1x1 arrays, REML defaults and the positional output file. Use when old pylmm code, commands or tutorials fail with AttributeError, ModuleNotFoundError or unrecognized-argument errors.
---

# Migrate old pylmm code and commands

pylmm's Python names were made PEP 8 (Phase 5.4), the API was cleaned up (5b.1) and the two old programs
became one `pylmm` command (5b.2), all as **clean breaks: no aliases**. The complete maps:

* [docs/migration/name-map.csv](../../docs/migration/name-map.csv): every renamed Python name
  (`old_name, new_name, kind, module, phase, notes`; `kind=signature` rows are changed types/defaults).
* [docs/migration/cli-map.csv](../../docs/migration/cli-map.csv): every old command and flag (`kind` =
  renamed, value→flag, removed, semantics-changed, ...).
* [CHANGELOG.md](../../CHANGELOG.md): the why of each change.

`scripts/migrate.py` (in this skill) applies both maps. The blocks below run in CI, in order, from the
repository root.

## 1. Rewrite a Python script

```python
import subprocess, sys
OLD = '''\
import numpy as np
from pylmm import lmm
from pylmm import input

IN = input.plink("data/snps.132k.clean.noX", type="b", normGenotype=True)
IN.getSNPIterator()
snps = np.array([IN.next()[0] for _ in range(50)]).T
Y = IN.getPhenos("data/snps.132k.clean.noX.fake.phenos")[:, 0]
K = lmm.calculateKinship(snps)
L = lmm.LMM(Y, K, Kva=[], Kve=[])
hmax, beta, sigma, LL = L.fit(REML=True)
print("h", L.optH, "sigma", sigma[0, 0], "grid", L.LLs.shape, L.H[-1])
ts, ps, b, bv = L.association(snps[:, 0], REML=True, returnBeta=True)
TS, PS = lmm.GWAS(Y, snps, K, REML=True, refit=False)
'''
with open("old_analysis.py", "w") as f:
    f.write(OLD)
p = subprocess.run([sys.executable, "skills/migrate-from-old-pylmm/scripts/migrate.py", "old_analysis.py"],
                   capture_output=True, text=True, check=True)
NEW = p.stdout
print(NEW)
print(p.stderr)                            # NOTE lines: what still needs a human
```

The result still has one problem the script can only flag: `sigma[0, 0]` (sigma is a float now). Fix it
and run the migrated code:

```python
NEW = NEW.replace("sigma[0, 0]", "sigma")
assert "plink_io.PlinkReader(" in NEW and "next(IN)" in NEW and "kva=None" in NEW
assert "reml=True" in NEW and "return_beta=True" in NEW and "lmm.gwas(" in NEW
exec(compile(NEW, "migrated.py", "exec"), {})
```

`python skills/migrate-from-old-pylmm/scripts/migrate.py -i FILE...` rewrites in place. It works on
Python tokens (strings and comments are left alone): functions and classes are renamed anywhere, methods
and attributes only after a `.`, parameters only as keyword arguments of the call that owns them. Review
the diff: it cannot see types.

## 2. Rewrite command lines

```python
for old in [
    "pylmmGWAS -v --bfile data/x --kfile x.kin --phenofile x.phenos -p 1 --covfile x.cov out.txt",
    "python scripts/pylmmKinship.py -v --bfile data/x -e foo --eigenNpz x.npz -n 500 x.kin",
    "pylmmGWAS --emmaSNP g.txt --emmaPHENO p.txt --kfile K.txt --noMean --removeMissingGenotypes out.txt",
]:
    r = subprocess.run([sys.executable, "skills/migrate-from-old-pylmm/scripts/migrate.py", "--command", old],
                       capture_output=True, text=True, check=True)
    note = r.stderr.strip().replace("\n", "; ")
    print(r.stdout.strip() + ("    # " + note if note else ""))
```

A shell script or notes file (any file not ending in `.py`) passed to the script has every
`pylmmGWAS`/`pylmmKinship` line rewritten.

## 3. The traps (what a search-and-replace gets wrong)

| old | new | why it matters |
|---|---|---|
| `pylmmGWAS -p N` (0-based) | `pylmm gwas --pheno-col N+1` (1-based, like plink2) | renaming the flag alone tests the **next** column; or use `--pheno-name` |
| positional `OUTFILE` | `-o/--out FILE` | required option now |
| `pylmmKinship -e VALUE` | `--eigen-text` (a flag; the value was ignored) | |
| `--kfile`, `--phenofile`, `--covfile`, `--noMean` | `--kinship`, `--pheno`, `--covar`, `--no-intercept` | |
| `--REML` / `--ML` | `--reml` (default) / `--ml` | `--ML` is a real ML fit now, null and test |
| `--kfile2` | removed | it never worked; use `pylmm.misc.fit_two` in Python |
| `reader.next()` | `next(reader)` | Python 3 iterator protocol |
| `LMM(Y, K, Kva=[], Kve=[])` | `LMM(Y, K)` or `kva=None, kve=None` | an empty list raises `ValueError` now |
| `sigma[0, 0]`, `betaVar[0, 0]`, `Q[0][0]` | `sigma`, `beta_var`, `Q` | floats now (they were (1, 1) arrays); `beta` is still (q, 1) |
| `L.LL(h)` (ML by default) | `L.log_likelihood(h)` (**REML** by default) | pass `reml=False` for the old value; returns `LogLikResult(ll, beta, sigma, xx_inv)` |
| `GWAS(..., REML=False)` | `gwas(..., reml=False)` | was a REML test in disguise (bug 1); now ML throughout |
| `--refit`, `--removeMissingGenotypes` | `--refit`, `--remove-missing-genotypes` | refit by REML now (they silently used ML): outputs change |
| `getPhenos`, `getCovariates` | `read_phenos`, `read_covariates` | rows matched by (FID, IID); absent individuals are `nan` rows, not skipped |
| `plink.getSNPIterator()` | `PlinkReader.iter_snps()` | renamed twice (5.4 `get_snp_iterator`, 5b.1 `iter_snps`) |
| `misc.fitTwo(y, ..., wgrids=)` | `misc.fit_two(Y, ..., ngrids=)` | |
| `from pylmm import input` | `from pylmm import plink_io` | `input` shadowed the builtin |
| `pylmm.plink_io.IndivKey` | `pylmm.samples.SampleID` | |

Behaviour changes with no rename (check numbers, not just that the code runs):

* A kinship file `K` with a `K.id` next to it is aligned by ID; without it, a warning and genotype order.
* Phenotype/covariate rows are matched by ID (file order no longer matters); a missing covariate drops
  the individual instead of failing.
* Untestable SNPs (constant, collinear, all missing) give `nan` instead of `t ≈ 1e-8` or a crash.
* Messages go to the `pylmm` logger (`pylmm: warning: ...`), errors exit 2 with one line.
* The default GWAS output (`SNP_ID BETA BETA_SD F_STAT P_VALUE`) is byte-identical for the same inputs.

## 4. When the script is not enough

* Look the name up: `grep -i oldName docs/migration/name-map.csv`; for a flag,
  `grep -- --oldFlag docs/migration/cli-map.csv`.
* `AttributeError: 'LMM' object has no attribute 'optH'` and similar: the attribute is in the map
  (`opt_h`, `opt_sigma`, `opt_beta`, `opt_ll`, `kva`, `kve`, `yt`, `x0t`, `ll_grid`, `h_grid`).
* `ModuleNotFoundError: pylmm.input` → `pylmm.plink_io`; `python -m pylmm.cli.gwas` exits 2 with a
  pointer to `pylmm gwas`.
* `pylmm: error: unrecognized arguments: ...` → the old flag; see `pylmm gwas --help`.
