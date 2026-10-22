---
name: inbred-strain-panels
description: Data preparation for inbred strain panels with replicate animals (mouse HMDP, BXD, inbred rat or plant lines) - strain aliases and coisogenic labels with a genotype-identity audit, joining a published supplement or MPD export to strain genotypes, expanding genotypes and kinship to animals, PLINK-safe strain IDs, imputing over strains. Use when phenotypes are measured on several animals per strain or when reproducing an HMDP-style mouse GWAS.
---

# Inbred strain panels with replicate animals

In an inbred panel the **genotypes belong to strains** and the **phenotypes to animals**: one genotype
vector per strain, several animals per strain, often from different files or databases. Most mistakes
happen in the join between the two, not in the model. This skill is the data-preparation recipe; the
model side lives elsewhere:

* individual-level vs strain-mean set-up and heritability: `estimate-heritability` section 7;
* power, LOCO and λ_GC in panels with few distinct genotypes: `run-gwas` sections 1-2 (`pylmm gwas
  --loco`), `simulate-lmm-data` section 6, `custom-association-analysis` (`pylmm.loco`);
* comparing the result with a published scan: `interpret-gwas-results/comparison-worksheet.md`.

Every block below runs in CI, in order, on a small synthetic panel built in the first block. The worked
example at the end sketches the two published HMDP studies reproduced in
[docs/replication.md](../../docs/replication.md) (Farber et al. 2011, Bennett et al. 2010); their data
files are not in the repository.

**Planned, not yet in pylmm**: a strain-aware option that expands strain genotypes to animals
and imputes over strains, and marker metadata (position, counted allele) carried through the EMMA path.
Until then, use the recipes here. Available: EMMA's IBS kinship (`calculate_kinship(G, kind="ibs")`,
`pylmm kinship --kinship-type ibs`, section 3) and leave-one-chromosome-out scans, **recommended for these
panels**: a K from all SNPs absorbs much of a causal SNP's signal here, and LOCO roughly doubled QTL power
in S8 (`pylmm gwas --loco [--kinship-type ibs]`; expect λ_GC well above 1 for a polygenic trait under LOCO,
linked signal rather than inflation: never genomic-control it; for EMMA
input give the chromosomes with `--emma-map data/hmdp.liver.snps.map.tsv`, whose rows are the bundled
matrix's; `pylmm.loco` in the library). On a per-animal BED, `--loco` builds its kinships over the animals,
as `pylmm kinship` does.

## 1. The two inputs

* **Strain genotypes**: strains × SNPs, coded 0/1 for inbred strains (homozygous one way or the other),
  `nan` for a missing call, with the strain names as the genotype source spells them. For the bundled
  HMDP matrix: `data/hmdp.liver.snps` (SNPs × strains), `data/hmdp.liver.strains` and the marker map
  `data/hmdp.liver.snps.map.tsv` (see [data/README.md](../../data/README.md)).
* **Animal phenotypes**: one row per animal, `(strain, animal ID, value)`, with the strain names as the
  *phenotype* source spells them (a paper supplement, an MPD export). These two spellings differ.

```python
import hashlib
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, "skills/convert-genotype-formats/scripts")
import plink_tools
from pylmm.lmm import LMM, calculate_kinship, gwas

rng = np.random.default_rng(42)
m = 600                                             # SNPs (one chromosome here)
founders = rng.integers(0, 2, size=(8, m)).astype(float)
G_unique = founders[rng.integers(0, 8, size=(34, m)), np.arange(m)]   # 34 distinct strain genotypes
G_unique[rng.random(G_unique.shape) < 0.04] = np.nan                  # 4% missing calls

# the genotype source: 36 columns, the last label three times with identical columns (as in HMDP)
geno_labels = ["S%02d/J" % i for i in range(32)] + ["BTBR T+ tf/J"] + ["AXB-19/PgnJ"] * 3
G_cols = np.vstack([G_unique, G_unique[-1], G_unique[-1]])
snp_alleles = [("A", "G") if j % 3 else (None, None) for j in range(m)]  # some markers lack letters

# the phenotype source spells some strains differently, and has a strain with no genotypes
pheno_strains = ["S%02d/J" % i for i in range(32)] + [
    "BTBR T<+> Itpr3<tf>/J", "AXB19a/PgnJ", "AXB19/PgnJ", "AXB19b/PgnJ", "BXH20/KccJ"]
reps = rng.integers(2, 12, size=len(pheno_strains))                  # 2-11 animals per strain
animals = [(s, "m%03d" % (100 * i + k)) for i, (s, r) in enumerate(zip(pheno_strains, reps))
           for k in range(r)]
print(len(geno_labels), "genotype columns;", len(animals), "animals in", len(pheno_strains), "strains")
```

## 2. Strain names: aliases, duplicates and coisogenic sublines

Match strain names **exactly**, through an explicit alias table that you write down and keep with the
analysis. Never fuzzy-match (`BXD-1/TyJ` vs `BXD1/TyJ` may be a spelling or a different strain), and never
match by position.

**Audit genotype identity before dropping anything.** A label that appears more than once in the
genotype source may be one strain typed twice (identical columns: keep one vector), coisogenic sublines
that share a genotype (keep every phenotype group, give each the shared vector), or two different strains
under one name (different columns: you cannot tell which is which; stop and find out). Different labels
with identical columns are worth knowing too: they are not independent observations of the genotype.

```python
def vector_key(v):
    """Identity of a genotype vector, missing calls included."""
    return hashlib.sha256(np.where(np.isnan(v), -1.0, v).tobytes()).hexdigest()

vectors_of_label = defaultdict(set)
labels_of_vector = defaultdict(set)
for label, v in zip(geno_labels, G_cols):
    vectors_of_label[label].add(vector_key(v))
    labels_of_vector[vector_key(v)].add(label)
counts = Counter(geno_labels)
for label, n_cols in counts.items():
    if n_cols > 1:
        same = len(vectors_of_label[label]) == 1
        print("%s: %d columns, %s" % (label, n_cols, "identical" if same else "DIFFERENT - stop"))
        assert same
shared = [sorted(s) for s in labels_of_vector.values() if len(s) > 1]
print("different labels sharing one vector:", shared)
geno = {label: G_cols[geno_labels.index(label)] for label in counts}   # one vector per label
```

Then map every phenotype strain to a genotype label. Record why each alias holds (in the Farber
replication: an exact join of animal IDs *and* all trait values between the paper supplement and the MPD
export), and list the strains you exclude and why:

```python
alias = {                                           # phenotype-source name -> genotype label
    "BTBR T<+> Itpr3<tf>/J": "BTBR T+ tf/J",        # MPD renamed the strain between exports
    "AXB19a/PgnJ": "AXB-19/PgnJ",                   # coisogenic sublines: one shared genotype
    "AXB19/PgnJ": "AXB-19/PgnJ",
    "AXB19b/PgnJ": "AXB-19/PgnJ",
}
label_of = {s: alias.get(s, s) for s in pheno_strains}
strains = [s for s in pheno_strains if label_of[s] in geno]          # phenotype strains we can use
excluded = {s: "no genotype column" for s in pheno_strains if label_of[s] not in geno}
print(len(strains), "strains kept; excluded:", excluded)
assert excluded == {"BXH20/KccJ": "no genotype column"}
```

Animal IDs need the same care: an ID repeated within a strain is two records you cannot tell apart (fix it
from the source, or drop both); the same ID in two strains is fine when `(strain, ID)` is unique.

```python
assert len(set(animals)) == len(animals), "a (strain, animal) pair occurs twice"
animals = [(s, a) for s, a in animals if s in strains]
si = np.array([strains.index(s) for s, _ in animals])               # animal -> strain row
r_i = np.bincount(si)                                               # replicates per strain
print(len(animals), "animals kept; replicates per strain %d-%d" % (r_i.min(), r_i.max()))
```

## 3. Strain-level genotype matrix: QC and imputation over strains

Do SNP QC and imputation **over strains**, the unit that carries the genotype. Counting a strain's calls
once per animal weights a strain with 11 animals about 5 times as much as one with 2.

```python
G_raw = np.vstack([geno[label_of[s]] for s in strains])             # strains x SNPs, nan = missing
f = np.nanmean(G_raw, axis=0)                                       # allele-1 frequency over strains
miss = np.isnan(G_raw).mean(axis=0)
keep = (np.minimum(f, 1 - f) >= 0.05) & (miss < 0.2)
G_raw = G_raw[:, keep]
snp_alleles = [a for a, k in zip(snp_alleles, keep) if k]
G_imp = np.where(np.isnan(G_raw), np.nanmean(G_raw, axis=0), G_raw)   # mean-impute over STRAINS
K_strains = calculate_kinship(G_imp)
print(G_raw.shape[1], "SNPs pass QC;", int(np.isnan(G_raw).any(axis=0).sum()), "have a missing call")
```

**IBS kinship.** EMMA-era HMDP papers used EMMA's identity-by-state kinship: the share of SNPs at which
two strains carry the same allele (0 / 0.5 / 1 genotypes; missing calls at the SNP's mean over the rows
given, here strains; diagonal 1). It weights every SNP equally, where the standardized K up-weights rare
alleles. Its scale differs a lot: the diagonal is 1, but the off-diagonals of unrelated strains are about
0.5, so its centred scale `c` is well below 1 and its `h` is larger; compare PVE, `h c / (h c + 1 - h)`
(`estimate-heritability` section 3). Test statistics change only through the kinship's shape.

```python
from pylmm.kinship import kinship_scale

K_strains_ibs = calculate_kinship(G_raw, kind="ibs")   # imputes over strains itself (EMMA's rule)
for name, K_ in (("standardized", K_strains), ("ibs", K_strains_ibs)):
    print(name, kinship_scale(K_))
assert kinship_scale(K_strains_ibs).centered_mean_diag < kinship_scale(K_strains).centered_mean_diag
```

## 4. Expanding strains to animals

The individual-level model has one row per animal: `K_animals = Z K_strains Z'` with `Z` the animal ×
strain incidence matrix, and each animal's genotype row is its strain's. Expand genotypes by **row
indexing** (`G[si]`), not `Z @ G`: a matrix product spreads one missing call to every animal.

```python
Z = np.eye(len(strains))[si]
K_animals = Z @ K_strains @ Z.T
W_animals = G_imp[si]                                               # imputed over strains, expanded
n_distinct = len({vector_key(v) for v in G_imp})
print(K_animals.shape, "rank", np.linalg.matrix_rank(K_animals), "from", n_distinct,
      "distinct strain genotypes (K is centred: one less)")

# a phenotype with a polygenic strain effect, one causal SNP and animal-level noise
x = G_imp[:, 50]
y_strain = 0.8 * (x - x.mean()) / x.std() + rng.multivariate_normal(np.zeros(len(strains)), 0.5 * K_strains)
y = 2 + y_strain[si] + rng.normal(scale=0.8, size=len(si))
ts, ps = gwas(y, W_animals, K_animals)
ps = np.asarray(ps)
print("causal SNP p=%.1e, rank %d" % (ps[50], int((ps < ps[50]).sum()) + 1))
```

## 5. The imputation unit: strains vs animals

`PlinkReader` mean-imputes a missing call over the **animals** of the file when `norm_genotype=True`
(the default, and what `pylmm gwas` uses on a per-animal BED); with `norm_genotype=False` it leaves
the missing calls as `nan`. In an inbred panel the strain is the unit, so the strain-level choice is to
impute over strains (section 3) and then expand. The two differ only on SNPs with a missing call, and only
when replicates are unequal:

```python
W_by_animal = G_raw[si]                                             # expanded with the missing calls
W_by_animal = np.where(np.isnan(W_by_animal), np.nanmean(W_by_animal, axis=0), W_by_animal)
_, ps_animal = gwas(y, W_by_animal, K_animals)
d = np.abs(np.log10(ps) - np.log10(ps_animal))
has_missing = np.isnan(G_raw).any(axis=0)
print("max |d -log10 p|: %.2g on complete SNPs, %.2g on SNPs with a missing call"
      % (d[~has_missing].max(), d[has_missing].max()))
assert d[~has_missing].max() < 1e-8 and d[has_missing].max() > 0
```

In the Bennett 2010 HDL replication (95 strains, 2–17 animals each) the difference reached **0.28 in
−log10 p** on SNPs with missing calls; everywhere else the CLI matched the library to 8e-13. A third
choice, `gwas` on `G_raw[si]` with the `nan`s left in, tests each SNP on the called animals only.

**Recipe until the planned strain-aware option exists**: impute over strains in Python (section 3) and
scan with the library (section 4), or use the EMMA strain-mean path (section 8), which takes imputed
values. A PLINK BED holds hard calls only, so a BED written for `pylmm gwas` cannot carry the
strain-imputed values: the CLI on it imputes over animals. That is fine for a cross-check, and you should
say which you report.

## 6. PLINK files: strain names with spaces, markers without allele letters

`pylmm gwas` reads whitespace-separated files, so a strain name with a space (`BTBR T+ tf/J`) cannot be
an FID. Map names to tokens with `plink_tools.sanitize_ids`, which refuses two names that collide, and
write the map next to the files so results can be translated back. `write_bed` and friends reject IDs with
whitespace and point at this recipe.

Markers without allele letters (10,043 rows of the HMDP map have an unknown base): pass `None` for that
marker and `write_bed` writes the placeholders `1` (A1) and `2` (A2). The counted allele is **A2**, the one
whose copies you pass: here the allele coded 1 in the strain matrix. (Do not use `0` as a placeholder:
PLINK reads allele `0` as missing.)

```python
token = plink_tools.sanitize_ids(strains)                          # strain name -> FID token
ids = [(token[s], a) for s, a in animals]                           # FID = strain token, IID = animal
with open("strain_names.tsv", "w") as f:
    f.write("fid\tstrain\tgenotype_label\n")
    f.writelines("%s\t%s\t%s\n" % (token[s], s, label_of[s]) for s in strains)

plink_tools.write_bed("panel", 2 * G_raw[si], fids=[f for f, _ in ids], iids=[i for _, i in ids],
                      a1=[a for a, _ in snp_alleles], a2=[b for _, b in snp_alleles])
plink_tools.write_pheno("panel.pheno", ids, y, ["trait"])
plink_tools.write_kinship("panel.kin", K_animals, ids)              # panel.kin + panel.kin.id
bim = plink_tools.read_bim("panel")
print(token["BTBR T<+> Itpr3<tf>/J"], "|", bim["a1"][0], bim["a2"][0], "|", bim["a1"][1], bim["a2"][1])
```

```bash
pylmm gwas --bfile panel --kinship panel.kin --pheno panel.pheno -o panel.tsv.gz -v 2>&1 | grep "null model"
```

The CLI imputes over animals, so it matches `ps_animal` (section 5), not `ps`:

```python
from pylmm.sumstats import read_sumstats

cli = read_sumstats("panel.tsv.gz")
assert np.allclose(cli["neg_log_10_p_value"], -np.log10(ps_animal), atol=1e-6)
# effect_allele is A1, the allele coded 0 here (write_bed was given copies of A2, the allele coded 1)
assert list(cli["effect_allele"][:2]) == [bim["a1"][0], bim["a1"][1]]
```

## 7. Unequal replicates and heritability

With `r_i` animals in strain `i`, the strain means have residual variances `Ve / r_i`; pylmm's model has
one residual variance, so fit the **individual level** (one row per animal, as above) when replicates are
unequal, and use the harmonic mean `r_h` of the `r_i` to convert between individual-level h² and the
strain-mean (entry-mean) h². The formulas, and what to expect in small panels, are in
`estimate-heritability` section 7.

```python
h = LMM(y, K_animals).fit()[0]
r_h = len(r_i) / np.sum(1.0 / r_i)
print("individual-level h2 = %.2f; harmonic mean r = %.1f; strain-mean h2 = %.2f"
      % (h, r_h, r_h * h / (r_h * h + 1 - h)))
```

Report which one you give, and its interval (`estimate-heritability` section 5). In the Bennett HDL
replication: individual-level SNP h² 0.80 [0.75, 0.85], strain-mean h² 0.87 [0.57, 0.98] from 95 means,
r_h = 6.0.

## 8. The EMMA strain-mean path

`pylmm gwas --emma-geno` reads a SNPs × strains matrix (any numbers, so imputed values are fine), a
1 × strains phenotype row and a strains × strains kinship, **all matched by column position**. EMMA input
has no IDs: no `.id` file is needed, none is warned about, and a `.id` that exists is ignored with a
warning. So keep the strain order yourself, and remember that strain means of unequal `r_i` violate the
one-residual-variance model (section 7).

```python
y_means = np.array([y[si == i].mean() for i in range(len(strains))])
np.savetxt("strains.geno", G_imp.T, fmt="%.6g")                    # SNPs x strains
np.savetxt("strains.pheno", y_means[None, :], fmt="%.6g")
np.savetxt("strains.kin", K_strains)
with open("strains.order.txt", "w") as f:
    f.write("\n".join(strains) + "\n")                              # the order every file is in
```

```bash
pylmm gwas --emma-geno strains.geno --emma-pheno strains.pheno --kinship strains.kin -o strains.tsv.gz -v 2>&1 | grep "null model"
```

The output names SNPs `SNP_1, SNP_2, ...` in genotype row order, with `NA` chromosome, position and
alleles: pass a marker map with one line per genotype row (`--emma-map`, header `chr`, `bp` or
`bp_<build>`, `allele0`, `allele1`; for the bundled HMDP matrix `data/hmdp.liver.snps.map.tsv`, before any
QC subsetting) and they are filled in, the effect allele being the allele coded 1 (`--emma-coding count`
if the values are 0/1/2 counts rather than the fraction 0/0.5/1).

## 9. On the bundled HMDP matrix

The audit of section 2 on the real data (2000 rows to keep it quick): the three `AXB-19/PgnJ` columns are
identical, and the marker map gives positions and the base coded 1.

```python
names = [s.strip().strip('"') for s in open("data/hmdp.liver.strains")]
H = np.genfromtxt("data/hmdp.liver.snps", max_rows=2000, missing_values="NA", filling_values=np.nan).T
cols = [i for i, s in enumerate(names) if s == "AXB-19/PgnJ"]
assert len(cols) == 3 and len({vector_key(H[i]) for i in cols}) == 1
marker_map = np.genfromtxt("data/hmdp.liver.snps.map.tsv", names=True, dtype=None, encoding="utf-8",
                           max_rows=2000)
print("columns", [c + 1 for c in cols], "identical;", "row 1: chr%s:%d, 0=%s 1=%s (%s)"
      % tuple(marker_map[["chr", "bp_b37", "allele0", "allele1", "status"]][0]))
```

## 10. Worked example: Farber 2011 and Bennett 2010 (sketch)

Both reproduced published HMDP scans from their phenotypes with the bundled genotypes
([docs/replication.md](../../docs/replication.md) has the numbers). The preparation, in the order above:

1. **Phenotypes.** Farber 2011 BMD: the paper's Table S1 (878 animals, 97 strains); the later MPD export
   (`HMDPpheno1`, 877 records) joined to it by animal ID *and* all three trait values supplied the strain
   aliases and four explicit animal-ID repairs. Bennett 2010 HDL: MPD `HMDPpheno2` (706 males, 99 strains;
   its two animal files disagreed only on the BTBR name). Units, sex and diet checked against the paper.
2. **Strain names.** Exact matches plus a written alias table. The three identical `AXB-19/PgnJ` columns
   are the sublines AXB19a/19/19b (Farber's AXB-18/19/20): all three phenotype groups kept, one shared
   vector. Strains with no verified genotypes (the BXH KccJ strains, BXD22/TyJ) excluded and listed. The
   first pass dropped the three AXB columns as "ambiguous" before checking identity; the audit showed
   that was unnecessary.
3. **Genotypes.** Positions and alleles from the marker map (build 37). SJL/J, absent from the bundled
   matrix, added from the Broad file in the validated row order. Autosomes only, MAF ≥ 0.05 and missing
   < 0.2 over strains (105,130 and 105,249 SNPs), mean-imputed over strains.
4. **Kinship.** Both the standardized `calculate_kinship` K and an EMMA-style IBS K (the worksheet's
   hand-written version, `interpret-gwas-results/comparison-worksheet.md`; `kind="ibs"` is EMMA's own
   formula, which differs from it only on the diagonal of strains with imputed calls and by a constant
   and a scale, so the same p-values up to that diagonal), expanded to animals with `Z`.
5. **Scan.** Individual level, REML; null-fit scans and a per-SNP refit (`fit(X=x)` then
   `association(x)`), the latter closest to EMMA. A per-animal BED and `pylmm gwas` as the CLI
   cross-check (section 6), with FIDs from `sanitize_ids` and placeholder alleles.
6. **Compare** against pre-registered intervals (published peak ± 1.3 Mb, build 37) with the worksheet.
