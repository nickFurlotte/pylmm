# plink2-written kinship fixtures

Small files written by **PLINK v2.0.0-a.7.13 M1 (15 Oct 2026)** (macOS arm64), so that
`tests/test_kinship_io.py` checks pylmm's GCTA GRM and `.rel` readers against an
external tool, not against files pylmm wrote itself.

`geno.bed/.bim/.fam`: 50 samples x 300 SNPs from `tests/synthetic_bed.py`
(`write_synthetic_bed(prefix, 50, 300, seed=3)`; 2% missing calls). 50 samples is the
smallest set plink2 computes allele frequencies from without `--read-freq`.

The commands (run in this directory, with plink2 on the PATH):

```text
plink2 --bfile geno --make-grm-bin --out geno                 # geno.grm.bin / .grm.N.bin / .grm.id
plink2 --bfile geno --make-rel --out geno_tri                 # text, lower triangle (the default)
plink2 --bfile geno --make-rel bin4 square --out geno_sq4     # float32 square .rel.bin
# the same samples without an FID column (.psam header "#IID"):
plink2 --bfile geno --make-pgen --out tmp
(echo "#IID"; tail -n +2 tmp.psam | cut -f2) > tmpi.psam
cp tmp.pgen tmpi.pgen; cp tmp.pvar tmpi.pvar
plink2 --pfile tmpi --make-rel triangle bin --out geno_iid    # float64 triangle; .rel.id "#IID"
```

The `.log` and temporary files were deleted. These are plink2's GRMs (frequencies from
these 50 samples, missingness-corrected), not pylmm's: they are not positive
semi-definite and differ from `pylmm kinship`'s K; the tests compare the file layouts
and IDs, not the kinship definitions.
