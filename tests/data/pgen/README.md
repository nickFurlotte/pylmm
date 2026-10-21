# PGEN fixtures (plink2-written)

Small PLINK 2 file sets for `tests/test_pgen.py`, written by **PLINK v2.0.0-a.7.13 M1
(15 Oct 2026)** (macOS arm64), so that pylmm's PGEN reader is checked against plink2's own
conversions, not against files pylmm wrote itself.

The inputs come from `make_inputs.py` (seeded, numpy only; its docstring describes them):
`hc.bed/.bim/.fam` (53 samples x 40 SNPs, missing calls, an all-missing and two monomorphic
SNPs, SNVs and indels), `swap.txt` (8 SNPs whose REF to flip), `dos.vcf` (21 samples x 12
variants with GT, DS and GP; IIDs only) and `multi.vcf` (one multiallelic variant).

The commands, run in this directory with plink2 on the PATH:

```text
python make_inputs.py
plink2 --bfile hc --make-pgen --out hc                                # BED -> PGEN (REF = A2)
plink2 --bfile hc --make-pgen vzs --out hczst                         # the same, .pvar.zst
plink2 --bfile hc --ref-allele force swap.txt 2 1 --make-pgen --out swap   # 8 REFs flipped
plink2 --pfile swap --make-bed --out swap                             # PGEN -> BED (A2 = REF)
plink2 --vcf dos.vcf dosage=DS --make-pgen --out dos                  # imputed dosages
plink2 --pfile dos --make-bed --out dos                               # its hard calls (threshold 0.1)
plink2 --vcf multi.vcf --make-pgen --out multi                        # keeps the multiallelic variant
rm -f *.log
```

The outputs are deterministic: rerunning the commands gave byte-identical files.

What they are used for:

| files | checks |
|---|---|
| `hc.pgen` vs `hc.bed` | a BED converted to PGEN reads bit for bit as the BED (raw and standardised); `pylmm gwas` and `pylmm kinship` outputs are byte-identical with `--pfile hc` and `--bfile hc`. `plink2 --pfile hc --make-bed` gives back `hc.bed` byte for byte (checked once by hand, not in the tests) |
| `hczst.*` | the `.pvar.zst` variant (read with pgenlib's parser) |
| `swap.pgen` vs `swap.bed` | a PGEN whose REF is not the original A2 reads as the BED plink2 makes of it: pylmm counts REF; the flipped SNPs read as `1 -` those of `hc` |
| `dos.pgen` vs `dos.vcf`, `dos.bed` | dosages are `1 - DS/2` (to 2^-16); `--hardcalls` reads as plink2's BED of it |
| `multi.*` | a multiallelic variant is an `InputError` |

One difference is expected between `hc.bim` and the variants pylmm reads from `hc.pvar`:
snp12's A1 is PLINK 1's missing-allele code `0` in `hc.bim` (written by hand), and plink2
writes `.` (to the `.pvar` and to any `.bim` it makes).
