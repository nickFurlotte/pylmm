"""Write the plink2 inputs of the PGEN test fixtures (see README.md in this directory).

Run from this directory, ``python make_inputs.py``, then the plink2 commands of
README.md.  Deterministic (seeded); needs numpy only.

* ``hc.bed/.bim/.fam``: 53 samples (not a multiple of 4, so BED rows are padded) in
  families of 1-3, 40 SNPs on chromosomes 1, 2 and 22 with assorted alleles (SNVs and
  indels), 6% missing calls, one all-missing SNP (snp5), one monomorphic SNP (snp9) and
  one monomorphic SNP whose A1 is PLINK 1's missing-allele code ``0`` (snp12).
* ``swap.txt``: 8 SNP IDs with their BED A1 allele, for ``plink2 --ref-allele``: in
  ``swap.pgen`` those SNPs have REF = the BED's A1, the other allele than in ``hc.pgen``.
* ``dos.vcf``: 21 samples (IIDs only), 12 biallelic variants with GT, DS and GP; DS has
  free values (3 decimals), the exact values 0, 0.5, 1, 1.5 and 2, and missing values.
* ``multi.vcf``: 10 samples, 3 variants, the second multiallelic (ALT ``C,T``).
"""

import numpy as np

ALLELES = [("A", "G"), ("C", "T"), ("G", "A"), ("T", "C"), ("AT", "A"), ("G", "GTTA")]
SWAPPED = (0, 3, 4, 8, 13, 21, 30, 39)  # 0-based SNPs whose REF swap.pgen flips


def write_bed(prefix, rng):
    n, m = 53, 40
    p = rng.uniform(0.1, 0.5, m)
    counts = rng.binomial(2, p, size=(n, m))  # copies of A2
    miss = rng.random((n, m)) < 0.06
    counts[:, 8] = 2  # snp9: monomorphic A2/A2
    counts[:, 11] = 2  # snp12: monomorphic, A1 = "0"
    miss[:, 8] = miss[:, 11] = False
    miss[:, 4] = True  # snp5: no called genotype
    codes = np.array([0, 2, 3], dtype=np.uint8)[counts]  # 00 hom A1, 10 het, 11 hom A2
    codes[miss] = 1  # 01 missing
    nb = (n + 3) // 4
    padded = np.zeros((nb * 4, m), dtype=np.uint8)
    padded[:n] = codes
    q = padded.reshape(nb, 4, m)
    packed = q[:, 0] | (q[:, 1] << 2) | (q[:, 2] << 4) | (q[:, 3] << 6)
    with open(prefix + ".bed", "wb") as f:
        f.write(b"\x6c\x1b\x01")
        f.write(np.ascontiguousarray(packed.T).tobytes())
    chrom = ["1"] * 20 + ["2"] * 15 + ["22"] * 5
    with open(prefix + ".bim", "w") as f:
        for j in range(m):
            a1, a2 = ALLELES[j % len(ALLELES)]
            if j == 11:
                a1 = "0"
            f.write(
                "%s\tsnp%d\t%.4f\t%d\t%s\t%s\n"
                % (chrom[j], j + 1, 0.01 * (j + 1), 1000 + 37 * j, a1, a2)
            )
    with open(prefix + ".fam", "w") as f:
        i = 0
        fam = 0
        while i < n:
            fam += 1
            for _ in range(min(1 + fam % 3, n - i)):
                f.write("fam%d\tind%d\t0\t0\t%d\t-9\n" % (fam, i + 1, 1 + i % 2))
                i += 1
    with open("swap.txt", "w") as f:
        for j in SWAPPED:
            f.write("snp%d\t%s\n" % (j + 1, ALLELES[j % len(ALLELES)][0]))


def write_vcf_header(f, samples):
    f.write("##fileformat=VCFv4.2\n")
    f.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
    f.write('##FORMAT=<ID=DS,Number=A,Type=Float,Description="ALT dosage">\n')
    f.write('##FORMAT=<ID=GP,Number=G,Type=Float,Description="Genotype probabilities">\n')
    for c in ("1", "2"):
        f.write("##contig=<ID=%s>\n" % c)
    f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t%s\n" % "\t".join(samples))


def write_dosage_vcf(fn, rng):
    n, m = 21, 12
    samples = ["s%02d" % (i + 1) for i in range(n)]
    with open(fn, "w") as f:
        write_vcf_header(f, samples)
        for j in range(m):
            ds = np.round(rng.uniform(0, 2, n), 3)
            ds[:5] = [0.0, 0.5, 1.0, 1.5, 2.0]  # exact values
            ds = np.roll(ds, j)
            missing = rng.random(n) < 0.08
            fields = []
            for d, miss in zip(ds, missing):
                if miss:
                    fields.append("./.:.:.")
                    continue
                gt = "0/0" if d < 0.5 else ("0/1" if d < 1.5 else "1/1")
                # GP with the same expected ALT count: P(het) + 2 P(hom ALT) = d
                p2 = max(0.0, d - 1.0)
                p1 = d - 2 * p2
                p0 = 1.0 - p1 - p2
                fields.append("%s:%.3f:%.4f,%.4f,%.4f" % (gt, d, p0, p1, p2))
            ref, alt = ALLELES[j % 4]
            chrom = "1" if j < 8 else "2"
            f.write(
                "%s\t%d\tv%d\t%s\t%s\t.\tPASS\t.\tGT:DS:GP\t%s\n"
                % (chrom, 500 + 100 * j, j + 1, ref, alt, "\t".join(fields))
            )


def write_multi_vcf(fn, rng):
    n = 10
    samples = ["m%d" % (i + 1) for i in range(n)]
    with open(fn, "w") as f:
        write_vcf_header(f, samples)
        for j, (ref, alt) in enumerate([("A", "G"), ("A", "C,T"), ("G", "T")]):
            k = 3 if "," in alt else 2
            gts = ["%d/%d" % tuple(sorted(rng.integers(0, k, 2))) for _ in range(n)]
            f.write(
                "1\t%d\tm%d\t%s\t%s\t.\tPASS\t.\tGT\t%s\n"
                % (100 * (j + 1), j + 1, ref, alt, "\t".join(gts))
            )


if __name__ == "__main__":
    rng = np.random.default_rng(20261021)
    write_bed("hc", rng)
    write_dosage_vcf("dos.vcf", rng)
    write_multi_vcf("multi.vcf", rng)
