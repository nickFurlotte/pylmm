"""A seeded synthetic PLINK BED/BIM/FAM writer for the tests (no dependency on ``sims``).

``sims/bench/bed_blocks.py`` keeps its own copy (the benchmark cannot import
from ``tests/``); ``tests/sims/test_bench_bed_blocks.py`` checks that the two
write the same files.
"""

import numpy as np


def write_synthetic_bed(prefix, n, m, missing=0.02, seed=0):
    """Write a random SNP-major BED/BIM/FAM set; return its raw genotypes (fraction of A2).

    Allele frequencies are uniform on (0.05, 0.5), genotypes Hardy-Weinberg,
    each call missing with probability `missing`.  The ``.bim`` spreads the
    SNPs over three chromosomes with increasing positions.  The returned
    ``(n, m)`` array holds 0, 0.5, 1 or nan, as ``PlinkReader(...,
    norm_genotype=False)`` reads them.
    """
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.05, 0.5, m)
    counts = rng.binomial(2, p, size=(n, m))  # copies of A2
    miss = rng.random((n, m)) < missing
    # 2-bit codes: 0 copies of A2 -> 00, 1 -> 10, 2 -> 11, missing -> 01
    codes = np.array([0, 2, 3], dtype=np.uint8)[counts]
    codes[miss] = 1
    nb = (n + 3) // 4
    padded = np.zeros((nb * 4, m), dtype=np.uint8)
    padded[:n] = codes
    q = padded.reshape(nb, 4, m)
    packed = q[:, 0] | (q[:, 1] << 2) | (q[:, 2] << 4) | (q[:, 3] << 6)
    with open(prefix + ".bed", "wb") as f:
        f.write(b"\x6c\x1b\x01")
        f.write(np.ascontiguousarray(packed.T).tobytes())
    chrom = np.repeat(["1", "2", "3"], [m - 2 * (m // 3), m // 3, m // 3])
    with open(prefix + ".bim", "w") as f:
        for j in range(m):
            f.write("%s\tsnp%d\t%.3f\t%d\tA\tG\n" % (chrom[j], j + 1, j / 1000, 1000 * (j + 1)))
    with open(prefix + ".fam", "w") as f:
        for i in range(n):
            f.write("F%d I%d 0 0 0 -9\n" % (i + 1, i + 1))
    G = counts / 2.0
    G[miss] = np.nan
    return G
