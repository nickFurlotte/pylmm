#!/usr/bin/env python3
"""Regenerate the two CLI goldens that Phase 5.2 changed on purpose.

    uv run python tests/golden/generate_phase5_2.py [OUTDIR]   # default: tests/golden

Before Phase 5.2, `pylmmGWAS.py --refit` and `--removeMissingGenotypes`
silently fitted the per-SNP model by ML while the default null fit was REML
(bug 1).  Both now use REML, like the rest of the CLI, so these two files
are recomputed with the current (Python 3) code.  Everything else in
tests/golden/ is still the Phase 1 Python 2.7 output; the py2 versions of
these two files are kept in tests/golden/py2_original/.  See README.md.

Since Phase 5b.2 it runs the same models through the new command line,
`pylmm gwas --refit` / `--remove-missing-genotypes` (byte-identical output);
the golden file names keep the old flag names.
"""

import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import helpers as h  # noqa: E402

FILES = ((["--refit"], "refit"), (["--remove-missing-genotypes"], "removeMissing"))


def main(outdir=HERE):
    work = tempfile.mkdtemp()
    try:
        base = h.write_bed_subset(work, h.BED_GWAS_SMALL_NSNPS)
        common = [
            "gwas",
            "-v",
            "--bfile",
            base,
            "--kinship",
            h.README_KIN,
            "--pheno",
            h.FAKE_PHENOS,
        ]
        for flags, tag in FILES:
            name = "gwas_cli_bed_snps%d_%s.out" % (h.BED_GWAS_SMALL_NSNPS, tag)
            out = os.path.join(work, name)
            h.run_cli([*common, *flags, "-o", out], cwd=work)
            shutil.copy(out, os.path.join(outdir, name))
            print("wrote", os.path.join(outdir, name))
    finally:
        shutil.rmtree(work)


if __name__ == "__main__":
    main(*sys.argv[1:])
