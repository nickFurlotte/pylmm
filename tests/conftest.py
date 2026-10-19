import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import helpers

if helpers.REPO not in sys.path:
    sys.path.insert(0, helpers.REPO)


def pytest_addoption(parser):
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="also run tests marked @pytest.mark.slow",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="slow; use --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


def pytest_terminal_summary(terminalreporter):
    """Report the snippets run and skipped by tests/test_docs.py, if it ran."""
    mod = sys.modules.get("test_docs")
    stats = getattr(mod, "STATS", None)
    if not stats:
        return
    tr = terminalreporter
    executed = sum(s["executed"] for s in stats.values())
    skipped = [x for s in stats.values() for x in s["skipped"]]
    tr.section("docs snippets (tests/test_docs.py)")
    tr.write_line(
        "%d executed, %d skipped, in %d files (%.1f s)"
        % (executed, len(skipped), len(stats), sum(s["seconds"] for s in stats.values()))
    )
    for where, reason in skipped:
        tr.write_line("  skipped %s: %s" % (where, reason))


@pytest.fixture(scope="session")
def hmdp():
    """(Y, K, snps) for the hmdp data, all from text files.

    snps is individuals x first 1000 SNPs with nan for missing.  Tests that
    mutate must copy.
    """
    Y = np.genfromtxt(helpers.HMDP_Y)
    K = np.loadtxt(helpers.HMDP_K_TXT)
    snps = helpers.load_hmdp_snps()
    return Y, K, snps


@pytest.fixture(scope="session")
def bed_null():
    """(Y, K, kva, kve) for the README BED data (1219 individuals)."""
    from scipy import linalg

    from pylmm import plink_io

    reader = plink_io.PlinkReader(helpers.BED_BASE, type="b", pheno_file=helpers.FAKE_PHENOS)
    n = len(reader.indivs)
    K = helpers.load_readme_kinship(n)
    Y = reader.phenos[:, 0]
    kva, kve = linalg.eigh(K)
    return Y, K, kva, kve
