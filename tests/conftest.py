import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import helpers  # noqa: E402

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
    """(Y, K, Kva, Kve) for the README BED data (1219 individuals)."""
    from scipy import linalg
    from pylmm import input as pinput

    IN = pinput.plink(helpers.BED_BASE, type="b", phenoFile=helpers.FAKE_PHENOS)
    n = len(IN.indivs)
    K = helpers.load_readme_kinship(n)
    Y = IN.phenos[:, 0]
    Kva, Kve = linalg.eigh(K)
    return Y, K, Kva, Kve
