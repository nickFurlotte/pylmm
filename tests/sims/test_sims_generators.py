"""Genotype / kinship / phenotype generators in sims/."""

import numpy as np
import pytest

from sims import genotypes, kinship, phenotypes


# ---------------------------------------------------------------------------
# genotypes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spec", [
    {"source": "unrelated", "n": 50, "m": 40, "seed": 3},
    {"source": "balding_nichols", "n": 50, "m": 40, "k": 3, "fst": 0.1, "seed": 3},
    {"source": "balding_nichols", "n": 50, "m": 40, "k": 3, "fst": 0.1, "admixture": 0.5,
     "seed": 3},
    {"source": "family", "n_families": 10, "n_sibs": 5, "m": 40, "seed": 3},
])
def test_synthetic_shapes_and_seeds(spec):
    a = genotypes.make(spec)
    b = genotypes.make(spec)
    c = genotypes.make(dict(spec, seed=4))
    assert a.G.shape == (50, 40)
    assert len(a.snp_ids) == 40 and len(a.indiv_ids) == 50
    assert set(np.unique(a.G)) <= {0.0, 1.0, 2.0}
    np.testing.assert_array_equal(a.G, b.G)
    assert not np.array_equal(a.G, c.G)


def test_family_with_parents_shape():
    g = genotypes.family(4, 3, 20, include_parents=True, seed=1)
    assert g.G.shape == (4 * 2 + 4 * 3, 20)
    assert g.meta["role"].count("parent") == 8


def test_unrelated_maf():
    g = genotypes.unrelated(4000, 200, maf=0.2, seed=0)
    freq = g.G.mean(axis=0) / 2
    assert abs(freq.mean() - 0.2) < 0.005
    lo, hi = 0.1, 0.3
    g = genotypes.unrelated(10, 5000, maf=["uniform", lo, hi], seed=0)
    assert lo <= g.meta["maf"].min() and g.meta["maf"].max() <= hi


def test_balding_nichols_fst():
    """Population frequencies spread with variance ~ F p (1 - p)."""
    fst = 0.1
    g = genotypes.balding_nichols(20, 20000, k=4, fst=fst, maf=0.3, seed=0)
    P = g.meta["pop_freqs"]
    est = P.var(axis=0, ddof=1).mean() / (0.3 * 0.7)
    assert abs(est - fst) < 0.01
    assert list(np.bincount(g.pops)) == [5, 5, 5, 5]


def test_family_kinship_structure():
    g = genotypes.family(30, 4, 5000, seed=0)
    K = kinship.kinship(g.G)
    same = g.pops[:, None] == g.pops[None, :]
    off = ~np.eye(len(g.pops), dtype=bool)
    sib = K[same & off].mean()
    unrel = K[~same].mean()
    # GRM relatedness of full sibs ~0.5 (minus a small sample-mean shrinkage)
    assert 0.4 < sib < 0.55, sib
    assert abs(unrel) < 0.05, unrel


def test_with_missing_rate():
    G = np.zeros((200, 200))
    Gm = genotypes.with_missing(G, 0.1, np.random.default_rng(0))
    assert abs(np.isnan(Gm).mean() - 0.1) < 0.01
    assert not np.isnan(G).any()


def test_get_is_cached_and_readonly():
    spec = {"source": "unrelated", "n": 20, "m": 10, "seed": 9}
    a = genotypes.get(spec)
    assert genotypes.get(dict(spec)) is a
    with pytest.raises(ValueError):
        a.G[0, 0] = 1.0
    b = a.copy()
    b.G[0, 0] = 5.0           # copies are writable


def test_subset():
    g = genotypes.unrelated(10, 8, seed=0)
    s = g.subset(snps=[1, 3], indivs=slice(0, 4))
    np.testing.assert_array_equal(s.G, g.G[:4][:, [1, 3]])
    assert s.snp_ids == ["snp_1", "snp_3"]


def test_bed_subset_matches_pylmm_reader():
    import helpers
    g = genotypes.bed(n_snps=5, n_indivs=30)
    ref = helpers.read_bed_snps(5, norm=False)[:30]
    np.testing.assert_array_equal(np.isnan(g.G), np.isnan(ref))
    np.testing.assert_array_equal(np.nan_to_num(g.G), np.nan_to_num(ref))
    r = genotypes.bed(snps=[2, 4], indivs=[0, 5, 7])
    np.testing.assert_array_equal(np.nan_to_num(r.G), np.nan_to_num(ref[[0, 5, 7]][:, [2, 4]]))
    a = genotypes.bed(n_snps=4, select="random", seed=1)
    assert a.G.shape == (1219, 4)


def test_hmdp_subset():
    import helpers
    g = genotypes.hmdp(n_snps=20)
    ref = helpers.load_hmdp_snps(20)
    np.testing.assert_array_equal(np.nan_to_num(g.G, nan=-1), np.nan_to_num(ref, nan=-1))
    e = genotypes.hmdp(snps=[3, 10], n_indivs=7, select="even")
    assert e.G.shape == (7, 2)


# ---------------------------------------------------------------------------
# kinship
# ---------------------------------------------------------------------------

def test_kinship_matches_pylmm_and_does_not_mutate():
    from pylmm.lmm import calculateKinship
    g = genotypes.with_missing(genotypes.unrelated(30, 50, seed=1).G, 0.05,
                               np.random.default_rng(0))
    before = g.copy()
    K = kinship.kinship(g)
    np.testing.assert_array_equal(np.isnan(g), np.isnan(before))
    np.testing.assert_allclose(K, calculateKinship(before.copy()))


def test_kinship_get_cache_and_split():
    spec = {"source": "unrelated", "n": 30, "m": 60, "seed": 2}
    k1 = kinship.get(spec, snps=slice(0, 40))
    assert kinship.get(spec, snps=slice(0, 40)) is k1
    assert kinship.get(spec) is not k1
    np.testing.assert_allclose(k1.K, kinship.kinship(genotypes.get(spec).G[:, :40]))
    Kva, Kve = k1.eig_for_pylmm()
    Kva[0] = 123.0                         # pylmm clamps in place: cache unaffected
    assert k1.Kva[0] != 123.0
    kin, test = kinship.split_snps(100, 60, 30, np.random.default_rng(0))
    assert len(kin) == 60 and len(test) == 30 and not set(kin) & set(test)
    kin, test = kinship.split_snps(100, 60, 30, overlap=True)
    assert set(test) <= set(kin)


# ---------------------------------------------------------------------------
# phenotypes
# ---------------------------------------------------------------------------

def _kin(n=60, m=400, seed=0):
    return kinship.from_matrix(kinship.kinship(genotypes.unrelated(n, m, seed=seed).G))


def test_phenotype_shapes_and_seed():
    kin = _kin()
    a = phenotypes.simulate(kin, 0.5, np.random.default_rng(1), n_covariates=2,
                            missing_rate=0.2)
    b = phenotypes.simulate(kin, 0.5, np.random.default_rng(1), n_covariates=2,
                            missing_rate=0.2)
    np.testing.assert_array_equal(a.y, b.y)
    assert a.y.shape == (60,) and a.X0.shape == (60, 3)
    np.testing.assert_array_equal(np.isnan(a.y), a.missing)
    assert 0 < a.missing.sum() < 30
    np.testing.assert_allclose(a.sigma_g2 + a.sigma_e2, 1.0)


def test_single_causal_snp_explains_exact_share():
    g = genotypes.unrelated(500, 20, seed=0)
    kin = kinship.from_matrix(np.eye(500))
    ph = phenotypes.simulate(kin, 0.0, np.random.default_rng(0), G=g.G,
                             causal_idx=[3], causal_var=0.1)
    contrib = g.G[:, 3] * ph.beta[0]
    assert np.isclose(contrib.var(), 0.1)
    np.testing.assert_allclose(ph.y, contrib + ph.u + ph.e)
    np.testing.assert_allclose(ph.sigma_e2, 0.9)


def test_effect_split_normal():
    g = genotypes.unrelated(300, 50, seed=0)
    kin = kinship.from_matrix(np.eye(300))
    ph = phenotypes.simulate(kin, 0.3, np.random.default_rng(2), G=g.G,
                             causal_idx=[1, 2, 3, 4], causal_var=0.2, effects="normal")
    v = (ph.beta * g.G[:, [1, 2, 3, 4]].std(axis=0)) ** 2
    np.testing.assert_allclose(v.sum(), 0.2)
    assert np.isclose(ph.causal_var, 0.2)


def test_polygenic_covariance_matches_K():
    """Empirical cov of many u draws ~ sigma_g2 K, incl. a rank-deficient K."""
    g = genotypes.unrelated(40, 10, seed=0)       # m < n: K is singular
    kin = kinship.from_matrix(kinship.kinship(g.G))
    assert kin.Kva.min() < 1e-8
    rng = np.random.default_rng(0)
    U = np.array([phenotypes.sample_polygenic(kin.Kva, kin.Kve, 0.7, rng)
                  for _ in range(20000)])
    C = U.T @ U / len(U)
    err = np.linalg.norm(C - 0.7 * kin.K) / np.linalg.norm(0.7 * kin.K)
    assert err < 0.03, err


def test_choose_causal_skips_monomorphic():
    G = np.zeros((20, 6))
    G[:10, [1, 4]] = 1
    idx = phenotypes.choose_causal(6, 2, np.random.default_rng(0), G=G)
    assert list(idx) == [1, 4]


def test_empirical_h2_recovery_large_n():
    """Mean pylmm REML h over replicates ~ h2 (n=1500, unrelated; ~1 s)."""
    from sims import methods
    spec = {"source": "unrelated", "n": 1500, "m": 3000, "seed": 0}
    kin = kinship.get(spec)
    rng = np.random.default_rng(0)
    hs = [methods.pylmm_null(phenotypes.simulate(kin, 0.5, rng).y, kin)["h"]
          for _ in range(20)]
    assert abs(np.mean(hs) - 0.5) < 0.06, hs
