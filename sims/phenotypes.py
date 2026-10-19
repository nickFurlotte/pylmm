"""Phenotype generator for the pylmm model.

    y = X0 alpha + sum_j beta_j g_j + u + e,
    u ~ N(0, sigma_g^2 K),  e ~ N(0, sigma_e^2 I),  h2 = sigma_g^2 / (sigma_g^2 + sigma_e^2)

Variance bookkeeping (``var_total`` defaults to 1):

* the causal SNPs jointly explain ``causal_var * var_total`` (in-sample: each
  beta_j is scaled by the realised variance of g_j, so a single causal SNP
  explains exactly its share; with several SNPs their sample covariances add
  a little noise);
* the rest, ``(1 - causal_var) * var_total``, is split into
  sigma_g^2 = h2 * rest and sigma_e^2 = (1 - h2) * rest.

So ``h2`` is the variance-component parameter an LMM estimates when the
causal SNPs are fitted as fixed effects; when they are not, they inflate the
apparent h2.  The scale of sigma_g^2 is relative to K as given: with pylmm's
``calculateKinship`` the mean diagonal is ~1 for outbred synthetic data but
can differ for inbred data or when monomorphic SNPs shrink the scale.
Covariate effects come on top of var_total.

u is drawn through the eigendecomposition K = V diag(d) V':
u = V (sqrt(max(d, 0)) * z) * sigma_g, which is exact for PSD K and clamps the
tiny negative round-off eigenvalues of rank-deficient kinships to 0.
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Sequence, Union

import numpy as np


@dataclasses.dataclass
class Phenotype:
    y: np.ndarray  # length n, nan = missing
    X0: np.ndarray  # n x q covariates (first column = intercept)
    alpha: np.ndarray  # covariate effects (length q)
    causal_idx: np.ndarray  # columns of the genotype matrix passed in
    beta: np.ndarray  # per causal SNP effect (genotype scale given)
    u: np.ndarray  # polygenic effect
    e: np.ndarray  # noise
    h2: float  # sigma_g^2 / (sigma_g^2 + sigma_e^2)
    sigma_g2: float
    sigma_e2: float
    causal_var: float
    missing: np.ndarray  # boolean mask of y set to nan


def sample_polygenic(
    Kva: np.ndarray, Kve: np.ndarray, sigma_g2: float, rng: np.random.Generator
) -> np.ndarray:
    """One draw of u ~ N(0, sigma_g2 K) from the eigendecomposition of K."""
    d = np.clip(np.asarray(Kva, dtype=np.float64), 0.0, None)
    z = rng.standard_normal(len(d))
    return np.asarray(Kve) @ (np.sqrt(d) * z) * np.sqrt(sigma_g2)


def _effect_variances(
    n_causal: int, causal_var: Union[float, Sequence[float]], effects: str, rng: np.random.Generator
) -> np.ndarray:
    if np.ndim(causal_var) == 1:
        v = np.asarray(causal_var, dtype=np.float64)
        if len(v) != n_causal:
            raise ValueError("causal_var list must have one entry per causal SNP")
        return v
    if n_causal == 0:
        return np.zeros(0)
    if effects == "equal":
        w = np.ones(n_causal)
    elif effects == "normal":
        w = rng.standard_normal(n_causal) ** 2
    else:
        raise ValueError("effects must be 'equal' or 'normal'")
    return float(causal_var) * w / w.sum()


def simulate(
    kin,
    h2: float,
    rng: np.random.Generator,
    G: Optional[np.ndarray] = None,
    causal_idx: Optional[Sequence[int]] = None,
    causal_var: Union[float, Sequence[float]] = 0.0,
    effects: str = "equal",
    X0: Optional[np.ndarray] = None,
    alpha: Optional[Sequence[float]] = None,
    n_covariates: int = 0,
    covariate_effect: float = 0.5,
    mu: float = 0.0,
    var_total: float = 1.0,
    missing_rate: float = 0.0,
) -> Phenotype:
    """Simulate one phenotype vector.

    kin            a :class:`sims.kinship.Kinship` (or any object with Kva, Kve)
    h2             polygenic heritability in [0, 1]
    G, causal_idx  genotype matrix (n x m, complete) and the columns that are
                   causal; ``causal_var`` is their joint variance explained
                   (fraction of var_total) or a per-SNP list; ``effects`` is
                   'equal' or 'normal' (random split).  Signs are random.
    X0, alpha      covariates and their effects.  If X0 is None it is
                   [1, Z] with ``n_covariates`` N(0, 1) columns Z; the default
                   alpha is [mu, covariate_effect, ...].
    missing_rate   fraction of y set to nan (completely at random)
    """
    if not 0.0 <= h2 <= 1.0:
        raise ValueError("h2 must be in [0, 1]")
    n = len(kin.Kva)
    causal_idx = np.asarray([] if causal_idx is None else causal_idx, dtype=np.int64)
    if np.ndim(causal_var) == 0 and not 0.0 <= causal_var < 1.0:
        raise ValueError("causal_var must be in [0, 1)")
    v = _effect_variances(len(causal_idx), causal_var, effects, rng)
    total_causal = float(v.sum())

    if X0 is None:
        Z = rng.standard_normal((n, n_covariates))
        X0 = np.hstack([np.ones((n, 1)), Z])
    X0 = np.asarray(X0, dtype=np.float64)
    if alpha is None:
        alpha = np.r_[mu, np.full(X0.shape[1] - 1, covariate_effect)]
    alpha = np.asarray(alpha, dtype=np.float64)

    y = X0 @ alpha
    beta = np.zeros(len(causal_idx))
    if len(causal_idx):
        if G is None:
            raise ValueError("causal SNPs need the genotype matrix G")
        Gc = np.asarray(G[:, causal_idx], dtype=np.float64)
        if np.isnan(Gc).any():
            raise ValueError("causal genotypes must be complete (add missingness afterwards)")
        sd = Gc.std(axis=0)
        sign = rng.choice([-1.0, 1.0], size=len(causal_idx))
        with np.errstate(divide="ignore", invalid="ignore"):
            beta = np.where(sd > 0, sign * np.sqrt(v * var_total) / sd, 0.0)
        y = y + Gc @ beta

    rest = (1.0 - total_causal) * var_total
    sigma_g2, sigma_e2 = h2 * rest, (1.0 - h2) * rest
    u = sample_polygenic(kin.Kva, kin.Kve, sigma_g2, rng)
    e = rng.standard_normal(n) * np.sqrt(sigma_e2)
    y = y + u + e

    missing = np.zeros(n, dtype=bool)
    if missing_rate > 0:
        missing = rng.random(n) < missing_rate
        y = y.copy()
        y[missing] = np.nan
    return Phenotype(
        y=y,
        X0=X0,
        alpha=alpha,
        causal_idx=causal_idx,
        beta=beta,
        u=u,
        e=e,
        h2=float(h2),
        sigma_g2=sigma_g2,
        sigma_e2=sigma_e2,
        causal_var=total_causal,
        missing=missing,
    )


def choose_causal(
    m: int,
    n_causal: int,
    rng: np.random.Generator,
    G: Optional[np.ndarray] = None,
    min_maf: float = 0.0,
) -> np.ndarray:
    """Random sorted causal columns; with G, only polymorphic SNPs whose
    sample MAF is >= min_maf (genotypes assumed 0/1/2 or 0/0.5/1)."""
    pool = np.arange(m)
    if G is not None:
        Gm = np.asarray(G, dtype=np.float64)
        top = np.nanmax(Gm)
        freq = np.nanmean(Gm, axis=0) / (top if top > 0 else 1.0)
        maf = np.minimum(freq, 1 - freq)
        pool = pool[(np.nanstd(Gm, axis=0) > 0) & (maf >= min_maf)]
    if n_causal > len(pool):
        raise ValueError("only %d eligible causal SNPs" % len(pool))
    return np.sort(rng.choice(pool, size=n_causal, replace=False))
