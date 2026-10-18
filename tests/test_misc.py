
import numpy as np

import helpers as h
from pylmm import lmm, misc


def test_fit_two(hmdp):
    Y, K, snps = hmdp
    K2 = lmm.calculateKinship(snps.copy())
    h1, h2, e, beta, sigma, LL = misc.fitTwo(Y, K, K2)
    exp = h.load_kv('misc_fitTwo.kv.txt')
    got = {'h1': h1, 'h2': h2, 'e': e, 'sigma': np.asarray(sigma).sum(), 'LL': LL}
    for i, b in enumerate(np.asarray(beta).ravel()):
        got['beta_%d' % i] = b
    assert sorted(got) == sorted(exp)
    for k in sorted(got):
        h.assert_close(got[k], exp[k], rtol=h.RTOL_FIT, atol=h.ATOL_FIT,
                       what='fitTwo ' + k)
    np.testing.assert_allclose(h1 + h2 + e, 1.0, rtol=0, atol=1e-12)
    for v in (h1, h2, e):
        assert 0.0 <= v <= 1.0
