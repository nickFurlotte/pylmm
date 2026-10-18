# S5: edge cases and robustness

_Last run: 2026-10-18T12:32:14, git 30e14d5, jobs=2, wall 324s; total task time over all runs 644s._

One row per case (params), worst outcome over replicates.  outcome: OK, silent (dubious input, no warning), nan (unexpected nan/inf), crash (exception; message in crash_err), wrong (finite but incorrect vs the oracle, or finite where the statistic is not estimable).  severity: critical = silent wrong answer, high = crash/nan on valid input, medium = unclear error/silent on garbage input, low = warnings only.  n_nan_unexp counts SNPs with nan where the statistic is estimable; n_finite_unexp counts SNPs with a finite t where it is not.  max_abs_dh / max_dt compare with the dense oracle (h, and relative t).  Sorted by severity.  See s5_edge_cases.md for the Phase 5 target list.

| family | variant | path | knobs | outcome | severity | frac | crash_err | n_nan_unexp | n_finite_unexp | max_abs_dh | max_dt | runtime_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| missing | mcar | cli_remove | gm=0.01, pm=0.2 | wrong | critical | 1 |  | 0 | 0 | 0.0004986 | 0.04282 | 1.076 |
| missing | mcar | lib | gm=0.2, pm=0.2 | wrong | critical | 0.17 |  | 0 | 0 | 4.572e-08 | 0.002186 | 0.114 |
| smalln | n_k | lib | n=10, k=9 | wrong | critical | 1 |  | 0 | 17 |  |  | 0.025 |
| smalln | n_k | lib | n=10, k=12 | wrong | critical | 1 |  | 0 | 18 |  |  | 0.019 |
| smalln | n_k | lib | n=20, k=19 | wrong | critical | 1 |  | 0 | 16 |  |  | 0.016 |
| smalln | n_k | lib | n=20, k=22 | wrong | critical | 1 |  | 0 | 15 |  |  | 0.018 |
| smalln | n_k | lib | n=50, k=49 | wrong | critical | 1 |  | 0 | 9 |  |  | 0.036 |
| pheno | constant | lib |  | wrong | critical | 0.5 | null fit: BracketError: The algorithm terminated without finding a valid bracket. Consider trying different initial points. | GWAS: BracketError: The algorithm terminated without finding a valid bracket. Consider trying different initial points. | 0 | 0 |  |  | 0.054 |
| pheno | constant | cli_impute |  | wrong | critical | 0.67 | scipy.optimize._optimize.BracketError: The algorithm terminated without finding a valid bracket. Consider trying different initial points. | 0 | 0 |  |  | 1.174 |
| io | kfile_big | cli_impute |  | wrong | critical | 1 |  | 0 | 0 |  |  | 1.491 |
| io | kfile_small | cli_impute |  | wrong | critical | 1 |  | 0 | 0 |  |  | 1.558 |
| missing | mcar | cli_remove | gm=0.01, pm=0 | wrong | high | 0.83 |  | 0 | 0 | 0.0004985 | 0.0505 | 1.064 |
| missing | mcar | cli_remove | gm=0.05, pm=0 | wrong | high | 1 |  | 0 | 0 | 0.0004912 | 0.05918 | 1.106 |
| missing | mcar | cli_remove | gm=0.05, pm=0.2 | wrong | high | 1 |  | 0 | 0 | 0.0003758 | 0.04959 | 1.143 |
| missing | mcar | cli_remove | gm=0.2, pm=0 | wrong | high | 1 |  | 0 | 0 | 0.0004475 | 0.0573 | 1.197 |
| missing | mcar | cli_remove | gm=0.2, pm=0.2 | wrong | high | 1 |  | 0 | 0 | 0.0004192 | 0.09989 | 1.181 |
| missing | mcar | cli_remove | gm=0.5, pm=0 | wrong | high | 1 |  | 0 | 0 | 0.0004661 | 0.07586 | 1.208 |
| missing | mcar | cli_remove | gm=0.5, pm=0.2 | wrong | high | 1 |  | 0 | 0 | 0.0004509 | 0.0627 | 1.036 |
| degenerate | eq_cov_x2 | cli_remove |  | wrong | high | 0.5 | numpy.linalg.LinAlgError: A singular matrix detected: slice(s) [0] are singular. | 0 | 1 | 0.000462 | 3.359e-08 | 1.08 |
| kin | nonpsd | lib | noise=0.3 | wrong | high | 0.33 |  | 0 | 0 |  |  | 0.011 |
| kin | wrong_order | lib |  | wrong | high | 1 |  | 0 | 0 | 3.601e-08 | 6.72e-08 | 0.011 |
| kin | upper_only | lib |  | wrong | high | 1 |  | 0 | 0 |  |  | 0.01 |
| degenerate | allmiss | lib |  | crash | high | 1 | GWAS: LinAlgError: A singular matrix detected: slice(s) [0] are singular. |  |  |  |  | 0.018 |
| degenerate | allmiss | lib_impute |  | crash | high | 1 | GWAS: LinAlgError: A singular matrix detected: slice(s) [0] are singular. |  |  |  |  | 0.012 |
| degenerate | allmiss | cli_impute |  | crash | high | 1 | IndexError: boolean index did not match indexed array along axis 0; size of axis is 0 but size of corresponding boolean axis is 100 |  |  |  |  | 1.192 |
| smalln | n_k | lib | n=20, k=2 | crash | high | 0.17 | null fit: BracketError: The algorithm terminated without finding a valid bracket. Consider trying different initial points. | GWAS: BracketError: The algorithm terminated without finding a valid bracket. Consider trying different initial points. | 0 | 0 | 0.01 | 1.543e-08 | 0.01 |
| degenerate | mono_dosage | lib |  | wrong | medium | 0.67 |  | 0 | 1 | 5.458e-08 | 7.587e-08 | 0.016 |
| degenerate | mono_dosage | lib_impute |  | wrong | medium | 1 |  | 0 | 1 | 4.812e-08 | 1.142e-07 | 0.012 |
| degenerate | mono_ydrop | cli_impute |  | wrong | medium | 0.67 |  | 0 | 1 | 0.0003746 | 1.328e-07 | 1.063 |
| degenerate | single_minor_ydrop | cli_impute |  | wrong | medium | 0.17 |  | 0 | 1 | 0.0003931 | 1.106e-07 | 1.139 |
| degenerate | eq_cov | lib |  | wrong | medium | 0.33 |  | 0 | 1 | 2.93e-08 | 4.689e-08 | 0.013 |
| degenerate | eq_cov | lib_impute |  | wrong | medium | 0.33 |  | 0 | 1 | 4.642e-08 | 5.953e-08 | 0.012 |
| degenerate | eq_cov | cli_impute |  | wrong | medium | 0.83 |  | 0 | 1 | 0.0003755 | 5.557e-08 | 1.057 |
| degenerate | eq_cov | cli_remove |  | wrong | medium | 0.33 |  | 0 | 1 | 0.0004956 | 4.554e-08 | 1.074 |
| degenerate | eq_cov_x2 | lib |  | wrong | medium | 0.5 |  | 0 | 1 | 2.956e-08 | 3.241e-08 | 0.013 |
| degenerate | eq_cov_x2 | lib_impute |  | wrong | medium | 1 |  | 0 | 1 | 2.807e-08 | 6.002e-08 | 0.014 |
| degenerate | eq_cov_x2 | cli_impute |  | wrong | medium | 0.67 |  | 0 | 1 | 0.0004201 | 3.127e-08 | 1.08 |
| smalln | n_k | lib | n=20, k=18 | wrong | medium | 0.17 | null fit: BracketError: The algorithm terminated without finding a valid bracket. Consider trying different initial points. | GWAS: BracketError: The algorithm terminated without finding a valid bracket. Consider trying different initial points. | 0 | 19 |  |  | 0.117 |
| smalln | n_k | lib | n=50, k=48 | wrong | medium | 0.33 | null fit: BracketError: The algorithm terminated without finding a valid bracket. Consider trying different initial points. | GWAS: BracketError: The algorithm terminated without finding a valid bracket. Consider trying different initial points. | 0 | 20 |  |  | 0.263 |
| smalln | n_k | lib | n=50, k=52 | wrong | medium | 1 |  | 0 | 17 |  |  | 0.046 |
| smalln | n_k | lib | n=10, k=8 | crash | medium | 1 | null fit: BracketError: The algorithm terminated without finding a valid bracket. Consider trying different initial points. | GWAS: BracketError: The algorithm terminated without finding a valid bracket. Consider trying different initial points. |  |  |  |  | 0.106 |
| kin | nan_K | lib |  | crash | medium | 1 | null fit: ValueError: array must not contain infs or NaNs | GWAS: ValueError: array must not contain infs or NaNs |  |  |  |  | 0 |
| cov | collinear_dup | lib |  | crash | medium | 1 | null fit: LinAlgError: A singular matrix detected: slice(s) [0] are singular. | GWAS: LinAlgError: A singular matrix detected: slice(s) [0] are singular. |  |  |  |  | 0.004 |
| cov | collinear_dup | cli_impute |  | crash | medium | 1 | numpy.linalg.LinAlgError: A singular matrix detected: slice(s) [0] are singular. |  |  |  |  | 1.21 |
| cov | collinear_comb | lib |  | crash | medium | 1 | null fit: LinAlgError: A singular matrix detected: slice(s) [0] are singular. | GWAS: LinAlgError: A singular matrix detected: slice(s) [0] are singular. |  |  |  |  | 0.004 |
| cov | intercept_dup | lib |  | crash | medium | 1 | null fit: LinAlgError: A singular matrix detected: slice(s) [0] are singular. | GWAS: LinAlgError: A singular matrix detected: slice(s) [0] are singular. |  |  |  |  | 0.002 |
| cov | intercept_dup | cli_impute |  | crash | medium | 1 | numpy.linalg.LinAlgError: A singular matrix detected: slice(s) [0] are singular. |  |  |  |  | 1.068 |
| cov | zero_cov | lib |  | crash | medium | 1 | null fit: LinAlgError: A singular matrix detected: slice(s) [0] are singular. | GWAS: LinAlgError: A singular matrix detected: slice(s) [0] are singular. |  |  |  |  | 0.002 |
| cov | zero_cov | cli_impute |  | crash | medium | 1 | numpy.linalg.LinAlgError: A singular matrix detected: slice(s) [0] are singular. |  |  |  |  | 0.949 |
| cov | cov_nan | lib |  | crash | medium | 1 | null fit: ValueError: array must not contain infs or NaNs | GWAS: ValueError: array must not contain infs or NaNs |  |  |  |  | 0.002 |
| pheno | all_nan | lib |  | crash | medium | 1 | null fit: LinAlgError: A singular matrix detected: slice(s) [0] are singular. |  |  |  |  | 0 |
| pheno | all_nan | cli_impute |  | crash | medium | 1 | numpy.linalg.LinAlgError: A singular matrix detected: slice(s) [0] are singular. |  |  |  |  | 0.965 |
| io | nophenofile | cli_impute |  | crash | medium | 1 | TypeError: stat: path should be string, bytes, os.PathLike or integer, not NoneType |  |  |  |  | 0.913 |
| io | pheno_missing_indiv | cli_impute |  | crash | medium | 1 | IndexError: boolean index did not match indexed array along axis 0; size of axis is 100 but size of corresponding boolean axis is 99 |  |  |  |  | 1.146 |
| io | kfile_header | cli_impute |  | crash | medium | 1 | ValueError: string or file could not be read to its end due to unmatched data |  |  |  |  | 1.507 |
| kin | nonpsd | lib | noise=0.1 | silent | medium | 1 |  | 0 | 0 |  |  | 0.011 |
| kin | asym | lib |  | silent | medium | 1 |  | 0 | 0 |  |  | 0.015 |
| pheno | const_noise | lib | noise=1e-12 | silent | medium | 1 |  | 0 | 0 |  |  | 0.01 |
| cov | cov_scale | lib | scale=1e-08 | OK | low | 1 |  | 0 | 0 | 3.226e-08 | 4.997e-08 | 0.015 |
| cov | cov_scale | lib | scale=1e+08 | OK | low | 1 |  | 0 | 0 | 2.695e-08 | 5.754e-08 | 0.014 |
| missing | mcar | lib | gm=0, pm=0 | OK | none | 1 |  | 0 | 0 | 7.087e-08 | 9.647e-08 | 0.012 |
| missing | mcar | lib_impute | gm=0, pm=0 | OK | none | 1 |  | 0 | 0 | 3.166e-08 | 7.813e-08 | 0.023 |
| missing | mcar | lib | gm=0, pm=0.05 | OK | none | 1 |  | 0 | 0 | 4.55e-08 | 8.317e-08 | 0.018 |
| missing | mcar | lib_impute | gm=0, pm=0.05 | OK | none | 1 |  | 0 | 0 | 3.32e-08 | 3.875e-08 | 0.018 |
| missing | mcar | lib | gm=0, pm=0.2 | OK | none | 1 |  | 0 | 0 | 3.69e-08 | 3.655e-08 | 0.013 |
| missing | mcar | lib_impute | gm=0, pm=0.2 | OK | none | 1 |  | 0 | 0 | 5.843e-08 | 7.454e-08 | 0.011 |
| missing | mcar | cli_impute | gm=0, pm=0 | OK | none | 1 |  | 0 | 0 | 0.0004821 | 7.658e-08 | 1.159 |
| missing | mcar | cli_remove | gm=0, pm=0 | OK | none | 1 |  | 0 | 0 | 0.0003071 | 6.931e-08 | 1.262 |
| missing | mcar | cli_impute | gm=0, pm=0.2 | OK | none | 1 |  | 0 | 0 | 0.0004782 | 3.311e-08 | 1.274 |
| missing | mcar | cli_remove | gm=0, pm=0.2 | OK | none | 1 |  | 0 | 0 | 0.0004512 | 7.589e-08 | 1.054 |
| missing | mcar | lib | gm=0.01, pm=0 | OK | none | 1 |  | 0 | 0 | 3.155e-08 | 9.929e-08 | 0.097 |
| missing | mcar | lib_impute | gm=0.01, pm=0 | OK | none | 1 |  | 0 | 0 | 2.89e-08 | 6.483e-08 | 0.012 |
| missing | mcar | lib | gm=0.01, pm=0.05 | OK | none | 1 |  | 0 | 0 | 3.868e-08 | 5.82e-08 | 0.062 |
| missing | mcar | lib_impute | gm=0.01, pm=0.05 | OK | none | 1 |  | 0 | 0 | 5.848e-08 | 8.278e-08 | 0.014 |
| missing | mcar | lib | gm=0.01, pm=0.2 | OK | none | 1 |  | 0 | 0 | 3.702e-08 | 4.538e-08 | 0.077 |
| missing | mcar | lib_impute | gm=0.01, pm=0.2 | OK | none | 1 |  | 0 | 0 | 4.299e-08 | 6.218e-08 | 0.013 |
| missing | mcar | cli_impute | gm=0.01, pm=0 | OK | none | 1 |  | 0 | 0 | 0.000491 | 6.573e-08 | 1.16 |
| missing | mcar | cli_impute | gm=0.01, pm=0.2 | OK | none | 1 |  | 0 | 0 | 0.0003353 | 7.106e-08 | 1.003 |
| missing | mcar | lib | gm=0.05, pm=0 | OK | none | 1 |  | 0 | 0 | 4.053e-08 | 7.98e-08 | 0.131 |
| missing | mcar | lib_impute | gm=0.05, pm=0 | OK | none | 1 |  | 0 | 0 | 2.568e-08 | 3.679e-08 | 0.013 |
| missing | mcar | lib | gm=0.05, pm=0.05 | OK | none | 1 |  | 0 | 0 | 3.642e-08 | 5.777e-08 | 0.117 |
| missing | mcar | lib_impute | gm=0.05, pm=0.05 | OK | none | 1 |  | 0 | 0 | 3.856e-08 | 4.636e-08 | 0.013 |
| missing | mcar | lib | gm=0.05, pm=0.2 | OK | none | 1 |  | 0 | 0 | 3.529e-08 | 4.566e-08 | 0.102 |
| missing | mcar | lib_impute | gm=0.05, pm=0.2 | OK | none | 1 |  | 0 | 0 | 3.942e-08 | 4.341e-08 | 0.011 |
| missing | mcar | cli_impute | gm=0.05, pm=0 | OK | none | 1 |  | 0 | 0 | 0.0004798 | 7.693e-08 | 1.011 |
| missing | mcar | cli_impute | gm=0.05, pm=0.2 | OK | none | 1 |  | 0 | 0 | 0.0004629 | 3.126e-08 | 1.24 |
| missing | mcar | lib | gm=0.2, pm=0 | OK | none | 1 |  | 0 | 0 | 6.745e-08 | 9.315e-08 | 0.116 |
| missing | mcar | lib_impute | gm=0.2, pm=0 | OK | none | 1 |  | 0 | 0 | 2.55e-08 | 2.926e-08 | 0.012 |
| missing | mcar | lib | gm=0.2, pm=0.05 | OK | none | 1 |  | 0 | 0 | 2.987e-08 | 4.535e-08 | 0.111 |
| missing | mcar | lib_impute | gm=0.2, pm=0.05 | OK | none | 1 |  | 0 | 0 | 3.499e-08 | 3.966e-08 | 0.011 |
| missing | mcar | lib_impute | gm=0.2, pm=0.2 | OK | none | 1 |  | 0 | 0 | 2.879e-08 | 3.411e-08 | 0.015 |
| missing | mcar | cli_impute | gm=0.2, pm=0 | OK | none | 1 |  | 0 | 0 | 0.000493 | 5.286e-08 | 0.979 |
| missing | mcar | cli_impute | gm=0.2, pm=0.2 | OK | none | 1 |  | 0 | 0 | 0.004268 | 0.01239 | 0.996 |
| missing | mcar | lib | gm=0.5, pm=0 | OK | none | 1 |  | 0 | 0 | 3.237e-08 | 2.913e-08 | 0.135 |
| missing | mcar | lib_impute | gm=0.5, pm=0 | OK | none | 1 |  | 0 | 0 | 4.344e-08 | 8.507e-08 | 0.013 |
| missing | mcar | lib | gm=0.5, pm=0.05 | OK | none | 1 |  | 0 | 0 | 1.757e-08 | 5.82e-08 | 0.104 |
| missing | mcar | lib_impute | gm=0.5, pm=0.05 | OK | none | 1 |  | 0 | 0 | 3.166e-08 | 3.33e-08 | 0.011 |
| missing | mcar | lib | gm=0.5, pm=0.2 | OK | none | 1 |  | 0 | 0 | 3.671e-08 | 5.574e-08 | 0.098 |
| missing | mcar | lib_impute | gm=0.5, pm=0.2 | OK | none | 1 |  | 0 | 0 | 5.712e-08 | 7.353e-08 | 0.011 |
| missing | mcar | cli_impute | gm=0.5, pm=0 | OK | none | 1 |  | 0 | 0 | 0.0004818 | 6.806e-08 | 0.965 |
| missing | mcar | cli_impute | gm=0.5, pm=0.2 | OK | none | 1 |  | 0 | 0 | 0.0004755 | 3.601e-08 | 1.001 |
| degenerate | mono | lib |  | OK | none | 1 |  | 0 | 0 | 3.085e-08 | 5.622e-08 | 0.014 |
| degenerate | mono | lib_impute |  | OK | none | 1 |  | 0 | 0 | 5.682e-08 | 1.175e-07 | 0.011 |
| degenerate | mono | cli_impute |  | OK | none | 1 |  | 0 | 0 | 0.0004582 | 4.885e-08 | 1.119 |
| degenerate | mono | cli_remove |  | OK | none | 1 |  | 0 | 0 | 0.0004632 | 1.232e-07 | 1.048 |
| degenerate | mono_ydrop | lib |  | OK | none | 1 |  | 0 | 0 | 5.346e-08 | 7.761e-08 | 0.011 |
| degenerate | mono_ydrop | lib_impute |  | OK | none | 1 |  | 0 | 0 | 4.562e-08 | 5.426e-08 | 0.012 |
| degenerate | mono_ydrop | cli_remove |  | OK | none | 1 |  | 0 | 0 | 0.000395 | 5.335e-08 | 1.102 |
| degenerate | mono_xdrop | lib |  | OK | none | 1 |  | 0 | 0 | 3.445e-08 | 7.145e-08 | 0.012 |
| degenerate | mono_xdrop | lib_impute |  | OK | none | 1 |  | 0 | 0 | 7.57e-08 | 1.289e-07 | 0.012 |
| degenerate | mono_xdrop | cli_impute |  | OK | none | 1 |  | 0 | 0 | 0.0003833 | 2.668e-08 | 1.172 |
| degenerate | mono_xdrop | cli_remove |  | OK | none | 1 |  | 0 | 0 | 0.0004796 | 4.07e-08 | 1.022 |
| degenerate | allmiss | cli_remove |  | OK | none | 1 |  | 0 | 0 | 0.0004876 | 5.824e-08 | 1.045 |
| degenerate | single_minor | lib |  | OK | none | 1 |  | 0 | 0 | 4.561e-08 | 5.059e-08 | 0.012 |
| degenerate | single_minor | lib_impute |  | OK | none | 1 |  | 0 | 0 | 2.85e-08 | 5.428e-08 | 0.019 |
| degenerate | single_minor | cli_impute |  | OK | none | 1 |  | 0 | 0 | 0.0004029 | 5.739e-08 | 1.111 |
| degenerate | single_minor | cli_remove |  | OK | none | 1 |  | 0 | 0 | 0.0003969 | 5.308e-08 | 1.027 |
| degenerate | single_minor_ydrop | lib |  | OK | none | 1 |  | 0 | 0 | 3.679e-08 | 8.804e-08 | 0.011 |
| degenerate | single_minor_ydrop | lib_impute |  | OK | none | 1 |  | 0 | 0 | 4.111e-08 | 4.525e-08 | 0.011 |
| degenerate | single_minor_ydrop | cli_remove |  | OK | none | 1 |  | 0 | 0 | 0.0004171 | 5.865e-08 | 0.997 |
| degenerate | eq_intercept | lib |  | OK | none | 1 |  | 0 | 0 | 4.135e-08 | 1.134e-07 | 0.011 |
| degenerate | eq_intercept | lib_impute |  | OK | none | 1 |  | 0 | 0 | 4.739e-08 | 6.789e-08 | 0.012 |
| degenerate | eq_intercept | cli_impute |  | OK | none | 1 |  | 0 | 0 | 0.0004338 | 8.52e-08 | 1.141 |
| degenerate | eq_intercept | cli_remove |  | OK | none | 1 |  | 0 | 0 | 0.0004497 | 6.067e-08 | 1.045 |
| smalln | n_k | lib | n=10, k=0 | OK | none | 1 |  | 0 | 0 | 0.01 | 2.049e-08 | 0.009 |
| smalln | n_k | lib | n=10, k=2 | OK | none | 1 |  | 0 | 0 | 0.01 | 1.853e-09 | 0.009 |
| smalln | n_k | lib | n=10, k=5 | OK | none | 1 |  | 0 | 0 | 0.01 | 2.706e-08 | 0.009 |
| smalln | n_k | lib | n=10, k=7 | OK | none | 1 |  | 0 | 0 | 0.01 | 1.159e-13 | 0.01 |
| smalln | n_k | lib | n=20, k=0 | OK | none | 1 |  | 0 | 0 | 3.452e-08 | 6.211e-08 | 0.009 |
| smalln | n_k | lib | n=20, k=10 | OK | none | 1 |  | 0 | 0 | 0.01 | 2.249e-08 | 0.011 |
| smalln | n_k | lib | n=20, k=17 | OK | none | 1 |  | 0 | 0 | 0.01 | 9.876e-08 | 0.014 |
| smalln | n_k | lib | n=50, k=0 | OK | none | 1 |  | 0 | 0 | 0.01 | 3.693e-08 | 0.011 |
| smalln | n_k | lib | n=50, k=2 | OK | none | 1 |  | 0 | 0 | 3.056e-08 | 5.507e-08 | 0.013 |
| smalln | n_k | lib | n=50, k=25 | OK | none | 1 |  | 0 | 0 | 0.004568 | 0.006233 | 0.02 |
| smalln | n_k | lib | n=50, k=47 | OK | none | 1 |  | 0 | 0 | 0.01 | 1.027e-11 | 0.028 |
| kin | lowrank | lib | m_kin=2 | OK | none | 1 |  | 0 | 0 | 1.903e-07 | 1.486e-08 | 0.011 |
| kin | lowrank | lib | m_kin=5 | OK | none | 1 |  | 0 | 0 | 4.842e-07 | 2.527e-08 | 0.01 |
| kin | lowrank | lib | m_kin=20 | OK | none | 1 |  | 0 | 0 | 4.38e-07 | 9.71e-08 | 0.011 |
| kin | identity | lib |  | OK | none | 1 |  | 0 | 0 |  | 1.178e-15 | 0.009 |
| kin | dup_pairs | lib |  | OK | none | 1 |  | 0 | 0 | 5.603e-07 | 6.93e-07 | 0.011 |
| kin | dup_all | lib |  | OK | none | 1 |  | 0 | 0 | 6.197e-07 | 7.412e-08 | 0.01 |
| kin | scale | lib | scale=0.0001 | OK | none | 1 |  | 0 | 0 | 0.009936 |  | 0.011 |
| kin | scale | lib | scale=10000 | OK | none | 1 |  | 0 | 0 | 3.621e-11 | 4.664e-08 | 0.012 |
| cov | cov_nan | cli_impute |  | OK | none | 1 | gwas.py: error: The covariate file %s contains missing values. At this time we are not dealing with this case.  Either remove those individuals with missing val |  |  |  |  | 1.007 |
| cov | many | lib | k=5 | OK | none | 1 |  | 0 | 0 | 3.951e-08 | 5.573e-08 | 0.015 |
| cov | many | lib | k=20 | OK | none | 1 |  | 0 | 0 | 4.286e-08 | 5.357e-08 | 0.022 |
| cov | many | lib | k=40 | OK | none | 1 |  | 0 | 0 | 6.604e-08 | 8.306e-08 | 0.035 |
| cov | many | lib | k=80 | OK | none | 1 |  | 0 | 0 | 2.013e-07 | 2.055e-07 | 0.122 |
| pheno | scale | lib | scale=1e-06 | OK | none | 1 |  | 0 | 0 | 9.677e-08 | 1.756e-07 | 0.012 |
| pheno | scale | lib | scale=1e+06 | OK | none | 1 |  | 0 | 0 | 1.324e-07 | 2.195e-07 | 0.012 |
| pheno | offset | lib | offset=1e+09 | OK | none | 1 |  | 0 | 0 | 4.303e-07 | 3.659e-05 | 0.012 |
| pheno | binary | lib |  | OK | none | 1 |  | 0 | 0 | 1.95e-08 | 1.957e-08 | 0.011 |
| pheno | heavy_t2 | lib |  | OK | none | 1 |  | 0 | 0 | 3.746e-08 | 6.402e-08 | 0.015 |
| pheno | lognormal | lib |  | OK | none | 1 |  | 0 | 0 | 6.809e-08 | 8.161e-08 | 0.011 |
| pheno | inf_y | lib |  | OK | none | 1 |  | 20 | 0 |  |  | 0.01 |
| pheno | few_obs | lib | n_obs=3 | OK | none | 1 |  | 0 | 0 | 0.01 | 1.443e-15 | 0.008 |
| pheno | binary | cli_impute |  | OK | none | 1 |  | 0 | 0 | 0.0004563 | 3.997e-08 | 1.124 |
| pheno | inf_y | cli_impute |  | OK | none | 1 |  | 20 | 0 |  |  | 1.131 |
| io | pheno_shuffled | cli_impute |  | OK | none | 1 |  | 0 | 0 | 0.0004709 | 7.009e-08 | 1.069 |
| io | pheno_extra_indiv | cli_impute |  | OK | none | 1 |  | 0 | 0 | 0.000454 | 3.272e-08 | 0.995 |
| io | pheno_neg9 | cli_impute |  | OK | none | 1 |  | 0 | 0 | 0.000427 | 1.373e-07 | 1.425 |

## Outcome counts by family

| family | OK | crash | silent | wrong |
|---|---|---|---|---|
| cov | 7 | 8 | 0 | 0 |
| degenerate | 23 | 3 | 0 | 12 |
| io | 3 | 3 | 0 | 2 |
| kin | 8 | 1 | 2 | 3 |
| missing | 41 | 0 | 0 | 9 |
| pheno | 10 | 2 | 1 | 2 |
| smalln | 11 | 2 | 0 | 8 |
