# Example: null calibration (synthetic unrelated)

_Last run: 2026-10-18T12:09:50, git 792f71b, jobs=6, wall 23s; total task time over all runs 112s._

Null SNPs only, so lambda_GC should be ~1 and t1e_<alpha> ~ alpha for both methods.  dh_oracle / dll_oracle / t_vs_oracle_* compare pylmm with the independent dense REML oracle (sims/oracle.py).  All columns (with standard errors over replicates) are in summary.csv.

| n | h2 | m_test | n_reps | h_hat | lambda_lmm | lambda_ols | t1e_0.05_lmm | t1e_0.01_lmm | t1e_1e-03_lmm | t1e_0.05_ols | t1e_0.01_ols | t1e_1e-03_ols | h_oracle | dh_oracle | dll_oracle | t_vs_oracle_max_abs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200 | 0 | 2000 | 20 | 0.1619 | 0.9971 | 0.9971 | 0.0491 | 0.01035 | 0.00085 | 0.04975 | 0.0101 | 0.000875 | 0.1619 | 4.326e-09 | 2.842e-15 | 9.438e-09 |
| 200 | 0.5 | 2000 | 20 | 0.6047 | 1.012 | 1.005 | 0.0494 | 0.0101 | 0.0011 | 0.04875 | 0.0099 | 0.00095 | 0.6055 | -0.0008156 | -0.001736 | 0.0006173 |
| 500 | 0 | 2000 | 20 | 0.04205 | 0.9808 | 0.9793 | 0.04877 | 0.0094 | 0.0008 | 0.04875 | 0.009325 | 0.000825 | 0.04205 | -8.685e-10 | -5.684e-13 | 7.291e-09 |
| 500 | 0.5 | 2000 | 20 | 0.5139 | 1.019 | 1.023 | 0.05025 | 0.0093 | 0.0009 | 0.05008 | 0.0101 | 0.001025 | 0.5139 | 2.275e-08 | -5.684e-15 | 3.824e-08 |
