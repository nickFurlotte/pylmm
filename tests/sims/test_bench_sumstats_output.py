"""The Phase 6.2b benchmark sims/bench/sumstats_output.py (the cost of the summary-statistics output)."""

from sims.bench import sumstats_output


def test_bench_sumstats_output_quick(capsys):
    assert sumstats_output.main(["--quick", "--repeats", "1"]) == 0
    out = capsys.readouterr().out
    assert "| sumstats .tsv.gz |" in out
    assert "t_stat == -F_STAT: True; p_value == P_VALUE: True" in out
