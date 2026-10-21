"""The `pylmm` command line itself (Phase 5b.2): the parser (required options,
mutually exclusive groups, validation), exit codes, `--version`,
`python -m pylmm`, and logging (-v/-q/--log-file/--debug, library warnings).

The numerical outputs are covered by test_cli_gwas.py and test_kinship.py."""

import argparse
import contextlib
import csv
import io
import logging
import os
import subprocess
import sys
from importlib.metadata import entry_points

import helpers as h
import numpy as np
import pytest

import pylmm
from pylmm import cli
from pylmm.cli import gwas as cli_gwas
from pylmm.lmm import calculate_kinship

N = 30


def run(argv):
    """`pylmm <argv>` in-process: (exit code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = cli.main(argv)
    return rc, out.getvalue(), err.getvalue()


def python_m(args, cwd):
    """`python -m pylmm <args>` in a subprocess: (exit code, stdout, stderr)."""
    env = dict(os.environ)
    env["PYTHONPATH"] = h.REPO + os.pathsep + env.get("PYTHONPATH", "")
    p = subprocess.run(
        [sys.executable, "-m", "pylmm", *args],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=h.CLI_TIMEOUT,
        check=False,
    )
    return p.returncode, p.stdout, p.stderr


@pytest.fixture
def emma(tmp_path):
    """Small EMMA inputs: 5 SNPs x 30 individuals, a phenotype and a kinship."""
    rng = np.random.default_rng(0)
    G = rng.binomial(2, 0.4, size=(N, 5)).astype(float)
    snps = str(tmp_path / "g.emma")
    np.savetxt(snps, G.T)
    ph = str(tmp_path / "p.emma")
    np.savetxt(ph, rng.standard_normal(N)[None, :])
    kin = str(tmp_path / "k.txt")
    np.savetxt(kin, calculate_kinship(rng.binomial(2, 0.4, size=(N, 100)).astype(float)))
    out = str(tmp_path / "out.tsv")
    argv = ["gwas", "--emma-geno", snps, "--emma-pheno", ph, "--kinship", kin, "-o", out]
    return {"argv": argv, "out": out, "kin": kin, "dir": tmp_path}


# ---------------------------------------------------------------------------
# entry points, --version, --help
# ---------------------------------------------------------------------------


def test_version():
    rc, out, _ = run(["--version"])
    assert rc == 0 and out.strip() == "pylmm " + pylmm.__version__


def test_python_m_pylmm(tmp_path):
    rc, out, _ = python_m(["--version"], tmp_path)
    assert rc == 0 and out.startswith("pylmm ")
    rc, out, _ = python_m(["--help"], tmp_path)
    assert rc == 0 and "gwas" in out and "kinship" in out


def test_python_m_pylmm_gwas_runs(emma):
    rc, _, err = python_m(emma["argv"], emma["dir"])
    assert rc == 0, err
    assert open(emma["out"]).readline() == "SNP_ID\tBETA\tBETA_SD\tF_STAT\tP_VALUE\n"


def test_console_script_entry_point():
    (ep,) = entry_points(group="console_scripts", name="pylmm")
    assert ep.value == "pylmm.cli:main"
    names = {e.name for e in entry_points(group="console_scripts")}
    assert not names & {"pylmmGWAS", "pylmmKinship"}


@pytest.mark.parametrize("command", ["gwas", "kinship"])
def test_help_has_examples(command):
    rc, out, _ = run([command, "--help"])
    assert rc == 0
    assert out.startswith("usage: pylmm %s" % command)
    assert "examples:" in out and "pylmm %s --bfile" % command in out


# ---------------------------------------------------------------------------
# usage errors: exit code 2
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        ([], "required: COMMAND"),
        (["frobnicate"], "invalid choice"),
        (["gwas", "--kinship", "k", "-o", "o"], "one of the arguments --bfile --tfile --emma-geno"),
        (["gwas", "--bfile", "b", "-o", "o"], "one of the arguments --kinship --grm is required"),
        (["gwas", "--bfile", "b", "--kinship", "k"], "required: -o/--out"),
        (["gwas", "--bfile", "b", "--tfile", "t", "--kinship", "k", "-o", "o"], "not allowed"),
        (["gwas", "--bfile", "b", "--kinship", "k", "-o", "o", "--reml", "--ml"], "not allowed"),
        (
            [
                "gwas",
                "--bfile",
                "b",
                "--kinship",
                "k",
                "-o",
                "o",
                "--pheno",
                "p",
                "--emma-pheno",
                "q",
            ],
            "not allowed",
        ),
        (
            [
                "gwas",
                "--bfile",
                "b",
                "--kinship",
                "k",
                "-o",
                "o",
                "--covar",
                "c",
                "--emma-covar",
                "d",
            ],
            "not allowed",
        ),
        (["gwas", "--bfile", "b", "--kinship", "k", "-o", "o", "-v", "-q"], "not allowed"),
        (["gwas", "--bfile", "b", "--kinship", "k", "-o", "o", "--id-match", "fid"], "choose"),
        (["gwas", "--bfile", "b", "--kinship", "k", "-o", "o", "--pheno-col", "0"], "not positive"),
        (
            ["gwas", "--bfile", "b", "--kinship", "k", "-o", "o", "--pheno-col", "-1"],
            "not positive",
        ),
        (["gwas", "--bfile", "b", "--kinship", "k", "-o", "o", "--pheno-col", "x"], "integer"),
        (
            [
                "gwas",
                "--emma-geno",
                "g",
                "--emma-pheno",
                "p",
                "--pheno-name",
                "y",
                "--kinship",
                "k",
                "-o",
                "o",
            ],
            "--pheno-name needs --pheno",
        ),
        (["kinship", "-o", "o"], "one of the arguments --bfile --tfile --emma-geno"),
        (["kinship", "--bfile", "b"], "required: -o/--out"),
        (["kinship", "--bfile", "b", "--emma-num-snps", "3", "-o", "o"], "only to --emma-geno"),
        (["kinship", "--bfile", "b", "--block-size", "0", "-o", "o"], "not positive"),
    ],
)
def test_usage_errors_exit_2(argv, message):
    rc, _, err = run(argv)
    assert rc == 2
    assert "usage: pylmm" in err and message in err


@pytest.mark.parametrize(
    "old",
    [
        ["--kfile", "k"],
        ["--phenofile", "p"],
        ["--covfile", "c"],
        ["--emmaSNP", "g"],
        ["--REML"],
        ["--ML"],
        ["--noMean"],
        ["--no-mean"],
        ["--pheno-index", "0"],
        ["--removeMissingGenotypes"],
        ["-p", "0"],
    ],
)
def test_old_flags_are_gone(old):
    """A clean break: the pylmmGWAS flags are unknown options (exit 2)."""
    rc, _, err = run(["gwas", "--bfile", "b", "--kinship", "k", "-o", "o", *old])
    assert rc == 2 and "unrecognized arguments" in err


def test_old_positional_output_is_gone():
    rc, _, err = run(["gwas", "--bfile", "b", "--kinship", "k", "out.txt"])
    assert rc == 2 and "required: -o/--out" in err


def test_input_error_exit_2_without_traceback(emma):
    argv = [a if a != emma["kin"] else emma["kin"] + ".missing" for a in emma["argv"]]
    rc, _, err = run(argv)
    assert rc == 2
    assert "pylmm: error: --kinship %s.missing: no such file" % emma["kin"] in err
    assert "no kinship ID file" not in err and "Traceback" not in err
    assert not os.path.exists(emma["out"])  # nothing written before the inputs are valid
    rc, _, err = run([*argv, "--debug"])
    assert rc == 2 and "Traceback" in err


@pytest.mark.parametrize("command", ["gwas", "kinship"])
def test_missing_genotype_file_exit_2(command, tmp_path):
    nope = str(tmp_path / "nope")
    argv = [command, "--bfile", nope, "-o", str(tmp_path / "o")]
    if command == "gwas":
        argv += ["--kinship", "k"]
    rc, _, err = run(argv)
    assert rc == 2 and "Traceback" not in err
    assert "pylmm: error: --bfile %s: [Errno 2]" % nope in err and "nope.fam" in err


def test_output_is_checked_before_the_work(emma, monkeypatch):
    """-o (and --eigen-npz) are checked up front, and a check leaves no file."""

    def boom(args):
        raise AssertionError("the inputs were read before the output was checked")

    monkeypatch.setattr(cli_gwas, "load_inputs", boom)
    argv = [*emma["argv"][:-1], str(emma["dir"] / "no" / "dir.tsv")]
    rc, _, err = run(argv)
    assert rc == 2 and "pylmm: error: --out %s" % argv[-1] in err
    snps, d = emma["argv"][2], emma["dir"]
    rc, _, err = run(["kinship", "--emma-geno", snps, "-o", str(d / "no" / "k")])
    assert rc == 2 and "pylmm: error: --out %s" % (d / "no" / "k") in err
    npz = str(d / "no" / "e")  # np.savez would write e.npz
    rc, _, err = run(["kinship", "--emma-geno", snps, "-o", str(d / "k"), "--eigen-npz", npz])
    assert rc == 2 and "pylmm: error: --eigen-npz %s.npz" % npz in err
    assert not os.path.exists(d / "k")  # the check of -o left no file behind
    monkeypatch.undo()
    out = emma["out"]
    assert run([*emma["argv"]])[0] == 0 and os.path.getsize(out) > 0


@pytest.mark.parametrize("command", ["gwas", "kinship"])
def test_log_file_cannot_be_the_output(command, emma):
    out = emma["out"]
    argv = [command, "--emma-geno", emma["argv"][2], "-o", out, "--log-file", out]
    if command == "gwas":
        argv = [*emma["argv"], "--log-file", out]
    rc, _, err = run(argv)
    assert rc == 2 and "--log-file %s is also the -o/--out file" % out in err


def test_debug_argument_dump_has_no_callables(emma):
    rc, _, err = run([*emma["argv"], "--debug"])
    line = next(x for x in err.splitlines() if x.startswith("pylmm: debug: arguments:"))
    assert rc == 0 and "'kinship':" in line
    assert "function" not in line and "ArgumentParser" not in line


def test_unwritable_output_exit_2(emma):
    argv = [*emma["argv"][:-1], str(emma["dir"] / "no" / "such" / "dir.tsv")]
    rc, _, err = run(argv)
    assert rc == 2 and "--out" in err and "Traceback" not in err


def test_unexpected_error_exit_1(emma, monkeypatch):
    def boom(args):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(cli_gwas, "load_inputs", boom)
    rc, _, err = run(emma["argv"])
    assert rc == 1
    assert "pylmm: error: unexpected error: RuntimeError: kaboom" in err
    assert "--debug" in err and "Traceback" not in err
    rc, _, err = run([*emma["argv"], "--debug"])
    assert rc == 1 and "Traceback" in err and "kaboom" in err


@pytest.mark.parametrize("command", ["gwas", "kinship"])
def test_python_m_old_module_is_a_tombstone(command, tmp_path):
    """`python -m pylmm.cli.gwas` (the pre-5b.2 invocation) is not an alias:
    one error line pointing at `pylmm <command>`, exit 2 (it used to do
    nothing and exit 0)."""
    env = dict(os.environ)
    env["PYTHONPATH"] = h.REPO + os.pathsep + env.get("PYTHONPATH", "")
    p = subprocess.run(
        [sys.executable, "-m", "pylmm.cli." + command, "--bfile", "x", "-o", "y"],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=h.CLI_TIMEOUT,
        check=False,
    )
    assert p.returncode == 2 and p.stdout == ""
    assert p.stderr == (
        "pylmm: error: python -m pylmm.cli.%s was removed; use `pylmm %s` / "
        "`python -m pylmm %s` (see docs/migration/cli-map.csv)\n" % (command, command, command)
    )
    assert not os.path.exists(tmp_path / "y")


def _write_tped(d, n=N, m=6, bad_allele_at=None):
    """A 1/2-recoded TPED/TFAM/phenotype set; ``bad_allele_at`` puts an "A" in that SNP."""
    rng = np.random.default_rng(5)
    base = str(d / "t")
    with open(base + ".tfam", "w") as f, open(base + ".phenos", "w") as ph:
        for i in range(n):
            f.write("F%d I%d 0 0 1 -9\n" % (i, i))
            ph.write("F%d I%d %.6f\n" % (i, i, rng.standard_normal()))
    with open(base + ".tped", "w") as f, open(base + ".map", "w") as mp:
        for j in range(m):
            al = []
            for g in rng.integers(0, 3, n):
                al += {0: ["1", "1"], 1: ["1", "2"], 2: ["2", "2"]}[int(g)]
            if j == bad_allele_at:
                al[3] = "A"
            f.write("1 rs%d 0 %d %s\n" % (j, j + 1, " ".join(al)))
            mp.write("1 rs%d 0 %d\n" % (j, j + 1))
    return base


@pytest.mark.parametrize("command", ["gwas", "kinship"])
def test_input_error_while_reading_snps_exit_2(command, tmp_path):
    """A malformed SNP found mid-scan (after the up-front checks) is an input
    error, exit 2, not an unexpected error (exit 1)."""
    good = _write_tped(tmp_path)
    kin = str(tmp_path / "k.txt")
    assert run(["kinship", "--tfile", good, "-o", kin])[0] == 0
    bad = _write_tped(tmp_path, bad_allele_at=4)
    out = str(tmp_path / "o")
    argv = ["kinship", "--tfile", bad, "-o", out]
    if command == "gwas":
        argv = ["gwas", "--tfile", bad, "--kinship", kin, "-o", out]
    rc, _, err = run(argv)
    assert rc == 2, err
    assert "pylmm: error: TPED file %s.tped, SNP rs4: allele code 'A'" % bad in err
    assert "unexpected" not in err and "Traceback" not in err


def test_truncated_tped_exit_2(tmp_path):
    base = _write_tped(tmp_path)
    with open(base + ".map", "a") as f:  # one SNP more in the .map than in the .tped
        f.write("1 rs99 0 99\n")
    rc, _, err = run(["kinship", "--tfile", base, "-o", str(tmp_path / "k")])
    assert rc == 2 and "fewer lines than the .map" in err and "unexpected" not in err


def test_input_error_is_a_value_error():
    from pylmm import plink_io, samples

    assert plink_io.InputError is samples.InputError
    assert issubclass(samples.InputError, ValueError)


def _emma_snps(d, m=7, blank_line=False):
    rng = np.random.default_rng(6)
    fn = str(d / "g.emma")
    G = rng.binomial(2, 0.4, size=(m, N)) / 2.0
    with open(fn, "w") as f:
        for j, row in enumerate(G):
            f.write(" ".join("%g" % v for v in row) + "\n")
            if blank_line and j == 2:
                f.write("\n")
    return fn


def test_kinship_emma_num_snps_is_inferred(tmp_path):
    """--emma-num-snps is optional: the SNPs are counted (blank lines
    skipped); giving the right count changes nothing."""
    g = _emma_snps(tmp_path)
    a, b, c = (str(tmp_path / x) for x in ("a", "b", "c"))
    assert run(["kinship", "--emma-geno", g, "-o", a])[0] == 0
    assert run(["kinship", "--emma-geno", g, "--emma-num-snps", "7", "-o", b])[0] == 0
    with open(a) as fa, open(b) as fb:
        assert fa.read() == fb.read()
    (tmp_path / "blank").mkdir()
    blank = _emma_snps(tmp_path / "blank", blank_line=True)
    assert run(["kinship", "--emma-geno", blank, "--emma-num-snps", "7", "-o", c])[0] == 0
    np.testing.assert_array_equal(np.loadtxt(c), np.loadtxt(a))


@pytest.mark.parametrize("m", ["6", "8", "100"])
def test_kinship_wrong_emma_num_snps_exit_2(m, tmp_path):
    """A wrong count used to compute K from the first M SNPs (too small) or
    crash with StopIteration (too large)."""
    g = _emma_snps(tmp_path)
    out = str(tmp_path / "k")
    rc, _, err = run(["kinship", "--emma-geno", g, "--emma-num-snps", m, "-o", out])
    assert rc == 2
    assert "pylmm: error: --emma-num-snps %s: the --emma-geno file %s has 7 SNPs" % (m, g) in err
    assert not os.path.exists(out)


def test_emma_row_of_the_wrong_length_exit_2(tmp_path):
    g = _emma_snps(tmp_path)
    with open(g, "a") as f:
        f.write("0 1 0\n")
    rc, _, err = run(["kinship", "--emma-geno", g, "-o", str(tmp_path / "k")])
    assert rc == 2 and "EMMA genotype file %s, line 8: 3 values, expected %d" % (g, N) in err


# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------


def test_default_level_shows_warnings_not_info(emma):
    rc, out, err = run(emma["argv"])
    assert rc == 0 and out == ""
    assert "pylmm: info:" not in err
    # an EMMA kinship has no ID file and needs none: no warning either
    assert err == ""


def test_verbose_logs_progress(emma):
    rc, _, err = run([*emma["argv"], "-v"])
    assert rc == 0
    assert "pylmm: info: Reading SNP input..." in err
    assert "pylmm: info: null model: heritability=" in err
    assert h.null_fit_from_log(err)


def _asymmetric(emma):
    K = np.loadtxt(emma["kin"])
    K[0, 1] += 0.01
    np.savetxt(emma["kin"], K)


def test_library_warnings_are_logged(emma):
    """warnings.warn in the library becomes one `pylmm: warning:` line."""
    _asymmetric(emma)
    rc, _, err = run(emma["argv"])
    assert rc == 0
    assert "pylmm: warning: kinship file %s is not symmetric" % emma["kin"] in err
    assert "UserWarning" not in err and "warnings.warn" not in err


def test_quiet_hides_warnings(emma):
    _asymmetric(emma)
    rc, _, err = run([*emma["argv"], "-q"])
    assert rc == 0 and err == ""


def test_log_file_records_info_and_warnings(emma):
    _asymmetric(emma)
    log_file = str(emma["dir"] / "run.log")
    rc, _, err = run([*emma["argv"], "-q", "--log-file", log_file])
    assert rc == 0 and err == ""
    text = open(log_file).read()
    assert " INFO Reading SNP input..." in text
    assert " WARNING kinship file" in text and "is not symmetric" in text


def test_log_file_that_cannot_be_opened_exit_2(emma):
    rc, _, err = run([*emma["argv"], "--log-file", str(emma["dir"] / "no" / "x.log")])
    assert rc == 2 and "pylmm: error: --log-file" in err


def test_debug_logs_debug_messages(emma):
    rc, _, err = run([*emma["argv"], "--debug"])
    assert rc == 0 and "pylmm: debug: arguments:" in err


def test_logging_is_restored_after_main(emma):
    """main() removes its handlers, so repeated in-process calls do not stack
    them and the `pylmm` logger is left as it was."""
    logger = logging.getLogger("pylmm")
    before = (list(logger.handlers), logger.level, logger.propagate)
    for _ in range(2):
        rc, _, err = run([*emma["argv"], "-v"])
        assert rc == 0 and err.count("Reading SNP input") == 1
    assert (list(logger.handlers), logger.level, logger.propagate) == before


# ---------------------------------------------------------------------------
# docs/migration/cli-map.csv
# ---------------------------------------------------------------------------

CLI_MAP_KINDS = {
    "renamed",
    "value→flag",
    "removed",
    "removed-earlier",
    "new",
    "behaviour-changed",
    "semantics-changed",
    "unchanged",
}


def _cli_map():
    with open(os.path.join(h.REPO, "docs", "migration", "cli-map.csv"), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_cli_map_schema():
    rows = _cli_map()
    assert list(rows[0]) == [
        "old_command",
        "old_flag",
        "old_takes_value",
        "new_command",
        "new_flag",
        "new_takes_value",
        "kind",
        "notes",
    ]
    for row in rows:
        assert row["kind"] in CLI_MAP_KINDS, row
        for side in ("old", "new"):
            # flags only: no VALUE or (positional) text in the flag columns
            assert all(f.startswith("-") for f in row[side + "_flag"].split("/") if f), row
            assert row[side + "_takes_value"] in ("yes", "no", ""), row
        assert (row["old_command"] == "") == (row["kind"] == "new"), row


def test_cli_map_covers_every_option():
    """Every option of `pylmm gwas` / `pylmm kinship` has a row, and every
    new flag of the map exists and takes a value as the map says."""
    parser = cli.build_parser()
    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    actual = {}
    for name, p in sub.choices.items():
        for action in p._actions:
            for opt in action.option_strings:
                actual[("pylmm " + name, opt)] = action.nargs != 0
    mapped = set()
    for row in _cli_map():
        for command in row["new_command"].split(", "):
            for opt in row["new_flag"].split("/"):
                if opt and command != "pylmm":
                    assert (command, opt) in actual, row
                    assert actual[(command, opt)] == (row["new_takes_value"] == "yes"), row
                    mapped.add((command, opt))
    assert set(actual) - mapped == set()
