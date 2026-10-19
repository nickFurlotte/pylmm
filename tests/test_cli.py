"""The `pylmm` command line itself (Phase 5b.2): the parser (required options,
mutually exclusive groups, validation), exit codes, `--version`,
`python -m pylmm`, and logging (-v/-q/--log-file/--debug, library warnings).

The numerical outputs are covered by test_cli_gwas.py and test_kinship.py."""

import contextlib
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
        (["gwas", "--bfile", "b", "-o", "o"], "required: --kinship"),
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
        (["gwas", "--bfile", "b", "--kinship", "k", "-o", "o", "--pheno-index", "-1"], "negative"),
        (["gwas", "--bfile", "b", "--kinship", "k", "-o", "o", "--pheno-index", "x"], "integer"),
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
        (["kinship", "--emma-geno", "g", "-o", "o"], "--emma-geno needs --emma-num-snps"),
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
    assert "pylmm: error: kinship file" in err and "Traceback" not in err
    assert not os.path.exists(emma["out"])  # nothing written before the inputs are valid
    rc, _, err = run([*argv, "--debug"])
    assert rc == 2 and "Traceback" in err


def test_missing_genotype_file_exit_2(tmp_path):
    rc, _, err = run(
        ["gwas", "--bfile", str(tmp_path / "nope"), "--kinship", "k", "-o", str(tmp_path / "o")]
    )
    assert rc == 2 and "pylmm: error:" in err and "nope.fam" in err and "Traceback" not in err


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
