# Historical one-shot script, kept for the record: it is the snake_case rename that produced the
# 1.0.0 API, and the source of the 5.4 rows of name-map.csv.  It is not meant to be run again, and
# `--apply` refuses to run on the current tree (pylmm/plink_io.py already exists).  `--csv` still
# rewrites the 5.4 rows of name-map.csv from RENAMES and keeps the later rows.
"""The Phase 5.4 snake_case rename, as a reviewable script.

    uv run python docs/migration/rename_5_4.py --apply   # rewrite the tree
    uv run python docs/migration/rename_5_4.py --csv     # write name-map.csv

RENAMES below is the whole mapping.  Rows whose module is a pylmm module are
the public (and private) pylmm API and are written to name-map.csv, the source
for the migration skill; rows with module "-" are local names in tests/,
sims/ and scripts/ and are not part of the API.

The rewrite works on Python tokens, never on raw text: only NAME tokens are
renamed, so strings (file contents, golden keys, npz array names, CLI help)
and comments are left alone.  A name is renamed according to its context:

  attr  after a "." (obj.name), except on the optparse `options` object,
        whose attribute names are CLI dests (they set the --help metavars;
        the CLI is redesigned in Phase 5b);
  bare  anywhere else (definitions, parameters, locals, keyword arguments),
        except keyword arguments of np.savez (the .npz array names Kva/Kve
        are a file format).

Run it once on the Phase 5.3 tree; it is not idempotent by design (it
refuses to run if pylmm/plink_io.py already exists).
"""

import argparse
import csv
import io
import os
import subprocess
import sys
import tokenize

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# (old, new, kind, module, context)  context: "both" | "attr" | "bare"
# kind: function | method | class | attribute | parameter | module | local
# module: where the name lives AFTER the rename -- the module for a function
# or class, module.Class for a method or attribute, the function or method
# for a parameter.  Several owners are separated by ";" (one CSV row each).
LM, IO, M, CG, CK = (
    "pylmm.lmm",
    "pylmm.plink_io",
    "pylmm.misc",
    "pylmm.cli.gwas",
    "pylmm.cli.kinship",
)
RENAMES = [
    # --- modules -----------------------------------------------------------
    ("pylmm.input", "pylmm.plink_io", "module", "pylmm", "-"),
    # --- pylmm.lmm ---------------------------------------------------------
    ("matrixMult", "_matrix_mult", "function", LM, "both"),
    ("calculateKinship", "calculate_kinship", "function", LM, "both"),
    ("GWAS", "gwas", "function", LM, "both"),
    ("_residualOnSpan", "_residual_on_span", "function", LM, "both"),
    ("_columnList", "_column_list", "function", LM, "both"),
    ("_checkPhenotype", "_check_phenotype", "function", LM, "both"),
    ("_checkDesignSize", "_check_design_size", "function", LM, "both"),
    ("_checkKinship", "_check_kinship", "method", LM + ".LMM", "both"),
    ("_warnNegativeEigenvalues", "_warn_negative_eigenvalues", "method", LM + ".LMM", "both"),
    ("getMLSoln", "ml_solution", "method", LM + ".LMM", "both"),
    ("LL_brent", "_neg_log_likelihood", "method", LM + ".LMM", "both"),
    ("LL", "log_likelihood", "method", LM + ".LMM", "both"),
    ("getMax", "_get_max", "method", LM + ".LMM", "both"),
    ("plotFit", "plot_fit", "method", LM + ".LMM", "both"),
    ("meanAndVar", "mean_and_var", "method", LM + ".LMM", "both"),
    ("optH", "opt_h", "attribute", LM + ".LMM", "attr"),
    ("optLL", "opt_ll", "attribute", LM + ".LMM", "attr"),
    ("optBeta", "opt_beta", "attribute", LM + ".LMM", "attr"),
    ("optSigma", "opt_sigma", "attribute", LM + ".LMM", "attr"),
    ("Yt", "yt", "attribute", LM + ".LMM", "both"),
    ("X0t", "x0t", "attribute", LM + ".LMM", "both"),
    ("X0t_stack", "x0t_stack", "attribute", LM + ".LMM", "both"),
    ("LLs", "ll_grid", "attribute", LM + ".LMM", "both"),
    ("H", "h_grid", "attribute", LM + ".LMM", "attr"),
    ("Kva", "kva", "attribute", LM + ".LMM", "both"),
    ("Kve", "kve", "attribute", LM + ".LMM", "both"),
    (
        "Kva",
        "kva",
        "parameter",
        f"{LM}.gwas;{LM}.LMM.__init__;{LM}.LMM._warn_negative_eigenvalues",
        "-",
    ),
    ("Kve", "kve", "parameter", f"{LM}.gwas;{LM}.LMM.__init__", "-"),
    (
        "REML",
        "reml",
        "parameter",
        f"{LM}.gwas;{LM}.LMM.fit;{LM}.LMM.association;{LM}.LMM.log_likelihood;"
        f"{LM}.LMM._neg_log_likelihood;{LM}.LMM._get_max",
        "both",
    ),
    ("K_ids", "kinship_ids", "parameter", f"{LM}.gwas;{LM}.LMM.__init__;{LM}.check_ids", "bare"),
    ("Ys", "ys", "parameter", f"{LM}.subset_testable", "-"),
    ("X0s", "x0s", "parameter", f"{LM}.subset_testable", "-"),
    ("returnBeta", "return_beta", "parameter", LM + ".LMM.association", "bare"),
    # --- pylmm.misc --------------------------------------------------------
    ("fitTwo", "fit_two", "function", M, "both"),
    # --- pylmm.plink_io (was pylmm.input) ----------------------------------
    ("plink", "PlinkReader", "class", IO, "both"),
    ("getSNPIterator", "get_snp_iterator", "method", IO + ".PlinkReader", "both"),
    ("getSNPIterator_emma", "_get_snp_iterator_emma", "method", IO + ".PlinkReader", "both"),
    ("getSNPIterator_tped", "_get_snp_iterator_tped", "method", IO + ".PlinkReader", "both"),
    ("getSNPIterator_bed", "_get_snp_iterator_bed", "method", IO + ".PlinkReader", "both"),
    ("getGenos_tped", "_get_genos_tped", "method", IO + ".PlinkReader", "both"),
    ("formatBinaryGenotypes", "_format_binary_genotypes", "method", IO + ".PlinkReader", "both"),
    ("normalizeGenotype", "normalize_genotype", "method", IO + ".PlinkReader", "both"),
    ("getPhenos", "get_phenos", "method", IO + ".PlinkReader", "both"),
    ("getPhenosEMMA", "get_phenos_emma", "method", IO + ".PlinkReader", "both"),
    ("_emmaColumns", "_emma_columns", "method", IO + ".PlinkReader", "both"),
    ("_readTable", "_read_table", "method", IO + ".PlinkReader", "both"),
    ("_alignRows", "_align_rows", "method", IO + ".PlinkReader", "both"),
    ("getIndivs", "_read_fam", "method", IO + ".PlinkReader", "both"),
    ("readKinship", "read_kinship", "method", IO + ".PlinkReader", "both"),
    ("getCovariatesEMMA", "get_covariates_emma", "method", IO + ".PlinkReader", "both"),
    ("getCovariates", "get_covariates", "method", IO + ".PlinkReader", "both"),
    (
        "kFile",
        "k_file",
        "parameter",
        f"{IO}.PlinkReader.__init__;{IO}.PlinkReader.read_kinship",
        "bare",
    ),
    (
        "phenoFile",
        "pheno_file",
        "parameter",
        f"{IO}.PlinkReader.__init__;{IO}.PlinkReader.get_phenos",
        "bare",
    ),
    ("normGenotype", "norm_genotype", "parameter", IO + ".PlinkReader.__init__", "bare"),
    ("readKFile", "read_k_file", "parameter", IO + ".PlinkReader.__init__", "bare"),
    ("fastLMM_kinship", "fastlmm_kinship", "parameter", IO + ".PlinkReader.__init__", "bare"),
    (
        "emmaFile",
        "emma_file",
        "parameter",
        f"{IO}.PlinkReader.get_phenos_emma;{IO}.PlinkReader.get_covariates_emma",
        "bare",
    ),
    ("covFile", "cov_file", "parameter", IO + ".PlinkReader.get_covariates", "bare"),
    ("isMissing", "is_missing", "parameter", IO + ".PlinkReader._read_table", "bare"),
    ("kFile", "k_file", "attribute", IO + ".PlinkReader", "attr"),
    ("phenoFile", "pheno_file", "attribute", IO + ".PlinkReader", "attr"),
    ("normGenotype", "norm_genotype", "attribute", IO + ".PlinkReader", "attr"),
    ("readKFile", "read_k_file", "attribute", IO + ".PlinkReader", "attr"),
    ("fastLMM_kinship_style", "fastlmm_kinship", "attribute", IO + ".PlinkReader", "attr"),
    ("snpFileHandle", "snp_file_handle", "attribute", IO + ".PlinkReader", "attr"),
    ("numSNPs", "num_snps", "attribute", IO + ".PlinkReader", "attr"),
    ("BytestoRead", "_bytes_to_read", "attribute", IO + ".PlinkReader", "attr"),
    # --- pylmm.cli ---------------------------------------------------------
    ("printOutHead", "_print_out_head", "function", CG, "both"),
    ("outputResult", "_output_result", "function", CG, "both"),
    ("readKinshipFile", "read_kinship_file", "function", CG, "both"),
    ("_isNumber", "_is_number", "function", CG, "both"),
    ("buildParser", "build_parser", "function", f"{CG};{CK}", "both"),
    ("_plainWarning", "_plain_warning", "function", CG, "both"),
    ("readEigenFile", "read_eigen_file", "function", CG, "both"),
    ("betaSD", "beta_sd", "parameter", CG + "._output_result", "bare"),
    # --- local names (pylmm internals, tests, sims, scripts) ---------------
    ("IN", "reader", "local", "-", "bare"),
    ("TS", "t_stats", "local", "-", "bare"),
    ("PS", "p_values", "local", "-", "bare"),
    ("Ys", "ys", "local", "-", "bare"),
    ("X0s", "x0s", "local", "-", "bare"),
    ("Ks", "ks", "local", "-", "bare"),
    ("Ls", "ls", "local", "-", "bare"),
    ("XX", "xx", "local", "-", "bare"),
    ("XX_i", "xx_inv", "local", "-", "bare"),
    ("Xt", "xt", "local", "-", "bare"),
    ("LL_REML_part", "ll_reml_part", "local", "-", "bare"),
    ("HOpt", "h_opt", "local", "-", "bare"),
    ("betaSTDERR", "beta_stderr", "local", "-", "bare"),
    ("betaVAR", "beta_var", "local", "-", "bare"),
    ("betaVar", "beta_var", "local", "-", "bare"),
    ("K_n", "k_centered", "local", "-", "bare"),
    ("Res", "results", "local", "-", "bare"),
    ("KK", "kept", "local", "-", "bare"),
    ("famFile", "fam_file", "local", "-", "bare"),
    ("famId", "fam_id", "local", "-", "bare"),
    ("indivId", "indiv_id", "local", "-", "bare"),
    ("outFile", "out_file", "local", "-", "bare"),
    ("missingCov", "missing_cov", "local", "-", "bare"),
    ("nAllMissing", "n_all_missing", "local", "-", "bare"),
    ("K_j", "k_j", "local", "-", "bare"),
    ("basicGroup", "basic_group", "local", "-", "bare"),
    ("advancedGroup", "advanced_group", "local", "-", "bare"),
    ("Qx", "qx", "local", "-", "bare"),
    ("RG", "rg", "local", "-", "bare"),
    ("_as_X", "_as_x", "local", "-", "both"),
    ("_prep_K", "_prep_kinship", "local", "-", "both"),
    ("logdetH", "logdet_h", "local", "-", "bare"),
    ("XtHiX", "xt_hi_x", "local", "-", "bare"),
    ("logdet_XtHiX", "logdet_xt_hi_x", "local", "-", "bare"),
    ("logdet_XtX", "logdet_xt_x", "local", "-", "bare"),
    ("Xf", "xf", "local", "-", "bare"),
    ("Qf", "qf", "local", "-", "bare"),
    ("Gc", "g_causal", "local", "-", "bare"),
    ("Gm", "g_missing", "local", "-", "bare"),
    ("G_test", "g_test", "local", "-", "bare"),
    ("G_dis", "g_dis", "local", "-", "bare"),
    ("G_in", "g_in", "local", "-", "bare"),
    ("fT", "f_t", "local", "-", "bare"),
    ("fG", "f_g", "local", "-", "bare"),
    ("fC", "f_c", "local", "-", "bare"),
    ("fB", "f_b", "local", "-", "bare"),
    ("Gk", "g_kin", "local", "-", "bare"),
    ("Gt", "g_test", "local", "-", "both"),
    ("K_true", "k_true", "local", "-", "bare"),
    ("K_lib", "k_lib", "local", "-", "bare"),
    ("K_fix", "k_fix", "local", "-", "bare"),
    ("K_cli", "k_cli", "local", "-", "bare"),
    ("K_mono", "k_mono", "local", "-", "bare"),
    ("K_ref", "k_ref", "local", "-", "attr"),
    ("Gall", "g_all", "local", "-", "bare"),
    ("G_raw", "g_raw", "local", "-", "bare"),
    ("Ki", "k_ok", "local", "-", "bare"),
    ("L201", "ll_201", "local", "-", "bare"),
    ("L099", "ll_099", "local", "-", "bare"),
    ("Gs", "g_std", "local", "-", "bare"),
    ("Kn", "k_noisy", "local", "-", "bare"),
    ("Klow", "k_low", "local", "-", "bare"),
    ("Kc", "k_clamped", "local", "-", "bare"),
    ("Xa", "xa", "local", "-", "bare"),
    ("Xw", "xw", "local", "-", "bare"),
    ("Ym", "y_missing", "local", "-", "bare"),
    ("null_LL", "null_ll", "local", "-", "bare"),
    (
        "test_polygenic_covariance_matches_K",
        "test_polygenic_covariance_matches_k",
        "local",
        "-",
        "bare",
    ),
    (
        "test_identity_K_flat_profile_reduces_to_ols",
        "test_identity_k_flat_profile_reduces_to_ols",
        "local",
        "-",
        "bare",
    ),
    ("test_boundary_h1_with_pd_K", "test_boundary_h1_with_pd_k", "local", "-", "bare"),
    ("test_hmdp_LL_grid", "test_hmdp_ll_grid", "local", "-", "bare"),
]

# Per-file overrides of the bare map: {path: {old: new}}.
FILE_BARE = {
    # Gt here is the whitened genotype matrix (next to yt, Xt), not "test SNPs".
    "sims/studies/s4_bug_confirmation.py": {"Gt": "gt", "input": "plink_io", "pinput": "plink_io"},
    # `from pylmm import input`: the module is renamed.
    "pylmm/cli/gwas.py": {"input": "plink_io"},
    "pylmm/cli/kinship.py": {"input": "plink_io"},
    # `from pylmm import input as pinput`: the alias is no longer needed.
    "tests/conftest.py": {"input": "plink_io", "pinput": "plink_io"},
    "tests/helpers.py": {"input": "plink_io", "pinput": "plink_io"},
    "tests/sims/test_s4_reproducers.py": {"input": "plink_io", "pinput": "plink_io"},
    "sims/genotypes.py": {"input": "plink_io", "pinput": "plink_io"},
}

# Literal fix-ups after the token rewrite: (path, old, new, expected count).
# Only for things the token rule cannot know: an import alias that became
# `plink_io as plink_io`, and attribute names held in strings.
LITERAL = [
    ("tests/conftest.py", "import plink_io as plink_io", "import plink_io", 1),
    ("tests/helpers.py", "import plink_io as plink_io", "import plink_io", 1),
    ("tests/sims/test_s4_reproducers.py", "import plink_io as plink_io", "import plink_io", 1),
    ("sims/genotypes.py", "import plink_io as plink_io", "import plink_io", 1),
    ("sims/studies/s4_bug_confirmation.py", "import plink_io as plink_io", "import plink_io", 1),
    (
        "sims/studies/s4_bug_confirmation.py",
        'getattr(L, "LLs", None)',
        'getattr(L, "ll_grid", None)',
        1,
    ),
    # test_example_script reads example.py's module globals by name
    ("tests/test_example_script.py", 'g["TS"]', 'g["t_stats"]', 1),
    ("tests/test_example_script.py", 'g["PS"]', 'g["p_values"]', 1),
]

EXCLUDE = ("tests/golden/generate_golden.py", "tests/golden/py2_original/", "docs/")


def files():
    out = subprocess.run(
        ["git", "ls-files", "*.py"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.split()
    return [
        f
        for f in out
        if not f.startswith(EXCLUDE) and f.startswith(("pylmm/", "tests/", "sims/", "scripts/"))
    ]


def maps():
    attr, bare = {}, {}
    for old, new, _kind, _mod, ctx in RENAMES:
        if ctx in ("both", "attr"):
            assert attr.get(old, new) == new, old
            attr[old] = new
        if ctx in ("both", "bare"):
            assert bare.get(old, new) == new, old
            bare[old] = new
    return attr, bare


SKIP = {tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT, tokenize.INDENT, tokenize.DEDENT}


def rewrite(path, src, attr, bare):
    bare = dict(bare, **FILE_BARE.get(path, {}))
    toks = [t for t in tokenize.generate_tokens(io.StringIO(src).readline) if t.type not in SKIP]
    lines = src.splitlines(keepends=True)
    offs = [0]
    for ln in lines:
        offs.append(offs[-1] + len(ln))
    edits = []
    callee = []  # name before each open paren
    for i, t in enumerate(toks):
        if t.type == tokenize.OP:
            if t.string in "([{":
                prev = toks[i - 1] if i else None
                callee.append(
                    prev.string if prev is not None and prev.type == tokenize.NAME else None
                )
            elif t.string in ")]}":
                callee.pop()
            continue
        if t.type != tokenize.NAME:
            continue
        prev = toks[i - 1] if i else None
        if prev is not None and prev.type == tokenize.OP and prev.string == ".":
            base = toks[i - 2] if i > 1 else None
            if base is not None and base.string == "options":
                continue
            new = attr.get(t.string)
        else:
            nxt = toks[i + 1] if i + 1 < len(toks) else None
            is_kw = nxt is not None and nxt.type == tokenize.OP and nxt.string == "="
            if is_kw and callee and callee[-1] == "savez":
                continue
            new = bare.get(t.string)
        if new and new != t.string:
            (r, c), (r2, c2) = t.start, t.end
            assert r == r2
            edits.append((offs[r - 1] + c, offs[r - 1] + c2, new))
    out = src
    for a, b, new in sorted(edits, reverse=True):
        out = out[:a] + new + out[b:]
    return out, len(edits)


def collisions(path, src, out):
    """New names that already existed in the file (worth a look)."""
    before = {
        t.string
        for t in tokenize.generate_tokens(io.StringIO(src).readline)
        if t.type == tokenize.NAME
    }
    after = {
        t.string
        for t in tokenize.generate_tokens(io.StringIO(out).readline)
        if t.type == tokenize.NAME
    }
    attr, bare = maps()
    new_names = (
        set(attr.values()) | set(bare.values()) | set(FILE_BARE.get(path, {}).values())
    ) & after
    return sorted(n for n in new_names if n in before)


def apply():
    if os.path.exists(os.path.join(REPO, "pylmm", "plink_io.py")):
        sys.exit("pylmm/plink_io.py exists: the rename has already been applied")
    subprocess.run(["git", "mv", "pylmm/input.py", "pylmm/plink_io.py"], cwd=REPO, check=True)
    attr, bare = maps()
    total = 0
    for f in files():
        p = os.path.join(REPO, f)
        with open(p) as fh:
            src = fh.read()
        out, n = rewrite(f, src, attr, bare)
        if n:
            c = collisions(f, src, out)
            print("%-45s %4d renames%s" % (f, n, ("   check: " + ", ".join(c)) if c else ""))
            total += n
            with open(p, "w") as fh:
                fh.write(out)
    for f, old, new, count in LITERAL:
        p = os.path.join(REPO, f)
        with open(p) as fh:
            s = fh.read()
        assert s.count(old) == count, (f, old, s.count(old))
        with open(p, "w") as fh:
            fh.write(s.replace(old, new))
    print("total: %d token renames" % total)


# CSV-only rows: renames made by hand after the token pass (so not in RENAMES,
# which --apply would rewrite across the tree).  `next` is the py2 iterator
# spelling; callers should use `next(reader)`.
EXTRA_ROWS = [
    ("next", "__next__", "method", IO + ".PlinkReader"),
    ("y", "Y", "parameter", M + ".fit_two"),
    ("wgrids", "ngrids", "parameter", M + ".fit_two"),
]


def write_csv():
    path = os.path.join(REPO, "docs", "migration", "name-map.csv")
    # Rows of later phases (5b, ...) are kept as they are.
    later = []
    if os.path.isfile(path):
        with open(path) as fh:
            later = [r for r in csv.DictReader(fh) if r.get("phase", "5.4") != "5.4"]
    cols = ["old_name", "new_name", "kind", "module", "phase", "notes"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(cols)
        for old, new, kind, mod, _ctx in RENAMES:
            for owner in mod.split(";"):
                if owner.startswith("pylmm"):
                    w.writerow([old, new, kind, owner, "5.4", ""])
        for row in EXTRA_ROWS:
            w.writerow([*row, "5.4", ""])
        for r in later:
            w.writerow([r[c] for c in cols])
    print("wrote", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--apply", action="store_true", help="rewrite the tree")
    ap.add_argument("--csv", action="store_true", help="write docs/migration/name-map.csv")
    a = ap.parse_args()
    if a.apply:
        apply()
    if a.csv:
        write_csv()
    if not (a.apply or a.csv):
        ap.print_help()
