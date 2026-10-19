"""Rewrite old pylmm code and command lines to the current API and CLI.

Driven by the repository's migration tables, ``docs/migration/name-map.csv``
(Python names) and ``docs/migration/cli-map.csv`` (command-line flags).

    python skills/migrate-from-old-pylmm/scripts/migrate.py old_script.py      # print the result
    python skills/migrate-from-old-pylmm/scripts/migrate.py -i old_script.py   # rewrite in place
    python skills/migrate-from-old-pylmm/scripts/migrate.py --command "pylmmGWAS -v --bfile x --kfile x.kin out.txt"

Files ending in ``.py`` are rewritten as Python; any other file (shell
scripts, Makefiles, notes) has its ``pylmmGWAS`` / ``pylmmKinship`` command
lines rewritten.  Notes about things that need a human look (changed return
types and defaults, removed options) go to stderr as ``NOTE file:line: ...``.

Python rewriting works on tokens, so strings and comments are left alone:

* functions and classes are renamed wherever the name appears;
* methods and attributes only after a ``.`` (``L.getMLSoln`` ->
  ``L.ml_solution``), so a local variable called ``LL`` survives;
* parameters only as keyword arguments of a call to the function or method
  that owns them (``LMM(Y, K, Kva=..)`` -> ``kva=``, ``gwas(..., REML=..)``
  -> ``reml=``); ``y=`` of another function is untouched;
* ``pylmm.input`` -> ``pylmm.plink_io`` (and the module alias ``input.``);
* ``obj.next()`` -> ``next(obj)``; ``kva=[]`` / ``kve=[]`` -> ``None``.

It cannot see types, so a method name shared with another library (``fit``
is not renamed, but e.g. ``getPhenos``) is renamed on any object.  Review
the diff.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import re
import shlex
import sys
import tokenize

# realpath, not abspath: the script is also reached through the .claude/skills -> ../skills
# symlink, and ../../.. must climb out of the real skills/ folder to find docs/.
HERE = os.path.dirname(os.path.realpath(__file__))
MAP_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "..", "docs", "migration"))

# Notes for patterns whose meaning changed without a name change.
PY_NOTES = [
    (
        re.compile(r"\[\s*0\s*,\s*0\s*\]|\[\s*0\s*\]\s*\[\s*0\s*\]"),
        "[0, 0] / [0][0] indexing: sigma, Q, beta_var and log-likelihoods are floats now "
        "(they were (1, 1) arrays); drop the indexing",
    ),
    (
        re.compile(r"\.log_likelihood\((?![^)]*reml)"),
        "log_likelihood() defaults to REML now (it defaulted to ML); pass reml=False for ML",
    ),
    (
        re.compile(r"\bgwas\([^)]*reml\s*=\s*False"),
        "gwas(reml=False) is a real ML scan now (null fit and test); it used to return the REML t",
    ),
    (
        re.compile(r"\bfit_two\("),
        "misc.fit_two: the phenotype parameter is Y (was y) and wgrids is ngrids",
    ),
    (
        re.compile(r"\b_get_max\("),
        "getMax is private now (_get_max) and may change without notice; call fit() (or "
        "fit(reml=False)) and read opt_h / opt_ll instead",
    ),
    (
        re.compile(r"\bTrue\s*-\s*(?:np|numpy)\.isnan\("),
        "Python 2 idiom True - np.isnan(...) (negation by subtraction) raises TypeError on "
        "current numpy: use ~np.isnan(...)",
    ),
    (
        re.compile(r"\.read_phenos\(|\.read_covariates\("),
        "read_phenos/read_covariates match rows by (FID, IID) and give nan rows for absent "
        "individuals; read_covariates has no intercept column (add np.ones yourself)",
    ),
]


def _read_csv(name: str, map_dir: str) -> list[dict[str, str]]:
    with open(os.path.join(map_dir, name), newline="") as f:
        return list(csv.DictReader(f))


def _owner_names(module: str) -> set[str]:
    """Callee names that own a parameter: pylmm.lmm.LMM.fit -> {fit}, LMM.__init__ -> {LMM}."""
    parts = module.split(".")
    if parts[-1] == "__init__":
        return {parts[-2]}
    return {parts[-1]}


class NameMap:
    """The Python renames of name-map.csv, chained (old -> final name) and split by context."""

    def __init__(self, map_dir: str = MAP_DIR) -> None:
        rows = _read_csv("name-map.csv", map_dir)
        self.anywhere: dict[str, str] = {}  # functions, classes
        self.attr: dict[str, str] = {}  # methods, attributes: only after "."
        self.param: dict[str, dict[str, str]] = {}  # old -> {owner: new}
        self.removed: dict[str, str] = {}
        for r in rows:
            old, new, kind = r["old_name"], r["new_name"], r["kind"]
            if kind in ("signature", "module", "type alias") or old == new:
                continue
            if not new:
                self.removed[old] = r["notes"]
                continue
            if kind in ("function", "class"):
                self.anywhere[old] = new
            elif kind in ("method", "attribute"):
                if old == "next":
                    continue  # handled as next(obj)
                self.attr[old] = new
            elif kind == "parameter":
                for owner in _owner_names(r["module"]):
                    self.param.setdefault(old, {})[owner] = new
        # a name renamed twice (5.4, then 5b.1): follow the chain to the final name
        for table in (self.anywhere, self.attr):
            for old in list(table):
                seen = {old}
                while table[old] in table and table[old] not in seen:
                    seen.add(table[old])
                    table[old] = table[table[old]]
        # parameters are owned by the NEW callee name; also accept the old one
        renamed_callees = {**self.anywhere, **self.attr}
        inverse = {}
        for old, new in renamed_callees.items():
            inverse.setdefault(new, set()).add(old)
        inverse.setdefault("PlinkReader", set()).add("plink")
        for owners in self.param.values():
            for owner, new in list(owners.items()):
                for alias in inverse.get(owner, ()):
                    owners.setdefault(alias, new)


def _prepass(src: str) -> str:
    src = re.sub(r"\b([A-Za-z_][\w.]*)\.next\(\)", r"next(\1)", src)
    src = re.sub(r"\b(kva|kve|Kva|Kve)(\s*=\s*)\[\s*\]", r"\1\2None", src)
    src = re.sub(r"\bpylmm\.input\b", "pylmm.plink_io", src)
    src = re.sub(
        r"(\bfrom[ \t]+pylmm[ \t]+import[ \t]+(?:[\w \t,]*?\b)?)input\b", r"\1plink_io", src
    )
    return src


def rewrite_python(src: str, names: NameMap) -> str:
    src = _prepass(src)
    imported_input_module = re.search(r"\bfrom\s+pylmm\s+import\b[^\n]*\bplink_io\b", src)
    toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    out = []
    callees: list[str | None] = []  # the callee of each open bracket
    prev_name: str | None = None
    for i, tok in enumerate(toks):
        t = tok
        if tok.type == tokenize.OP:
            if tok.string == "(":
                callees.append(prev_name)
            elif tok.string in "[{":
                callees.append(None)
            elif tok.string in ")]}" and callees:
                callees.pop()
        if tok.type == tokenize.NAME:
            prev = toks[i - 1].string if i else ""
            nxt = toks[i + 1].string if i + 1 < len(toks) else ""
            new = None
            if prev == "." and tok.string in names.attr:
                new = names.attr[tok.string]
            elif tok.string in names.anywhere:
                new = names.anywhere[tok.string]
            elif (
                nxt == "="
                and prev in ("(", ",")
                and callees
                and callees[-1] is not None
                and tok.string in names.param
            ):
                new = names.param[tok.string].get(callees[-1])
            elif tok.string == "input" and nxt == "." and imported_input_module and prev != ".":
                new = "plink_io"
            if new:
                t = tok._replace(string=new)
            prev_name = t.string
        elif not (tok.type == tokenize.OP and tok.string == "."):
            prev_name = None
        out.append(t)
    return tokenize.untokenize(out)


def python_notes(src: str) -> list[tuple[int, str]]:
    notes = []
    for lineno, line in enumerate(src.splitlines(), 1):
        for pat, msg in PY_NOTES:
            if pat.search(line):
                notes.append((lineno, msg))
    return notes


# ---------------------------------------------------------------------------
# command lines
# ---------------------------------------------------------------------------

OLD_PROGRAMS = {
    "pylmmGWAS": "pylmmGWAS",
    "pylmmGWAS.py": "pylmmGWAS",
    "scripts/pylmmGWAS.py": "pylmmGWAS",
    "pylmmKinship": "pylmmKinship",
    "pylmmKinship.py": "pylmmKinship",
    "scripts/pylmmKinship.py": "pylmmKinship",
}


class CliMap:
    def __init__(self, map_dir: str = MAP_DIR) -> None:
        self.flags: dict[str, dict[str, tuple[str, bool, str, str]]] = {}
        for r in _read_csv("cli-map.csv", map_dir):
            prog = r["old_command"]
            if prog not in ("pylmmGWAS", "pylmmKinship") or not r["old_flag"]:
                continue
            for flag in r["old_flag"].split("/"):
                self.flags.setdefault(prog, {})[flag] = (
                    r["new_flag"],
                    r["old_takes_value"] == "yes",
                    r["kind"],
                    r["notes"],
                )


def rewrite_command(cmd: str, cli: CliMap) -> tuple[str, list[str]]:
    """Translate one old command line; returns (new command, notes)."""
    words = shlex.split(cmd.replace("\\\n", " "))
    notes: list[str] = []
    # find the program: `pylmmGWAS ...`, `python scripts/pylmmGWAS.py ...`, `python -m pylmm.cli.gwas`
    start, prog = None, None
    for i, w in enumerate(words):
        base = w if w in OLD_PROGRAMS else os.path.basename(w)
        if w in OLD_PROGRAMS or base in OLD_PROGRAMS:
            start, prog = i, OLD_PROGRAMS.get(w, OLD_PROGRAMS.get(base))
            break
        if (
            w == "-m"
            and i + 1 < len(words)
            and words[i + 1] in ("pylmm.cli.gwas", "pylmm.cli.kinship")
        ):
            start, prog = i + 1, "pylmmGWAS" if words[i + 1].endswith("gwas") else "pylmmKinship"
            break
    if prog is None:
        return cmd, notes
    head = words[:start]  # e.g. `uv run python -m` or `python`: the new program needs neither
    while head and (head[-1] == "-m" or os.path.basename(head[-1]).startswith("python")):
        head.pop()
    flags = cli.flags[prog]
    out = [*head, "pylmm", "gwas" if prog == "pylmmGWAS" else "kinship"]
    positional = []
    args = words[start + 1 :]
    i = 0
    while i < len(args):
        a = args[i]
        value = None
        if a.startswith("--") and "=" in a:
            a, value = a.split("=", 1)
        if not a.startswith("-") or a == "-":
            positional.append(a)
            i += 1
            continue
        if a not in flags:
            notes.append("unknown old flag %s: left as is" % a)
            out.append(args[i])
            i += 1
            continue
        new_flag, takes_value, kind, note = flags[a]
        if takes_value and value is None:
            i += 1
            value = args[i] if i < len(args) else ""
        i += 1
        if a in ("--refit", "--removeMissingGenotypes"):
            notes.append(
                "%s: each SNP's fit is now by REML, like the null fit (the old program refit by "
                "ML), so the output changes; see CHANGELOG.md" % a
            )
        if a == "-p":
            out += ["--pheno-col", str(int(value) + 1)]
            notes.append("-p %s (0-based) -> --pheno-col %d (1-based)" % (value, int(value) + 1))
        elif kind in ("removed", "removed-earlier") or not new_flag:
            notes.append("%s removed: %s" % (a, note))
        elif kind == "value→flag":
            out.append(new_flag.split("/")[-1])
        else:
            forms = new_flag.split("/")  # e.g. "-v/--verbose": keep the old spelling if it is one
            out.append(a if a in forms else forms[-1])
            if takes_value:
                out.append(value)
    if positional:
        out += ["-o", positional[-1]]
        if len(positional) > 1:
            notes.append("extra positional arguments dropped: %s" % " ".join(positional[:-1]))
    else:
        notes.append("no output file found; pylmm %s needs -o FILE" % out[len(head) + 1])
    return shlex.join(out), notes


def rewrite_text(text: str, cli: CliMap) -> tuple[str, list[tuple[int, str]]]:
    """Rewrite every old command line of a shell script or text file."""
    lines = text.split("\n")
    out, notes = [], []
    i = 0
    while i < len(lines):
        start = i
        logical = lines[i]
        while logical.endswith("\\") and i + 1 < len(lines):
            i += 1
            logical = logical[:-1] + " " + lines[i].lstrip()
        i += 1
        if re.search(r"pylmmGWAS|pylmmKinship|pylmm\.cli\.(gwas|kinship)", logical) and not (
            logical.lstrip().startswith("#")
        ):
            indent = logical[: len(logical) - len(logical.lstrip())]
            new, ns = rewrite_command(logical.strip(), cli)
            out.append(indent + new)
            notes += [(start + 1, n) for n in ns]
        else:
            out.extend(lines[start:i])
    return "\n".join(out), notes


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("paths", nargs="*", help=".py files (Python) or other files (command lines)")
    p.add_argument("-i", "--in-place", action="store_true", help="rewrite the files in place")
    p.add_argument("--command", help="translate one old command line and print it")
    p.add_argument("--map-dir", default=MAP_DIR, help="directory of name-map.csv and cli-map.csv")
    args = p.parse_args(argv)
    cli = CliMap(args.map_dir)
    if args.command:
        new, notes = rewrite_command(args.command, cli)
        print(new)
        for n in notes:
            print("NOTE: %s" % n, file=sys.stderr)
        return 0
    if not args.paths:
        p.error("give files to rewrite, or --command")
    names = NameMap(args.map_dir)
    for path in args.paths:
        with open(path) as f:
            src = f.read()
        if path.endswith(".py"):
            new = rewrite_python(src, names)
            notes = python_notes(new)
        else:
            new, notes = rewrite_text(src, cli)
        for lineno, msg in notes:
            print("NOTE %s:%d: %s" % (path, lineno, msg), file=sys.stderr)
        if args.in_place:
            with open(path, "w") as f:
                f.write(new)
        else:
            sys.stdout.write(new)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
