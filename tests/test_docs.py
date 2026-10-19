"""Executable documentation: run every code block of AGENTS.md, docs/guide.md and skills/.

The docs promise that their snippets run; this test keeps that promise.

What is not executed: ``README.md`` and ``docs/research/*.md`` (and the other
reports under ``docs/``).  The README's example commands scan the full
bundled dataset (minutes) and are covered instead by ``just readme-example``
and the CLI tests that compare the same command with the Phase 1 goldens
(tests/test_cli_gwas.py); the research notes are background surveys of other
tools' formats and methods, with sketches of external APIs, not instructions
that an agent or user is meant to run.

How blocks are run
------------------
* Each Markdown file is one session.  Its blocks run IN ORDER, sharing one
  Python namespace (``python`` blocks) and one working directory (all
  blocks), so a later block may use the variables and files of an earlier
  one in the same file.  Files do not share anything.
* The working directory is a fresh temporary directory that holds symlinks
  to the repository's ``data/``, ``docs/``, ``sims/`` and ``skills/``, so
  repo-relative paths such as ``data/snps.132k.clean.noX`` work as they do
  from a checkout, and every output file goes to the temporary directory.
* ``python`` / ``py`` blocks: ``exec`` in the file's namespace.
* ``bash`` / ``sh`` / ``shell`` blocks: ``bash -euo pipefail``, in the same
  directory.  The first command must start with ``pylmm`` or ``uv run``;
  any other shell block must be marked ``skip-ci`` (below).  ``uv`` is a shim
  that drops ``run`` and its options and runs the command in this test's
  environment (so ``uv run --group sims python -m sims ...`` uses the
  already-installed interpreter); ``pylmm`` is this environment's console
  script.  ``SIMS_RESULTS_DIR`` points into the temporary directory.
* Known non-code languages are not run: ``text``, ``txt``, ``csv``, ``tsv``,
  ``yaml``, ``yml``, ``toml``, ``json``, ``ini``, ``console-output``,
  ``diff``, ``markdown``, ``md`` (`NON_CODE_LANGS`).  Any other language
  (a typo such as ``pyhton``, or ``python3``, ``zsh``, ``console``) is an
  error unless the block is marked ``skip-ci``, so a block is never silently
  not run.  A fence without a language is an error: write ``text`` for output.

Fences
------
Fences are three or more backticks or tildes (CommonMark).  A block closes at
a line of the same character, at least as long as its opening, with nothing
after it, so a four-backtick block can show a three-backtick fence as text.
A line inside a block that opens a fence of the same kind (same character, at
least as long, with a language) is an error: it is almost always a block
that was never closed.

Opting out
----------
The info string of a block can carry (after the language):

* ``skip-ci`` -- never run, e.g. it needs plink2, bcftools, a network
  download or the full test suite.  Write the reason after it, as in
  ```` ```bash skip-ci: needs plink2 ````.  Skips are counted and listed in
  the pytest summary.
* ``needs=mod1,mod2`` -- run only if these Python modules import (e.g.
  ``needs=pandas`` for blocks that use the ``sims`` dependency group);
  otherwise skipped (counted) with the reason "module X not installed".

Fences must start in column 0 (an indented fence is an error, since it would
silently not be run).

Statistics (blocks executed and skipped, per file) are kept in the pytest
config stash (`tests/conftest.py`'s ``docs_stats`` fixture) and printed in the
terminal summary, whatever the import mode.
"""

from __future__ import annotations

import glob
import importlib.util
import os
import re
import stat
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DOC_FILES = sorted(
    [
        os.path.join(REPO, "AGENTS.md"),
        os.path.join(REPO, "docs", "guide.md"),
        *glob.glob(os.path.join(REPO, "skills", "**", "*.md"), recursive=True),
    ]
)
LINKED_DIRS = ("data", "docs", "sims", "skills")
PY_LANGS = {"python", "py"}
SH_LANGS = {"bash", "sh", "shell"}
NON_CODE_LANGS = {
    "text",
    "txt",
    "csv",
    "tsv",
    "yaml",
    "yml",
    "toml",
    "json",
    "ini",
    "console-output",
    "diff",
    "markdown",
    "md",
}
SH_PREFIXES = ("pylmm ", "uv run ")

# An opening or closing fence: 3+ backticks or 3+ tildes, then the info string.
_FENCE = re.compile(r"^(`{3,}|~{3,})(.*)$")
_INDENTED_FENCE = re.compile(r"^[ \t]+(`{3,}|~{3,})")


@dataclass
class Block:
    path: str
    line: int  # 1-based line of the opening fence
    lang: str
    flags: str
    code: str

    @property
    def where(self) -> str:
        return "%s:%d" % (os.path.relpath(self.path, REPO), self.line)


def parse_blocks(path: str) -> list[Block]:
    """The fenced code blocks of a Markdown file, in order."""
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    rel = os.path.relpath(path, REPO)
    blocks = []
    i = 0
    while i < len(lines):
        if _INDENTED_FENCE.match(lines[i]):
            raise AssertionError(
                "%s:%d: indented code fence; start fences in column 0 so the docs test sees them"
                % (rel, i + 1)
            )
        m = _FENCE.match(lines[i])
        if not m:
            i += 1
            continue
        fence, info = m.group(1), m.group(2).strip()
        if fence[0] == "`" and "`" in info:
            i += 1  # inline code such as ```x``` on its own line, not a fence
            continue
        start = i
        i += 1
        body = []
        while i < len(lines):
            inner = _FENCE.match(lines[i])
            if inner and inner.group(1)[0] == fence[0] and len(inner.group(1)) >= len(fence):
                if not inner.group(2).strip():
                    break  # the closing fence
                raise AssertionError(
                    "%s:%d: a fence inside a block (unclosed block opened at line %d?)"
                    % (rel, i + 1, start + 1)
                )
            if _INDENTED_FENCE.match(lines[i]) and lines[i].lstrip().startswith(fence):
                raise AssertionError(
                    "%s:%d: an indented fence inside a block (unclosed block opened at line %d?)"
                    % (rel, i + 1, start + 1)
                )
            body.append(lines[i])
            i += 1
        if i == len(lines):
            raise AssertionError("%s:%d: unclosed code block" % (rel, start + 1))
        i += 1
        lang, _, flags = info.partition(" ")
        blocks.append(Block(path, start + 1, lang.lower(), flags.strip(), "\n".join(body) + "\n"))
    return blocks


def skip_reason(block: Block) -> str | None:
    """Why the block is not run, or None to run it."""
    m = re.search(r"\bskip-ci\b:?\s*(.*)", block.flags)
    if m:
        return m.group(1).strip() or "skip-ci"
    m = re.search(r"\bneeds=(\S+)", block.flags)
    if m:
        for mod in m.group(1).split(","):
            if importlib.util.find_spec(mod) is None:
                return "module %s not installed" % mod
    return None


def _first_command(code: str) -> str:
    for line in code.splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            return s
    return ""


_UV_SHIM = '''\
#!{python}
"""`uv run [options] CMD ...` -> CMD ... in the docs test's environment."""
import os, sys
args = sys.argv[1:]
if not args or args[0] != "run":
    sys.exit("docs-test uv shim: only `uv run` is supported, got %r" % args)
args = args[1:]
takes_value = {{"--group", "--with", "--python", "-p", "--extra", "--from", "--project",
               "--directory", "--env-file", "--resolution"}}
while args and args[0].startswith("-"):
    a = args.pop(0)
    if a in takes_value:
        args.pop(0)
if not args:
    sys.exit("docs-test uv shim: no command after `uv run`")
if args[0] == "python":
    args[0] = {python!r}
os.execvp(args[0], args)
'''


def _session_dir(tmp_path) -> tuple[str, dict]:
    """The working directory (with repo symlinks) and the environment of a doc session."""
    work = tmp_path / "work"
    work.mkdir()
    for d in LINKED_DIRS:
        os.symlink(os.path.join(REPO, d), work / d)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    shim = bindir / "uv"
    shim.write_text(_UV_SHIM.format(python=sys.executable))
    shim.chmod(shim.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join(
        [str(bindir), os.path.dirname(sys.executable), env.get("PATH", "")]
    )
    env["PYTHONPATH"] = os.pathsep.join([REPO, env.get("PYTHONPATH", "")])
    env["SIMS_RESULTS_DIR"] = str(tmp_path / "sims-results")
    env["MPLBACKEND"] = "Agg"
    return str(work), env


def run_doc(path: str, tmp_path, monkeypatch) -> dict:
    blocks = parse_blocks(path)
    work, env = _session_dir(tmp_path)
    monkeypatch.chdir(work)
    for k in ("SIMS_RESULTS_DIR", "MPLBACKEND"):
        monkeypatch.setenv(k, env[k])
    namespace: dict = {"__name__": "__docs__"}
    stats: dict = {"executed": 0, "skipped": [], "seconds": 0.0}
    begin = time.time()
    for b in blocks:
        if not b.lang:
            raise AssertionError(
                "%s: code block without a language; use ```text for output or ```python/```bash"
                % b.where
            )
        if b.lang in NON_CODE_LANGS:
            continue
        reason = skip_reason(b)
        if reason is not None:
            stats["skipped"].append((b.where, reason))
            continue
        if b.lang not in PY_LANGS and b.lang not in SH_LANGS:
            raise AssertionError(
                "%s: unknown code block language %r; use python/py, bash/sh/shell, one of the "
                "non-code languages (%s), or mark the block skip-ci: <reason>"
                % (b.where, b.lang, ", ".join(sorted(NON_CODE_LANGS)))
            )
        if b.lang in PY_LANGS:
            try:
                exec(compile(b.code, b.where, "exec"), namespace)
            except Exception as e:
                raise AssertionError(
                    "%s: python block failed: %s: %s\n%s"
                    % (b.where, type(e).__name__, e, textwrap.indent(b.code, "    "))
                ) from e
        else:
            first = _first_command(b.code)
            if not first.startswith(SH_PREFIXES):
                raise AssertionError(
                    "%s: a shell block must start with `pylmm` or `uv run`, or be marked "
                    "`skip-ci: <reason>`; it starts with %r" % (b.where, first)
                )
            p = subprocess.run(
                ["bash", "-euo", "pipefail", "-c", b.code],
                cwd=work,
                env=env,
                capture_output=True,
                text=True,
                timeout=600,
            )
            if p.returncode != 0:
                raise AssertionError(
                    "%s: shell block exited %d\n%s\n--- stdout ---\n%s\n--- stderr ---\n%s"
                    % (b.where, p.returncode, textwrap.indent(b.code, "    "), p.stdout, p.stderr)
                )
        stats["executed"] += 1
    stats["seconds"] = time.time() - begin
    return stats


@pytest.mark.parametrize("path", DOC_FILES, ids=lambda p: os.path.relpath(p, REPO))
def test_doc_snippets_run(path, tmp_path, monkeypatch, docs_stats):
    docs_stats[os.path.relpath(path, REPO)] = run_doc(path, tmp_path, monkeypatch)


def test_doc_files_exist():
    assert os.path.isfile(os.path.join(REPO, "AGENTS.md"))
    assert os.path.isfile(os.path.join(REPO, "docs", "guide.md"))
    assert glob.glob(os.path.join(REPO, "skills", "*", "SKILL.md"))


_NAME = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


@pytest.mark.parametrize(
    "skill_md",
    sorted(glob.glob(os.path.join(REPO, "skills", "*", "SKILL.md"))),
    ids=lambda p: os.path.basename(os.path.dirname(p)),
)
def test_skill_frontmatter(skill_md):
    """Agent Skills format: YAML frontmatter with `name` (= the folder) and `description`."""
    with open(skill_md, encoding="utf-8") as f:
        text = f.read()
    assert text.startswith("---\n"), "%s: no YAML frontmatter" % skill_md
    front, sep, _body = text[4:].partition("\n---\n")
    assert sep, "%s: unterminated frontmatter" % skill_md
    fields = {}
    for line in front.splitlines():
        key, colon, value = line.partition(":")
        assert colon and key.strip(), "%s: frontmatter line %r is not `key: value`" % (
            skill_md,
            line,
        )
        fields[key.strip()] = value.strip()
    folder = os.path.basename(os.path.dirname(skill_md))
    assert fields.get("name") == folder, "name must equal the folder name %r" % folder
    assert _NAME.match(folder) and len(folder) <= 64
    desc = fields.get("description", "")
    assert 50 <= len(desc) <= 1024, "description must be 50-1024 characters"
    assert "<" not in desc and ">" not in desc
    # keep the values plain YAML scalars (no PyYAML here to parse anything fancier)
    for key, value in fields.items():
        assert ": " not in value and " #" not in value, "%s: quote-free plain value needed" % key
        assert not value.startswith(("'", '"', "[", "{", "|", ">")), key


def test_claude_skills_symlink():
    """`.claude/skills` is a relative symlink to `../skills`, so Claude Code finds the skills."""
    link = os.path.join(REPO, ".claude", "skills")
    assert os.path.islink(link)
    assert os.readlink(link) == "../skills"
    assert os.path.samefile(link, os.path.join(REPO, "skills"))


# -- the parser and the language rules themselves --------------------------------


def _doc(tmp_path, text: str) -> str:
    path = tmp_path / "doc.md"
    path.write_text(text)
    return str(path)


def test_parse_tilde_and_long_backtick_fences(tmp_path):
    text = "~~~python\nx = 1\n~~~\n````markdown\n```python\nnot run\n```\n````\n```text\n~~~\n```\n"
    blocks = parse_blocks(_doc(tmp_path, text))
    assert [(b.lang, b.code) for b in blocks] == [
        ("python", "x = 1\n"),
        ("markdown", "```python\nnot run\n```\n"),
        ("text", "~~~\n"),
    ]


@pytest.mark.parametrize(
    "text",
    [
        "```python\nx = 1\n```python\ny = 2\n```\n",  # an unclosed block
        "````python\nx = 1\n```\n",  # a closing fence shorter than the opening one
        "  ```python\nx = 1\n```\n",  # indented
        "~~~python\nx = 1\n  ~~~\n",  # indented closing fence
    ],
)
def test_parse_rejects_bad_fences(tmp_path, text):
    with pytest.raises(AssertionError, match=r"fence|unclosed"):
        parse_blocks(_doc(tmp_path, text))


@pytest.mark.parametrize("lang", ["pyhton", "python3", "zsh", "console"])
def test_unknown_language_is_an_error(tmp_path, monkeypatch, lang):
    path = _doc(tmp_path, "```%s\nprint(1)\n```\n" % lang)
    with pytest.raises(AssertionError, match="unknown code block language %r" % lang):
        run_doc(path, tmp_path, monkeypatch)


def test_non_code_and_skipped_blocks(tmp_path, monkeypatch):
    text = "".join("```%s\nanything\n```\n" % lang for lang in sorted(NON_CODE_LANGS))
    text += "```zsh skip-ci: needs zsh\nsetopt x\n```\n```python\nok = 1\n```\n"
    stats = run_doc(_doc(tmp_path, text), tmp_path, monkeypatch)
    assert stats["executed"] == 1
    assert [reason for _, reason in stats["skipped"]] == ["needs zsh"]
