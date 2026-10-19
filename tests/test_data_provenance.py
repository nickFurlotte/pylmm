"""data/README.md checksums, the HMDP marker map and scripts/build_hmdp_marker_map.py.

The script's download step needs the network and is not run here; its join and checks run on
small synthetic files, and the committed map is checked against the bundled data.
"""

from __future__ import annotations

import doctest
import gzip
import hashlib
import os
import re
import subprocess
import sys

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(REPO, "data")
sys.path.insert(0, os.path.join(REPO, "scripts"))
import build_hmdp_marker_map as bmm  # noqa: E402

MAP = os.path.join(DATA, "hmdp.liver.snps.map.tsv")
HAVE_DATA = os.path.isfile(os.path.join(DATA, "README.md")) and os.path.isfile(MAP)
needs_data = pytest.mark.skipif(
    not HAVE_DATA, reason="the bundled data files are absent (moved out of the repository)"
)


def _readme_checksums() -> dict[str, str]:
    if not os.path.isfile(os.path.join(DATA, "README.md")):
        return {}
    with open(os.path.join(DATA, "README.md"), encoding="utf-8") as f:
        text = f.read()
    pairs = re.findall(r"^([0-9a-f]{64})  (\S+)$", text, flags=re.MULTILINE)
    return {name: digest for digest, name in pairs}


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _tracked_data_files() -> list[str] | None:
    """Files tracked by git under data/ (relative to data/), or None if git is unavailable."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z", "--", "data/"],
            cwd=REPO,
            capture_output=True,
            check=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return sorted(p[len("data/") :] for p in out.split("\0") if p)


@needs_data
def test_readme_lists_every_tracked_data_file():
    tracked = _tracked_data_files()
    if tracked is None:
        pytest.skip("git is not available or this is not a git checkout")
    listed = _readme_checksums()
    # untracked local files in data/ are ignored; every tracked one must be listed
    expected = sorted(f for f in tracked if f != "README.md" and not f.startswith("."))
    missing = [f for f in expected if f not in listed]
    assert not missing, "tracked but not listed in data/README.md: %s" % missing
    # everything listed exists and matches its hash
    for name, digest in listed.items():
        path = os.path.join(DATA, name)
        assert os.path.isfile(path), "%s is listed in data/README.md but absent" % name
        assert _sha256(path) == digest, "%s does not match its README hash" % name


@pytest.mark.parametrize("name", sorted(_readme_checksums()))
@needs_data
def test_data_checksums(name):
    assert _sha256(os.path.join(DATA, name)) == _readme_checksums()[name]


@needs_data
def test_marker_map_matches_the_bundled_data():
    with open(MAP) as f:
        header = f.readline().rstrip("\n").split("\t")
        rows = [line.rstrip("\n").split("\t") for line in f]
    assert tuple(header) == bmm.COLUMNS
    with open(os.path.join(DATA, "hmdp.liver.snps")) as f:
        n_emma = sum(1 for line in f if line.strip())
    assert len(rows) == n_emma == 132285
    assert [int(r[0]) for r in rows] == list(range(1, n_emma + 1))
    status = {}
    for r in rows:
        status[r[5]] = status.get(r[5], 0) + 1
        assert r[3] in {".", "A", "C", "G", "T"} and r[4] in {".", "A", "C", "G", "T"}
        if r[5] == "concordant":
            assert "." not in (r[3], r[4]) and r[3] != r[4]
    assert status == {"concordant": 122242, "uninformative": 10043}
    # every SNP of the bundled BED (IDs mm37-<chr>-<bp>) is at a mapped build-37 position
    pos = {(r[1], r[2]) for r in rows}
    with open(os.path.join(DATA, "snps.132k.clean.noX.bim")) as f:
        bim = [line.split() for line in f]
    assert all((b[0], b[3]) in pos and b[1] == "mm37-%s-%s" % (b[0], b[3]) for b in bim)


def test_script_doctests():
    assert doctest.testmod(bmm).failed == 0


def _broad(tmp_path, strains, rows):
    """A tiny Broad-style file: 2 header lines, then mm37-<chr>-<bp>, oxford name, calls."""
    lines = [
        "\t\t" + "\t".join(strains),
        "build37_loc\toxford_name\t" + "\t".join("CI" for _ in strains),
    ]
    for loc, calls in rows:
        lines.append("%s\t\t%s" % (loc, "\t".join("%s:0.01" % c for c in calls)))
    path = tmp_path / "broad.txt.gz"
    with gzip.open(path, "wt") as f:
        f.write("\n".join(lines) + "\n")
    return str(path)


@pytest.fixture
def tiny(tmp_path):
    # EMMA strains: S1, S2, S2 (a repeated label, identical columns), S3 (not in Broad)
    (tmp_path / "strains").write_text('"S1"\n"S2"\n"S2"\n"S3"\n')
    (tmp_path / "emma").write_text("0 1 1 0\n1 1 1 NA\n0 0 0 1\n")
    broad = _broad(
        tmp_path,
        ["S0", "S1", "S2"],
        [("mm37-1-100", "GAT"), ("mm37-1-250", "CCC"), ("mm37-X-7", "TTNoCall")],
    )
    return tmp_path, broad


def test_build_map_on_tiny_files(tiny):
    tmp_path, broad = tiny
    out = tmp_path / "map.tsv"
    args = ["--broad", broad, "--no-checksum", "--emma", str(tmp_path / "emma")]
    assert bmm.main([*args, "--strains", str(tmp_path / "strains"), "--out", str(out)]) == 0
    assert out.read_text().splitlines() == [
        "row\tchr\tbp_b37\tallele0\tallele1\tstatus",
        "1\t1\t100\tA\tT\tconcordant",
        "2\t1\t250\t.\tC\tuninformative",
        "3\tX\t7\tT\t.\tuninformative",
    ]


def test_build_map_rejects_discordant_rows_and_row_count_mismatch(tiny):
    tmp_path, broad = tiny
    (tmp_path / "emma").write_text("0 0 0 0\n1 1 1 NA\n0 0 0 1\n")  # row 1: code 0 is A and T
    out = tmp_path / "map.tsv"
    args = ["--broad", broad, "--no-checksum", "--emma", str(tmp_path / "emma")]
    assert bmm.main([*args, "--strains", str(tmp_path / "strains"), "--out", str(out)]) == 1
    assert not out.exists()
    (tmp_path / "emma").write_text("0 1 1 0\n1 1 1 NA\n")
    assert bmm.main([*args, "--strains", str(tmp_path / "strains"), "--out", str(out)]) == 1
    assert not out.exists()


@pytest.mark.parametrize(
    ("emma", "message"),
    [
        ("0 1 1 0\n1 1 1 NA\n0 0 0 1\n1 1 1 1\n", "after 3 rows: the EMMA file has more"),
        ("0 1 1 0\n1 1 1 NA\n0 0 0 1\n1 1 1 1\n0 0 0 0\n", "after 3 rows: the EMMA file has more"),
        ("0 1 1 0\n1 1 1 NA\n", "after 2 rows: the Broad file has more"),
        ("0 0 0 0\n1 1 1 NA\n0 0 0 1\n", "1 discordant rows"),
    ],
    ids=["emma+1", "emma+2", "broad+1", "discordant"],
)
def test_main_refuses_bad_inputs_on_stderr(tiny, capsys, emma, message):
    tmp_path, broad = tiny
    (tmp_path / "emma").write_text(emma)
    out = tmp_path / "map.tsv"
    args = ["--broad", broad, "--no-checksum", "--emma", str(tmp_path / "emma")]
    code = bmm.main([*args, "--strains", str(tmp_path / "strains"), "--out", str(out)])
    captured = capsys.readouterr()
    assert code == 1
    assert not out.exists()
    assert message in captured.err
    assert captured.out == ""


def test_build_map_checks_the_checksum(tiny, capsys):
    tmp_path, broad = tiny
    args = [
        "--broad",
        broad,
        "--emma",
        str(tmp_path / "emma"),
        "--strains",
        str(tmp_path / "strains"),
    ]
    assert bmm.main([*args, "--out", str(tmp_path / "map.tsv")]) == 1
    assert "expected %s" % bmm.BROAD_SHA256 in capsys.readouterr().err
