"""notebooks/pylmm_tour.ipynb: committed without outputs, and runs top to bottom.

The execution test needs the `notebooks` group and is skipped when nbclient/ipykernel are not
installed.  The executed copy stays in memory; nothing is written into the repository.
"""

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
NOTEBOOK = REPO / "notebooks" / "pylmm_tour.ipynb"


def test_tour_notebook_is_committed_clean():
    """No outputs, execution counts or per-cell execution timestamps in git (stdlib only)."""
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for i, cell in enumerate(nb["cells"]):
        assert not cell.get("outputs"), "cell %d has outputs" % i
        assert cell.get("execution_count") is None, "cell %d has an execution count" % i
        assert "execution" not in cell.get("metadata", {}), "cell %d has execution metadata" % i


def test_tour_notebook_executes():
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    nb = nbformat.read(NOTEBOOK, as_version=4)
    client = nbclient.NotebookClient(
        nb, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(REPO)}}
    )
    client.execute()
