"""Execute notebooks/pylmm_tour.ipynb top to bottom (needs the `notebooks` group).

Skipped when nbclient/ipykernel are not installed.  The executed copy stays in
memory; nothing is written into the repository.
"""

from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")
pytest.importorskip("ipykernel")

REPO = Path(__file__).resolve().parent.parent
NOTEBOOK = REPO / "notebooks" / "pylmm_tour.ipynb"


def test_tour_notebook_executes():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    client = nbclient.NotebookClient(
        nb, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(REPO)}}
    )
    client.execute()
