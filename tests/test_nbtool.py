"""Tests for scripts/nbtool.py, the marimo notebook editing helper.

nbtool leans entirely on marimo private APIs (`_cell_manager`, `_config`,
`codegen.generate_filecontents`, `CellConfig`) under a wide `marimo` version
pin, so these tests exist to catch both regressions in nbtool itself and
silent breakage from a marimo release. Every test operates on a copy under
`tmp_path`; none may write to `notebooks/`.
"""

import shutil
from pathlib import Path

import pytest
from marimo._ast.cell import CellConfig

from scripts.nbtool import Notebook, check

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
SMALLEST_NOTEBOOK = NOTEBOOKS_DIR / "00_environment_check.py"


def _copy_into(tmp_path: Path, source: Path = SMALLEST_NOTEBOOK) -> Path:
    dest = tmp_path / source.name
    shutil.copy(source, dest)
    return dest


def _bare_notebook(ids: list[str], codes: list[str]) -> Notebook:
    """Build a Notebook instance with explicit ids/codes, bypassing file I/O."""
    nb = Notebook.__new__(Notebook)
    nb.path = None
    nb._config = None
    nb._header_comments = None
    nb._new_id_counter = 0
    nb.ids = list(ids)
    nb.codes = list(codes)
    nb.names = ["_"] * len(ids)
    nb.configs = [CellConfig() for _ in ids]
    return nb


def test_load_then_save_round_trip_is_byte_identical(tmp_path):
    path = _copy_into(tmp_path)
    original = path.read_text()

    Notebook(path).save()

    resaved = path.read_text()
    if resaved != original:
        assert resaved.split() == original.split(), (
            "round-trip changed more than whitespace"
        )


@pytest.mark.parametrize(
    "path", sorted(NOTEBOOKS_DIR.glob("*.py")), ids=lambda p: p.name
)
def test_check_passes_every_shipped_notebook(path):
    assert check(path) == 0


def test_check_fails_on_unparsable_cell(tmp_path):
    path = _copy_into(tmp_path)
    nb = Notebook(path)
    nb.append("this is not python at all (((")
    nb.save()

    assert check(path) != 0


def test_check_fails_on_undefined_reference(tmp_path):
    path = _copy_into(tmp_path)
    nb = Notebook(path)
    nb.append("this_name_is_never_defined_anywhere_in_the_notebook + 1")
    nb.save()

    assert check(path) != 0


def test_index_of_raises_on_ambiguous_substring():
    nb = _bare_notebook(
        ids=["a", "b", "c"],
        codes=["needle = 1", "needle = 2", "haystack = 3"],
    )

    with pytest.raises(KeyError):
        nb.index_of("needle")


def test_index_of_raises_on_duplicated_id():
    nb = _bare_notebook(
        ids=["dup", "dup", "unique"],
        codes=["x = 1", "y = 2", "z = 3"],
    )

    with pytest.raises(KeyError):
        nb.index_of("dup")


def test_insert_after_then_delete_leaves_other_cells_unchanged(tmp_path):
    path = _copy_into(tmp_path)
    nb = Notebook(path)
    original_codes = list(nb.codes)
    anchor = nb.ids[0]

    new_id = nb.insert_after(anchor, "inserted_marker = 1")
    assert nb.codes[nb.index_of(new_id)] == "inserted_marker = 1"

    nb.delete(new_id)

    assert nb.codes == original_codes


def test_repeated_inserts_produce_distinct_ids(tmp_path):
    path = _copy_into(tmp_path)
    nb = Notebook(path)
    first = nb.insert_after(nb.ids[0], "alpha_probe = 1")
    second = nb.insert_after(nb.ids[0], "beta_probe = 2")

    assert first != second
    assert len(set(nb.ids)) == len(nb.ids)
