"""Edit marimo notebooks from a script, without a running kernel.

marimo notebooks store each cell as a decorated function whose signature lists
the globals the cell reads and whose return tuple lists what it defines. Hand
editing those is error prone; this module rewrites the file with marimo's own
code generator, so signatures and return tuples are always recomputed for you.

Only use this when no marimo session is running against the file — a live
kernel overwrites file edits on its next save.

Usage:

    pixi run python scripts/nbtool.py list notebooks/01_foundations.py
    pixi run python scripts/nbtool.py check notebooks/01_foundations.py

or from Python:

    from scripts.nbtool import Notebook

    nb = Notebook("notebooks/01_foundations.py")
    nb.replace("Kclp", "x = 1")
    nb.insert_after("Kclp", "y = x + 1", hide_code=True)
    nb.delete("oldcell")
    nb.save()
"""

from __future__ import annotations

import builtins
import re
import sys
from pathlib import Path

from marimo._ast import codegen
from marimo._ast.cell import CellConfig
from marimo._ast.load import load_app


class Notebook:
    """A marimo notebook opened for structural editing."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        app = load_app(self.path)
        data = list(app._cell_manager.cell_data())
        self._config = app._config
        self._header_comments = codegen.get_header_comments(self.path)
        self._new_id_counter = 0
        self.ids = [cd.cell_id for cd in data]
        self.codes = [cd.code or "" for cd in data]
        self.names = [cd.name for cd in data]
        self.configs = [cd.config for cd in data]

    def _next_id(self) -> str:
        self._new_id_counter += 1
        return f"new{self._new_id_counter}"

    def index_of(self, target: str) -> int:
        """Index of a cell by id, or by a unique substring of its code."""
        id_matches = [i for i, cell_id in enumerate(self.ids) if cell_id == target]
        if id_matches:
            if len(id_matches) > 1:
                raise KeyError(f"{target!r} id is duplicated at indices {id_matches}")
            return id_matches[0]
        matches = [i for i, code in enumerate(self.codes) if target in code]
        if not matches:
            raise KeyError(f"no cell matches {target!r}")
        if len(matches) > 1:
            raise KeyError(
                f"{target!r} matches {len(matches)} cells: {[self.ids[i] for i in matches]}"
            )
        return matches[0]

    def code(self, target: str) -> str:
        return self.codes[self.index_of(target)]

    def replace(self, target: str, code: str) -> None:
        self.codes[self.index_of(target)] = code

    def set_hide_code(self, target: str, hide: bool) -> None:
        i = self.index_of(target)
        old = self.configs[i]
        self.configs[i] = CellConfig(
            column=old.column, disabled=old.disabled, hide_code=hide
        )

    def insert_after(self, target: str, code: str, *, hide_code: bool = True) -> str:
        i = self.index_of(target) + 1
        new_id = self._next_id()
        self.codes.insert(i, code)
        self.names.insert(i, "_")
        self.configs.insert(i, CellConfig(hide_code=hide_code))
        self.ids.insert(i, new_id)
        return new_id

    def insert_before(self, target: str, code: str, *, hide_code: bool = True) -> str:
        i = self.index_of(target)
        new_id = self._next_id()
        self.codes.insert(i, code)
        self.names.insert(i, "_")
        self.configs.insert(i, CellConfig(hide_code=hide_code))
        self.ids.insert(i, new_id)
        return new_id

    def append(self, code: str, *, hide_code: bool = True) -> str:
        new_id = self._next_id()
        self.codes.append(code)
        self.names.append("_")
        self.configs.append(CellConfig(hide_code=hide_code))
        self.ids.append(new_id)
        return new_id

    def delete(self, target: str) -> None:
        i = self.index_of(target)
        for seq in (self.ids, self.codes, self.names, self.configs):
            del seq[i]

    def save(self) -> None:
        self.path.write_text(
            codegen.generate_filecontents(
                self.codes,
                self.names,
                self.configs,
                config=self._config,
                header_comments=self._header_comments,
            )
        )


def describe(path: str | Path) -> None:
    """Print one line per cell: index, id, hidden flag, heading, first line."""
    nb = Notebook(path)
    for i, (cell_id, code, config) in enumerate(zip(nb.ids, nb.codes, nb.configs)):
        stripped = code.strip()
        heading = re.search(r"^#+ .*$", stripped, re.M)
        first = stripped.splitlines()[0][:58] if stripped else ""
        flag = "hidden" if config.hide_code else "SHOWN "
        print(
            f"{i:3d} {cell_id} {flag} | {first} | {heading.group(0)[:56] if heading else ''}"
        )


# Names marimo/Python inject into every cell's namespace without a defining
# cell; flagging these as unresolved would be a false positive.
_IMPLICIT_GLOBALS = {"__file__", "__name__", "__doc__"}


def check(path: str | Path) -> int:
    """Verify the notebook forms a valid graph. Returns a process exit code."""
    app = load_app(Path(path))
    cells = list(app._cell_manager.cell_data())
    defs: dict[str, list[str]] = {}
    for cd in cells:
        for name in cd.cell.defs if cd.cell else ():
            defs.setdefault(name, []).append(cd.cell_id)
    unresolved = [
        (cd.cell_id, ref)
        for cd in cells
        if cd.cell
        for ref in cd.cell.refs
        if ref not in defs
        and ref not in _IMPLICIT_GLOBALS
        and not hasattr(builtins, ref)
    ]
    duplicated = {k: v for k, v in defs.items() if len(v) > 1}
    # marimo serializes a cell it cannot parse as app._unparsable_cell(...);
    # such cells have cd.cell is None, so the checks above silently skip them.
    unparsable = [cd.cell_id for cd in cells if cd.cell is None]
    print(f"cells: {len(cells)}")
    print(f"unresolved refs: {unresolved}")
    print(f"multiply defined: {duplicated}")
    print(f"unparsable cells: {unparsable}")
    return 1 if unresolved or duplicated or unparsable else 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(
            f"usage: {sys.argv[0]} <list|check> <path>\n"
            "  list  <path>  print one line per cell\n"
            "  check <path>  validate the notebook's cell graph"
        )
    command, target = sys.argv[1], sys.argv[2]
    if command == "list":
        describe(target)
    elif command == "check":
        raise SystemExit(check(target))
    else:
        raise SystemExit(f"unknown command {command!r}; expected 'list' or 'check'")
