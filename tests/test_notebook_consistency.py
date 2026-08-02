"""Execution checks for the notebook cells changed by the reviewer revision."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
from IPython.core.interactiveshell import InteractiveShell

matplotlib.use("Agg")

NOTEBOOK = Path("tensor_network_HHL.ipynb")


def test_application_and_timing_cells_execute_in_order() -> None:
    """Catch stale names when application cells change but timing cells do not."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    shell = InteractiveShell()
    cell_indices = (
        4,
        9,
        12,
        15,
        18,
        19,
        22,
        24,
        25,
        28,
        30,
        31,
        34,
        36,
        37,
        69,
        70,
        71,
        72,
    )

    for index in cell_indices:
        source = "".join(notebook["cells"][index]["source"])
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="FigureCanvasAgg is non-interactive"
            )
            result = shell.run_cell(source, store_history=False)
        error = result.error_before_exec or result.error_in_exec
        assert error is None, f"notebook cell {index} failed: {error}"


def test_every_code_cell_compiles_and_stored_outputs_are_clear() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    for index, cell in enumerate(notebook["cells"]):
        source = "".join(cell.get("source", []))
        unexpected_controls = {
            character
            for character in source
            if ord(character) < 32 and character != "\n"
        }
        assert unexpected_controls == set(), (
            f"notebook cell {index} contains control characters: "
            f"{unexpected_controls!r}"
        )

        if cell["cell_type"] != "code":
            continue
        compile(source, f"notebook-cell-{index}", "exec")
        assert cell["execution_count"] is None
        assert cell["outputs"] == []


def test_small_qiskit_statevector_example_executes() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    shell = InteractiveShell()

    for index in (4, 42, 46, 49):
        result = shell.run_cell(
            "".join(notebook["cells"][index]["source"]), store_history=False
        )
        error = result.error_before_exec or result.error_in_exec
        assert error is None, f"notebook cell {index} failed: {error}"


def test_matrix_product_state_cells_are_inert_by_default() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    shell = InteractiveShell()

    for index in (50, 64, 65, 66):
        source = "".join(notebook["cells"][index]["source"])
        result = shell.run_cell(source, store_history=False)
        error = result.error_before_exec or result.error_in_exec
        assert error is None, f"notebook cell {index} failed: {error}"

    assert "backend_mps" not in shell.user_ns
