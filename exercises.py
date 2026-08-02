"""Interactive exercise feedback for Marimo course notebooks."""

from __future__ import annotations

import ast
import inspect
import textwrap
import traceback
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Literal

import marimo as mo
from xarray import DataTree


@dataclass(frozen=True)
class ExerciseOutcome:
    """Result of attempting a learner-owned exercise function."""

    status: Literal["incomplete", "error", "success"]
    value: Any = None
    message: str = ""
    traceback_text: str = ""


def _has_ellipsis(function: Callable[[], Any]) -> bool:
    """Return whether the function source still contains an ellipsis scaffold."""
    try:
        source = textwrap.dedent(inspect.getsource(function))
    except (OSError, TypeError):
        return False
    tree = ast.parse(source)
    return any(
        isinstance(node, ast.Constant) and node.value is Ellipsis
        for node in ast.walk(tree)
    )


def run_exercise(function: Callable[[], Any]) -> ExerciseOutcome:
    """Run a learner function while preserving incomplete and error feedback."""
    if _has_ellipsis(function):
        return ExerciseOutcome(
            status="incomplete", message="Please complete the exercise."
        )
    try:
        return ExerciseOutcome(status="success", value=function())
    except NotImplementedError:
        return ExerciseOutcome(
            status="incomplete", message="Please complete the exercise."
        )
    except Exception as error:
        return ExerciseOutcome(
            status="error",
            message=f"{type(error).__name__}: {error}",
            traceback_text=traceback.format_exc(),
        )


def exercise(function: Callable[[], Any]) -> Callable[[], Any]:
    """Decorate a learner exercise with immediate Marimo feedback."""

    @wraps(function)
    def wrapped() -> Any:
        outcome = run_exercise(function)
        if outcome.status == "incomplete":
            return mo.callout(mo.md(outcome.message), kind="warn")
        if outcome.status == "error":
            return mo.callout(
                mo.md(f"**{outcome.message}**\n\n```text\n{outcome.traceback_text}\n```"),
                kind="danger",
            )
        if isinstance(outcome.value, DataTree):
            return mo.callout(
                mo.md(
                    "Exercise completed, but it returned a raw `DataTree`. "
                    "Return a plot, table, or `mo.vstack` with learner feedback instead."
                ),
                kind="warn",
            )
        success = mo.callout(mo.md("Exercise complete."), kind="success")
        if outcome.value is None:
            return success
        return mo.vstack([success, outcome.value])

    return wrapped
