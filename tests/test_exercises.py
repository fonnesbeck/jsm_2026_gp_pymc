import pytest

from exercises import exercise, run_exercise


def test_run_exercise_reports_ellipsis_without_calling_function():
    called = False

    def incomplete():
        nonlocal called
        called = True
        ...

    outcome = run_exercise(incomplete)

    assert outcome.status == "incomplete"
    assert outcome.message == "Please complete the exercise."
    assert called is False


def test_run_exercise_exposes_runtime_error_traceback():
    def broken():
        raise ValueError("basis and outcome dimensions differ")

    outcome = run_exercise(broken)

    assert outcome.status == "error"
    assert "ValueError: basis and outcome dimensions differ" in outcome.traceback_text


def test_run_exercise_preserves_successful_result():
    result = object()

    def complete():
        return result

    outcome = run_exercise(complete)

    assert outcome.status == "success"
    assert outcome.value is result


def test_exercise_wrapper_returns_incomplete_feedback():
    @exercise
    def incomplete():
        ...

    feedback = incomplete()

    assert "Please complete the exercise." in feedback.text


def test_exercise_wrapper_includes_error_traceback():
    @exercise
    def broken():
        raise ValueError("invalid likelihood support")

    feedback = broken()

    assert "ValueError: invalid likelihood support" in feedback.text


def test_exercise_wrapper_wraps_successful_result():
    @exercise
    def complete():
        return "useful result"

    feedback = complete()

    assert "Exercise complete." in feedback.text
