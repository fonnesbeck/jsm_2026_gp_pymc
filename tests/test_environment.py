"""The workshop environment must provide every library the notebooks import."""


def test_patsy_available():
    import patsy

    assert hasattr(patsy, "dmatrix")


def test_preliz_available():
    # Run in subprocess to avoid pytest's assertion-rewrite hook interaction.
    # pytest imports assert_rewrite before preliz, and preliz's dependency on
    # IPython.core.ultratb → pygments.styles.get_style_by_name creates a
    # deadlock in StyleMeta.__new__. Direct in-process import hangs forever;
    # subprocess isolates the import and avoids the hook conflict.
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import preliz as pz; assert hasattr(pz, 'maxent'); assert hasattr(pz, 'Gamma')",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
