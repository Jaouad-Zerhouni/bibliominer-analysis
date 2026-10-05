"""What the distribution must contain to be installable as it is.

A comment in `pyproject.toml` guarantees nothing: it does not run. These
tests keep the promises the package makes to whoever installs it from
PyPI, without the repository around it.
"""
from pathlib import Path

import bibliominer_analysis


def test_the_scimago_reference_travels_with_the_package():
    """Without it, `pip install` would ship a module that knows nothing about
    journals any more.

    `metrics/scimago.py` reads it next to the module (`data_ref/`), not from
    the repository: it must therefore be IN the distribution. It is the only
    data file of the package, and its absence would only show at run time, on
    the user's machine.
    """
    root = Path(bibliominer_analysis.__file__).parent
    reference = root / "data_ref" / "scimagojr_2025.csv"

    assert reference.is_file(), "the SCImago reference table is missing from the distribution"
    assert reference.stat().st_size > 1_000_000, "reference table suspiciously truncated"


def test_the_declared_version_is_the_one_the_module_exposes():
    """The version lives in `__init__.py`; `pyproject.toml` reads it from there.

    Two copies always end up diverging, and the module's copy is the one users
    read.
    """
    assert bibliominer_analysis.__version__
    assert bibliominer_analysis.__version__[0].isdigit()


def test_the_public_api_is_importable():
    for name in bibliominer_analysis.__all__:
        assert hasattr(bibliominer_analysis, name), name
