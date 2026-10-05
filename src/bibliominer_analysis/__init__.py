"""Bibliometric analysis of a Scopus corpus cleaned with Bibliominer."""

# Before the imports: `report` copies the version into the report's README.
__version__ = "0.1.0"

from .model.corpus import Corpus  # noqa: E402
from .report import Report  # noqa: E402
from .io.reader import NotCleanedError  # noqa: E402

__all__ = ["Corpus", "NotCleanedError", "Report", "__version__"]
