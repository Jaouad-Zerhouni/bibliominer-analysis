"""Analyse bibliométrique d'un corpus Scopus nettoyé par Bibliominer."""

# Avant les imports : `report` recopie la version dans le README du rapport.
__version__ = "0.1.0"

from .model.corpus import Corpus  # noqa: E402
from .report import Report  # noqa: E402

__all__ = ["Corpus", "Report", "__version__"]
