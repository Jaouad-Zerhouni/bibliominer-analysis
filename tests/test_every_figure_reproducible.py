"""Le même export donne le même fichier, octet pour octet, pour CHAQUE dessin.

Les réseaux et les barres l'étaient ; les cartes du monde, le dendrogramme,
les cartes de densité, le treemap et les Sankey portaient encore la date
(PNG, SVG, PDF) ou des identifiants SVG tirés au hasard. Et l'export JPG
passait `metadata=None`, que matplotlib 3.6, le minimum déclaré, refuse.

Une figure par module de dessin, dans les quatre formats.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from bibliominer_analysis import Corpus

SYNTHETIC = Path(__file__).parent / "data" / "synthetic_cleaned.csv"
ONE_PER_MODULE = ["documents-per-year",   # render.py
                  "network-keywords",     # network.py
                  "country-map",          # world.py
                  "three-fields",         # charts.py (Sankey)
                  "dendrogram"]           # charts.py


@pytest.fixture(scope="module")
def corpus():
    return Corpus.from_csv(SYNTHETIC)


@pytest.mark.parametrize("fmt", ["png", "svg", "pdf", "jpg"])
@pytest.mark.parametrize("name", ONE_PER_MODULE)
def test_two_exports_are_identical(corpus, name, fmt):
    first = corpus.figure(name, fmt=fmt)
    assert first, "empty file"
    assert corpus.figure(name, fmt=fmt) == first
