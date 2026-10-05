"""The same export gives the same file, byte for byte, for EVERY drawing.

Networks and bars already did; world maps, the dendrogram, density maps,
the treemap and the Sankey diagrams still carried the date (PNG, SVG, PDF)
or randomly drawn SVG identifiers. And the JPG export passed
`metadata=None`, which matplotlib 3.6, the declared minimum, rejects.

One figure per drawing module, in the four formats.
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
