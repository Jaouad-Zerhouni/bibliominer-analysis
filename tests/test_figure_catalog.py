"""The catalogue: every figure of the interface, produced in Python.

The interface and the package must offer the same figures, with the same
options. The coverage test (every figure on screen has an entry) lives on
the API side, where the interface code is; here we check that the
catalogue keeps its promises: the screen's defaults, options, formats,
filters.
"""

from __future__ import annotations

import pandas as pd
import pytest

from bibliominer_analysis import Corpus
from bibliominer_analysis.figures.catalog import CATALOG
from bibliominer_analysis.figures.render import FigureSpec

PNG = b"\x89PNG\r\n\x1a\n"


def _aff(parent, city, country):
    return "parent 1: %s, city: %s, country: %s" % (parent, city, country)


RABAT = _aff("Mohammed V University", "Rabat", "Morocco")
MADRID = _aff("Universidad Politecnica", "Madrid", "Spain")
PARIS = _aff("Sorbonne University", "Paris", "France")


@pytest.fixture(scope="module")
def corpus():
    rows = []
    for i in range(24):
        year = 2016 + i % 9
        authors = ["Idri A.", "Hosni M.", "Garcia J.", "Martin P."][i % 4:] + ["Idri A."]
        affs = [RABAT, MADRID, PARIS][: 1 + i % 3]
        refs = ";".join("ref%d | 10.1/%d | %d | Author%d | Reference title number %d on estimation"
                        % (k, k, 2000 + k, k, k) for k in range(i % 5, i % 5 + 4))
        rows.append({
            "Title": "Effort estimation study number %d with ensembles" % i,
            "Year": str(year), "Cited by": str(3 * i % 17), "EID": "eid-%d" % i,
            "Source title": ["J Soft", "Inf Sci", "IEEE Access"][i % 3],
            "Document Type": ["Article", "Conference Paper"][i % 2],
            "Authors": "; ".join(dict.fromkeys(authors)),
            "Author full names": "; ".join(dict.fromkeys(authors)),
            "Affiliations": "; ".join(affs),
            "Author Keywords": "; ".join(["effort estimation", "ensemble",
                                          "machine learning", "deep learning",
                                          "software"][i % 3: i % 3 + 3]),
            "Abstract": "Software effort estimation with ensemble learning improves "
                        "estimation accuracy for software projects. Deep learning helps.",
            "References": refs,
        })
    return Corpus.from_dataframe(pd.DataFrame(rows))


def test_le_catalogue_est_decrit(corpus):
    table = Corpus.figure_catalog()
    assert len(table) == len(CATALOG) >= 80
    assert set(table["section"]) == {"Corpus", "Actors", "Impact", "Concepts", "Networks"}
    assert table["name"].is_unique
    row = table.set_index("name").loc["top-authors"]
    assert row["page"] == "Authors" and "n=20" in row["options"]


def test_chaque_figure_se_dessine_ou_dit_pourquoi(corpus):
    """On a small corpus some figures have no data (SCImago, open access...):
    they SAY so (ValueError), they do not crash."""
    drawn = 0
    for name in CATALOG:
        try:
            assert corpus.figure(name, dpi=50).startswith(PNG), name
            drawn += 1
        except ValueError as exc:
            assert "no data" in str(exc) or "empty" in str(exc) or "without" in str(exc), (name, exc)
    assert drawn >= 60


def test_defauts_de_l_ecran_et_options(corpus):
    assert len(corpus.figure_spec("top-authors").categories) <= 20
    assert len(corpus.figure_spec("top_authors", n=2).categories) == 2    # snake_case accepted
    with pytest.raises(TypeError, match="accepted: n"):
        corpus.figure("top-authors", top=10)
    with pytest.raises(KeyError):
        corpus.figure("no-such-figure")


def test_une_periode_se_choisit_par_filter(corpus):
    spec = corpus.filter(years=(2023, 2024)).figure_spec("documents-per-year")
    assert spec.categories == ["2023", "2024"]


def test_svg_et_fichier(corpus, tmp_path):
    svg = corpus.figure("documents-per-year", fmt="svg")
    assert b"<svg" in svg[:500]
    out = corpus.figure("co-word-network", path=tmp_path / "co_word.svg", min_weight=1)
    assert (tmp_path / "co_word.svg").read_bytes() == out and b"<svg" in out[:500]


def test_figure_spec_seulement_pour_les_figures_simples(corpus):
    assert isinstance(corpus.figure_spec("citations-per-year"), FigureSpec)
    assert corpus.figure_spec("citations-per-year").title == "Citations per year"
    with pytest.raises(TypeError, match="dedicated renderer"):
        corpus.figure_spec("collaboration-authors")


def test_haute_resolution_par_defaut(corpus):
    """300 dpi by default: the same figure is larger than at 100 dpi."""
    assert len(corpus.figure("documents-per-year")) > len(corpus.figure("documents-per-year", dpi=100))


def test_a_catalog_figure_takes_the_style_of_the_export_dialog(corpus):
    """What the interface's export dialog sets, Python sets too."""
    svg = corpus.figure("documents-per-year", fmt="svg",
                        style={"kind": "pie", "show_title": True,
                               "subtitle": "Years 2016-2024", "value_labels": True})
    assert b"Documents per year" in svg and b"Years 2016-2024" in svg


def test_a_style_key_that_is_not_a_style_is_refused(corpus):
    with pytest.raises(TypeError, match="style"):
        corpus.figure("documents-per-year", style={"series": []})


def test_the_report_files_a_styled_corpus_figure_under_its_period(corpus):
    import io
    import zipfile

    from bibliominer_analysis import Report
    recent = corpus.filter(years=(2016, 2019))
    report = Report().add_corpus_figure(recent, "top-authors", period="2016-2019",
                                        style={"show_title": True, "palette": "gradient"})
    names = zipfile.ZipFile(io.BytesIO(report.to_bytes())).namelist()
    assert "2-actors/figures/top-authors_2016-2019.png" in names
