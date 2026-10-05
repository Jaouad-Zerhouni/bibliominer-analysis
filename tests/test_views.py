"""Results as the interface shows them, available in Python.

Document list, normalised citations, citation balance, the "Network lab"
network, density map: the API only calls them now, so a package user gets
exactly the tables on screen.
"""

from __future__ import annotations

import pandas as pd
import pytest

from bibliominer_analysis import Corpus
from bibliominer_analysis.views import NETWORK_UNITS

T1 = "A systematic review of ensemble effort estimation methods"
T2 = "Improved estimation of software development effort using ensembles"
T3 = "Deep learning for effort estimation in agile software projects"


def _aff(parent, city, country):
    return "parent 1: %s, city: %s, country: %s" % (parent, city, country)


def _doc(i, title, source, year, cited, authors, affs, refs="", keywords="",
         abstract="", doi=""):
    return {
        "Title": title, "Year": str(year), "Cited by": str(cited),
        "EID": "eid-%d" % i, "Source title": source, "Document Type": "Article",
        "Authors": authors, "Author full names": authors,
        "Affiliations": affs, "References": refs, "Author Keywords": keywords,
        "Abstract": abstract, "DOI": doi,
    }


RABAT = _aff("Mohammed V University", "Rabat", "Morocco")
MADRID = _aff("Universidad Politecnica", "Madrid", "Spain")
ABSTRACT = ("Software effort estimation with ensemble methods improves "
            "software effort estimation accuracy.")


@pytest.fixture(scope="module")
def corpus():
    """Doc 3 (J3) cites doc 1 (J1) and doc 2 (J2); doc 4 (J3) cites doc 1."""
    return Corpus.from_dataframe(pd.DataFrame([
        _doc(1, T1, "J1", 2016, 100, "Idri A.; Hosni M.", RABAT + "; " + MADRID,
             keywords="effort estimation; ensemble", abstract=ABSTRACT,
             doi="10.1000/aaa"),
        _doc(2, T2, "J2", 2020, 10, "Hosni M.; Garcia J.", RABAT + "; " + MADRID,
             keywords="effort estimation; machine learning", abstract=ABSTRACT),
        _doc(3, T3, "J3", 2022, 4, "Idri A.; Garcia J.", RABAT + "; " + MADRID,
             keywords="machine learning; ensemble", abstract=ABSTRACT,
             refs="ref1 | 10.1000/aaa | 2016 | Idri | %s;ref2 |  | 2020 | Hosni | %s" % (T1, T2)),
        _doc(4, "A fourth paper on agile teams", "J3", 2022, 0, "Idri A.", RABAT,
             keywords="effort estimation; ensemble", abstract=ABSTRACT,
             refs="ref1 | 10.1000/aaa | 2016 | Idri | %s" % T1),
    ]))


# --- documents ---------------------------------------------------------------

def test_document_list_colonnes_et_tri(corpus):
    docs = corpus.document_list()
    assert list(docs.columns) == ["title", "first_author", "year", "source",
                                  "doc_type", "citations", "local_citations", "doi"]
    assert list(docs["citations"]) == [100, 10, 4, 0]
    assert docs.loc[0, "first_author"] == "Idri A."
    # doc 1 is cited by doc 3 and doc 4, doc 2 by doc 3
    assert dict(zip(docs["title"], docs["local_citations"]))[T1] == 2
    assert dict(zip(docs["title"], docs["local_citations"]))[T2] == 1


def test_document_list_autres_tris(corpus):
    assert list(corpus.document_list(sort="year")["year"])[:2] == [2022, 2022]
    titles = list(corpus.document_list(sort="title")["title"])
    assert titles == sorted(titles)
    assert len(corpus.document_list(2)) == 2
    assert len(corpus.document_list(None)) == 4
    with pytest.raises(ValueError):
        corpus.document_list(sort="doi")


def test_most_normalized_documents(corpus):
    top = corpus.most_normalized_documents()
    assert list(top.columns) == ["title", "first_author", "year", "source", "citations",
                                 "year_mean_citations", "normalized_citations"]
    values = list(top["normalized_citations"])
    assert values == sorted(values, reverse=True)
    # Alone in its year, a document is worth exactly the mean: 1.
    assert dict(zip(top["title"], top["normalized_citations"]))[T1] == pytest.approx(1.0)


# --- citation directe --------------------------------------------------------

def test_citation_balance_sources(corpus):
    b = corpus.citation_balance("sources", min_weight=1)
    rows = {r.label: (r.received, r.emitted, r.balance) for r in b.itertuples()}
    assert rows["J1"] == (2, 0, 2)
    assert rows["J2"] == (1, 0, 1)
    assert rows["J3"] == (0, 3, -3)
    assert list(b["balance"]) == sorted(b["balance"], reverse=True)


def test_citation_balance_reprend_le_reseau(corpus):
    """The table and the drawing come from the same network."""
    graph = corpus.citation_graph("sources", min_weight=1)
    labels = {n["label"] for n in graph["nodes"]}
    assert set(corpus.citation_balance("sources", min_weight=1)["label"]) == labels
    assert "summary" in graph and "community" in graph["nodes"][0]


# --- networks ----------------------------------------------------------------

@pytest.mark.parametrize("unit", NETWORK_UNITS)
def test_network_toutes_les_unites(corpus, unit):
    graph = corpus.network(unit, top_n=20, min_weight=1)
    assert {"nodes", "edges", "summary"} <= set(graph)
    ids = {n["id"] for n in graph["nodes"]}
    assert all(e["source"] in ids and e["target"] in ids for e in graph["edges"])


def test_network_unite_inconnue(corpus):
    with pytest.raises(ValueError):
        corpus.network("journals")


def test_network_normalisation_garde_le_poids_brut(corpus):
    graph = corpus.network("keywords", min_weight=1, normalization="association")
    assert graph["edges"]
    assert all("raw_weight" in e for e in graph["edges"])


def test_network_superposition_par_annee(corpus):
    graph = corpus.network("authors", min_weight=1, overlay=True)
    assert any(n.get("overlay_year") is not None for n in graph["nodes"])


def test_density_map_deterministe(corpus):
    first = corpus.density_map("keywords", min_weight=1, size=16)
    second = corpus.density_map("keywords", min_weight=1, size=16)
    assert first == second
    assert {"nodes", "grid", "n_nodes", "n_edges"} == set(first)
    assert all({"x", "y"} <= set(n) for n in first["nodes"])


def test_term_network(corpus):
    graph = corpus.term_network("abstract", ngram=2, min_weight=1, min_documents=1)
    assert graph["nodes"] and "summary" in graph
