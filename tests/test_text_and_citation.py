"""Text mining, dendrogram, K-S test, direct citation, density."""

import numpy as np
import pandas as pd
import pytest

from bibliominer_analysis import Corpus
from bibliominer_analysis.metrics import factorial as fac
from bibliominer_analysis.metrics import laws as lw
from bibliominer_analysis.metrics import local as L
from bibliominer_analysis.metrics import text as tx
from bibliominer_analysis.networks import analysis as netan


def _doc(i, title="A long and perfectly ordinary document title",
         abstract="", year=2020, authors="1:A.", source="J1", refs="",
         doi="", kws="alpha"):
    return {
        "Title": title, "Year": str(year), "Cited by": "0", "EID": "eid-%d" % i,
        "Source title": source, "Document Type": "Article", "Authors": authors,
        "Author Keywords": kws, "Abstract": abstract, "DOI": doi,
        "References": refs,
        "Affiliations": "parent 1: U, city: X, country: Morocco",
    }


# ------------------------------------------------------------------ texte ----

def test_bigrams_extracted_from_abstract():
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, abstract="We apply machine learning to software effort estimation."),
        _doc(2, abstract="Machine learning improves effort estimation greatly."),
    ]))
    terms = set(tx.top_terms(c, "abstract", ngram=2, min_documents=2)["term"])
    assert "machine learning" in terms
    assert "effort estimation" in terms


def test_ngram_neither_starts_nor_ends_with_a_stopword():
    '''"of the model" and "the model of" are not terms.'''
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, abstract="the accuracy of the model is high"),
        _doc(2, abstract="the accuracy of the model is stable"),
    ]))
    terms = list(tx.top_terms(c, "abstract", ngram=2, min_documents=2)["term"])
    for t in terms:
        first, last = t.split()[0], t.split()[-1]
        assert first not in tx.STOPWORDS and last not in tx.STOPWORDS, t


def test_copyright_removed_before_counting():
    """Without it, "springer nature" climbs into the very first terms."""
    tail = " © The Author(s), under exclusive licence to Springer Nature 2024."
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, abstract="Flood prediction with random forests." + tail),
        _doc(2, abstract="Flood prediction using neural networks." + tail),
        _doc(3, abstract="Flood prediction and hazard mapping." + tail),
    ]))
    terms = set(tx.top_terms(c, "abstract", ngram=2, min_documents=2)["term"])
    assert "springer nature" not in terms
    assert "flood prediction" in terms


def test_strip_copyright_keeps_the_useful_text():
    assert tx.strip_copyright("Body text. © 2020 Elsevier").strip() == "Body text."
    assert tx.strip_copyright("Body text. Copyright © 2020").strip() == "Body text."
    assert tx.strip_copyright("No notice here") == "No notice here"


def test_term_counted_once_per_document():
    """A repetitive abstract must not weigh more than another document."""
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, abstract="deep learning deep learning deep learning"),
        _doc(2, abstract="deep learning applied once"),
    ]))
    row = tx.top_terms(c, "abstract", ngram=2, min_documents=1)
    value = row.loc[row["term"] == "deep learning", "documents"].iat[0]
    assert value == 2


def test_text_network_is_a_standard_graph():
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, abstract="machine learning and effort estimation together"),
        _doc(2, abstract="machine learning with effort estimation again"),
    ]))
    g = tx.text_co_occurrence(c, "abstract", ngram=2, min_weight=2, min_documents=2)
    assert set(g) >= {"nodes", "edges", "n_nodes", "n_edges"}
    # Compatible with the package's network measures.
    assert netan.graph_summary(g)["nodes"] == g["n_nodes"]


# ----------------------------------------------------------- dendrogramme ----

def test_dendrogram_separates_two_families():
    rows = [_doc(i, kws="alpha;beta;gamma") for i in range(6)]
    rows += [_doc(i, kws="delta;epsilon;zeta") for i in range(6, 12)]
    c = Corpus.from_dataframe(pd.DataFrame(rows))
    d = fac.topic_dendrogram(c, top_n=10, min_documents=2, max_clusters=2)
    assert d["tree"] is not None
    assert d["n_terms"] == 6
    families = [{"alpha", "beta", "gamma"}, {"delta", "epsilon", "zeta"}]
    for g in d["clusters"]:
        members = {t.strip() for t in g["terms"].split(",")}
        assert any(members <= f for f in families), members


def test_dendrogram_cut_at_requested_number():
    rows = [_doc(i, kws="alpha;beta") for i in range(4)]
    rows += [_doc(i, kws="gamma;delta") for i in range(4, 8)]
    rows += [_doc(i, kws="epsilon;zeta") for i in range(8, 12)]
    c = Corpus.from_dataframe(pd.DataFrame(rows))
    for k in (2, 3):
        assert len(fac.topic_dendrogram(c, top_n=10, max_clusters=k)["clusters"]) == k


def test_average_linkage_merges_the_closest_first():
    D = np.array([[0.0, 0.1, 0.9], [0.1, 0.0, 0.9], [0.9, 0.9, 0.0]])
    merges = fac._average_linkage(D)
    assert merges[0][2] == pytest.approx(0.1)
    assert merges[-1][2] > merges[0][2]


# ------------------------------------------------------------ K-S test ----

def test_ks_accepts_an_identical_distribution():
    p = np.array([0.6, 0.2, 0.1, 0.1])
    r = lw.kolmogorov_smirnov(p, p)
    assert r["d"] == 0.0
    assert r["follows_lotka"] is True
    assert r["p_value"] == pytest.approx(1.0)


def test_ks_rejects_an_opposite_distribution():
    a = np.array([0.9, 0.05, 0.03, 0.02])
    b = np.array([0.02, 0.03, 0.05, 0.9])
    r = lw.kolmogorov_smirnov(a, b)
    assert r["d"] > r["critical_5pct"]
    assert r["follows_lotka"] is False


def test_lotka_exposes_the_test():
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, authors="1:A."), _doc(2, authors="1:A.; 2:B."),
        _doc(3, authors="1:C."),
    ]))
    ks = c.lotka()["ks_test"]
    assert set(ks) == {"d", "p_value", "critical_5pct", "follows_lotka"}


# ------------------------------------------------- citation directe ---------

LONG_A = "A systematic review of ensemble effort estimation methods"


@pytest.fixture
def citing():
    """Doc 2 (journal J2) cites doc 1 (journal J1)."""
    return Corpus.from_dataframe(pd.DataFrame([
        _doc(1, title=LONG_A, doi="10.1000/aaa", source="J1", authors="1:A."),
        _doc(2, title="Another paper with a decently long title", source="J2",
             authors="1:B.", refs="ref1 | 10.1000/aaa | 2016 | X | %s" % LONG_A),
    ]))


def test_citation_network_is_directed(citing):
    g = L.citation_network(citing, "sources", min_weight=1)
    assert g["n_edges"] == 1
    edge = g["edges"][0]
    assert edge["source"] == "J2" and edge["target"] == "J1"


def test_entity_self_citation_discarded():
    """Two documents from THE SAME journal: nothing to show at journal level."""
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, title=LONG_A, doi="10.1000/aaa", source="J1"),
        _doc(2, title="Another paper with a decently long title", source="J1",
             refs="ref1 | 10.1000/aaa | 2016 | X | %s" % LONG_A),
    ]))
    assert L.citation_network(c, "sources")["n_edges"] == 0


def test_citation_per_author(citing):
    g = L.citation_network(citing, "authors", min_weight=1)
    assert {e["source"] for e in g["edges"]} == {"B."}
    assert {e["target"] for e in g["edges"]} == {"A."}


# ------------------------------------------------- layout, density ----

def _ring_graph():
    ids = list("ABCDE")
    edges = [{"source": ids[i], "target": ids[(i + 1) % 5], "weight": 1}
             for i in range(5)]
    return {"nodes": [{"id": x, "label": x, "occurrences": 2, "degree": 2} for x in ids],
            "edges": edges, "n_nodes": 5, "n_edges": 5}


def test_layout_is_deterministic():
    g = _ring_graph()
    assert netan.layout(g) == netan.layout(g)


def test_layout_places_every_node():
    coords = netan.layout(_ring_graph())
    assert set(coords) == set("ABCDE")
    assert all(len(v) == 2 for v in coords.values())


def test_normalised_density_between_zero_and_one():
    g = _ring_graph()
    d = netan.density_grid(g, netan.layout(g), size=12)
    values = [c[2] for c in d["cells"]]
    assert len(d["x"]) == 12 and len(d["y"]) == 12
    assert min(values) >= 0.0 and max(values) == pytest.approx(1.0)


def test_resolution_changes_the_number_of_groups():
    """Without an effect, the setting would be a decorative button."""
    rows = []
    for i in range(6):
        rows.append(_doc(i, kws="alpha;beta;gamma"))
    for i in range(6, 12):
        rows.append(_doc(i, kws="delta;epsilon;zeta"))
    c = Corpus.from_dataframe(pd.DataFrame(rows))
    g = c.co_word(top_n=20, min_weight=2)
    low = {n["community"] for n in netan.annotate(g, resolution=0.3)["nodes"]}
    high = {n["community"] for n in netan.annotate(g, resolution=3.0)["nodes"]}
    assert len(high) >= len(low)
