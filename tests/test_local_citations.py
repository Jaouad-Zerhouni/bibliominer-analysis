"""Local citations: matching reference -> document, and its consequences.

These are the most important tests of the set: everything that follows
(most locally cited documents and authors, historiograph) rests on this
matching. A WRONG match is worse than a missing one, since it creates a
lineage that does not exist.
"""

import pandas as pd
import pytest

from bibliominer_analysis import Corpus
from bibliominer_analysis.metrics import local as L


def _doc(i, title, authors="1:A.", doi="", refs="", year="2020", cited="0"):
    return {
        "Title": title, "Year": year, "Cited by": cited,
        "EID": "eid-%d" % i, "Source title": "J1", "Document Type": "Article",
        "Authors": authors, "Author Keywords": "alpha", "DOI": doi,
        "Affiliations": "parent 1: U, city: X, country: Morocco",
        "References": refs,
    }


LONG_A = "A systematic review of ensemble effort estimation methods"
LONG_B = "Improved estimation of software development effort using ensembles"


@pytest.fixture
def corpus():
    """Doc 3 cites doc 1 by DOI and doc 2 by title."""
    return Corpus.from_dataframe(pd.DataFrame([
        _doc(1, LONG_A, doi="10.1000/aaa", year="2016", cited="100"),
        _doc(2, LONG_B, year="2017", cited="10"),
        _doc(3, "A third paper", year="2019", refs=(
            "ref1 | 10.1000/aaa | 2016 | Varela | %s;"
            "ref2 |  | 2017 | Okafor | %s" % (LONG_A, LONG_B))),
    ]))


def test_matching_by_doi_and_by_title(corpus):
    pairs = L.citation_pairs(corpus)
    assert len(pairs) == 2
    assert set(pairs["via"]) == {"doi", "title"}
    assert set(pairs["citing"]) == {"eid-3"}
    assert set(pairs["cited"]) == {"eid-1", "eid-2"}


def test_the_doi_prevails_over_the_title():
    """When both agree, the pair counts only ONCE, through the DOI."""
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, LONG_A, doi="10.1000/aaa"),
        _doc(2, "Citing", refs="ref1 | 10.1000/aaa | 2016 | X | %s" % LONG_A),
    ]))
    pairs = L.citation_pairs(c)
    assert len(pairs) == 1
    assert pairs.iloc[0]["via"] == "doi"


def test_doi_normalised_despite_url_prefix():
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, LONG_A, doi="https://doi.org/10.1000/AAA"),
        _doc(2, "Citing", refs="ref1 | doi:10.1000/aaa | 2016 | X | other title"),
    ]))
    assert len(L.citation_pairs(c)) == 1


def test_too_short_title_refused():
    '''"Machine learning" must NEVER serve as a matching key.'''
    short = "Machine learning"
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, short),
        _doc(2, "Citing", refs="ref1 |  | 2016 | X | %s" % short),
    ]))
    assert L.citation_pairs(c).empty


def test_no_self_citation():
    """A document citing itself must not inflate its own score."""
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, LONG_A, doi="10.1000/aaa",
             refs="ref1 | 10.1000/aaa | 2016 | X | %s" % LONG_A),
    ]))
    assert L.citation_pairs(c).empty
    assert int(L.local_citations(c)["local_citations"].sum()) == 0


def test_most_locally_cited_documents(corpus):
    d = L.most_local_cited_documents(corpus)
    assert len(d) == 2
    row = d[d["title"] == LONG_A].iloc[0]
    assert row["local_citations"] == 1
    assert row["global_citations"] == 100
    # 1 local citation out of 100 global ones
    assert row["lc_gc_ratio"] == 1.0


def test_labels_disambiguated():
    """Two articles by the same author in the same year get a suffix."""
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, LONG_A, authors="1:Varela A.", doi="10.1000/aaa", year="2016"),
        _doc(2, LONG_B, authors="1:Varela A.", doi="10.1000/bbb", year="2016"),
        _doc(3, "Citing paper", year="2019", refs=(
            "ref1 | 10.1000/aaa | 2016 | X | t1;ref2 | 10.1000/bbb | 2016 | X | t2")),
    ]))
    labels = list(L.most_local_cited_documents(c)["label"])
    assert len(set(labels)) == 2, "two identical labels would make the graph wrong"
    assert all(x.startswith("VARELA A., 2016") for x in labels)


def test_locally_cited_authors_full_counting():
    """Every author receives all the local citations of the document."""
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, LONG_A, authors="1:A.; 2:B.", doi="10.1000/aaa"),
        _doc(2, "Citing", refs="ref1 | 10.1000/aaa | 2016 | X | t"),
    ]))
    a = L.most_local_cited_authors(c).set_index("author")["local_citations"]
    assert a["A."] == 1 and a["B."] == 1


def test_historiograph_keeps_only_internal_links(corpus):
    h = L.historiograph(corpus, n=25)
    ids = {n["id"] for n in h["nodes"]}
    # eid-3 receives no citation: it does not enter the graph, so its outgoing
    # links must not appear in it either.
    assert "eid-3" not in ids
    for e in h["edges"]:
        assert e["source"] in ids and e["target"] in ids


def test_corpus_without_references_does_not_break():
    c = Corpus.from_dataframe(pd.DataFrame([_doc(1, LONG_A)]))
    assert L.citation_pairs(c).empty
    assert L.most_local_cited_documents(c).empty
    assert L.historiograph(c)["n_nodes"] == 0
    assert int(L.local_citations(c)["local_citations"].sum()) == 0
