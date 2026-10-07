"""Tests of the h and g indices.

The expected values are computed BY HAND from the canonical definitions,
not produced by the code itself. That is the only way an indicator test is
worth anything: otherwise it only engraves the error.
"""

import pandas as pd
import pytest

from bibliominer_analysis import Corpus
from bibliominer_analysis.metrics.impact import g_index, h_index, i10_index


# --- h-index ---------------------------------------------------------------

def test_h_index_classic_case():
    # [10, 8, 5, 4, 3]: 4 articles have >= 4 citations; the 5th has only 3.
    assert h_index([10, 8, 5, 4, 3]) == 4


def test_h_index_one_highly_cited_article():
    # A single article, even cited 100 times, only gives an h of 1.
    assert h_index([100]) == 1


def test_h_index_no_citation():
    assert h_index([0, 0, 0]) == 0


def test_h_index_empty_series():
    assert h_index([]) == 0


def test_h_index_order_independent():
    assert h_index([3, 10, 4, 8, 5]) == h_index([10, 8, 5, 4, 3])


def test_h_index_all_equal():
    # 5 articles with 5 citations: h = 5.
    assert h_index([5, 5, 5, 5, 5]) == 5


# --- g-index ---------------------------------------------------------------

def test_g_index_classic_case():
    # [10, 8, 5, 4, 3] -> cumulative [10, 18, 23, 27, 30]; g^2 [1, 4, 9, 16, 25].
    # 30 >= 25 so g = 5.
    assert g_index([10, 8, 5, 4, 3]) == 5


def test_g_index_greater_or_equal_to_h():
    """Fundamental property: g >= h, always."""
    for serie in ([10, 8, 5, 4, 3], [100], [1, 1, 1], [25, 8, 5, 3, 3],
                  [0, 0, 1], [7, 7, 7, 7]):
        assert g_index(serie) >= h_index(serie), serie


def test_g_index_highly_cited_article_counts():
    # h is capped at 1, but g goes up: 25 citations cover 5^2.
    assert h_index([25, 0, 0, 0, 0]) == 1
    assert g_index([25, 0, 0, 0, 0]) == 5


def test_g_index_empty_and_zero():
    assert g_index([]) == 0
    assert g_index([0, 0]) == 0


def test_i10_index():
    assert i10_index([10, 9, 11, 0]) == 2


# --- at corpus level -------------------------------------------------------

@pytest.fixture
def corpus() -> Corpus:
    """4 documents by the same author, citations 10 / 8 / 5 / 1."""
    rows = []
    for i, cites in enumerate([10, 8, 5, 1], start=1):
        rows.append({
            "Title": "Doc %d" % i, "Year": str(2018 + i), "Cited by": str(cites),
            "EID": "eid-%d" % i, "Source title": "J", "Document Type": "Article",
            "Authors": "1:Okafor M.; 2:Varela A.",
            "Author full names": "1:Okafor, Maya (222); 2:Varela, Ana (111)",
            "Author(s) ID": "1:222; 2:111",
            "Affiliations": "parent 1: Moulay Ismail University, city: Meknes, country: Morocco",
        })
    return Corpus.from_dataframe(pd.DataFrame(rows))


def test_corpus_impact(corpus):
    imp = corpus.impact()
    assert imp["documents"] == 4
    assert imp["citations"] == 24
    assert imp["h_index"] == 3          # 10, 8, 5 >= 3; the 4th has only 1
    assert imp["i10_index"] == 1        # only the document with 10 citations
    assert imp["uncited"] == 0


def test_authors_impact(corpus):
    df = corpus.authors_impact()
    assert set(df.columns) >= {"author", "documents", "citations",
                               "h_index", "g_index", "first_author"}
    okafor = df[df["author"] == "Okafor M."].iloc[0]
    assert okafor["documents"] == 4
    assert okafor["citations"] == 24
    assert okafor["h_index"] == 3
    assert okafor["first_author"] == 4    # first on all 4 documents
    varela = df[df["author"] == "Varela A."].iloc[0]
    assert varela["first_author"] == 0     # always second


def test_authors_impact_period(corpus):
    df = corpus.authors_impact()
    okafor = df[df["author"] == "Okafor M."].iloc[0]
    assert (okafor["first_year"], okafor["last_year"]) == (2019, 2022)


def test_institutions_impact(corpus):
    df = corpus.institutions_impact()
    assert df.iloc[0]["institution"] == "Moulay Ismail University"
    assert df.iloc[0]["documents"] == 4
    assert df.iloc[0]["h_index"] == 3
    assert df.iloc[0]["country"] == "Morocco"


def test_summary_counts_the_affiliations(corpus):
    s = corpus.summary()
    assert s["affiliations"] == 1        # a single distinct affiliation
    assert s["institutions"] == 1


def test_empty_corpus_indices_at_zero():
    c = Corpus.from_dataframe(pd.DataFrame([{"Title": "x"}]))
    assert c.impact()["h_index"] == 0
    assert c.authors_impact().empty
