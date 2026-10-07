"""Institutions: production over time and distribution by country."""

import pandas as pd

from bibliominer_analysis import Corpus
from bibliominer_analysis.io.schema import INDEPENDENT_LABEL


def _doc(i, year, affs):
    return {
        "Title": "Doc %d" % i, "Year": str(year), "Cited by": "0",
        "EID": "eid-%d" % i, "Source title": "J", "Document Type": "Article",
        "Authors": "1:A.", "Affiliations": affs,
    }


def _aff(parent, country="Morocco"):
    return "parent 1: %s, city: X, country: %s" % (parent, country)


def test_institutions_over_time_complete_series():
    """Every institution must have ALL the years, even at zero."""
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, 2020, _aff("Univ A")),
        _doc(2, 2022, _aff("Univ A")),
        _doc(3, 2021, _aff("Univ B")),
    ]))
    t = c.institutions_over_time()
    assert sorted(t["year"].unique()) == [2020, 2021, 2022]
    # 2 institutions x 3 years = 6 rows, without gaps
    assert len(t) == 6
    a = t[t["institution"] == "Univ A"].set_index("year")
    assert list(a["documents"]) == [1, 0, 1]
    assert list(a["cumulative"]) == [1, 1, 2]


def test_institutions_over_time_excludes_independents():
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, 2020, _aff("Univ A")),
        _doc(2, 2020, _aff(INDEPENDENT_LABEL)),
    ]))
    t = c.institutions_over_time()
    assert INDEPENDENT_LABEL not in set(t["institution"])


def test_institutions_over_time_limit():
    rows = [_doc(i, 2020, _aff("U%d" % i)) for i in range(5)]
    c = Corpus.from_dataframe(pd.DataFrame(rows))
    assert c.institutions_over_time(n=2)["institution"].nunique() == 2


def test_institutions_by_country():
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, 2020, _aff("Univ A", "Morocco")),
        _doc(2, 2020, _aff("Univ A", "Morocco")),
        _doc(3, 2020, _aff("Univ B", "Morocco")),
        _doc(4, 2020, _aff("Univ C", "Spain")),
    ]))
    t = c.institutions_by_country().set_index("country")
    assert t.loc["Morocco", "institutions"] == 2
    assert t.loc["Morocco", "documents"] == 3
    assert t.loc["Morocco", "top_institution"] == "Univ A"
    assert t.loc["Spain", "institutions"] == 1


def test_institutions_empty_corpus():
    c = Corpus.from_dataframe(pd.DataFrame([{"Title": "x"}]))
    assert c.institutions_over_time().empty
    assert c.institutions_by_country().empty
