"""Corpus profile: ageing, concentration, self-citation, access."""

import pandas as pd
import pytest

from bibliominer_analysis import Corpus
from bibliominer_analysis.metrics import access as acc
from bibliominer_analysis.metrics import aging, concentration as conc
from bibliominer_analysis.metrics import selfcitation as sf

LONG = "A systematic review of ensemble effort estimation methods"


def _doc(i, year=2020, refs="", authors="1:A.", oa="", cited="0",
         aff="parent 1: U, city: Rabat, country: Morocco"):
    return {
        "Title": "Document %d with a reasonably long title" % i,
        "Year": str(year), "Cited by": str(cited), "EID": "eid-%d" % i,
        "Source title": "J1", "Document Type": "Article", "Authors": authors,
        "Author Keywords": "alpha", "References": refs, "Affiliations": aff,
        "Open Access": oa,
    }


# ------------------------------------------------------------ Price ---------

def test_price_index():
    """Three recent references out of four -> 75 %."""
    refs = ";".join(["ref%d |  | %d | X | t%d" % (i, y, i)
                     for i, y in enumerate([2019, 2018, 2017, 2000])])
    c = Corpus.from_dataframe(pd.DataFrame([_doc(1, 2020, refs=refs)]))
    p = aging.price_index(c, window=5)
    assert p["references"] == 4
    assert p["price_index"] == pytest.approx(75.0)


def test_median_age_and_half_life_coincide():
    refs = ";".join(["ref%d |  | %d | X | t%d" % (i, y, i)
                     for i, y in enumerate([2019, 2015, 2010])])
    c = Corpus.from_dataframe(pd.DataFrame([_doc(1, 2020, refs=refs)]))
    p = aging.price_index(c)
    assert p["median_age"] == p["half_life"] == 5.0


def test_outlier_reference_year_discarded():
    refs = "ref1 |  | 1500 | X | t1;ref2 |  | 2018 | X | t2"
    c = Corpus.from_dataframe(pd.DataFrame([_doc(1, 2020, refs=refs)]))
    assert aging.price_index(c)["references"] == 1


def test_age_distribution_sums_to_hundred():
    refs = ";".join(["ref%d |  | %d | X | t%d" % (i, y, i)
                     for i, y in enumerate([2019, 2018, 2018])])
    c = Corpus.from_dataframe(pd.DataFrame([_doc(1, 2020, refs=refs)]))
    d = aging.reference_age_distribution(c)
    assert d["cumulative_share"].iloc[-1] == pytest.approx(100.0, abs=0.1)


# ----------------------------------------------------- concentration --------

def test_gini_zero_when_everyone_produces_the_same():
    assert conc.gini([5, 5, 5, 5]) == pytest.approx(0.0, abs=1e-9)


def test_gini_close_to_one_when_one_produces():
    assert conc.gini([100, 0, 0, 0, 0, 0, 0, 0, 0, 0]) > 0.85


def test_lorenz_starts_at_zero_and_reaches_hundred():
    lz = conc.lorenz([1, 2, 3, 4])
    assert lz.iloc[0]["value_share"] == pytest.approx(0.0)
    assert lz.iloc[-1]["value_share"] == pytest.approx(100.0)


def test_concentration_per_dimension():
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, authors="1:A."), _doc(2, authors="1:A."), _doc(3, authors="1:B."),
    ]))
    r = conc.concentration(c, "authors")
    assert r["entities"] == 2
    assert r["documents"] == 3
    assert r["cr4"] == pytest.approx(100.0)


def test_summary_covers_every_dimension():
    c = Corpus.from_dataframe(pd.DataFrame([_doc(1)]))
    assert list(conc.concentration_summary(c)["unit"]) == list(conc.UNITS)


# ------------------------------------------------------ auto-citation -------

def test_self_citation_detected():
    """Doc 2 cites doc 1, same author -> author self-citation."""
    c = Corpus.from_dataframe(pd.DataFrame([
        {**_doc(1, authors="1:A."), "Title": LONG, "DOI": "10.1000/aaa"},
        {**_doc(2, authors="1:A."), "References": "r1 | 10.1000/aaa | 2016 | X | %s" % LONG},
    ]))
    r = sf.self_citation_rate(c, "authors")
    assert r["citations"] == 1
    assert r["self_citations"] == 1
    assert r["self_rate"] == pytest.approx(100.0)


def test_external_citation_not_counted_as_self():
    c = Corpus.from_dataframe(pd.DataFrame([
        {**_doc(1, authors="1:A."), "Title": LONG, "DOI": "10.1000/aaa"},
        {**_doc(2, authors="1:B."), "References": "r1 | 10.1000/aaa | 2016 | X | %s" % LONG},
    ]))
    r = sf.self_citation_rate(c, "authors")
    assert r["self_citations"] == 0
    assert r["external"] == 1


def test_nested_levels():
    """An author self-citation is necessarily also a country self-citation."""
    c = Corpus.from_dataframe(pd.DataFrame([
        {**_doc(1, authors="1:A."), "Title": LONG, "DOI": "10.1000/aaa"},
        {**_doc(2, authors="1:A."), "References": "r1 | 10.1000/aaa | 2016 | X | %s" % LONG},
    ]))
    s = sf.self_citation_summary(c).set_index("level")["self_citations"]
    assert s["authors"] <= s["institutions"] <= s["countries"]


# ---------------------------------------------------------- acces ouvert ----

def test_access_routes_extracted():
    assert set(acc.routes_of("All Open Access; Gold Open Access")) == {"Gold"}
    assert set(acc.routes_of("All Open Access; Gold Open Access; Green Open Access")) \
        == {"Gold", "Green"}
    assert acc.routes_of("") == []
    assert acc.routes_of(None) == []


def test_status_two_rows_only():
    """Scopus never writes "closed": inventing the row would be lying."""
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, oa="All Open Access; Gold Open Access"), _doc(2),
    ]))
    st = acc.access_status(c)
    assert list(st["status"]) == ["Open access", "Not flagged"]
    assert list(st["documents"]) == [1, 1]
    assert st["share"].sum() == pytest.approx(100.0, abs=0.2)


def test_an_article_can_combine_two_routes():
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, oa="All Open Access; Gold Open Access; Green Open Access"),
    ]))
    r = acc.access_routes(c).set_index("route")["documents"]
    assert r["Gold"] == 1 and r["Green"] == 1


def test_yearly_share_relative_to_all_documents():
    """Restricting to the flagged documents would give 100 % every year."""
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, year=2020, oa="All Open Access; Gold Open Access"),
        _doc(2, year=2020),
        _doc(3, year=2020),
    ]))
    row = acc.access_over_time(c).iloc[0]
    assert row["documents"] == 3
    assert row["open_access"] == 1
    assert row["share"] == pytest.approx(33.3, abs=0.1)


def test_corpus_without_open_access_does_not_break():
    c = Corpus.from_dataframe(pd.DataFrame([_doc(1)]))
    assert len(acc.access_status(c)) == 2
    assert acc.access_routes(c)["documents"].sum() == 0
    assert acc.access_summary(c)["open_documents"] == 0
