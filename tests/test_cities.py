"""Cities: rankings, collaboration scope, city network."""

import pandas as pd
import pytest

from bibliominer_analysis import Corpus
from bibliominer_analysis.metrics import cities as cit


def _doc(i, affs, cited="0", year=2020):
    return {
        "Title": "Document %d with a reasonably long title" % i,
        "Year": str(year), "Cited by": str(cited), "EID": "eid-%d" % i,
        "Source title": "J1", "Document Type": "Article", "Authors": "1:A.",
        "Author Keywords": "alpha", "Affiliations": affs,
    }


RABAT = "parent 1: Univ A, city: Rabat, country: Morocco"
MEKNES = "parent 1: Univ B, city: Meknes, country: Morocco"
RABAT2 = "parent 1: Univ C, city: Rabat, country: Morocco"
MADRID = "parent 1: Univ D, city: Madrid, country: Spain"


def test_reach_distinguishes_national_from_local():
    """The SCP/MCP split puts these two documents together; we do not."""
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, RABAT + ";" + RABAT2),      # two institutions, one city
        _doc(2, RABAT + ";" + MEKNES),      # two cities, one country
        _doc(3, RABAT + ";" + MADRID),      # two countries
        _doc(4, RABAT),                     # a single affiliation
    ]))
    s = c.collaboration_scale().set_index("scale")["documents"]
    assert s["Local (same city)"] == 1
    assert s["National (same country)"] == 1
    assert s["International"] == 1
    assert s["Single affiliation"] == 1


def test_reach_sums_to_one_hundred_percent():
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, RABAT + ";" + MEKNES), _doc(2, RABAT + ";" + MADRID),
    ]))
    assert c.collaboration_scale()["share"].sum() == pytest.approx(100.0, abs=0.2)


def test_reach_always_four_rows():
    """An empty category would read as missing data."""
    c = Corpus.from_dataframe(pd.DataFrame([_doc(1, RABAT)]))
    s = c.collaboration_scale()
    assert len(s) == 4
    assert list(s["scale"]) == ["Single affiliation", "Local (same city)",
                                "National (same country)", "International"]


def test_a_single_affiliation_is_not_a_local_collaboration():
    """Confusing them would artificially inflate local collaboration."""
    c = Corpus.from_dataframe(pd.DataFrame([_doc(1, RABAT), _doc(2, RABAT)]))
    s = c.collaboration_scale().set_index("scale")["documents"]
    assert s["Single affiliation"] == 2
    assert s["Local (same city)"] == 0


def test_top_cities_counts_the_institutions():
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, RABAT + ";" + RABAT2), _doc(2, RABAT),
    ]))
    row = c.top_cities().set_index("city").loc["Rabat"]
    assert row["documents"] == 2
    assert row["institutions"] == 2          # Univ A and Univ C
    assert row["country"] == "Morocco"


def test_a_document_counts_for_each_city():
    c = Corpus.from_dataframe(pd.DataFrame([_doc(1, RABAT + ";" + MEKNES)]))
    t = c.top_cities().set_index("city")["documents"]
    assert t["Rabat"] == 1 and t["Meknes"] == 1


def test_network_marks_national_and_international():
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, RABAT + ";" + MEKNES),
        _doc(2, RABAT + ";" + MADRID),
    ]))
    g = c.co_city(min_weight=1)
    scopes = {(e["source"], e["target"]): e["scope"] for e in g["edges"]}
    assert set(scopes.values()) == {"national", "international"}
    for (a, b), scope in scopes.items():
        if {a, b} == {"Rabat", "Meknes"}:
            assert scope == "national"
        if {a, b} == {"Rabat", "Madrid"}:
            assert scope == "international"


def test_network_carries_each_citys_country():
    c = Corpus.from_dataframe(pd.DataFrame([_doc(1, RABAT + ";" + MADRID)]))
    g = c.co_city(min_weight=1)
    country = {n["id"]: n["country"] for n in g["nodes"]}
    assert country["Rabat"] == "Morocco"
    assert country["Madrid"] == "Spain"


def test_hierarchy_relative_to_the_country_total():
    """The share is computed BEFORE any truncation, otherwise the denominator
    lies."""
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, RABAT), _doc(2, RABAT), _doc(3, RABAT), _doc(4, MEKNES),
    ]))
    h = c.city_hierarchy().set_index("city")
    assert h.loc["Rabat", "share_of_country"] == pytest.approx(75.0)
    assert h.loc["Meknes", "share_of_country"] == pytest.approx(25.0)


def test_cities_over_time_without_gap():
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, RABAT, year=2018), _doc(2, RABAT, year=2021),
    ]))
    t = c.cities_over_time(n=1)
    assert list(t["year"]) == [2018, 2019, 2020, 2021]
    assert list(t["cumulative"]) == [1, 1, 1, 2]


def test_corpus_without_city_does_not_break():
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, "parent 1: Univ A, country: Morocco"),
    ]))
    assert c.top_cities().empty
    assert c.cities_impact().empty
    assert c.co_city()["n_nodes"] == 0
    assert len(c.collaboration_scale()) == 4


def test_hierarchy_lists_every_institution_of_the_city():
    """The leader alone hid the other institutions of the city.

    Counted in AFFILIATIONS: a document co-signed by two laboratories of
    university A gives it two.
    """
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, RABAT + ";" + RABAT.replace("Univ A", "Univ A, subparent: Lab 2")),
        _doc(2, RABAT), _doc(3, RABAT2),
    ]))
    h = c.city_hierarchy().set_index("city")
    assert h.loc["Rabat", "institution_affiliations"] == "Univ A (3); Univ C (1)"
    assert h.loc["Rabat", "top_institution"] == "Univ A"
    assert h.loc["Rabat", "institutions"] == 2


def test_hierarchy_tie_broken_by_name():
    c = Corpus.from_dataframe(pd.DataFrame([_doc(1, RABAT2), _doc(2, RABAT)]))
    h = c.city_hierarchy().set_index("city")
    assert h.loc["Rabat", "institution_affiliations"] == "Univ A (1); Univ C (1)"
    assert h.loc["Rabat", "top_institution"] == "Univ A"
