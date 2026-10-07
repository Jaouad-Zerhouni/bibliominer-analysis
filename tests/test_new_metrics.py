"""Journals, timelines, RPYS, factorial analysis, annotated networks."""

import numpy as np
import pandas as pd
import pytest

from bibliominer_analysis import Corpus
from bibliominer_analysis.metrics import factorial as fac
from bibliominer_analysis.metrics import impact as imp
from bibliominer_analysis.metrics import sources as src
from bibliominer_analysis.metrics import spectroscopy as spec
from bibliominer_analysis.metrics import timeline as tl
from bibliominer_analysis.networks import analysis as netan


def _doc(i, year, cited="0", source="J1", authors="1:A.", kws="alpha", refs=""):
    return {
        "Title": "Document number %d with a reasonably long title" % i,
        "Year": str(year), "Cited by": str(cited), "EID": "eid-%d" % i,
        "Source title": source, "Document Type": "Article", "Authors": authors,
        "Author Keywords": kws, "References": refs,
        "Affiliations": "parent 1: U, city: X, country: Morocco",
    }


# --------------------------------------------------------------- m-index ----

def test_m_index_relates_h_to_age():
    assert imp.m_index(10, 2016, 2025) == 1.0     # 10 ans -> 1,0
    assert imp.m_index(5, 2021, 2025) == 1.0      # 5 ans  -> 1,0
    assert imp.m_index(3, 2024, 2025) == 1.5


def test_m_index_absent_without_year():
    assert imp.m_index(5, None, 2025) is None
    assert imp.m_index(5, 2020, None) is None


def test_m_index_uses_the_last_year_of_the_corpus():
    """A corpus that stops in 2020 must not see its m drop every January."""
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, 2018, cited="5"), _doc(2, 2020, cited="5"),
    ]))
    row = src.sources_impact(c).iloc[0]
    # h = 2 over 2018-2020, i.e. 3 years: 2/3 = 0.67
    assert row["m_index"] == pytest.approx(0.67, abs=0.01)


# ---------------------------------------------------------------- journals --

def test_sources_over_time_fills_the_empty_years():
    """Without the zeros, the cumulative curve would be wrong."""
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, 2018, source="J1"), _doc(2, 2021, source="J1"),
    ]))
    t = src.sources_over_time(c, n=1)
    assert list(t["year"]) == [2018, 2019, 2020, 2021]
    assert list(t["documents"]) == [1, 0, 0, 1]
    assert list(t["cumulative"]) == [1, 1, 1, 2]


# ----------------------------------------------------------- chronologies ----

def test_word_dynamics_is_cumulative_and_without_gap():
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, 2018, kws="alpha"), _doc(2, 2020, kws="alpha"),
    ]))
    t = tl.word_dynamics(c, n=1)
    assert list(t["cumulative"]) == [1, 1, 2]


def test_authors_over_time_omits_years_without_publication():
    """Here the absence of a point IS the information: an interruption."""
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, 2018, authors="1:A."), _doc(2, 2021, authors="1:A."),
    ]))
    t = tl.authors_over_time(c, n=1)
    assert list(t["year"]) == [2018, 2021]


def test_mean_citations_corrected_for_age():
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, 2020, cited="10"), _doc(2, 2024, cited="10"),
    ]))
    t = tl.average_citations_per_year(c).set_index("year")
    # 2020 had 5 years to accumulate, 2024 only one
    assert t.loc[2020, "citable_years"] == 5
    assert t.loc[2020, "mean_citations_per_year"] == 2.0
    assert t.loc[2024, "mean_citations_per_year"] == 10.0


# ------------------------------------------------------------------ RPYS ----

def test_rpys_detects_a_peak():
    """An over-cited year must stand out through its deviation, not its total."""
    refs = ";".join(
        ["ref%d |  | 1990 | X | title %d" % (i, i) for i in range(20)] +
        ["ref%d |  | %d | X | title %d" % (i, 1988 + (i % 5), i) for i in range(20, 30)]
    )
    c = Corpus.from_dataframe(pd.DataFrame([_doc(1, 2020, refs=refs)]))
    t = spec.reference_spectroscopy(c).set_index("year")
    assert t.loc[1990, "deviation"] > 0
    assert bool(t.loc[1990, "is_peak"])


def test_rpys_discards_outlier_years():
    refs = "ref1 |  | 1500 | X | t1;ref2 |  | 2015 | X | t2"
    c = Corpus.from_dataframe(pd.DataFrame([_doc(1, 2020, refs=refs)]))
    years = list(spec.reference_spectroscopy(c)["year"])
    assert 1500 not in years


# ------------------------------------------------------- analyse factorielle --

@pytest.fixture
def themed():
    """Two disjoint families of terms: the CA must separate them."""
    rows = []
    for i in range(6):
        rows.append(_doc(i, 2020, kws="alpha;beta;gamma"))
    for i in range(6, 12):
        rows.append(_doc(i, 2021, kws="delta;epsilon;zeta"))
    return Corpus.from_dataframe(pd.DataFrame(rows))


def test_ca_separates_two_families(themed):
    r = fac.conceptual_structure(themed, "CA", top_n=10, min_documents=2)
    assert len(r["terms"]) == 6
    groups = {}
    for _, row in r["terms"].iterrows():
        groups.setdefault(row["cluster"], set()).add(row["keyword"])
    families = [{"alpha", "beta", "gamma"}, {"delta", "epsilon", "zeta"}]
    for members in groups.values():
        assert any(members <= f for f in families), members


def test_ca_is_deterministic(themed):
    a = fac.conceptual_structure(themed, "CA", top_n=10)
    b = fac.conceptual_structure(themed, "CA", top_n=10)
    pd.testing.assert_frame_equal(a["terms"], b["terms"])


def test_mds_also_produces_two_dimensions(themed):
    r = fac.conceptual_structure(themed, "MDS", top_n=10, min_documents=2)
    assert r["method"] == "MDS"
    assert len(r["explained"]) == 2
    assert set(r["terms"].columns) >= {"keyword", "dim1", "dim2", "cluster"}


def test_too_small_corpus_returns_empty():
    c = Corpus.from_dataframe(pd.DataFrame([_doc(1, 2020, kws="alpha")]))
    r = fac.conceptual_structure(c, "CA")
    assert r["terms"].empty and r["clusters"] == []


# ------------------------------------------------------ networks: measures ----

@pytest.fixture
def bridge_graph():
    """Two triangles joined by a single node: B is the BRIDGE."""
    return {
        "nodes": [{"id": x, "label": x, "occurrences": 3, "degree": 2}
                  for x in ("A", "B", "C", "D", "E")],
        "edges": [
            {"source": "A", "target": "B", "weight": 1},
            {"source": "B", "target": "C", "weight": 1},
            {"source": "C", "target": "D", "weight": 1},
            {"source": "D", "target": "E", "weight": 1},
        ],
        "n_nodes": 5, "n_edges": 4,
    }


def test_betweenness_finds_the_bridge(bridge_graph):
    g = netan.annotate(bridge_graph)
    b = {n["id"]: n["betweenness"] for n in g["nodes"]}
    assert b["C"] > b["B"] > b["A"]
    assert b["A"] == 0.0 and b["E"] == 0.0


def test_pagerank_varies_and_sums_to_one(bridge_graph):
    """A uniform PageRank would signal a fallback computation, hence a bug."""
    g = netan.annotate(bridge_graph)
    pr = [n["pagerank"] for n in g["nodes"]]
    assert len(set(pr)) > 1, "uniform PageRank = computation not done"
    assert sum(pr) == pytest.approx(1.0, abs=1e-3)


def test_normalisation_keeps_the_raw_weight(bridge_graph):
    g = netan.normalize(bridge_graph, "association")
    for e in g["edges"]:
        assert e["raw_weight"] == 1
        # 1 / (3 x 3)
        assert e["weight"] == pytest.approx(1 / 9)
    assert g["normalization"] == "association"


def test_known_normalisations(bridge_graph):
    for method in netan.NORMALIZATIONS:
        g = netan.normalize(bridge_graph, method)
        assert len(g["edges"]) == 4, method


def test_graph_summary(bridge_graph):
    s = netan.graph_summary(bridge_graph)
    assert s["nodes"] == 5 and s["edges"] == 4
    assert s["components"] == 1
    assert s["diameter"] == 4


def test_overlay_by_year():
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, 2018, kws="alpha"), _doc(2, 2022, kws="alpha"),
        _doc(3, 2020, kws="beta"),
    ]))
    years = netan.overlay_years(c, "keywords")
    assert years["alpha"] == 2020.0
    assert years["beta"] == 2020.0


# ------------------------------------------------------- e / i10 / groups ---

def test_e_index_captures_the_core_excess():
    """Two identical h, very different excesses: e must separate them."""
    from bibliominer_analysis.metrics.impact import e_index, h_index

    steady = [3, 3, 3]        # h = 3, sum of the core = 9 = h^2 -> e = 0
    boosted = [100, 100, 100]     # h = 3, sum = 300 -> e = sqrt(291)
    assert h_index(steady) == h_index(boosted) == 3
    assert e_index(steady) == 0.0
    assert e_index(boosted) == pytest.approx((300 - 9) ** 0.5, abs=0.01)


def test_e_index_never_negative():
    """Every article of the core has at least h citations: the sum reaches h^2."""
    from bibliominer_analysis.metrics.impact import e_index
    for cites in ([], [0], [1], [5, 4, 3, 2, 1], [7, 7, 7, 1]):
        assert e_index(cites) >= 0.0


def test_complete_per_author_indices():
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, 2018, cited="30", authors="1:A."),
        _doc(2, 2019, cited="12", authors="1:A."),
        _doc(3, 2020, cited="2", authors="1:A."),
    ]))
    row = c.authors_impact(1).iloc[0]
    for col in ("h_index", "g_index", "i10_index", "e_index", "m_index"):
        assert col in row.index, col
    assert row["i10_index"] == 2          # 30 and 12 exceed 10
    assert row["h_index"] == 2


def test_signatory_groups_always_four_rows():
    """A missing category must be 0, not disappear."""
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, 2020, authors="1:A."),
        _doc(2, 2020, authors="1:A.; 2:B.; 3:C.; 4:D.; 5:E."),
    ]))
    g = c.authorship_groups()
    assert list(g["group"]) == ["1 author", "2 authors", "3 authors", "4+ authors"]
    counts = dict(zip(g["group"], g["documents"]))
    assert counts["1 author"] == 1
    assert counts["2 authors"] == 0
    assert counts["4+ authors"] == 1
    assert g["share"].sum() == pytest.approx(100.0, abs=0.2)


def test_groups_aggregate_beyond_four():
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, 2020, authors="1:A.; 2:B.; 3:C.; 4:D."),
        _doc(2, 2020, authors="1:A.; 2:B.; 3:C.; 4:D.; 5:E.; 6:F."),
    ]))
    g = c.authorship_groups().set_index("group")
    assert g.loc["4+ authors", "documents"] == 2


def test_authorship_ranks_sum_to_the_documents():
    """The invariant that makes the table checkable by eye."""
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, 2020, authors="1:A.; 2:B."),
        _doc(2, 2020, authors="1:B.; 2:A."),
        _doc(3, 2020, authors="1:C.; 2:B.; 3:A."),
        _doc(4, 2020, authors="1:C.; 2:D.; 3:E.; 4:A."),
    ]))
    a = c.authors_impact(None).set_index("author")
    total = (a["first_author"] + a["second_author"]
             + a["third_author"] + a["later_author"])
    assert (total == a["documents"]).all()

    author_row = a.loc["A."]
    assert author_row["documents"] == 4
    assert author_row["first_author"] == 1
    assert author_row["second_author"] == 1
    assert author_row["third_author"] == 1
    assert author_row["later_author"] == 1


def test_supervisor_profile_visible():
    """Never first nor second: that is what the single column hid."""
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(i, 2020, authors="1:X.; 2:Y.; 3:Senior.") for i in range(5)
    ]))
    senior = c.authors_impact(None).set_index("author").loc["Senior."]
    assert senior["first_author"] == 0
    assert senior["second_author"] == 0
    assert senior["third_author"] == 5
