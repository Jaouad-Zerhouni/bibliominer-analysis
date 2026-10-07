"""Growth and collaboration.

All expected values are computed BY HAND from the published formulas. An
indicator test that reuses the code's output tests nothing.
"""

import math

import pandas as pd
import pytest

from bibliominer_analysis import Corpus


def _doc(i, year, authors):
    return {
        "Title": "Doc %d" % i, "Year": str(year), "Cited by": "0",
        "EID": "eid-%d" % i, "Source title": "J", "Document Type": "Article",
        "Authors": authors,
        "Affiliations": "parent 1: U, city: X, country: Morocco",
    }


def _corpus(specs):
    """specs: list of (year, number of authors)."""
    rows = []
    for i, (year, n_auth) in enumerate(specs, start=1):
        names = "; ".join("%d:A%d." % (k, k) for k in range(1, n_auth + 1))
        rows.append(_doc(i, year, names))
    return Corpus.from_dataframe(pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# CAGR / AGR
# ---------------------------------------------------------------------------

def test_cagr_yearly_doubling():
    """1 doc en 2020, 2 en 2021, 4 en 2022 -> +100 %/an."""
    c = _corpus([(2020, 1)] + [(2021, 1)] * 2 + [(2022, 1)] * 4)
    assert c.cagr() == 100.0


def test_cagr_stable():
    c = _corpus([(2020, 1), (2021, 1), (2022, 1)])
    assert c.cagr() == 0.0


def test_cagr_single_year():
    assert _corpus([(2020, 1), (2020, 1)]).cagr() is None


def test_agr_year_by_year():
    """2 docs en 2020, 3 en 2021 -> +50 % ; 3 -> 6 -> +100 %."""
    c = _corpus([(2020, 1)] * 2 + [(2021, 1)] * 3 + [(2022, 1)] * 6)
    a = c.agr().set_index("year")
    assert pd.isna(a.loc[2020, "agr"])        # no previous year
    assert a.loc[2021, "agr"] == 50.0
    assert a.loc[2022, "agr"] == 100.0
    assert list(a["cumulative"]) == [2, 5, 11]


def test_agr_empty_year_without_division_by_zero():
    """A year without documents must not produce an infinite rate."""
    c = _corpus([(2020, 1), (2022, 1)])       # 2021 empty
    a = c.agr().set_index("year")
    assert a.loc[2021, "documents"] == 0
    assert pd.isna(a.loc[2022, "agr"])        # division by zero avoided


# ---------------------------------------------------------------------------
# RGR / doubling time
# ---------------------------------------------------------------------------

def test_rgr_and_doubling():
    """Cumulative 1, 2, 4 -> RGR = ln2 every year -> doubling in 1 year."""
    c = _corpus([(2020, 1), (2021, 1), (2022, 1), (2022, 1)])
    r = c.rgr_doubling_time().set_index("year")
    assert list(r["cumulative"]) == [1, 2, 4]
    assert r.loc[2021, "rgr"] == pytest.approx(math.log(2), abs=1e-3)
    assert r.loc[2021, "doubling_time"] == pytest.approx(1.0, abs=0.02)
    assert r.loc[2022, "doubling_time"] == pytest.approx(1.0, abs=0.02)


def test_rgr_decreases_on_linear_growth():
    """At constant production, RGR decreases: that is the expected behaviour."""
    c = _corpus([(2020 + i, 1) for i in range(5)])
    r = c.rgr_doubling_time()["rgr"].dropna().tolist()
    assert r == sorted(r, reverse=True)


# ---------------------------------------------------------------------------
# Cochran
# ---------------------------------------------------------------------------

def test_cochran_infinite_population():
    """n0 = 1.96² × 0.25 / 0.05² = 384,16 -> 385."""
    c = _corpus([(2020, 1)] * 10)
    r = c.cochran_sample_size()
    assert r["sample_size_infinite"] == 385
    assert r["z_score"] == 1.96


def test_cochran_finite_population_correction():
    """Over 100 documents: n = 385 / (1 + 384/100) = 79.5 -> 80."""
    c = _corpus([(2020, 1)] * 100)
    r = c.cochran_sample_size()
    assert r["population"] == 100
    assert r["sample_size"] == 80


def test_cochran_never_exceeds_the_corpus():
    c = _corpus([(2020, 1)] * 10)
    r = c.cochran_sample_size()
    assert r["sample_size"] <= 10
    assert r["sampling_fraction"] <= 100


def test_cochran_tighter_margin_needs_more():
    c = _corpus([(2020, 1)] * 1000)
    large = c.cochran_sample_size(margin=0.10)["sample_size"]
    tight = c.cochran_sample_size(margin=0.03)["sample_size"]
    assert tight > large


# ---------------------------------------------------------------------------
# Collaboration
# ---------------------------------------------------------------------------

def test_degree_of_collaboration():
    """3 co-authored documents out of 4 -> C = 0.75."""
    c = _corpus([(2020, 1), (2020, 2), (2020, 3), (2020, 4)])
    r = c.collaboration_indicators()
    assert r["single_authored"] == 1
    assert r["multi_authored"] == 3
    assert r["degree_of_collaboration"] == 0.75


def test_lawani_collaboration_index():
    """CI = Σ(j·f_j)/N = (1+2+3+4)/4 = 2.5 authors per document."""
    c = _corpus([(2020, 1), (2020, 2), (2020, 3), (2020, 4)])
    assert c.collaboration_indicators()["collaboration_index"] == 2.5


def test_collaboration_coefficient():
    """CC = 1 − Σ(f_j/j)/N = 1 − (1/1+1/2+1/3+1/4)/4 = 1 − 0,520833 = 0,479167."""
    c = _corpus([(2020, 1), (2020, 2), (2020, 3), (2020, 4)])
    r = c.collaboration_indicators()
    assert r["collaborative_coefficient"] == pytest.approx(0.4792, abs=1e-4)
    # MCC = CC × A/(A−1) = 0,479167 × 4/3
    assert r["modified_collaborative_coefficient"] == pytest.approx(0.6389, abs=1e-4)


def test_all_solo():
    c = _corpus([(2020, 1), (2021, 1)])
    r = c.collaboration_indicators()
    assert r["degree_of_collaboration"] == 0.0
    assert r["collaboration_index"] == 1.0
    assert r["collaborative_coefficient"] == 0.0
    assert r["modified_collaborative_coefficient"] == 0.0     # A = 1


def test_aapp():
    """3 documents, 2 distinct authors -> 1.5 documents per author."""
    rows = [_doc(1, 2020, "1:A."), _doc(2, 2020, "1:A."), _doc(3, 2020, "1:B.")]
    c = Corpus.from_dataframe(pd.DataFrame(rows))
    r = c.collaboration_indicators()
    assert r["authors"] == 2
    assert r["aapp"] == 1.5


def test_authorship_pattern():
    c = _corpus([(2020, 1), (2020, 2), (2020, 2)])
    p = c.authorship_pattern().set_index("authors")
    assert p.loc[1, "documents"] == 1
    assert p.loc[2, "documents"] == 2
    assert p.loc[2, "share"] == pytest.approx(66.67, abs=0.01)


# ---------------------------------------------------------------------------
# CAI
# ---------------------------------------------------------------------------

def test_cai_reference_a_cent():
    """Two periods with identical composition -> CAI = 100 everywhere."""
    specs = [(2000, 1), (2000, 2)] + [(2010, 1), (2010, 2)]
    c = _corpus(specs)
    t = c.cai(block_years=5)
    assert len(t) >= 2
    for col in ("cai_single", "cai_two"):
        vals = t[col].dropna().tolist()
        assert all(v == 100.0 for v in vals), (col, vals)


def test_cai_over_representation():
    """Period 1: single authors only. Period 2: co-authored only."""
    specs = [(2000, 1), (2000, 1)] + [(2010, 2), (2010, 2)]
    t = _corpus(specs).cai(block_years=5).set_index("period")
    p1, p2 = t.index[0], t.index[-1]
    assert t.loc[p1, "cai_single"] == 200.0     # 2× the expected share
    assert t.loc[p2, "cai_single"] == 0.0
    assert t.loc[p2, "cai_two"] == 200.0


def test_cai_empty_corpus():
    c = Corpus.from_dataframe(pd.DataFrame([{"Title": "x"}]))
    assert c.cai().empty


# ---------------------------------------------------------------------------
# Price's law
# ---------------------------------------------------------------------------

def test_price_law():
    """9 authors -> theoretical core = √9 = 3."""
    rows = []
    for i in range(9):
        rows.append(_doc(i, 2020, "1:A%d." % i))
    c = Corpus.from_dataframe(pd.DataFrame(rows))
    r = c.price_law()
    assert r["authors"] == 9
    assert r["expected_core"] == 3
    # 9 authors with one signature each: 3 of them make 3/9 = 33.3 %
    assert r["observed_share"] == pytest.approx(33.3, abs=0.1)


def test_growth_summary_empty_corpus():
    c = Corpus.from_dataframe(pd.DataFrame([{"Title": "x"}]))
    s = c.growth_summary()
    assert s["cagr"] is None and s["rgr_mean"] is None


# ---------------------------------------------------------------------------
# Least-squares trend
# ---------------------------------------------------------------------------

def test_trend_perfect_linear():
    """2, 4, 6, 8 documents -> pente exactement 2, R² = 1."""
    specs = []
    for i, n in enumerate([2, 4, 6, 8]):
        specs += [(2020 + i, 1)] * n
    c = _corpus(specs)
    res = c.trend_forecast(horizon=2)
    assert res["fit"]["slope"] == pytest.approx(2.0, abs=1e-6)
    assert res["fit"]["r2"] == pytest.approx(1.0, abs=1e-6)
    t = res["table"]
    fut = t[t["kind"] == "forecast"]
    assert list(fut["year"]) == [2024, 2025]
    assert fut.iloc[0]["fitted"] == pytest.approx(10.0, abs=1e-6)
    assert fut.iloc[1]["fitted"] == pytest.approx(12.0, abs=1e-6)


def test_trend_exponential():
    """1, 2, 4, 8 -> facteur multiplicatif annuel = 2."""
    specs = []
    for i, n in enumerate([1, 2, 4, 8]):
        specs += [(2020 + i, 1)] * n
    res = _corpus(specs).trend_forecast(horizon=1, model="exponential")
    assert res["fit"]["slope"] == pytest.approx(2.0, abs=1e-3)
    assert res["fit"]["r2"] == pytest.approx(1.0, abs=1e-3)


def test_trend_projection_never_negative():
    """A downward slope must not predict a negative number."""
    specs = []
    for i, n in enumerate([10, 6, 2]):
        specs += [(2020 + i, 1)] * n
    res = _corpus(specs).trend_forecast(horizon=5)
    assert (res["table"]["fitted"] >= 0).all()


def test_trend_requires_three_years():
    """Two points always give an R² of 1: that is not a trend."""
    res = _corpus([(2020, 1), (2021, 1)]).trend_forecast()
    assert res["fit"] is None
    # The message is shown as is in the interface, which is in English.
    assert "three years" in res["message"].lower()


def test_a_year_without_any_document_does_not_break_production():
    """Found: a year WITHOUT documents in the series (2014, 2016, no 2015)
    broke `by_year`, an error 500 on the Production screen."""
    import pandas as pd
    from bibliominer_analysis import Corpus
    rows = [{"EID": "e1", "Title": "A", "Year": "2014", "Cited by": "4",
             "Authors": "A.", "Source title": "J", "Document Type": "Article"},
            {"EID": "e2", "Title": "B", "Year": "2016", "Cited by": "2",
             "Authors": "B.", "Source title": "J", "Document Type": "Article"}]
    g = Corpus.from_dataframe(pd.DataFrame(rows)).production_by_year()
    assert list(g["year"]) == [2014, 2015, 2016]
    assert list(g["documents"]) == [1, 0, 1]
    assert pd.isna(g.loc[g["year"] == 2015, "citations_per_doc"]).all()
    assert g.loc[g["year"] == 2014, "citations_per_doc"].iloc[0] == 4.0
