"""Fixes from the formula audit, September 2026.

Each test carries a value computed BY HAND from the published definition,
and each one failed on the code from before the audit: that is what
proves the defect is fixed, not merely that the code runs.
"""

import math

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from bibliominer_analysis import Corpus
from bibliominer_analysis.io import parsers as P
from bibliominer_analysis.metrics import growth as gr
from bibliominer_analysis.metrics import laws as lw
from bibliominer_analysis.metrics import selfcitation as sc
from bibliominer_analysis.metrics import themes as th
from bibliominer_analysis.networks import analysis as na


def _doc(i, year=2020, authors="1:A.", source="J1", kws="", refs="",
         title=None, doi="", ids="", full=""):
    return {
        "Title": title or "Document number %d with a long enough title" % i,
        "Year": str(year), "Cited by": "0", "EID": "eid-%d" % i,
        "Source title": source, "Document Type": "Article",
        "Authors": authors, "Author(s) ID": ids, "Author full names": full,
        "Author Keywords": kws, "References": refs, "DOI": doi,
        "Affiliations": "parent 1: U, city: X, country: Morocco",
    }


def _corpus(rows):
    return Corpus.from_dataframe(pd.DataFrame(rows))


# --- Lotka: Kolmogorov-Smirnov test (Pao, 1985) ----------------------------

def test_ks_critical_value_on_the_number_of_authors():
    """1.36/√N with N = 1,000 authors, not N = 4 productivity levels.

    With the old N = 4, the critical value was 0.68 and D = 0.1 passed.
    """
    obs = np.array([0.70, 0.15, 0.10, 0.05])
    theo = np.array([0.60, 0.20, 0.12, 0.08])
    r = lw.kolmogorov_smirnov(obs, theo, sample_size=1000)
    assert r["critical_5pct"] == pytest.approx(1.36 / math.sqrt(1000), abs=1e-4)
    assert r["d"] == pytest.approx(0.10, abs=1e-4)
    assert r["follows_lotka"] is False


def test_lotka_ks_covers_levels_without_author():
    """A has 3 documents, B, C, D, E have 1; nobody has 2.

    Observed shares over 1..3: 0.8 / 0 / 0.2. Strict Lotka (1/x²)/ζ(2):
    0.6079 / 0.1520 / 0.0675. Cumulative: 0.8 / 0.8 / 1.0 against
    0.6079 / 0.7599 / 0.8275 -> D = 0.1921. By skipping level 2, the old
    computation compared 1.0 with 0.6755 and found D = 0.3245.
    """
    c = _corpus([_doc(1, authors="1:A.; 2:B."), _doc(2, authors="1:A.; 2:C."),
                 _doc(3, authors="1:A.; 2:D.; 3:E.")])
    ks = c.lotka()["ks_test_strict"]
    assert ks["d"] == pytest.approx(0.1921, abs=1e-4)
    assert ks["critical_5pct"] == pytest.approx(1.36 / math.sqrt(5), abs=1e-4)


# --- MCC (Savanur & Srikanth, 2010) ----------------------------------------

def test_mcc_a_is_the_total_number_of_authors():
    """3 articles, each with 2 authors, all different: 6 authors.

    CC = 1 - (3 × 1/2)/3 = 0.5. MCC = 0.5 × 6/5 = 0.6. With A = maximum per
    article (2), the old computation gave 1.0: "maximum collaboration".
    """
    c = _corpus([_doc(1, authors="1:A.; 2:B."), _doc(2, authors="1:C.; 2:D."),
                 _doc(3, authors="1:E.; 2:F.")])
    r = c.collaboration_indicators()
    assert r["collaborative_coefficient"] == pytest.approx(0.5)
    assert r["modified_collaborative_coefficient"] == pytest.approx(0.6)


# --- Bradford ---------------------------------------------------------------

def test_bradford_one_journal_filling_two_zones():
    """J1: 8 documents out of 10 fills zones 1 AND 2 (thresholds 3.33 and 6.67).

    The next two journals are therefore in zone 3. The old `if` only moved
    forward one zone, and put J2 in zone 2.
    """
    rows = [_doc(i, source="J1") for i in range(1, 9)]
    rows += [_doc(9, source="J2"), _doc(10, source="J3")]
    table = _corpus(rows).bradford()["table"]
    zones = dict(zip(table["source"], table["zone"]))
    assert zones == {"J1": 1, "J2": 3, "J3": 3}


# --- Eigenvector centrality -------------------------------------------------

def test_eigenvector_of_a_star():
    """A 3-branch star: λ = √3, vector (√3, 1, 1, 1)/√6.

    Centre = 0.7071, leaves = 0.4082. Bipartite graph: the old iteration
    oscillated without converging and stopped at 0.5 for all four nodes,
    centre and leaves indistinguishable.
    """
    A = np.zeros((4, 4))
    for leaf in (1, 2, 3):
        A[0, leaf] = A[leaf, 0] = 1.0
    x = na._eigenvector(A)
    assert x[0] == pytest.approx(math.sqrt(3) / math.sqrt(6), abs=1e-6)
    assert x[1] == pytest.approx(1 / math.sqrt(6), abs=1e-6)


def test_summary_modularity_follows_the_displayed_communities():
    graph = {
        "nodes": [{"id": "a", "community": 1}, {"id": "b", "community": 1},
                  {"id": "c", "community": 2}, {"id": "d", "community": 2}],
        "edges": [{"source": "a", "target": "b", "weight": 3},
                  {"source": "c", "target": "d", "weight": 3},
                  {"source": "b", "target": "c", "weight": 1}],
    }
    G = nx.Graph()
    for e in graph["edges"]:
        G.add_edge(e["source"], e["target"], weight=e["weight"])
    expected = nx.community.modularity(G, [{"a", "b"}, {"c", "d"}], weight="weight")
    assert na.graph_summary(graph)["modularity"] == pytest.approx(round(expected, 3))


# --- Callon's thematic map --------------------------------------------------

def test_callon_density_on_the_equivalence_index():
    """{a, b} co-occurring in 2 documents: e = 2²/(2·2) = 1, density = 100·1/2 = 50.
    {c, d} in 3 documents: e = 3²/(3·3) = 1, density = 50 too.

    On raw counts, the old computation gave 100 and 150: the more frequent
    theme looked more "developed" with an identical structure.
    """
    rows = [_doc(i, kws="a;b") for i in (1, 2)] + [_doc(i, kws="c;d") for i in (3, 4, 5)]
    clusters = th.thematic_map(_corpus(rows), min_weight=2)["clusters"]
    assert sorted(clusters["density"]) == [50.0, 50.0]


# --- Growth ----------------------------------------------------------------

def test_quantile_for_a_non_tabulated_confidence():
    """0.975 two-sided -> z = 2.2414; the old rounding to 0.97 gave 2.1705."""
    assert gr._z_score(0.975) == pytest.approx(2.2414, abs=5e-4)


def test_doubling_time_on_the_exact_rgr():
    """1,500 documents, then 1: RGR = ln(1501/1500) = 0.000666.

    Dt = ln 2 / RGR = 1,040.1 years. Computed on the RGR rounded to 0.0007,
    the old result was 990.2.
    """
    rows = [_doc(i, year=2020) for i in range(1, 1501)] + [_doc(1501, year=2021)]
    table = gr.rgr_doubling_time(_corpus(rows))
    dt = float(table.loc[table["year"] == 2021, "doubling_time"].iat[0])
    assert dt == pytest.approx(math.log(2) / math.log(1501 / 1500), abs=0.01)


def test_partial_last_year_flagged():
    from datetime import date
    this_year = date.today().year
    rows = [_doc(i, year=y) for i, y in enumerate(
        [this_year - 2, this_year - 1, this_year - 1, this_year], start=1)]
    fit = gr.trend_forecast(_corpus(rows))["fit"]
    assert fit["last_year_partial"] is True


# --- Matching the author columns --------------------------------------------

def test_a_missing_identifier_does_not_shift_the_following_ones():
    """Roe has no identifier: the column holds two for three authors.

    By rank, the old code gave Poe's identifier to Roe, and none to Poe.
    """
    a = P.parse_authors("Doe J.; Roe A.; Poe B.", "", "111111; 333333")
    assert [x["scopus_id"] for x in a] == [None, None, None]


def test_the_full_name_attached_by_surname():
    a = P.parse_authors("Doe J.; Roe A.; Poe B.",
                        "Doe, John (111111); Poe, Bob (333333)", "")
    assert [x["scopus_id"] for x in a] == ["111111", None, "333333"]


def test_the_bracketed_id_prevails_over_the_column():
    a = P.parse_authors("Doe J.; Roe A.", "Doe, John (111111); Roe, Ann (222222)",
                        "222222; 111111")
    assert [x["scopus_id"] for x in a] == ["111111", "222222"]


# --- Auto-citation ----------------------------------------------------------

def test_two_namesakes_do_not_self_cite():
    """Two "Smith J." with different identifiers: one cites the other.

    It is NOT a self-citation. Going by the name, the old computation counted
    it.
    """
    cited = _doc(1, authors="1:Smith J.", ids="1:111111", doi="10.1000/found",
                 title="The foundational paper that everybody cites here")
    citing = _doc(2, authors="1:Smith J.", ids="1:222222",
                  refs="ref1 | 10.1000/found | 2019 | Smith J. | "
                       "The foundational paper that everybody cites here")
    out = sc.authors_self_citation(_corpus([cited, citing]), min_citations=1)
    assert len(out) == 1
    assert int(out["self_citations"].iat[0]) == 0
