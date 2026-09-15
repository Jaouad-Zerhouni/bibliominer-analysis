"""Croissance et collaboration.

Toutes les valeurs attendues sont calculées À LA MAIN depuis les formules
publiées. Un test d'indicateur qui reprend la sortie du code ne teste rien.
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
    """specs : liste de (année, nb_auteurs)."""
    rows = []
    for i, (year, n_auth) in enumerate(specs, start=1):
        names = "; ".join("%d:A%d." % (k, k) for k in range(1, n_auth + 1))
        rows.append(_doc(i, year, names))
    return Corpus.from_dataframe(pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# CAGR / AGR
# ---------------------------------------------------------------------------

def test_cagr_doublement_annuel():
    """1 doc en 2020, 2 en 2021, 4 en 2022 -> +100 %/an."""
    c = _corpus([(2020, 1)] + [(2021, 1)] * 2 + [(2022, 1)] * 4)
    assert c.cagr() == 100.0


def test_cagr_stable():
    c = _corpus([(2020, 1), (2021, 1), (2022, 1)])
    assert c.cagr() == 0.0


def test_cagr_une_seule_annee():
    assert _corpus([(2020, 1), (2020, 1)]).cagr() is None


def test_agr_annee_par_annee():
    """2 docs en 2020, 3 en 2021 -> +50 % ; 3 -> 6 -> +100 %."""
    c = _corpus([(2020, 1)] * 2 + [(2021, 1)] * 3 + [(2022, 1)] * 6)
    a = c.agr().set_index("year")
    assert pd.isna(a.loc[2020, "agr"])        # pas d'année précédente
    assert a.loc[2021, "agr"] == 50.0
    assert a.loc[2022, "agr"] == 100.0
    assert list(a["cumulative"]) == [2, 5, 11]


def test_agr_annee_vide_sans_division_par_zero():
    """Une année sans document ne doit pas produire un taux infini."""
    c = _corpus([(2020, 1), (2022, 1)])       # 2021 vide
    a = c.agr().set_index("year")
    assert a.loc[2021, "documents"] == 0
    assert pd.isna(a.loc[2022, "agr"])        # division par zéro évitée


# ---------------------------------------------------------------------------
# RGR / temps de doublement
# ---------------------------------------------------------------------------

def test_rgr_et_doublement():
    """Cumuls 1, 2, 4 -> RGR = ln2 chaque année -> doublement en 1 an."""
    c = _corpus([(2020, 1), (2021, 1), (2022, 1), (2022, 1)])
    r = c.rgr_doubling_time().set_index("year")
    assert list(r["cumulative"]) == [1, 2, 4]
    assert r.loc[2021, "rgr"] == pytest.approx(math.log(2), abs=1e-3)
    assert r.loc[2021, "doubling_time"] == pytest.approx(1.0, abs=0.02)
    assert r.loc[2022, "doubling_time"] == pytest.approx(1.0, abs=0.02)


def test_rgr_decroit_sur_croissance_lineaire():
    """À production constante, le RGR décroît : c'est le comportement attendu."""
    c = _corpus([(2020 + i, 1) for i in range(5)])
    r = c.rgr_doubling_time()["rgr"].dropna().tolist()
    assert r == sorted(r, reverse=True)


# ---------------------------------------------------------------------------
# Cochran
# ---------------------------------------------------------------------------

def test_cochran_population_infinie():
    """n0 = 1.96² × 0.25 / 0.05² = 384,16 -> 385."""
    c = _corpus([(2020, 1)] * 10)
    r = c.cochran_sample_size()
    assert r["sample_size_infinite"] == 385
    assert r["z_score"] == 1.96


def test_cochran_correction_population_finie():
    """Sur 100 documents : n = 385 / (1 + 384/100) = 79,5 -> 80."""
    c = _corpus([(2020, 1)] * 100)
    r = c.cochran_sample_size()
    assert r["population"] == 100
    assert r["sample_size"] == 80


def test_cochran_ne_depasse_jamais_le_corpus():
    c = _corpus([(2020, 1)] * 10)
    r = c.cochran_sample_size()
    assert r["sample_size"] <= 10
    assert r["sampling_fraction"] <= 100


def test_cochran_marge_plus_serree_demande_plus():
    c = _corpus([(2020, 1)] * 1000)
    large = c.cochran_sample_size(margin=0.10)["sample_size"]
    tight = c.cochran_sample_size(margin=0.03)["sample_size"]
    assert tight > large


# ---------------------------------------------------------------------------
# Collaboration
# ---------------------------------------------------------------------------

def test_degre_de_collaboration():
    """3 documents co-signés sur 4 -> C = 0,75."""
    c = _corpus([(2020, 1), (2020, 2), (2020, 3), (2020, 4)])
    r = c.collaboration_indicators()
    assert r["single_authored"] == 1
    assert r["multi_authored"] == 3
    assert r["degree_of_collaboration"] == 0.75


def test_indice_de_collaboration_lawani():
    """CI = Σ(j·f_j)/N = (1+2+3+4)/4 = 2,5 auteurs par document."""
    c = _corpus([(2020, 1), (2020, 2), (2020, 3), (2020, 4)])
    assert c.collaboration_indicators()["collaboration_index"] == 2.5


def test_coefficient_de_collaboration():
    """CC = 1 − Σ(f_j/j)/N = 1 − (1/1+1/2+1/3+1/4)/4 = 1 − 0,520833 = 0,479167."""
    c = _corpus([(2020, 1), (2020, 2), (2020, 3), (2020, 4)])
    r = c.collaboration_indicators()
    assert r["collaborative_coefficient"] == pytest.approx(0.4792, abs=1e-4)
    # MCC = CC × A/(A−1) = 0,479167 × 4/3
    assert r["modified_collaborative_coefficient"] == pytest.approx(0.6389, abs=1e-4)


def test_tout_en_solo():
    c = _corpus([(2020, 1), (2021, 1)])
    r = c.collaboration_indicators()
    assert r["degree_of_collaboration"] == 0.0
    assert r["collaboration_index"] == 1.0
    assert r["collaborative_coefficient"] == 0.0
    assert r["modified_collaborative_coefficient"] == 0.0     # A = 1


def test_aapp():
    """3 documents, 2 auteurs distincts -> 1,5 document par auteur."""
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
    """Deux périodes de composition identique -> CAI = 100 partout."""
    specs = [(2000, 1), (2000, 2)] + [(2010, 1), (2010, 2)]
    c = _corpus(specs)
    t = c.cai(block_years=5)
    assert len(t) >= 2
    for col in ("cai_single", "cai_two"):
        vals = t[col].dropna().tolist()
        assert all(v == 100.0 for v in vals), (col, vals)


def test_cai_sur_representation():
    """Période 1 : que du solo. Période 2 : que du co-signé."""
    specs = [(2000, 1), (2000, 1)] + [(2010, 2), (2010, 2)]
    t = _corpus(specs).cai(block_years=5).set_index("period")
    p1, p2 = t.index[0], t.index[-1]
    assert t.loc[p1, "cai_single"] == 200.0     # 2× la part attendue
    assert t.loc[p2, "cai_single"] == 0.0
    assert t.loc[p2, "cai_two"] == 200.0


def test_cai_corpus_vide():
    c = Corpus.from_dataframe(pd.DataFrame([{"Title": "x"}]))
    assert c.cai().empty


# ---------------------------------------------------------------------------
# Loi de Price
# ---------------------------------------------------------------------------

def test_price_law():
    """9 auteurs -> noyau théorique = √9 = 3."""
    rows = []
    for i in range(9):
        rows.append(_doc(i, 2020, "1:A%d." % i))
    c = Corpus.from_dataframe(pd.DataFrame(rows))
    r = c.price_law()
    assert r["authors"] == 9
    assert r["expected_core"] == 3
    # 9 auteurs à une signature : 3 d'entre eux font 3/9 = 33,3 %
    assert r["observed_share"] == pytest.approx(33.3, abs=0.1)


def test_growth_summary_corpus_vide():
    c = Corpus.from_dataframe(pd.DataFrame([{"Title": "x"}]))
    s = c.growth_summary()
    assert s["cagr"] is None and s["rgr_mean"] is None


# ---------------------------------------------------------------------------
# Tendance par moindres carrés
# ---------------------------------------------------------------------------

def test_trend_lineaire_parfait():
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


def test_trend_exponentiel():
    """1, 2, 4, 8 -> facteur multiplicatif annuel = 2."""
    specs = []
    for i, n in enumerate([1, 2, 4, 8]):
        specs += [(2020 + i, 1)] * n
    res = _corpus(specs).trend_forecast(horizon=1, model="exponential")
    assert res["fit"]["slope"] == pytest.approx(2.0, abs=1e-3)
    assert res["fit"]["r2"] == pytest.approx(1.0, abs=1e-3)


def test_trend_projection_jamais_negative():
    """Une pente descendante ne doit pas prédire un nombre négatif."""
    specs = []
    for i, n in enumerate([10, 6, 2]):
        specs += [(2020 + i, 1)] * n
    res = _corpus(specs).trend_forecast(horizon=5)
    assert (res["table"]["fitted"] >= 0).all()


def test_trend_exige_trois_annees():
    """Deux points donnent toujours un R² de 1 : ce n'est pas une tendance."""
    res = _corpus([(2020, 1), (2021, 1)]).trend_forecast()
    assert res["fit"] is None
    assert "trois" in res["message"].lower()
