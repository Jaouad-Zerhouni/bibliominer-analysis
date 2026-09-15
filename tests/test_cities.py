"""Les villes : classements, portee de collaboration, reseau inter-villes."""

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


def test_portee_distingue_national_de_local():
    """Le partage SCP/MCP range ces deux documents ensemble ; pas nous."""
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, RABAT + ";" + RABAT2),      # deux institutions, une ville
        _doc(2, RABAT + ";" + MEKNES),      # deux villes, un pays
        _doc(3, RABAT + ";" + MADRID),      # deux pays
        _doc(4, RABAT),                     # une seule affiliation
    ]))
    s = c.collaboration_scale().set_index("scale")["documents"]
    assert s["Local (same city)"] == 1
    assert s["National (same country)"] == 1
    assert s["International"] == 1
    assert s["Single affiliation"] == 1


def test_portee_somme_a_cent_pour_cent():
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, RABAT + ";" + MEKNES), _doc(2, RABAT + ";" + MADRID),
    ]))
    assert c.collaboration_scale()["share"].sum() == pytest.approx(100.0, abs=0.2)


def test_portee_toujours_quatre_lignes():
    """Une categorie vide se lirait comme une donnee manquante."""
    c = Corpus.from_dataframe(pd.DataFrame([_doc(1, RABAT)]))
    s = c.collaboration_scale()
    assert len(s) == 4
    assert list(s["scale"]) == ["Single affiliation", "Local (same city)",
                                "National (same country)", "International"]


def test_une_seule_affiliation_n_est_pas_une_collaboration_locale():
    """Les confondre gonflerait artificiellement la collaboration locale."""
    c = Corpus.from_dataframe(pd.DataFrame([_doc(1, RABAT), _doc(2, RABAT)]))
    s = c.collaboration_scale().set_index("scale")["documents"]
    assert s["Single affiliation"] == 2
    assert s["Local (same city)"] == 0


def test_top_cities_compte_les_institutions():
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, RABAT + ";" + RABAT2), _doc(2, RABAT),
    ]))
    row = c.top_cities().set_index("city").loc["Rabat"]
    assert row["documents"] == 2
    assert row["institutions"] == 2          # Univ A et Univ C
    assert row["country"] == "Morocco"


def test_un_document_compte_pour_chaque_ville():
    c = Corpus.from_dataframe(pd.DataFrame([_doc(1, RABAT + ";" + MEKNES)]))
    t = c.top_cities().set_index("city")["documents"]
    assert t["Rabat"] == 1 and t["Meknes"] == 1


def test_reseau_marque_national_et_international():
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


def test_reseau_porte_le_pays_de_chaque_ville():
    c = Corpus.from_dataframe(pd.DataFrame([_doc(1, RABAT + ";" + MADRID)]))
    g = c.co_city(min_weight=1)
    country = {n["id"]: n["country"] for n in g["nodes"]}
    assert country["Rabat"] == "Morocco"
    assert country["Madrid"] == "Spain"


def test_hierarchie_rapporte_au_total_du_pays():
    """La part se calcule AVANT toute troncature, sinon le denominateur ment."""
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, RABAT), _doc(2, RABAT), _doc(3, RABAT), _doc(4, MEKNES),
    ]))
    h = c.city_hierarchy().set_index("city")
    assert h.loc["Rabat", "share_of_country"] == pytest.approx(75.0)
    assert h.loc["Meknes", "share_of_country"] == pytest.approx(25.0)


def test_villes_dans_le_temps_sans_trou():
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, RABAT, year=2018), _doc(2, RABAT, year=2021),
    ]))
    t = c.cities_over_time(n=1)
    assert list(t["year"]) == [2018, 2019, 2020, 2021]
    assert list(t["cumulative"]) == [1, 1, 1, 2]


def test_corpus_sans_ville_ne_casse_pas():
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, "parent 1: Univ A, country: Morocco"),
    ]))
    assert c.top_cities().empty
    assert c.cities_impact().empty
    assert c.co_city()["n_nodes"] == 0
    assert len(c.collaboration_scale()) == 4
