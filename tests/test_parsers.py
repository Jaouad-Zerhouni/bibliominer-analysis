"""Tests des parseurs : c'est ici que se joue la fidélité au format.

Les cas ne sont pas inventés — ils viennent d'exports réels, y compris les
cas dégradés (ancien format non indexé, affiliation brute, référence non
réconciliée), parce que l'utilisateur a des fichiers dans les deux états.
"""

from bibliominer_analysis.io import parsers as P
from bibliominer_analysis.io.schema import INDEPENDENT_LABEL


# --- auteurs ---------------------------------------------------------------

def test_authors_indexes_bibliominer():
    a = P.parse_authors(
        "1:Idri A.; 2:Hosni M.; 3:Abran A.",
        "1:Idri, Ali (6602789810); 2:Hosni, Mohamed (57189341317); 3:Abran, Alain (7004233119)",
        "1:6602789810; 2:57189341317; 3:7004233119",
    )
    assert [x["position"] for x in a] == [1, 2, 3]
    assert [x["name"] for x in a] == ["Idri A.", "Hosni M.", "Abran A."]
    assert a[0]["full_name"] == "Idri, Ali"
    assert a[0]["scopus_id"] == "6602789810"


def test_authors_scopus_brut_sans_index():
    """Ancien export : pas de préfixe « n: ». La position retombe sur le rang."""
    a = P.parse_authors("Idri A.; Hosni M.",
                        "Idri, Ali (6602789810); Hosni, Mohamed (57189341317)",
                        "6602789810; 57189341317")
    assert [x["position"] for x in a] == [1, 2]
    assert a[1]["scopus_id"] == "57189341317"


def test_authors_colonnes_desalignees():
    """Un export abîmé ne doit pas décaler les noms ni lever d'exception."""
    a = P.parse_authors("A.; B.; C.", "A, Alpha (11111)", "")
    assert len(a) == 3
    assert a[0]["scopus_id"] == "11111"
    assert a[2]["full_name"] is None


def test_authors_cellule_vide():
    assert P.parse_authors("", "", "") == []
    assert P.parse_authors(None) == []


# --- affiliations ----------------------------------------------------------

def test_affiliation_etiquetee():
    r = P.parse_affiliation(
        "subparent: ENSMR, parent 1: National School of Mineral Industry, "
        "city: Rabat, country: Morocco")
    assert r["labelled"] is True
    assert r["subparent"] == "ENSMR"
    assert r["parent1"] == "National School of Mineral Industry"
    assert r["city"] == "Rabat"
    assert r["country"] == "Morocco"
    assert r["parent2"] is None and r["region"] is None


def test_affiliation_etiquetee_champs_omis():
    """Les champs vides sont OMIS à l'export : la lecture ne doit pas dépendre
    de la position."""
    r = P.parse_affiliation("parent 1: University of Murcia, city: Murcia, "
                            "region: Murcia, country: Spain")
    assert r["parent1"] == "University of Murcia"
    assert r["region"] == "Murcia"
    assert r["subparent"] is None


def test_affiliation_independent():
    r = P.parse_affiliation("parent 1: %s, city: Rabat, country: Morocco"
                            % INDEPENDENT_LABEL)
    assert r["parent1"] == INDEPENDENT_LABEL


def test_affiliation_brute_scopus():
    """Non nettoyée : seuls pays et ville sont fiables, par position finale."""
    r = P.parse_affiliation(
        "National School of Applied Sciences of Oujda, Mohamed 1st University, "
        "Oujda, Morocco")
    assert r["labelled"] is False
    assert r["country"] == "Morocco"
    assert r["city"] == "Oujda"
    assert r["parent1"] == "Mohamed 1st University"


def test_affiliation_vide():
    r = P.parse_affiliation("")
    assert r["country"] is None and r["raw"] == ""


# --- références ------------------------------------------------------------

def test_references_reconciliees():
    cell = ("ref1 | 10.1007/s11749-016-0481-7 | 2016 | Biau, Scornet | "
            "A random forest guided tour ; "
            "ref2 | 10.1145/2939672.2939785 | 2016 | Chen T., Guestrin C. | "
            "XGBoost: A Scalable Tree Boosting System")
    r = P.parse_references(cell)
    assert len(r) == 2
    assert r[0]["ref_pos"] == 1
    assert r[0]["ref_doi"] == "10.1007/s11749-016-0481-7"
    assert r[0]["ref_year"] == 2016
    assert r[1]["ref_title"].startswith("XGBoost")


def test_reference_non_structuree_conservee():
    """Sans barres verticales, on garde le texte plutôt que de le perdre."""
    r = P.parse_references("Some old raw reference, 1998")
    assert len(r) == 1
    assert r[0]["ref_doi"] is None
    assert "1998" in r[0]["ref_title"]


def test_references_vides():
    assert P.parse_references("") == []


# --- mots-clés -------------------------------------------------------------

def test_keywords_dedoublonnes_sans_casse():
    k = P.parse_keywords("Machine Learning; machine learning; Random Forest")
    assert k == ["Machine Learning", "Random Forest"]
