"""Tests of the parsers: this is where fidelity to the format is decided.

The cases are not invented, they come from real exports, including the
degraded ones (old non-indexed format, raw affiliation, unreconciled
reference), because users have files in both states.
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
    """Old export: no "n:" prefix. The position falls back on the rank."""
    a = P.parse_authors("Idri A.; Hosni M.",
                        "Idri, Ali (6602789810); Hosni, Mohamed (57189341317)",
                        "6602789810; 57189341317")
    assert [x["position"] for x in a] == [1, 2]
    assert a[1]["scopus_id"] == "57189341317"


def test_authors_colonnes_desalignees():
    """A damaged export must neither shift the names nor raise an exception."""
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
    """Empty fields are OMITTED from the export: reading must not depend on the
    position."""
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
    """Not cleaned: only country and city are reliable, by final position."""
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


# --- references ------------------------------------------------------------

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
    """Without vertical bars, the text is kept rather than lost."""
    r = P.parse_references("Some old raw reference, 1998")
    assert len(r) == 1
    assert r[0]["ref_doi"] is None
    assert "1998" in r[0]["ref_title"]


def test_references_vides():
    assert P.parse_references("") == []


# --- keywords --------------------------------------------------------------

def test_keywords_dedoublonnes_sans_casse():
    k = P.parse_keywords("Machine Learning; machine learning; Random Forest")
    assert k == ["Machine Learning", "Random Forest"]


def test_raw_scopus_references_are_split_per_reference_not_per_author():
    """Found: in the raw Scopus export, the authors of a reference are separated
    by ";", like the references themselves. Split on ";", each reference
    became as many "references" as it had authors (8,854 instead of 2,847 on a
    real corpus), with neither title nor year: no local citations, no
    historiograph."""
    from bibliominer_analysis.io.parsers import parse_references
    cell = ("Ali A.; Gravino C., A systematic literature review of software effort "
            "prediction, Journal of software: evolution and process, 31, 10, (2019); "
            "Azzeh M.; Nassif A. B.; Minku L. L., An empirical evaluation of ensemble "
            "adjustment methods, Journal of Systems and Software, 103, pp. 36-52, (2015)")
    refs = parse_references(cell)
    assert len(refs) == 2
    assert refs[0]["ref_year"] == 2019
    assert refs[0]["ref_authors"] == "Ali A., Gravino C."
    assert refs[0]["ref_title"] == "A systematic literature review of software effort prediction"
    assert refs[1]["ref_year"] == 2015
    assert refs[1]["ref_authors"].startswith("Azzeh M.")


def test_reconciled_references_are_read_as_before():
    from bibliominer_analysis.io.parsers import parse_references
    refs = parse_references("ref1 | 10.1/x | 2016 | Biau G., Scornet E. | A random "
                            "forest guided tour ; ref2 |  | 2019 | Doe J. | Other title")
    assert [r["ref_doi"] for r in refs] == ["10.1/x", None]
    assert refs[0]["ref_title"] == "A random forest guided tour"
