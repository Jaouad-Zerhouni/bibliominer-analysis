"""Réseaux déclinés par unité d'analyse : institutions, pays, auteurs cités."""

import pandas as pd

from bibliominer_analysis import Corpus
from bibliominer_analysis.io.schema import INDEPENDENT_LABEL


def _doc(i, affs, refs="", authors="1:A."):
    return {
        "Title": "Doc %d" % i, "Year": "2021", "Cited by": "0",
        "EID": "eid-%d" % i, "Source title": "J", "Document Type": "Article",
        "Authors": authors, "Affiliations": affs, "References": refs,
    }


def _aff(parent, country, city="X"):
    return "parent 1: %s, city: %s, country: %s" % (parent, city, country)


# --- collaboration entre institutions --------------------------------------

def test_co_institution_poids():
    """Univ A et Univ B ensemble sur 2 documents -> lien de poids 2."""
    both = _aff("Univ A", "Morocco") + "; " + _aff("Univ B", "Spain")
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, both), _doc(2, both),
        _doc(3, _aff("Univ A", "Morocco") + "; " + _aff("Univ C", "Egypt")),
    ]))
    g = c.co_institution(min_weight=1)
    w = {tuple(sorted((e["source"], e["target"]))): e["weight"] for e in g["edges"]}
    assert w[("Univ A", "Univ B")] == 2
    assert w[("Univ A", "Univ C")] == 1


def test_co_institution_exclut_les_independants():
    """« Independent researcher » n'est pas un établissement."""
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, _aff("Univ A", "Morocco") + "; " + _aff(INDEPENDENT_LABEL, "Morocco")),
        _doc(2, _aff("Univ A", "Morocco") + "; " + _aff(INDEPENDENT_LABEL, "Morocco")),
    ]))
    g = c.co_institution(min_weight=1)
    assert all(INDEPENDENT_LABEL not in n["label"] for n in g["nodes"])
    assert g["n_edges"] == 0        # il ne reste qu'une institution


# --- collaboration entre pays ----------------------------------------------

def test_co_country():
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, _aff("U1", "Morocco") + "; " + _aff("U2", "Spain")),
        _doc(2, _aff("U1", "Morocco") + "; " + _aff("U3", "Spain")),
        _doc(3, _aff("U1", "Morocco")),
    ]))
    g = c.co_country(min_weight=1)
    assert g["n_nodes"] == 2
    assert g["edges"][0]["weight"] == 2      # Maroc-Espagne sur 2 documents
    occ = {n["label"]: n["occurrences"] for n in g["nodes"]}
    assert occ["Morocco"] == 3               # présent sur les 3 documents


def test_co_country_document_national_sans_lien():
    c = Corpus.from_dataframe(pd.DataFrame([_doc(1, _aff("U1", "Morocco"))]))
    g = c.co_country(min_weight=1)
    assert g["n_edges"] == 0 and g["n_nodes"] == 0


# --- co-citation d'auteurs --------------------------------------------------

def test_co_citation_authors_premier_auteur():
    """La convention ACA retient le PREMIER auteur de chaque référence."""
    refs = ("ref1 | 10.1/a | 2015 | Biau G., Scornet E. | Random forests ; "
            "ref2 | 10.1/b | 2016 | Chen T., Guestrin C. | XGBoost")
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, _aff("U", "Morocco"), refs=refs),
        _doc(2, _aff("U", "Morocco"), refs=refs),
    ]))
    g = c.co_citation_authors(min_weight=2)
    labels = sorted(n["label"] for n in g["nodes"])
    assert labels == ["Biau G.", "Chen T."]
    assert g["edges"][0]["weight"] == 2


def test_co_citation_authors_regroupe_les_graphies():
    """« Chen T. » et « Chen T. » cités par deux documents = un seul nœud."""
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, _aff("U", "Morocco"),
             refs="ref1 | 10.1/b | 2016 | Chen T., Guestrin C. | XGBoost ; "
                  "ref2 | 10.1/c | 2017 | Doe J. | Autre"),
        _doc(2, _aff("U", "Morocco"),
             refs="ref1 | 10.1/d | 2018 | chen t., Other | Suite ; "
                  "ref2 | 10.1/c | 2017 | Doe J. | Autre"),
    ]))
    g = c.co_citation_authors(min_weight=2)
    ids = {n["id"] for n in g["nodes"]}
    assert "chen t." in ids            # les deux graphies ont fusionné
    assert g["n_nodes"] == 2


def test_co_citation_authors_sans_auteurs():
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, _aff("U", "Morocco"), refs="ref1 | 10.1/a | 2015 | | Sans auteur")]))
    assert c.co_citation_authors(min_weight=1)["n_nodes"] == 0


def test_tous_les_reseaux_sur_corpus_vide():
    c = Corpus.from_dataframe(pd.DataFrame([{"Title": "x"}]))
    for g in (c.co_institution(), c.co_country(), c.co_citation_authors()):
        assert g["n_nodes"] == 0 and g["n_edges"] == 0
