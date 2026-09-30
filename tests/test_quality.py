"""Qualite des donnees : le piege des valeurs manquantes, et la fiabilite."""

import numpy as np
import pandas as pd
import pytest

from bibliominer_analysis import Corpus
from bibliominer_analysis.metrics import quality as q

LONG = "A systematic review of ensemble effort estimation methods"


def _doc(i, year=2020, title=None, doi="", issn="", abstract="Some abstract text",
         authors="1:A.", kws="alpha", refs="r1 | 10.9/x | 2015 | X | some ref title",
         aff="parent 1: U, city: Rabat, country: Morocco"):
    return {
        "Title": title or ("Document %d with a reasonably long title" % i),
        "Year": str(year), "Cited by": "1", "EID": "eid-%d" % i,
        "Source title": "J1", "Document Type": "Article", "Authors": authors,
        "Author Keywords": kws, "Abstract": abstract, "DOI": doi, "ISSN": issn,
        "References": refs, "Affiliations": aff,
    }


# --------------------------------------------------- le piege des absences ---

def test_is_filled_attrape_toutes_les_ecritures_d_une_absence():
    """« nan » et « None » sont des CHAINES apres conversion : le piege.

    Un test naif `valeur != ""` les compte comme remplies, et le tableau de
    completude annonce 100 % partout, un controle faux d'une facon
    particulierement traitresse, puisqu'il rassure.
    """
    s = pd.Series(["a", "", None, np.nan, "nan", "None", "  ", "NA", "null", "b"])
    mask = q.is_filled(s)
    assert list(mask) == [True, False, False, False, False, False,
                          False, False, False, True]
    assert int(mask.sum()) == 2


def test_completude_ne_ment_pas_sur_un_champ_vide():
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, issn="1542-4863"), _doc(2), _doc(3), _doc(4),
    ]))
    f = q.field_completeness(c).set_index("field")
    assert f.loc["issn", "filled"] == 1
    assert f.loc["issn", "share"] == pytest.approx(25.0)
    assert f.loc["issn", "status"] == q.LIMITED


# ------------------------------------------------------------- fiabilite -----

def test_statuts_suivent_les_seuils():
    assert q._status(100.0) == q.READY
    assert q._status(90.0) == q.READY
    assert q._status(89.9) == q.PARTIAL
    assert q._status(50.0) == q.PARTIAL
    assert q._status(49.9) == q.LIMITED


def test_fiabilite_couvre_toutes_les_familles():
    c = Corpus.from_dataframe(pd.DataFrame([_doc(1), _doc(2)]))
    r = q.indicator_readiness(c)
    assert len(r) == 10
    assert set(r.columns) == {"analysis", "status", "coverage", "basis", "note"}
    assert set(r["status"]) <= {q.READY, q.PARTIAL, q.LIMITED}


def test_resume_compte_les_analyses_fiables():
    c = Corpus.from_dataframe(pd.DataFrame([_doc(1), _doc(2)]))
    s = q.quality_summary(c)
    assert s["analyses"] == s["ready"] + s["partial"] + s["limited"]
    assert 0 <= s["mean_coverage"] <= 100


# ------------------------------------------------------------- doublons ------

def test_doublon_par_doi():
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, doi="10.1000/aaa"), _doc(2, doi="10.1000/aaa"), _doc(3),
    ]))
    d = q.duplicates(c)
    assert len(d) == 1
    assert d.iloc[0]["kind"] == "DOI"
    assert d.iloc[0]["documents"] == 2


def test_doublon_par_titre():
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, title=LONG), _doc(2, title=LONG.upper()), _doc(3),
    ]))
    d = q.duplicates(c)
    assert len(d) == 1
    assert d.iloc[0]["kind"] == "Title"


def test_titre_court_ne_declenche_pas_un_faux_doublon():
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, title="Editorial"), _doc(2, title="Editorial"),
    ]))
    assert q.duplicates(c).empty


def test_corpus_propre_sans_doublon():
    c = Corpus.from_dataframe(pd.DataFrame([_doc(1), _doc(2)]))
    assert q.duplicates(c).empty


# ------------------------------------------------------------ anomalies ------

def test_document_sans_annee_signale():
    c = Corpus.from_dataframe(pd.DataFrame([_doc(1), _doc(2, year="")]))
    a = q.anomalies(c).set_index("check")
    assert a.loc["Missing year", "documents"] == 1
    assert a.loc["Missing year", "severity"] == "high"


def test_document_sans_mot_cle_signale():
    c = Corpus.from_dataframe(pd.DataFrame([_doc(1), _doc(2, kws="")]))
    a = q.anomalies(c).set_index("check")
    assert a.loc["No keyword", "documents"] == 1


def test_anomalies_toujours_toutes_les_lignes():
    """Une verification absente se lirait comme non effectuee."""
    c = Corpus.from_dataframe(pd.DataFrame([_doc(1)]))
    a = q.anomalies(c)
    assert len(a) == 7
    assert (a["documents"] == 0).all()


def test_tables_liees_evaluees():
    c = Corpus.from_dataframe(pd.DataFrame([_doc(1)]))
    t = q.table_completeness(c)
    assert set(t["table"]) == {"authors", "affiliations", "keywords", "references"}


# ------------------------------------------ « No single city » (site: virtual)

def test_a_virtual_site_is_read_and_not_a_missing_city():
    """« site: virtual » : l'utilisateur a coché « No single city » au
    nettoyage (laboratoire virtuel). Pas de ville PAR DÉCISION : ni une
    ville nommée « virtual », ni une ville manquante."""
    docs = [_doc(1), _doc(2, aff="parent 1: LIRIMA, site: virtual, country: France")]
    c = Corpus.from_dataframe(pd.DataFrame(docs))
    lirima = c.affiliations[c.affiliations["parent1"] == "LIRIMA"].iloc[0]
    assert lirima["site"] == "virtual"
    assert not isinstance(lirima["city"], str) or not lirima["city"]
    assert lirima["country"] == "France"

    table = q.table_completeness(c)
    city = table[(table["table"] == "affiliations") & (table["field"] == "city")].iloc[0]
    assert city["share"] == 100.0 and city["total"] == 1
