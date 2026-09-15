"""Appariement SCImago : ISSN, titre, et ce qu'on refuse de deviner."""

import pandas as pd
import pytest

from bibliominer_analysis import Corpus
from bibliominer_analysis.metrics import scimago as sc


def _doc(i, source, issn="", cited="0", year=2020):
    return {
        "Title": "Document %d with a reasonably long title" % i,
        "Year": str(year), "Cited by": str(cited), "EID": "eid-%d" % i,
        "Source title": source, "Document Type": "Article", "Authors": "1:A.",
        "Author Keywords": "alpha", "ISSN": issn,
        "Affiliations": "parent 1: U, city: X, country: Morocco",
    }


def test_referentiel_charge():
    t = sc.load_scimago()
    assert not t.empty, "le CSV SCImago doit voyager avec le package"
    assert set(t["quartile"].dropna().unique()) <= set(sc.QUARTILES)
    assert t["sjr"].max() > 1, "les decimales a la virgule doivent etre lues"


def test_normalisation_issn():
    assert sc.normalize_issn("1542-4863") == "15424863"
    assert sc.normalize_issn("0921030X") == "0921030X"
    assert sc.normalize_issn("trop court") == ""
    assert sc.normalize_issn(None) == ""


def test_appariement_par_issn():
    """CA-A Cancer Journal for Clinicians, Q1, premiere ligne du referentiel."""
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, "Whatever the export calls it", issn="1542-4863"),
    ]))
    row = c.scimago_sources().iloc[0]
    assert row["matched_by"] == "issn"
    assert row["quartile"] == "Q1"
    assert row["sjr"] > 100


def test_appariement_par_titre_quand_issn_absent():
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, "Ca-A Cancer Journal for Clinicians"),
    ]))
    row = c.scimago_sources().iloc[0]
    assert row["matched_by"] == "title"
    assert row["quartile"] == "Q1"


def test_revue_inconnue_reste_vide():
    """On n'invente pas un quartile : ce serait pire que de ne rien dire."""
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, "Journal Of Things That Do Not Exist At All"),
    ]))
    row = c.scimago_sources().iloc[0]
    assert row["matched_by"] == ""
    assert row["quartile"] is None
    assert pd.isna(row["sjr"])


def test_distribution_toujours_cinq_lignes():
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, "Ca-A Cancer Journal for Clinicians", issn="1542-4863"),
        _doc(2, "Journal Of Things That Do Not Exist"),
    ]))
    d = c.quartile_distribution()
    assert list(d["quartile"]) == ["Q1", "Q2", "Q3", "Q4", "Not indexed"]
    assert d["share"].sum() == pytest.approx(100.0, abs=0.2)
    counts = dict(zip(d["quartile"], d["documents"]))
    assert counts["Q1"] == 1
    assert counts["Not indexed"] == 1


def test_couverture_dit_ce_qui_manque():
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, "Ca-A Cancer Journal for Clinicians", issn="1542-4863"),
        _doc(2, "Journal Of Things That Do Not Exist"),
    ]))
    cov = c.scimago_coverage()
    assert cov["available"] is True
    assert cov["sources"] == 2
    assert cov["matched_sources"] == 1
    assert cov["matched_by_issn"] == 1
    assert cov["document_share"] == pytest.approx(50.0)


def test_quartiles_dans_le_temps():
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, "Ca-A Cancer Journal for Clinicians", issn="1542-4863", year=2019),
        _doc(2, "Journal Of Things That Do Not Exist", year=2019),
    ]))
    t = c.quartile_over_time()
    y = t[t["year"] == 2019].set_index("quartile")
    assert y.loc["Q1", "documents"] == 1
    assert y.loc["Not indexed", "documents"] == 1
    assert y.loc["Q1", "share"] == pytest.approx(50.0)
    # Les cinq categories sont presentes meme a zero.
    assert len(y) == 5


def test_titre_trop_court_refuse_comme_cle():
    """« Nature » ou « Cell » ne peuvent pas servir de cle d'appariement."""
    c = Corpus.from_dataframe(pd.DataFrame([_doc(1, "Cell")]))
    assert c.scimago_sources().iloc[0]["matched_by"] == ""


def test_referentiel_absent_ne_casse_pas(tmp_path):
    """Sans le fichier, l'analyse se degrade — elle ne s'interrompt pas."""
    missing = tmp_path / "absent.csv"
    c = Corpus.from_dataframe(pd.DataFrame([_doc(1, "Some Journal Name Here")]))
    out = c.scimago_sources(path=missing)
    assert len(out) == 1
    assert out.iloc[0]["matched_by"] == ""
    assert len(c.quartile_distribution(path=missing)) == 5
