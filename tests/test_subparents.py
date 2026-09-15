"""Niveau SOUS-PARENT : unités internes (laboratoires, écoles, départements).

Le cleaning distingue l'organisme mère (`parent 1`) de l'unité qui y travaille
(`subparent`). Les deux répondent à des questions différentes, et les tests
vérifient que le paramètre `level` change bien l'objet analysé.
"""

import pandas as pd
import pytest

from bibliominer_analysis import Corpus
from bibliominer_analysis.io.schema import INDEPENDENT_LABEL


def _doc(i, affs, year=2020, cites=0):
    return {
        "Title": "Doc %d" % i, "Year": str(year), "Cited by": str(cites),
        "EID": "eid-%d" % i, "Source title": "J", "Document Type": "Article",
        "Authors": "1:A.", "Affiliations": affs,
    }


def _aff(sub, parent, country="Morocco"):
    return "subparent: %s, parent 1: %s, city: X, country: %s" % (sub, parent, country)


@pytest.fixture
def corpus() -> Corpus:
    """Deux universités, quatre laboratoires."""
    return Corpus.from_dataframe(pd.DataFrame([
        _doc(1, _aff("LMAID", "Moulay Ismail University"), 2020, 10),
        _doc(2, _aff("LMAID", "Moulay Ismail University"), 2021, 4),
        _doc(3, _aff("MOSI", "Moulay Ismail University"), 2021, 2),
        _doc(4, _aff("ENSIAS", "Mohammed V University"), 2022, 6),
        _doc(5, _aff("ENSIAS", "Mohammed V University"), 2022, 0),
    ]))


def test_niveau_change_l_objet_analyse(corpus):
    parents = set(corpus.top_institutions(level="parent")["institution"])
    subs = set(corpus.top_institutions(level="subparent")["institution"])
    assert parents == {"Moulay Ismail University", "Mohammed V University"}
    assert subs == {"LMAID", "MOSI", "ENSIAS"}


def test_comptes_par_niveau(corpus):
    p = corpus.top_institutions(level="parent").set_index("institution")
    s = corpus.top_institutions(level="subparent").set_index("institution")
    assert p.loc["Moulay Ismail University", "documents"] == 3   # 1 + 2 + 3
    assert s.loc["LMAID", "documents"] == 2
    assert s.loc["MOSI", "documents"] == 1


def test_impact_par_niveau(corpus):
    s = corpus.institutions_impact(level="subparent").set_index("institution")
    # LMAID : citations 10 et 4 -> h = 2
    assert s.loc["LMAID", "citations"] == 14
    assert s.loc["LMAID", "h_index"] == 2
    # ENSIAS : 6 et 0 -> h = 1
    assert s.loc["ENSIAS", "h_index"] == 1


def test_over_time_par_niveau(corpus):
    t = corpus.institutions_over_time(level="subparent")
    assert set(t["institution"]) == {"LMAID", "MOSI", "ENSIAS"}
    lmaid = t[t["institution"] == "LMAID"].set_index("year")
    assert list(lmaid["cumulative"]) == [1, 2, 2]      # 2020, 2021, 2022


def test_by_country_par_niveau(corpus):
    s = corpus.institutions_by_country(level="subparent").set_index("country")
    assert s.loc["Morocco", "institutions"] == 3       # 3 unités, pas 2 universités


def test_reseau_par_niveau():
    """Deux unités d'une MÊME université collaborant : invisible au niveau parent."""
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, _aff("LMAID", "Univ A") + "; " + _aff("MOSI", "Univ A")),
        _doc(2, _aff("LMAID", "Univ A") + "; " + _aff("MOSI", "Univ A")),
    ]))
    parent = c.co_institution(level="parent", min_weight=1)
    sub = c.co_institution(level="subparent", min_weight=1)
    assert parent["n_edges"] == 0        # une seule université -> aucun lien
    assert sub["n_edges"] == 1           # les deux labos collaborent
    assert sub["edges"][0]["weight"] == 2


def test_hierarchie(corpus):
    h = corpus.org_hierarchy()
    mi = h[h["parent"] == "Moulay Ismail University"].set_index("subparent")
    assert set(mi.index) == {"LMAID", "MOSI"}
    assert mi.loc["LMAID", "documents"] == 2
    assert mi.loc["LMAID", "parent_documents"] == 3
    # LMAID porte 2 des 3 documents de son université
    assert mi.loc["LMAID", "share"] == pytest.approx(66.7, abs=0.1)


def test_hierarchie_ignore_les_affiliations_sans_unite():
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, "parent 1: Univ A, city: X, country: Morocco"),
        _doc(2, _aff("Lab", "Univ A")),
    ]))
    h = c.org_hierarchy()
    assert list(h["subparent"]) == ["Lab"]
    assert h.iloc[0]["documents"] == 1
    assert h.iloc[0]["parent_documents"] == 2     # l'université en a bien 2


def test_subparent_exclut_les_independants():
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, _aff(INDEPENDENT_LABEL, "Univ A")),
        _doc(2, _aff("Lab", "Univ A")),
    ]))
    assert INDEPENDENT_LABEL not in set(
        c.top_institutions(level="subparent")["institution"])


def test_niveau_invalide_leve_une_erreur(corpus):
    with pytest.raises(ValueError, match="parent"):
        corpus.top_institutions(level="departement")


def test_corpus_sans_sous_unite():
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, "parent 1: Univ A, city: X, country: Morocco")]))
    assert c.top_institutions(level="subparent").empty
    assert c.org_hierarchy().empty
