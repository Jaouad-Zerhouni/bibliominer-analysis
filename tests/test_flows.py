"""Diagramme à trois champs : double co-occurrence."""

import pandas as pd
import pytest

from bibliominer_analysis import Corpus


def _doc(i, authors, kws, source):
    return {
        "Title": "Doc %d" % i, "Year": "2020", "Cited by": "0",
        "EID": "eid-%d" % i, "Source title": source, "Document Type": "Article",
        "Authors": authors, "Author Keywords": kws,
        "Affiliations": "parent 1: U, city: X, country: Morocco",
    }


@pytest.fixture
def corpus():
    return Corpus.from_dataframe(pd.DataFrame([
        _doc(1, "1:A.", "alpha", "J1"),
        _doc(2, "1:A.", "alpha", "J1"),
        _doc(3, "1:B.", "beta", "J2"),
    ]))


def test_liens_et_poids(corpus):
    """A -> alpha porte 2 documents ; alpha -> J1 aussi."""
    t = corpus.three_fields()
    w = {(r["source"], r["target"]): r["value"] for _, r in t.iterrows()}
    assert w[("A.", "alpha")] == 2
    assert w[("alpha", "J1")] == 2
    assert w[("B.", "beta")] == 1


def test_profondeur_des_colonnes(corpus):
    """La profondeur aligne les noeuds ; sans elle le rendu les place mal."""
    t = corpus.three_fields()
    gauche = t[t["depth"] == 0]
    milieu = t[t["depth"] == 1]
    assert set(gauche["source"]) == {"A.", "B."}
    assert set(milieu["source"]) == {"alpha", "beta"}


def test_limite_par_colonne():
    rows = [_doc(i, "1:A%d." % i, "k%d" % i, "S%d" % i) for i in range(8)]
    c = Corpus.from_dataframe(pd.DataFrame(rows))
    t = c.three_fields(n=3)
    assert t[t["depth"] == 0]["source"].nunique() <= 3


def test_meme_dimension_deux_fois_est_desambiguisee():
    """Sans préfixe, un noeud apparaîtrait dans deux colonnes et bouclerait."""
    c = Corpus.from_dataframe(pd.DataFrame([_doc(1, "1:A.", "alpha", "J1")]))
    t = c.three_fields(left="authors", middle="keywords", right="authors")
    assert any(str(v).startswith("3\u00b7 ") for v in t["target"])


def test_champ_inconnu(corpus):
    with pytest.raises(ValueError, match="champ inconnu"):
        corpus.three_fields(left="galaxies")


def test_corpus_vide():
    c = Corpus.from_dataframe(pd.DataFrame([{"Title": "x"}]))
    assert c.three_fields().empty
