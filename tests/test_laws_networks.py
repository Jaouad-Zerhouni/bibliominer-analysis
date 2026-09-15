"""Tests des lois bibliométriques et des réseaux.

Les corpus de test sont construits pour que la réponse soit calculable À LA
MAIN : c'est la seule façon de détecter une régression dans un indicateur.
"""

import numpy as np
import pandas as pd
import pytest

from bibliominer_analysis import Corpus


def _doc(i, authors, keywords="", refs="", source="J1", cites=0):
    return {
        "Title": "Doc %d" % i, "Year": "2020", "Cited by": str(cites),
        "EID": "eid-%d" % i, "Source title": source, "Document Type": "Article",
        "Authors": authors, "Author Keywords": keywords, "References": refs,
        "Affiliations": "parent 1: Univ A, city: Rabat, country: Morocco",
    }


# ---------------------------------------------------------------------------
# Lotka
# ---------------------------------------------------------------------------

def test_lotka_distribution():
    """3 auteurs à 1 doc, 1 auteur à 3 docs -> distribution {1: 3, 3: 1}."""
    rows = [
        _doc(1, "1:A.; 2:X."),
        _doc(2, "1:A.; 2:Y."),
        _doc(3, "1:A.; 2:Z."),
    ]
    c = Corpus.from_dataframe(pd.DataFrame(rows))
    res = c.lotka()
    t = res["table"].set_index("documents_written")["n_authors"]
    assert t[1] == 3          # X, Y, Z
    assert t[3] == 1          # A
    assert res["total_authors"] == 4


def test_lotka_part_theorique():
    """La part théorique de Lotka pour x=1 vaut 1/zeta(2) = 6/pi^2 ≈ 0.6079."""
    c = Corpus.from_dataframe(pd.DataFrame([_doc(1, "1:A."), _doc(2, "1:B.")]))
    row = c.lotka()["table"].iloc[0]
    assert row["documents_written"] == 1
    assert row["share_lotka"] == pytest.approx(6 / np.pi ** 2, abs=1e-3)


def test_lotka_corpus_vide():
    c = Corpus.from_dataframe(pd.DataFrame([{"Title": "x"}]))
    assert c.lotka()["table"].empty


# ---------------------------------------------------------------------------
# Bradford
# ---------------------------------------------------------------------------

def test_bradford_zones_couvrent_tout():
    """Chaque source appartient à une zone, et les zones somment au total."""
    rows = []
    i = 0
    # 1 source à 6 docs, 2 sources à 2 docs, 6 sources à 1 doc = 16 documents
    for src, n in [("Core", 6), ("B1", 2), ("B2", 2)] + [("S%d" % k, 1) for k in range(6)]:
        for _ in range(n):
            i += 1
            rows.append(_doc(i, "1:A.", source=src))
    c = Corpus.from_dataframe(pd.DataFrame(rows))
    res = c.bradford()

    assert res["total_documents"] == 16
    assert res["total_sources"] == 9
    assert res["zones"]["documents"].sum() == 16
    assert res["zones"]["sources"].sum() == 9
    # La source la plus productive est en zone 1, et en tête du classement.
    top = res["table"].iloc[0]
    assert top["source"] == "Core" and top["zone"] == 1


def test_bradford_cumul_croissant():
    rows = [_doc(i, "1:A.", source="S%d" % (i % 4)) for i in range(12)]
    c = Corpus.from_dataframe(pd.DataFrame(rows))
    cum = c.bradford()["table"]["cumulative"].tolist()
    assert cum == sorted(cum)
    assert cum[-1] == 12


# ---------------------------------------------------------------------------
# Zipf
# ---------------------------------------------------------------------------

def test_zipf_rangs_et_frequences():
    rows = [
        _doc(1, "1:A.", keywords="alpha; beta; gamma"),
        _doc(2, "1:A.", keywords="alpha; beta"),
        _doc(3, "1:A.", keywords="alpha"),
    ]
    c = Corpus.from_dataframe(pd.DataFrame(rows))
    t = c.zipf()["table"]
    assert list(t["rank"]) == [1, 2, 3]
    assert list(t["frequency"]) == [3, 2, 1]      # alpha, beta, gamma
    assert t.iloc[0]["keyword"] == "alpha"


def test_zipf_ajustement_sur_loi_parfaite():
    """Sur une distribution exactement en 1/rang, l'exposant doit valoir 1."""
    rows = []
    i = 0
    for rank, freq in enumerate([12, 6, 4, 3], start=1):   # ~ 12/rang
        for _ in range(freq):
            i += 1
            rows.append(_doc(i, "1:A.", keywords="w%d" % rank))
    c = Corpus.from_dataframe(pd.DataFrame(rows))
    fit = c.zipf()["fit"]
    assert fit["exponent"] == pytest.approx(1.0, abs=0.1)
    assert fit["r2"] > 0.99


# ---------------------------------------------------------------------------
# Réseaux
# ---------------------------------------------------------------------------

def test_co_word_poids():
    """alpha et beta ensemble dans 2 documents -> lien de poids 2."""
    rows = [
        _doc(1, "1:A.", keywords="alpha; beta"),
        _doc(2, "1:A.", keywords="alpha; beta"),
        _doc(3, "1:A.", keywords="alpha; gamma"),
    ]
    c = Corpus.from_dataframe(pd.DataFrame(rows))
    g = c.co_word(min_weight=1)
    edge = {(e["source"], e["target"]): e["weight"] for e in g["edges"]}
    assert edge[("alpha", "beta")] == 2
    assert edge[("alpha", "gamma")] == 1


def test_co_word_filtre_les_liens_faibles():
    rows = [_doc(1, "1:A.", keywords="alpha; beta"),
            _doc(2, "1:A.", keywords="alpha; gamma")]
    c = Corpus.from_dataframe(pd.DataFrame(rows))
    g = c.co_word(min_weight=2)
    assert g["n_edges"] == 0
    assert g["n_nodes"] == 0        # aucun nœud isolé ne subsiste


def test_co_authorship():
    rows = [_doc(1, "1:A.; 2:B."), _doc(2, "1:A.; 2:B."), _doc(3, "1:A.; 2:C.")]
    c = Corpus.from_dataframe(pd.DataFrame(rows))
    g = c.co_authorship()
    w = {tuple(sorted((e["source"], e["target"]))): e["weight"] for e in g["edges"]}
    assert w[tuple(sorted(("name:A.", "name:B.")))] == 2
    assert w[tuple(sorted(("name:A.", "name:C.")))] == 1


def test_co_citation_par_doi():
    """Deux références partagées par deux documents -> lien de poids 2."""
    r = ("ref1 | 10.1/x | 2015 | Doe J. | Paper X ; "
         "ref2 | 10.1/y | 2016 | Roe R. | Paper Y")
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, "1:A.", refs=r), _doc(2, "1:B.", refs=r)]))
    g = c.co_citation(min_weight=2)
    assert g["n_nodes"] == 2
    assert g["edges"][0]["weight"] == 2
    assert any("Paper X" in n["label"] for n in g["nodes"])


def test_co_citation_ignore_les_refs_non_identifiables():
    """Sans DOI ni titre exploitable, une référence ne peut pas être rapprochée."""
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, "1:A.", refs="ref1 | | | | ab ; ref2 | | | | cd")]))
    assert c.co_citation(min_weight=1)["n_nodes"] == 0


def test_reseaux_corpus_vide():
    c = Corpus.from_dataframe(pd.DataFrame([{"Title": "x"}]))
    for g in (c.co_word(), c.co_citation(), c.co_authorship()):
        assert g["n_nodes"] == 0 and g["n_edges"] == 0


# ---------------------------------------------------------------------------
# Carte des pays
# ---------------------------------------------------------------------------

def test_country_map_collaboration_internationale():
    rows = [
        _doc(1, "1:A."),                                    # Maroc seul
        dict(_doc(2, "1:B."),
             Affiliations=("parent 1: Univ A, city: Rabat, country: Morocco; "
                           "parent 1: Univ B, city: Madrid, country: Spain")),
    ]
    c = Corpus.from_dataframe(pd.DataFrame(rows))
    m = c.country_map().set_index("country")
    assert m.loc["Morocco", "documents"] == 2
    assert m.loc["Morocco", "sca"] == 1      # doc 1 seulement
    assert m.loc["Morocco", "mca"] == 1      # doc 2, co-signé
    assert m.loc["Spain", "mca"] == 1
    assert m.loc["Spain", "mca_ratio"] == 100.0
