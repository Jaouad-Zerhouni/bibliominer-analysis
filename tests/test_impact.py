"""Tests des indices h et g.

Les valeurs attendues sont calculées À LA MAIN à partir des définitions
canoniques, pas produites par le code lui-même. C'est la seule façon qu'un
test d'indicateur ait une valeur : sinon il ne fait que graver l'erreur.
"""

import pandas as pd
import pytest

from bibliominer_analysis import Corpus
from bibliominer_analysis.metrics.impact import g_index, h_index, i10_index


# --- h-index ---------------------------------------------------------------

def test_h_index_cas_classique():
    # [10, 8, 5, 4, 3] : 4 articles ont >= 4 citations ; le 5e n'en a que 3.
    assert h_index([10, 8, 5, 4, 3]) == 4


def test_h_index_un_article_tres_cite():
    # Un seul article, même cité 100 fois, ne donne qu'un h de 1.
    assert h_index([100]) == 1


def test_h_index_aucune_citation():
    assert h_index([0, 0, 0]) == 0


def test_h_index_serie_vide():
    assert h_index([]) == 0


def test_h_index_ordre_indifferent():
    assert h_index([3, 10, 4, 8, 5]) == h_index([10, 8, 5, 4, 3])


def test_h_index_tous_egaux():
    # 5 articles à 5 citations : h = 5.
    assert h_index([5, 5, 5, 5, 5]) == 5


# --- g-index ---------------------------------------------------------------

def test_g_index_cas_classique():
    # [10, 8, 5, 4, 3] -> cumul [10, 18, 23, 27, 30] ; g^2 [1, 4, 9, 16, 25].
    # 30 >= 25 donc g = 5.
    assert g_index([10, 8, 5, 4, 3]) == 5


def test_g_index_superieur_ou_egal_au_h():
    """Propriété fondamentale : g >= h, toujours."""
    for serie in ([10, 8, 5, 4, 3], [100], [1, 1, 1], [25, 8, 5, 3, 3],
                  [0, 0, 1], [7, 7, 7, 7]):
        assert g_index(serie) >= h_index(serie), serie


def test_g_index_article_tres_cite_compte():
    # h plafonne à 1, mais g remonte : 25 citations couvrent 5^2.
    assert h_index([25, 0, 0, 0, 0]) == 1
    assert g_index([25, 0, 0, 0, 0]) == 5


def test_g_index_vide_et_zero():
    assert g_index([]) == 0
    assert g_index([0, 0]) == 0


def test_i10_index():
    assert i10_index([10, 9, 11, 0]) == 2


# --- au niveau du corpus ---------------------------------------------------

@pytest.fixture
def corpus() -> Corpus:
    """4 documents d'un même auteur, citations 10 / 8 / 5 / 1."""
    rows = []
    for i, cites in enumerate([10, 8, 5, 1], start=1):
        rows.append({
            "Title": "Doc %d" % i, "Year": str(2018 + i), "Cited by": str(cites),
            "EID": "eid-%d" % i, "Source title": "J", "Document Type": "Article",
            "Authors": "1:Hosni M.; 2:Idri A.",
            "Author full names": "1:Hosni, Mohamed (222); 2:Idri, Ali (111)",
            "Author(s) ID": "1:222; 2:111",
            "Affiliations": "parent 1: Moulay Ismail University, city: Meknes, country: Morocco",
        })
    return Corpus.from_dataframe(pd.DataFrame(rows))


def test_corpus_impact(corpus):
    imp = corpus.impact()
    assert imp["documents"] == 4
    assert imp["citations"] == 24
    assert imp["h_index"] == 3          # 10, 8, 5 >= 3 ; le 4e n'a qu'1
    assert imp["i10_index"] == 1        # seul le document à 10 citations
    assert imp["uncited"] == 0


def test_authors_impact(corpus):
    df = corpus.authors_impact()
    assert set(df.columns) >= {"author", "documents", "citations",
                               "h_index", "g_index", "first_author"}
    hosni = df[df["author"] == "Hosni M."].iloc[0]
    assert hosni["documents"] == 4
    assert hosni["citations"] == 24
    assert hosni["h_index"] == 3
    assert hosni["first_author"] == 4    # premier sur les 4 documents
    idri = df[df["author"] == "Idri A."].iloc[0]
    assert idri["first_author"] == 0     # toujours second


def test_authors_impact_periode(corpus):
    df = corpus.authors_impact()
    hosni = df[df["author"] == "Hosni M."].iloc[0]
    assert (hosni["first_year"], hosni["last_year"]) == (2019, 2022)


def test_institutions_impact(corpus):
    df = corpus.institutions_impact()
    assert df.iloc[0]["institution"] == "Moulay Ismail University"
    assert df.iloc[0]["documents"] == 4
    assert df.iloc[0]["h_index"] == 3
    assert df.iloc[0]["country"] == "Morocco"


def test_summary_compte_les_affiliations(corpus):
    s = corpus.summary()
    assert s["affiliations"] == 1        # une seule affiliation distincte
    assert s["institutions"] == 1


def test_corpus_vide_indices_a_zero():
    c = Corpus.from_dataframe(pd.DataFrame([{"Title": "x"}]))
    assert c.impact()["h_index"] == 0
    assert c.authors_impact().empty
