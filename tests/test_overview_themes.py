"""Fiche signalétique, documents les plus cités, sujets et carte thématique."""

import pandas as pd
import pytest

from bibliominer_analysis import Corpus


def _doc(i, year, cites, authors, kws="", refs="", affs=None):
    return {
        "Title": "Doc %d" % i, "Year": str(year), "Cited by": str(cites),
        "EID": "eid-%d" % i, "Source title": "J%d" % (i % 3),
        "Document Type": "Article", "Authors": authors,
        "Author Keywords": kws, "References": refs,
        "Affiliations": affs or "parent 1: U, city: X, country: Morocco",
    }


@pytest.fixture
def corpus() -> Corpus:
    """4 documents 2020-2023, dont un solo et un maroco-espagnol."""
    return Corpus.from_dataframe(pd.DataFrame([
        _doc(1, 2020, 10, "1:A.; 2:B.", kws="alpha; beta",
             refs="ref1 | 10.1/x | 2010 | Doe J. | Fondation"),
        _doc(2, 2021, 6, "1:A.; 2:C.", kws="alpha; gamma",
             refs="ref1 | 10.1/x | 2010 | Doe J. | Fondation"),
        _doc(3, 2022, 2, "1:A.", kws="alpha",
             refs="ref1 | 10.1/y | 2012 | Roe R. | Autre"),
        _doc(4, 2023, 0, "1:D.; 2:E.", kws="delta",
             affs=("parent 1: U, city: X, country: Morocco; "
                   "parent 1: V, city: Y, country: Spain")),
    ]))


# --- fiche signalétique -----------------------------------------------------

def test_main_information(corpus):
    m = corpus.main_information()
    assert m["documents"] == 4
    assert m["timespan"] == "2020-2023"
    assert m["total_citations"] == 18
    assert m["single_authored_documents"] == 1        # Doc 3
    assert m["authors_per_document"] == 1.75          # (2+2+1+2)/4
    assert m["collaboration_index"] == 2.0            # moyenne sur les co-signés
    assert m["international_documents"] == 1          # Doc 4
    assert m["international_share"] == 25.0


def test_age_moyen_relatif_au_corpus(corpus):
    """L'âge se compte depuis l'année la plus RÉCENTE du corpus, pas depuis
    aujourd'hui — sinon la valeur changerait chaque année."""
    m = corpus.main_information()
    # années 2020..2023, la plus récente = 2023 -> âges 3,2,1,0 -> moyenne 1.5
    assert m["document_average_age"] == 1.5


def test_taux_de_croissance(corpus):
    """1 document en 2020, 1 en 2023 -> croissance nulle."""
    assert corpus.main_information()["annual_growth_rate"] == 0.0


def test_main_information_une_seule_annee():
    c = Corpus.from_dataframe(pd.DataFrame([_doc(1, 2020, 0, "1:A.")]))
    assert c.main_information()["annual_growth_rate"] is None


# --- documents et références les plus cités ---------------------------------

def test_most_cited_documents(corpus):
    d = corpus.most_cited_documents()
    assert d.iloc[0]["title"] == "Doc 1"
    assert d.iloc[0]["citations"] == 10
    assert d.iloc[0]["first_author"] == "A."
    # 10 citations, publié en 2020, corpus jusqu'en 2023 -> 4 ans -> 2.5/an
    assert d.iloc[0]["citations_per_year"] == 2.5


def test_most_cited_references(corpus):
    r = corpus.most_cited_references()
    top = r.iloc[0]
    assert top["local_citations"] == 2       # « Fondation » cité par Doc 1 et 2
    assert "Fondation" in str(top["reference"])


# --- sujets dans le temps ---------------------------------------------------

def test_trend_topics_mediane(corpus):
    t = corpus.trend_topics(min_documents=1).set_index("keyword")
    # alpha apparaît en 2020, 2021, 2022 -> médiane 2021
    assert t.loc["alpha", "year_median"] == 2021
    assert t.loc["alpha", "documents"] == 3
    assert t.loc["delta", "year_median"] == 2023


def test_trend_topics_seuil(corpus):
    """Le seuil écarte les termes trop rares pour être interprétables."""
    t = corpus.trend_topics(min_documents=3)
    assert list(t["keyword"]) == ["alpha"]


# --- carte thématique -------------------------------------------------------

def test_thematic_map_quadrants():
    """Deux groupes bien séparés : chacun doit être identifié."""
    rows = []
    # Groupe 1 : a, b, c toujours ensemble. Groupe 2 : x, y toujours ensemble.
    for i in range(4):
        rows.append(_doc(i, 2020 + i, 0, "1:A.", kws="a; b; c"))
    for i in range(4, 7):
        rows.append(_doc(i, 2020, 0, "1:A.", kws="x; y"))
    # Un document fait le pont, pour créer des liens externes.
    rows.append(_doc(9, 2021, 0, "1:A.", kws="a; x"))
    c = Corpus.from_dataframe(pd.DataFrame(rows))

    res = c.thematic_map(min_weight=2)
    cl = res["clusters"]
    assert len(cl) >= 2
    assert set(cl.columns) >= {"label", "terms", "centrality", "density", "quadrant"}
    assert cl["quadrant"].isin(
        {"motor", "basic", "niche", "emerging_declining"}).all()
    assert res["medians"]["centrality"] >= 0


def test_thematic_map_corpus_vide():
    c = Corpus.from_dataframe(pd.DataFrame([{"Title": "x"}]))
    assert c.thematic_map()["clusters"].empty


# --- couplage bibliographique -----------------------------------------------

def test_bibliographic_coupling(corpus):
    """Doc 1 et Doc 2 partagent la référence 10.1/x -> lien de poids 1."""
    g = corpus.bibliographic_coupling(min_weight=1)
    pairs = {(e["source"], e["target"]): e["weight"] for e in g["edges"]}
    assert pairs.get(("eid-1", "eid-2")) == 1
    labels = {n["id"]: n["label"] for n in g["nodes"]}
    assert "Doc 1" in labels["eid-1"]


def test_bibliographic_coupling_seuil(corpus):
    """Avec un seuil de 2 références communes, plus aucun lien ici."""
    assert corpus.bibliographic_coupling(min_weight=2)["n_edges"] == 0
