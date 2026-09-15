"""Tests du Corpus et des indicateurs, sur un mini-corpus contrôlé.

Même discipline que le golden du cleaning : un jeu de données minuscule dont
on connaît les réponses à la main, pour qu'une régression saute aux yeux.
"""

import pandas as pd
import pytest

from bibliominer_analysis import Corpus
from bibliominer_analysis.io.schema import INDEPENDENT_LABEL


@pytest.fixture
def mini() -> Corpus:
    """3 documents, 2 pays, 1 chercheur indépendant, 1 co-publication."""
    df = pd.DataFrame([
        {
            "Title": "Doc A", "Year": "2020", "Cited by": "10",
            "Source title": "Journal X", "Document Type": "Article",
            "EID": "eid-A", "DOI": "10.1/a",
            "Authors": "1:Idri A.; 2:Hosni M.",
            "Author full names": "1:Idri, Ali (111); 2:Hosni, Mohamed (222)",
            "Author(s) ID": "1:111; 2:222",
            "Affiliations": "parent 1: University of Murcia, city: Murcia, country: Spain",
            "Author Keywords": "Machine learning; Random forest",
            "References": "ref1 | 10.9/x | 2015 | Doe J. | Old paper",
        },
        {
            "Title": "Doc B", "Year": "2021", "Cited by": "5",
            "Source title": "Journal X", "Document Type": "Article",
            "EID": "eid-B", "DOI": "10.1/b",
            "Authors": "1:Hosni M.",
            "Author full names": "1:Hosni, Mohamed (222)",
            "Author(s) ID": "1:222",
            "Affiliations": ("parent 1: Moulay Ismail University, city: Meknes, country: Morocco; "
                             "parent 1: University of Murcia, city: Murcia, country: Spain"),
            "Author Keywords": "machine learning",
            "References": "ref1 | 10.9/x | 2015 | Doe J. | Old paper ; ref2 | | | | Raw ref",
        },
        {
            "Title": "Doc C", "Year": "2021", "Cited by": "0",
            "Source title": "Journal Y", "Document Type": "Conference paper",
            "EID": "eid-C", "DOI": "",
            "Authors": "1:Solo S.",
            "Author full names": "1:Solo, Sam (333)",
            "Author(s) ID": "1:333",
            "Affiliations": "parent 1: %s, city: Rabat, country: Morocco" % INDEPENDENT_LABEL,
            "Author Keywords": "",
            "References": "",
        },
    ])
    return Corpus.from_dataframe(df)


def test_six_tables_presentes(mini):
    t = mini.tables()
    assert set(t) == {"documents", "authors", "affiliations",
                      "author_affiliations", "keywords", "references"}
    assert len(t["documents"]) == 3
    assert len(t["authors"]) == 4          # 2 + 1 + 1
    assert len(t["affiliations"]) == 4     # 1 + 2 + 1
    assert len(t["references"]) == 3       # 1 + 2 + 0


def test_summary(mini):
    s = mini.summary()
    assert s["documents"] == 3
    assert (s["year_min"], s["year_max"]) == (2020, 2021)
    assert s["authors"] == 3               # Idri, Hosni, Solo
    assert s["countries"] == 2             # Spain, Morocco
    assert s["citations"] == 15
    assert s["documents_without_references"] == 1   # Doc C


def test_production_by_year_sans_trou(mini):
    g = mini.production_by_year()
    assert list(g["year"]) == [2020, 2021]
    assert list(g["documents"]) == [1, 2]
    assert list(g["cumulative"]) == [1, 3]


def test_pays_compte_une_fois_par_document(mini):
    """Doc B est maroco-espagnol : il compte 1 fois pour chaque pays."""
    g = mini.production_by_country().set_index("country")["documents"]
    assert g["Spain"] == 2       # Doc A + Doc B
    assert g["Morocco"] == 2     # Doc B + Doc C


def test_institutions_excluent_les_independants(mini):
    """« Independent researcher » n'est pas une institution."""
    inst = list(mini.top_institutions()["institution"])
    assert INDEPENDENT_LABEL not in inst
    assert "University of Murcia" in inst


def test_top_authors_compte_les_premiers_auteurs(mini):
    g = mini.top_authors().set_index("author")
    assert g.loc["Hosni M.", "documents"] == 2
    assert g.loc["Hosni M.", "first_author"] == 1     # premier sur Doc B seulement
    assert g.loc["Idri A.", "first_author"] == 1


def test_keywords_insensibles_a_la_casse(mini):
    g = mini.top_keywords().set_index("keyword")["documents"]
    assert g.get("Machine learning", g.get("machine learning")) == 2


def test_filter_ne_modifie_pas_l_original(mini):
    f = mini.filter(years=(2021, 2021))
    assert len(f) == 2
    assert len(mini) == 3                       # l'original est intact
    assert set(f.authors["eid"]) <= {"eid-B", "eid-C"}


def test_filter_par_pays(mini):
    f = mini.filter(countries=["Spain"])
    assert set(f.documents["eid"]) == {"eid-A", "eid-B"}


def test_corpus_vide_ne_plante_pas():
    c = Corpus.from_dataframe(pd.DataFrame([{"Title": "Sans rien"}]))
    assert c.summary()["documents"] == 1
    assert c.production_by_year().empty        # aucune année exploitable
    assert c.top_authors().empty


def test_colonnes_indispensables():
    with pytest.raises(ValueError, match="Colonnes indispensables"):
        Corpus.from_dataframe(pd.DataFrame([{"Autre": "x"}]))
