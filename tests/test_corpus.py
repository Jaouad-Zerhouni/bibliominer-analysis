"""Tests of the Corpus and its indicators, on a small controlled corpus.

Same discipline as the cleaning's golden file: a tiny dataset whose answers
are known by hand, so that a regression is obvious.
"""

import pandas as pd
import pytest

from bibliominer_analysis import Corpus
from bibliominer_analysis.io.schema import INDEPENDENT_LABEL


@pytest.fixture
def mini() -> Corpus:
    """3 documents, 2 countries, 1 independent researcher, 1 co-publication."""
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


def test_six_tables_present(mini):
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


def test_production_by_year_without_gap(mini):
    g = mini.production_by_year()
    assert list(g["year"]) == [2020, 2021]
    assert list(g["documents"]) == [1, 2]
    assert list(g["cumulative"]) == [1, 3]


def test_country_counted_once_per_document(mini):
    """Doc B is Moroccan-Spanish: it counts once for each country."""
    g = mini.production_by_country().set_index("country")["documents"]
    assert g["Spain"] == 2       # Doc A + Doc B
    assert g["Morocco"] == 2     # Doc B + Doc C


def test_institutions_exclude_independents(mini):
    '''"Independent researcher" is not an institution.'''
    inst = list(mini.top_institutions()["institution"])
    assert INDEPENDENT_LABEL not in inst
    assert "University of Murcia" in inst


def test_top_authors_counts_the_first_authors(mini):
    g = mini.top_authors().set_index("author")
    assert g.loc["Hosni M.", "documents"] == 2
    assert g.loc["Hosni M.", "first_author"] == 1     # first author on Doc B only
    assert g.loc["Idri A.", "first_author"] == 1


def test_keywords_case_insensitive(mini):
    g = mini.top_keywords().set_index("keyword")["documents"]
    assert g.get("Machine learning", g.get("machine learning")) == 2


def test_filter_does_not_modify_the_original(mini):
    f = mini.filter(years=(2021, 2021))
    assert len(f) == 2
    assert len(mini) == 3                       # the original is untouched
    assert set(f.authors["eid"]) <= {"eid-B", "eid-C"}


def test_filter_by_country(mini):
    f = mini.filter(countries=["Spain"])
    assert set(f.documents["eid"]) == {"eid-A", "eid-B"}


def test_empty_corpus_does_not_crash():
    c = Corpus.from_dataframe(pd.DataFrame([{"Title": "Nothing at all"}]))
    assert c.summary()["documents"] == 1
    assert c.production_by_year().empty        # no usable year
    assert c.top_authors().empty


def test_required_columns():
    with pytest.raises(ValueError, match="Required columns missing"):
        Corpus.from_dataframe(pd.DataFrame([{"Other": "x"}]))
