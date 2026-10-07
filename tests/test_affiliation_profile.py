"""Affiliation profile: one affiliation, two, more, and authors with several
affiliations."""
import pandas as pd
import pytest

from bibliominer_analysis import Corpus

A = "subparent: ENSIAS, parent 1: Mohammed V University, city: Rabat, country: Morocco"
B = "subparent: ENSAM, parent 1: Moulay Ismail University, city: Meknes, country: Morocco"
C = "subparent: FS, parent 1: University of Murcia, city: Murcia, country: Spain"


def _doc(i, authors, full_names, ids, affs, awa, year=2020):
    return {"Title": "Document %d" % i, "Year": str(year), "EID": "eid-%d" % i,
            "Cited by": "0", "Source title": "J", "Document Type": "Article",
            "Authors": authors, "Author full names": full_names,
            "Author(s) ID": ids, "Affiliations": affs,
            "Authors with affiliations": awa}


@pytest.fixture
def corpus():
    return Corpus.from_dataframe(pd.DataFrame([
        # a single author, a single affiliation
        _doc(1, "Solo A.", "Solo, Ann (1)", "1", A, f"Solo A., {A}"),
        # two authors, two affiliations, including a DOUBLE affiliation
        _doc(2, "Okafor M.; Varela A.", "Okafor, Maya (2); Varela, Ana (3)", "2; 3",
             f"{A}; {B}", f"Okafor M., {A}, {B}; Varela A., {A}"),
        # three affiliations
        _doc(3, "Okafor M.; Juan C.", "Okafor, Maya (2); Prado, Juan (4)", "2; 4",
             f"{A}; {B}; {C}", f"Okafor M., {B}; Juan C., {C}"),
    ]))


def test_documents_by_affiliation_count(corpus):
    table = corpus.documents_by_affiliation_count().set_index("affiliations")
    assert table.loc[1, "documents"] == 1
    assert table.loc[2, "documents"] == 1
    assert table.loc[3, "documents"] == 1
    # 3 x 33.33: rounding each share leaves one hundredth.
    assert table["share"].sum() == pytest.approx(100.0, abs=0.05)


def test_the_profile_counts_single_author_and_single_affiliation(corpus):
    p = corpus.affiliation_profile()
    assert (p["documents"], p["single_author_documents"]) == (3, 1)
    assert p["single_affiliation_documents"] == 1
    assert p["two_affiliation_documents"] == 1
    assert p["many_affiliation_documents"] == 1
    assert p["single_author_share"] == pytest.approx(33.33, abs=0.01)


def test_an_author_with_two_institutions_on_one_article_is_counted(corpus):
    p = corpus.affiliation_profile()
    assert p["authors_with_double_affiliation"] == 1        # Okafor, article 2
    authors = corpus.authors_by_affiliation_count().set_index("author")
    assert authors.loc["Okafor M.", "max_in_one_document"] == 2
    assert authors.loc["Okafor M.", "institutions"] == 2
    assert "Varela A." not in authors.index                   # a single institution


def test_an_author_who_changed_institution_is_not_a_double_affiliation():
    """Two institutions from one article to the next: that is mobility, not a
    declared double affiliation."""
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, "Move A.", "Move, Ann (9)", "9", A, f"Move A., {A}", year=2018),
        _doc(2, "Move A.", "Move, Ann (9)", "9", B, f"Move A., {B}", year=2022),
    ]))
    row = c.authors_by_affiliation_count().iloc[0]
    assert (row["institutions"], row["max_in_one_document"]) == (2, 1)
    assert c.affiliation_profile()["authors_with_double_affiliation"] == 0


def test_an_empty_corpus_does_not_break():
    c = Corpus.from_dataframe(pd.DataFrame([
        {"Title": "t", "Year": "2020", "EID": "e", "Cited by": "0",
         "Source title": "J", "Document Type": "Article", "Authors": "",
         "Author full names": "", "Author(s) ID": "", "Affiliations": "",
         "Authors with affiliations": ""}]))
    assert c.authors_by_affiliation_count().empty
    assert c.affiliation_profile()["single_affiliation_documents"] == 0
