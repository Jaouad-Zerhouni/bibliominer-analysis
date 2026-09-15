"""Citations locales : rapprochement référence → document, et ses conséquences.

Ce sont les tests les plus importants du lot : tout ce qui suit (documents et
auteurs les plus cités localement, historiographe) repose sur ce rapprochement.
Une correspondance FAUSSE y est plus grave qu'une correspondance manquante,
puisqu'elle fabrique une filiation qui n'existe pas.
"""

import pandas as pd
import pytest

from bibliominer_analysis import Corpus
from bibliominer_analysis.metrics import local as L


def _doc(i, title, authors="1:A.", doi="", refs="", year="2020", cited="0"):
    return {
        "Title": title, "Year": year, "Cited by": cited,
        "EID": "eid-%d" % i, "Source title": "J1", "Document Type": "Article",
        "Authors": authors, "Author Keywords": "alpha", "DOI": doi,
        "Affiliations": "parent 1: U, city: X, country: Morocco",
        "References": refs,
    }


LONG_A = "A systematic review of ensemble effort estimation methods"
LONG_B = "Improved estimation of software development effort using ensembles"


@pytest.fixture
def corpus():
    """Doc 3 cite doc 1 par DOI et doc 2 par titre."""
    return Corpus.from_dataframe(pd.DataFrame([
        _doc(1, LONG_A, doi="10.1000/aaa", year="2016", cited="100"),
        _doc(2, LONG_B, year="2017", cited="10"),
        _doc(3, "A third paper", year="2019", refs=(
            "ref1 | 10.1000/aaa | 2016 | Idri | %s;"
            "ref2 |  | 2017 | Hosni | %s" % (LONG_A, LONG_B))),
    ]))


def test_rapprochement_par_doi_et_par_titre(corpus):
    pairs = L.citation_pairs(corpus)
    assert len(pairs) == 2
    assert set(pairs["via"]) == {"doi", "title"}
    assert set(pairs["citing"]) == {"eid-3"}
    assert set(pairs["cited"]) == {"eid-1", "eid-2"}


def test_le_doi_prime_sur_le_titre():
    """Quand les deux concordent, la paire ne compte qu'UNE fois, via le DOI."""
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, LONG_A, doi="10.1000/aaa"),
        _doc(2, "Citing", refs="ref1 | 10.1000/aaa | 2016 | X | %s" % LONG_A),
    ]))
    pairs = L.citation_pairs(c)
    assert len(pairs) == 1
    assert pairs.iloc[0]["via"] == "doi"


def test_doi_normalise_malgre_le_prefixe_url():
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, LONG_A, doi="https://doi.org/10.1000/AAA"),
        _doc(2, "Citing", refs="ref1 | doi:10.1000/aaa | 2016 | X | autre titre"),
    ]))
    assert len(L.citation_pairs(c)) == 1


def test_titre_trop_court_refuse():
    """« Machine learning » ne doit JAMAIS servir de clé de rapprochement."""
    short = "Machine learning"
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, short),
        _doc(2, "Citing", refs="ref1 |  | 2016 | X | %s" % short),
    ]))
    assert L.citation_pairs(c).empty


def test_pas_d_auto_citation():
    """Un document qui se cite lui-même ne doit pas gonfler son propre score."""
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, LONG_A, doi="10.1000/aaa",
             refs="ref1 | 10.1000/aaa | 2016 | X | %s" % LONG_A),
    ]))
    assert L.citation_pairs(c).empty
    assert int(L.local_citations(c)["local_citations"].sum()) == 0


def test_documents_les_plus_cites_localement(corpus):
    d = L.most_local_cited_documents(corpus)
    assert len(d) == 2
    row = d[d["title"] == LONG_A].iloc[0]
    assert row["local_citations"] == 1
    assert row["global_citations"] == 100
    # 1 citation locale sur 100 mondiales
    assert row["lc_gc_ratio"] == 1.0


def test_etiquettes_desambiguisees():
    """Deux articles du même auteur la même année reçoivent un suffixe."""
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, LONG_A, authors="1:Idri A.", doi="10.1000/aaa", year="2016"),
        _doc(2, LONG_B, authors="1:Idri A.", doi="10.1000/bbb", year="2016"),
        _doc(3, "Citing paper", year="2019", refs=(
            "ref1 | 10.1000/aaa | 2016 | X | t1;ref2 | 10.1000/bbb | 2016 | X | t2")),
    ]))
    labels = list(L.most_local_cited_documents(c)["label"])
    assert len(set(labels)) == 2, "deux libellés identiques rendraient le graphe faux"
    assert all(x.startswith("IDRI A., 2016") for x in labels)


def test_auteurs_cites_localement_comptage_entier():
    """Chaque signataire reçoit la totalité des citations locales du document."""
    c = Corpus.from_dataframe(pd.DataFrame([
        _doc(1, LONG_A, authors="1:A.; 2:B.", doi="10.1000/aaa"),
        _doc(2, "Citing", refs="ref1 | 10.1000/aaa | 2016 | X | t"),
    ]))
    a = L.most_local_cited_authors(c).set_index("author")["local_citations"]
    assert a["A."] == 1 and a["B."] == 1


def test_historiographe_ne_garde_que_les_liens_internes(corpus):
    h = L.historiograph(corpus, n=25)
    ids = {n["id"] for n in h["nodes"]}
    # eid-3 ne recoit aucune citation : il n'entre pas dans le graphe, donc
    # ses liens sortants ne doivent pas y figurer non plus.
    assert "eid-3" not in ids
    for e in h["edges"]:
        assert e["source"] in ids and e["target"] in ids


def test_corpus_sans_reference_ne_casse_pas():
    c = Corpus.from_dataframe(pd.DataFrame([_doc(1, LONG_A)]))
    assert L.citation_pairs(c).empty
    assert L.most_local_cited_documents(c).empty
    assert L.historiograph(c)["n_nodes"] == 0
    assert int(L.local_citations(c)["local_citations"].sum()) == 0
