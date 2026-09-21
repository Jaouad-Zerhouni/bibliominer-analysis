"""Double rattachement (« parent 2 ») et auteur à plusieurs affiliations.

Deux défauts constatés sur des fichiers nettoyés réels :
  - « parent 2 » était lu puis ignoré partout : le second organisme d'une
    affiliation n'apparaissait dans aucun classement ;
  - le lien auteur -> affiliation se faisait au RANG (auteur 3 -> affiliation
    3) : dès qu'un auteur en avait deux, ses voisins héritaient d'une
    affiliation qui n'était pas la leur, et sa seconde disparaissait.
"""
import pandas as pd

from bibliominer_analysis import Corpus
from bibliominer_analysis.io import parsers as P

ENSIAS = "subparent: ENSIAS, parent 1: Mohammed V University, city: Rabat, country: Morocco"
MURCIA = ("subparent: Faculty of Computer Science, parent 1: University of Murcia, "
          "city: Murcia, country: Spain")
ENSAM = "subparent: ENSAM, parent 1: Moulay Ismail University, city: Meknès, country: Morocco"
JOINT = ("subparent: LIRMM, parent 1: University of Montpellier, parent 2: CNRS, "
         "city: Montpellier, country: France")


def _doc(i, affs, awa="", authors="1:Doe J.", **extra):
    row = {"Title": "Document %d" % i, "Year": "2022", "EID": "eid-%d" % i,
           "Cited by": "3", "Source title": "J", "Document Type": "Article",
           "Authors": authors, "Affiliations": affs,
           "Authors with affiliations": awa}
    row.update(extra)
    return row


# --- auteur à plusieurs affiliations -----------------------------------------

def _hosni_paper():
    return _doc(
        1, "; ".join([ENSIAS, MURCIA, ENSAM]),
        authors="1:Hosni M.; 2:Carrillo de Gea J.M.; 3:Idri A.",
        awa="; ".join([f"Hosni M., {ENSIAS}, {ENSAM}",
                       f"Carrillo de Gea J.M., {MURCIA}",
                       f"Idri A., {ENSIAS}"]))


def test_an_author_with_two_affiliations_keeps_both():
    c = Corpus.from_dataframe(pd.DataFrame([_hosni_paper()]))
    link = c.author_affiliations
    assert sorted(link.loc[link["position"] == 1, "aff_pos"]) == [1, 3]


def test_the_next_authors_keep_their_own_affiliation():
    """Au rang, Idri (3e auteur) recevait ENSAM Meknès, l'affiliation n°3."""
    c = Corpus.from_dataframe(pd.DataFrame([_hosni_paper()]))
    link = c.author_affiliations.merge(c.affiliations, on=["eid", "aff_pos"])
    cities = dict(zip(link["position"].astype(int).astype(str) + link["city"],
                      link["city"]))
    assert "3Rabat" in cities and "3Meknès" not in cities
    assert set(link.loc[link["position"] == 2, "city"]) == {"Murcia"}


def test_a_second_affiliation_starting_with_parent_1_is_split():
    """Sans « subparent », la seconde affiliation commence par « parent 1 »,
    qui REMONTE dans l'ordre des libellés après « country »."""
    cairo = "parent 1: African Disaster Mitigation Research Center, city: Cairo, country: Egypt"
    blocks = P.author_affiliation_positions(
        f"Chourak M., {ENSIAS}, {cairo}", P.parse_affiliations(f"{ENSIAS}; {cairo}"))
    assert [pos for pos, _ in blocks] == [1, 2]


def test_a_raw_scopus_export_matches_by_text():
    affs = "ENSIAS, Mohammed V University, Rabat, Morocco; ENSAM, Moulay Ismail University, Meknes, Morocco"
    blocks = P.author_affiliation_positions(
        "Hosni M., ENSAM, Moulay Ismail University, Meknes, Morocco, "
        "ENSIAS, Mohammed V University, Rabat, Morocco",
        P.parse_affiliations(affs))
    assert [pos for pos, _ in blocks] == [2, 1]


# --- double rattachement ------------------------------------------------------

def test_parent_2_is_a_parse_field():
    assert P.parse_affiliation(JOINT)["parent2"] == "CNRS"


def test_both_parents_are_ranked_as_institutions():
    c = Corpus.from_dataframe(pd.DataFrame([_doc(1, JOINT), _doc(2, JOINT)]))
    top = c.top_institutions(10).set_index("institution")
    assert top.loc["University of Montpellier", "documents"] == 2
    assert top.loc["CNRS", "documents"] == 2
    assert c.summary()["institutions"] == 2


def test_the_unit_appears_under_both_parents():
    c = Corpus.from_dataframe(pd.DataFrame([_doc(1, JOINT)]))
    h = c.org_hierarchy(10)
    assert set(h.loc[h["subparent"] == "LIRMM", "parent"]) == {
        "University of Montpellier", "CNRS"}


def test_the_city_lists_both_parents():
    c = Corpus.from_dataframe(pd.DataFrame([_doc(1, JOINT)]))
    row = c.city_hierarchy().set_index("city").loc["Montpellier"]
    assert row["institutions"] == 2
    assert row["institution_affiliations"] == "CNRS (1); University of Montpellier (1)"
    assert c.top_cities().set_index("city").loc["Montpellier", "institutions"] == 2


def test_a_parent_2_equal_to_parent_1_counts_once():
    same = JOINT.replace("parent 2: CNRS", "parent 2: University of Montpellier")
    c = Corpus.from_dataframe(pd.DataFrame([_doc(1, same)]))
    assert c.summary()["institutions"] == 1
    assert c.city_hierarchy().iloc[0]["institution_affiliations"] == \
        "University of Montpellier (1)"
