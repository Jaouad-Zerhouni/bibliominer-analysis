"""The interface charts that are neither bars, nor lines, nor scatter plots
(world maps, treemap, word cloud, Sankey, dendrogram, density map) each
have their rendering in the package.

And the computation fixes that come with them: one spelling per author
and per keyword, the thematic map measured on all co-occurrences.
"""

from __future__ import annotations

import pandas as pd
import pytest

from bibliominer_analysis.figures import (atlas_name, render_dendrogram,
                                          render_density_map, render_thematic_evolution,
                                          render_three_fields, render_treemap,
                                          render_word_cloud, render_world_map)
from bibliominer_analysis.figures.world import _atlas
from bibliominer_analysis.io.reader import unify_spellings

PNG = b"\x89PNG\r\n\x1a\n"


@pytest.mark.parametrize("corpus_name, atlas", [
    ("United States", "United States of America"),
    ("Russian Federation", "Russia"),
    ("Viet Nam", "Vietnam"),
    ("Czech Republic", "Czechia"),
    ("Korea, Republic of", "South Korea"),
    ("Côte d’Ivoire", "Côte d'Ivoire"),
    ("Morocco", "Morocco"),
    ("morocco", "Morocco"),
])
def test_noms_de_pays_vers_le_fond_de_carte(corpus_name, atlas):
    assert atlas_name(corpus_name) == atlas


def test_pays_absent_du_fond_de_carte():
    assert atlas_name("Hong Kong") is None


def test_aucun_anneau_ne_traverse_la_carte():
    """A ring that crosses the 180° line must not jump from one edge to the
    other (otherwise a horizontal band strikes across the map)."""
    shapes, _ = _atlas()
    for name, rings in shapes.items():
        for ring in rings:
            jumps = [abs(b[0] - a[0]) for a, b in zip(ring, ring[1:])]
            assert max(jumps, default=0) <= 180, name


def test_cartes_du_monde_png():
    values = {"Morocco": 85, "Spain": 69, "United States": 3, "Hong Kong": 1}
    assert render_world_map(values, dpi=60).startswith(PNG)
    links = [("Morocco", "Spain", 40), ("Spain", "United States", 2)]
    assert render_world_map(values, links=links, dpi=60).startswith(PNG)


def test_graphiques_sans_axes_png():
    kw = pd.DataFrame({"keyword": ["Machine learning", "Classification", "Usability"],
                       "documents": [24, 18, 7]})
    assert render_treemap(kw, dpi=60).startswith(PNG)
    assert render_word_cloud(kw, dpi=60).startswith(PNG)
    links = pd.DataFrame({"source": ["A", "A", "k1", "k2"], "target": ["k1", "k2", "S", "S"],
                          "value": [3, 1, 2, 1], "depth": [0, 0, 1, 1]})
    assert render_three_fields(links, dpi=60).startswith(PNG)
    evolution = {
        "periods": [{"label": "P1"}, {"label": "P2"}],
        "nodes": [{"name": "P1: ML", "period": "P1", "occurrences": 5},
                  {"name": "P1: Alone", "period": "P1", "occurrences": 2},
                  {"name": "P2: ML", "period": "P2", "occurrences": 7}],
        "flows": [{"source": "P1: ML", "target": "P2: ML", "value": 4}],
    }
    assert render_thematic_evolution(evolution, dpi=60).startswith(PNG)
    tree = {"height": 1.0, "children": [{"name": "a", "height": 0.0},
                                         {"height": 0.5, "children": [{"name": "b"}, {"name": "c"}]}]}
    assert render_dendrogram({"tree": tree}, dpi=60).startswith(PNG)
    density = {"x": [0, 1], "y": [0, 1], "cells": [[0, 0, 1.0], [1, 1, 0.5]], "max": 1}
    graph = {"nodes": [{"label": "a", "x": 0.2, "y": 0.3, "occurrences": 3}]}
    assert render_density_map(density, graph, dpi=60).startswith(PNG)


def _tables(authors, keywords):
    return {"authors": pd.DataFrame(authors), "keywords": pd.DataFrame(keywords)}


def test_une_graphie_par_auteur_selon_son_identifiant():
    t = unify_spellings(_tables(
        {"eid": ["d1", "d2", "d3", "d4"],
         "name": ["Fernández-Alemán J.L.", "Fernández-Alemán J.L.",
                  "Fernandez-Aleman J.L.", "Noid A."],
         "scopus_id": ["1", "1", "1", None]},
        {"eid": [], "keyword": [], "kind": []}))
    names = list(t["authors"]["name"])
    assert names[:3] == ["Fernández-Alemán J.L."] * 3
    assert names[3] == "Noid A."


def test_une_graphie_par_mot_cle_sans_la_casse():
    t = unify_spellings(_tables(
        {"eid": [], "name": [], "scopus_id": []},
        {"eid": ["d1", "d2", "d3", "d4"],
         "keyword": ["Machine learning", "Machine learning", "Machine Learning", "ML"],
         "kind": ["author"] * 4}))
    assert list(t["keywords"]["keyword"]) == ["Machine learning"] * 3 + ["ML"]


def test_mots_cles_de_types_differents_restent_separes():
    t = unify_spellings(_tables(
        {"eid": [], "name": [], "scopus_id": []},
        {"eid": ["d1", "d2", "d3"], "keyword": ["Ecology", "ecology", "ecology"],
         "kind": ["author", "index", "index"]}))
    assert list(t["keywords"]["keyword"]) == ["Ecology", "ecology", "ecology"]


def test_organisations_meme_nom_a_la_ponctuation_pres():
    tables = _tables({"eid": [], "name": [], "scopus_id": []},
                     {"eid": [], "keyword": [], "kind": []})
    tables["affiliations"] = pd.DataFrame({
        "eid": ["d1", "d2", "d3", "d4"],
        "subparent": ["Faculty of Sciences Oujda - FSO", "Faculty of Sciences Oujda - FSO",
                      "Faculty of Sciences Oujda-FSO", "Facultad de Informática"],
        "parent1": ["Mohammed 1st University"] * 3 + ["Universidad de Murcia"]})
    out = unify_spellings(tables)["affiliations"]
    assert list(out["subparent"][:3]) == ["Faculty of Sciences Oujda - FSO"] * 3
    # A translation is NOT merged here: that is the cleaning's job.
    assert out["subparent"][3] == "Facultad de Informática"


def test_colonnes_non_textuelles_intactes():
    """A boolean indicator (labelled) keeps its type: only a column of NAMES is
    normalised."""
    tables = _tables({"eid": [], "name": [], "scopus_id": []},
                     {"eid": [], "keyword": [], "kind": []})
    tables["affiliations"] = pd.DataFrame({
        "eid": ["d1", "d2"], "labelled": [True, False],
        "subparent": ["Lab A", "lab a"], "parent1": ["U", "U"]})
    out = unify_spellings(tables)["affiliations"]
    assert out["labelled"].dtype == bool
    assert list(out["labelled"]) == [True, False]


@pytest.mark.parametrize("name, expected", [
    ("Morocco", True), ("United States", True), ("Russian Federation", True),
    ("Singapore", True), ("Hong Kong", True), ("Viet Nam", True), ("Kosovo", True),
    ("University Moulay Ismail of Meknes", False), ("Meknes", False), ("", False),
])
def test_reconnaissance_des_pays(name, expected):
    from bibliominer_analysis.io.countries import is_country
    assert is_country(name) is expected


def test_petits_pays_places_a_leur_capitale():
    from bibliominer_analysis.io.countries import atlas_name, position
    assert atlas_name("Singapore") is None          # absent from the 1:110 m base map
    lon, lat = position("Singapore")
    assert 103 < lon < 105 and 0 < lat < 2
    assert render_world_map({"Singapore": 3, "Morocco": 5}, dpi=60).startswith(PNG)
    assert render_world_map({"Singapore": 3, "Morocco": 5},
                            links=[("Singapore", "Morocco", 2)], dpi=60).startswith(PNG)


def test_affiliation_brute_sans_pays_reconnu():
    """The last segment is a country only if it is one: otherwise the geography
    stays empty, and that segment is the organisation."""
    from bibliominer_analysis.io import parsers as P
    r = P.parse_affiliation("LIMIE Laboratory, University Moulay Ismail of Meknes")
    assert r["country"] is None and r["city"] is None
    assert r["parent1"] == "University Moulay Ismail of Meknes"
    r = P.parse_affiliation("ENSIAS, Mohammed V University, Rabat, Morocco")
    assert r["country"] == "Morocco" and r["city"] == "Rabat"


def test_trois_champs_meme_nom_dans_deux_colonnes():
    """A keyword and a journal with the same name stay two distinct nodes."""
    links = pd.DataFrame({"source": ["A", "Energies"], "target": ["Energies", "Energies"],
                          "value": [2, 2], "depth": [0, 1]})
    assert render_three_fields(links, dpi=60).startswith(PNG)


def test_first_author_countries_is_the_same_count_under_its_true_name():
    """The SCP/MCP table is about the FIRST author; the old name remains an
    alias, for existing scripts."""
    from bibliominer_analysis import Corpus
    rows = pd.DataFrame([{
        "Title": "Paper", "Year": "2024", "EID": "e1",
        "Authors": "A B.; C D.", "Author full names": "A, Bob (1); C, Dan (2)",
        "Author(s) ID": "1; 2",
        "Affiliations": "Lab, Univ X, Rabat, Morocco; Lab, Univ Y, Madrid, Spain",
        "Authors with affiliations": "A B., Lab, Univ X, Rabat, Morocco; C D., Lab, Univ Y, Madrid, Spain",
    }])
    c = Corpus.from_dataframe(rows)
    first = c.first_author_countries()
    assert first.equals(c.corresponding_author_countries())
    assert list(first["country"]) == ["Morocco"] and int(first["mcp"].iloc[0]) == 1
