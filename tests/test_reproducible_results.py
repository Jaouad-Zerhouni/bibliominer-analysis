"""Mêmes données, même résultat : quel que soit l'ordre des lignes, la graine
de hachage de Python, ou la version de pandas.

Défauts constatés sur de vrais corpus :
  - la carte thématique nommait un groupe « Software Testing Effort » à une
    exécution et « ISBSG » à la suivante (ex æquo départagés par l'ordre d'un
    ``set``, que Python mélange à chaque lancement) ;
  - le diagramme à trois champs échangeait deux auteurs à égalité ;
  - trier un export Scopus autrement changeait les références gardées dans le
    réseau de co-citation, l'étiquette d'une référence, et le « 2018-b » d'un
    document ;
  - sous pandas 3, 25 méthodes sur 100 rendaient d'autres résultats.
"""
import os
import pathlib
import subprocess
import sys
import textwrap

import pandas as pd
import pytest

from bibliominer_analysis import Corpus
from bibliominer_analysis._stable import ordered_communities, top_by_count

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"


def _doc(i, kws, refs="", authors="1:Doe J.", affs="", cited="0", year=2020):
    return {"Title": "Document %d" % i, "Year": str(year), "Cited by": cited,
            "EID": "eid-%02d" % i, "Source title": "J%d" % (i % 3),
            "Document Type": "Article", "Authors": authors,
            "Author Keywords": kws, "References": refs, "Affiliations": affs}


def _corpus_rows():
    # Deux thèmes dont les termes sont À ÉGALITÉ de fréquence : c'est
    # précisément le cas où l'ordre du hasard décidait du nom du groupe.
    rows = []
    for i in range(12):
        theme = ("zeta; alpha; mu" if i % 2 else "omega; beta; nu")
        rows.append(_doc(i, theme, authors="1:Author%d A.; 2:Author%d B." % (i % 4, (i + 1) % 4),
                         affs="parent 1: Univ %d, city: City%d, country: Country%d" % (i % 3, i % 3, i % 2)))
    return rows


def test_ties_are_broken_alphabetically():
    assert top_by_count({"b": 2, "a": 2, "c": 3}, 2) == [("c", 3), ("a", 2)]
    assert ordered_communities([{"y", "z"}, {"b", "a"}, {"q"}]) == [{"a", "b"}, {"y", "z"}, {"q"}]


def test_thematic_map_names_groups_the_same_way_whatever_the_row_order():
    rows = _corpus_rows()
    a = Corpus.from_dataframe(pd.DataFrame(rows)).thematic_map(min_weight=1)
    b = Corpus.from_dataframe(pd.DataFrame(rows[::-1])).thematic_map(min_weight=1)
    labels = sorted(a["clusters"]["label"])
    assert labels == sorted(b["clusters"]["label"])
    assert labels == ["alpha", "beta"]          # à égalité : l'ordre alphabétique


_SCRIPT = textwrap.dedent("""
    import json, sys, pandas as pd
    sys.path.insert(0, %(src)r)
    from bibliominer_analysis import Corpus
    c = Corpus.from_dataframe(pd.read_json(sys.argv[1]))
    out = {"thematic_map": c.thematic_map(min_weight=1)["clusters"].to_dict("records"),
           "three_fields": c.three_fields().to_dict("records"),
           "co_word": [(n["id"], n.get("community")) for n in
                       Corpus.attach_layout(c.co_word(min_weight=1))["nodes"]]}
    # La figure elle-même, octet pour octet : l'ordre des nœuds décide de
    # l'ordre de dessin et de ceux qui reçoivent une étiquette.
    import hashlib
    from bibliominer_analysis.networks.analysis import annotate
    from bibliominer_analysis.figures.network import render_network
    g = annotate(c.co_word(min_weight=1), communities=True)
    out["figure"] = hashlib.sha256(render_network(g, title="t", fmt="svg")).hexdigest()
    print(json.dumps(out, sort_keys=True, default=str))
""")


def test_results_do_not_depend_on_the_python_hash_seed(tmp_path):
    data = tmp_path / "corpus.json"
    pd.DataFrame(_corpus_rows()).to_json(data)
    script = tmp_path / "run.py"
    script.write_text(_SCRIPT % {"src": str(SRC)}, encoding="utf-8")
    outputs = set()
    for seed in ("1", "2", "3"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        done = subprocess.run([sys.executable, str(script), str(data)], env=env,
                              capture_output=True, text=True, check=True)
        outputs.add(done.stdout)
    assert len(outputs) == 1


@pytest.mark.parametrize("root", [SRC])
def test_no_astype_str_in_the_package(root):
    """Sous pandas 3, ``astype(str)`` laisse une valeur manquante MANQUANTE au
    lieu d'écrire « nan » : les filtres « non vide » la laissaient passer.
    ``map(str)`` rend le même texte sous pandas 2 et 3."""
    offenders = [str(p.relative_to(root)) for p in root.rglob("*.py")
                 if ".astype(str)" in p.read_text(encoding="utf-8")]
    assert offenders == []
