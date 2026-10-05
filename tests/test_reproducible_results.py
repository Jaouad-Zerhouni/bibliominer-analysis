"""Same data, same result: whatever the order of the rows, Python's hash seed,
or the pandas version.

Defects found on real corpora:
  - the thematic map named a cluster "Software Testing Effort" in one run
    and "ISBSG" in the next (ties broken by the order of a ``set``, which
    Python shuffles at every launch);
  - the three-field plot swapped two tied authors;
  - sorting a Scopus export differently changed the references kept in the
    co-citation network, the label of a reference, and the "2018-b" of a
    document;
  - under pandas 3, 25 methods out of 100 returned different results.
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
    # Two themes whose terms are TIED in frequency: precisely the case where the
    # order of chance decided the name of the cluster.
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
    assert labels == ["alpha", "beta"]          # tied: alphabetical order


_SCRIPT = textwrap.dedent("""
    import json, sys, pandas as pd
    sys.path.insert(0, %(src)r)
    from bibliominer_analysis import Corpus
    c = Corpus.from_dataframe(pd.read_json(sys.argv[1]))
    out = {"thematic_map": c.thematic_map(min_weight=1)["clusters"].to_dict("records"),
           "three_fields": c.three_fields().to_dict("records"),
           "co_word": [(n["id"], n.get("community")) for n in
                       Corpus.attach_layout(c.co_word(min_weight=1))["nodes"]]}
    # The figure itself, byte for byte: the order of the nodes decides the
    # drawing order and which nodes receive a label.
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
    """Under pandas 3, ``astype(str)`` leaves a missing value MISSING instead of
    writing "nan": the "not empty" filters let it through. ``map(str)`` gives
    the same text under pandas 2 and 3."""
    offenders = [str(p.relative_to(root)) for p in root.rglob("*.py")
                 if ".astype(str)" in p.read_text(encoding="utf-8")]
    assert offenders == []
