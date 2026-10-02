"""Flux entre trois dimensions du corpus (diagramme de Sankey).

Relier trois dimensions, typiquement auteurs → mots-clés → sources, montre
« qui travaille sur quoi et publie où », ce qu'aucun classement isolé ne dit.

Le calcul est une double co-occurrence : deux valeurs sont reliées quand elles
apparaissent dans le MÊME document, et l'épaisseur du lien est le nombre de
documents partagés.

Un garde-fou compte plus que tous les autres : on ne retient que les valeurs
les plus fréquentes de chaque colonne. Sans ce filtre, un corpus de 3 000
documents produit des milliers de rubans d'épaisseur 1, illisibles et longs à
calculer.
"""

from __future__ import annotations

from .._stable import top_by_count

from collections import Counter
from typing import Dict, List, Set

import pandas as pd

#: Dimensions disponibles et la façon d'en extraire les valeurs par document.
FIELDS = ("authors", "keywords", "sources", "countries", "institutions")


def _values_by_doc(corpus, field: str) -> Dict[str, Set[str]]:
    """{eid -> valeurs} pour la dimension demandée."""
    if field == "authors":
        a = corpus.authors
        a = a[a["name"].notna() & (a["name"].map(str).str.strip() != "")]
        src = a[["eid", "name"]].rename(columns={"name": "v"})
    elif field == "keywords":
        k = corpus.keywords
        k = k[(k["kind"] == "author") & k["keyword"].notna()]
        src = k[["eid", "keyword"]].rename(columns={"keyword": "v"})
    elif field == "sources":
        d = corpus.documents
        d = d[d["source"].notna() & (d["source"].map(str).str.strip() != "")]
        src = d[["eid", "source"]].rename(columns={"source": "v"})
    elif field == "countries":
        f = corpus.affiliations
        f = f[f["country"].notna() & (f["country"].map(str).str.strip() != "")]
        src = f[["eid", "country"]].rename(columns={"country": "v"})
    elif field == "institutions":
        from .production import _org_frame
        f = _org_frame(corpus, "parent")
        src = f[["eid", "org"]].rename(columns={"org": "v"})
    else:
        raise ValueError("champ inconnu : %r (attendu : %s)"
                         % (field, ", ".join(FIELDS)))

    out: Dict[str, Set[str]] = {}
    if src.empty:
        return out
    for eid, v in zip(src["eid"], src["v"]):
        out.setdefault(eid, set()).add(str(v).strip())
    return out


def _top_values(by_doc: Dict[str, Set[str]], n: int) -> Set[str]:
    counts: Counter = Counter()
    for vals in by_doc.values():
        counts.update(vals)
    return {v for v, _ in top_by_count(counts, n)}


def _links(left: Dict[str, Set[str]], right: Dict[str, Set[str]],
           keep_left: Set[str], keep_right: Set[str],
           depth: int) -> List[Dict[str, object]]:
    """Liens entre deux colonnes, pondérés par documents partagés."""
    pairs: Counter = Counter()
    for eid, lvals in left.items():
        rvals = right.get(eid)
        if not rvals:
            continue
        for a in lvals & keep_left:
            for b in rvals & keep_right:
                pairs[(a, b)] += 1
    return [{"source": a, "target": b, "value": int(w), "depth": depth}
            for (a, b), w in pairs.items()]


def three_fields(corpus, left: str = "authors", middle: str = "keywords",
                 right: str = "sources", n: int = 10,
                 min_weight: int = 1) -> pd.DataFrame:
    """Liens d'un diagramme à trois champs.

    Colonnes : ``source``, ``target``, ``value``, ``depth``.

    ``depth`` est la position de la colonne d'origine (0 pour la gauche, 1 pour
    le milieu) : le rendu s'en sert pour aligner les nœuds, sans quoi une
    valeur présente dans deux colonnes serait dessinée au mauvais endroit.

    Les libellés des colonnes sont préfixés quand deux colonnes portent la même
    dimension, sinon un nœud apparaîtrait des deux côtés et le diagramme
    boucler ait sur lui-même.
    """
    empty = pd.DataFrame(columns=["source", "target", "value", "depth"])
    fields = [left, middle, right]
    for f in fields:
        if f not in FIELDS:
            raise ValueError("champ inconnu : %r (attendu : %s)"
                             % (f, ", ".join(FIELDS)))

    cols = [_values_by_doc(corpus, f) for f in fields]
    if any(not c for c in cols):
        return empty

    # Préfixe de désambiguïsation : seulement si la dimension se répète.
    prefixes = ["", "", ""]
    for i, f in enumerate(fields):
        if fields.count(f) > 1:
            prefixes[i] = "%d· " % (i + 1)

    tops = [_top_values(c, n) for c in cols]
    for i, pfx in enumerate(prefixes):
        if pfx:
            cols[i] = {eid: {pfx + v for v in vals} for eid, vals in cols[i].items()}
            tops[i] = {pfx + v for v in tops[i]}

    links = (_links(cols[0], cols[1], tops[0], tops[1], 0)
             + _links(cols[1], cols[2], tops[1], tops[2], 1))
    links = [l for l in links if l["value"] >= min_weight]
    if not links:
        return empty

    return (pd.DataFrame(links)
              .sort_values(["depth", "value", "source", "target"],
                           ascending=[True, False, True, True], kind="stable")
              .reset_index(drop=True))
