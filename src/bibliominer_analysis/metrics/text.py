"""Fouille de texte : les termes des **titres et résumés**.

Les mots-clés sont choisis par les auteurs ou ajoutés par la base. Ils
décrivent ce que les auteurs *déclarent* étudier. Les titres et surtout les
résumés disent ce qu'ils écrivent réellement — et les deux divergent souvent :
un sujet peut traverser tout un corpus sans jamais apparaître comme mot-clé.

On extrait des **n-grammes** (1, 2 ou 3 mots). Les bigrammes sont presque
toujours plus informatifs que les unigrammes en bibliométrie : « machine » et
« learning » séparés ne disent rien, « machine learning » dit tout.

Deux précautions qui décident de la qualité du résultat :

  - les **mots vides** sont retirés, y compris le vocabulaire de rédaction
    scientifique (« paper », « results », « proposed »), qui sinon occupe
    toutes les premières places sans rien apprendre ;
  - un n-gramme n'est retenu que s'il ne **commence ni ne finit** par un mot
    vide : « of the model » et « the model of » seraient sinon comptés comme
    des termes distincts du seul « model ».
"""

from __future__ import annotations

from .._stable import top_by_count

import re
import unicodedata
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

#: Mots vides de l'anglais, et vocabulaire de rédaction scientifique. Ce
#: second bloc est le plus important : sans lui, « results », « study » et
#: « paper » trustent le haut de tous les classements.
STOPWORDS = frozenset("""
a about above after again against all also am an and any are as at be because
been before being below between both but by can cannot could did do does doing
down during each few for from further had has have having he her here hers
herself him himself his how i if in into is it its itself just me more most my
myself no nor not now of off on once only or other others ought our ours
ourselves out over own same she should so some such than that the their theirs
them themselves then there these they this those through to too under until up
very was we were what when where which while who whom why will with would you
your yours yourself yourselves

paper article study studies research researches propose proposed proposes
present presented presents result results show shows shown using use used uses
approach approaches method methods methodology based obtain obtained new novel
different various several many much high low large small case cases work works
find found analysis analyze analyzed data dataset datasets model models
performance experiment experiments experimental evaluate evaluated evaluation
conclusion conclusions introduction discussion aim aims objective objectives
provide provides provided consider considered considering compare compared
comparison significant significantly respectively however therefore thus moreover
furthermore additionally finally overall given due within among across per via
one two three first second third best better good well may might must need needs
value values number numbers time times year years type types term terms level
levels order set sets part parts problem problems solution solutions
""".split())

_WORD = re.compile(r"[a-z][a-z0-9\-]+")

#: Champs de texte exploitables.
FIELDS = ("abstract", "title", "both")


def _normalise(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    s = unicodedata.normalize("NFKD", value)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.lower()


def _tokens(text: str) -> List[str]:
    return _WORD.findall(_normalise(text))


def _ngrams(tokens: Sequence[str], size: int) -> List[str]:
    """n-grammes dont ni le premier ni le dernier mot n'est un mot vide."""
    if size == 1:
        return [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    out = []
    for i in range(len(tokens) - size + 1):
        window = tokens[i:i + size]
        if window[0] in STOPWORDS or window[-1] in STOPWORDS:
            continue
        # Un n-gramme entièrement composé de mots vides internes n'apprend rien
        # non plus : on exige au moins deux mots pleins.
        if sum(1 for w in window if w not in STOPWORDS) < 2:
            continue
        if any(len(w) < 3 for w in window):
            continue
        out.append(" ".join(window))
    return out


#: Mention de copyright en fin de résumé Scopus. On coupe le texte à partir de
#: là — et on remonte sur un « Copyright » qui précéderait le symbole.
_COPYRIGHT = re.compile(r"(?:\bcopyright\s*)?[©ⓒ]", re.I)


def strip_copyright(text: Any) -> str:
    """Retire la mention d'éditeur qui clôt presque tous les résumés Scopus.

    Sans cela « springer nature », « elsevier » ou « rights reserved » montent
    dans les tout premiers termes du corpus, où ils ne décrivent évidemment
    aucun sujet.

    On coupe au symbole plutôt que de mettre « nature » ou « science » sur une
    liste noire : ce sont des mots parfaitement légitimes ailleurs, et les
    interdire ferait disparaître de vrais thèmes.
    """
    if not isinstance(text, str):
        return ""
    m = _COPYRIGHT.search(text)
    return text[:m.start()] if m else text


def _text_series(corpus, field: str) -> pd.Series:
    docs = corpus.documents
    title = docs["title"].fillna("")
    abstract = docs["abstract"].fillna("").map(strip_copyright)
    if field == "title":
        raw = title
    elif field == "abstract":
        raw = abstract
    else:
        raw = title + ". " + abstract
    return pd.Series(raw.to_numpy(), index=docs["eid"].to_numpy())


def _per_document(corpus, field: str, ngram: int,
                  extra_stopwords: Optional[Iterable[str]] = None) -> Dict[str, set]:
    """Ensemble des termes de chaque document. Un terme compte UNE fois par
    document, quel que soit le nombre de répétitions : sinon un résumé bavard
    pèserait autant que dix articles."""
    stop = set(STOPWORDS)
    if extra_stopwords:
        stop |= {str(w).strip().lower() for w in extra_stopwords if str(w).strip()}

    out: Dict[str, set] = {}
    for eid, text in _text_series(corpus, field).items():
        toks = _tokens(text)
        if stop is not STOPWORDS:
            # On refiltre avec la liste enrichie fournie par l'appelant.
            terms = [g for g in _ngrams(toks, ngram)
                     if not any(w in stop and w not in STOPWORDS for w in g.split())]
        else:
            terms = _ngrams(toks, ngram)
        if terms:
            out[str(eid)] = set(terms)
    return out


def top_terms(corpus, field: str = "abstract", ngram: int = 1,
              n: Optional[int] = 50, min_documents: int = 2,
              extra_stopwords: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Termes les plus fréquents des titres ou résumés.

    Colonnes : ``term``, ``documents``, ``share``.

    ``documents`` est un nombre de DOCUMENTS, pas d'occurrences : c'est la
    mesure qui résiste à un résumé qui répéterait vingt fois le même mot.
    """
    per_doc = _per_document(corpus, field, ngram, extra_stopwords)
    empty = pd.DataFrame(columns=["term", "documents", "share"])
    if not per_doc:
        return empty

    counts = Counter()
    for terms in per_doc.values():
        counts.update(terms)

    total = len(corpus.documents) or 1
    rows = [{"term": t, "documents": c, "share": round(100.0 * c / total, 1)}
            for t, c in counts.items() if c >= min_documents]
    if not rows:
        return empty

    out = pd.DataFrame(rows).sort_values(
        ["documents", "term"], ascending=[False, True], kind="stable").reset_index(drop=True)
    return out.head(n) if n else out


def text_co_occurrence(corpus, field: str = "abstract", ngram: int = 2,
                       top_n: int = 50, min_weight: int = 2,
                       min_documents: int = 2,
                       extra_stopwords: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """Réseau de co-occurrence des termes du texte.

    Même structure que les autres réseaux du paquet, donc directement
    compatible avec les centralités et les normalisations de
    `networks.analysis`.
    """
    per_doc = _per_document(corpus, field, ngram, extra_stopwords)
    if not per_doc:
        return {"nodes": [], "edges": [], "n_nodes": 0, "n_edges": 0}

    counts = Counter()
    for terms in per_doc.values():
        counts.update(terms)
    counts = Counter({t: c for t, c in counts.items() if c >= min_documents})
    if not counts:
        return {"nodes": [], "edges": [], "n_nodes": 0, "n_edges": 0}

    keep = {t for t, _ in top_by_count(counts, top_n)}
    pairs: Counter = Counter()
    for terms in per_doc.values():
        present = sorted(terms & keep)
        for i, a in enumerate(present):
            for b in present[i + 1:]:
                pairs[(a, b)] += 1

    from ..networks.build import _to_graph

    return _to_graph(Counter({t: counts[t] for t in keep}), pairs,
                     {t: t for t in keep}, min_weight)


def text_trend(corpus, field: str = "abstract", ngram: int = 2,
               n: int = 20, min_documents: int = 2,
               extra_stopwords: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Position dans le temps des termes du texte.

    Colonnes : ``term``, ``documents``, ``year_q1``, ``year_median``,
    ``year_q3``. Même lecture que `trend_topics`, mais sur le texte plutôt que
    sur les mots-clés — et c'est souvent là qu'on voit un sujet monter avant
    qu'il ne devienne un mot-clé déclaré.
    """
    import numpy as np

    per_doc = _per_document(corpus, field, ngram, extra_stopwords)
    empty = pd.DataFrame(columns=["term", "documents", "year_q1",
                                  "year_median", "year_q3"])
    if not per_doc:
        return empty

    years = corpus.documents[["eid", "year"]].copy()
    years["year"] = pd.to_numeric(years["year"], errors="coerce")
    year_of = dict(zip(years["eid"].map(str), years["year"]))

    rows = []
    for eid, terms in per_doc.items():
        y = year_of.get(eid)
        if pd.isna(y):
            continue
        for t in terms:
            rows.append((t, float(y)))
    if not rows:
        return empty

    df = pd.DataFrame(rows, columns=["term", "year"])
    g = (df.groupby("term")
           .agg(documents=("year", "size"),
                year_q1=("year", lambda s: float(np.percentile(s, 25))),
                year_median=("year", "median"),
                year_q3=("year", lambda s: float(np.percentile(s, 75))))
           .reset_index())
    g = g[g["documents"] >= min_documents]
    if g.empty:
        return empty

    for c in ("year_q1", "year_median", "year_q3"):
        g[c] = g[c].round().astype("Int64")
    return (g.sort_values(["year_median", "documents"], ascending=[True, False], kind="stable")
             .reset_index(drop=True).head(n))
