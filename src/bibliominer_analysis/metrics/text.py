"""Text mining: the terms of **titles and abstracts**.

Keywords are chosen by the authors or added by the database. They
describe what the authors *declare* they study. Titles, and above all
abstracts, say what they actually write, and the two often diverge: a
subject can run through a whole corpus without ever appearing as a
keyword.

**n-grams** (1, 2 or 3 words) are extracted. Bigrams are almost always
more informative than unigrams in bibliometrics: "machine" and "learning"
apart say nothing, "machine learning" says everything.

Two precautions decide the quality of the result:

  - **stop words** are removed, including the vocabulary of scientific
    writing ("paper", "results", "proposed"), which otherwise takes all
    the top places without teaching anything;
  - an n-gram is kept only if it **neither starts nor ends** with a stop
    word: "of the model" and "the model of" would otherwise be counted as
    terms distinct from "model" alone.
"""

from __future__ import annotations

from .._stable import top_by_count

import re
import unicodedata
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

#: English stop words, and the vocabulary of scientific writing. The second
#: block is the most important: without it, "results", "study" and "paper"
#: monopolise the top of every ranking.
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

#: Usable text fields.
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
    """n-grams whose first and last words are not stop words."""
    if size == 1:
        return [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    out = []
    for i in range(len(tokens) - size + 1):
        window = tokens[i:i + size]
        if window[0] in STOPWORDS or window[-1] in STOPWORDS:
            continue
        # An n-gram made entirely of internal stop words teaches nothing either: at
        # least two content words are required.
        if sum(1 for w in window if w not in STOPWORDS) < 2:
            continue
        if any(len(w) < 3 for w in window):
            continue
        out.append(" ".join(window))
    return out


#: Copyright notice at the end of a Scopus abstract. The text is cut from
#: there, going back to a "Copyright" that would precede the symbol.
_COPYRIGHT = re.compile(r"(?:\bcopyright\s*)?[©ⓒ]", re.I)


def strip_copyright(text: Any) -> str:
    """Removes the publisher notice that ends almost every Scopus abstract.

    Without it, "springer nature", "elsevier" or "rights reserved" climb into
    the very first terms of the corpus, where they obviously describe no
    subject.

    The text is cut at the symbol rather than putting "nature" or "science"
    on a blacklist: these are perfectly legitimate words elsewhere, and
    banning them would make real themes disappear.
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
    """Set of the terms of each document. A term counts ONCE per document,
    however many times it is repeated: otherwise a wordy abstract would weigh
    as much as ten articles."""
    stop = set(STOPWORDS)
    if extra_stopwords:
        stop |= {str(w).strip().lower() for w in extra_stopwords if str(w).strip()}

    out: Dict[str, set] = {}
    for eid, text in _text_series(corpus, field).items():
        toks = _tokens(text)
        if stop is not STOPWORDS:
            # Filtered again with the extended list supplied by the caller.
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
    """Most frequent terms of titles or abstracts.

    Columns: ``term``, ``documents``, ``share``.

    ``documents`` is a number of DOCUMENTS, not of occurrences: it is the
    measure that resists an abstract repeating the same word twenty times.
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
    """Co-occurrence network of the text terms.

    Same structure as the package's other networks, therefore directly
    compatible with the centralities and normalisations of
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
    """Position in time of the text terms.

    Columns: ``term``, ``documents``, ``year_q1``, ``year_median``,
    ``year_q3``. Same reading as `trend_topics`, but on the text rather than
    the keywords, and that is often where a subject is seen rising before it
    becomes a declared keyword.
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
