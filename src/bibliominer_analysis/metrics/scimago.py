"""Enriching journals with **SCImago** (SJR, quartile, and the rest).

The corpus says how many times an article was cited. It says nothing
about the **venue**: publishing in a Q1 journal and in a Q4 journal are
not comparable, and this information exists nowhere in a Scopus export.
SCImago provides it, with much more than the quartile alone.

Matching is done in two steps, from the safest to the least safe:

  1. **by ISSN**, an identifier, hence unambiguous. SCImago often lists
     several per journal (print and electronic): all are indexed.
  2. **by normalised title**, when the ISSN is missing or matches nothing.
     Less safe, so the method used is ALWAYS returned in the
     ``matched_by`` column: a reader must be able to discard weak matches
     themselves.

What is not found stays **empty**, never guessed. A journal absent from
SCImago (unindexed conference proceedings, a journal too recent) has no
quartile; writing "Q4" by default would be an invention.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

#: Locations searched in order. The first is the one shipped with the
#: package; the following ones let an application provide its own version
#: without reinstalling.
_SEARCH_PATHS = (
    Path(__file__).resolve().parent.parent / "data_ref" / "scimagojr_2025.csv",
)

#: Columns kept, with their output name. Not everything SCImago publishes is
#: useful here: what describes the journal is kept, not what describes the
#: file.
_COLUMNS = {
    "Sourceid": "scimago_id",
    "Title": "scimago_title",
    "Type": "source_type",
    "Publisher": "publisher",
    "Country": "publisher_country",
    "Region": "publisher_region",
    "Coverage": "coverage",
    "Categories": "categories",
    "Areas": "areas",
}

#: Numeric columns: SCImago writes decimals with a COMMA.
_NUMERIC = {
    "SJR": "sjr",
    "H index": "source_h_index",
    "Total Docs. (2025)": "docs_year",
    "Total Docs. (3years)": "docs_3y",
    "Total Refs.": "total_refs",
    "Total Citations (3years)": "citations_3y",
    "Citable Docs. (3years)": "citable_docs_3y",
    "Citations / Doc. (2years)": "citations_per_doc_2y",
    "Ref. / Doc.": "refs_per_doc",
    "%Female": "female_share",
    "Overton": "overton",
}

QUARTILES = ("Q1", "Q2", "Q3", "Q4")

_cache: Optional[pd.DataFrame] = None
_index_cache: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------

def default_path() -> Optional[Path]:
    for p in _SEARCH_PATHS:
        if p.exists():
            return p
    return None


def normalize_issn(raw: Any) -> str:
    """ISSN reduced to its eight characters, same rules as the cleaning."""
    if raw is None:
        return ""
    s = re.sub(r"[^0-9Xx]", "", str(raw)).upper()
    return s if len(s) == 8 else ""


def _split_issn_field(cell: Any) -> List[str]:
    """SCImago lists several ISSNs per journal, separated by commas."""
    if not isinstance(cell, str):
        return []
    out = []
    for part in cell.replace('"', "").split(","):
        n = normalize_issn(part)
        if n:
            out.append(n)
    return out


def normalize_title(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    s = unicodedata.normalize("NFKD", value)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9 ]+", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def _to_number(series: pd.Series) -> pd.Series:
    """Decimal commas, no thousands separator."""
    return pd.to_numeric(
        series.map(str).str.replace(".", "", regex=False)
                          .str.replace(",", ".", regex=False)
                          .str.strip(),
        errors="coerce")


def load_scimago(path: Optional[Any] = None, force: bool = False) -> pd.DataFrame:
    """Loads the SCImago reference table. The result is cached.

    Returns an empty DataFrame if the file is missing: the absence of the
    reference table must degrade the analysis, never stop it.
    """
    global _cache, _index_cache
    if _cache is not None and not force and path is None:
        return _cache

    target = Path(path) if path else default_path()
    if target is None or not target.exists():
        empty = pd.DataFrame(columns=list(_COLUMNS.values()) + list(_NUMERIC.values())
                             + ["quartile", "issns"])
        if path is None:
            _cache, _index_cache = empty, None
        return empty

    df = pd.read_csv(target, sep=";", dtype=str, encoding="utf-8",
                     on_bad_lines="skip")
    # "Publisher" appears TWICE in the file; pandas suffixes the second one.
    # The first is kept and the duplicate ignored.
    df = df.loc[:, ~df.columns.duplicated()]

    out = pd.DataFrame(index=df.index)
    for src, dst in _COLUMNS.items():
        out[dst] = df[src].map(str).str.strip().str.strip('"') if src in df.columns else None
    for src, dst in _NUMERIC.items():
        out[dst] = _to_number(df[src]) if src in df.columns else np.nan

    quartile = (df["SJR Best Quartile"].map(str).str.strip().str.upper()
                if "SJR Best Quartile" in df.columns else pd.Series("", index=df.index))
    out["quartile"] = quartile.where(quartile.isin(QUARTILES), None)
    out["issns"] = df["Issn"].map(_split_issn_field) if "Issn" in df.columns else [[]] * len(df)
    out["scimago_rank"] = _to_number(df["Rank"]) if "Rank" in df.columns else np.nan

    if "Open Access" in df.columns:
        out["open_access"] = df["Open Access"].map(str).str.strip().str.lower().eq("yes")
    if "Open Access Diamond" in df.columns:
        out["open_access_diamond"] = (df["Open Access Diamond"].map(str)
                                      .str.strip().str.lower().eq("yes"))

    if path is None:
        _cache, _index_cache = out, None
    return out


def _index(path: Optional[Any] = None) -> Dict[str, Any]:
    """Index ISSN -> row and normalised title -> row."""
    global _index_cache
    if _index_cache is not None and path is None:
        return _index_cache

    table = load_scimago(path)
    by_issn: Dict[str, int] = {}
    by_title: Dict[str, int] = {}
    if not table.empty:
        for pos, (issns, title) in enumerate(zip(table["issns"], table["scimago_title"])):
            for issn in issns or []:
                by_issn.setdefault(issn, pos)
            t = normalize_title(title)
            # A short title is too ambiguous to serve as a key.
            if len(t) >= 8:
                by_title.setdefault(t, pos)

    idx = {"table": table, "by_issn": by_issn, "by_title": by_title}
    if path is None:
        _index_cache = idx
    return idx


# ---------------------------------------------------------------------------
# Appariement
# ---------------------------------------------------------------------------

_OUT_COLS = ["source", "documents", "citations", "matched_by", "quartile", "sjr",
             "scimago_rank", "source_h_index", "citations_per_doc_2y",
             "source_type", "publisher", "publisher_country", "publisher_region",
             "open_access", "coverage", "categories", "areas", "docs_3y",
             "citations_3y", "refs_per_doc", "female_share", "scimago_title"]


def enrich_sources(corpus, path: Optional[Any] = None) -> pd.DataFrame:
    """Every journal of the corpus, enriched with what SCImago knows about it.

    Columns: those of the corpus (``source``, ``documents``, ``citations``),
    then ``matched_by`` ("issn", "title" or empty) and the SCImago measures.

    A journal that is not found keeps its SCImago columns **empty**. That is
    the normal case for conference proceedings, which SCImago only partly
    indexes, and saying so is better than hiding it.
    """
    docs = corpus.documents
    if docs.empty or "source" not in docs.columns:
        return pd.DataFrame(columns=_OUT_COLS)

    d = docs[["eid", "source"]].copy()
    d["source"] = d["source"].map(str).str.strip()
    d = d[(d["source"] != "") & (d["source"].str.lower() != "nan")]
    if d.empty:
        return pd.DataFrame(columns=_OUT_COLS)

    d["citations"] = pd.to_numeric(docs["cited_by"], errors="coerce").fillna(0).astype(int)
    d["issn"] = docs["issn"].map(normalize_issn) if "issn" in docs.columns else ""

    grouped = (d.groupby("source")
                 .agg(documents=("eid", "nunique"),
                      citations=("citations", "sum"),
                      issn=("issn", lambda s: next((x for x in s if x), "")))
                 .reset_index())

    idx = _index(path)
    table, by_issn, by_title = idx["table"], idx["by_issn"], idx["by_title"]

    positions: List[Optional[int]] = []
    methods: List[str] = []
    for issn, title in zip(grouped["issn"], grouped["source"]):
        pos = by_issn.get(issn) if issn else None
        method = "issn" if pos is not None else ""
        if pos is None:
            t = normalize_title(title)
            if len(t) >= 8:
                pos = by_title.get(t)
                method = "title" if pos is not None else ""
        positions.append(pos)
        methods.append(method)

    grouped["matched_by"] = methods
    extra = [c for c in _OUT_COLS if c not in ("source", "documents", "citations",
                                               "matched_by")]
    for col in extra:
        if col in table.columns:
            grouped[col] = [table.iloc[p][col] if p is not None else None
                            for p in positions]
        else:
            grouped[col] = None

    grouped = grouped.sort_values(["documents", "citations"],
                                  ascending=False, kind="stable").reset_index(drop=True)
    return grouped[_OUT_COLS]


def quartile_distribution(corpus, path: Optional[Any] = None) -> pd.DataFrame:
    """Documents distributed by the SCImago quartile of their journal.

    Columns: ``quartile``, ``sources``, ``documents``, ``share``,
    ``citations``, ``citations_per_document``.

    The five rows are always present, ``Not indexed`` included. It is the most
    important row of the table: it says what share of the corpus escapes the
    ranking, and therefore how representative the other four are.
    """
    cols = ["quartile", "sources", "documents", "share", "citations",
            "citations_per_document"]
    enriched = enrich_sources(corpus, path)
    order = list(QUARTILES) + ["Not indexed"]
    if enriched.empty:
        return pd.DataFrame([{"quartile": q, "sources": 0, "documents": 0,
                              "share": 0.0, "citations": 0,
                              "citations_per_document": 0.0} for q in order],
                            columns=cols)

    e = enriched.copy()
    e["bucket"] = e["quartile"].where(e["quartile"].isin(QUARTILES), "Not indexed")
    total = int(e["documents"].sum())

    rows = []
    for q in order:
        sub = e[e["bucket"] == q]
        docs = int(sub["documents"].sum())
        cites = int(sub["citations"].sum())
        rows.append({
            "quartile": q,
            "sources": int(len(sub)),
            "documents": docs,
            "share": round(100.0 * docs / total, 1) if total else 0.0,
            "citations": cites,
            "citations_per_document": round(cites / docs, 2) if docs else 0.0,
        })
    return pd.DataFrame(rows, columns=cols)


def quartile_over_time(corpus, path: Optional[Any] = None) -> pd.DataFrame:
    """Yearly evolution of the publication profile by quartile.

    Columns: ``year``, ``quartile``, ``documents``, ``share``.

    It is the reading that shows **moving up the ranks**: a laboratory going
    from Q3 to Q1 in five years, or the reverse. A total per quartile does not
    say it.

    **Caveat to state with the figure**: the quartile comes from ONE edition
    of SCImago (the one of the loaded file) and applies to all years. A
    journal that is Q1 today may have been Q2 at the time of publication. The
    curve therefore shows in which journals, ranked by their CURRENT rank, the
    corpus published each year, not the quartile they had at the time.
    """
    cols = ["year", "quartile", "documents", "share"]
    enriched = enrich_sources(corpus, path)
    if enriched.empty:
        return pd.DataFrame(columns=cols)

    bucket = dict(zip(enriched["source"],
                      enriched["quartile"].where(
                          enriched["quartile"].isin(QUARTILES), "Not indexed")))

    d = corpus.documents[["eid", "source"]].copy()
    d["year"] = pd.to_numeric(corpus.documents["year"], errors="coerce")
    d = d.dropna(subset=["year"])
    d["quartile"] = d["source"].map(str).str.strip().map(bucket).fillna("Not indexed")
    if d.empty:
        return pd.DataFrame(columns=cols)

    counts = (d.groupby(["year", "quartile"])["eid"].nunique()
                .rename("documents").reset_index())
    order = list(QUARTILES) + ["Not indexed"]
    years = sorted(counts["year"].unique())
    grid = pd.MultiIndex.from_product([years, order], names=["year", "quartile"])
    counts = (counts.set_index(["year", "quartile"]).reindex(grid, fill_value=0)
                    .reset_index())
    totals = counts.groupby("year")["documents"].transform("sum")
    counts["share"] = np.where(totals > 0,
                               (100.0 * counts["documents"] / totals).round(1), 0.0)
    counts["year"] = counts["year"].astype(int)
    return counts[cols].reset_index(drop=True)


def scimago_coverage(corpus, path: Optional[Any] = None) -> Dict[str, Any]:
    """What the matching managed to do, and what it missed.

    An enriched table without this count reads as if it covered the whole
    corpus. It never covers it entirely.
    """
    enriched = enrich_sources(corpus, path)
    available = default_path() is not None or not load_scimago(path).empty
    if enriched.empty:
        return {"available": available, "sources": 0, "matched_sources": 0,
                "matched_by_issn": 0, "matched_by_title": 0,
                "documents": 0, "matched_documents": 0, "document_share": 0.0}

    total_docs = int(enriched["documents"].sum())
    matched = enriched[enriched["matched_by"] != ""]
    matched_docs = int(matched["documents"].sum())
    return {
        "available": available,
        "sources": int(len(enriched)),
        "matched_sources": int(len(matched)),
        "matched_by_issn": int((enriched["matched_by"] == "issn").sum()),
        "matched_by_title": int((enriched["matched_by"] == "title").sum()),
        "documents": total_docs,
        "matched_documents": matched_docs,
        "document_share": round(100.0 * matched_docs / total_docs, 1) if total_docs else 0.0,
    }


# ---------------------------------------------------------------------------
# Interdisciplinarity, only possible thanks to the reference table
# ---------------------------------------------------------------------------

def _split_areas(cell: Any) -> List[str]:
    """"Business, Management and Accounting; Computer Science" -> two areas.

    The semicolon separates the areas; the comma belongs to the area's NAME.
    Splitting on the comma would break "Biochemistry, Genetics and Molecular
    Biology" into three false areas.
    """
    if not isinstance(cell, str) or not cell.strip():
        return []
    return [p.strip() for p in cell.split(";") if p.strip()]


def _strip_quartile(label: str) -> str:
    """"Oncology (Q1)" -> "Oncology": the category, without its rank."""
    return re.sub(r"\s*\(Q[1-4]\)\s*$", "", label).strip()


def subject_areas(corpus, path: Optional[Any] = None) -> pd.DataFrame:
    """Subject areas of the corpus, according to the journals.

    Columns: ``area``, ``sources``, ``documents``, ``share``, ``citations``.

    A document whose journal covers three areas counts for each: the sum
    therefore deliberately exceeds the number of documents. It is this
    multiple membership that MAKES interdisciplinarity.
    """
    cols = ["area", "sources", "documents", "share", "citations"]
    enriched = enrich_sources(corpus, path)
    if enriched.empty:
        return pd.DataFrame(columns=cols)

    rows = []
    for r in enriched.itertuples():
        for area in _split_areas(getattr(r, "areas", None)):
            rows.append({"area": area, "source": r.source,
                         "documents": r.documents, "citations": r.citations})
    if not rows:
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(rows)
    total = int(enriched["documents"].sum())
    out = (df.groupby("area")
             .agg(sources=("source", "nunique"),
                  documents=("documents", "sum"),
                  citations=("citations", "sum"))
             .reset_index())
    out["share"] = (100.0 * out["documents"] / total).round(1) if total else 0.0
    return out.sort_values(["documents", "citations"],
                           ascending=False, kind="stable").reset_index(drop=True)[cols]


def subject_categories(corpus, n: Optional[int] = 25,
                       path: Optional[Any] = None) -> pd.DataFrame:
    """Fine-grained categories, more precise than the areas, with their
    quartile.

    Columns: ``category``, ``sources``, ``documents``, ``best_quartile``.
    """
    cols = ["category", "sources", "documents", "best_quartile"]
    enriched = enrich_sources(corpus, path)
    if enriched.empty:
        return pd.DataFrame(columns=cols)

    rows = []
    for r in enriched.itertuples():
        for raw in _split_areas(getattr(r, "categories", None)):
            m = re.search(r"\((Q[1-4])\)\s*$", raw)
            rows.append({"category": _strip_quartile(raw), "source": r.source,
                         "documents": r.documents,
                         "quartile": m.group(1) if m else None})
    if not rows:
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(rows)
    out = (df.groupby("category")
             .agg(sources=("source", "nunique"),
                  documents=("documents", "sum"),
                  # The BEST quartile reached: "Q1" sorts before "Q4", hence the min on the
                  # string.
                  best_quartile=("quartile",
                                 lambda x: x.dropna().min() if x.notna().any() else None))
             .reset_index())
    out = out.sort_values(["documents", "sources"],
                          ascending=False, kind="stable").reset_index(drop=True)
    return (out[cols].head(n) if n else out[cols])


def interdisciplinarity(corpus, path: Optional[Any] = None) -> Dict[str, Any]:
    """Disciplinary diversity of the corpus.

    Returns: ``areas``, ``shannon``, ``simpson``, ``evenness``, ``top_area``,
    ``top_area_share``, ``coverage``.

    - **Simpson**: probability that two documents drawn at random belong to
      different areas. Directly interpretable.
    - **Shannon**: entropy of the distribution.
    - **Evenness**: Shannon relative to its maximum. It is the measure to
      compare between corpora, because it does not depend on the NUMBER of
      areas; raw entropy, on the other hand, rises mechanically with it.
    """
    areas = subject_areas(corpus, path)
    base = {"areas": 0, "shannon": None, "simpson": None, "evenness": None,
            "top_area": None, "top_area_share": None, "coverage": 0.0}
    if areas.empty:
        return base

    counts = areas["documents"].to_numpy(dtype=float)
    total = counts.sum()
    if total <= 0:
        return base

    p = counts / total
    shannon = float(-(p * np.log(p)).sum())
    k = int(p.size)
    cov = scimago_coverage(corpus, path)
    return {
        "areas": k,
        "shannon": round(shannon, 3),
        "simpson": round(float(1.0 - (p ** 2).sum()), 3),
        "evenness": round(shannon / np.log(k), 3) if k > 1 else 0.0,
        "top_area": str(areas.iloc[0]["area"]),
        "top_area_share": float(areas.iloc[0]["share"]),
        # Diversity only reads on the matched share: without this figure one would
        # believe it describes the whole corpus.
        "coverage": cov["document_share"],
    }


def journal_open_access(corpus, path: Optional[Any] = None) -> pd.DataFrame:
    """Documents by the access of their JOURNAL (a property of the venue).

    To be distinguished from `access.access_status`, which is about the
    ARTICLE. An open article in a subscription journal exists: that is hybrid.

    "Unknown" is not "paid": a journal absent from the reference table has no
    known status.
    """
    cols = ["access", "sources", "documents", "share", "citations",
            "citations_per_document"]
    enriched = enrich_sources(corpus, path)
    order = ["Open access", "Subscription", "Unknown"]
    if enriched.empty:
        return pd.DataFrame([{"access": a, "sources": 0, "documents": 0,
                              "share": 0.0, "citations": 0,
                              "citations_per_document": 0.0} for a in order],
                            columns=cols)

    def bucket(v) -> str:
        # `is True` fails on a numpy boolean: np.bool_(True) IS not the True
        # singleton. All journals therefore fell into "Unknown".
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "Unknown"
        try:
            return "Open access" if bool(v) else "Subscription"
        except (TypeError, ValueError):
            return "Unknown"

    e = enriched.copy()
    e["access"] = e["open_access"].map(bucket)
    total = int(e["documents"].sum())

    rows = []
    for a in order:
        sub = e[e["access"] == a]
        docs = int(sub["documents"].sum())
        cites = int(sub["citations"].sum())
        rows.append({
            "access": a,
            "sources": int(len(sub)),
            "documents": docs,
            "share": round(100.0 * docs / total, 1) if total else 0.0,
            "citations": cites,
            "citations_per_document": round(cites / docs, 2) if docs else 0.0,
        })
    return pd.DataFrame(rows, columns=cols)
