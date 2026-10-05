"""Production and actor indicators.

Every function takes a `Corpus` and returns a DataFrame ready to be
displayed or exported, never a figure. Columns have stable names: the
interface relies on them.

One rule holds everywhere: the same document is NEVER counted twice. An
institution cited by three authors of the same article counts for one
document, not three; otherwise the rankings are wrong.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


def _citations(corpus) -> pd.Series:
    """eid -> number of citations, as an integer, never NaN."""
    d = corpus.documents
    return pd.to_numeric(d["cited_by"], errors="coerce").fillna(0).astype(int)


def by_year(corpus) -> pd.DataFrame:
    """Documents and citations per year, with no gap in the series.

    Missing years are added at zero: a production curve with absent years
    reads wrongly.
    """
    d = corpus.documents.copy()
    d["year"] = pd.to_numeric(d["year"], errors="coerce")
    d["citations"] = _citations(corpus)
    d = d.dropna(subset=["year"])
    if d.empty:
        return pd.DataFrame(columns=["year", "documents", "citations",
                                     "citations_per_doc", "cumulative"])

    g = (d.groupby(d["year"].astype(int))
           .agg(documents=("eid", "nunique"), citations=("citations", "sum"))
           .reset_index())

    full = pd.DataFrame({"year": range(int(g["year"].min()),
                                       int(g["year"].max()) + 1)})
    g = full.merge(g, on="year", how="left").fillna({"documents": 0, "citations": 0})
    g["documents"] = g["documents"].astype(int)
    g["citations"] = g["citations"].astype(int)
    # A year WITHOUT documents (added at zero above) has no mean: empty, not
    # zero. `where` keeps a NUMERIC column; `replace(0, pd.NA)` made it
    # "object", and `round` failed (an error 500 on Production as soon as a
    # year was missing from the series).
    per_doc = g["citations"] / g["documents"].where(g["documents"] > 0)
    g["citations_per_doc"] = per_doc.astype(float).round(2)
    g["cumulative"] = g["documents"].cumsum()
    return g


def by_country(corpus, n: Optional[int] = None) -> pd.DataFrame:
    """Documents per country of affiliation.

    A document with two Moroccan authors counts ONCE for Morocco. A
    Moroccan-Spanish document counts once for each country: the sum therefore
    deliberately exceeds the number of documents.
    """
    a = corpus.affiliations
    a = a[a["country"].notna() & (a["country"].map(str).str.strip() != "")]
    if a.empty:
        return pd.DataFrame(columns=["country", "documents", "citations"])

    pairs = a[["eid", "country"]].drop_duplicates()
    cites = corpus.documents[["eid"]].copy()
    cites["citations"] = _citations(corpus)
    pairs = pairs.merge(cites, on="eid", how="left")

    g = (pairs.groupby("country")
              .agg(documents=("eid", "nunique"), citations=("citations", "sum"))
              .reset_index()
              .sort_values(["documents", "citations"], ascending=False, kind="stable")
              .reset_index(drop=True))
    return g.head(n) if n else g


#: Column of the `affiliations` table for the requested level of analysis.
#:   "parent"    -> the parent organisation (university, company, hospital)
#:   "subparent" -> the internal unit (laboratory, school, department)
#: The two levels answer different questions: the first locates the
#: institution, the second identifies the team that actually produces.
_ORG_LEVELS = {"parent": "parent1", "subparent": "subparent"}


def org_column(level: str = "parent") -> str:
    """Column to analyse for the requested level."""
    try:
        return _ORG_LEVELS[level]
    except KeyError:
        raise ValueError("level must be 'parent' or 'subparent', not %r"
                         % level) from None


#: The two parent organisations an affiliation can name.
PARENT_COLUMNS = ("parent1", "parent2")


def institution_rows(affiliations: pd.DataFrame) -> pd.DataFrame:
    """One row per (affiliation, parent organisation), column ``institution``.

    An affiliation with a DOUBLE attachment ("parent 1: University A,
    parent 2: CNRS") belongs to both organisations: it counts for each, with
    full counting, as a co-signed document counts for each country. Reading
    ``parent1`` alone made the second organisation disappear from every
    ranking. A ``parent 2`` identical to ``parent 1`` counts only once.
    """
    frames = []
    for col in PARENT_COLUMNS:
        if col not in affiliations.columns:
            continue
        names = affiliations[col].astype("string").str.strip()
        kept = affiliations[names.notna() & (names != "")].copy()
        kept["institution"] = names[kept.index].map(str)
        frames.append(kept)
    if not frames:
        return affiliations.iloc[0:0].assign(institution=pd.Series(dtype=object))
    out = pd.concat(frames, ignore_index=True)
    key = [c for c in ("eid", "aff_pos") if c in out.columns] + ["institution"]
    return out.drop_duplicates(subset=key).reset_index(drop=True)


def _org_frame(corpus, level: str = "parent") -> pd.DataFrame:
    """Usable affiliations for this level, column renamed `org`.

    At "parent" level, an affiliation with a double attachment gives one row
    per organisation (see `institution_rows`).

    Researchers without an affiliation are EXCLUDED: the cleaning marks them
    "Independent researcher"; it is not an organisation and its presence at
    the top of a ranking would make no sense.
    """
    from ..io.schema import INDEPENDENT_LABEL

    col = org_column(level)
    if level == "parent":
        a = institution_rows(corpus.affiliations)
        a = a[a["institution"] != INDEPENDENT_LABEL]
        if a.empty:
            return pd.DataFrame(columns=["eid", "org", "country", "parent1"])
        return pd.DataFrame({
            "eid": a["eid"].to_numpy(),
            "org": a["institution"].to_numpy(),
            "country": a["country"].to_numpy(),
            "parent1": a["institution"].to_numpy(),
        })

    a = corpus.affiliations
    a = a[a[col].notna() & (a[col].map(str).str.strip() != "")]
    a = a[a[col] != INDEPENDENT_LABEL]
    if a.empty:
        return pd.DataFrame(columns=["eid", "org", "country", "parent1"])

    # Built column by column: at "parent" level, `col` IS "parent1", and a list
    # selection would make it appear twice; pandas then refuses to group on a
    # duplicated column.
    out = pd.DataFrame({
        "eid": a["eid"].to_numpy(),
        "org": a[col].to_numpy(),
        "country": a["country"].to_numpy(),
        # `parent1` is used to attach a unit to its organisation; at parent level
        # the two columns coincide, which is correct.
        "parent1": a["parent1"].to_numpy(),
    })
    return out


def top_institutions(corpus, n: int = 20, level: str = "parent") -> pd.DataFrame:
    """Ranking of the organisations, at the requested level."""
    a = _org_frame(corpus, level)
    if a.empty:
        return pd.DataFrame(columns=["institution", "documents", "citations",
                                     "country"])

    pairs = a[["eid", "org", "country"]].drop_duplicates(subset=["eid", "org"])
    cites = corpus.documents[["eid"]].copy()
    cites["citations"] = _citations(corpus)
    pairs = pairs.merge(cites, on="eid", how="left")

    g = (pairs.groupby("org")
              .agg(documents=("eid", "nunique"),
                   citations=("citations", "sum"),
                   country=("country", lambda s: s.dropna().mode().iat[0]
                            if not s.dropna().empty else None))
              .reset_index()
              .rename(columns={"org": "institution"})
              .sort_values(["documents", "citations"], ascending=False, kind="stable")
              .reset_index(drop=True))
    return g.head(n)


def institutions_over_time(corpus, n: int = 10, cumulative: bool = True,
                           level: str = "parent") -> pd.DataFrame:
    """Yearly production of the top `n` organisations, at the requested level.

    Long format: ``institution``, ``year``, ``documents``, ``cumulative``.
    Every year of the corpus appears for EACH organisation, including at
    zero; otherwise the curves would be interrupted where it published
    nothing, which reads as an absence of data rather than an absence of
    production.
    """
    aff = _org_frame(corpus, level)
    empty = pd.DataFrame(columns=["institution", "year", "documents", "cumulative"])
    if aff.empty:
        return empty

    years = corpus.documents[["eid", "year"]].copy()
    years["year"] = pd.to_numeric(years["year"], errors="coerce")
    pairs = (aff[["eid", "org"]].drop_duplicates()
                .merge(years, on="eid", how="left")
                .dropna(subset=["year"]))
    if pairs.empty:
        return empty
    pairs["year"] = pairs["year"].astype(int)

    top = (pairs.groupby("org")["eid"].nunique()
                .sort_values(ascending=False, kind="stable").head(n).index)
    pairs = pairs[pairs["org"].isin(top)]

    counts = (pairs.groupby(["org", "year"])["eid"].nunique()
                   .reset_index(name="documents"))

    all_years = range(int(pairs["year"].min()), int(pairs["year"].max()) + 1)
    grid = pd.MultiIndex.from_product([list(top), list(all_years)],
                                      names=["org", "year"]).to_frame(index=False)
    out = grid.merge(counts, on=["org", "year"], how="left").fillna({"documents": 0})
    out["documents"] = out["documents"].astype(int)
    out["cumulative"] = out.groupby("org")["documents"].cumsum()
    if not cumulative:
        out = out.drop(columns=["cumulative"])
    return out.rename(columns={"org": "institution"}).reset_index(drop=True)


def institutions_by_country(corpus, n: int = 20,
                            level: str = "parent") -> pd.DataFrame:
    """Organisations grouped by country: how many, and which ones dominate."""
    aff = _org_frame(corpus, level)
    aff = aff[aff["country"].notna()] if not aff.empty else aff
    empty = pd.DataFrame(columns=["country", "institutions", "documents",
                                  "top_institution"])
    if aff.empty:
        return empty

    pairs = aff[["eid", "org", "country"]].drop_duplicates()
    per_inst = pairs.groupby(["country", "org"])["eid"].nunique()

    rows = []
    for country, g in per_inst.groupby(level=0):
        g = g.droplevel(0).sort_values(ascending=False, kind="stable")
        rows.append({
            "country": country,
            "institutions": int(len(g)),
            "documents": int(pairs.loc[pairs["country"] == country, "eid"].nunique()),
            "top_institution": g.index[0],
        })
    return (pd.DataFrame(rows)
              .sort_values(["documents", "institutions"], ascending=False, kind="stable")
              .reset_index(drop=True).head(n))


def org_hierarchy(corpus, n: int = 20, min_documents: int = 1) -> pd.DataFrame:
    """ORGANISATION -> UNITS hierarchy, as the cleaning established it.

    It is the analytical counterpart of the cleaning's "organisations" view:
    under each parent organisation, the internal units attached to it, with
    their volume and their citations.

    Columns: ``parent``, ``subparent``, ``documents``, ``citations``,
    ``parent_documents``, ``share``.

    ``share`` is the unit's share WITHIN its organisation: it says whether a
    laboratory carries most of its university's production or is only one
    component among others.

    A unit with a double attachment appears under BOTH its organisations.
    """
    from ..io.schema import INDEPENDENT_LABEL

    base = institution_rows(corpus.affiliations)
    base = base[base["institution"] != INDEPENDENT_LABEL]
    # What follows reads "parent1": here it is the organisation of THIS row,
    # first or second attachment.
    base = base.assign(parent1=base["institution"])

    # `a` keeps only the rows CARRYING a unit, but the organisation's total is
    # computed on `base`: a university with a document that mentions no unit
    # still has one more, and ignoring it would artificially inflate the units'
    # share.
    a = base[base["subparent"].notna()
             & (base["subparent"].map(str).str.strip() != "")]
    empty = pd.DataFrame(columns=["parent", "subparent", "documents",
                                  "citations", "parent_documents", "share"])
    if a.empty:
        return empty

    cites = corpus.documents[["eid"]].copy()
    cites["citations"] = _citations(corpus)

    pairs = a[["eid", "parent1", "subparent"]].drop_duplicates()
    pairs = pairs.merge(cites, on="eid", how="left")

    g = (pairs.groupby(["parent1", "subparent"])
              .agg(documents=("eid", "nunique"), citations=("citations", "sum"))
              .reset_index()
              .rename(columns={"parent1": "parent"}))

    # Total of the organisation: counted on DISTINCT documents, otherwise a
    # document naming two units of the same organisation would count it twice.
    parent_totals = (base[["eid", "parent1"]].drop_duplicates()
                      .groupby("parent1")["eid"].nunique()
                      .rename("parent_documents"))
    g = g.merge(parent_totals, left_on="parent", right_index=True, how="left")
    g["share"] = (100 * g["documents"] / g["parent_documents"]).round(1)

    g = g[g["documents"] >= min_documents]
    return (g.sort_values(["parent_documents", "parent", "documents"],
                          ascending=[False, True, False], kind="stable")
             .reset_index(drop=True).head(n))


def top_authors(corpus, n: int = 20) -> pd.DataFrame:
    """Ranking of the authors, with the number of times in FIRST position.

    `first_author` only exists because the cleaning indexes the authors:
    without this information, the lead of a work cannot be told from a
    co-author.
    """
    a = corpus.authors
    a = a[a["name"].notna() & (a["name"].map(str).str.strip() != "")]
    if a.empty:
        return pd.DataFrame(columns=["author", "scopus_id", "documents",
                                     "citations", "first_author"])

    cites = corpus.documents[["eid"]].copy()
    cites["citations"] = _citations(corpus)
    a = a.merge(cites, on="eid", how="left")
    a["is_first"] = pd.to_numeric(a["position"], errors="coerce").eq(1)

    # Grouping key: the Scopus identifier if it exists (reliable), otherwise the
    # name; two namesakes without an identifier stay indistinguishable.
    a["key"] = a["scopus_id"].fillna("name:" + a["name"].map(str))

    g = (a.groupby("key")
           .agg(author=("name", lambda s: s.mode().iat[0] if not s.empty else None),
                scopus_id=("scopus_id", "first"),
                documents=("eid", "nunique"),
                citations=("citations", "sum"),
                first_author=("is_first", "sum"))
           .reset_index(drop=True)
           .sort_values(["documents", "citations"], ascending=False, kind="stable")
           .reset_index(drop=True))
    g["first_author"] = g["first_author"].astype(int)
    return g.head(n)


def top_sources(corpus, n: int = 20) -> pd.DataFrame:
    d = corpus.documents.copy()
    d["citations"] = _citations(corpus)
    d = d[d["source"].notna() & (d["source"].map(str).str.strip() != "")]
    if d.empty:
        return pd.DataFrame(columns=["source", "documents", "citations"])
    return (d.groupby("source")
             .agg(documents=("eid", "nunique"), citations=("citations", "sum"))
             .reset_index()
             .sort_values(["documents", "citations"], ascending=False, kind="stable")
             .reset_index(drop=True)
             .head(n))


def top_keywords(corpus, n: int = 30, kind: str = "author") -> pd.DataFrame:
    """Most frequent keywords. `kind`: 'author', 'index' or 'all'.

    Comparison ignores case, but the displayed spelling is the one most used
    by the authors.
    """
    k = corpus.keywords
    if kind != "all":
        k = k[k["kind"] == kind]
    k = k[k["keyword"].notna()]
    if k.empty:
        return pd.DataFrame(columns=["keyword", "documents"])

    k = k.copy()
    k["norm"] = k["keyword"].map(str).str.strip().str.lower()
    g = (k.groupby("norm")
           .agg(keyword=("keyword", lambda s: s.mode().iat[0]),
                documents=("eid", "nunique"))
           .reset_index(drop=True)
           .sort_values("documents", ascending=False, kind="stable")
           .reset_index(drop=True))
    return g.head(n)


def document_types(corpus) -> pd.DataFrame:
    d = corpus.documents
    d = d[d["doc_type"].notna() & (d["doc_type"].map(str).str.strip() != "")]
    if d.empty:
        return pd.DataFrame(columns=["doc_type", "documents", "share"])
    g = (d.groupby("doc_type")
           .agg(documents=("eid", "nunique"))
           .reset_index()
           .sort_values("documents", ascending=False, kind="stable")
           .reset_index(drop=True))
    total = g["documents"].sum()
    g["share"] = (100 * g["documents"] / total).round(1) if total else 0.0
    return g
