"""Reading a cleaned CSV -> six "tidy" tables.

It is the only costly step of the package: it is done ONCE, and every
indicator is then only a grouping over these tables.

    documents            one row per document
    authors              one row per (document, author)            + position
    affiliations         one row per (document, affiliation)
    author_affiliations  one row per (document, author, affiliation)
    keywords             one row per (document, keyword)
    references           one row per (document, cited reference)
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from . import parsers as P
from . import schema as S

PathLike = Union[str, Path]

TABLE_NAMES = ("documents", "authors", "affiliations",
               "author_affiliations", "keywords", "references")


def read_csv(path: PathLike) -> pd.DataFrame:
    """Reads a Scopus/Bibliominer export without ever converting types.

    Everything is read as text: an 11-digit Scopus identifier would
    otherwise become a float, and "0123" would lose its zero. Conversions are
    done later, column by column, where we know what we are handling.
    """
    return pd.read_csv(path, dtype=str, keep_default_na=False,
                       na_values=[""], encoding="utf-8-sig", low_memory=False)


class NotCleanedError(ValueError):
    """The file is not the one exported by the Bibliominer cleaning."""


_INDEX = re.compile(S.AUTHOR_INDEX_PATTERN)


def _numbered_in_order(cell: str) -> bool:
    '''"1:A.; 2:B.; 3:C." yes; "A.; B." or "1:A.; 3:B." no.'''
    parts = [p for p in (x.strip() for x in cell.split(S.LIST_SEP)) if p]
    for expected, part in enumerate(parts, 1):
        m = _INDEX.match(part)
        if not m or int(m.group(1)) != expected:
            return False
    return bool(parts)


def check_cleaned(df: pd.DataFrame) -> None:
    """Refuses a file that did not go through the Bibliominer cleaning.

    The analysis relies on what the cleaning guarantees: authors aligned from
    one column to the next, affiliations labelled down to the city,
    reconciled references. On a raw Scopus export it still ran and gave wrong
    numbers without saying so (an author counted under two spellings, missing
    cities, co-citation on free text).

    The signature of the cleaned file: its final export, and only that,
    numbers the authors in order, in the three author columns
    ("1:Idri A.; 2:Hosni M."). Every non-empty cell must be numbered in full,
    as the export writes it; a single line that is not, and the file is not
    (or no longer) the one the cleaning produced.
    """
    columns = [c for c in (S.COL_AUTHORS, S.COL_AUTHOR_FULL, S.COL_AUTHOR_IDS)
               if c in df.columns]
    filled = bad = 0
    example = ""
    for col in columns:
        for row, value in df[col].items():
            cell = P._cell(value)
            if not cell:
                continue
            filled += 1
            if not _numbered_in_order(cell):
                bad += 1
                if not example:
                    shown = cell if len(cell) <= 60 else cell[:57] + "..."
                    # +2: the header line, and a numbering that starts at 1.
                    line = row + 2 if isinstance(row, int) else row
                    example = ' For example, line %s, column "%s": "%s".' % (line, col, shown)
    if S.COL_AUTHORS in columns and filled and not bad:
        return
    if bad:
        found = "%d author cell(s) out of %d are not numbered.%s" % (bad, filled, example)
    else:
        found = 'The file has no "%s" column filled in.' % S.COL_AUTHORS
    raise NotCleanedError(
        "This file has not been cleaned with Bibliominer. The analysis reads "
        "the file exported by the Bibliominer cleaning application, where "
        'authors are numbered in order ("1:Idri A.; 2:Hosni M."). %s '
        "Clean the Scopus export first, then import the file the cleaning "
        "exports." % found)


def _doc_id(row: Dict[str, Any], fallback: int) -> str:
    """Stable identifier of a document: EID, otherwise DOI, otherwise a hash of
    the title. Needed to join the six tables together."""
    for col in (S.COL_EID, S.COL_DOI):
        v = P._cell(row.get(col))
        if v:
            return v
    title = P._cell(row.get(S.COL_TITLE))
    if title:
        return "t:" + hashlib.md5(title.lower().encode("utf-8"), usedforsecurity=False).hexdigest()[:16]
    return "row:%d" % fallback


def _to_int(value: Any) -> Optional[int]:
    s = P._cell(value)
    if not s:
        return None
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return None


def build_tables(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Raw DataFrame -> the six tidy tables."""
    missing = [c for c in S.REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "Required columns missing: %s. "
            "This file does not look like a Scopus export." % ", ".join(missing)
        )

    docs: List[Dict[str, Any]] = []
    authors: List[Dict[str, Any]] = []
    affs: List[Dict[str, Any]] = []
    auth_affs: List[Dict[str, Any]] = []
    kws: List[Dict[str, Any]] = []
    refs: List[Dict[str, Any]] = []

    def col(row, name):
        return P._cell(row.get(name)) if name in df.columns else ""

    for i, row in enumerate(df.to_dict("records")):
        eid = _doc_id(row, i)

        docs.append({
            "eid": eid,
            "title": col(row, S.COL_TITLE) or None,
            "year": _to_int(row.get(S.COL_YEAR)),
            "source": col(row, S.COL_SOURCE) or None,
            "source_abbr": col(row, S.COL_SOURCE_ABBR) or None,
            "doc_type": col(row, S.COL_DOC_TYPE) or None,
            "language": col(row, S.COL_LANGUAGE) or None,
            "cited_by": _to_int(row.get(S.COL_CITED_BY)) or 0,
            "doi": col(row, S.COL_DOI) or None,
            "link": col(row, S.COL_LINK) or None,
            "abstract": col(row, S.COL_ABSTRACT) or None,
            "open_access": col(row, S.COL_OPEN_ACCESS) or None,
            "issn": col(row, S.COL_ISSN) or None,
            "publisher": col(row, S.COL_PUBLISHER) or None,
        })

        # --- authors ------------------------------------------------------
        parsed_authors = P.parse_authors(row.get(S.COL_AUTHORS),
                                         row.get(S.COL_AUTHOR_FULL),
                                         row.get(S.COL_AUTHOR_IDS))
        for a in parsed_authors:
            authors.append(dict(a, eid=eid))

        # --- affiliations --------------------------------------------------
        parsed_affs = P.parse_affiliations(row.get(S.COL_AFFILIATIONS))
        for j, a in enumerate(parsed_affs, start=1):
            affs.append(dict(a, eid=eid, aff_pos=j))

        # --- author x affiliation ------------------------------------------
        # "Authors with affiliations" has ONE block per author, in author order. A
        # block can carry SEVERAL affiliations: one row per (author, affiliation),
        # each pointing to the identical affiliation of the document, never "the
        # n-th one", which has nothing to do with the author.
        awa = P.split_list(row.get(S.COL_AUTHORS_AFF))
        if awa and parsed_authors:
            for rank, item in enumerate(awa, start=1):
                if rank > len(parsed_authors):
                    break
                for aff_pos, text in P.author_affiliation_positions(item, parsed_affs):
                    auth_affs.append({
                        "eid": eid,
                        "position": parsed_authors[rank - 1]["position"],
                        "aff_pos": aff_pos,
                        "raw": text or None,
                    })

        # --- keywords -------------------------------------------------------
        for kw in P.parse_keywords(row.get(S.COL_AUTHOR_KW)):
            kws.append({"eid": eid, "keyword": kw, "kind": "author"})
        for kw in P.parse_keywords(row.get(S.COL_INDEX_KW)):
            kws.append({"eid": eid, "keyword": kw, "kind": "index"})

        # --- references -----------------------------------------------------
        for r in P.parse_references(row.get(S.COL_REFERENCES)):
            refs.append(dict(r, eid=eid))

    tables = {
        "documents": pd.DataFrame(docs),
        "authors": pd.DataFrame(authors),
        "affiliations": pd.DataFrame(affs),
        "author_affiliations": pd.DataFrame(auth_affs),
        "keywords": pd.DataFrame(kws),
        "references": pd.DataFrame(refs),
    }
    return {name: _ensure_columns(name, t) for name, t in tables.items()}


#: Guaranteed columns of each table, even on an empty corpus: downstream code
#: must never have to test whether a column exists.
_EXPECTED = {
    "documents": ["eid", "title", "year", "source", "source_abbr", "doc_type",
                  "language", "cited_by", "doi", "link", "abstract",
                  "open_access", "issn", "publisher"],
    "authors": ["eid", "position", "name", "full_name", "scopus_id"],
    "affiliations": ["eid", "aff_pos", "raw", "labelled"] + list(S.AFF_COLUMNS),
    "author_affiliations": ["eid", "position", "aff_pos", "raw"],
    "keywords": ["eid", "keyword", "kind"],
    "references": ["eid", "ref_pos", "ref_doi", "ref_year", "ref_authors",
                   "ref_title", "ref_raw"],
}


def _ensure_columns(name: str, table: pd.DataFrame) -> pd.DataFrame:
    cols = _EXPECTED[name]
    for c in cols:
        if c not in table.columns:
            table[c] = pd.Series(dtype="object")
    return table[cols]


def _most_used(values: pd.Series) -> str:
    """The most used spelling; on a tie, the first in alphabetical order (same
    result from one run to the next)."""
    counts = values.value_counts()
    return sorted(counts[counts == counts.max()].index)[0]


def _loose_key(text: str) -> str:
    """Comparison key: lower case, no accents, letters and digits only."""
    import unicodedata
    s = unicodedata.normalize("NFKD", text)
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    return "".join(ch for ch in s if ch.isalnum())


def unify_spellings(tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """A single spelling per author and per keyword, for ALL the analyses.

    Scopus writes the same author in several ways from one article to the
    next ("Fernández-Alemán J.L.", "Fernandez-Aleman J.L.", "Fernández Alemán
    J.L.") under the same identifier; and authors write "Machine learning" or
    "Machine Learning". The rankings already grouped by identifier or ignoring
    case, but the authors' evolution, the keyword dynamics and the three-field
    plot grouped by raw text: the same author, the same word, appeared twice
    there, each with part of its documents.

    Every Scopus identifier therefore takes its most used name, and every
    keyword (same type, same text ignoring case) its most used spelling. An
    author without an identifier keeps their name as is: two namesakes stay
    indistinguishable.
    """
    out = dict(tables)

    a = out.get("authors")
    if a is not None and not a.empty and {"scopus_id", "name"} <= set(a.columns):
        known = a["scopus_id"].notna() & a["name"].notna()
        if known.any():
            a = a.copy()
            names = a.loc[known, "name"].map(str).str.strip()
            best = names.groupby(a.loc[known, "scopus_id"]).agg(_most_used)
            a.loc[known, "name"] = a.loc[known, "scopus_id"].map(best)
            out["authors"] = a

    # Organisations: the same name up to case, accents and punctuation
    # ("Faculty of Sciences Oujda - FSO" / "Faculty of Sciences Oujda-FSO") =
    # a single spelling. Never a translation, nor a matching of different
    # names: that is the cleaning's work.
    f = out.get("affiliations")
    if f is not None and not f.empty:
        f = f.copy()
        for col in ("subparent", "parent1", "parent2"):
            # NAMES only: a boolean or numeric column (an indicator) has no spelling,
            # converting it to text would destroy it.
            if col not in f.columns or pd.api.types.is_bool_dtype(f[col])                     or pd.api.types.is_numeric_dtype(f[col]):
                continue
            known = f[col].notna() & (f[col].map(str).str.strip() != "")
            if not known.any():
                continue
            text = f.loc[known, col].map(str).str.strip()
            loose = text.map(_loose_key)
            best = text.groupby(loose).agg(_most_used)
            f.loc[known, col] = loose.map(best)
        out["affiliations"] = f

    k = out.get("keywords")
    if k is not None and not k.empty and "keyword" in k.columns:
        known = k["keyword"].notna()
        if known.any():
            k = k.copy()
            text = k.loc[known, "keyword"].map(str).str.strip()
            kind = (k.loc[known, "kind"].map(str) if "kind" in k.columns
                    else pd.Series("", index=text.index))
            norm = kind + "|" + text.str.lower()
            best = text.groupby(norm).agg(_most_used)
            k.loc[known, "keyword"] = norm.map(best)
            out["keywords"] = k
    return out
