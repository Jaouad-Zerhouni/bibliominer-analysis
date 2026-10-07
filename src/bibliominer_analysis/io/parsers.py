"""Parsers for the packed cells of the Scopus export.

Every function takes ONE cell and returns a list of dictionaries. They are
pure, stateless, and never raise: an empty or malformed cell returns an
empty list. A damaged row must not bring down the analysis of a
3,000-document corpus.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from . import schema as S

_INDEX_RE = re.compile(S.AUTHOR_INDEX_PATTERN)
_AWA_MULTI_RE = re.compile(S.AWA_MULTI_AFFILIATION_PATTERN)
#: "Varela, Ana (59990000001)" -> name + Scopus identifier
_FULLNAME_RE = re.compile(r"^(.*?)\s*\((\d{5,})\)\s*$")


def _cell(value: Any) -> str:
    """Normalises a cell: NaN, None and "nan" become an empty string."""
    if value is None:
        return ""
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return ""
    return s


def split_list(value: Any) -> List[str]:
    """Splits a Scopus list cell ("A; B; C") into non-empty items."""
    s = _cell(value)
    if not s:
        return []
    return [p.strip() for p in s.split(S.LIST_SEP) if p.strip()]


def strip_index(item: str) -> "tuple[Optional[int], str]":
    '''"3:Lindqvist A." -> (3, 'Lindqvist A.'). Without a prefix -> (None, item).'''
    m = _INDEX_RE.match(item or "")
    if not m:
        return None, (item or "").strip()
    return int(m.group(1)), item[m.end():].strip()


# ---------------------------------------------------------------------------
# Authors
# ---------------------------------------------------------------------------

def _by_index(items: List[str]) -> Optional[Dict[int, str]]:
    """{position -> value} when EVERY entry carries its "n:" prefix."""
    out: Dict[int, str] = {}
    for raw in items:
        pos, value = strip_index(raw)
        if pos is None:
            return None
        out[pos] = value
    return out


def _surname_key(text: str) -> str:
    # The final dot of a lone initial ("A.") distinguishes nobody.
    return text.strip().rstrip(".").strip().lower()


def _surname_of_short(name: str) -> str:
    '''"El Tazari N." -> "el tazari": the short name without its initials.'''
    tokens = name.strip().split()
    while len(tokens) > 1 and "." in tokens[-1]:
        tokens.pop()
    return _surname_key(" ".join(tokens))


def _surname_of_full(full: str) -> str:
    """'El Tazari, Nora (59990000004)' -> 'el tazari'."""
    return _surname_key(full.split(",")[0])


def _align(names: List[tuple], items: List[str], by_surname) -> Dict[int, str]:
    """Author rank -> value of a parallel column, WITHOUT ever guessing.

    1. Both columns are indexed ("n:", cleaned file): matching by position,
       the only certain one.
    2. Same length: the Scopus columns are aligned, matching by rank.
    3. Different lengths: a shift is certain but its place unknown. Only what
       a surname attaches to ONE single author is then matched
       (`by_surname`), and nothing at all for a column without names.

    Fixed defect: matching was done by rank even when the lengths differed. An
    identifier missing in the middle of the list shifted all the following
    ones, and an author's h-index, co-authorship and self-citation were
    attributed to their neighbour.
    """
    if not items:
        return {}
    indexed_items = _by_index(items)
    if indexed_items is not None and all(pos is not None for pos, _ in names):
        return {rank: indexed_items[pos] for rank, (pos, _) in enumerate(names, 1)
                if pos in indexed_items}
    if len(items) == len(names):
        return {rank: strip_index(raw)[1] for rank, raw in enumerate(items, 1)}
    if by_surname is None:
        return {}
    wanted: Dict[str, List[int]] = {}
    for rank, (_, name) in enumerate(names, 1):
        wanted.setdefault(_surname_of_short(name), []).append(rank)
    out: Dict[int, str] = {}
    for raw in items:
        value = strip_index(raw)[1]
        ranks = wanted.get(by_surname(value), [])
        if len(ranks) == 1:
            out[ranks[0]] = value
    return out


def parse_authors(authors: Any, full_names: Any = None,
                  author_ids: Any = None) -> List[Dict[str, Any]]:
    """Merges the three author columns into an ordered list.

    Returns, per author: ``position``, ``name``, ``full_name``,
    ``scopus_id``.

    The POSITION comes from the "n:" prefix written by the cleaning; without
    it, the rank in the list is used. Matching the three columns follows
    `_align`: by position, by rank if the lengths agree, and never by rank
    when they differ; a damaged export must not shift identifiers from one
    author to another.
    """
    names = [strip_index(raw) for raw in split_list(authors)]
    fulls = _align(names, split_list(full_names), _surname_of_full)
    ids = _align(names, split_list(author_ids), None)

    out: List[Dict[str, Any]] = []
    for rank, (pos, name) in enumerate(names, start=1):
        full = fulls.get(rank, "")
        sid = ids.get(rank, "")

        # The identifier can come from the dedicated column OR from the brackets of
        # the full name. The brackets take precedence: the identifier there is
        # ATTACHED to the person's name and cannot slip to a neighbour, while the
        # dedicated column can be shifted.
        m = _FULLNAME_RE.match(full) if full else None
        if m:
            full, sid = m.group(1).strip(), m.group(2)

        out.append({
            "position": pos if pos is not None else rank,
            "name": name,
            "full_name": full or None,
            "scopus_id": sid or None,
        })
    return out


# ---------------------------------------------------------------------------
# Affiliations
# ---------------------------------------------------------------------------

def parse_affiliation(segment: str) -> Dict[str, Optional[str]]:
    """An affiliation -> its components.

    Bibliominer format (labelled):
        ``subparent: X, parent 1: Y, city: C, country: K``
    Empty fields are OMITTED from the export: the LABELS are read, never the
    positions.

    Raw Scopus format (not cleaned): nothing can be asserted about the
    intermediate segments. The only reliable convention is applied: the LAST
    segment is the country, the one before the city, provided that the last
    segment IS a recognised country (`io.countries.is_country`); the rest is
    ``None`` rather than a guess.
    """
    rec: Dict[str, Optional[str]] = {c: None for c in S.AFF_COLUMNS}
    rec["raw"] = (segment or "").strip()
    seg = rec["raw"]
    if not seg:
        return rec

    parts = [p.strip() for p in seg.split(",") if p.strip()]
    labelled = 0
    for part in parts:
        if ":" not in part:
            continue
        label, _, value = part.partition(":")
        key = S.AFF_LABELS.get(label.strip().lower())
        if key and value.strip():
            rec[key] = value.strip()
            labelled += 1

    if labelled:
        rec["labelled"] = True
        return rec

    # --- repli : export Scopus brut ---------------------------------------
    rec["labelled"] = False
    from .countries import is_country
    if not is_country(parts[-1]):
        # The last segment is not a country ("..., University Moulay Ismail of
        # Meknes"): it still became the "country", and the city the segment before.
        # Without a recognised country, the geography stays empty rather than
        # wrong; the rightmost segment is the parent organisation.
        rec["parent1"] = parts[-1]
        if len(parts) >= 2:
            rec["subparent"] = parts[-2]
        return rec
    rec["country"] = parts[-1]
    if len(parts) >= 2:
        rec["city"] = parts[-2]
    if len(parts) >= 3:
        # The rightmost segment before the geography is the most encompassing: it
        # is the parent organisation in Scopus's writing order.
        rec["parent1"] = parts[-3]
    if len(parts) >= 4:
        rec["subparent"] = parts[-4]
    return rec


def parse_affiliations(cell: Any) -> List[Dict[str, Optional[str]]]:
    """All the affiliations of a document."""
    return [parse_affiliation(s) for s in split_list(cell)]


#: Writing order of the labels in a Bibliominer affiliation. A label that
#: comes back, or goes back up in this order, opens the next affiliation.
_AFF_ORDER = {key: rank for rank, key in enumerate(S.AFF_COLUMNS)}


def _aff_key(rec: Dict[str, Optional[str]]) -> tuple:
    return tuple((rec.get(c) or "").strip().lower() for c in S.AFF_COLUMNS)


def _labelled_blocks(parts: List[str]) -> "tuple[str, List[str]]":
    """Segments of a labelled AWA block -> (name, [text of each affiliation])."""
    name_parts: List[str] = []
    affs: List[List[str]] = []
    seen: Dict[str, int] = {}
    last_rank = -1
    for part in parts:
        label = part.partition(":")[0].strip().lower() if ":" in part else ""
        key = S.AFF_LABELS.get(label)
        if key is None:
            if affs:
                affs[-1].append(part)      # a value containing a comma
            else:
                name_parts.append(part)
            continue
        rank = _AFF_ORDER[key]
        if not affs or key in seen or rank <= last_rank:
            affs.append([])
            seen = {}
        affs[-1].append(part)
        seen[key] = rank
        last_rank = rank
    return ", ".join(name_parts), [", ".join(a) for a in affs]


def author_affiliation_positions(block: str,
                                 doc_affs: List[Dict[str, Optional[str]]]
                                 ) -> List["tuple[Optional[int], str]"]:
    """The affiliations of ONE author, read from their "Authors with
    affiliations" block: [(position in the Affiliations column, text)].

    An author can have several, written one after the other in the same
    block:
        ``Okafor M., subparent: ENSIAS, ..., country: Morocco, subparent: ENSAM, ...``
    Each one is matched to the IDENTICAL affiliation of the document. Matching
    by rank (author 3 -> affiliation 3) gave an author their neighbour's
    affiliation as soon as a previous author had two, or two authors shared
    the same one.

    Raw Scopus export (not labelled): the affiliations of the document whose
    text appears in the block are kept. Nothing recognised: a single entry,
    without a position, that keeps the text.
    """
    _, text = strip_index(block)
    # "[2 affiliations] Okafor M., ...": the cleaning's mark tells the eye that
    # this author carries several. It is not part of the name.
    text = _AWA_MULTI_RE.sub("", text)
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        return []

    labelled = any(S.AFF_LABELS.get(p.partition(":")[0].strip().lower())
                   for p in parts if ":" in p)
    found = (_labelled_positions(parts, doc_affs) if labelled
             else _raw_positions(parts, doc_affs))
    return found or [(None, ", ".join(parts[1:]) or text)]


def _labelled_positions(parts: List[str], doc_affs) -> list:
    _, aff_texts = _labelled_blocks(parts)
    keys = [_aff_key(a) for a in doc_affs]
    out = []
    for aff_text in aff_texts:
        key = _aff_key(parse_affiliation(aff_text))
        out.append((keys.index(key) + 1 if key in keys else None, aff_text))
    return out


def _raw_positions(parts: List[str], doc_affs) -> list:
    """Raw export: the affiliations of the document whose text appears in the
    author's block are kept.

    A SHORT affiliation can be a piece of a long one: "ENSAM, University
    Moulay Ismail of Meknes" is the end of "IEST Research Team, AIDTM
    Laboratory, ENSAM, University Moulay Ismail of Meknes". Keeping it would
    give the author an affiliation they do not have. The longest matches are
    therefore kept first, and those that fall INSIDE a match already kept are
    left out.
    """
    tail = ", ".join(parts[1:]).lower()
    spans = []
    for pos, aff in enumerate(doc_affs, start=1):
        raw = (aff.get("raw") or "").strip().lower()
        at = tail.find(raw) if raw else -1
        if at >= 0:
            spans.append((at, at + len(raw), pos, aff.get("raw")))

    kept = []
    for start, end, pos, raw in sorted(spans, key=lambda s: -(s[1] - s[0])):
        if any(k[0] <= start and end <= k[1] for k in kept):
            continue
        kept.append((start, end, pos, raw))
    kept.sort()
    return [(pos, raw) for _, _, pos, raw in kept]


# ---------------------------------------------------------------------------
# Reconciled references
# ---------------------------------------------------------------------------

#: A reconciled reference starts with its "refN |" key.
_RECONCILED_REF = re.compile(r"^\s*ref\d+\s*\|")
#: In RAW Scopus text, a reference ends with its year in brackets; the next
#: one starts after the ";" that follows. A plain split on ";" ALSO cut
#: between the authors of the same reference ("Ali A.; Brandt C., A
#: systematic..."): each author became a "reference" without a title or a
#: year.
_RAW_REF_END = re.compile(r"(?<=\(\d{4}\))\s*;\s*")
_YEAR = re.compile(r"\((\d{4})\)")
_DOI = re.compile(r"10\.\d{4,9}/[^\s,;]+")


def _reconciled(item: str, i: int) -> Dict[str, Any]:
    fields = [f.strip() for f in item.split(S.REF_FIELD_SEP)]
    pos_raw, doi, year, authors, title = fields[0], fields[1], fields[2], fields[3], fields[4]
    m = re.search(r"(\d+)", pos_raw)
    return {
        "ref_pos": int(m.group(1)) if m else i,
        "ref_doi": (doi or None) or None,
        "ref_year": _to_int(year),
        "ref_authors": authors or None,
        "ref_title": title or None,
        "ref_raw": item,
    }


def _raw_scopus(item: str, i: int) -> Dict[str, Any]:
    '''"Ali A.; Brandt C., A systematic literature review..., Journal..., 31,
    (2019)" -> authors, title, year, DOI.

    Authors are separated by ";", the last one is followed by the title after
    a comma. When the form is not that one, the whole text stays the title:
    information is never lost.
    '''
    years = _YEAR.findall(item)
    doi = _DOI.search(item)
    parts = [p.strip() for p in item.split(";")]
    last_author, _, rest = parts[-1].partition(", ")
    authors = [p for p in parts[:-1] if p] + ([last_author] if rest else [])
    title = rest.split(", ")[0].strip() if rest else item
    return {
        "ref_pos": i,
        "ref_doi": doi.group(0).rstrip(".").lower() if doi else None,
        "ref_year": _to_int(years[-1]) if years else None,
        "ref_authors": ", ".join(authors) or None,
        "ref_title": title or None,
        "ref_raw": item,
    }


def parse_references(cell: Any) -> List[Dict[str, Any]]:
    """The references of a document, in the two possible forms.

    Reconciled by the cleaning:
        ``ref1 | 10.1007/... | 2016 | Lane, Moss | A guided tour of tree ensembles``
    Raw, as Scopus exports them:
        ``Ali A.; Brandt C., A systematic literature review..., (2019); ...``

    A reference that cannot be read keeps its whole text in ``ref_title``:
    information is never lost, even unstructured.
    """
    s = _cell(cell)
    if not s:
        return []
    if _RECONCILED_REF.match(s):
        out = []
        for i, item in enumerate(split_list(s), start=1):
            if len(item.split(S.REF_FIELD_SEP)) >= 5:
                out.append(_reconciled(item, i))
            else:
                out.append({"ref_pos": i, "ref_doi": None, "ref_year": None,
                            "ref_authors": None, "ref_title": item, "ref_raw": item})
        return out
    items = (_RAW_REF_END.split(s) if _YEAR.search(s)
             else [p for p in s.split(S.LIST_SEP)])
    return [_raw_scopus(item.strip(), i)
            for i, item in enumerate((x for x in items if x.strip()), start=1)]


def _to_int(value: Any) -> Optional[int]:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Keywords
# ---------------------------------------------------------------------------

def parse_keywords(cell: Any) -> List[str]:
    """Keywords of a cell, deduplicated ignoring case but keeping the spelling."""
    seen, out = set(), []
    for kw in split_list(cell):
        k = kw.strip()
        low = k.lower()
        if k and low not in seen:
            seen.add(low)
            out.append(k)
    return out
