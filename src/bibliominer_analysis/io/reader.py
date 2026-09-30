"""Lecture d'un CSV nettoyé -> six tables « tidy ».

C'est la seule étape coûteuse du package : elle se fait UNE fois, et tous les
indicateurs ne sont ensuite que des regroupements sur ces tables.

    documents            une ligne par document
    authors              une ligne par (document, auteur)          + position
    affiliations         une ligne par (document, affiliation)
    author_affiliations  une ligne par (document, auteur, affiliation)
    keywords             une ligne par (document, mot-clé)
    references           une ligne par (document, référence citée)
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from . import parsers as P
from . import schema as S

PathLike = Union[str, Path]

TABLE_NAMES = ("documents", "authors", "affiliations",
               "author_affiliations", "keywords", "references")


def read_csv(path: PathLike) -> pd.DataFrame:
    """Lit un export Scopus/Bibliominer sans jamais convertir les types.

    Tout est lu en texte : un identifiant Scopus de 11 chiffres deviendrait
    sinon un flottant, et « 0123 » perdrait son zéro. Les conversions se font
    plus loin, colonne par colonne, là où on sait ce qu'on manipule.
    """
    return pd.read_csv(path, dtype=str, keep_default_na=False,
                       na_values=[""], encoding="utf-8-sig", low_memory=False)


def _doc_id(row: Dict[str, Any], fallback: int) -> str:
    """Identifiant stable d'un document : EID, sinon DOI, sinon empreinte du
    titre. Nécessaire pour joindre les six tables entre elles."""
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
    """DataFrame brut -> les six tables tidy."""
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

        # --- auteurs ------------------------------------------------------
        parsed_authors = P.parse_authors(row.get(S.COL_AUTHORS),
                                         row.get(S.COL_AUTHOR_FULL),
                                         row.get(S.COL_AUTHOR_IDS))
        for a in parsed_authors:
            authors.append(dict(a, eid=eid))

        # --- affiliations --------------------------------------------------
        parsed_affs = P.parse_affiliations(row.get(S.COL_AFFILIATIONS))
        for j, a in enumerate(parsed_affs, start=1):
            affs.append(dict(a, eid=eid, aff_pos=j))

        # --- auteur x affiliation ------------------------------------------
        # « Authors with affiliations » a UN bloc par auteur, dans l'ordre des
        # auteurs. Un bloc peut porter PLUSIEURS affiliations : une ligne par
        # (auteur, affiliation), chacune pointant l'affiliation identique du
        # document, jamais « la n-ième », qui n'a aucun rapport avec l'auteur.
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

        # --- mots-clés ------------------------------------------------------
        for kw in P.parse_keywords(row.get(S.COL_AUTHOR_KW)):
            kws.append({"eid": eid, "keyword": kw, "kind": "author"})
        for kw in P.parse_keywords(row.get(S.COL_INDEX_KW)):
            kws.append({"eid": eid, "keyword": kw, "kind": "index"})

        # --- références -----------------------------------------------------
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


#: Colonnes garanties de chaque table, même sur un corpus vide : le code aval
#: ne doit jamais avoir à tester l'existence d'une colonne.
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
    """La graphie la plus employée ; à égalité, la première dans l'ordre
    alphabétique (résultat identique d'une exécution à l'autre)."""
    counts = values.value_counts()
    return sorted(counts[counts == counts.max()].index)[0]


def _loose_key(text: str) -> str:
    """Clé de comparaison : minuscules, sans accents, lettres et chiffres seuls."""
    import unicodedata
    s = unicodedata.normalize("NFKD", text)
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    return "".join(ch for ch in s if ch.isalnum())


def unify_spellings(tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Une seule graphie par auteur et par mot-clé, pour TOUTES les analyses.

    Scopus écrit le même auteur de plusieurs façons d'un article à l'autre
    (« Fernández-Alemán J.L. », « Fernandez-Aleman J.L. », « Fernández Alemán
    J.L. ») sous un même identifiant ; et les auteurs écrivent « Machine
    learning » ou « Machine Learning ». Les classements regroupaient déjà par
    identifiant ou sans la casse, mais l'évolution des auteurs, la dynamique
    des mots-clés et le diagramme à trois champs regroupaient par texte brut :
    un même auteur, un même mot, y apparaissaient deux fois, chacun avec une
    partie de ses documents.

    Chaque identifiant Scopus prend donc son nom le plus employé, et chaque
    mot-clé (même type, même texte sans la casse) sa graphie la plus
    employée. Un auteur sans identifiant garde son nom tel quel : deux
    homonymes restent indiscernables.
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

    # Organisations : même nom à la casse, aux accents et à la ponctuation
    # près (« Faculty of Sciences Oujda - FSO » / « Faculty of Sciences
    # Oujda-FSO ») = une seule graphie. Jamais de traduction ni de
    # rapprochement de noms différents : c'est le travail du nettoyage.
    f = out.get("affiliations")
    if f is not None and not f.empty:
        f = f.copy()
        for col in ("subparent", "parent1", "parent2"):
            # Des NOMS seulement : une colonne booléenne ou numérique (un
            # indicateur) n'a pas de graphie, la convertir en texte la
            # détruirait.
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
