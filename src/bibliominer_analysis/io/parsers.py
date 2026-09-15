"""Parseurs des cellules compactées de l'export Scopus.

Chaque fonction prend UNE cellule et renvoie une liste de dictionnaires. Elles
sont pures, sans état, et ne lèvent jamais : une cellule vide ou mal formée
renvoie une liste vide. Une ligne abîmée ne doit pas faire tomber l'analyse
d'un corpus de 3 000 documents.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from . import schema as S

_INDEX_RE = re.compile(S.AUTHOR_INDEX_PATTERN)
#: « Idri, Ali (6602789810) » -> nom + identifiant Scopus
_FULLNAME_RE = re.compile(r"^(.*?)\s*\((\d{5,})\)\s*$")


def _cell(value: Any) -> str:
    """Normalise une cellule : NaN, None et « nan » deviennent une chaîne vide."""
    if value is None:
        return ""
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return ""
    return s


def split_list(value: Any) -> List[str]:
    """Découpe une cellule-liste Scopus (« A; B; C ») en éléments non vides."""
    s = _cell(value)
    if not s:
        return []
    return [p.strip() for p in s.split(S.LIST_SEP) if p.strip()]


def strip_index(item: str) -> "tuple[Optional[int], str]":
    """« 3:Abran A. » -> (3, 'Abran A.'). Sans préfixe -> (None, item)."""
    m = _INDEX_RE.match(item or "")
    if not m:
        return None, (item or "").strip()
    return int(m.group(1)), item[m.end():].strip()


# ---------------------------------------------------------------------------
# Auteurs
# ---------------------------------------------------------------------------

def _by_index(items: List[str]) -> Optional[Dict[int, str]]:
    """{position -> valeur} quand CHAQUE entrée porte son préfixe « n: »."""
    out: Dict[int, str] = {}
    for raw in items:
        pos, value = strip_index(raw)
        if pos is None:
            return None
        out[pos] = value
    return out


def _surname_key(text: str) -> str:
    # Le point final d'une initiale seule (« A. ») ne distingue personne.
    return text.strip().rstrip(".").strip().lower()


def _surname_of_short(name: str) -> str:
    """« El Baida M. » -> « el baida » : le nom court sans ses initiales."""
    tokens = name.strip().split()
    while len(tokens) > 1 and "." in tokens[-1]:
        tokens.pop()
    return _surname_key(" ".join(tokens))


def _surname_of_full(full: str) -> str:
    """« El Baida, Maelaynayn (58834078300) » -> « el baida »."""
    return _surname_key(full.split(",")[0])


def _align(names: List[tuple], items: List[str], by_surname) -> Dict[int, str]:
    """Rang d'auteur -> valeur d'une colonne parallèle, SANS jamais deviner.

    1. Les deux colonnes sont indexées (« n: », fichier nettoyé) : appariement
       par position, le seul certain.
    2. Même longueur : les colonnes Scopus sont alignées, appariement par rang.
    3. Longueurs différentes : un décalage est certain mais son endroit
       inconnu. On n'apparie alors que ce qu'un patronyme rattache à UN seul
       auteur (`by_surname`), et rien du tout pour une colonne sans nom.

    Défaut corrigé : l'appariement se faisait par rang même quand les
    longueurs différaient. Un identifiant manquant au milieu de la liste
    décalait tous les suivants, et le h-index, la co-signature et
    l'auto-citation d'un auteur étaient attribués à son voisin.
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
    """Fusionne les trois colonnes d'auteurs en une liste ordonnée.

    Renvoie, par auteur : ``position``, ``name``, ``full_name``, ``scopus_id``.

    La POSITION vient du préfixe « n: » posé par le cleaning ; sans lui on
    retombe sur le rang dans la liste. L'appariement des trois colonnes suit
    `_align` : par position, par rang si les longueurs concordent, et jamais
    par rang quand elles diffèrent — un export abîmé ne doit pas décaler les
    identifiants d'un auteur à l'autre.
    """
    names = [strip_index(raw) for raw in split_list(authors)]
    fulls = _align(names, split_list(full_names), _surname_of_full)
    ids = _align(names, split_list(author_ids), None)

    out: List[Dict[str, Any]] = []
    for rank, (pos, name) in enumerate(names, start=1):
        full = fulls.get(rank, "")
        sid = ids.get(rank, "")

        # L'identifiant peut venir de la colonne dédiée OU des parenthèses du
        # nom complet. Les parenthèses priment : l'identifiant y est COLLÉ au
        # nom de la personne et ne peut pas glisser vers un voisin, alors que
        # la colonne dédiée, elle, peut être décalée.
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
    """Une affiliation -> ses composants.

    Format Bibliominer (étiqueté) :
        ``subparent: X, parent 1: Y, city: C, country: K``
    Les champs vides sont OMIS à l'export : on lit donc les LIBELLÉS, jamais
    les positions.

    Format brut Scopus (non nettoyé) : on ne peut rien affirmer sur les
    segments intermédiaires. On applique la seule convention fiable — le
    DERNIER segment est le pays, l'avant-dernier la ville — et on laisse le
    reste à ``None`` plutôt que de deviner.
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
    if len(parts) >= 1:
        rec["country"] = parts[-1]
    if len(parts) >= 2:
        rec["city"] = parts[-2]
    if len(parts) >= 3:
        # Le segment le plus à droite avant la géo est le plus englobant :
        # c'est l'organisme mère dans l'ordre d'écriture Scopus.
        rec["parent1"] = parts[-3]
    if len(parts) >= 4:
        rec["subparent"] = parts[-4]
    return rec


def parse_affiliations(cell: Any) -> List[Dict[str, Optional[str]]]:
    """Toutes les affiliations d'un document."""
    return [parse_affiliation(s) for s in split_list(cell)]


# ---------------------------------------------------------------------------
# Références réconciliées
# ---------------------------------------------------------------------------

def parse_references(cell: Any) -> List[Dict[str, Any]]:
    """Références réconciliées par le cleaning.

        ``ref1 | 10.1007/... | 2016 | Biau, Scornet | A random forest guided tour``

    Une référence brute (sans barres verticales) est conservée telle quelle
    dans ``ref_title`` : on ne perd jamais l'information, même non structurée.
    """
    out: List[Dict[str, Any]] = []
    for i, item in enumerate(split_list(cell), start=1):
        fields = [f.strip() for f in item.split(S.REF_FIELD_SEP)]
        if len(fields) >= 5:
            pos_raw, doi, year, authors, title = fields[0], fields[1], fields[2], fields[3], fields[4]
            m = re.search(r"(\d+)", pos_raw)
            out.append({
                "ref_pos": int(m.group(1)) if m else i,
                "ref_doi": (doi or None) or None,
                "ref_year": _to_int(year),
                "ref_authors": authors or None,
                "ref_title": title or None,
                "ref_raw": item,
            })
        else:
            out.append({
                "ref_pos": i, "ref_doi": None, "ref_year": None,
                "ref_authors": None, "ref_title": item, "ref_raw": item,
            })
    return out


def _to_int(value: Any) -> Optional[int]:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Mots-clés
# ---------------------------------------------------------------------------

def parse_keywords(cell: Any) -> List[str]:
    """Mots-clés d'une cellule, dédoublonnés sans casse mais graphie conservée."""
    seen, out = set(), []
    for kw in split_list(cell):
        k = kw.strip()
        low = k.lower()
        if k and low not in seen:
            seen.add(low)
            out.append(k)
    return out
