"""Les pays : les reconnaître, les nommer comme le fond de carte, les placer.

Utilisé à la LECTURE d'un fichier brut (le dernier segment d'une affiliation
Scopus n'est un pays que s'il est reconnu ici) et par les cartes du monde.
Aucune dépendance graphique : ce module se charge sans matplotlib.

Sources, dans ``data_ref`` :
  - ``countries-110m.json``, le fond de carte de l'interface (world-atlas,
    Natural Earth, licence ISC) ;
  - ``countries.csv``, les 249 pays et territoires ISO 3166-1 (+ Kosovo),
    avec leur nom sur le fond de carte, ou la position de leur capitale
    quand ils sont trop petits pour y figurer.
"""

from __future__ import annotations

import json
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_ATLAS = Path(__file__).resolve().parent.parent / "data_ref" / "countries-110m.json"
#: Les pays et territoires ISO 3166-1 (+ Kosovo) : noms officiel et courant,
#: nom sur le fond de carte, et position (lon, lat) de la capitale pour ceux
#: que le fond au 1:110 m ne dessine pas (Singapour, Bahreïn, Malte…).
_COUNTRIES = Path(__file__).resolve().parent.parent / "data_ref" / "countries.csv"

#: Graphie du corpus (sans accents, minuscules) -> nom du fond de carte.
ATLAS_ALIASES: Dict[str, str] = {
    "united states": "United States of America",
    "usa": "United States of America",
    "us": "United States of America",
    "russian federation": "Russia",
    "viet nam": "Vietnam",
    "czech republic": "Czechia",
    "korea": "South Korea",
    "republic of korea": "South Korea",
    "korea, republic of": "South Korea",
    "korea, democratic people's republic of": "North Korea",
    "democratic people's republic of korea": "North Korea",
    "iran, islamic republic of": "Iran",
    "islamic republic of iran": "Iran",
    "syrian arab republic": "Syria",
    "lao people's democratic republic": "Laos",
    "bosnia and herzegovina": "Bosnia and Herz.",
    "central african republic": "Central African Rep.",
    "democratic republic of the congo": "Dem. Rep. Congo",
    "democratic republic of congo": "Dem. Rep. Congo",
    "dr congo": "Dem. Rep. Congo",
    "congo, the democratic republic of the": "Dem. Rep. Congo",
    "republic of the congo": "Congo",
    "republic of congo": "Congo",
    "dominican republic": "Dominican Rep.",
    "equatorial guinea": "Eq. Guinea",
    "south sudan": "S. Sudan",
    "solomon islands": "Solomon Is.",
    "falkland islands": "Falkland Is.",
    "ivory coast": "Côte d'Ivoire",
    "cote d'ivoire": "Côte d'Ivoire",
    "north macedonia": "Macedonia",
    "tanzania, united republic of": "Tanzania",
    "united republic of tanzania": "Tanzania",
    "taiwan, province of china": "Taiwan",
    "state of palestine": "Palestine",
    "palestine, state of": "Palestine",
    "falkland islands (malvinas)": "Falkland Is.",
    "palestinian territory": "Palestine",
    "timor leste": "Timor-Leste",
    "east timor": "Timor-Leste",
    "eswatini": "eSwatini",
    "swaziland": "eSwatini",
    "brunei darussalam": "Brunei",
    "moldova, republic of": "Moldova",
    "republic of moldova": "Moldova",
    "libyan arab jamahiriya": "Libya",
    "western sahara": "W. Sahara",
    "turkiye": "Turkey",
    "the netherlands": "Netherlands",
    "the bahamas": "Bahamas",
    "the gambia": "Gambia",
    "uk": "United Kingdom",
    "great britain": "United Kingdom",
}


def _key(name: object) -> str:
    text = unicodedata.normalize("NFKD", str(name or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(text.lower().replace("’", "'").split())


@lru_cache(maxsize=1)
def _atlas() -> Tuple[Dict[str, List[List[Tuple[float, float]]]], Dict[str, str]]:
    """{nom -> anneaux (lon, lat)} et {clé normalisée -> nom}."""
    topo = json.loads(_ATLAS.read_text(encoding="utf-8"))
    sx, sy = topo["transform"]["scale"]
    tx, ty = topo["transform"]["translate"]
    arcs = []
    for arc in topo["arcs"]:
        x = y = 0
        pts = []
        for dx, dy in arc:
            x += dx
            y += dy
            pts.append((x * sx + tx, y * sy + ty))
        arcs.append(pts)

    def ring(indices: Sequence[int]) -> List[Tuple[float, float]]:
        out: List[Tuple[float, float]] = []
        for i in indices:
            pts = arcs[i] if i >= 0 else arcs[~i][::-1]
            out.extend(pts if not out else pts[1:])
        return out

    shapes: Dict[str, List[List[Tuple[float, float]]]] = {}
    for geo in topo["objects"]["countries"]["geometries"]:
        name = geo.get("properties", {}).get("name")
        if not name or "arcs" not in geo:
            continue
        polygons = geo["arcs"] if geo["type"] == "MultiPolygon" else [geo["arcs"]]
        # L'anneau extérieur de chaque polygone ; les trous (lacs) sont
        # négligeables à cette échelle.
        shapes[name] = [_unwrap(ring(poly[0])) for poly in polygons]
    return shapes, {_key(n): n for n in shapes}


def _unwrap(ring_: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Un anneau qui franchit la ligne des 180° (Russie orientale, Fidji)
    saute de +180 à -180 : dessiné tel quel, il barre toute la carte d'une
    bande horizontale. On le recolle d'un seul côté."""
    out = [ring_[0]]
    for x, y in ring_[1:]:
        px = out[-1][0]
        while x - px > 180:
            x -= 360
        while px - x > 180:
            x += 360
        out.append((x, y))
    xs = [x for x, _ in out]
    if min(xs) < -180:
        out = [(x + 360, y) for x, y in out]
    return out


@lru_cache(maxsize=1)
def _countries() -> Dict[str, Dict[str, str]]:
    """{nom normalisé -> ligne de la table des pays}, pour tous ses noms."""
    import csv
    out: Dict[str, Dict[str, str]] = {}
    with _COUNTRIES.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            for col in ("name", "common_name", "official_name"):
                if row.get(col):
                    out.setdefault(_key(row[col]), row)
    return out


def _country_row(country: object) -> Optional[Dict[str, str]]:
    k = _key(country)
    row = _countries().get(k)
    if row is None and k in ATLAS_ALIASES:
        target = _key(ATLAS_ALIASES[k])
        row = next((r for r in _countries().values() if _key(r.get("atlas")) == target), None)
    if row is None:
        atlas = _atlas()[1].get(k)
        row = next((r for r in _countries().values() if atlas and r.get("atlas") == atlas), None)
    return row


def is_country(name: object) -> bool:
    """Ce texte est-il un pays (ou un territoire) ? Tous les noms ISO,
    officiels et courants, les graphies Scopus connues et ceux du fond de
    carte sont reconnus, sans tenir compte de la casse ni des accents."""
    k = _key(name)
    return bool(k) and (k in _countries() or k in ATLAS_ALIASES or k in _atlas()[1])


def atlas_name(country: object) -> Optional[str]:
    """Le nom du fond de carte pour un pays du corpus, ou None s'il n'y est pas
    (un micro-État, par exemple, absent de l'échelle 1:110 m)."""
    shapes, by_key = _atlas()
    k = _key(country)
    if k in by_key:
        return by_key[k]
    alias = ATLAS_ALIASES.get(k)
    if alias in shapes:
        return alias
    row = _country_row(country)
    return row["atlas"] if row and row.get("atlas") in shapes else None


def position(country: object) -> Optional[Tuple[float, float]]:
    """(lon, lat) où placer un pays : le centre de son territoire sur le fond
    de carte, ou sa capitale s'il est trop petit pour y figurer."""
    name = atlas_name(country)
    if name:
        return centroids()[name]
    row = _country_row(country)
    if row and row.get("lon") and row.get("lat"):
        return float(row["lon"]), float(row["lat"])
    return None


def _centroid(rings: List[List[Tuple[float, float]]]) -> Tuple[float, float]:
    """Centre du plus grand polygone (aire signée de la formule du lacet) :
    la France se place en métropole, pas entre Paris et la Guyane."""
    best, best_area = rings[0], -1.0
    for r in rings:
        area = abs(sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(r, r[1:] + r[:1]))) / 2
        if area > best_area:
            best, best_area = r, area
    a = cx = cy = 0.0
    for (x0, y0), (x1, y1) in zip(best, best[1:] + best[:1]):
        f = x0 * y1 - x1 * y0
        a += f
        cx += (x0 + x1) * f
        cy += (y0 + y1) * f
    if abs(a) < 1e-12:
        xs, ys = zip(*best)
        return sum(xs) / len(xs), sum(ys) / len(ys)
    return cx / (3 * a), cy / (3 * a)


@lru_cache(maxsize=1)
def centroids() -> Dict[str, Tuple[float, float]]:
    """{nom du fond de carte -> (lon, lat)}."""
    return {n: _centroid(r) for n, r in _atlas()[0].items()}
