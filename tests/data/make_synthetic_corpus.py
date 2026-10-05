"""A FICTITIOUS corpus in the format of a file cleaned by Bibliominer.

Real corpora are users' data: they are not in the repository, and without
them the end-to-end tests were skipped by the CI. This corpus replaces
them wherever no real file is given: same columns, same conventions
(authors indexed "1:Name", labelled affiliations, references reconciled
"refN | DOI | year | authors | title"), and enough structure for every
analysis to have something to compute.

Everything is invented and DETERMINISTIC (fixed seed): authors built from
syllables, imaginary institutions in real cities (maps need countries),
DOIs under Crossref's test prefix (10.5555). Only the journals are real,
taken from the package's SCImago table, so that quartiles have matches.

    python make_synthetic_corpus.py            # rewrites synthetic_cleaned.csv
    python make_synthetic_corpus.py 10000 big.csv
"""
from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEED = 20261002

SYLLABLES = ["ar", "bel", "cor", "dan", "el", "fen", "gar", "hol", "is", "jor", "kel", "lan",
             "mor", "nel", "or", "pal", "quin", "ros", "sel", "tor", "ul", "val", "wen", "zar"]
GIVEN = ["Ana", "Bruno", "Clara", "Dimitri", "Elif", "Farid", "Greta", "Hugo", "Ines", "Jonas",
         "Karim", "Lena", "Malik", "Nora", "Omar", "Paula", "Rami", "Sara", "Tomas", "Yasmin"]

# (city, region, country): real cities, for the maps and the countries.
PLACES = [("Rabat", "Rabat-Sale-Kenitra", "Morocco"), ("Meknes", "Fes-Meknes", "Morocco"),
          ("Murcia", "Murcia", "Spain"), ("Madrid", "Madrid", "Spain"),
          ("Paris", "Ile-de-France", "France"), ("Montreal", "Quebec", "Canada"),
          ("Milan", "Lombardy", "Italy"), ("Berlin", "Berlin", "Germany"),
          ("Boston", "Massachusetts", "United States"), ("Sao Paulo", "Sao Paulo", "Brazil")]
INSTITUTION_WORDS = ["Northern", "Coastal", "Central", "Royal", "Polytechnic", "Metropolitan"]
UNITS = ["School of Computing", "Faculty of Sciences", "Data Science Lab", "Software Engineering Group"]

TOPICS = ["effort estimation", "machine learning", "ensemble learning", "deep learning",
          "software quality", "defect prediction", "data mining", "feature selection",
          "case-based reasoning", "neural networks", "fuzzy logic", "systematic review",
          "requirements engineering", "mobile health", "cloud computing", "explainability",
          "transfer learning", "random forest", "support vector machines", "clustering"]
DOC_TYPES = [("Article", 0.55), ("Conference Paper", 0.35), ("Review", 0.06), ("Book Chapter", 0.04)]
OPEN_ACCESS = ["", "", "All Open Access; Gold Open Access", "All Open Access; Green Open Access"]


def _name(rng: random.Random) -> str:
    surname = "".join(rng.choice(SYLLABLES) for _ in range(rng.choice((2, 3)))).capitalize()
    return surname


def _journals(n: int) -> list:
    """Real journals (name, ISSN) from the SCImago table shipped with the
    package."""
    table = HERE.parents[1] / "src" / "bibliominer_analysis" / "data_ref" / "scimagojr_2025.csv"
    out = []
    with table.open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter=";"):
            if "Computer Science" in (row.get("Areas") or "") and row.get("Type") == "journal":
                issn = (row.get("Issn") or "").split(",")[0].strip()
                if len(issn) == 8:
                    out.append((row["Title"], issn))
            if len(out) >= n:
                break
    return out


def build(n_docs: int = 180, seed: int = SEED) -> list:
    rng = random.Random(seed)
    authors = []
    for k in range(max(40, n_docs // 4)):
        surname, given = _name(rng), rng.choice(GIVEN)
        authors.append({"short": f"{surname} {given[0]}.", "full": f"{surname}, {given}",
                        "id": str(57000000000 + k)})
    institutions = []
    for k, place in enumerate(PLACES * 2):
        name = f"{rng.choice(INSTITUTION_WORDS)} University of {place[0]}"
        institutions.append({"parent": name, "unit": UNITS[k % len(UNITS)], "place": place})
    for a in authors:
        a["inst"] = rng.choice(institutions)
    journals = _journals(15) or [("Journal of Example Studies", "00000000")]

    years = list(range(2014, 2026))
    docs = []
    for i in range(n_docs):
        year = rng.choices(years, weights=[1 + j for j in range(len(years))])[0]
        team = rng.sample(authors, rng.choice((1, 2, 3, 3, 4, 5)))
        topics = rng.sample(TOPICS, 4)
        doi = f"10.5555/synthetic.{year}.{i:05d}"
        docs.append({"i": i, "year": year, "team": team, "topics": topics, "doi": doi,
                     "journal": rng.choice(journals), "eid": f"2-s2.0-{85000000000 + i}"})

    # Shared references (co-citation), and internal citations (local
    # citations): an article cites older articles of the corpus.
    pool = [{"doi": f"10.5555/reference.{k:04d}", "year": rng.randint(1995, 2020),
             "authors": f"{_name(rng)} {rng.choice(GIVEN)[0]}., {_name(rng)} {rng.choice(GIVEN)[0]}.",
             "title": f"On {rng.choice(TOPICS)} and {rng.choice(TOPICS)}"} for k in range(120)]
    rows = []
    for d in docs:
        refs = rng.sample(pool, rng.randint(8, 18))
        older = [o for o in docs if o["year"] < d["year"]]
        for o in rng.sample(older, min(len(older), rng.randint(0, 3))):
            refs.append({"doi": o["doi"], "year": o["year"],
                         "authors": ", ".join(a["short"] for a in o["team"]),
                         "title": f"Study {o['i']} on {o['topics'][0]}"})
        references = " ; ".join(f"ref{k + 1} | {r['doi']} | {r['year']} | {r['authors']} | {r['title']}"
                                for k, r in enumerate(refs))
        affs, awa = [], []
        for a in d["team"]:
            city, region, country = a["inst"]["place"]
            label = (f"subparent: {a['inst']['unit']}, parent 1: {a['inst']['parent']}, "
                     f"city: {city}, region: {region}, country: {country}")
            if label not in affs:
                affs.append(label)
            awa.append(f"{a['short']}, {label}")
        doc_type = rng.choices([t for t, _ in DOC_TYPES], weights=[w for _, w in DOC_TYPES])[0]
        title = f"Study {d['i']} on {d['topics'][0]} with {d['topics'][1]}"
        rows.append({
            "Authors": "; ".join(f"{k + 1}:{a['short']}" for k, a in enumerate(d["team"])),
            "Author full names": "; ".join(f"{k + 1}:{a['full']} ({a['id']})"
                                           for k, a in enumerate(d["team"])),
            "Author(s) ID": "; ".join(f"{k + 1}:{a['id']}" for k, a in enumerate(d["team"])),
            "Title": title,
            "Year": str(d["year"]),
            "Source title": d["journal"][0],
            "Cited by": str(int(rng.expovariate(1 / 12)) if rng.random() > 0.15 else ""),
            "DOI": d["doi"],
            "Link": f"https://www.scopus.com/inward/record.uri?eid={d['eid']}&partnerID=40",
            "Affiliations": "; ".join(affs),
            "Authors with affiliations": "; ".join(awa),
            "Abstract": (f"This study addresses {d['topics'][0]} using {d['topics'][1]}. "
                         f"We compare {d['topics'][2]} and {d['topics'][3]} on public datasets, "
                         f"and the results show that {d['topics'][1]} improves {d['topics'][0]}."),
            "Author Keywords": "; ".join(d["topics"][:3]),
            "Index Keywords": "; ".join(d["topics"]),
            "References": references,
            "Correspondence Address": f"{d['team'][0]['short']}; {d['team'][0]['inst']['parent']}",
            "ISSN": d["journal"][1],
            "Language of Original Document": "English",
            "Document Type": doc_type,
            "Publication Stage": "Final",
            "Open Access": rng.choice(OPEN_ACCESS),
            "Source": "Scopus",
            "EID": d["eid"],
        })
    return rows


def write(rows: list, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 180
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "synthetic_cleaned.csv"
    write(build(n), out)
    print(f"{n} documents -> {out}")
