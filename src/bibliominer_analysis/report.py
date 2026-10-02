"""Le rapport : figures et tableaux d'une étude, rangés dans un seul ZIP.

C'est le même rapport que le panier « Add to report » de l'interface, l'API
appelle cette classe, et un utilisateur du paquet le construit en Python :

    >>> from bibliominer_analysis import Corpus, Report
    >>> corpus = Corpus.from_csv("cleaned.csv").filter(years=(2023, 2025))
    >>> report = Report(filters={"years": "2023-2025"})
    >>> report.add_table("Actors", "Top 10 authors", corpus.top_authors(10))
    >>> report.add_indicators("Impact", "Collaboration",
    ...                       corpus.collaboration_indicators())
    >>> report.add_figure("Corpus", "Annual production",
    ...                   corpus.figure_spec("production_by_year"))
    >>> report.save("report.zip")

Ce qu'il contient ::

    README.txt                      l'index : chaque fichier, son titre, les filtres
    all_tables.xlsx                 tous les tableaux, une feuille chacun
    1-corpus/figures/<nom>.png      300 dpi, qualité impression
    1-corpus/figures/<nom>.svg      vectoriel : s'agrandit sans perte
    1-corpus/corpus_tables.xlsx     les tableaux de la section, une feuille
                                    chacun, et une feuille « Contents »
    2-actors/…  3-impact/…  4-concepts/…  5-networks/…

UN classeur par section, pas un fichier par tableau : une étude compte une
centaine de tableaux, et vingt fichiers Excel dans un dossier ne se
parcourent pas. Son nom porte la section (``actors_tables.xlsx``) : Excel
refuse d'ouvrir ensemble deux classeurs du même nom. La feuille « Contents »
donne le titre ENTIER de chaque feuille, sa période et ses filtres : un nom
de feuille est coupé à 31 caractères.

Une entrée calculée sur une PÉRIODE (``period="2010-2013"``) la porte dans le
nom de son fichier (``top-authors_2010-2013.png``) ou de sa feuille. Deux
classements des auteurs sur deux périodes donnent deux fichiers, deux
feuilles, jamais un seul écrasé.

Une section par groupe de l'interface (Corpus, Actors, Impact, Concepts,
Networks), dans l'ordre du menu ; une section inconnue vient ensuite.
"""

from __future__ import annotations

import dataclasses
import io
import re
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import pandas as pd

from . import __version__
from ._xlsx import sheet_name, workbook_bytes
from .figures.render import FigureSpec, render_figure

__all__ = ["Report", "SECTIONS", "table_workbook"]

#: Les groupes du menu de l'interface, dans leur ordre : un dossier chacun.
SECTIONS = ("Corpus", "Actors", "Impact", "Concepts", "Networks")

#: Résolution des figures matricielles : celle qu'exigent les revues.
PRINT_DPI = 300

_EPOCH = (1980, 1, 1, 0, 0, 0)
_SIGNATURES = ((b"\x89PNG", "png"), (b"\xff\xd8", "jpg"), (b"%PDF", "pdf"),
               (b"<?xml", "svg"), (b"<svg", "svg"))

FigureLike = Union[FigureSpec, bytes, Mapping[str, bytes], Callable[[str], bytes]]
TableLike = Union[pd.DataFrame, Sequence[Mapping[str, Any]], Mapping[str, Any]]


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").strip().lower())
    return s.strip("-") or "item"


def _frame(table: TableLike) -> pd.DataFrame:
    if isinstance(table, pd.DataFrame):
        return table.reset_index(drop=True)
    if isinstance(table, Mapping):
        return pd.DataFrame([dict(table)])
    return pd.DataFrame(list(table))


def table_workbook(table: TableLike, name: str = "Table") -> bytes:
    """Un tableau seul, en classeur Excel d'une feuille.

    L'export « Excel » d'un tableau de l'interface : écrit par le paquet,
    comme les classeurs du rapport, sans dépendance de plus."""
    return workbook_bytes([(name or "Table", _frame(table))])


def _format_of(data: bytes) -> str:
    head = data[:64].lstrip()
    for signature, ext in _SIGNATURES:
        if head.startswith(signature):
            return ext
    raise ValueError("unknown image format: expected PNG, JPEG, SVG or PDF bytes")


@dataclasses.dataclass
class _Table:
    section: str
    name: str
    frame: pd.DataFrame
    note: str = ""
    title: str = ""
    period: str = ""


@dataclasses.dataclass
class _Figure:
    section: str
    name: str
    files: Dict[str, bytes]
    note: str = ""
    title: str = ""
    period: str = ""


def _book(section: str) -> str:
    """Le classeur des tableaux d'une section : ``actors_tables.xlsx``."""
    return f"{_slug(section).replace('-', '_')}_tables.xlsx"


def _stem(item: Union[_Table, _Figure]) -> str:
    """Le nom de fichier d'une entrée : son nom, puis sa période."""
    stem = _slug(item.name)
    return f"{stem}_{_slug(item.period)}" if item.period.strip() else stem


class Report:
    """Un rapport en construction : on y ajoute tableaux, indicateurs et
    figures, puis `save` écrit le ZIP.

    ``filters``  les filtres de l'étude (période, types…), recopiés dans le
                 README : un chiffre sans son périmètre ne se cite pas.
    ``formats``  les formats de chaque figure : PNG (``dpi``) et SVG par défaut.
    """

    def __init__(self, title: str = "Bibliometric report", *,
                 filters: Optional[Mapping[str, Any]] = None,
                 dpi: int = PRINT_DPI,
                 formats: Iterable[str] = ("png", "svg"),
                 created: Optional[datetime] = None) -> None:
        self.title = title
        self.filters = dict(filters or {})
        self.dpi = int(dpi)
        self.formats = tuple(formats)
        self.created = created
        self._items: List[Union[_Table, _Figure]] = []

    def __len__(self) -> int:
        return len(self._items)

    # -- ajouter --------------------------------------------------------------

    def add_table(self, section: str, name: str, table: TableLike,
                  note: str = "", *, title: str = "", period: str = "") -> "Report":
        """Un tableau (DataFrame, liste de lignes ou dictionnaire) : un
        feuille du classeur de sa section (``<section>_tables.xlsx``), et une feuille de
        ``all_tables.xlsx``.

        ``title``  le titre écrit dans le README (à défaut, ``name``) ;
        ``period`` la période du tableau, ajoutée au nom du fichier."""
        self._items.append(_Table(section, name, _frame(table), note, title, period))
        return self

    def add_indicators(self, section: str, name: str,
                       values: Union[Mapping[str, Any], TableLike],
                       definitions: Optional[Mapping[str, str]] = None,
                       *, period: str = "") -> "Report":
        """Des indicateurs (ceux des tuiles de l'interface) : un tableau
        ``indicator | value | definition``. Un dictionnaire imbriqué ou une
        liste n'est pas un indicateur : ces valeurs-là sont écartées."""
        if isinstance(values, Mapping):
            definitions = definitions or {}
            rows = [{"indicator": key, "value": value,
                     "definition": definitions.get(key, "")}
                    for key, value in values.items()
                    if not isinstance(value, (Mapping, list, tuple, pd.DataFrame))]
            if not any(row["definition"] for row in rows):
                rows = [{"indicator": r["indicator"], "value": r["value"]} for r in rows]
            return self.add_table(section, name, rows, period=period)
        return self.add_table(section, name, values, period=period)

    def add_figure(self, section: str, name: str, figure: FigureLike,
                   note: str = "", *, title: str = "", period: str = "") -> "Report":
        """Une figure, dans chacun des formats du rapport.

        ``figure`` peut être une `FigureSpec` (rendue ici, à ``dpi``), une
        fonction ``fmt -> octets`` (``lambda fmt: render_network(g, fmt=fmt)``),
        un dictionnaire ``{"png": …, "svg": …}`` déjà rendu, ou les octets
        d'une seule image. ``title`` : le titre du README (à défaut, ``name``) ;
        ``period`` : la période de la figure, ajoutée au nom du fichier.
        """
        files: Dict[str, bytes] = {}
        if isinstance(figure, FigureSpec):
            spec = dataclasses.replace(figure, dpi=max(int(figure.dpi or 0), self.dpi))
            files = {fmt: render_figure(spec, fmt=fmt) for fmt in self.formats}
        elif callable(figure):
            files = {fmt: figure(fmt) for fmt in self.formats}
        elif isinstance(figure, Mapping):
            files = {str(ext).lower().lstrip("."): bytes(data) for ext, data in figure.items()}
        else:
            data = bytes(figure)
            files = {_format_of(data): data}
        if not files:
            raise ValueError(f"'{name}': no image to file")
        self._items.append(_Figure(section, name, files, note, title, period))
        return self

    def add_corpus_figure(self, corpus: Any, name: str, *, period: str = "",
                          style: Optional[Mapping[str, Any]] = None,
                          **options: Any) -> "Report":
        """Une figure du catalogue de l'interface, rangée dans SA section et
        sous SON titre, l'équivalent du bouton « Add to report » :

            >>> report.add_corpus_figure(corpus, "top-authors", n=10)
            >>> recent = corpus.filter(years=(2010, 2013))
            >>> report.add_corpus_figure(recent, "top-authors", period="2010-2013",
            ...                          style={"show_title": True, "palette": "gradient"})
        """
        from .figures.catalog import entry, figure_bytes
        e = entry(name)
        return self.add_figure(e.section, e.name,
                               lambda fmt: figure_bytes(corpus, e.name, fmt=fmt,
                                                        dpi=self.dpi,
                                                        style=dict(style or {}), **options),
                               title=e.title, period=period)

    # -- écrire ---------------------------------------------------------------

    def _folders(self) -> Dict[str, str]:
        extra = [s for s in dict.fromkeys(i.section for i in self._items)
                 if s not in SECTIONS]
        order = list(SECTIONS) + extra
        return {section: f"{k}-{_slug(section)}" for k, section in enumerate(order, start=1)}

    def _readme(self, index: List[tuple]) -> str:
        when = (self.created or datetime.now()).strftime("%Y-%m-%d %H:%M")
        lines = [self.title, "=" * len(self.title), "",
                 f"Generated by bibliominer-analysis {__version__} on {when}."]
        if self.filters:
            lines.append("Filters: " + "; ".join(f"{k}: {v}" for k, v in self.filters.items()
                                                  if v not in (None, "", [], ())))
        else:
            lines.append("Filters: none (the whole corpus).")
        lines += ["",
                  f"Figures: PNG at {self.dpi} dpi (print quality) and SVG (vector: "
                  "scales without loss, editable in Inkscape or Illustrator).",
                  "Tables: Excel (.xlsx), one workbook per folder (its first sheet, "
                  "Contents, lists the tables); all_tables.xlsx holds every table, one "
                  "sheet each.",
                  ""]
        folder = None
        for path, title, note in index:
            top = path.split("/", 1)[0]
            if top != folder:
                folder = top
                lines += ["", f"{folder}/"]
            lines.append(f"  {path.split('/', 1)[1]:<48} {title}" + (f" ({note})" if note else ""))
        return "\n".join(lines) + "\n"

    def to_bytes(self) -> bytes:
        """Le ZIP du rapport, en mémoire."""
        if not self._items:
            raise ValueError("the report is empty: add a table or a figure first")
        folders = self._folders()
        used: set = set()
        index: List[tuple] = []
        entries: List[tuple] = []

        def unique(path: str) -> str:
            stem, ext = path.rsplit(".", 1)
            candidate, n = path, 2
            while candidate in used:
                candidate = f"{stem}-{n}.{ext}"
                n += 1
            used.add(candidate)
            return candidate

        sheets = []
        # Les tableaux d'une section : un classeur, une feuille chacun.
        section_sheets: Dict[str, List[tuple]] = {}
        section_taken: Dict[str, set] = {}
        # Dans l'ordre du menu, et dans l'ordre d'ajout à l'intérieur d'une
        # section (tri stable).
        rank = {section: int(folder.split("-", 1)[0]) for section, folder in folders.items()}
        for item in sorted(self._items, key=lambda i: rank[i.section]):
            base = f"{folders[item.section]}"
            shown = item.title or item.name
            # La période dans l'index aussi : deux « Top authors » sur deux
            # périodes ne se distinguent pas autrement dans le README.
            listed = f"{shown}, {item.period}" if item.period else shown
            if isinstance(item, _Table):
                label = f"{shown} ({item.period})" if item.period else shown
                taken = section_taken.setdefault(item.section, {"contents"})
                sheet = sheet_name(f"{shown} {item.period}".strip(), taken)
                section_sheets.setdefault(item.section, []).append(
                    (sheet, shown, item.period, item.note, item.frame))
                sheets.append((f"{item.section} - {label}", item.frame))
                index.append((f"{base}/{_book(item.section)} > {sheet}", listed, item.note))
            else:
                stem = unique(f"{base}/figures/{_stem(item)}.fig")[:-4]
                for ext, data in item.files.items():
                    path = f"{stem}.{ext}"
                    used.add(path)
                    entries.append((path, data))
                exts = ", ".join(sorted(item.files))
                index.append((f"{stem}.{{{exts}}}" if len(item.files) > 1
                              else f"{stem}.{next(iter(item.files))}", listed, item.note))

        for section, rows in section_sheets.items():
            contents = pd.DataFrame([
                {"sheet": sheet, "table": title, "period": period, "filters": note,
                 "rows": len(frame)}
                for sheet, title, period, note, frame in rows])
            entries.append((f"{folders[section]}/{_book(section)}",
                            workbook_bytes([("Contents", contents)]
                                           + [(sheet, frame) for sheet, *_, frame in rows])))

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            def write(path: str, data: bytes) -> None:
                info = zipfile.ZipInfo(path, date_time=_EPOCH)
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, data)

            write("README.txt", self._readme(index).encode("utf-8"))
            if sheets:
                write("all_tables.xlsx", workbook_bytes(sheets))
            for path, data in entries:
                write(path, data)
        return buffer.getvalue()

    def save(self, path: Union[str, Path]) -> Path:
        """Écrit le ZIP et renvoie son chemin."""
        target = Path(path)
        target.write_bytes(self.to_bytes())
        return target
