"""The report: the figures and tables of a study, gathered in a single ZIP.

It is the same report as the interface's "Add to report" basket: the API
calls this class, and a package user builds it in Python:

    >>> from bibliominer_analysis import Corpus, Report
    >>> corpus = Corpus.from_csv("cleaned.csv").filter(years=(2023, 2025))
    >>> report = Report(filters={"years": "2023-2025"})
    >>> report.add_table("Actors", "Top 10 authors", corpus.top_authors(10))
    >>> report.add_indicators("Impact", "Collaboration",
    ...                       corpus.collaboration_indicators())
    >>> report.add_figure("Corpus", "Annual production",
    ...                   corpus.figure_spec("production_by_year"))
    >>> report.save("report.zip")

What it contains ::

    README.txt                      the index: every file, its title, the filters
    all_tables.xlsx                 every table, one sheet each
    1-corpus/figures/<name>.png     300 dpi, print quality
    1-corpus/figures/<name>.svg     vector: scales without loss
    1-corpus/corpus_tables.xlsx     the section's tables, one sheet each,
                                    and a "Contents" sheet
    2-actors/…  3-impact/…  4-concepts/…  5-networks/…

ONE workbook per section, not one file per table: a study has about a
hundred tables, and twenty Excel files in a folder are not browsable. Its
name carries the section (``actors_tables.xlsx``): Excel refuses to open
two workbooks with the same name at once. The "Contents" sheet gives the
FULL title of each sheet, its period and its filters: a sheet name is cut
at 31 characters.

An entry computed over a PERIOD (``period="2010-2013"``) carries it in
its file name (``top-authors_2010-2013.png``) or sheet name. Two author
rankings over two periods give two files, two sheets, never a single
overwritten one.

One section per group of the interface (Corpus, Actors, Impact, Concepts,
Networks), in the order of the menu; an unknown section comes after.
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

#: The groups of the interface menu, in their order: one folder each.
SECTIONS = ("Corpus", "Actors", "Impact", "Concepts", "Networks")

#: Resolution of raster figures: the one journals require.
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
    """A single table, as a one-sheet Excel workbook.

    The interface's "Excel" export of a table: written by the package, like
    the report's workbooks, without any extra dependency."""
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
    """The workbook of a section's tables: ``actors_tables.xlsx``."""
    return f"{_slug(section).replace('-', '_')}_tables.xlsx"


def _stem(item: Union[_Table, _Figure]) -> str:
    """The file name of an entry: its name, then its period."""
    stem = _slug(item.name)
    return f"{stem}_{_slug(item.period)}" if item.period.strip() else stem


class Report:
    """A report under construction: tables, indicators and figures are added to
    it, then `save` writes the ZIP.

    ``filters``  the study's filters (period, types...), copied into the
                 README: a figure without its scope cannot be cited.
    ``formats``  the formats of each figure: PNG (``dpi``) and SVG by default.
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
        """A table (DataFrame, list of rows or dictionary): a sheet of its section's
        workbook (``<section>_tables.xlsx``), and a sheet of ``all_tables.xlsx``.

        ``title``  the title written in the README (``name`` by default);
        ``period`` the table's period, added to the file name."""
        self._items.append(_Table(section, name, _frame(table), note, title, period))
        return self

    def add_indicators(self, section: str, name: str,
                       values: Union[Mapping[str, Any], TableLike],
                       definitions: Optional[Mapping[str, str]] = None,
                       *, period: str = "") -> "Report":
        """Indicators (those of the interface tiles): a table
        ``indicator | value | definition``. A nested dictionary or a list is not
        an indicator: such values are left out."""
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
        """A figure, in each of the report's formats.

        ``figure`` can be a `FigureSpec` (rendered here, at ``dpi``), a function
        ``fmt -> bytes`` (``lambda fmt: render_network(g, fmt=fmt)``), an already
        rendered dictionary ``{"png": ..., "svg": ...}``, or the bytes of a single
        image. ``title``: the README title (``name`` by default); ``period``: the
        figure's period, added to the file name.
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
        """A figure from the interface catalogue, filed in ITS section and under ITS
        title, the equivalent of the "Add to report" button:

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

    # -- writing --------------------------------------------------------------

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
        """The report's ZIP, in memory."""
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
        # The tables of a section: one workbook, one sheet each.
        section_sheets: Dict[str, List[tuple]] = {}
        section_taken: Dict[str, set] = {}
        # In the order of the menu, and in the order they were added within a
        # section (stable sort).
        rank = {section: int(folder.split("-", 1)[0]) for section, folder in folders.items()}
        for item in sorted(self._items, key=lambda i: rank[i.section]):
            base = f"{folders[item.section]}"
            shown = item.title or item.name
            # The period in the index too: two "Top authors" over two periods cannot be
            # told apart otherwise in the README.
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
        """Writes the ZIP and returns its path."""
        target = Path(path)
        target.write_bytes(self.to_bytes())
        return target
