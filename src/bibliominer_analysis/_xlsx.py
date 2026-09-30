"""Un classeur Excel (.xlsx), écrit sans aucune dépendance.

Pourquoi ne pas passer par ``DataFrame.to_excel`` ? Parce qu'il exige
openpyxl ou xlsxwriter : une dépendance de plus pour tout utilisateur du
paquet, rien que pour écrire des tableaux. Un .xlsx n'est qu'un ZIP de
quelques fichiers XML ; on en écrit le strict nécessaire, des feuilles,
une ligne d'en-tête en gras et figée, des largeurs de colonnes lisibles.

Le fichier est DÉTERMINISTE : aucune date de création, des entrées ZIP
datées du 1er janvier 1980. Deux exports des mêmes tableaux donnent les
mêmes octets.
"""

from __future__ import annotations

import io
import math
import numbers
import re
import zipfile
from typing import Any, List, Sequence, Tuple
from xml.sax.saxutils import escape

import pandas as pd

__all__ = ["workbook_bytes", "sheet_name"]

_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_PKG = "http://schemas.openxmlformats.org/package/2006/relationships"
_HEAD = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
_EPOCH = (1980, 1, 1, 0, 0, 0)

#: Caractères interdits dans un XML 1.0, un titre copié d'un PDF en contient.
_ILLEGAL = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f￾￿]")
#: Excel refuse ces caractères dans un nom de feuille.
_SHEET_FORBIDDEN = re.compile(r"[\[\]:*?/\\]")
#: Limite d'Excel pour le texte d'une cellule.
_CELL_MAX = 32767


def sheet_name(name: str, taken: set) -> str:
    """Un nom de feuille valide et unique : 31 caractères au plus, sans
    ``[]:*?/\\``, jamais deux fois le même (Excel ignore la casse)."""
    base = _SHEET_FORBIDDEN.sub(" ", _ILLEGAL.sub("", str(name))).strip().strip("'")
    base = (base or "Sheet")[:31]
    candidate, n = base, 2
    while candidate.lower() in taken:
        suffix = f" ({n})"
        candidate = base[:31 - len(suffix)] + suffix
        n += 1
    taken.add(candidate.lower())
    return candidate


def _column(index: int) -> str:
    """0 -> A, 25 -> Z, 26 -> AA."""
    letters = ""
    index += 1
    while index:
        index, rest = divmod(index - 1, 26)
        letters = chr(65 + rest) + letters
    return letters


def _missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):          # listes, tableaux : pas « manquants »
        return False


def _cell(ref: str, value: Any, style: int = 0) -> str:
    if _missing(value):
        return ""
    s = f' s="{style}"' if style else ""
    if isinstance(value, bool):
        return f'<c r="{ref}" t="b"{s}><v>{int(value)}</v></c>'
    if isinstance(value, numbers.Number) and not isinstance(value, complex):
        number = float(value)
        if math.isinf(number):
            value = "inf" if number > 0 else "-inf"
        else:
            text = str(int(number)) if number.is_integer() and abs(number) < 1e15 else repr(number)
            return f'<c r="{ref}"{s}><v>{text}</v></c>'
    text = _ILLEGAL.sub("", str(value))[:_CELL_MAX]
    return (f'<c r="{ref}" t="inlineStr"{s}><is><t xml:space="preserve">'
            f'{escape(text)}</t></is></c>')


def _width(values: List[Any]) -> float:
    longest = max((len(str(v)) for v in values if not _missing(v)), default=0)
    return float(min(max(longest + 2, 8), 60))


def _sheet_xml(frame: pd.DataFrame) -> str:
    columns = [str(c) for c in frame.columns]
    rows = frame.itertuples(index=False, name=None)
    widths = [_width([name] + frame.iloc[:, k].head(500).tolist())
              for k, name in enumerate(columns)]

    out = [_HEAD, f'<worksheet xmlns="{_MAIN}">',
           # En-tête figé : il reste visible quand on fait défiler 2 000 lignes.
           '<sheetViews><sheetView workbookViewId="0">'
           '<pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>'
           '</sheetView></sheetViews>']
    if columns:
        out.append("<cols>" + "".join(
            f'<col min="{k + 1}" max="{k + 1}" width="{w}" customWidth="1"/>'
            for k, w in enumerate(widths)) + "</cols>")
    out.append("<sheetData>")
    out.append('<row r="1">' + "".join(
        _cell(f"{_column(k)}1", name, style=1) for k, name in enumerate(columns)) + "</row>")
    for r, row in enumerate(rows, start=2):
        out.append(f'<row r="{r}">' + "".join(
            _cell(f"{_column(k)}{r}", value) for k, value in enumerate(row)) + "</row>")
    out.append("</sheetData></worksheet>")
    return "".join(out)


_STYLES = (
    _HEAD + f'<styleSheet xmlns="{_MAIN}">'
    '<fonts count="2"><font><sz val="11"/><name val="Calibri"/></font>'
    '<font><b/><sz val="11"/><name val="Calibri"/></font></fonts>'
    '<fills count="2"><fill><patternFill patternType="none"/></fill>'
    '<fill><patternFill patternType="gray125"/></fill></fills>'
    '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
    '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
    '<cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
    '<xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/></cellXfs>'
    '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
    '</styleSheet>'
)


def workbook_bytes(sheets: Sequence[Tuple[str, pd.DataFrame]]) -> bytes:
    """Un classeur : une feuille par ``(nom, tableau)``, dans cet ordre."""
    if not sheets:
        sheets = [("Sheet", pd.DataFrame())]
    taken: set = set()
    named = [(sheet_name(name, taken), frame) for name, frame in sheets]

    content_types = (
        _HEAD + '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/'
        'vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/styles.xml" ContentType="application/'
        'vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        + "".join(
            f'<Override PartName="/xl/worksheets/sheet{k}.xml" ContentType="application/'
            'vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            for k in range(1, len(named) + 1))
        + "</Types>")
    root_rels = (
        _HEAD + f'<Relationships xmlns="{_PKG}">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/'
        '2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
    workbook = (
        _HEAD + f'<workbook xmlns="{_MAIN}" xmlns:r="{_REL}"><sheets>'
        + "".join(f'<sheet name="{escape(name, {chr(34): "&quot;"})}" sheetId="{k}" r:id="rId{k}"/>'
                  for k, (name, _) in enumerate(named, start=1))
        + "</sheets></workbook>")
    workbook_rels = (
        _HEAD + f'<Relationships xmlns="{_PKG}">'
        + "".join(
            f'<Relationship Id="rId{k}" Type="http://schemas.openxmlformats.org/officeDocument/'
            f'2006/relationships/worksheet" Target="worksheets/sheet{k}.xml"/>'
            for k in range(1, len(named) + 1))
        + f'<Relationship Id="rId{len(named) + 1}" Type="http://schemas.openxmlformats.org/'
        'officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>')

    parts = [("[Content_Types].xml", content_types), ("_rels/.rels", root_rels),
             ("xl/workbook.xml", workbook), ("xl/_rels/workbook.xml.rels", workbook_rels),
             ("xl/styles.xml", _STYLES)]
    parts += [(f"xl/worksheets/sheet{k}.xml", _sheet_xml(frame))
              for k, (_, frame) in enumerate(named, start=1)]

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path, text in parts:
            info = zipfile.ZipInfo(path, date_time=_EPOCH)
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, text.encode("utf-8"))
    return buffer.getvalue()
