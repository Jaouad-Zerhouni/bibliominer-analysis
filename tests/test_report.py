"""Le rapport : un ZIP rangé par section, figures PNG 300 dpi + SVG, tableaux
Excel. Le même que celui du panier « Add to report » de l'interface."""

from __future__ import annotations

import io
import re
import zipfile
from datetime import datetime

import pandas as pd
import pytest

from bibliominer_analysis import Report
from bibliominer_analysis._xlsx import sheet_name, workbook_bytes
from bibliominer_analysis.figures.render import FigureSpec, Series

SPEC = FigureSpec(kind="bar", categories=["2020", "2021"],
                  series=[Series(name="documents", values=[3, 5])],
                  x_label="Year", y_label="Documents")


def _cells(data: bytes, sheet: int = 1):
    xml = zipfile.ZipFile(io.BytesIO(data)).read(f"xl/worksheets/sheet{sheet}.xml").decode()
    return [a or b for a, b in re.findall(
        r'<t xml:space="preserve">([^<]*)</t>|<v>([^<]*)</v>', xml)]


def _report():
    r = Report(filters={"years": "2023-2025"}, created=datetime(2026, 1, 1))
    r.add_table("Actors", "Top 10 authors",
                pd.DataFrame({"author": ["Idri A.", "Hosni M."], "documents": [12, 9]}),
                note="years 2023-2025")
    r.add_indicators("Impact", "Collaboration", {"CAGR": 12.5, "nested": {"x": 1}},
                     definitions={"CAGR": "Compound annual growth rate"})
    r.add_figure("Corpus", "Annual production", SPEC)
    r.add_figure("Networks", "Map", {"png": b"\x89PNG\r\n\x1a\nxx", "svg": b"<svg/>"})
    return r


def test_structure_du_zip():
    names = zipfile.ZipFile(io.BytesIO(_report().to_bytes())).namelist()
    assert names[:2] == ["README.txt", "all_tables.xlsx"]
    assert "1-corpus/figures/annual-production.png" in names
    assert "1-corpus/figures/annual-production.svg" in names
    assert "2-actors/tables/top-10-authors.xlsx" in names
    assert "3-impact/tables/collaboration.xlsx" in names
    assert "5-networks/figures/map.svg" in names


def test_figure_rendue_a_300_dpi():
    z = zipfile.ZipFile(io.BytesIO(_report().to_bytes()))
    png = z.read("1-corpus/figures/annual-production.png")
    width = int.from_bytes(png[16:20], "big")
    assert width == int(SPEC.width_in * 300)           # 7,2 pouces à 300 dpi


def test_tableaux_excel_et_indicateurs():
    z = zipfile.ZipFile(io.BytesIO(_report().to_bytes()))
    assert _cells(z.read("2-actors/tables/top-10-authors.xlsx")) == [
        "author", "documents", "Idri A.", "12", "Hosni M.", "9"]
    # un indicateur imbriqué n'est pas un indicateur : écarté
    assert _cells(z.read("3-impact/tables/collaboration.xlsx")) == [
        "indicator", "value", "definition", "CAGR", "12.5", "Compound annual growth rate"]


def test_readme_dit_les_filtres_et_les_notes():
    readme = zipfile.ZipFile(io.BytesIO(_report().to_bytes())).read("README.txt").decode()
    assert "Filters: years: 2023-2025" in readme
    assert "Top 10 authors (years 2023-2025)" in readme
    assert readme.index("1-corpus/") < readme.index("2-actors/") < readme.index("5-networks/")


def test_deterministe():
    assert _report().to_bytes() == _report().to_bytes()


def test_rapport_vide_refuse():
    with pytest.raises(ValueError):
        Report().to_bytes()


def test_noms_de_feuilles_excel():
    taken: set = set()
    assert sheet_name("A/B:C*?", taken) == "A B C"
    assert sheet_name("x" * 40, taken) == "x" * 31
    assert sheet_name("X" * 40, taken) == "X" * 27 + " (2)"   # Excel ignore la casse


def test_cellules_vides_et_caracteres_interdits():
    data = workbook_bytes([("S", pd.DataFrame({"a": [None, float("nan"), "x\x01y"],
                                                "b": [pd.NA, 1.5, True]}))])
    assert _cells(data) == ["a", "b", "1.5", "xy", "1"]


def test_figure_du_catalogue(tmp_path):
    from bibliominer_analysis import Corpus
    c = Corpus.from_dataframe(pd.DataFrame([{
        "Title": "T%d" % i, "Year": str(2020 + i), "Cited by": "1", "EID": "e%d" % i,
        "Source title": "J", "Document Type": "Article", "Authors": "A.; B.",
        "Affiliations": "parent 1: U, city: X, country: Morocco"} for i in range(3)]))
    r = Report().add_corpus_figure(c, "documents-per-year")
    names = zipfile.ZipFile(io.BytesIO(r.to_bytes())).namelist()
    assert "1-corpus/figures/documents-per-year.png" in names
    path = r.save(tmp_path / "report.zip")
    assert zipfile.is_zipfile(path)
