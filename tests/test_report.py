"""The report: a ZIP organised by section, PNG figures at 300 dpi + SVG,
Excel tables. The same as the interface's "Add to report" basket."""

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
                pd.DataFrame({"author": ["Varela A.", "Okafor M."], "documents": [12, 9]}),
                note="years 2023-2025")
    r.add_indicators("Impact", "Collaboration", {"CAGR": 12.5, "nested": {"x": 1}},
                     definitions={"CAGR": "Compound annual growth rate"})
    r.add_figure("Corpus", "Annual production", SPEC)
    r.add_figure("Networks", "Map", {"png": b"\x89PNG\r\n\x1a\nxx", "svg": b"<svg/>"})
    return r


def test_zip_structure():
    names = zipfile.ZipFile(io.BytesIO(_report().to_bytes())).namelist()
    assert names[:2] == ["README.txt", "all_tables.xlsx"]
    assert "1-corpus/figures/annual-production.png" in names
    assert "1-corpus/figures/annual-production.svg" in names
    # One workbook per section, not one file per table.
    assert "2-actors/actors_tables.xlsx" in names
    assert "3-impact/impact_tables.xlsx" in names
    assert not any("/tables/" in n for n in names)
    assert "5-networks/figures/map.svg" in names


def test_figure_rendue_a_300_dpi():
    z = zipfile.ZipFile(io.BytesIO(_report().to_bytes()))
    png = z.read("1-corpus/figures/annual-production.png")
    width = int.from_bytes(png[16:20], "big")
    assert width == int(SPEC.width_in * 300)           # 7.2 inches at 300 dpi


def test_excel_tables_and_indicators():
    z = zipfile.ZipFile(io.BytesIO(_report().to_bytes()))
    # Sheet 1: the contents; the tables follow, in the order they were added.
    assert _cells(z.read("2-actors/actors_tables.xlsx"), sheet=2) == [
        "author", "documents", "Varela A.", "12", "Okafor M.", "9"]
    # a nested indicator is not an indicator: left out
    assert _cells(z.read("3-impact/impact_tables.xlsx"), sheet=2) == [
        "indicator", "value", "definition", "CAGR", "12.5", "Compound annual growth rate"]


def test_readme_states_the_filters_and_notes():
    readme = zipfile.ZipFile(io.BytesIO(_report().to_bytes())).read("README.txt").decode()
    assert "Filters: years: 2023-2025" in readme
    assert "Top 10 authors (years 2023-2025)" in readme
    assert readme.index("1-corpus/") < readme.index("2-actors/") < readme.index("5-networks/")


def test_deterministic():
    assert _report().to_bytes() == _report().to_bytes()


def test_empty_report_refused():
    with pytest.raises(ValueError):
        Report().to_bytes()


def test_excel_sheet_names():
    taken: set = set()
    assert sheet_name("A/B:C*?", taken) == "A B C"
    assert sheet_name("x" * 40, taken) == "x" * 31
    assert sheet_name("X" * 40, taken) == "X" * 27 + " (2)"   # Excel ignores case


def test_empty_cells_and_forbidden_characters():
    data = workbook_bytes([("S", pd.DataFrame({"a": [None, float("nan"), "x\x01y"],
                                                "b": [pd.NA, 1.5, True]}))])
    assert _cells(data) == ["a", "b", "1.5", "xy", "1"]


def test_catalog_figure(tmp_path):
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
