"""Figures exported or added to the report read ON THEIR OWN.

Reported by the user: a single colour, no title, no choice of form. Now:
title and subtitle (the period) written on the figure, a gradient or one
hue per category, values at the end of the bars, and the form of choice:
bars, lollipops, areas, pie, donut. And a report entry computed over a
period carries that period in its file name.
"""
from __future__ import annotations

import io
import zipfile

import pytest

from bibliominer_analysis import Report
from bibliominer_analysis.figures.render import (
    FigureError, FigureSpec, Series, _gradient, _spread, render_figure,
)

CATS = ["Idri A.", "Hosni M.", "Abran A.", "Toval A.", "Nassif A.B.",
        "Azzeh M.", "Ouhbi S.", "Benali M.", "Kharbouch A.", "Garcia G."]
VALUES = [42, 31, 27, 19, 17, 15, 12, 11, 9, 8]


def _spec(kind, **extra):
    base = dict(kind=kind, categories=CATS, series=[Series("Documents", VALUES)],
                title="Top authors", x_label="Documents", y_label="Author",
                show_title=True, subtitle="Years 2010-2013")
    base.update(extra)
    return FigureSpec(**base)


@pytest.mark.parametrize("kind, extra", [
    ("bar", {"palette": "gradient", "value_labels": True}),
    ("bar", {"palette": "category", "orientation": "horizontal"}),
    ("lollipop", {"orientation": "horizontal", "color": "#1baf7a"}),
    ("lollipop", {"value_labels": True}),
    ("line", {"value_labels": True}),
    ("area", {"color": "#4a3aa7"}),
    ("pie", {"value_labels": True}),
    ("donut", {}),
])
def test_every_form_renders_in_every_format(kind, extra):
    spec = _spec(kind, **extra)
    assert render_figure(spec, "png").startswith(b"\x89PNG")
    assert b"<svg" in render_figure(spec, "svg")[:400]
    assert render_figure(spec, "pdf").startswith(b"%PDF")


def test_a_styled_figure_is_still_reproducible_byte_for_byte():
    spec = _spec("bar", palette="gradient", value_labels=True)
    assert render_figure(spec, "svg") == render_figure(spec, "svg")


def test_the_title_is_written_only_on_request():
    titled = render_figure(_spec("bar"), "svg")
    plain = render_figure(_spec("bar", show_title=False), "svg")
    assert b"Top authors" in titled and b"Years 2010-2013" in titled
    assert b"Years 2010-2013" not in plain


def test_the_gradient_runs_from_light_to_the_chosen_colour():
    colours = _gradient("#2a78d6", [1, 5, 10], "#ffffff")
    assert colours[-1] == "#2a78d6"
    assert colours[0] != colours[-1]


def test_more_categories_than_colours_never_repeat_a_colour():
    colours = _spread(["#2a78d6", "#eb6834", "#1baf7a"], 7)
    assert len(set(colours)) == 7


def test_a_pie_refuses_what_it_cannot_show():
    with pytest.raises(FigureError, match="single series"):
        render_figure(_spec("pie", series=[Series("a", VALUES), Series("b", VALUES)]))
    with pytest.raises(FigureError, match="positive"):
        render_figure(_spec("pie", series=[Series("a", [-1] + VALUES[1:])]))


def test_an_unknown_palette_is_refused():
    with pytest.raises(FigureError, match="palette"):
        render_figure(_spec("bar", palette="rainbow"))


def _names(report):
    return zipfile.ZipFile(io.BytesIO(report.to_bytes())).namelist()


def test_the_period_is_in_the_file_name_and_two_periods_make_two_files():
    report = Report()
    report.add_figure("Actors", "top-authors", _spec("bar"), title="Top authors",
                      period="2010-2013")
    report.add_figure("Actors", "top-authors", _spec("bar"), title="Top authors",
                      period="2014-2016")
    report.add_table("Actors", "top-authors", [{"author": "Idri A.", "documents": 42}],
                     period="2010-2013")
    names = _names(report)
    assert "2-actors/figures/top-authors_2010-2013.png" in names
    assert "2-actors/figures/top-authors_2014-2016.png" in names
    book = zipfile.ZipFile(io.BytesIO(report.to_bytes())).read("2-actors/actors_tables.xlsx")
    contents = zipfile.ZipFile(io.BytesIO(book)).read("xl/worksheets/sheet1.xml").decode()
    assert "top-authors 2010-2013" in contents
    readme = zipfile.ZipFile(io.BytesIO(report.to_bytes())).read("README.txt").decode()
    assert "Top authors" in readme


def test_without_a_period_the_file_keeps_its_plain_name():
    report = Report().add_figure("Corpus", "documents-per-year", _spec("bar"))
    assert "1-corpus/figures/documents-per-year.png" in _names(report)
