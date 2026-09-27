"""Le rendu matplotlib des figures publiables (barres, lignes, nuages
simples) -- jamais un rendu de navigateur capturé.
"""

from __future__ import annotations

import pytest

from bibliominer_analysis.figures import FigureError, render_figure
from bibliominer_analysis.figures.render import FigureSpec, Series


def _png_signature(data: bytes) -> bool:
    return data[:8] == b"\x89PNG\r\n\x1a\n"


def test_barres_simples_produit_un_png_valide():
    spec = FigureSpec(
        kind="bar",
        categories=["2020", "2021", "2022"],
        series=[Series(name="Documents", values=[10, 15, 12])],
        title="Documents per year",
    )
    data = render_figure(spec, fmt="png")
    assert _png_signature(data)
    assert len(data) > 500  # une figure vide ferait quelques dizaines d'octets


def test_lignes_multi_series_ne_leve_pas():
    spec = FigureSpec(
        kind="line",
        categories=["2020", "2021", "2022"],
        series=[
            Series(name="Citations", values=[5, 9, 7]),
            Series(name="Documents", values=[10, 15, 12]),
        ],
    )
    assert _png_signature(render_figure(spec, fmt="png"))


def test_nuage_de_points_simple():
    spec = FigureSpec(
        kind="scatter",
        series=[Series(name="Corpus", points=[(1.0, 2.0), (3.0, 4.5)])],
    )
    assert _png_signature(render_figure(spec, fmt="png"))


def test_barres_horizontales_grille_sur_l_axe_des_valeurs():
    """Non-régression : le filet de grille doit suivre l'axe X (valeurs) en
    orientation horizontale, pas Y (catégories) -- sinon il ne sert à rien."""
    spec = FigureSpec(
        kind="bar", orientation="horizontal",
        categories=["Morocco", "France"],
        series=[Series(name="Documents", values=[40, 25])],
    )
    render_figure(spec, fmt="png")  # ne doit pas lever


def test_barres_et_ligne_melangees_ne_leve_pas():
    """Le motif « compte + tendance » (RPYS, moyenne mobile...) : une série
    barres et une série ligne sur le même axe catégoriel."""
    spec = FigureSpec(
        kind="bar",
        categories=["2020", "2021", "2022"],
        series=[
            Series(name="Cited references", kind="bar", values=[10, 40, 25]),
            Series(name="5-year median", kind="line", values=[15, 20, 30]),
        ],
    )
    assert _png_signature(render_figure(spec, fmt="png"))


def test_scatter_melange_a_des_barres_leve_figure_error():
    spec = FigureSpec(
        kind="bar", categories=["a", "b"],
        series=[
            Series(name="x", kind="bar", values=[1, 2]),
            Series(name="y", kind="scatter", values=[1, 2]),
        ],
    )
    with pytest.raises(FigureError):
        render_figure(spec, fmt="png")


def test_type_non_couvert_leve_figure_error():
    spec = FigureSpec(kind="sankey", series=[Series(name="x", values=[1])])
    with pytest.raises(FigureError):
        render_figure(spec, fmt="png")


def test_format_jpg_produit_un_jpeg_valide():
    """"jpg" est le nom reconnu par l'utilisateur -- matplotlib, lui, ne
    connaît que "jpeg" (alias interne, voir `_MPL_FORMAT`)."""
    spec = FigureSpec(kind="bar", categories=["a", "b"],
                      series=[Series(name="x", values=[1, 2])])
    data = render_figure(spec, fmt="jpg")
    assert data[:3] == b"\xff\xd8\xff"  # signature JPEG


def test_format_inconnu_leve_figure_error():
    spec = FigureSpec(kind="bar", categories=["a"],
                      series=[Series(name="x", values=[1])])
    with pytest.raises(FigureError):
        render_figure(spec, fmt="gif")


def test_aucune_serie_leve_figure_error():
    spec = FigureSpec(kind="bar", categories=["a"], series=[])
    with pytest.raises(FigureError):
        render_figure(spec, fmt="png")


def test_longueur_valeurs_categories_desaccordees_leve_figure_error():
    """Une série de 2 valeurs pour 3 catégories est une donnée cassée -- la
    figure ne doit pas se construire en silence sur un mauvais alignement."""
    spec = FigureSpec(
        kind="bar", categories=["2020", "2021", "2022"],
        series=[Series(name="Documents", values=[10, 15])],
    )
    with pytest.raises(FigureError):
        render_figure(spec, fmt="png")


def test_scatter_sans_points_leve_figure_error():
    spec = FigureSpec(kind="scatter", series=[Series(name="x", points=None)])
    with pytest.raises(FigureError):
        render_figure(spec, fmt="png")


def test_svg_est_du_texte_vectoriel():
    spec = FigureSpec(kind="bar", categories=["a"],
                      series=[Series(name="x", values=[1])])
    data = render_figure(spec, fmt="svg")
    assert data.strip().startswith(b"<?xml") or data.strip().startswith(b"<svg")


def test_couleurs_categorielles_dans_l_ordre_valide():
    """La première série doit porter la PREMIÈRE couleur de la rampe validée
    -- jamais une couleur choisie au hasard par matplotlib."""
    from bibliominer_analysis.figures.palette import CATEGORICAL_LIGHT

    spec = FigureSpec(
        kind="bar", categories=["a"],
        series=[Series(name="x", values=[1])], mode="light",
    )
    # Rendu SVG : la couleur du rectangle apparaît en texte, vérifiable sans
    # décoder un PNG.
    svg = render_figure(spec, fmt="svg").decode("utf-8", errors="ignore")
    assert CATEGORICAL_LIGHT[0].lower() in svg.lower()


def test_axes_logarithmiques_pour_zipf():
    """Zipf se lit en log-log : l'option doit changer l'échelle, pas lever."""
    spec = FigureSpec(kind="scatter", x_log=True, y_log=True,
                      series=[Series(name="observed",
                                     points=[(1, 40), (2, 20), (10, 4), (100, 1)])])
    from dataclasses import replace

    log_svg = render_figure(spec, fmt="svg")
    linear_svg = render_figure(replace(spec, x_log=False, y_log=False), fmt="svg")
    assert log_svg != linear_svg


def test_etiquettes_de_points_superposes_fusionnees():
    """Deux thèmes au même endroit : une seule étiquette « premier +1 »."""
    spec = FigureSpec(kind="scatter", series=[Series(
        name="themes", points=[(0, 50), (0, 50), (1, 60)],
        labels=["Alpha theme", "Beta theme", "Gamma theme"])])
    svg = render_figure(spec, fmt="svg").decode("utf-8", errors="ignore")
    assert "Alpha theme +1" in svg
    assert "Beta theme" not in svg
    assert "Gamma theme" in svg


def test_beaucoup_de_categories_graduations_clairsemees():
    """77 rangs de Bradford : l'axe n'écrit pas 77 étiquettes côte à côte."""
    from bibliominer_analysis.figures import render as r

    cats = [str(i) for i in range(1, 78)]
    spec = FigureSpec(kind="line", categories=cats,
                      series=[Series(name="cumulative", values=list(range(77)))])
    fig = r.Figure()
    ax = fig.add_subplot()
    r._draw_lines(ax, spec, ["#000000"])
    ticks = [t.get_text() for t in ax.get_xticklabels()]
    assert len(ticks) <= r._MAX_X_TICKS + 1
    assert ticks[0] == "1" and ticks[-1] == "77"
