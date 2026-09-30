"""Les figures se rejouent à l'identique, et ne dérèglent pas la session.

Deux défauts corrigés :

  - les barres, lignes et nuages exportés en SVG ou PDF portaient la DATE
    d'export : deux exports des mêmes données différaient à chaque fois ;
  - importer le package posait `matplotlib.use("Agg")` et changeait les
    polices de TOUTE la session, dans un notebook, les `plt.show()` de
    l'utilisateur cessaient de s'afficher.
"""

import pytest

from bibliominer_analysis.figures import render_figure
from bibliominer_analysis.figures.render import FigureSpec, Series

SPEC = FigureSpec(kind="bar", categories=["2020", "2021"],
                  series=[Series(name="Documents", values=[3, 5])])


@pytest.mark.parametrize("fmt", ["png", "svg", "pdf", "jpg"])
def test_a_simple_figure_is_byte_identical_across_exports(fmt):
    assert render_figure(SPEC, fmt=fmt) == render_figure(SPEC, fmt=fmt)


def test_exports_carry_no_date():
    assert b"<dc:date>" not in render_figure(SPEC, fmt="svg")
    assert b"CreationDate" not in render_figure(SPEC, fmt="pdf")


def test_rendering_leaves_the_user_session_untouched():
    import matplotlib

    backend = matplotlib.get_backend()
    fonts = list(matplotlib.rcParams["font.sans-serif"])
    salt = matplotlib.rcParams["svg.hashsalt"]

    import bibliominer_analysis.figures.network  # noqa: F401
    render_figure(SPEC, fmt="svg")

    assert matplotlib.get_backend() == backend
    assert list(matplotlib.rcParams["font.sans-serif"]) == fonts
    assert matplotlib.rcParams["svg.hashsalt"] == salt
