"""Figures PUBLIABLES, générées par Python (matplotlib) — jamais capturées
depuis un rendu de navigateur.

Une capture de graphique ECharts dépend de l'état d'affichage au moment du
clic : thème clair/sombre, taille de fenêtre, police effectivement chargée.
Une figure d'article doit être reproductible à l'identique depuis les
mêmes données, ce qu'un script Python garantit et qu'une capture d'écran ne
garantit jamais.

`render_figure` est le point d'entrée unique : le TYPE de graphique décide
de la fonction interne appelée, mais l'appelant (le routeur FastAPI) n'a
qu'une seule fonction à connaître.
"""

from .render import FigureError, render_figure

__all__ = ["render_figure", "FigureError"]
