"""Figures PUBLIABLES, générées par Python (matplotlib), jamais capturées
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
from .world import atlas_name, render_world_map
from .charts import (render_dendrogram, render_density_map, render_thematic_evolution,
                     render_three_fields, render_treemap, render_word_cloud)
from .network import render_network

__all__ = ["render_figure", "FigureError", "render_network", "render_world_map",
           "atlas_name", "render_treemap", "render_word_cloud", "render_three_fields",
           "render_thematic_evolution", "render_dendrogram", "render_density_map"]
