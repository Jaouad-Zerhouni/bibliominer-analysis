"""La MÊME palette que l'interface — recopiée, pas réinventée.

Source de vérité : `analysis_service/frontend/src/theme/palette.ts`, validée
par `scripts/validate_palette.js` (bande de luminosité, plancher de chroma,
séparation daltonisme). Une figure exportée qui utiliserait d'autres teintes
que celles vues à l'écran romprait le lien entre ce que l'utilisateur a lu et
ce qu'il publie — et re-court le script de validation ici demanderait une
dépendance Node dans un paquet Python. Recopiée, avec le fichier source cité,
pour qu'une modification de l'un rappelle de vérifier l'autre.
"""

from __future__ import annotations

from typing import Dict, List

CATEGORICAL_LIGHT: List[str] = [
    "#2a78d6",  # 1 bleu
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 jaune
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 vert
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 rouge
]

CATEGORICAL_DARK: List[str] = [
    "#3987e5", "#d95926", "#199e70", "#c98500",
    "#d55181", "#008300", "#9085e9", "#e66767",
]

CHROME: Dict[str, Dict[str, str]] = {
    "light": {
        "surface": "#fcfcfb",
        "text_primary": "#0b0b0b",
        "text_secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
    },
    "dark": {
        "surface": "#1a1a19",
        "text_primary": "#ffffff",
        "text_secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
    },
}


def categorical(mode: str) -> List[str]:
    return list(CATEGORICAL_DARK if mode == "dark" else CATEGORICAL_LIGHT)


def chrome(mode: str) -> Dict[str, str]:
    return CHROME["dark"] if mode == "dark" else CHROME["light"]


#: Réglages matplotlib appliqués LE TEMPS D'UN RENDU, jamais à la session.
#:
#: Une PILE de repli, jamais un seul nom : "Inter" peut manquer sans que
#: matplotlib échoue -- il descend la liste jusqu'à une police installée,
#: DejaVu Sans en dernier recours (toujours livrée avec matplotlib).
#:
#: Ils étaient posés sur `plt.rcParams` à l'import, avec `matplotlib.use("Agg")` :
#: importer le package changeait les polices et le moteur d'affichage de TOUTE
#: la session. Dans un notebook, les `plt.show()` de l'utilisateur cessaient
#: alors de s'afficher.
#:
#: `svg.hashsalt` : en SVG, matplotlib tire au hasard les identifiants de ses
#: éléments (`<g id="...">`). On fixe le grain, sans quoi deux rendus
#: identiques donneraient deux fichiers différents.
RENDER_RC = {
    "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "svg.hashsalt": "bibliominer-analysis",
}

#: matplotlib horodate ses fichiers : deux rendus des MÊMES données donnaient
#: deux octets différents. On retire la date (et la version du logiciel, qui
#: changerait le fichier à chaque mise à jour de matplotlib) : une figure
#: publiée doit pouvoir se rejouer et se comparer octet pour octet. JPEG
#: n'accepte pas de métadonnées dans matplotlib : rien à retirer.
#:
#: Défaut corrigé : seuls les RÉSEAUX en profitaient. Les barres, lignes et
#: nuages exportés en SVG ou PDF portaient la date, et changeaient à chaque
#: export.
NO_TIMESTAMP = {
    "png": {"Software": None, "Date": None},
    "svg": {"Date": None},
    "pdf": {"CreationDate": None},
}
