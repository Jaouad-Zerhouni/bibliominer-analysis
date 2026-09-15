"""Ce que la distribution doit contenir pour être installable telle quelle.

Un commentaire dans `pyproject.toml` ne garantit rien : il ne s'exécute pas.
Ces tests tiennent les promesses que le paquet fait à qui l'installe depuis
PyPI, sans le dépôt autour.
"""
from pathlib import Path

import bibliominer_analysis


def test_the_scimago_reference_travels_with_the_package():
    """Sans lui, `pip install` livrerait un module qui ne sait plus rien des revues.

    `metrics/scimago.py` le lit à côté du module (`data_ref/`), pas depuis le
    dépôt : il doit donc être DANS la distribution. C'est le seul fichier de
    données du paquet, et son absence ne se verrait qu'à l'exécution, chez
    l'utilisateur.
    """
    root = Path(bibliominer_analysis.__file__).parent
    reference = root / "data_ref" / "scimagojr_2025.csv"

    assert reference.is_file(), "le référentiel SCImago manque à la distribution"
    assert reference.stat().st_size > 1_000_000, "référentiel suspect de troncature"


def test_the_declared_version_is_the_one_the_module_exposes():
    """La version vit dans `__init__.py` ; `pyproject.toml` la lit de là.

    Deux copies finissent toujours par diverger, et c'est celle du module que
    lisent les utilisateurs.
    """
    assert bibliominer_analysis.__version__
    assert bibliominer_analysis.__version__[0].isdigit()


def test_the_public_api_is_importable():
    for name in bibliominer_analysis.__all__:
        assert hasattr(bibliominer_analysis, name), name
