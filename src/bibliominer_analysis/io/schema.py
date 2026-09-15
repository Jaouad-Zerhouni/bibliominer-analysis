"""Contrat de format entre `cleaning_service` et l'analyse.

Tout ce qui décrit la FORME du fichier d'entrée vit ici, et nulle part
ailleurs : le jour où le cleaning change une convention, on modifie ce
fichier et les parseurs suivent.

Le package accepte DEUX niveaux de fichier :

  - « bibliominer » : sortie de `cleaning_service` (auteurs indexés,
    affiliations étiquetées, références réconciliées) ;
  - « scopus brut » : un export Scopus non nettoyé.

Les parseurs détectent le niveau tout seuls et se replient sans se plaindre :
un corpus brut donne simplement des tables moins riches.
"""

from __future__ import annotations

# --- Colonnes de l'export Scopus utilisées par l'analyse --------------------
COL_AUTHORS = "Authors"
COL_AUTHOR_FULL = "Author full names"
COL_AUTHOR_IDS = "Author(s) ID"
COL_AUTHORS_AFF = "Authors with affiliations"
COL_AFFILIATIONS = "Affiliations"
COL_TITLE = "Title"
COL_YEAR = "Year"
COL_SOURCE = "Source title"
COL_SOURCE_ABBR = "Abbreviated Source Title"
COL_CITED_BY = "Cited by"
COL_DOI = "DOI"
COL_LINK = "Link"
COL_ABSTRACT = "Abstract"
COL_AUTHOR_KW = "Author Keywords"
COL_INDEX_KW = "Index Keywords"
COL_REFERENCES = "References"
COL_DOC_TYPE = "Document Type"
COL_LANGUAGE = "Language of Original Document"
COL_OPEN_ACCESS = "Open Access"
COL_ISSN = "ISSN"
COL_ISBN = "ISBN"
COL_PUBLISHER = "Publisher"
COL_EID = "EID"

#: Colonne indispensable — sans elle on ne sait pas de quel corpus on parle.
REQUIRED_COLUMNS = (COL_TITLE,)

# --- Séparateurs ------------------------------------------------------------
#: Sépare les éléments d'une liste dans une cellule Scopus (auteurs, mots-clés,
#: affiliations, références).
LIST_SEP = ";"

#: Sépare les champs À L'INTÉRIEUR d'une référence réconciliée :
#:     ref1 | 10.1007/... | 2016 | Biau, Scornet | A random forest guided tour
REF_FIELD_SEP = "|"
REF_FIELDS = ("ref_pos", "ref_doi", "ref_year", "ref_authors", "ref_title")

#: Préfixe d'indexation des auteurs posé par le cleaning : « 3:Abran A. ».
AUTHOR_INDEX_PATTERN = r"^\s*(\d+)\s*:\s*"

# --- Affiliations étiquetées ------------------------------------------------
#: Libellés écrits par le cleaning, dans l'ordre de sortie. Un champ vide est
#: OMIS à l'export : on ne peut donc pas se fier à la position, seulement aux
#: libellés — c'est précisément pourquoi ils existent.
AFF_LABELS = {
    "subparent": "subparent",
    "parent 1": "parent1",
    "parent 2": "parent2",
    "city": "city",
    "region": "region",
    "country": "country",
}

#: Colonnes de la table `affiliations`, dans l'ordre.
AFF_COLUMNS = ("subparent", "parent1", "parent2", "city", "region", "country")

#: Libellé canonique posé par le cleaning sur un chercheur sans rattachement.
INDEPENDENT_LABEL = "Independent researcher"
