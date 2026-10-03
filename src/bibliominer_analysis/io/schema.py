"""Contrat de format entre le nettoyage (`bibliominer-cleaning`) et l'analyse.

Tout ce qui décrit la FORME du fichier d'entrée vit ici, et nulle part
ailleurs : le jour où le cleaning change une convention, on modifie ce
fichier et les parseurs suivent.

Un FICHIER doit être la sortie du nettoyage Bibliominer (auteurs indexés,
affiliations étiquetées, références réconciliées) : `Corpus.from_csv` refuse
un export Scopus brut (`reader.check_cleaned`). Les parseurs gardent leurs
replis pour une cellule isolée qui ne suit pas la convention, et pour les
tableaux assemblés en mémoire (`Corpus.from_dataframe`).
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

#: Colonne indispensable, sans elle on ne sait pas de quel corpus on parle.
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

#: Marque posée par le cleaning devant un auteur à PLUSIEURS affiliations :
#: « [2 affiliations] Hosni M., subparent: … ». C'est une aide de lecture
#: humaine ; elle ne fait pas partie du nom.
AWA_MULTI_AFFILIATION_PATTERN = r"^\s*\[(\d+)\s+affiliations\]\s*"

# --- Affiliations étiquetées ------------------------------------------------
#: Libellés écrits par le cleaning, dans l'ordre de sortie. Un champ vide est
#: OMIS à l'export : on ne peut donc pas se fier à la position, seulement aux
#: libellés, c'est précisément pourquoi ils existent.
AFF_LABELS = {
    "subparent": "subparent",
    "parent 1": "parent1",
    "parent 2": "parent2",
    "city": "city",
    # « site: virtual » : l'utilisateur a coché « No single city » au
    # nettoyage (laboratoire virtuel, plusieurs sites). Pas de ville, par
    # DÉCISION, ce n'est pas une ville manquante.
    "site": "site",
    "region": "region",
    "country": "country",
}

#: Colonnes de la table `affiliations`, dans l'ordre d'écriture du cleaning
#: (l'ordre sert aussi à découper les blocs d'auteurs).
AFF_COLUMNS = ("subparent", "parent1", "parent2", "city", "site", "region",
               "country")

#: Valeur de `site` pour une affiliation sans ville unique.
VIRTUAL_SITE = "virtual"

#: Libellé canonique posé par le cleaning sur un chercheur sans rattachement.
INDEPENDENT_LABEL = "Independent researcher"
