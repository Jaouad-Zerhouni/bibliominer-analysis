"""Format contract between the cleaning (`bibliominer-cleaning`) and the
analysis.

Everything that describes the SHAPE of the input file lives here, and
nowhere else: the day the cleaning changes a convention, this file changes
and the parsers follow.

A FILE must be the output of the Bibliominer cleaning (indexed authors,
labelled affiliations, reconciled references): `Corpus.from_csv` refuses a
raw Scopus export (`reader.check_cleaned`). The parsers keep their
fallbacks for a single cell that does not follow the convention, and for
tables assembled in memory (`Corpus.from_dataframe`).
"""

from __future__ import annotations

# --- Scopus export columns used by the analysis -----------------------------
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

#: Required column: without it, there is no knowing which corpus is meant.
REQUIRED_COLUMNS = (COL_TITLE,)

# --- Separators -------------------------------------------------------------
#: Separates the items of a list in a Scopus cell (authors, keywords,
#: affiliations, references).
LIST_SEP = ";"

#: Separates the fields INSIDE a reconciled reference:
#:     ref1 | 10.1007/... | 2016 | Lane, Moss | A guided tour of tree ensembles
REF_FIELD_SEP = "|"
REF_FIELDS = ("ref_pos", "ref_doi", "ref_year", "ref_authors", "ref_title")

#: Author index prefix written by the cleaning: "3:Lindqvist A.".
AUTHOR_INDEX_PATTERN = r"^\s*(\d+)\s*:\s*"

#: Mark written by the cleaning before an author with SEVERAL affiliations:
#: "[2 affiliations] Okafor M., subparent: ...". It is a reading aid for
#: humans; it is not part of the name.
AWA_MULTI_AFFILIATION_PATTERN = r"^\s*\[(\d+)\s+affiliations\]\s*"

# --- Labelled affiliations --------------------------------------------------
#: Labels written by the cleaning, in output order. An empty field is
#: OMITTED from the export: the position cannot be trusted, only the labels,
#: which is precisely why they exist.
AFF_LABELS = {
    "subparent": "subparent",
    "parent 1": "parent1",
    "parent 2": "parent2",
    "city": "city",
    # "site: virtual": the user ticked "No single city" during cleaning
    # (virtual laboratory, several sites). No city, by DECISION: it is not a
    # missing city.
    "site": "site",
    "region": "region",
    "country": "country",
}

#: Columns of the `affiliations` table, in the order the cleaning writes them
#: (the order is also used to split the author blocks).
AFF_COLUMNS = ("subparent", "parent1", "parent2", "city", "site", "region",
               "country")

#: Value of `site` for an affiliation without a single city.
VIRTUAL_SITE = "virtual"

#: Canonical label written by the cleaning for a researcher without affiliation.
INDEPENDENT_LABEL = "Independent researcher"
