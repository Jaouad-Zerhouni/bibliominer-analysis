"""`Corpus.from_csv` ne lit que le fichier exporté par le nettoyage.

Signature : les auteurs numérotés dans l'ordre (« 1:Idri A.; 2:Hosni M. »)
dans les trois colonnes d'auteurs, ce que seul l'export du nettoyage écrit.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from bibliominer_analysis import Corpus, NotCleanedError

SYNTHETIC = Path(__file__).parent / "data" / "synthetic_cleaned.csv"


def _write(tmp_path, rows):
    path = tmp_path / "corpus.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


CLEAN = {"Title": "T", "Year": "2024", "Authors": "1:Idri A.; 2:Hosni M.",
         "Author full names": "1:Idri, Ali (11); 2:Hosni, Mohamed (22)",
         "Author(s) ID": "1:11; 2:22"}


def test_the_cleaned_export_loads():
    assert Corpus.from_csv(SYNTHETIC).summary()["documents"] == 180


def test_numbered_authors_load(tmp_path):
    path = _write(tmp_path, [CLEAN, dict(CLEAN, Authors="1:Solo S.",
                                         **{"Author full names": "", "Author(s) ID": ""})])
    assert Corpus.from_csv(path).summary()["documents"] == 2


def test_a_raw_scopus_export_is_refused(tmp_path):
    raw = {"Title": "T", "Authors": "Idri A.; Hosni M.",
           "Author full names": "Idri, Ali (11); Hosni, Mohamed (22)", "Author(s) ID": "11; 22"}
    with pytest.raises(NotCleanedError, match="not been cleaned with Bibliominer") as err:
        Corpus.from_csv(_write(tmp_path, [raw, raw]))
    message = str(err.value)
    assert "6 author cell(s) out of 6 are not numbered" in message
    assert 'line 2, column "Authors": "Idri A.; Hosni M."' in message
    assert "Clean the Scopus export first" in message


def test_one_unnumbered_line_is_enough(tmp_path):
    """Un fichier nettoyé retouché à la main n'est plus le livrable."""
    edited = dict(CLEAN, Authors="Added B.")
    with pytest.raises(NotCleanedError, match='line 3, column "Authors": "Added B."'):
        Corpus.from_csv(_write(tmp_path, [CLEAN, edited]))


@pytest.mark.parametrize("authors", ["1:Idri A.; 3:Hosni M.", "2:Idri A.", "1:Idri A.; Hosni M."])
def test_the_numbers_must_run_1_2_3(tmp_path, authors):
    with pytest.raises(NotCleanedError):
        Corpus.from_csv(_write(tmp_path, [dict(CLEAN, Authors=authors)]))


def test_a_file_without_authors_is_refused(tmp_path):
    with pytest.raises(NotCleanedError, match='no "Authors" column filled in'):
        Corpus.from_csv(_write(tmp_path, [{"Title": "T", "Year": "2024"}]))


def test_the_error_is_a_value_error(tmp_path):
    """Les appelants qui attrapaient `ValueError` (l'API) le reçoivent."""
    with pytest.raises(ValueError):
        Corpus.from_csv(_write(tmp_path, [{"Title": "T", "Authors": "Doe J."}]))
