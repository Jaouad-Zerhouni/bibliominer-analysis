"""`Corpus.from_csv` only reads the file exported by the cleaning.

Signature: the authors numbered in order ("1:Varela A.; 2:Okafor M.") in the
three author columns, which only the cleaning export writes.
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


CLEAN = {"Title": "T", "Year": "2024", "Authors": "1:Varela A.; 2:Okafor M.",
         "Author full names": "1:Varela, Ana (11); 2:Okafor, Maya (22)",
         "Author(s) ID": "1:11; 2:22"}


def test_the_cleaned_export_loads():
    assert Corpus.from_csv(SYNTHETIC).summary()["documents"] == 180


def test_numbered_authors_load(tmp_path):
    path = _write(tmp_path, [CLEAN, dict(CLEAN, Authors="1:Solo S.",
                                         **{"Author full names": "", "Author(s) ID": ""})])
    assert Corpus.from_csv(path).summary()["documents"] == 2


def test_a_raw_scopus_export_is_refused(tmp_path):
    raw = {"Title": "T", "Authors": "Varela A.; Okafor M.",
           "Author full names": "Varela, Ana (11); Okafor, Maya (22)", "Author(s) ID": "11; 22"}
    with pytest.raises(NotCleanedError, match="not been cleaned with Bibliominer") as err:
        Corpus.from_csv(_write(tmp_path, [raw, raw]))
    message = str(err.value)
    assert "6 author cell(s) out of 6 are not numbered" in message
    assert 'line 2, column "Authors": "Varela A.; Okafor M."' in message
    assert "Clean the Scopus export first" in message


def test_one_unnumbered_line_is_enough(tmp_path):
    """A cleaned file edited by hand is no longer the deliverable."""
    edited = dict(CLEAN, Authors="Added B.")
    with pytest.raises(NotCleanedError, match='line 3, column "Authors": "Added B."'):
        Corpus.from_csv(_write(tmp_path, [CLEAN, edited]))


@pytest.mark.parametrize("authors", ["1:Varela A.; 3:Okafor M.", "2:Varela A.", "1:Varela A.; Okafor M."])
def test_the_numbers_must_run_1_2_3(tmp_path, authors):
    with pytest.raises(NotCleanedError):
        Corpus.from_csv(_write(tmp_path, [dict(CLEAN, Authors=authors)]))


def test_a_file_without_authors_is_refused(tmp_path):
    with pytest.raises(NotCleanedError, match='no "Authors" column filled in'):
        Corpus.from_csv(_write(tmp_path, [{"Title": "T", "Year": "2024"}]))


def test_the_error_is_a_value_error(tmp_path):
    """Callers that caught `ValueError` (the API) still receive it."""
    with pytest.raises(ValueError):
        Corpus.from_csv(_write(tmp_path, [{"Title": "T", "Authors": "Doe J."}]))
