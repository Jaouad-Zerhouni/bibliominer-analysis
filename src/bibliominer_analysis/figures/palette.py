"""The SAME palette as the interface, copied, not reinvented.

Source of truth: `analysis_service/frontend/src/theme/palette.ts`,
validated by `scripts/validate_palette.js` (lightness band, chroma floor,
colour-blind separation). An exported figure using other hues than those
seen on screen would break the link between what the user read and what
they publish, and re-running the validation script here would need a Node
dependency in a Python package. Copied, with the source file named, so
that changing one reminds us to check the other.
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


#: Typographic hyphens that Arial and DejaVu lack ("Fernández‐Alemán" was
#: shown with an empty box): they are written as a plain hyphen.
_HYPHENS = str.maketrans({"\u2010": "-", "\u2011": "-", "\u2012": "-"})


def printable(text: object) -> str:
    """A text the font knows how to draw."""
    return str(text if text is not None else "").translate(_HYPHENS)


def short_text(text: object, limit: int = 24) -> str:
    """A short text on one line, shortened with "…"."""
    s = printable(text).strip()
    return s if len(s) <= limit else s[: limit - 1].rstrip() + "…"


def tick_label(text: object, width: int = 34, lines: int = 2) -> str:
    """A readable axis label: wrapped at word boundaries on ``lines`` lines of at
    most ``width`` characters, then shortened with "…".

    A 120-character journal name pushed the bars out of the frame: matplotlib
    had no room left for the axes and the figure collapsed.
    """
    words = printable(text).split()
    out, current = [], ""
    for word in words:
        candidate = (current + " " + word).strip()
        if len(candidate) <= width or not current:
            current = candidate
            continue
        out.append(current)
        current = word
        if len(out) == lines:
            break
    else:
        out.append(current)
        return "\n".join(line[:width] for line in out)
    # Some text is left: the last line ends with "…".
    last = out[-1]
    out[-1] = (last[: width - 1].rstrip() + "…") if len(last) >= width else last + " …"
    return "\n".join(out)


def categorical(mode: str) -> List[str]:
    return list(CATEGORICAL_DARK if mode == "dark" else CATEGORICAL_LIGHT)


def chrome(mode: str) -> Dict[str, str]:
    return CHROME["dark"] if mode == "dark" else CHROME["light"]


#: matplotlib settings applied FOR THE DURATION OF ONE RENDERING, never to
#: the session.
#:
#: A fallback STACK, never a single name: "Inter" may be missing without
#: matplotlib failing -- it walks down the list to an installed font, DejaVu
#: Sans as a last resort (always shipped with matplotlib).
#:
#: They used to be set on `plt.rcParams` at import time, with
#: `matplotlib.use("Agg")`: importing the package changed the fonts and the
#: display backend of the WHOLE session. In a notebook, the user's
#: `plt.show()` then stopped displaying anything.
#:
#: `svg.hashsalt`: in SVG, matplotlib draws the identifiers of its elements
#: at random (`<g id="...">`). The seed is fixed, otherwise two identical
#: renderings would give two different files.
RENDER_RC = {
    "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "svg.hashsalt": "bibliominer-analysis",
}

#: matplotlib timestamps its files: two renderings of the SAME data gave
#: different bytes. The date is removed (and the software version, which
#: would change the file at every matplotlib update): a published figure must
#: be reproducible and comparable byte for byte. JPEG accepts no metadata in
#: matplotlib: nothing to remove.
#:
#: Fixed defect: only NETWORKS benefited from it. Bars, lines and scatters
#: exported to SVG or PDF carried the date, and changed at every export.
NO_TIMESTAMP = {
    "png": {"Software": None, "Date": None},
    "svg": {"Date": None},
    "pdf": {"CreationDate": None},
}


def no_timestamp(fmt: str) -> dict:
    """The `savefig` arguments that remove the date; none for JPEG.

    `metadata=None` is NOT passed: matplotlib 3.6 (the declared minimum)
    rejects any `metadata` argument for a JPEG, `None` included, and the JPG
    export failed. Recent versions ignore it, hence a failure that only an
    installation at the minimum versions revealed.
    """
    meta = NO_TIMESTAMP.get(fmt)
    return {"metadata": meta} if meta else {}
