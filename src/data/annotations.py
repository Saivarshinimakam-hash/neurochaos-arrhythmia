"""
src/data/annotations.py
-----------------------
AAMI EC57:2012-style 5-class annotation mapping for the MIT-BIH
Arrhythmia Database.

Background
----------
The MIT-BIH database uses a rich set of beat symbols (40+).  For
inter-laboratory comparability the ANSI/AAMI EC57:2012 standard reduces
these to five clinically meaningful superclasses:

    N  — Normal and near-normal (LBBB, RBBB, atrial/junctional escape)
    S  — Supraventricular ectopic (APB, aberrated APB, junctional PB)
    V  — Ventricular ectopic (PVC, ventricular escape)
    F  — Fusion of ventricular and normal beat
    Q  — Paced, fusion of paced and normal, unclassifiable

Any symbol NOT listed in the mapping is treated as UNKNOWN and excluded
from the dataset by default.

This module is intentionally kept pure (no I/O, no NumPy dependency) so
it can be unit-tested without the full data pipeline.

References
----------
- ANSI/AAMI EC57:2012 — Testing and reporting performance results of
  cardiac rhythm and ST segment measurement algorithms.
- PhysioNet MIT-BIH Arrhythmia Database annotation key:
  https://www.physionet.org/content/mitdb/1.0.0/
"""

from __future__ import annotations

from typing import Optional

# ---------------------------------------------------------------------------
# Canonical AAMI mapping
# ---------------------------------------------------------------------------
# Each key is the AAMI superclass label.
# Each value is the exhaustive list of wfdb beat annotation symbols that
# belong to that class.
#
# Symbol reference (wfdb / PhysioNet convention):
#   N  – Normal beat
#   L  – Left bundle branch block beat
#   R  – Right bundle branch block beat
#   B  – Bundle branch block beat (unspecified)
#   e  – Atrial escape beat
#   j  – Nodal (junctional) escape beat
#   A  – Atrial premature beat
#   a  – Aberrated atrial premature beat
#   J  – Nodal (junctional) premature beat
#   S  – Supraventricular premature beat
#   V  – Premature ventricular contraction
#   E  – Ventricular escape beat
#   F  – Fusion of ventricular and normal beat
#   /  – Paced beat
#   f  – Fusion of paced and normal beat
#   Q  – Unclassifiable beat

AAMI_SYMBOL_TO_CLASS: dict[str, str] = {
    # ── N class ────────────────────────────────────────────
    "N": "N",
    "L": "N",
    "R": "N",
    "B": "N",   # unspecified BBB — near-normal morphology
    "e": "N",
    "j": "N",
    # ── S class ────────────────────────────────────────────
    "A": "S",
    "a": "S",
    "J": "S",
    "S": "S",
    # ── V class ────────────────────────────────────────────
    "V": "V",
    "E": "V",
    # ── F class ────────────────────────────────────────────
    "F": "F",
    # ── Q class ────────────────────────────────────────────
    "/": "Q",
    "f": "Q",
    "Q": "Q",
}

# Integer label encoding (consistent with config.yaml class_labels)
AAMI_CLASS_TO_INDEX: dict[str, int] = {
    "N": 0,
    "S": 1,
    "V": 2,
    "F": 3,
    "Q": 4,
}

INDEX_TO_AAMI_CLASS: dict[int, str] = {v: k for k, v in AAMI_CLASS_TO_INDEX.items()}

# Human-readable descriptions for reports and UI
AAMI_CLASS_DESCRIPTIONS: dict[str, str] = {
    "N": "Normal / Near-normal",
    "S": "Supraventricular ectopic",
    "V": "Ventricular ectopic",
    "F": "Fusion beat",
    "Q": "Unknown / Paced",
}

# All known MIT-BIH beat symbols that are explicitly excluded
# (non-beat annotations such as rhythm labels, noise markers, etc.)
NON_BEAT_SYMBOLS: frozenset[str] = frozenset({
    "+",  # Rhythm change
    "~",  # Signal quality change
    "|",  # Isolated QRS-like artifact
    "s",  # ST change
    "T",  # T-wave change
    "*",  # Ventricular flutter wave
    "D",  # Idioventricular rhythm
    "=",  # Ventricular flutter wave
    "p",  # Non-conducted P-wave (pacemaker)
    "^",  # Non-conducted pacer spike
    "t",  # T-wave peak
    "u",  # U-wave peak
    "`",  # PQ junction
    "\'", # J-point
    "!",  # Ventricular flutter wave
    "[",  # Start of ventricular flutter/fibrillation
    "]",  # End of ventricular flutter/fibrillation
    "?",  # Beat not classified during learning
    "x",  # Non-conducted P-wave (block)
})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def symbol_to_aami(symbol: str) -> Optional[str]:
    """Map a single wfdb beat annotation symbol to its AAMI superclass.

    Parameters
    ----------
    symbol:
        A single-character wfdb beat annotation symbol (e.g. ``"N"``,
        ``"V"``, ``"A"``).

    Returns
    -------
    str or None
        The AAMI class label (``"N"``, ``"S"``, ``"V"``, ``"F"``,
        ``"Q"``), or ``None`` if the symbol is not a mappable beat.

    Examples
    --------
    >>> symbol_to_aami("N")
    'N'
    >>> symbol_to_aami("V")
    'V'
    >>> symbol_to_aami("A")
    'S'
    >>> symbol_to_aami("+") is None
    True
    """
    return AAMI_SYMBOL_TO_CLASS.get(symbol, None)


def symbol_to_index(symbol: str) -> Optional[int]:
    """Map a wfdb beat symbol directly to its integer class index.

    Returns ``None`` for non-beat or unmappable symbols.

    Examples
    --------
    >>> symbol_to_index("V")
    2
    >>> symbol_to_index("+") is None
    True
    """
    aami = symbol_to_aami(symbol)
    if aami is None:
        return None
    return AAMI_CLASS_TO_INDEX[aami]


def aami_to_index(aami_class: str) -> int:
    """Convert an AAMI class label to its integer index.

    Parameters
    ----------
    aami_class:
        One of ``"N"``, ``"S"``, ``"V"``, ``"F"``, ``"Q"``.

    Raises
    ------
    KeyError
        If *aami_class* is not a valid AAMI superclass.
    """
    if aami_class not in AAMI_CLASS_TO_INDEX:
        raise KeyError(
            f"'{aami_class}' is not a valid AAMI class. "
            f"Valid classes: {list(AAMI_CLASS_TO_INDEX.keys())}"
        )
    return AAMI_CLASS_TO_INDEX[aami_class]


def index_to_aami(index: int) -> str:
    """Convert an integer class index to its AAMI label.

    Raises
    ------
    KeyError
        If *index* is not in [0, 4].
    """
    if index not in INDEX_TO_AAMI_CLASS:
        raise KeyError(
            f"Index {index} is not valid. Valid indices: {list(INDEX_TO_AAMI_CLASS.keys())}"
        )
    return INDEX_TO_AAMI_CLASS[index]


def is_beat_symbol(symbol: str) -> bool:
    """Return True if *symbol* is a mappable beat annotation (beat or non-beat)."""
    return symbol in AAMI_SYMBOL_TO_CLASS


def get_class_names() -> list[str]:
    """Return the ordered list of AAMI class names (index 0→4)."""
    return [INDEX_TO_AAMI_CLASS[i] for i in sorted(INDEX_TO_AAMI_CLASS)]


def get_class_descriptions() -> list[str]:
    """Return human-readable descriptions ordered by class index."""
    return [AAMI_CLASS_DESCRIPTIONS[cls] for cls in get_class_names()]


def summarise_mapping() -> str:
    """Return a formatted string summarising the full symbol → class mapping.

    Useful for logging and documentation.
    """
    lines = ["AAMI 5-class mapping (MIT-BIH symbol → superclass → index)", ""]
    for cls_idx, cls_name in sorted(INDEX_TO_AAMI_CLASS.items()):
        symbols = [s for s, c in AAMI_SYMBOL_TO_CLASS.items() if c == cls_name]
        desc = AAMI_CLASS_DESCRIPTIONS[cls_name]
        lines.append(
            f"  [{cls_idx}] {cls_name}  ({desc})"
            f"\n       symbols: {', '.join(symbols)}"
        )
    return "\n".join(lines)
