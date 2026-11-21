"""Molecular weight of a protein from its amino acid sequence."""
from __future__ import annotations

import logging

__all__ = ["calculate_mw"]

logger = logging.getLogger(__name__)


# Average mass of water (g/mol). Added once to account for the two
# H atoms and one OH that remain after peptide bond condensation.
_WATER_MASS = 18.01528


# Average residue masses (g/mol). Standard amino acids agree with
# widely-cited references (e.g., ExPASy ProtParam). Non-standard codes:
# - B = average(D, N).
# - J = same as L and I, which both happen to equal 113.16.
# - O (pyrrolysine) = 255.31.
# - U (selenocysteine) = 150.04.
# - Z = average(E, Q).
#
# X (any amino acid): the unweighted mean of the 20 standard residues
# above, 118.885.
_STANDARD_RESIDUE_MASSES: dict[str, float] = {
    "A": 71.08,
    "C": 103.14,
    "D": 115.09,
    "E": 129.11,
    "F": 147.17,
    "G": 57.05,
    "H": 137.14,
    "I": 113.16,
    "K": 128.17,
    "L": 113.16,
    "M": 131.20,
    "N": 114.10,
    "P": 97.12,
    "Q": 128.13,
    "R": 156.19,
    "S": 87.08,
    "T": 101.10,
    "V": 99.13,
    "W": 186.21,
    "Y": 163.17,
}

_X_MASS = sum(_STANDARD_RESIDUE_MASSES.values()) / len(_STANDARD_RESIDUE_MASSES)

_RESIDUE_MASSES: dict[str, float] = {
    **_STANDARD_RESIDUE_MASSES,
    "B": (115.09 + 114.10) / 2,  # D-or-N
    "J": 113.16,                  # L-or-I (both 113.16)
    "O": 255.31,                  # pyrrolysine
    "U": 150.04,                  # selenocysteine
    "X": _X_MASS,                 # any standard residue
    "Z": (129.11 + 128.13) / 2,  # E-or-Q
}


def calculate_mw(sequence: str) -> float:
    """Molecular weight of a protein from its amino acid sequence.

    The weight is the sum of the average residue masses plus one water. Residue masses of the 20
    standard amino acids agree with ExPASy ProtParam. Ambiguous and rare codes use averages: ``B``
    (D or N), ``J`` (L or I), ``Z`` (E or Q), ``X`` (mean of the 20 standard residues, 118.885),
    ``O`` (pyrrolysine, 255.31) and ``U`` (selenocysteine, 150.04). The sequence is converted to upper
    case. Whitespace and digits are skipped silently; any other character is skipped with one warning
    per call that lists the distinct characters.

    Parameters
    ----------
    sequence : str
        Amino acid sequence in one-letter codes.

    Returns
    -------
    float
        Molecular weight (g/mol). NaN for a sequence without residues, so that a missing sequence is
        not mistaken for a protein of 18 Da.

    Examples
    --------
    >>> import math
    >>> from raven_toolbox.utils.protein import calculate_mw
    >>> round(calculate_mw("MKV"), 2)
    376.52
    >>> math.isnan(calculate_mw(""))
    True
    """
    upper = sequence.upper() if sequence else ""
    mw = _WATER_MASS
    n_residues = 0

    unknown_chars: set[str] = set()
    for ch in upper:
        if ch in _RESIDUE_MASSES:
            mw += _RESIDUE_MASSES[ch]
            n_residues += 1
        elif not ch.isspace() and not ch.isdigit():
            unknown_chars.add(ch)

    if unknown_chars:
        logger.warning(
            "calculate_mw: skipped %d unknown character(s): %s",
            len(unknown_chars),
            ", ".join(repr(c) for c in sorted(unknown_chars)),
        )

    if n_residues == 0:
        return float("nan")
    return mw
