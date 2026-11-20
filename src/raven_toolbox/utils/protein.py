"""Compute the molecular weight of a protein from its amino acid sequence."""
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
    """Compute the molecular weight of a protein from its sequence.

    Each residue contributes its average mass; one water mass is added
    once (representing the H + OH at the peptide termini, equivalent to
    one un-eliminated water relative to N residues + N water masses).

    Recognized amino acid codes (the sequence is uppercased first, so
    matching is case-insensitive):

        ACDEFGHIKLMNPQRSTVWY  - the 20 standard residues
        BJOZUX                - non-standard / ambiguous codes

    Whitespace and digits are skipped silently. Any other unrecognized
    character is also skipped, but triggers one warning per call
    listing every distinct offending character.

    Parameters
    ----------
    sequence
        Protein sequence as a string. May contain whitespace, digits,
        and other non-letter characters; those are ignored.

    Returns
    -------
    float
        Molecular weight in g/mol (Da). Returns ``nan`` for an
        empty/all-skipped sequence (no residues), so a missing sequence
        is not mistaken for an 18 Da "protein".
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
