"""Generate fresh sequential identifiers in a model's existing numbering scheme.

Ported from RAVEN MATLAB: manipulation/generateNewIds.m.
"""
from __future__ import annotations

import logging
import math
import re
from collections.abc import Iterable
from typing import Literal

import cobra

logger = logging.getLogger(__name__)

_ENTITY_ATTR = {"reactions": "reactions", "metabolites": "metabolites"}


def generate_new_ids(
    model: cobra.Model,
    entity_type: Literal["reactions", "metabolites"],
    prefix: str,
    *,
    quantity: int = 1,
    num_length: int = 4,
) -> list[str]:
    """Generate ``quantity`` new ids, sequentially numbered after the model's own.

    Existing ids are filtered to those starting with ``prefix`` and the prefix is
    stripped. When any remainder reads as a finite whole number of absolute value below 2**52,
    the largest number is the starting point and its text length is the width. A remainder is
    read as a plain decimal number with an optional sign, fraction and ``e`` exponent; ``inf``,
    ``nan``, ``2.5``, ``1e20``, ``1e400`` and text do not count. MATLAB ``str2double`` reads
    more as numbers than this does (hexadecimal ``0x10``, a thousands separator ``1,000``,
    complex ``3+0i``), so ids with such suffixes are numbered differently by
    ``generateNewIds``. When no remainder counts, the width is the length of the last remainder
    in string order and numbering starts at 1. ``num_length`` applies only when no id has the
    prefix. This is ``generateNewIds.m``: ``r_9``, ``r_10`` and ``r_2`` give ``r_11``, and
    ``r_0003`` beside ``r_abc`` gives ``r_0004``.

    Parameters
    ----------
    model
        A cobra model.
    entity_type
        Which id namespace to look in and generate for: ``"reactions"`` or
        ``"metabolites"``.
    prefix
        Prefix for all generated ids, e.g. ``"s_"`` or ``"r_"``.
    quantity
        Number of new ids to generate.
    num_length
        Width of the zero-padded numeric part, e.g. 4 gives ids like
        ``r_0001``. Ignored in favour of the model's own existing width
        whenever at least one id with this prefix is already present.

    Returns
    -------
    list[str]
        The generated ids, e.g. ``["r_0001", "r_0002"]``.
    """
    if entity_type not in _ENTITY_ATTR:
        raise ValueError(f"entity_type must be 'reactions' or 'metabolites', got {entity_type!r}")
    existing_ids = [e.id for e in getattr(model, _ENTITY_ATTR[entity_type])]
    return next_ids(existing_ids, prefix, quantity, num_length, label=entity_type)


_NUMBER = re.compile(r"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:e[+-]?[0-9]+)?|[+-]?inf", re.IGNORECASE)


def _str2double(text: str) -> float:
    """Signed decimal numbers with an ``e`` exponent, and ``inf``, as a float. Other text,
    including a ``d`` exponent, hexadecimal and a thousands separator, is NaN."""
    text = text.strip()
    if not _NUMBER.fullmatch(text):
        return math.nan
    return float(text)


def _suffix_number(text: str) -> float:
    """The numeric value of an id suffix, or NaN when the suffix is not a finite whole number
    (``inf``, ``1e400``, ``2.5``, ``1d2`` and text) or has an absolute value of 2**52 or more,
    where a double no longer separates consecutive whole numbers (RAVEN uses ``flintmax/2``)."""
    number = _str2double(text)
    if not math.isfinite(number) or number != math.floor(number) or abs(number) >= 2.0**52:
        return math.nan
    return number


def next_ids(
    existing_ids: Iterable[str],
    prefix: str,
    quantity: int = 1,
    num_length: int = 4,
    *,
    label: str = "entity",
) -> list[str]:
    """The id sequence of :func:`generate_new_ids` for a list of existing ids."""
    stripped = [eid[len(prefix):] for eid in existing_ids if eid.startswith(prefix)]

    last_id = 0
    if stripped:
        numbers = [_suffix_number(x) for x in stripped]
        valid = [i for i, n in enumerate(numbers) if not math.isnan(n)]
        if valid:
            best = max(valid, key=lambda i: (numbers[i], -i))
            last_id = int(numbers[best])
            num_length = len(stripped[best])
        else:
            num_length = len(sorted(stripped)[-1])
    else:
        logger.info(
            "No %s ids with prefix %r currently exist in the model. "
            "The first new id will be %r.",
            label, prefix, f"{prefix}{1:0{num_length}d}",
        )

    return [f"{prefix}{k + last_id:0{num_length}d}" for k in range(1, quantity + 1)]
