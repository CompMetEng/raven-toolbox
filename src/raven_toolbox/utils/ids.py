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
    stripped. When any remainder reads as a number (as MATLAB ``str2double``), the
    largest number is the starting point and its text length is the width. When none
    does, the width is the length of the last remainder in string order and numbering
    starts at 1. ``num_length`` applies only when no id has the prefix. This is
    ``generateNewIds.m``: ``r_9``, ``r_10`` and ``r_2`` give ``r_11``, and ``r_0003``
    beside ``r_abc`` gives ``r_0004``.

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


_NUMBER = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][+-]?\d+)?|[+-]?inf", re.IGNORECASE)


def _str2double(text: str) -> float:
    text = text.strip()
    if not _NUMBER.fullmatch(text):
        return math.nan
    return float(text.replace("d", "e").replace("D", "e"))


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
        numbers = [_str2double(x) for x in stripped]
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
