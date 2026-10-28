"""Template handling shared by the gap-filling algorithms."""
from __future__ import annotations

import cobra


def unique_templates(
    model: cobra.Model,
    templates: list[cobra.Model],
    renames: dict[str, str] | None = None,
) -> list[cobra.Model]:
    """Copies of ``templates`` whose reaction ids are unique against ``model`` and each other.

    A colliding id becomes ``<id>_<template id>``, then ``<id>_<template id>_2``, ... as in
    RAVEN's ``mergeModels``. When ``renames`` is given, it receives ``new id -> original id``
    for every renamed reaction.
    """
    taken = {r.id for r in model.reactions}
    out: list[cobra.Model] = []
    for t in templates:
        copy = t.copy()
        own = {r.id for r in copy.reactions}
        for rxn in list(copy.reactions):
            if rxn.id not in taken:
                continue
            new_id = f"{rxn.id}_{t.id}"
            suffix = 2
            while new_id in taken or new_id in own:
                new_id = f"{rxn.id}_{t.id}_{suffix}"
                suffix += 1
            if renames is not None:
                renames[new_id] = rxn.id
            own.discard(rxn.id)
            own.add(new_id)
            rxn.id = new_id
        taken |= own
        out.append(copy)
    return out
