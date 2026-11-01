"""Add transport reactions between compartments.

cobra has no transport-reaction function. For each metabolite, the function matches the
species by name across compartments (the source in ``from_compartment`` and the metabolite
with the same name in each target compartment), optionally creates the target metabolite, and
builds a ``-1 from / +1 to`` reaction with a sequential ``tr_0001`` ID. As in RAVEN's
``addTransport``, the subsystem of each reaction names the two compartments:
``Transport between <from> and <to>`` for a reversible reaction and
``Transport from <from> to <to>`` for an irreversible one (the compartment ids are used).
"""
from __future__ import annotations

import re
import warnings
from collections.abc import Iterable

import cobra
from cobra import Metabolite, Reaction

from raven_toolbox.manipulation.add import _new_met_id


def _index_by_name(mets: Iterable[Metabolite], compartment: str) -> dict[str, Metabolite]:
    """Index metabolites by name, with a warning when a name is duplicated.

    cobra allows several metabolites with the same name in one compartment. A dict keyed by
    name holds one of them, so the function warns and keeps the first.
    """
    out: dict[str, list[Metabolite]] = {}
    for m in mets:
        out.setdefault(m.name, []).append(m)
    chosen: dict[str, Metabolite] = {}
    for name, group in out.items():
        if len(group) > 1:
            warnings.warn(
                f"Multiple metabolites named {name!r} in compartment {compartment!r} "
                f"({[m.id for m in group]}); using {group[0].id!r} for transport.",
                stacklevel=3,
            )
        chosen[name] = group[0]
    return chosen


def _transport_id_factory(model: cobra.Model, prefix: str):
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    used = [int(m.group(1)) for r in model.reactions if (m := pattern.match(r.id))]
    counter = max(used) + 1 if used else 1
    reserved: set[str] = set()  # ids already handed out but not yet in model.reactions

    def next_id() -> str:
        nonlocal counter
        # A free id must appear within len(reactions)+len(reserved)+1 tries (only that
        # many are occupied); the bound turns any pathological case into a clear error
        # instead of spinning forever.
        for _ in range(len(model.reactions) + len(reserved) + 2):
            rid = f"{prefix}{counter:04d}"
            counter += 1
            if rid not in model.reactions and rid not in reserved:
                reserved.add(rid)
                return rid
        raise RuntimeError(f"could not allocate a free reaction id with prefix {prefix!r}.")

    return next_id


def add_transport_reactions(
    model: cobra.Model,
    from_compartment: str,
    to_compartments: str | Iterable[str],
    metabolite_names: str | Iterable[str] | None = None,
    *,
    reversible: bool = True,
    only_to_existing: bool = True,
    id_prefix: str = "tr_",
) -> list[Reaction]:
    """Add transport reactions from one compartment to one or more others.

    Parameters
    ----------
    model : cobra.Model
        Model the reactions are added to, in place.
    from_compartment : str
        Source compartment id.
    to_compartments : str or Iterable of str
        Target compartment id(s).
    metabolite_names : str or Iterable of str or None
        Names of metabolites to transport. Default: every metabolite in
        ``from_compartment``.
    reversible : bool
        If True (default), bounds span the cobra configuration default
        (reversible); otherwise lower bound 0.
    only_to_existing : bool
        If True (default), only transport a metabolite into a target
        compartment where a same-named metabolite already exists. If False,
        create the missing target metabolite (with name, formula, charge and
        annotation of the source) before adding the transport.
    id_prefix : str
        Prefix for the sequential reaction IDs (``tr_0001``, ...).

    Returns
    -------
    list of cobra.Reaction
        The transport reactions added, in creation order.
    """
    # cobra's `model.compartments` only lists compartments that have metabolites;
    # include registered-but-empty ones so transport can target an empty compartment.
    known = set(model.compartments) | set(model._compartments)
    if from_compartment not in known:
        raise ValueError(f"Compartment {from_compartment!r} is not in the model.")
    if isinstance(to_compartments, str):
        to_compartments = [to_compartments]
    else:
        to_compartments = list(to_compartments)
    for comp in to_compartments:
        if comp not in known:
            raise ValueError(f"Compartment {comp!r} is not in the model.")

    source = _index_by_name(
        (m for m in model.metabolites if m.compartment == from_compartment),
        from_compartment,
    )
    if metabolite_names is None:
        names = list(source)
    else:
        names = [metabolite_names] if isinstance(metabolite_names, str) else list(metabolite_names)
        missing = [n for n in names if n not in source]
        if missing:
            raise ValueError(
                f"Metabolites not found in compartment {from_compartment!r}: {missing}"
            )

    cfg = cobra.Configuration()
    bounds = (cfg.lower_bound, cfg.upper_bound) if reversible else (0.0, cfg.upper_bound)
    from_name = model.compartments.get(from_compartment) or from_compartment
    next_id = _transport_id_factory(model, id_prefix)

    added: list[Reaction] = []
    pending_coeffs: list[dict[Metabolite, float]] = []
    for to_comp in to_compartments:
        to_name = model.compartments.get(to_comp) or to_comp
        targets = _index_by_name(
            (m for m in model.metabolites if m.compartment == to_comp),
            to_comp,
        )
        for name in names:
            src = source[name]
            dst = targets.get(name)
            if dst is None:
                if only_to_existing:
                    continue
                dst = Metabolite(
                    _new_met_id(model, "m"),
                    name=name,
                    compartment=to_comp,
                    formula=src.formula,
                    charge=src.charge,
                )
                dst.annotation = dict(src.annotation)
                model.add_metabolites([dst])
                targets[name] = dst

            rxn = Reaction(next_id())
            rxn.name = f"{name} transport, {from_name}-{to_name}"
            rxn.bounds = bounds
            rxn.subsystem = (
                f"Transport between {from_compartment} and {to_comp}"
                if reversible
                else f"Transport from {from_compartment} to {to_comp}"
            )
            added.append(rxn)
            pending_coeffs.append({src: -1, dst: 1})

    model.add_reactions(added)  # one batch — per-reaction adds are super-linear at scale
    for rxn, coeffs in zip(added, pending_coeffs, strict=True):
        rxn.add_metabolites(coeffs)

    return added
