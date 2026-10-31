"""Set reaction bounds to a sign-aware percentage band around measured values, and set
exchange-flux bounds for a chosen set of metabolites.

cobra has no idiom for the variance band case (for example "5 ± 20 %"). Other bound-setting cases
are cobra one-liners:

* fixed lb and ub: ``reaction.lower_bound``, ``upper_bound`` or ``reaction.bounds``
* equality: ``reaction.bounds = (v, v)``
* objective: ``model.objective = {reaction: coeff}``
* unconstrained: ``reaction.bounds = cobra.Configuration().bounds``
* a medium: ``model.medium = {ex_id: uptake}`` (keyed by reaction id, one bound only)

``model.medium`` does not cover the case of ``set_exchange_bounds``: independent lb and ub per
metabolite, a compartment filter for media-only exchanges, and a check that every exchange reaction
agrees on which sign of flux is import before any bound is changed.
"""
from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence

import cobra
from cobra import Metabolite, Reaction

Number = int | float


def _resolve(model: cobra.Model, reactions) -> list[Reaction]:
    if isinstance(reactions, (str, Reaction)):
        reactions = [reactions]
    out: list[Reaction] = []
    for r in reactions:
        if isinstance(r, Reaction):
            out.append(r)
        elif r in model.reactions:
            out.append(model.reactions.get_by_id(r))
        else:
            raise ValueError(f"Reaction {r!r} not found in the model.")
    return out


def _broadcast(value, n: int) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value)] * n
    vals = [float(v) for v in value]
    if len(vals) != n:
        raise ValueError(
            f"Expected 1 or {n} values to match the reactions, got {len(vals)}."
        )
    return vals


def set_variance_bounds(
    model: cobra.Model,
    reactions: str | Reaction | Iterable,
    values: Number | Sequence[Number],
    percent: Number,
) -> list[Reaction]:
    """Constrain reactions to a ``±percent/2`` band around measured values.

    For a measured value ``v`` and ``percent`` ``p``, the bounds become
    ``v * (1 - p/200) .. v * (1 + p/200)``. ``percent`` is the total
    width, split into half above and half below. For a negative ``v`` the two are
    swapped so that ``lb <= ub``. For example, ``percent=5`` gives 97.5 % to 102.5 % of ``v``.

    Parameters
    ----------
    reactions
        Reaction IDs or objects.
    values
        Measured value per reaction; a scalar applies to all reactions.
    percent
        Total band width as a percentage.

    Returns
    -------
    list of cobra.Reaction
        The reactions whose bounds were set.
    """
    rxns = _resolve(model, reactions)
    half = percent / 200.0
    for rxn, v in zip(rxns, _broadcast(values, len(rxns)), strict=True):
        lo, hi = v * (1 - half), v * (1 + half)
        rxn.bounds = (hi, lo) if v < 0 else (lo, hi)
    return rxns


def _exchange_reactions(model: cobra.Model, media_only: bool) -> list[Reaction]:
    rxns = [r for r in model.reactions if r.boundary]
    if not media_only:
        return rxns
    extracellular_ids = {
        cid for cid, name in model.compartments.items() if name.lower() == "extracellular"
    }
    if not extracellular_ids:
        raise ValueError('Could not find any compartments named "extracellular".')
    return [r for r in rxns if next(iter(r.metabolites)).compartment in extracellular_ids]


def _resolve_met(model: cobra.Model, exchanged: list[Metabolite], token) -> Metabolite | None:
    """Match ``token`` (a Metabolite, id or name) against the exchanged metabolites only.

    A metabolite that is in the model but not exchanged is treated like a metabolite that
    does not exist."""
    if isinstance(token, Metabolite):
        return token if token in exchanged else None
    by_id = {m.id: m for m in exchanged}
    if token in by_id:
        return by_id[token]
    by_name = {m.name.lower(): m for m in exchanged}
    return by_name.get(str(token).lower())


def set_exchange_bounds(
    model: cobra.Model,
    mets: str | Metabolite | Iterable[str | Metabolite] | None = None,
    lb: Number | Sequence[Number] | None = None,
    ub: Number | Sequence[Number] | None = None,
    *,
    close_others: bool = True,
    media_only: bool = False,
) -> list[Metabolite]:
    """Set exchange-flux bounds for a chosen set of metabolites.

    Corresponds to RAVEN ``setExchangeBounds``. Exchange reactions are the boundary reactions of
    cobra (single metabolite). The function warns about every metabolite of the model that is
    exchanged by more than one reaction, whether or not it is in ``mets`` (RAVEN warns only about
    the metabolites in ``mets``). The bounds are set on all exchange reactions of each metabolite in
    ``mets``, as in RAVEN.

    Parameters
    ----------
    mets
        Metabolite objects, ids (exact) or names (case-insensitive), matched against the
        exchanged metabolites of the model only; a metabolite in the model that is not
        exchanged counts as unused, like one that does not exist. ``None`` (default)
        selects every exchanged metabolite, in which case ``lb`` and ``ub`` must each be a
        single value.
    lb, ub
        A single value for every metabolite in ``mets``, or one value per
        metabolite. Default to ``cobra.Configuration().bounds`` when not given.
    close_others
        Close import on every exchange reaction not covered by ``mets``; export is not
        changed. Default True. Set to False, with a warning, if the exchange reactions of the
        model do not agree on which sign of flux is import: closing import would then need a
        different bound (lb or ub) for each reaction, which this function does not do.
    media_only
        Only consider exchange reactions whose metabolite is in the compartment named
        "extracellular" (in ``model.compartments``, case-insensitive), which excludes, for
        example, sink and demand reactions of an intracellular metabolite. Raises
        ``ValueError`` if the model has no compartment with that name.

    Returns
    -------
    list of str or cobra.Metabolite
        The entries of ``mets`` (as supplied: ids, names or metabolite objects) that were not
        found among the exchanged metabolites of the model.
    """
    exch_rxns = _exchange_reactions(model, media_only)
    exchanged = list(dict.fromkeys(next(iter(r.metabolites)) for r in exch_rxns))

    signs = {next(iter(r.metabolites.values())) >= 0 for r in exch_rxns}
    if len(signs) > 1:
        warnings.warn(
            "Some exchange reactions differ in direction, and therefore have opposite "
            'meanings of lb and ub; forcing close_others=False.',
            stacklevel=2,
        )
        close_others = False
        direction = None
    else:
        direction = "forward" if True in signs else "backward"

    default_lb, default_ub = cobra.Configuration().bounds
    if lb is None:
        lb = default_lb
    if ub is None:
        ub = default_ub

    if mets is None:
        if isinstance(lb, Sequence) or isinstance(ub, Sequence):
            raise ValueError(
                "Only one upper and one lower bound may be provided if metabolites "
                "are not specified."
            )
        targets = exchanged
        lbs = _broadcast(lb, len(targets))
        ubs = _broadcast(ub, len(targets))
        unused: list[Metabolite] = []
    else:
        tokens = [mets] if isinstance(mets, (str, Metabolite)) else list(mets)
        lbs_in = _broadcast(lb, len(tokens))
        ubs_in = _broadcast(ub, len(tokens))
        targets = []
        lbs, ubs, unused = [], [], []
        for token, lo, hi in zip(tokens, lbs_in, ubs_in, strict=True):
            met = _resolve_met(model, exchanged, token)
            if met is None:
                unused.append(token)
                continue
            targets.append(met)
            lbs.append(lo)
            ubs.append(hi)

    if not targets:
        return unused

    met_to_rxns: dict[Metabolite, list[Reaction]] = {}
    for r in exch_rxns:
        met_to_rxns.setdefault(next(iter(r.metabolites)), []).append(r)
    repeated = [m for m, rs in met_to_rxns.items() if len(rs) > 1]
    if repeated:
        warnings.warn(
            "The following metabolites are involved in more than one exchange "
            f"reaction: {', '.join(m.name for m in repeated[:10])}"
            + (f", and {len(repeated) - 10} more." if len(repeated) > 10 else ""),
            stacklevel=2,
        )

    touched: set[Reaction] = set()
    for met, lo, hi in zip(targets, lbs, ubs, strict=True):
        for rxn in met_to_rxns[met]:
            rxn.lower_bound, rxn.upper_bound = lo, hi
            touched.add(rxn)

    if close_others:
        for rxn in exch_rxns:
            if rxn in touched:
                continue
            if direction == "backward":
                rxn.lower_bound = 0
            else:
                rxn.upper_bound = 0

    return unused
