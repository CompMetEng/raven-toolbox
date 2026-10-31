"""Interactive flux-network navigation: port of RAVEN's ``walkFluxes``.

Starting from a reaction, groups every other flux-carrying reaction connected
through a shared metabolite, labelled with its role (``produces`` or ``consumes``) and
ranked by its own flux magnitude. Two interfaces:

* :class:`FluxWalker`: a steppable, inspectable object (``.groups``,
  ``.step(n)``, ``.back()``) for use in a script or notebook without a blocking
  terminal loop.
* :func:`walk_fluxes`: the terminal loop, matching RAVEN's blocking
  ``input()``-driven navigator command for command (a number, ``b``, ``q``),
  built on :class:`FluxWalker`.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import cobra
import pandas as pd

__all__ = ["FluxWalker", "MetaboliteGroup", "NeighborReaction", "walk_fluxes"]

_HEADER_WIDTH = 68


@dataclass
class NeighborReaction:
    """One neighbouring reaction, as displayed under a metabolite group."""

    number: int
    reaction: str
    flux: float
    role: str  # "produces" | "consumes"
    name: str


@dataclass
class MetaboliteGroup:
    """One metabolite the current reaction touches, and its other carriers."""

    metabolite: str
    name: str
    role: str  # "produced" | "consumed"
    magnitude: float
    neighbors: list[NeighborReaction] = field(default_factory=list)


def _flux(fluxes: pd.Series | Mapping[str, float], rxn_id: str) -> float:
    try:
        return float(fluxes[rxn_id])
    except KeyError:
        return 0.0


def _neighbor_groups(
    model: cobra.Model,
    fluxes: pd.Series | Mapping[str, float],
    rxn_id: str,
    *,
    cutoff: float,
    max_per_met: int,
) -> tuple[list[MetaboliteGroup], list[str]]:
    """Groups of flux-carrying neighbours for ``rxn_id``, and their numbering.

    Matches ``walkFluxes.m``'s per-step computation: metabolites are visited
    in the model's own global order (not stoichiometry-dict insertion order),
    candidate neighbours are collected in the model's global reaction order
    before a stable sort by descending ``|flux|`` (so tied fluxes keep their
    original relative order, as MATLAB's ``sort`` does), and a
    reaction already numbered under an earlier metabolite keeps that number.
    """
    rxn = model.reactions.get_by_id(rxn_id)
    own_flux = _flux(fluxes, rxn_id)
    coeffs = rxn.metabolites  # {Metabolite: coefficient}, only nonzero entries

    seen: dict[str, int] = {}
    order: list[str] = []
    groups: list[MetaboliteGroup] = []

    for met in model.metabolites:
        coef = coeffs.get(met)
        if coef is None:
            continue
        net = coef * own_flux
        if abs(net) < cutoff:
            continue
        role = "consumed" if net < 0 else "produced"

        candidates = []
        for other in model.reactions:
            if other is rxn:
                continue
            other_coef = other.metabolites.get(met)
            if other_coef is None:
                continue
            other_flux = _flux(fluxes, other.id)
            if abs(other_coef * other_flux) <= cutoff:
                continue
            candidates.append(other)
        if not candidates:
            continue
        candidates.sort(key=lambda r: abs(_flux(fluxes, r.id)), reverse=True)
        candidates = candidates[:max_per_met]

        neighbors: list[NeighborReaction] = []
        for other in candidates:
            other_flux = _flux(fluxes, other.id)
            other_coef = other.metabolites[met]
            neigh_role = "consumes" if other_coef * other_flux < 0 else "produces"
            if other.id in seen:
                number = seen[other.id]
            else:
                number = len(order) + 1
                seen[other.id] = number
                order.append(other.id)
            neighbors.append(
                NeighborReaction(
                    number=number, reaction=other.id, flux=other_flux,
                    role=neigh_role, name=other.name or "",
                )
            )

        groups.append(
            MetaboliteGroup(
                metabolite=met.id, name=met.name or met.id,
                role=role, magnitude=abs(net), neighbors=neighbors,
            )
        )

    return groups, order


class FluxWalker:
    """Steppable navigator over a solved flux distribution.

    ``fluxes`` is indexed by reaction id (for example a :class:`cobra.Solution`'s
    ``.fluxes``, or the result of ``pfba(model).fluxes``). Only a metabolite
    whose net contribution from the current reaction is at least ``cutoff``
    in absolute value is shown, and at most ``max_per_met`` of its other
    flux-carrying reactions (``> cutoff``, ranked by their own ``|flux|``)
    are listed per metabolite.
    """

    def __init__(
        self,
        model: cobra.Model,
        fluxes: pd.Series | Mapping[str, float],
        start_rxn: str | int,
        *,
        cutoff: float = 1e-8,
        max_per_met: int = 8,
    ) -> None:
        self.model = model
        self.fluxes = fluxes
        self.cutoff = cutoff
        self.max_per_met = max_per_met
        self.current = self._resolve(start_rxn)
        self.history: list[str] = []
        self._groups: list[MetaboliteGroup] | None = None
        self._order: list[str] | None = None

    def _resolve(self, rxn: str | int) -> str:
        if isinstance(rxn, int):
            return self.model.reactions[rxn].id
        if rxn not in self.model.reactions:
            raise ValueError(f"Reaction {rxn!r} not found in model.")
        return rxn

    def _compute(self) -> None:
        if self._groups is None:
            self._groups, self._order = _neighbor_groups(
                self.model, self.fluxes, self.current,
                cutoff=self.cutoff, max_per_met=self.max_per_met,
            )

    @property
    def groups(self) -> list[MetaboliteGroup]:
        """Metabolite groups for the current reaction (computed on demand)."""
        self._compute()
        assert self._groups is not None
        return self._groups

    @property
    def neighbor_ids(self) -> list[str]:
        """Neighbour reaction ids in display order (index 0 is choice "1")."""
        self._compute()
        assert self._order is not None
        return self._order

    def flux(self, rxn_id: str | None = None) -> float:
        return _flux(self.fluxes, rxn_id if rxn_id is not None else self.current)

    def step(self, n: int) -> str:
        """Move to the ``n``-th (1-based) listed neighbour; returns its id."""
        ids = self.neighbor_ids
        if not (1 <= n <= len(ids)):
            raise ValueError(f"Choice must be between 1 and {len(ids)}, got {n}.")
        self.history.append(self.current)
        self.current = ids[n - 1]
        self._groups = None
        self._order = None
        return self.current

    def back(self) -> bool:
        """Return to the previous reaction; ``False`` if there is no history."""
        if not self.history:
            return False
        self.current = self.history.pop()
        self._groups = None
        self._order = None
        return True


def _render(walker: FluxWalker, print_fn) -> None:
    rule = "=" * _HEADER_WIDTH
    rxn = walker.model.reactions.get_by_id(walker.current)
    f = walker.flux()
    print_fn(f"\n{rule}")
    if rxn.name:
        print_fn(f"  [{rxn.id}]  {rxn.name}\n  flux: {f:+.6g}")
    else:
        print_fn(f"  [{rxn.id}]  flux: {f:+.6g}")
    print_fn(f"  {rxn.build_reaction_string(use_metabolite_names=True)}")
    if rxn.gene_reaction_rule:
        print_fn(f"  genes: {rxn.gene_reaction_rule}")
    print_fn(rule)

    for grp in walker.groups:
        print_fn(f"\n  {grp.name}  [{grp.role}, {grp.magnitude:.4g}]")
        for nb in grp.neighbors:
            name = nb.name if len(nb.name) <= 28 else nb.name[:25] + "..."
            print_fn(f"  {nb.number:3d}. {nb.reaction:<14s}  {nb.flux:+9.4g}  {nb.role:<8s}  {name}")

    n_neighbors = len(walker.neighbor_ids)
    print_fn("")
    if n_neighbors == 0:
        print_fn("  (no flux-carrying neighbours at this cutoff)")
        print_fn("  b: go back   q: quit")
    else:
        print_fn(f"  1-{n_neighbors}: step to reaction   b: go back   q: quit")


def walk_fluxes(
    model: cobra.Model,
    fluxes: pd.Series | Mapping[str, float],
    start_rxn: str | int,
    *,
    cutoff: float = 1e-8,
    max_per_met: int = 8,
    input_fn=input,
    print_fn=print,
) -> None:
    """Interactively navigate a flux distribution reaction by reaction.

    Starting from ``start_rxn``, prints all flux-carrying reactions connected
    through shared metabolites, grouped by metabolite and labelled with each
    neighbour's role (``produces`` or ``consumes``). Entering a neighbour's number steps
    to it, ``b`` goes back, and ``q`` quits. ``input_fn`` and ``print_fn`` are
    injectable for scripting or testing; they default to the builtins.

    See :class:`FluxWalker` for a non-blocking, programmatic form of the same
    navigation (``.step(n)``, ``.back()``, ``.groups``).
    """
    walker = FluxWalker(model, fluxes, start_rxn, cutoff=cutoff, max_per_met=max_per_met)
    while True:
        _render(walker, print_fn)
        resp = input_fn("  > ").strip()

        if resp.lower() == "q":
            print_fn("Navigator closed.")
            return
        if resp.lower() == "b":
            if not walker.back():
                print_fn("  (no history)")
            continue

        n_neighbors = len(walker.neighbor_ids)
        try:
            choice = float(resp)
        except ValueError:
            choice = float("nan")
        if choice == choice and choice == round(choice) and 1 <= choice <= n_neighbors:
            walker.step(int(choice))
        else:
            print_fn(f"  Enter a number 1-{max(n_neighbors, 1)}, 'b', or 'q'.")
