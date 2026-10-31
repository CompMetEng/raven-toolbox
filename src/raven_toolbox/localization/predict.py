"""Sub-cellular localisation by MILP.

Assigns reactions to compartments by maximising per-gene localisation evidence minus
inter-compartment transport cost. Behaviour:

* The caller passes the set of reactions to (re-)place (``reactions_to_relocate``);
  all other reactions are pinned. Boundary reactions and existing inter-compartment
  transports are always pinned, also when listed.
* Incomplete models are accepted: no reaction is removed because a metabolite is not
  produced. Reactions whose genes all lack a score are reported in ``unplaced_reactions``.
* The MILP is solved with the solver of the model (Gurobi, HiGHS or GLPK).
* ``apply=False`` returns a :class:`LocalizationProposal` (a diff) and leaves the model unchanged.
* Multi-compartment placement is the default. A gene can be assigned to several compartments:
  its highest-scoring compartment is free, and every additional compartment costs
  ``multi_compartment_penalty``. Secondary compartments have lower predictor
  scores and are picked only when their score exceeds the explicit penalty.
  A high ``multi_compartment_penalty`` gives mono-localised genes.

Limitations:

* Isozymes are not separated: a reaction with isozymes requires all listed genes in its
  compartment. For per-isozyme placement, call
  :func:`raven_toolbox.manipulation.expand_model` first.
* Transports go through ``default_compartment``.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import cobra
import pandas as pd
from optlang.interface import Variable  # untyped (resolves to Any); used for container hints
from optlang.symbolics import Real, add, mul

from raven_toolbox.localization.scores import LocalizationScores


@dataclass
class LocalizationProposal:
    """What :func:`predict_localization` proposes, before applying it.

    All DataFrames have one row per item. With ``apply=False``, the proposal previews
    the changes; :func:`apply_localization` applies it.
    """

    moved: pd.DataFrame                       # rxn_id, from_compartment, to_compartment
    added_transports: pd.DataFrame            # met_id, compartment (other than default)
    gene_compartments: dict[str, list[str]]   # gene_id → list of compartments assigned
    unplaced_reactions: list[str] = field(default_factory=list)  # had no scored gene support
    objective: float = 0.0


@dataclass
class LocalizationResult:
    """Outcome of :func:`predict_localization` (when ``apply=True``)."""

    model: cobra.Model
    proposal: LocalizationProposal
    added_transports: list[cobra.Reaction] = field(default_factory=list)


# --------------------------------------------------------------------- helpers

def _reaction_compartment(rxn: cobra.Reaction) -> str | None:
    """Single compartment id if all metabolites share one, else ``None`` (transport)."""
    comps = {m.compartment for m in rxn.metabolites if m.compartment}
    return next(iter(comps)) if len(comps) == 1 else None


def _reaction_genes(rxn: cobra.Reaction) -> list[str]:
    """Genes on the reaction's GPR (flat list, without the AND or OR structure)."""
    return [g.id for g in rxn.genes]


# --------------------------------------------------------------------- the MILP

def predict_localization(
    model: cobra.Model,
    scores: LocalizationScores,
    reactions_to_relocate: Iterable[str],
    *,
    default_compartment: str = "c",
    transport_cost: float | Mapping[str, float] = 0.5,
    multi_compartment_penalty: float = 0.5,
    apply: bool = True,
    mip_gap: float | None = None,
    time_limit: float | None = None,
) -> LocalizationResult | LocalizationProposal:
    """Place a caller-specified set of reactions in compartments with a MILP.

    Returns a :class:`LocalizationProposal` (when ``apply=False``) or a
    :class:`LocalizationResult` (when ``apply=True``).

    ``reactions_to_relocate``: the reaction ids to (re-)place. All other reactions stay
    where they are. Boundary reactions and existing multi-compartment transports in
    this set are filtered out without a message (they are always pinned). An empty set, or a set
    without non-boundary reactions, gives an empty proposal (a :class:`LocalizationResult` holding a
    copy of the model when ``apply=True``).

    ``transport_cost``: a scalar (same cost per added transport) or a mapping
    ``{metabolite_id_base: cost}`` (the base id has the compartment suffix removed,
    for example ``"glc__D"`` matches ``"glc__D_c"`` and ``"glc__D_e"``). A metabolite missing from
    the mapping costs ``0.5``. Negative costs favour adding the transport.

    **Multi-compartment gene scoring (default):** a gene contributes its
    predictor score in each compartment it is assigned to. The highest-scoring compartment
    is free, and each additional compartment costs ``multi_compartment_penalty``. A
    secondary compartment is picked when its score (lower than the primary) exceeds the
    penalty; there is no hard cutoff. A large ``multi_compartment_penalty`` gives
    mono-localised genes.

    ``time_limit``: maximum solve time of the MILP in seconds (``None`` = no limit). With no limit,
    a model with more than about 5,000 gene-associated reactions, or with ambiguous scores, can make
    the solve run for a long time; ``time_limit=900`` stops it after 15 min with the incumbent
    solution. MATLAB RAVEN's ``predictLocalization`` has a default ``maxTime`` of 15 min (900 s),
    but it solves this problem with simulated annealing, so its ``maxTime`` is not a solver cutoff
    with an optimality guarantee.
    """
    # ---- 1. Scope: which reactions move, which are pinned. -----------------
    to_relocate = set(reactions_to_relocate)
    # Boundaries / transports always pin (even if listed).
    to_relocate -= {r.id for r in model.reactions
                    if r.boundary or _reaction_compartment(r) is None}
    if not to_relocate:
        return _empty_result(model, apply)

    # ---- 2. Compartments universe (model + scores). ------------------------
    compartments = sorted(set(model.compartments) | set(scores.compartments))
    if default_compartment not in compartments:
        raise ValueError(f"default_compartment={default_compartment!r} not in known "
                         f"compartments {compartments}")

    # ---- 3. Gather genes for the relocate-set, build score lookup. ---------
    # Genes only mentioned by pinned reactions don't enter the MILP.
    moving = [model.reactions.get_by_id(rid) for rid in sorted(to_relocate)]
    genes_in_scope: set[str] = set()
    unplaced: list[str] = []
    for r in moving:
        gs = _reaction_genes(r)
        scored = [g for g in gs if g in scores.df.index]
        if not gs:
            # GPR-less reaction: place it freely (no gene coupling). Allowed.
            continue
        if not scored:
            # All genes absent from predictor → no signal; report and skip.
            unplaced.append(r.id)
            continue
        genes_in_scope.update(scored)
    # Remove reactions we can't score from the placement set.
    placeable = [r for r in moving if r.id not in set(unplaced)]
    if not placeable:
        # Everything in the relocate set lacks scored genes — return a proposal with
        # only the unplaced list.
        prop = LocalizationProposal(
            moved=pd.DataFrame(columns=["rxn_id", "from_compartment", "to_compartment"]),
            added_transports=pd.DataFrame(columns=["met_id", "compartment"]),
            gene_compartments={}, unplaced_reactions=unplaced, objective=0.0)
        return prop if not apply else LocalizationResult(model, prop)

    # ---- 4. Per-metabolite transport cost. ---------------------------------
    def _met_cost(m_id: str) -> float:
        if not isinstance(transport_cost, (int, float)):
            base = m_id.rsplit("_", 1)[0]
            return float(transport_cost.get(base, transport_cost.get(m_id, 0.5)))
        return float(transport_cost)

    # ---- 5. Build the MILP. ------------------------------------------------
    model.solver  # noqa: B018 — ensure the solver is initialised so model.problem works
    prob = model.problem
    opt = prob.Model()

    # x[r, c] = 1 iff reaction r placed in c (only for r ∈ placeable)
    x: dict[tuple[str, str], Variable] = {
        (r.id, c): prob.Variable(f"x_{r.id}_{c}", type="binary")
        for r in placeable for c in compartments
    }
    # y[g, c] = 1 iff gene g assigned to c
    y: dict[tuple[str, str], Variable] = {
        (g, c): prob.Variable(f"y_{g}_{c}", type="binary")
        for g in genes_in_scope for c in compartments
    }
    # t[m_id, c] = 1 iff metabolite m (with id including its current compartment suffix)
    # needs a transport to compartment c (c ≠ default). One per (base met, c).
    met_keys: set[tuple[str, str]] = set()
    for r in placeable:
        for m in r.metabolites:
            for c in compartments:
                if c != default_compartment:
                    met_keys.add((m.id, c))
    t: dict[tuple[str, str], Variable] = {
        k: prob.Variable(f"t_{k[0]}_{k[1]}", type="binary") for k in met_keys
    }

    cons: list = []
    # 5a. Each placeable reaction goes to exactly one compartment.
    for r in placeable:
        cons.append(prob.Constraint(add([mul([Real(1.0), x[r.id, c]]) for c in compartments]),
                                     lb=1.0, ub=1.0, name=f"place_{r.id}"))
    # 5b. Gene-reaction coupling: if r placed in c, every scored gene of r must be in c.
    for r in placeable:
        for g in _reaction_genes(r):
            if g not in genes_in_scope:
                continue
            for c in compartments:
                # x[r,c] − y[g,c] ≤ 0
                cons.append(prob.Constraint(x[r.id, c] - y[g, c], ub=0.0,
                                             name=f"gene_{r.id}_{g}_{c}"))
    # 5c. Gene assignment: each gene in scope lands in ≥1 compartment. Multi is allowed;
    # the multi_compartment_penalty in the objective keeps extras from coming for free.
    for g in genes_in_scope:
        s = add([mul([Real(1.0), y[g, c]]) for c in compartments])
        cons.append(prob.Constraint(s, lb=1.0, name=f"gene_one_{g}"))
    # 5d. Transport requirement: t[m,c] ≥ x[r,c] whenever r touches m and c ≠ default.
    for r in placeable:
        for m in r.metabolites:
            for c in compartments:
                if c == default_compartment:
                    continue
                # x[r,c] − t[m,c] ≤ 0
                cons.append(prob.Constraint(x[r.id, c] - t[m.id, c], ub=0.0,
                                             name=f"trans_{r.id}_{m.id}_{c}"))

    opt.add(list(x.values()) + list(y.values()) + list(t.values()) + cons)

    # 5e. Objective.
    obj_terms = []
    # + per-gene per-compartment localisation score (rows missing → 0)
    score_lookup = scores.df  # gene_id × compartment → float
    for g in genes_in_scope:
        for c in compartments:
            if c not in score_lookup.columns:
                continue
            val = score_lookup.at[g, c]
            s = 0.0 if pd.isna(val) else float(val)
            if s:
                obj_terms.append(mul([Real(s), y[g, c]]))
    # − transport cost per added transport
    for (m_id, _c), tvar in t.items():
        cost = _met_cost(m_id)
        if cost:
            obj_terms.append(mul([Real(-cost), tvar]))
    # − multi-compartment penalty per *extra* compartment (the primary is free).
    # Per gene: penalty * (Σ_c y[g,c] - 1) = penalty * Σ_c y[g,c] - penalty (constant). The
    # constant doesn't affect optimisation but is added back to the reported objective so
    # the value matches the "primary free" intent the user reads off the proposal.
    constant_offset = 0.0
    if multi_compartment_penalty:
        for yvar in y.values():
            obj_terms.append(mul([Real(-multi_compartment_penalty), yvar]))
        constant_offset = multi_compartment_penalty * len(genes_in_scope)

    opt.objective = prob.Objective(add(obj_terms) if obj_terms else Real(0.0), direction="max")
    if time_limit is not None:
        opt.configuration.timeout = int(time_limit)
    if mip_gap is not None:
        try:  # Gurobi-specific
            opt.problem.Params.MIPGap = mip_gap
        except Exception:  # noqa: BLE001
            pass

    opt.optimize()
    if opt.status not in ("optimal", "feasible", "suboptimal", "time_limit"):
        raise RuntimeError(f"localisation MILP did not solve (status: {opt.status}).")

    # ---- 6. Read the solution into a proposal. -----------------------------
    moved_rows: list[dict] = []
    for r in placeable:
        chosen = None
        for c in compartments:
            if (x[r.id, c].primal or 0.0) >= 0.5:
                chosen = c
                break
        from_c = _reaction_compartment(r)
        if chosen and chosen != from_c:
            moved_rows.append({"rxn_id": r.id, "from_compartment": from_c,
                                "to_compartment": chosen})
    moved = pd.DataFrame(moved_rows, columns=["rxn_id", "from_compartment", "to_compartment"])

    transp_rows: list[dict] = []
    for (m_id, c), tvar in t.items():
        if (tvar.primal or 0.0) >= 0.5:
            transp_rows.append({"met_id": m_id, "compartment": c})
    added_transports = pd.DataFrame(transp_rows, columns=["met_id", "compartment"])

    gene_comps: dict[str, list[str]] = {}
    for g in genes_in_scope:
        in_c = [c for c in compartments if (y[g, c].primal or 0.0) >= 0.5]
        gene_comps[g] = in_c

    proposal = LocalizationProposal(
        moved=moved, added_transports=added_transports, gene_compartments=gene_comps,
        unplaced_reactions=unplaced,
        objective=float((opt.objective.value or 0.0) + constant_offset))

    if not apply:
        return proposal
    new_model, transports = apply_localization(model, proposal, default_compartment=default_compartment)
    return LocalizationResult(model=new_model, proposal=proposal, added_transports=transports)


# --------------------------------------------------------------------- apply

def apply_localization(
    model: cobra.Model,
    proposal: LocalizationProposal,
    *,
    default_compartment: str = "c",
) -> tuple[cobra.Model, list[cobra.Reaction]]:
    """Apply a :class:`LocalizationProposal` to ``model``: move reactions, add the
    inter-compartment transports of the proposal, and return ``(model_copy, added)``.

    The returned model is a deep copy of the input (the original is unchanged). Moved
    reactions get their metabolites' compartment suffix swapped (for example ``A_c → A_m``);
    new compartment-specific metabolite copies are added as needed. Each added
    transport is a passive diffusion ``M[default] ⇌ M[c]`` (RAVEN convention),
    named ``tr_<met>_<c>``.
    """
    out = model.copy()
    added: list[cobra.Reaction] = []

    # 1. Move each reaction by remapping its metabolites to the target compartment.
    for _, row in proposal.moved.iterrows():
        rxn = out.reactions.get_by_id(row["rxn_id"])
        target = row["to_compartment"]
        new_stoich: dict[cobra.Metabolite, float] = {}
        old = list(rxn.metabolites.items())
        # Clear current stoichiometry first so cobra updates the constraints cleanly.
        rxn.subtract_metabolites(dict(old))
        for m, coeff in old:
            m_new = _met_in_compartment(out, m, target)
            new_stoich[m_new] = coeff
        rxn.add_metabolites(new_stoich)

    # 2. Add transports between default and each requested compartment.
    for _, row in proposal.added_transports.iterrows():
        m_id, c = row["met_id"], row["compartment"]
        if m_id not in out.metabolites:
            continue
        m_src = out.metabolites.get_by_id(m_id)
        if m_src.compartment == c:
            continue  # already there; no transport needed
        m_default = _met_in_compartment(out, m_src, default_compartment)
        m_dest = _met_in_compartment(out, m_src, c)
        if m_default.id == m_dest.id:
            continue
        tr_id = f"tr_{_base_met_id(m_src)}_{c}"
        if tr_id in out.reactions:
            continue
        tr = cobra.Reaction(tr_id, lower_bound=-1000, upper_bound=1000)
        tr.add_metabolites({m_default: -1.0, m_dest: 1.0})
        tr.notes["localization"] = "added by predict_localization"
        out.add_reactions([tr])
        added.append(out.reactions.get_by_id(tr_id))

    return out, added


def _base_met_id(m: cobra.Metabolite) -> str:
    """Remove the trailing ``_<compartment>`` suffix (or return the id unchanged)."""
    if m.compartment and m.id.endswith(f"_{m.compartment}"):
        return m.id[: -(len(m.compartment) + 1)]
    return m.id


def _met_in_compartment(model: cobra.Model, source: cobra.Metabolite,
                        compartment: str) -> cobra.Metabolite:
    """Return the copy of ``source`` in ``compartment``, created if absent."""
    if source.compartment == compartment:
        return source
    base = _base_met_id(source)
    new_id = f"{base}_{compartment}"
    if new_id in model.metabolites:
        return model.metabolites.get_by_id(new_id)
    new_met = cobra.Metabolite(new_id, name=source.name, compartment=compartment,
                               formula=source.formula, charge=source.charge)
    new_met.notes = dict(source.notes or {})
    model.add_metabolites([new_met])
    return new_met


def _empty_result(model: cobra.Model, apply_flag: bool):
    proposal = LocalizationProposal(
        moved=pd.DataFrame(columns=["rxn_id", "from_compartment", "to_compartment"]),
        added_transports=pd.DataFrame(columns=["met_id", "compartment"]),
        gene_compartments={}, unplaced_reactions=[], objective=0.0)
    return proposal if not apply_flag else LocalizationResult(model.copy(), proposal)
