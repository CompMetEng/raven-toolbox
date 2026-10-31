"""Generic "apply this YAML condition" loader.

Schema (all keys optional):

.. code-block:: yaml

    name: my_condition
    description: free-form text.

    # Set exchange reactions to bounds (0, 1000) before any per-reaction change.
    # The string "in" or "out" (in any case; RAVEN's getExchangeRxns direction: the
    # boundary metabolite is the reaction's substrate or product respectively) resets only
    # that direction. Any other truthy value, such as "all", "both" or true, resets every
    # exchange in either direction. A falsy value (false, 0, "" or null) resets nothing.
    # RAVEN passes the value to getExchangeRxns as the reaction type and also accepts
    # "uptake", "excrete", "reverse" and "blocked"; here those strings reset every exchange.
    prelude:
      reset_exchanges: out                # "in" | "out" | any other truthy value

    # Remove metabolites from a pseudoreaction and, when charge_balance_met is
    # given, rebalance H+ (or any other balance metabolite) so that the total
    # charge is zero. With charge_balance_met, raises ValueError if any remaining
    # participant has no charge set, because the charge balance is then unknown.
    cofactor_pseudoreaction:
      rxn_id: r_4598
      remove_mets:
        - { met: s_3714 }                 # comment field is informational
      charge_balance_met: s_0794          # optional

    # Add (combine=True) to a reaction's stoichiometry.
    biomass_stoichiometry_delta:
      rxn_id: r_4041
      add:
        - { met: s_0689, coef:  0.08 }
        - { met: s_0687, coef: -0.08 }
        - { met: s_0794, coef: -0.16 }

    # Per-reaction bounds. An omitted lb or ub leaves that bound unchanged. A
    # reaction that is not in the model is skipped with a warning.
    bounds:
      - { rxn: r_1654, lb: -1000 }
      - { rxn: r_1992, lb: 0 }
      - { rxn: r_1663, lb: 0, ub: 0 }

    # Emits a warning if the number of applied bound entries with
    # ``lb == -1000`` differs from this value.
    expected_uptake_count: 15

The steps run in the order prelude, cofactor pseudoreaction, biomass stoichiometry
delta, bounds. Project-specific extensions (for example yeast-GEM's
``amino_acid_ratio``) are handled by the caller before or after
:func:`apply_condition`.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import cobra
from ruamel.yaml import YAML

# A safe loader keeps the parsed document as plain dict / list / scalars,
# which matches what callers expect from ``load_condition``. ruamel.yaml
# is already a transitive dependency via cobra, so we don't take on
# PyYAML on top.
_SAFE_YAML = YAML(typ="safe")

#: ``prelude.reset_exchanges`` puts every exchange reaction at this
#: upper bound (and lower bound = 0).
DEFAULT_RESET_EXCHANGES_UPPER_BOUND: float = 1000.0


def load_condition(path: str | Path) -> dict[str, Any]:
    """Parse a condition YAML file into a plain dict.

    Parameters
    ----------
    path : str or Path
        Path of the YAML file, in the schema described in the module docstring.

    Returns
    -------
    dict
        The parsed condition. A file without content gives an empty dict, which
        :func:`apply_condition` applies as a condition without steps (as RAVEN ``parseYAML``
        and ``applyCondition`` do).

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Condition file not found: {path}")
    with open(path) as f:
        data = _SAFE_YAML.load(f)
    return {} if data is None else data


def apply_condition(
    model: cobra.Model,
    condition: dict[str, Any] | str | Path,
) -> cobra.Model:
    """Apply a parsed condition (or a YAML file path) to ``model`` in place.

    Returns the same model object.

    Parameters
    ----------
    model : cobra.Model
        Model to modify in place.
    condition : dict or str or Path
        A condition already parsed into a dict, or the path of a condition YAML file that is
        read with :func:`load_condition`. The schema is described in the module docstring.
    """
    cfg = condition if isinstance(condition, dict) else load_condition(condition)

    _apply_prelude(model, cfg)
    _apply_cofactor_pseudoreaction(model, cfg)
    _apply_biomass_stoichiometry_delta(model, cfg)
    n_uptake = _apply_bounds(model, cfg)
    _check_uptake_count(cfg, n_uptake)

    return model


def set_reaction_bounds(rxn: cobra.Reaction, lb: float, ub: float) -> None:
    """Set both bounds, bypassing cobra's ``lb <= ub`` validator.

    Some conditions set an infeasible state with ``lb > ub`` (for example to force
    flux via a sentinel bound), which cobra's setter rejects. When ``lb > ub``, the
    function writes the private bound attributes of the reaction directly; otherwise
    it uses ``rxn.bounds``.

    Parameters
    ----------
    rxn : cobra.Reaction
        Reaction to modify in place.
    lb : float
        New lower bound.
    ub : float
        New upper bound.
    """
    if lb > ub:
        rxn._lower_bound = lb  # noqa: SLF001 — intentional bypass
        rxn._upper_bound = ub  # noqa: SLF001 — intentional bypass
    else:
        rxn.bounds = (lb, ub)


# --- step implementations ---------------------------------------------

def _exchange_direction(rxn: cobra.Reaction) -> str | None:
    """RAVEN ``getExchangeRxns`` rule for "in" and "out", based on the boundary metabolite.

    "out": nothing is produced within the model (``sum(S > 0) == 0``), so the
    boundary metabolite is implicitly the reaction's product. "in": nothing is
    consumed (``sum(S < 0) == 0``), so the boundary metabolite is implicitly the
    substrate. A reaction with both products and reactants returns ``None``.
    The rule applies to reactions with any number of metabolites, as in RAVEN, and is
    wider than cobra's ``Reaction.boundary``.
    """
    has_product = any(c > 0 for c in rxn.metabolites.values())
    has_reactant = any(c < 0 for c in rxn.metabolites.values())
    if not has_product:
        return "out"
    if not has_reactant:
        return "in"
    return None


def _apply_prelude(model: cobra.Model, cfg: dict[str, Any]) -> None:
    prelude = cfg.get("prelude") or {}
    value = prelude.get("reset_exchanges")
    if not value:
        return
    direction = value.lower() if isinstance(value, str) else None
    if direction in ("in", "out"):
        # A named direction resets only that direction, matching RAVEN's
        # applyCondition, which forwards this same value straight through to
        # getExchangeRxns as its direction filter.
        targets = [r for r in model.reactions if _exchange_direction(r) == direction]
    else:
        # "all" / "both", or any other truthy value that isn't a direction
        # keyword (e.g. a bare `true`): reset every reaction RAVEN's
        # getExchangeRxns would call an exchange in either direction.
        targets = [r for r in model.reactions if _exchange_direction(r) is not None]
    for rxn in targets:
        rxn.lower_bound = 0
        rxn.upper_bound = DEFAULT_RESET_EXCHANGES_UPPER_BOUND


def _apply_cofactor_pseudoreaction(model: cobra.Model, cfg: dict[str, Any]) -> None:
    cp = cfg.get("cofactor_pseudoreaction")
    if not cp:
        return
    rxn = model.reactions.get_by_id(cp["rxn_id"])
    for entry in cp.get("remove_mets", []):
        met = model.metabolites.get_by_id(entry["met"])
        _set_coefficient(rxn, met, 0.0)
    balance_met_id = cp.get("charge_balance_met")
    if balance_met_id:
        balance_met = model.metabolites.get_by_id(balance_met_id)
        _set_coefficient(rxn, balance_met, 0.0)
        # Only the participating metabolites may be summed. A missing charge
        # (None) is unknown, not neutral -- treating it as 0 would silently
        # write the wrong balancing coefficient, matching RAVEN's
        # applyCondition, which raises rather than defaulting to 0 here.
        unknown = sorted(m.id for m in rxn.metabolites if m.charge is None)
        if unknown:
            raise ValueError(
                f"Cannot charge balance {rxn.id!r}: the following metabolites "
                "have no charge, so the charge balance is unknown rather "
                f"than zero: {', '.join(unknown)}"
            )
        total_charge = sum(
            m.charge * coef
            for m, coef in rxn.metabolites.items()
        )
        _set_coefficient(rxn, balance_met, -total_charge)


def _apply_biomass_stoichiometry_delta(model: cobra.Model, cfg: dict[str, Any]) -> None:
    delta = cfg.get("biomass_stoichiometry_delta")
    if not delta:
        return
    rxn = model.reactions.get_by_id(delta["rxn_id"])
    for entry in delta.get("add", []):
        met = model.metabolites.get_by_id(entry["met"])
        rxn.add_metabolites({met: float(entry["coef"])}, combine=True)


def _apply_bounds(model: cobra.Model, cfg: dict[str, Any]) -> int:
    n_uptake = 0
    for entry in cfg.get("bounds", []):
        try:
            rxn = model.reactions.get_by_id(entry["rxn"])
        except KeyError:
            warnings.warn(
                f"Reaction {entry['rxn']!r} not found in model; skipping.",
                stacklevel=3,
            )
            continue
        new_lb = float(entry["lb"]) if "lb" in entry else rxn.lower_bound
        new_ub = float(entry["ub"]) if "ub" in entry else rxn.upper_bound
        set_reaction_bounds(rxn, new_lb, new_ub)
        if entry.get("lb") == -1000:
            n_uptake += 1
    return n_uptake


def _check_uptake_count(cfg: dict[str, Any], n_uptake: int) -> None:
    expected = cfg.get("expected_uptake_count")
    if expected is None:
        return
    if n_uptake != expected:
        warnings.warn(
            f"Expected {expected} uptake reactions, applied {n_uptake}. "
            "Some referenced reactions may be missing from the model.",
            stacklevel=3,
        )


# --- helpers ----------------------------------------------------------

def _set_coefficient(rxn: cobra.Reaction, met: cobra.Metabolite, value: float) -> None:
    """Set the stoichiometric coefficient of ``met`` in ``rxn`` to ``value``.

    The function calls ``add_metabolites`` with ``combine=True`` and
    ``(value - current)``, which sets the coefficient without depending on cobra's
    removal of zero coefficients.
    """
    current = rxn.metabolites.get(met, 0.0)
    delta = float(value) - current
    if delta != 0:
        rxn.add_metabolites({met: delta}, combine=True)
