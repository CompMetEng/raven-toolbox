"""Typed enzyme-constrained ecModel substructure (``model.ec``).

Follows the layout of MATLAB GECKO's ``model.ec`` struct. Holds the per-reaction and
per-enzyme arrays that the GECKO toolbox attaches to a metabolic model when
making it enzyme-constrained: kcat values, enzyme molecular weights, the
sparse reaction-to-enzyme coupling matrix, and the provenance source tags,
free-text notes, EC numbers and sequences.

This module contains:

- the ``EcData`` dataclass (in-memory shape),
- the YAML schema for the ``ec-rxns``, ``ec-enzymes`` and ``gecko_light``
  top-level sections,
- the (de)serialisation helpers `ec_data_from_yaml_sections` and
  `ec_data_to_yaml_sections`.

The cobra-shaped portion of the document is handled by ``raven_toolbox.io.yaml``,
which calls into this module when ec sections are present. The same division
exists in RAVEN MATLAB: ``readYAMLmodel.m`` populates ``model.ec`` whenever the
YAML defines it.

YAML schema (one entry per row):

::

    ec-rxns:
      - id: R1_EXP_1
        kcat: 12.5          # turnover number, 1/s (0 == "no kcat assigned")
        source: brenda      # optional, omitted if empty
        notes: free-text    # optional, omitted if empty
        eccodes: "1.1.1.1"  # optional; scalar OR a list when multiple
        enzymes:            # column -> stoichiometric subunit count
          P12345: 1.0
          P67890: 2.0       # heteromeric complex with 2 copies of P67890
    ec-enzymes:
      - genes: G1           # gene name as it appears in cobra
        enzymes: P12345     # uniprot accession (or KEGG id)
        mw: 50000.0         # Da; omitted when unknown
        sequence: MAGIC     # protein sequence; omitted when empty
        concs: 0.005        # proteomics-measured concentration mg/gDCW;
                            #   omitted when not measured
    gecko_light: false      # top-level bool; defaults to false on load
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import sparse


@dataclass
class EcData:
    """Typed enzyme-constrained ecModel substructure attached as ``model.ec``.

    Field semantics match MATLAB GECKO's ``model.ec`` struct one-to-one.
    The fields form two parallel index spaces:

    - per-reaction arrays (``rxns``, ``kcat``, ``source``, ``notes``,
      ``eccodes``) of length ``n_rxns``;
    - per-enzyme arrays (``genes``, ``enzymes``, ``mw``, ``sequence``,
      ``concs``) of length ``n_enzymes``.

    The sparse ``rxn_enz_mat`` of shape ``(n_rxns, n_enzymes)`` connects the two spaces.
    Its ``[i, j]`` entry is the subunit count of enzyme j in reaction i (0 or 1 for
    single enzymes, above 1 for heteromeric complexes).

    Sentinels (as in MATLAB GECKO):

    - ``kcat == 0`` means "no kcat assigned" (real turnover numbers are positive).
    - ``mw == nan`` means "MW unknown" (the writer omits NaN mw entries).
    - ``concs == nan`` means "not measured" (the writer omits NaN concs).
    - empty strings in ``source`` / ``notes`` / ``eccodes`` / ``sequence``
      are omitted on write and restored as ``""`` on load.

    ``gecko_light`` marks the gecko-light layout: cobra reactions stay
    singular, ec.rxns carries one entry per isozyme distinguished by a
    ``###_`` counter prefix, and the per-enzyme ``prot_<id>`` and usage reactions
    are omitted in favour of the shared protein pool. ``False`` (default) is the
    full layout, where ``ec.rxns`` matches cobra reactions
    one-to-one after isozyme expansion.
    """
    gecko_light: bool = False
    rxns: list[str] = field(default_factory=list)
    kcat: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    source: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    eccodes: list[str] = field(default_factory=list)
    genes: list[str] = field(default_factory=list)
    enzymes: list[str] = field(default_factory=list)
    mw: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    sequence: list[str] = field(default_factory=list)
    concs: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    rxn_enz_mat: sparse.csr_matrix = field(
        default_factory=lambda: sparse.csr_matrix((0, 0), dtype=float),
    )

    @property
    def n_rxns(self) -> int:
        return len(self.rxns)

    @property
    def n_enzymes(self) -> int:
        return len(self.enzymes)

    def validate(self) -> None:
        """Raise ``ValueError`` if internal field lengths are inconsistent.

        Compares the lengths of the per-reaction arrays, the lengths of the per-enzyme
        arrays and the shape of the coupling matrix with ``n_rxns`` and ``n_enzymes``.
        """
        n_r, n_e = self.n_rxns, self.n_enzymes

        rxn_lengths = {
            "kcat": len(self.kcat),
            "source": len(self.source),
            "notes": len(self.notes),
            "eccodes": len(self.eccodes),
        }
        for name, length in rxn_lengths.items():
            if length != n_r:
                raise ValueError(
                    f"ec.{name} has length {length}, expected {n_r} "
                    f"(matching ec.rxns)"
                )

        # `ec.enzymes` itself is the reference length; check the remaining
        # per-enzyme arrays against it.
        enz_lengths = {
            "genes": len(self.genes),
            "mw": len(self.mw),
            "sequence": len(self.sequence),
            "concs": len(self.concs),
        }
        for name, length in enz_lengths.items():
            if length != n_e:
                raise ValueError(
                    f"ec.{name} has length {length}, expected {n_e} "
                    f"(matching ec.enzymes)"
                )

        if self.rxn_enz_mat.shape != (n_r, n_e):
            raise ValueError(
                f"ec.rxn_enz_mat has shape {self.rxn_enz_mat.shape}, "
                f"expected ({n_r}, {n_e})"
            )

    @staticmethod
    def empty(n_rxns: int, n_enzymes: int = 0, *,
              gecko_light: bool = False) -> EcData:
        """Preallocate an ``EcData`` with the sentinel values.

        Per-reaction string fields are empty strings; per-enzyme string fields are
        empty strings. ``kcat`` starts at 0 (0 marks "no kcat assigned"). ``mw`` and
        ``concs`` start at NaN (unknown). ``rxn_enz_mat`` is all-zero.

        Builders in the style of makeEcModel allocate the structure with this method
        and then fill it row by row.
        """
        return EcData(
            gecko_light=gecko_light,
            rxns=[""] * n_rxns,
            kcat=np.zeros(n_rxns, dtype=float),
            source=[""] * n_rxns,
            notes=[""] * n_rxns,
            eccodes=[""] * n_rxns,
            genes=[""] * n_enzymes,
            enzymes=[""] * n_enzymes,
            mw=np.full(n_enzymes, np.nan, dtype=float),
            sequence=[""] * n_enzymes,
            concs=np.full(n_enzymes, np.nan, dtype=float),
            rxn_enz_mat=sparse.lil_matrix(
                (n_rxns, n_enzymes), dtype=float,
            ).tocsr(),
        )


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #

def ec_data_from_yaml_sections(sections: dict) -> EcData | None:
    """Build an ``EcData`` from the ec-* top-level YAML sections.

    Returns ``None`` when ``ec-rxns`` and ``ec-enzymes`` are both absent; the
    caller then leaves ``model.ec = None``. Raises ``ValueError`` when exactly one
    of the two is present.

    ``sections`` is the dict of foreign top-level keys captured by the YAML
    loader. ``gecko_light`` defaults to ``False`` when the key is absent.
    """
    has_rxns = "ec-rxns" in sections
    has_enzymes = "ec-enzymes" in sections
    if not has_rxns and not has_enzymes:
        return None
    if has_rxns != has_enzymes:
        missing = "ec-enzymes" if has_rxns else "ec-rxns"
        raise ValueError(
            f"ecModel YAML is missing the `{missing}` top-level section; "
            "both ec-rxns and ec-enzymes are required."
        )

    return _build_ec_data(
        sections["ec-rxns"],
        sections["ec-enzymes"],
        gecko_light=bool(sections.get("gecko_light", False)),
    )


def _build_ec_data(
    ec_rxns_raw: list,
    ec_enzymes_raw: list,
    *,
    gecko_light: bool,
) -> EcData:
    """Construct an ``EcData`` from the parsed YAML lists.

    Missing optional fields are filled with sentinels (NaN for ``mw`` and ``concs``,
    empty string for ``source``, ``notes``, ``eccodes`` and ``sequence``, 0.0 for ``kcat``).
    Raises ``ValueError`` when an ec-rxns row references an enzyme that is absent
    from ec-enzymes. Warns when ec-rxns has entries but no row references an enzyme.
    """
    n_e = len(ec_enzymes_raw)
    genes = [str(e["genes"]) for e in ec_enzymes_raw]
    enzymes = [str(e["enzymes"]) for e in ec_enzymes_raw]
    mw = np.array(
        [float(e.get("mw", np.nan)) for e in ec_enzymes_raw], dtype=float,
    )
    sequence = [str(e.get("sequence", "")) for e in ec_enzymes_raw]
    concs = np.array(
        [float(e.get("concs", np.nan)) for e in ec_enzymes_raw], dtype=float,
    )

    enz_index = {eid: i for i, eid in enumerate(enzymes)}

    n_r = len(ec_rxns_raw)
    rxns = [str(r["id"]) for r in ec_rxns_raw]
    # 0 == "no kcat assigned"; real turnover numbers are always positive.
    kcat = np.array(
        [float(r.get("kcat", 0.0)) for r in ec_rxns_raw], dtype=float,
    )
    source = [str(r.get("source", "")) for r in ec_rxns_raw]
    notes = [str(r.get("notes", "")) for r in ec_rxns_raw]
    eccodes = [_canonicalize_eccodes(r.get("eccodes", "")) for r in ec_rxns_raw]

    mat = sparse.lil_matrix((n_r, n_e), dtype=float)
    for i, r in enumerate(ec_rxns_raw):
        for enz_id, stoich in (r.get("enzymes") or {}).items():
            j = enz_index.get(str(enz_id))
            if j is None:
                raise ValueError(
                    f"ec-rxns[{i}] (id={r.get('id')!r}) references enzyme "
                    f"{enz_id!r} that is not present in ec-enzymes."
                )
            mat[i, j] = float(stoich)

    if n_r > 0 and mat.nnz == 0:  # reactions present but none coupled to an enzyme
        warnings.warn(
            f"ec-rxns has {n_r} entries but none reference an enzyme; the "
            "reaction↔enzyme coupling matrix is all-zero (likely malformed "
            "ec-rxns/ec-enzymes).",
            stacklevel=2,
        )

    return EcData(
        gecko_light=gecko_light,
        rxns=rxns,
        kcat=kcat,
        source=source,
        notes=notes,
        eccodes=eccodes,
        genes=genes,
        enzymes=enzymes,
        mw=mw,
        sequence=sequence,
        concs=concs,
        rxn_enz_mat=mat.tocsr(),
    )


# --------------------------------------------------------------------------- #
# Save
# --------------------------------------------------------------------------- #

def ec_data_to_yaml_sections(ec: EcData) -> dict[str, Any]:
    """Serialise an ``EcData`` to a dict suitable for YAML emission.

    Returns a fresh dict with three keys: ``gecko_light`` (bool),
    ``ec-rxns`` (list of mappings), ``ec-enzymes`` (list of mappings).
    Values are native Python primitives (no numpy or ruamel scalars), so the YAML
    writer can dump them directly.

    Empty optional fields are omitted; the loader restores them.
    """
    return {
        "gecko_light": bool(ec.gecko_light),
        "ec-rxns": _build_ec_rxns_list(ec),
        "ec-enzymes": _build_ec_enzymes_list(ec),
    }


def _build_ec_rxns_list(ec: EcData) -> list[dict[str, Any]]:
    """Translate the per-reaction ec fields and the ``rxn_enz_mat`` rows to the
    list-of-mappings YAML form.

    Empty ``source`` / ``notes`` / ``eccodes`` strings are omitted.
    ``kcat`` is always written: a real turnover number when set,
    otherwise ``0`` (0 marks "no kcat assigned").
    """
    coo = ec.rxn_enz_mat.tocoo()
    per_row_enzymes: list[dict[str, float]] = [{} for _ in range(ec.n_rxns)]
    # All three arrays come from the same COO matrix, so they're guaranteed
    # equal length; strict=True turns any future drift into a loud TypeError
    # instead of silent truncation.
    for i, j, v in zip(coo.row, coo.col, coo.data, strict=True):
        per_row_enzymes[int(i)][ec.enzymes[int(j)]] = float(v)

    out: list[dict[str, Any]] = []
    for i in range(ec.n_rxns):
        entry: dict[str, Any] = {
            "id": ec.rxns[i],
            "kcat": float(ec.kcat[i]),
        }
        if ec.source[i]:
            entry["source"] = ec.source[i]
        if ec.notes[i]:
            entry["notes"] = ec.notes[i]
        if ec.eccodes[i]:
            entry["eccodes"] = _eccodes_to_yaml(ec.eccodes[i])
        entry["enzymes"] = per_row_enzymes[i]
        out.append(entry)
    return out


def _build_ec_enzymes_list(ec: EcData) -> list[dict[str, Any]]:
    """Translate per-enzyme ec fields to the list-of-mappings YAML form.

    NaN ``mw`` / ``concs`` and empty ``sequence`` are omitted; the loader
    restores them as NaN / empty string.
    """
    out: list[dict[str, Any]] = []
    for j in range(ec.n_enzymes):
        entry: dict[str, Any] = {
            "genes": ec.genes[j],
            "enzymes": ec.enzymes[j],
        }
        if not math.isnan(ec.mw[j]):
            entry["mw"] = float(ec.mw[j])
        if ec.sequence[j]:
            entry["sequence"] = ec.sequence[j]
        if not math.isnan(ec.concs[j]):
            entry["concs"] = float(ec.concs[j])
        out.append(entry)
    return out


# --------------------------------------------------------------------------- #
# eccodes representation helpers
# --------------------------------------------------------------------------- #

def _canonicalize_eccodes(value) -> str:
    """Convert an EC-codes field to a single `;`-joined string.

    The schema accepts a scalar string (`"1.1.1.1"`) or a list of
    strings (`["1.1.1.1", "1.1.99.40"]`); a scalar is returned unchanged and a list
    is joined with `;`. `None` gives an empty string.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return ";".join(str(v) for v in value)


def _eccodes_to_yaml(eccodes: str):
    """Convert the internal `;`-joined eccodes string back to the YAML form:
    a scalar string for one EC, a list for multiple."""
    parts = [p for p in eccodes.split(";") if p]
    if len(parts) <= 1:
        return ";".join(parts)
    return parts
