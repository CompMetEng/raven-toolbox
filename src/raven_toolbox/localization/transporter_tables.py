"""Curated tables from transporter family to coarse substrate class.

These hand-maintained tables are the coarse substrate layer of the evidence-aware transport scoring
(see the `design
<https://github.com/CompMetEng/raven-gecko-parity/blob/develop/docs/localization/transport_evidence_scoring.md>`_
on raven-gecko-parity). They have two consumers:

* ``scripts/build_transporter_data.py`` reads :data:`PFAM_TRANSPORTERS` to select the Pfam HMMs to
  fetch (from InterPro) into the bundled transporter HMM database.
* The annotation backend maps the Pfam family or TCDB TC-number of a gene to a coarse substrate
  class via :data:`PFAM_TRANSPORTERS` and :data:`TC_FAMILY_CLASS`, so that a transporter can be
  matched to the metabolites it plausibly carries.

Classes are drawn from :data:`COARSE_CLASSES`. Many carriers transport several substrate types, so a
family maps to a set of classes; a match succeeds on any overlap with the class(es) of the
metabolite. The build script verifies every accession against the live InterPro API and reports an
accession that is outdated or misspelled. The per-substrate ChEBI mapping (TCDB substrate table and
ChEBI ontology roll-up, see :mod:`raven_toolbox.localization.substrate_ontology`) is applied on top
of this coarse fallback.
"""
from __future__ import annotations

# Shared coarse substrate vocabulary — both metabolites and transporters map into this.
COARSE_CLASSES: frozenset[str] = frozenset({
    "sugar", "amino_acid", "carboxylate", "nucleotide", "nucleoside_base", "inorganic_ion",
    "phosphate_sulfate", "lipid_fatty_acid", "cofactor_vitamin", "amine_polyamine", "peptide", "other",
})

# Pfam accession -> (family name, coarse substrate classes). The build script fetches an HMM per key.
PFAM_TRANSPORTERS: dict[str, tuple[str, frozenset[str]]] = {
    # Mitochondrial carrier family (SLC25) — the cytosol<->mito carriers (malate/2-OG/citrate/OAA/
    # ATP-ADP/Pi/carnitine); the c<->m shuttles a blanket transport penalty wrongly drops.
    "PF00153": ("Mito_carr", frozenset({"carboxylate", "nucleotide", "cofactor_vitamin",
                                        "amino_acid"})),
    # Major Facilitator Superfamily (broad; sugars, drugs, organic acids)
    "PF07690": ("MFS_1", frozenset({"sugar", "carboxylate", "other"})),
    "PF05977": ("MFS_2", frozenset({"sugar", "other"})),
    "PF00083": ("Sugar_tr", frozenset({"sugar"})),
    # ABC transporters (very broad; keep low-specificity)
    "PF00005": ("ABC_tran", frozenset({"other"})),
    "PF00664": ("ABC_membrane", frozenset({"other"})),
    # Amino-acid transporters / permeases
    "PF00324": ("AA_permease", frozenset({"amino_acid"})),
    "PF13520": ("AA_permease_2", frozenset({"amino_acid"})),
    "PF01490": ("Aa_trans", frozenset({"amino_acid"})),
    # Aquaporins (water / glycerol / small neutral solutes)
    "PF00230": ("MIP", frozenset({"other"})),
    # Cation / proton antiporters + P-type ATPases (ions)
    "PF00999": ("Na_H_Exchanger", frozenset({"inorganic_ion"})),
    "PF00122": ("E1-E2_ATPase", frozenset({"inorganic_ion"})),
    "PF00689": ("Cation_ATPase_C", frozenset({"inorganic_ion"})),
    "PF02535": ("Zip", frozenset({"inorganic_ion"})),
    "PF01566": ("Nramp", frozenset({"inorganic_ion"})),
    # Nucleobase / nucleoside transporters
    "PF00860": ("Xan_ur_permease", frozenset({"nucleoside_base"})),
    "PF01733": ("Nucleoside_tran", frozenset({"nucleoside_base", "nucleotide"})),
    # Peptide transporters
    "PF00854": ("PTR2", frozenset({"peptide", "amino_acid"})),
    "PF03169": ("OPT", frozenset({"peptide"})),
    # Phosphate / triose-phosphate translocators
    "PF03151": ("TPT", frozenset({"phosphate_sulfate", "sugar"})),
    # Sodium:solute symporters
    "PF00474": ("SSF", frozenset({"sugar", "amino_acid"})),
    # Multidrug / drug-metabolite exporters (broad)
    "PF01554": ("MatE", frozenset({"other"})),
    "PF00893": ("Multi_Drug_Res", frozenset({"other"})),
    "PF00892": ("EamA", frozenset({"other"})),
}

# TCDB family prefix (first three TC levels) -> coarse substrate classes. Used when a DIAMOND hit to
# TCDB gives a TC number but no entry-specific substrate.
TC_FAMILY_CLASS: dict[str, frozenset[str]] = {
    "2.A.29": frozenset({"carboxylate", "nucleotide", "cofactor_vitamin"}),  # mitochondrial carrier
    "2.A.1": frozenset({"sugar", "carboxylate", "other"}),                    # MFS
    "2.A.2": frozenset({"sugar"}),                                            # GPH (glycoside-pentoside)
    "2.A.3": frozenset({"amino_acid", "amine_polyamine"}),                    # APC
    "2.A.4": frozenset({"inorganic_ion"}),                                    # CDF (cation diffusion)
    "2.A.5": frozenset({"inorganic_ion"}),                                    # ZIP
    "2.A.7": frozenset({"other"}),                                            # DMT
    "2.A.17": frozenset({"peptide"}),                                         # POT
    "2.A.21": frozenset({"sugar", "amino_acid"}),                             # SSS
    "2.A.47": frozenset({"carboxylate"}),                                     # DASS (dicarboxylate)
    "2.A.11": frozenset({"carboxylate"}),                                     # CitMHS
    "2.A.6": frozenset({"other"}),                                            # RND
    "2.A.66": frozenset({"other"}),                                           # MOP / MATE
    "3.A.1": frozenset({"other"}),                                            # ABC
    "3.A.3": frozenset({"inorganic_ion"}),                                    # P-type ATPase
    "1.A.8": frozenset({"other"}),                                            # MIP / aquaporin
}
