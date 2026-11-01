"""Tests for raven_toolbox.io.excel (exportToExcelFormat port, export only)."""
import cobra
import pytest

openpyxl = pytest.importorskip("openpyxl")

import numpy as np
from scipy import sparse

from raven_toolbox.io import EcData, export_to_excel
from raven_toolbox.manipulation import add_reactions_from_equations


@pytest.fixture
def model():
    m = cobra.Model("yeastGEM")
    m.name = "Yeast"
    m.compartments = {"c": "cytoplasm"}
    m.notes["metaData"] = {"taxonomy": "taxonomy/559292", "defaultLB": "-1000"}
    m.add_metabolites(
        [
            cobra.Metabolite("atp_c", name="ATP", formula="C10H16N5O13P3", charge=-4, compartment="c"),
            cobra.Metabolite("adp_c", name="ADP", compartment="c"),
        ]
    )
    m.metabolites.atp_c.annotation = {"kegg.compound": ["C00002"], "smiles": ["C1=NC"],
                                      "inchi": ["InChI=1S/X"]}
    m.metabolites.atp_c.notes = {"inchis": "InChI=1S/X"}
    add_reactions_from_equations(
        m,
        [{"id": "R1", "equation": "atp_c <=> adp_c", "name": "rxn one",
          "gene_reaction_rule": "G1", "subsystem": "glycolysis"}],
    )
    r = m.reactions.R1
    r.annotation = {"ec-code": ["1.1.1.1"], "kegg.reaction": ["R00001"]}
    r.notes = {"confidence_score": 2, "note": "a note", "references": "PMID:1"}
    r.objective_coefficient = 1
    return m


def _wb(path):
    return openpyxl.load_workbook(path)


def test_sheets_present(model, tmp_path):
    out = tmp_path / "m.xlsx"
    export_to_excel(model, out)
    wb = _wb(out)
    assert set(wb.sheetnames) == {"RXNS", "METS", "COMPS", "GENES", "MODEL"}


def test_rxns_sheet(model, tmp_path):
    out = tmp_path / "m.xlsx"
    export_to_excel(model, out)
    ws = _wb(out)["RXNS"]
    header = [c.value for c in ws[1]]
    row = {header[i]: c.value for i, c in enumerate(ws[2])}
    assert row["ID"] == "R1"
    assert row["NAME"] == "rxn one"
    assert "ATP[c]" in row["EQUATION"] and "<=>" in row["EQUATION"]
    assert row["EC-NUMBER"] == "1.1.1.1"
    assert row["GENE ASSOCIATION"] == "G1"
    assert row["SUBSYSTEM"] == "glycolysis"
    assert row["OBJECTIVE"] == 1
    assert row["CONFIDENCE SCORE"] == 2
    assert row["NOTE"] == "a note"
    assert row["MIRIAM"] == "kegg.reaction/R00001"  # ec-code excluded (own column)


def test_mets_sheet(model, tmp_path):
    out = tmp_path / "m.xlsx"
    export_to_excel(model, out)
    ws = _wb(out)["METS"]
    header = [c.value for c in ws[1]]
    rows = {
        r[header.index("REPLACEMENT ID")].value: {header[i]: c.value for i, c in enumerate(r)}
        for r in ws.iter_rows(min_row=2)
    }
    atp = rows["atp_c"]
    assert atp["ID"] == "ATP[c]"
    assert atp["NAME"] == "ATP"
    assert atp["InChI"] == "InChI=1S/X"
    assert atp["COMPOSITION"] == "C10H16N5O13P3"  # formula kept even when InChI present
    assert atp["CHARGE"] == -4
    assert atp["MIRIAM"] == "kegg.compound/C00002"  # smiles and inchi excluded


def test_model_sheet(model, tmp_path):
    out = tmp_path / "m.xlsx"
    export_to_excel(model, out)
    ws = _wb(out)["MODEL"]
    header = [c.value for c in ws[1]]
    row = {header[i]: c.value for i, c in enumerate(ws[2])}
    assert row["ID"] == "yeastGEM"
    assert row["NAME"] == "Yeast"
    assert row["TAXONOMY"] == "taxonomy/559292"
    assert row["DEFAULT LOWER"] == "-1000"


def test_genes_sheet(model, tmp_path):
    out = tmp_path / "m.xlsx"
    export_to_excel(model, out)
    ws = _wb(out)["GENES"]
    header = [c.value for c in ws[1]]
    row = {header[i]: c.value for i, c in enumerate(ws[2])}
    assert row["NAME"] == "G1"


def test_no_genes_skips_sheet(tmp_path):
    m = cobra.Model("t")
    m.add_metabolites([cobra.Metabolite("a_c", compartment="c")])
    add_reactions_from_equations(m, [{"id": "R1", "equation": "a_c -->"}])
    out = tmp_path / "m.xlsx"
    export_to_excel(m, out)
    assert "GENES" not in _wb(out).sheetnames


# --------------------------------------------------------------------------- #
# Enzyme-constrained (GECKO) models: ENZYMES / ENZRXNS sheets
# --------------------------------------------------------------------------- #

@pytest.fixture
def ec_data():
    # Two enzymes and two ec-reactions. One MW and one CONC are NaN (the
    # "unknown"/"not measured" sentinels), and one kcat is the 0 "unassigned"
    # sentinel, to check that NaN -> blank while 0 is kept.
    return EcData(
        gecko_light=False,
        rxns=["R1", "R2"],
        kcat=np.array([13.7, 0.0]),
        source=["brenda", ""],
        notes=["note1", ""],
        eccodes=["1.1.1.1", "2.7.1.1;2.7.1.2"],
        genes=["G1", "G2"],
        enzymes=["P1", "P2"],
        mw=np.array([51000.0, np.nan]),
        sequence=["MABC", "MDEF"],
        concs=np.array([np.nan, 0.5]),
        rxn_enz_mat=sparse.csr_matrix(np.array([[1.0, 2.0], [0.0, 1.0]])),
    )


def _rows_by_id(ws):
    header = [c.value for c in ws[1]]
    rows = ws.iter_rows(min_row=2, max_col=len(header), values_only=True)
    return header, {r[1]: dict(zip(header, r, strict=True)) for r in rows}


def test_ec_sheets_absent_without_ec(model, tmp_path):
    out = tmp_path / "m.xlsx"
    export_to_excel(model, out)
    names = set(_wb(out).sheetnames)
    assert "ENZYMES" not in names and "ENZRXNS" not in names


def test_ec_sheets_present(model, ec_data, tmp_path):
    model.ec = ec_data
    out = tmp_path / "m.xlsx"
    export_to_excel(model, out)
    assert {"ENZYMES", "ENZRXNS"} <= set(_wb(out).sheetnames)


def test_enzymes_sheet(model, ec_data, tmp_path):
    model.ec = ec_data
    out = tmp_path / "m.xlsx"
    export_to_excel(model, out)
    ws = _wb(out)["ENZYMES"]
    header, by_id = _rows_by_id(ws)
    assert header == ["#", "ID", "GENE", "MW", "SEQUENCE", "CONC"]
    assert by_id["P1"]["GENE"] == "G1"
    assert by_id["P1"]["MW"] == pytest.approx(51000.0)
    assert by_id["P1"]["SEQUENCE"] == "MABC"
    assert by_id["P1"]["CONC"] is None        # NaN -> blank
    assert by_id["P2"]["MW"] is None          # NaN -> blank
    assert by_id["P2"]["CONC"] == pytest.approx(0.5)
    # MW shown without decimals, CONC with 5 decimals
    assert ws.cell(row=2, column=4).number_format == "0"        # MW
    assert ws.cell(row=3, column=6).number_format == "0.00000"  # CONC (P2=0.5)


def test_enzrxns_sheet(model, ec_data, tmp_path):
    model.ec = ec_data
    out = tmp_path / "m.xlsx"
    export_to_excel(model, out)
    header, by_id = _rows_by_id(_wb(out)["ENZRXNS"])
    assert header == ["#", "ID", "KCAT", "SOURCE", "NOTE", "EC-NUMBER", "ENZYMES"]
    assert by_id["R1"]["KCAT"] == pytest.approx(13.7)
    assert by_id["R1"]["SOURCE"] == "brenda"
    assert by_id["R1"]["NOTE"] == "note1"
    assert by_id["R1"]["EC-NUMBER"] == "1.1.1.1"
    assert by_id["R1"]["ENZYMES"] == "P1:1;P2:2"   # subunit stoichiometry
    assert by_id["R2"]["KCAT"] == 0           # 0 "unassigned" sentinel kept
    assert by_id["R2"]["SOURCE"] is None      # empty -> blank
    assert by_id["R2"]["EC-NUMBER"] == "2.7.1.1;2.7.1.2"
    assert by_id["R2"]["ENZYMES"] == "P2:1"


def test_inchi_column_falls_back_to_annotation(model, tmp_path):
    """Without an ``inchis`` note, the InChI column takes ``annotation['inchi']``."""
    model.metabolites.atp_c.notes = {}
    out = tmp_path / "m.xlsx"
    export_to_excel(model, out)
    ws = _wb(out)["METS"]
    header = [c.value for c in ws[1]]
    atp = next({header[i]: c.value for i, c in enumerate(r)} for r in ws.iter_rows(min_row=2)
               if r[header.index("REPLACEMENT ID")].value == "atp_c")
    assert atp["InChI"] == "InChI=1S/X"
    assert atp["MIRIAM"] == "kegg.compound/C00002"
