"""Genome annotation data for gene-identifier mapping.

Port of RAVEN's ``downloadGenomeData`` (fetch a GFF3 + protein FASTA pair
from NCBI Datasets) and ``getGeneData`` (parse the GFF3 into a gene-mapping
table for :func:`raven_toolbox.curation.rename_model_genes`).
"""
from __future__ import annotations

import io
import re
import shutil
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import requests

__all__ = ["download_genome_data", "get_gene_data", "parse_gff_gene_table"]

_ACCESSION_RE = re.compile(r"^GC[FA]_\d+(\.\d+)?$")
_PERCENT_ENCODED_RE = re.compile(r"%([0-9A-Fa-f]{2})")
_GENE_TABLE_COLUMNS = [
    "locus_tag", "old_locus_tag", "GeneID", "gene_name", "GenBank_protein", "UniProt",
]


def download_genome_data(
    accession: str,
    *,
    output_dir: str | Path = ".",
    verbose: bool = True,
) -> tuple[Path, Path]:
    """Download the GFF3 annotation and protein FASTA for an NCBI genome assembly.

    Retrieves both files in one archive via the NCBI Datasets v2 API. When both files
    are already present in ``output_dir``, nothing is fetched; deleting them forces a
    new download.

    Parameters
    ----------
    accession:
        NCBI genome assembly accession, for example ``"GCF_000002595.2"``. Both
        RefSeq (``GCF_``) and GenBank (``GCA_``) prefixes are accepted
        (case-insensitively).
    output_dir:
        Directory to save the downloaded files in (default: current directory).
    verbose:
        Print download progress.

    Returns
    -------
    (gff_file, faa_file):
        Paths to the downloaded GFF3 annotation and protein FASTA files.

    Raises
    ------
    ValueError
        If ``accession`` does not start with ``GCF_`` or ``GCA_``.
    RuntimeError
        If the download or the extraction fails, or the archive lacks the GFF3 or the
        protein FASTA file.
    """
    if not accession.upper().startswith(("GCF_", "GCA_")):
        raise ValueError(
            "Accession must start with 'GCF_' (RefSeq) or 'GCA_' (GenBank). "
            f"Received: {accession!r}."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gff_file = output_dir / f"{accession}_genomic.gff"
    faa_file = output_dir / f"{accession}_protein.faa"

    if gff_file.is_file() and faa_file.is_file():
        if verbose:
            print(f"Genome data for {accession} already present, skipping download.")
        return gff_file, faa_file

    if verbose:
        print(f"Downloading genome data for {accession} from NCBI Datasets...")
    url = (
        "https://api.ncbi.nlm.nih.gov/datasets/v2alpha/genome/accession/"
        f"{accession}/download?include_annotation_type=GENOME_GFF,PROT_FASTA"
    )
    try:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Download failed for {accession}: {exc}") from exc

    with TemporaryDirectory() as tmp:
        extract_dir = Path(tmp)
        try:
            with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                archive.extractall(extract_dir)
        except zipfile.BadZipFile as exc:
            raise RuntimeError(f"Extraction failed for {accession}: {exc}") from exc

        data_parent = extract_dir / "ncbi_dataset" / "data"
        subdirs = (
            [p for p in data_parent.iterdir() if p.is_dir() and not p.name.startswith(".")]
            if data_parent.is_dir() else []
        )
        if not subdirs:
            raise RuntimeError(
                "NCBI Datasets did not return both a GFF3 annotation and a protein "
                f"FASTA for {accession}. The assembly may lack annotation."
            )
        src_gff = subdirs[0] / "genomic.gff"
        src_faa = subdirs[0] / "protein.faa"
        if not src_gff.is_file() or not src_faa.is_file():
            raise RuntimeError(
                "NCBI Datasets did not return both a GFF3 annotation and a protein "
                f"FASTA for {accession}. The assembly may lack annotation."
            )

        shutil.move(str(src_gff), str(gff_file))
        shutil.move(str(src_faa), str(faa_file))

    return gff_file, faa_file


def _decode_gff3(value: str) -> str:
    """Decode GFF3 percent-encoded characters (%XX). Unlike urllib's unquote,
    '+' is left untouched, since GFF3 does not use it to encode spaces."""
    return _PERCENT_ENCODED_RE.sub(lambda m: chr(int(m.group(1), 16)), value)


def _extract_attr(attrs: str, key: str) -> str:
    match = re.search(rf"(?:^|;){re.escape(key)}=([^;]+)", attrs)
    return _decode_gff3(match.group(1)) if match else ""


def _extract_dbxref_field(dbxref: str, prefix: str) -> str:
    """From a comma-separated Dbxref value like
    "Phytozome:Cre16.g651050,GeneID:5723799,GenBank:XM_001698190.2", extract
    the part after "prefix:". Returns "" if not found."""
    if not dbxref:
        return ""
    for part in dbxref.split(","):
        part = part.strip()
        if part.startswith(f"{prefix}:"):
            return part[len(prefix) + 1:]
    return ""


def _parse_gene_attrs(attrs: str) -> dict[str, str]:
    dbxref = _extract_attr(attrs, "Dbxref")
    uniprot = _extract_dbxref_field(dbxref, "UniProtKB/Swiss-Prot") or \
        _extract_dbxref_field(dbxref, "UniProtKB")
    return {
        "id": _extract_attr(attrs, "ID"),
        "name": _extract_attr(attrs, "Name"),
        "locus_tag": _extract_attr(attrs, "locus_tag"),
        "old_locus_tag": _extract_attr(attrs, "old_locus_tag"),
        "geneID": _extract_dbxref_field(dbxref, "GeneID"),
        "uniProt": uniprot,
    }


def _parse_cds_attrs(attrs: str) -> tuple[str, str]:
    parent = _extract_attr(attrs, "Parent")
    protein_id = _extract_attr(attrs, "protein_id")
    if not protein_id:
        # Some files put it in Dbxref as GenBank:WP_...
        protein_id = _extract_dbxref_field(_extract_attr(attrs, "Dbxref"), "GenBank")
    return parent, protein_id


def parse_gff_gene_table(gff_path: str | Path) -> pd.DataFrame:
    """Parse a GFF3 file into a gene-protein mapping table.

    Extracts gene and CDS pairs. The protein accession is taken from each
    CDS's ``protein_id`` (or, failing that, a ``Dbxref=GenBank:...`` entry),
    and the owning gene is resolved through the ``Parent`` chain
    (CDS -> mRNA -> gene for eukaryotes, CDS -> gene for prokaryotes), so
    the ``GenBank_protein`` column matches a protein FASTA's headers.

    Returns
    -------
    pandas.DataFrame
        Columns ``locus_tag``, ``old_locus_tag``, ``GeneID``, ``gene_name``,
        ``GenBank_protein``, ``UniProt``. Deduplicated (a CDS repeats once
        per exon) on every column except ``UniProt``, the same key as ``getGeneData.m``.
    """
    gene_map: dict[str, dict[str, str]] = {}
    mrna_to_gene: dict[str, str] = {}
    empty_gene = {"locus_tag": "", "old_locus_tag": "", "geneID": "", "name": "", "uniProt": ""}
    rows: list[tuple[str, str, str, str, str, str]] = []

    with open(gff_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 9:
                continue
            feature_type = fields[2]
            attrs = fields[8]

            if feature_type == "gene":
                gene = _parse_gene_attrs(attrs)
                if gene["id"]:
                    gene_map[gene["id"]] = gene
            elif feature_type == "mRNA":
                mrna_id = _extract_attr(attrs, "ID")
                parent = _extract_attr(attrs, "Parent")
                if mrna_id and parent:
                    mrna_to_gene[mrna_id] = parent
            elif feature_type == "CDS":
                parent, protein_id = _parse_cds_attrs(attrs)
                if not parent or not protein_id:
                    continue
                gene_key = mrna_to_gene.get(parent, parent)
                gene = gene_map.get(gene_key, empty_gene)
                rows.append((
                    gene["locus_tag"], gene["old_locus_tag"], gene["geneID"],
                    gene["name"], protein_id, gene["uniProt"],
                ))

    table = pd.DataFrame(rows, columns=_GENE_TABLE_COLUMNS)
    return table.drop_duplicates(subset=_GENE_TABLE_COLUMNS[:-1], keep="first").reset_index(drop=True)


def get_gene_data(
    accession_or_path: str | Path,
    *,
    output_file: str | Path | None = None,
    download_dir: str | Path = ".",
) -> pd.DataFrame:
    """Build a gene-mapping table from NCBI genome annotation files.

    Parameters
    ----------
    accession_or_path:
        An NCBI assembly accession (downloaded with
        :func:`download_genome_data` if not already present in
        ``download_dir``), or a path to a local GFF3 file.
    output_file:
        If given, also save the table as a tab-delimited file.
    download_dir:
        Directory to download to when ``accession_or_path`` is an accession.

    Returns
    -------
    pandas.DataFrame
        See :func:`parse_gff_gene_table`.

    Raises
    ------
    ValueError
        If ``accession_or_path`` is neither an existing file nor a
        recognised NCBI assembly accession.
    """
    path = Path(accession_or_path)
    if path.is_file():
        gff_path: str | Path = path
    elif _ACCESSION_RE.match(str(accession_or_path)):
        gff_path, _ = download_genome_data(str(accession_or_path), output_dir=download_dir)
    else:
        raise ValueError(
            f"Input {str(accession_or_path)!r} is neither an existing file nor a "
            "recognised NCBI assembly accession."
        )

    table = parse_gff_gene_table(gff_path)

    if output_file is not None:
        table.to_csv(output_file, sep="\t", index=False)
        print(f"Gene table saved to: {output_file}  ({len(table)} rows)")

    return table
