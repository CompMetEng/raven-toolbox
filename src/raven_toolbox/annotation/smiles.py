"""SMILES of metabolites from PubChem, with a local cache.

:func:`find_met_smiles` looks up the SMILES of the metabolites of a model by name and writes it
to the annotation of each metabolite; :func:`met_smiles` reads it back. Looking up a name needs
network access to PubChem, so every result is stored in a cache file that later calls read first.

The cache file is a tab-delimited file without header, with the columns ``name`` and ``smiles``.
Each result is appended as soon as it is known, so an interrupted run continues where it
stopped.
"""
from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import quote

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

if TYPE_CHECKING:
    import cobra


__all__ = ["find_met_smiles", "met_smiles"]

logger = logging.getLogger(__name__)

# PubChem deprecated the `CanonicalSMILES` property in 2025 in favour of
# `SMILES` (its canonical/absolute SMILES); `ConnectivitySMILES` replaces
# the old `IsomericSMILES`.
_PUBCHEM_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
    "{name}/property/SMILES/TXT"
)
_TIMEOUT_SECONDS = 30
_NUM_RETRIES = 10
_INTER_REQUEST_SLEEP = 0.05  # seconds; PubChem allows ~5 req/s
_PROT_PATTERN = re.compile(r"^prot_")
_FIRST_LINE_PATTERN = re.compile(r"^(\S*)\n")


def met_smiles(metabolite) -> str:
    """The SMILES of a metabolite, whatever form its annotation has.

    cobra keeps an annotation value as it was assigned, but returns a list for a model that was
    read from a file: a metabolite annotated in memory by :func:`find_met_smiles` has ``"CCO"``, and
    the same metabolite after it is saved and loaded has ``["CCO"]``. This function returns a string
    in both cases.

    Parameters
    ----------
    metabolite : cobra.Metabolite
        The metabolite.

    Returns
    -------
    str
        The first non-empty SMILES in ``metabolite.annotation["smiles"]``, or ``""`` if there is none.

    Examples
    --------
    >>> import cobra
    >>> from raven_toolbox.annotation import met_smiles
    >>> met = cobra.Metabolite("etoh")
    >>> met.annotation["smiles"] = ["CCO"]
    >>> met_smiles(met)
    'CCO'
    """
    value = metabolite.annotation.get("smiles", "")
    if isinstance(value, (list, tuple, set)):
        return next((str(v) for v in value if v), "")
    return str(value) if value else ""


def find_met_smiles(
    model: "cobra.Model",
    *,
    cache_path: str | Path,
) -> None:
    r"""Write the SMILES of the metabolites of a model to their annotation, from PubChem.

    The SMILES of a metabolite is found by its name (``Metabolite.name``) and written to
    ``metabolite.annotation["smiles"]``. A lookup is done once per distinct name and applies to every
    metabolite with that name, for example the same metabolite in several compartments. For each
    metabolite:

    1. Metabolites whose name starts with ``prot_`` are skipped.
    2. Metabolites without a name are skipped, with one warning that lists them.
    3. Metabolites that already have a SMILES are skipped.
    4. The name is looked up in the cache file. If the cache holds a name more than once, the last
       row is used.
    5. If the name is not in the cache, PubChem is asked, one request after the other with a pause
       of 0.05 s, and failed requests with a status such as 429 or 503 are retried up to 10 times.
       The first SMILES that PubChem returns is used. The result is appended to the cache.

    A name for which PubChem returns no SMILES gets a cache row with an empty SMILES, so that it is
    not asked for again; the same happens when the request fails because the network is not
    available. Delete such rows from the cache to look them up again. A metabolite without a SMILES
    gets no annotation.

    Network access to ``pubchem.ncbi.nlm.nih.gov`` is needed only for names that are not in the
    cache.

    Parameters
    ----------
    model : cobra.Model
        The model, changed in place.
    cache_path : str or pathlib.Path
        The cache file. It is created, with its folder, if it does not exist.

    See Also
    --------
    met_smiles : Reads the SMILES of a metabolite.

    Examples
    --------
    >>> import cobra, pathlib, tempfile
    >>> from raven_toolbox.annotation import find_met_smiles, met_smiles
    >>> cache = pathlib.Path(tempfile.mkdtemp()) / "smiles.tsv"
    >>> _ = cache.write_text("ethanol\tCCO\n")
    >>> model = cobra.Model("toy")
    >>> model.add_metabolites([cobra.Metabolite("etoh_c", name="ethanol")])
    >>> find_met_smiles(model, cache_path=cache)
    >>> met_smiles(model.metabolites.etoh_c)
    'CCO'
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Build map: metabolite name -> list of metabolite objects sharing it.
    name_to_mets: dict[str, list] = {}
    unnamed = []
    for met in model.metabolites:
        name = met.name
        if not name:
            unnamed.append(met.id)
            continue
        if _PROT_PATTERN.match(name):
            continue
        if met.annotation.get("smiles"):
            continue
        name_to_mets.setdefault(name, []).append(met)

    if unnamed:
        logger.warning(
            "find_met_smiles: %d metabolite(s) have no name and get no SMILES: %s",
            len(unnamed), ", ".join(unnamed),
        )

    if not name_to_mets:
        logger.info("find_met_smiles: nothing to look up.")
        return

    # Load existing cache.
    cache = _load_cache(cache_path)
    logger.info(
        "find_met_smiles: %d unique metabolite name(s) need SMILES; "
        "%d already in cache.",
        len(name_to_mets),
        sum(1 for n in name_to_mets if n in cache),
    )

    # Apply cached SMILES first (no network).
    for name in list(name_to_mets):
        if name in cache:
            smiles = cache[name]
            for met in name_to_mets[name]:
                if smiles:
                    met.annotation["smiles"] = smiles
            del name_to_mets[name]

    if not name_to_mets:
        logger.info("find_met_smiles: all matches resolved from cache.")
        return

    # Hit PubChem for the rest.
    session = _make_session()
    n_found = 0
    n_missing = 0

    for name in sorted(name_to_mets):
        smiles = _fetch_one_smiles(session, name)
        # Append to cache immediately, including empty results so we
        # do not retry next time.
        _append_cache(cache_path, name, smiles)

        if smiles:
            for met in name_to_mets[name]:
                met.annotation["smiles"] = smiles
            n_found += 1
        else:
            n_missing += 1
            logger.debug("No SMILES for: %s", name)

        time.sleep(_INTER_REQUEST_SLEEP)

    total = n_found + n_missing
    if total > 0:
        pct = 100.0 * n_found / total
        logger.info(
            "find_met_smiles: SMILES found for %d/%d unique names (%.0f%%); "
            "%d missing.",
            n_found, total, pct, n_missing,
        )


def _load_cache(path: Path) -> dict[str, str]:
    """Load the two-column TSV cache. Returns an empty dict if missing."""
    cache: dict[str, str] = {}
    if not path.is_file():
        return cache
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                cache[parts[0]] = parts[1]
            elif len(parts) == 1 and parts[0]:
                cache[parts[0]] = ""
    return cache


def _append_cache(path: Path, name: str, smiles: str) -> None:
    """Append one row to the TSV cache. Open and close per call so
    interrupted runs leave a consistent file."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{name}\t{smiles}\n")


def _make_session() -> requests.Session:
    """A requests.Session with retry/backoff for transient errors."""
    session = requests.Session()
    retry = Retry(
        total=_NUM_RETRIES,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _fetch_one_smiles(session: requests.Session, name: str) -> str:
    """Fetch SMILES from PubChem for one metabolite name. Returns
    empty string if unfound or unrecoverable error."""
    url = _PUBCHEM_URL.format(name=quote(name, safe=""))
    try:
        response = session.get(url, timeout=_TIMEOUT_SECONDS)
    except requests.RequestException as e:
        logger.warning("PubChem request failed for %r: %s", name, e)
        return ""

    if response.status_code in (400, 404, 500):
        return ""
    if response.status_code != 200:
        logger.warning(
            "PubChem returned HTTP %d for %r; treating as no match.",
            response.status_code, name,
        )
        return ""

    text = response.text
    if not text:
        return ""

    # Multiple SMILES may come on separate lines; take only the first.
    match = _FIRST_LINE_PATTERN.match(text)
    if match:
        return match.group(1).strip()
    return text.split("\n", 1)[0].strip()
