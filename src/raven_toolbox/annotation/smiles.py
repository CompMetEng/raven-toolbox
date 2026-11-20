"""SMILES lookup via PubChem with a local TSV cache.

The cache file is a two-column TSV with no header: ``name<TAB>smiles``.
Each successful lookup is appended immediately so interrupted runs
resume gracefully.
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
    """The metabolite's SMILES, whatever shape its annotation is in.

    cobra stores an annotation value as the object it was assigned, and
    returns a *list* for a model read back from a file: a metabolite
    annotated in memory by :func:`find_met_smiles` has ``"CCO"``,
    and the same metabolite after it is saved and loaded has
    ``["CCO"]``. Readers that assume a string get a list that compares
    equal to nothing and is not hashable.

    Returns the first non-empty entry, or ``""`` when there is none.
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
    """Populate ``metabolite.annotation['smiles']`` from PubChem.

    For every metabolite in the model:

    1. Skip if the name starts with ``prot_`` (enzyme pseudometabolite).
    2. Skip, with a warning, if the metabolite has no name.
    3. Skip if ``annotation['smiles']`` is already set.
    4. Look up the metabolite name in the local TSV cache. When the cache
       holds a name more than once, the last row is used.
    5. If still not found, query PubChem; cache the result.

    The same metabolite name may appear on many ``cobra.Metabolite``
    instances (different compartments). Lookup is done once per unique
    name and the SMILES is applied to every metabolite sharing that name.

    Parameters
    ----------
    model
        A cobra.Model. Mutated in place: each metabolite gains
        ``annotation['smiles']`` when a SMILES is found.
    cache_path
        Path to the SMILES cache TSV. The file is created if missing and
        appended to as new SMILES are downloaded.
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
