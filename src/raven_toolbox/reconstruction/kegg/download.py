"""Download and arrange a local KEGG flat-file dump.

Maintainer-side build-time tooling. Ports ``fetch_keggdb.sh``: fetches the KEGG
FTP source archives, extracts them, and moves and concatenates the files that the
parser (:mod:`.parse`) and HMM build (:mod:`.hmm`) consume. It uses the Python standard library
(``urllib``, ``tarfile``, ``gzip``, ``netrc``) and ``tqdm`` for progress bars, so it
needs neither ``wget``, ``tar``, ``gunzip`` nor Cygwin on Windows. The username and
password of a paid KEGG subscription are read from ``~/.netrc`` (mode 600) and are not
passed on the command line.

Requires an active KEGG FTP subscription. Add to ``~/.netrc``::

    machine ftp.kegg.net login YOUR_USER password YOUR_PASS

Example, run once per KEGG release::

    from raven_toolbox.reconstruction.kegg import download_kegg_dump, parse_kegg_dump
    download_kegg_dump("keggdb")            # -> keggdb/{reaction,compound,ko,...}
    parse_kegg_dump("keggdb", "artefacts")  # -> reference model + gzipped TSVs

The arranged dump contains: ``reaction``, ``reaction.lst``,
``reaction_mapformula.lst``, ``compound`` (compound + glycan concatenated),
``compound.inchi``, ``ko``, ``genes.pep`` (eukaryote + prokaryote proteomes
concatenated), and ``taxonomy``.
"""
from __future__ import annotations

import gzip
import logging
import netrc
import shutil
import tarfile
import urllib.request
from pathlib import Path

from tqdm import tqdm

logger = logging.getLogger(__name__)

KEGG_HOST = "ftp.kegg.net"
BASE_URL = "https://ftp.kegg.net"

# KEGG FTP paths fetched, mirroring fetch_keggdb.sh.
DEFAULT_FILES: tuple[str, ...] = (
    "kegg/ligand/reaction.tar.gz",
    "kegg/ligand/compound.tar.gz",
    "kegg/ligand/glycan.tar.gz",
    "kegg/genes/ko.tar.gz",
    "kegg/genes/fasta/eukaryotes.pep.gz",
    "kegg/genes/fasta/prokaryotes.pep.gz",
    "kegg/genes/misc/taxonomy",
)


# --------------------------------------------------------------------------- #
# Credentials
# --------------------------------------------------------------------------- #
def _resolve_auth(
    host: str,
    *,
    netrc_path: str | Path | None = None,
    auth: tuple[str, str] | None = None,
) -> tuple[str, str]:
    """Return ``(user, password)`` for ``host`` from ``auth`` or a ``.netrc`` file."""
    if auth is not None:
        return auth
    path = Path(netrc_path) if netrc_path else Path.home() / ".netrc"
    if not path.is_file():
        raise FileNotFoundError(
            f"No credentials given and {path} does not exist. Create it (chmod 600) "
            f"with a line:\n    machine {host} login YOUR_USER password YOUR_PASS"
        )
    try:
        creds = netrc.netrc(str(path)).authenticators(host)
    except (netrc.NetrcParseError, OSError) as exc:
        raise ValueError(
            f"Could not read credentials from {path}: {exc}. Ensure it is a valid "
            f".netrc (chmod 600) with a line:\n"
            f"    machine {host} login YOUR_USER password YOUR_PASS"
        ) from exc
    if not creds:
        raise ValueError(
            f"No credentials for '{host}' in {path}. Add a line:\n"
            f"    machine {host} login YOUR_USER password YOUR_PASS"
        )
    login, _, password = creds
    if not login or not password:
        raise ValueError(f"Incomplete credentials for '{host}' in {path}.")
    return login, password


def _build_opener(base_url: str, user: str, password: str) -> urllib.request.OpenerDirector:
    mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    mgr.add_password(None, base_url, user, password)
    return urllib.request.build_opener(
        urllib.request.HTTPBasicAuthHandler(mgr),
        urllib.request.HTTPDigestAuthHandler(mgr),
    )


# --------------------------------------------------------------------------- #
# Fetch
# --------------------------------------------------------------------------- #
def _copy_with_progress(
    src, dst, *, total: int | None, desc: str, progress: bool, chunk: int = 1 << 20
) -> None:
    """Copy ``src`` to ``dst`` in chunks, advancing a tqdm byte bar.

    ``total`` (for example from a ``Content-Length`` header) gives the bar a percentage;
    ``None`` (unknown size) gives a byte count with a transfer rate. The
    bar is shown only when ``progress`` is true.
    """
    with tqdm(
        total=total, desc=desc, unit="B", unit_scale=True, unit_divisor=1024,
        disable=not progress,
    ) as bar:
        while buf := src.read(chunk):
            dst.write(buf)
            bar.update(len(buf))


def fetch_kegg_files(
    dest: str | Path,
    *,
    files: tuple[str, ...] = DEFAULT_FILES,
    base_url: str = BASE_URL,
    host: str = KEGG_HOST,
    auth: tuple[str, str] | None = None,
    netrc_path: str | Path | None = None,
    force: bool = False,
    verbose: bool = True,
    progress: bool = True,
) -> list[Path]:
    """Download the raw KEGG archives into ``dest``, named by their basenames, and return the paths.

    Existing files are skipped unless ``force=True``. ``progress`` shows a per-file byte
    download bar, sized from the response ``Content-Length``; ``verbose`` enables the
    ``fetching`` and ``skip`` INFO log lines.

    Parameters
    ----------
    dest : str or Path
        Directory that receives the archives. It is created, with parents, if missing.
    files : tuple of str
        KEGG FTP paths to download, relative to ``base_url``. Each file is saved under its
        basename. Defaults to ``DEFAULT_FILES``: the reaction, compound, glycan and ko
        archives, the eukaryote and prokaryote proteomes, and the taxonomy file.
    base_url : str
        Base URL of the server. Defaults to ``"https://ftp.kegg.net"``. A trailing slash
        is removed.
    host : str
        Host name looked up in the ``.netrc`` file. Defaults to ``"ftp.kegg.net"``. It is
        unused when ``auth`` is given.
    auth : tuple of (str, str) or None
        ``(user, password)`` for HTTP basic or digest authentication. When None (default),
        the credentials of ``host`` are read from the ``.netrc`` file. A
        ``FileNotFoundError`` is raised if that file does not exist, and a ``ValueError``
        if it cannot be parsed or has no complete entry for ``host``.
    netrc_path : str or Path or None
        Path of the ``.netrc`` file. When None (default), ``~/.netrc`` is used. Ignored
        when ``auth`` is given.
    force : bool
        If True, files that already exist in ``dest`` are downloaded again. If False
        (default), they are skipped.
    verbose : bool
        If True (default), log a ``fetching`` line for each download and a ``skip`` line
        for each existing file at INFO level.
    progress : bool
        If True (default), show a byte progress bar for each download. The total comes
        from the ``Content-Length`` header; without it the bar shows a byte count.

    Returns
    -------
    list of Path
        Paths of the files in ``dest``, in the order of ``files``, including skipped
        files.
    """
    user, password = _resolve_auth(host, netrc_path=netrc_path, auth=auth)
    opener = _build_opener(base_url, user, password)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    out: list[Path] = []
    for path in files:
        target = dest / Path(path).name
        if target.exists() and not force:
            if verbose:
                logger.info("skip (exists): %s", target.name)
            out.append(target)
            continue
        url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
        if verbose:
            logger.info("fetching %s", path)
        with opener.open(url) as resp, open(target, "wb") as handle:
            length = resp.headers.get("Content-Length")
            _copy_with_progress(
                resp, handle,
                total=int(length) if length else None,
                desc=target.name, progress=progress,
            )
        out.append(target)
    return out


# --------------------------------------------------------------------------- #
# Extract / arrange
# --------------------------------------------------------------------------- #
def _gunzip(src: Path, target: Path, *, progress: bool = False) -> None:
    with gzip.open(src, "rb") as fh, open(target, "wb") as out:
        # Decompressed size is unknown up front, so the bar counts output bytes
        # with a rate rather than a percentage (the .pep proteomes are large).
        _copy_with_progress(fh, out, total=None, desc=f"gunzip {src.name}", progress=progress)


def _concat(sources: list[Path], target: Path) -> None:
    with open(target, "wb") as out:
        for src in sources:
            with open(src, "rb") as fh:
                shutil.copyfileobj(fh, out)


def extract_kegg_dump(dest: str | Path, *, progress: bool = False) -> dict[str, Path]:
    """Extract and arrange the downloaded archives into the flat dump layout.

    Mirrors the extract step of ``fetch_keggdb.sh``: untars the ``*.tar.gz`` archives,
    gunzips the ``*.pep.gz`` proteomes, moves the needed files out of their
    sub-directories, and concatenates compound and glycan, and the two proteomes.
    Tar extraction uses the ``data`` filter, which rejects paths outside ``dest``. Returns
    a mapping of logical name to path for the files produced. ``progress`` shows a
    bar over the untar step and a byte bar per proteome gunzip.

    The function makes no network access; :func:`download_kegg_dump` runs
    :func:`fetch_kegg_files` before it.

    Parameters
    ----------
    dest : str or Path
        Directory that holds the downloaded archives. The extracted files are written
        into it, and the archives and sub-directories are deleted. A
        ``FileNotFoundError`` is raised if the reaction, compound or ko archive did not
        yield its file.
    progress : bool
        If True, show a progress bar over the untar step and a byte bar for each proteome
        gunzip. Defaults to False.
    """
    dest = Path(dest)

    tars = sorted(dest.glob("*.tar.gz"))
    for tar_path in tqdm(tars, desc="untar", unit="archive", disable=not progress):
        with tarfile.open(tar_path) as tar:
            tar.extractall(dest, filter="data")
        tar_path.unlink()

    for gz_path in sorted(dest.glob("*.gz")):  # only the .pep.gz remain
        _gunzip(gz_path, gz_path.with_suffix(""), progress=progress)
        gz_path.unlink()

    def lift(rel: str, tmp: str) -> Path | None:
        src = dest / rel
        if src.is_file():
            shutil.move(str(src), str(dest / tmp))
            return dest / tmp
        return None

    reaction = lift("reaction/reaction", "_reaction")
    lift("reaction/reaction.lst", "reaction.lst")
    lift("reaction/reaction_mapformula.lst", "reaction_mapformula.lst")
    compound = lift("compound/compound", "_compound")
    lift("compound/compound.inchi", "compound.inchi")
    glycan = lift("glycan/glycan", "_glycan")
    ko = lift("ko/ko", "_ko")

    for subdir in ("reaction", "compound", "glycan", "ko"):
        path = dest / subdir
        if path.is_dir():
            shutil.rmtree(path)

    missing = [n for n, p in (("reaction", reaction), ("compound", compound), ("ko", ko)) if p is None]
    if missing:
        raise FileNotFoundError(
            f"KEGG archives did not yield required file(s): {missing}. "
            f"Check that the source .tar.gz archives are present in {dest}."
        )
    # Narrowing for the type checker: the `missing` check above raised if any were None.
    assert reaction is not None and compound is not None and ko is not None

    shutil.move(str(reaction), str(dest / "reaction"))
    shutil.move(str(ko), str(dest / "ko"))
    if glycan is not None:
        _concat([compound, glycan], dest / "compound")
        compound.unlink()
        glycan.unlink()
    else:
        shutil.move(str(compound), str(dest / "compound"))

    peps = [p for p in (dest / "eukaryotes.pep", dest / "prokaryotes.pep") if p.is_file()]
    if peps:
        _concat(peps, dest / "genes.pep")
        for pep in peps:
            pep.unlink()

    result: dict[str, Path] = {}
    for name in (
        "reaction",
        "reaction.lst",
        "reaction_mapformula.lst",
        "compound",
        "compound.inchi",
        "ko",
        "genes.pep",
        "taxonomy",
    ):
        path = dest / name
        if path.is_file():
            result[name] = path
    return result


def download_kegg_dump(
    dest: str | Path,
    *,
    files: tuple[str, ...] = DEFAULT_FILES,
    base_url: str = BASE_URL,
    host: str = KEGG_HOST,
    auth: tuple[str, str] | None = None,
    netrc_path: str | Path | None = None,
    force: bool = False,
    verbose: bool = True,
    progress: bool = True,
) -> dict[str, Path]:
    """Fetch and arrange a complete KEGG dump into ``dest``.

    Runs :func:`fetch_kegg_files` and then :func:`extract_kegg_dump`. Returns the mapping of
    logical name to path of the arranged dump, the input of
    :func:`raven_toolbox.reconstruction.kegg.parse_kegg_dump`.

    Parameters
    ----------
    dest : str or Path
        Directory that receives the dump. It is created, with parents, if missing.
    files : tuple of str
        KEGG FTP paths to download, relative to ``base_url``. Defaults to
        ``DEFAULT_FILES``. See :func:`fetch_kegg_files`.
    base_url : str
        Base URL of the server. Defaults to ``"https://ftp.kegg.net"``.
    host : str
        Host name looked up in the ``.netrc`` file. Defaults to ``"ftp.kegg.net"``.
        Unused when ``auth`` is given.
    auth : tuple of (str, str) or None
        ``(user, password)`` for the server. When None (default), the credentials are read
        from the ``.netrc`` file.
    netrc_path : str or Path or None
        Path of the ``.netrc`` file. When None (default), ``~/.netrc`` is used. Ignored
        when ``auth`` is given.
    force : bool
        If True, archives that already exist in ``dest`` are downloaded again. If False
        (default), they are skipped.
    verbose : bool
        If True (default), log the download, skip and extraction steps at INFO level.
    progress : bool
        If True (default), show download and extraction progress bars.

    Returns
    -------
    dict of str to Path
        Logical name to path of each arranged file that exists in ``dest``.
    """
    fetch_kegg_files(
        dest,
        files=files,
        base_url=base_url,
        host=host,
        auth=auth,
        netrc_path=netrc_path,
        force=force,
        verbose=verbose,
        progress=progress,
    )
    if verbose:
        logger.info("Extracting and arranging KEGG dump...")
    return extract_kegg_dump(dest, progress=progress)
