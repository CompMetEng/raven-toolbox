"""Tests for scripts/make_registry_snippet.py registry-entry helpers."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

# scripts/ is not a package; load the module directly by path.
_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_registry_snippet.py"
_spec = importlib.util.spec_from_file_location("make_registry_snippet", _SCRIPT)
mrs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mrs)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_data_entry_lists_files_with_urls_and_checksums(tmp_path):
    (tmp_path / "reference_model.yml.gz").write_bytes(b"model")
    (tmp_path / "ko_reaction.tsv.gz").write_bytes(b"table")
    (tmp_path / ".hidden").write_bytes(b"skip")  # hidden files ignored

    entry = mrs.data_entry("kegg", "kegg116", "https://x/rel/", tmp_path)
    assert entry["version"] == "kegg116"
    assert set(entry["files"]) == {"reference_model.yml.gz", "ko_reaction.tsv.gz"}
    ref = entry["files"]["reference_model.yml.gz"]
    assert ref["url"] == "https://x/rel/reference_model.yml.gz"  # trailing slash collapsed
    assert ref["sha256"] == _sha(b"model")


def test_data_entry_empty_dir_errors(tmp_path):
    with pytest.raises(SystemExit):
        mrs.data_entry("kegg", "v1", "https://x", tmp_path)


def test_binary_entry_parses_platform_from_filename(tmp_path):
    (tmp_path / "blast-2.16.0-linux-x86_64.zip").write_bytes(b"linux")
    (tmp_path / "blast-2.16.0-macos-arm64.zip").write_bytes(b"mac")
    (tmp_path / "other-1.0-linux-x86_64.zip").write_bytes(b"nope")  # different bundle

    entry = mrs.binary_entry("blast", "2.16.0", ["blastp", "makeblastdb"], "https://x", tmp_path)
    assert entry["provides"] == ["blastp", "makeblastdb"]
    assert set(entry["platforms"]) == {"linux-x86_64", "macos-arm64"}
    assert entry["platforms"]["macos-arm64"]["sha256"] == _sha(b"mac")
    assert entry["platforms"]["linux-x86_64"]["url"].endswith("blast-2.16.0-linux-x86_64.zip")


def test_binary_entry_no_zips_errors(tmp_path):
    with pytest.raises(SystemExit):
        mrs.binary_entry("blast", "2.16.0", ["blastp"], "https://x", tmp_path)


def test_release_base_url_builds_releases_download_prefix():
    # Default repo + an arbitrary tag → the canonical GitHub release-asset prefix.
    assert (
        mrs.release_base_url("v0.3.0")
        == "https://github.com/CompMetEng/toolbox-assets/releases/download/v0.3.0"
    )
    # --repo override is honoured; the fixed "releases/download" path is always present.
    assert (
        mrs.release_base_url("blast-2.16.0", repo="ORG/toolbox-assets")
        == "https://github.com/ORG/toolbox-assets/releases/download/blast-2.16.0"
    )


def test_render_is_valid_json_round_trip():
    entry = {"version": "v1", "files": {"a": {"url": "u", "sha256": "s"}}}
    text = mrs.render("kegg", entry)
    assert json.loads(text) == {"kegg": entry}
