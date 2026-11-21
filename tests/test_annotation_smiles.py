"""Tests for find_met_smiles (network mocked)."""
import logging

import cobra
import requests

from raven_toolbox.annotation import find_met_smiles


# --------------------------------------------------------------------------- #
# Fake HTTP layer
# --------------------------------------------------------------------------- #

class _FakeResponse:
    def __init__(self, status_code: int, text: str = ""):
        self.status_code = status_code
        self.text = text


def _install_fake_session(monkeypatch, name_to_response: dict):
    """Replace _make_session so URL fetches return canned responses.

    `name_to_response` maps URL-escaped metabolite names to
    (status_code, text). Names not in the map return 404.
    """
    from raven_toolbox.annotation import smiles as mod

    class _FakeSession:
        def get(self, url, timeout=None):
            for fragment, (status, text) in name_to_response.items():
                if fragment in url:
                    return _FakeResponse(status, text)
            return _FakeResponse(404, "")

    monkeypatch.setattr(mod, "_make_session", lambda: _FakeSession())
    # Skip the inter-request sleep in tests.
    monkeypatch.setattr(mod, "_INTER_REQUEST_SLEEP", 0.0)


def _build_model(metabolites: list[tuple[str, str]]) -> cobra.Model:
    """Build a cobra.Model from (id, name) tuples."""
    model = cobra.Model("test")
    for met_id, met_name in metabolites:
        m = cobra.Metabolite(met_id, name=met_name, compartment="c")
        model.add_metabolites([m])
    return model


def _smiles(model: cobra.Model, met_id: str) -> str:
    return model.metabolites.get_by_id(met_id).annotation.get("smiles", "")


# --------------------------------------------------------------------------- #
# Cache hits
# --------------------------------------------------------------------------- #

def test_uses_cache_without_network(monkeypatch, tmp_path):
    """If a SMILES is in the cache, no network call happens."""
    cache_path = tmp_path / "smilesDB.tsv"
    cache_path.write_text("glucose\tOC1OC(CO)C(O)C(O)C1O\n")

    model = _build_model([("glc_c", "glucose")])

    def _fail():
        raise AssertionError("network should not be called")

    from raven_toolbox.annotation import smiles as mod
    monkeypatch.setattr(mod, "_make_session", _fail)

    find_met_smiles(model, cache_path=cache_path)
    assert _smiles(model, "glc_c") == "OC1OC(CO)C(O)C(O)C1O"


def test_cache_with_empty_smiles_skips_network(monkeypatch, tmp_path):
    """A cached empty result means 'we already know there is no SMILES'.
    Should not retry the network for it."""
    cache_path = tmp_path / "smilesDB.tsv"
    cache_path.write_text("unknown_compound\t\n")

    model = _build_model([("u_c", "unknown_compound")])

    def _fail():
        raise AssertionError("network should not be called")

    from raven_toolbox.annotation import smiles as mod
    monkeypatch.setattr(mod, "_make_session", _fail)

    find_met_smiles(model, cache_path=cache_path)
    assert _smiles(model, "u_c") == ""


# --------------------------------------------------------------------------- #
# Network lookups
# --------------------------------------------------------------------------- #

def test_fetches_from_network_when_cache_missing(monkeypatch, tmp_path):
    cache_path = tmp_path / "smilesDB.tsv"
    _install_fake_session(monkeypatch, {
        "glucose": (200, "OC1OC(CO)C(O)C(O)C1O\n"),
    })

    model = _build_model([("glc_c", "glucose")])
    find_met_smiles(model, cache_path=cache_path)

    assert _smiles(model, "glc_c") == "OC1OC(CO)C(O)C(O)C1O"


def test_takes_first_line_when_multiple_returned(monkeypatch, tmp_path):
    """PubChem may return multiple SMILES separated by newlines; the
    function should keep only the first."""
    cache_path = tmp_path / "smilesDB.tsv"
    _install_fake_session(monkeypatch, {
        "ambiguous": (200, "C1CCCCC1\nCCCCCCC\nCCCC\n"),
    })

    model = _build_model([("a_c", "ambiguous")])
    find_met_smiles(model, cache_path=cache_path)

    assert _smiles(model, "a_c") == "C1CCCCC1"


def test_404_means_no_smiles(monkeypatch, tmp_path):
    cache_path = tmp_path / "smilesDB.tsv"
    _install_fake_session(monkeypatch, {})  # everything 404s

    model = _build_model([("x_c", "unknown_compound")])
    find_met_smiles(model, cache_path=cache_path)

    assert _smiles(model, "x_c") == ""


def test_400_is_a_final_no_match_and_is_cached(monkeypatch, tmp_path):
    """PubChem answers 400 for a name it cannot parse; asking again gives the same answer."""
    cache_path = tmp_path / "smilesDB.tsv"
    _install_fake_session(monkeypatch, {"bad%20name": (400, "")})
    model = _build_model([("w_c", "bad name")])
    find_met_smiles(model, cache_path=cache_path)
    assert _smiles(model, "w_c") == ""
    assert "bad name\t\n" in cache_path.read_text()


def test_a_server_error_is_not_cached(monkeypatch, tmp_path, caplog):
    """A 500 is not an answer about the name: no crash, no SMILES, nothing cached."""
    cache_path = tmp_path / "smilesDB.tsv"
    _install_fake_session(monkeypatch, {"weird": (500, "")})
    model = _build_model([("w_c", "weird")])
    with caplog.at_level(logging.WARNING):
        find_met_smiles(model, cache_path=cache_path)
    assert _smiles(model, "w_c") == ""
    assert not cache_path.exists() or cache_path.read_text() == ""
    assert "not cached" in caplog.text


def test_a_network_error_is_not_cached_and_the_next_name_is_still_asked(monkeypatch, tmp_path):
    cache_path = tmp_path / "smilesDB.tsv"
    from raven_toolbox.annotation import smiles as mod

    class _Session:
        def get(self, url, timeout=None):
            if "ethanol" in url:
                raise requests.ConnectionError("no network")
            return _FakeResponse(200, "OC1OC(CO)C(O)C(O)C1O\n")

    monkeypatch.setattr(mod, "_make_session", lambda: _Session())
    monkeypatch.setattr(mod, "_INTER_REQUEST_SLEEP", 0.0)
    model = _build_model([("etoh_c", "ethanol"), ("glc_c", "glucose")])
    find_met_smiles(model, cache_path=cache_path)

    assert _smiles(model, "etoh_c") == ""
    assert _smiles(model, "glc_c") == "OC1OC(CO)C(O)C(O)C1O"
    assert cache_path.read_text() == "glucose\tOC1OC(CO)C(O)C(O)C1O\n"


def test_the_pause_between_requests_keeps_to_five_requests_per_second():
    from raven_toolbox.annotation import smiles as mod
    assert mod._INTER_REQUEST_SLEEP >= 0.2


# --------------------------------------------------------------------------- #
# Cache write-back
# --------------------------------------------------------------------------- #

def test_appends_to_cache_after_fetch(monkeypatch, tmp_path):
    cache_path = tmp_path / "smilesDB.tsv"
    _install_fake_session(monkeypatch, {
        "glucose": (200, "OC1OC(CO)C(O)C(O)C1O\n"),
    })
    model = _build_model([("glc_c", "glucose")])
    find_met_smiles(model, cache_path=cache_path)

    text = cache_path.read_text()
    assert "glucose\tOC1OC(CO)C(O)C(O)C1O\n" in text


def test_caches_empty_result_too(monkeypatch, tmp_path):
    """Even when nothing is found, the empty result is cached so we
    do not retry next time."""
    cache_path = tmp_path / "smilesDB.tsv"
    _install_fake_session(monkeypatch, {})

    model = _build_model([("x_c", "unknown")])
    find_met_smiles(model, cache_path=cache_path)

    text = cache_path.read_text()
    assert "unknown\t\n" in text


def test_existing_cache_preserved_when_appending(monkeypatch, tmp_path):
    """Existing cache entries should not be lost."""
    cache_path = tmp_path / "smilesDB.tsv"
    cache_path.write_text("water\tO\n")

    _install_fake_session(monkeypatch, {
        "ethanol": (200, "CCO\n"),
    })
    model = _build_model([
        ("h2o_c", "water"),
        ("etoh_c", "ethanol"),
    ])
    find_met_smiles(model, cache_path=cache_path)

    text = cache_path.read_text()
    assert "water\tO\n" in text
    assert "ethanol\tCCO\n" in text


# --------------------------------------------------------------------------- #
# Filters and skip conditions
# --------------------------------------------------------------------------- #

def test_skips_the_prefixes_that_are_given(monkeypatch, tmp_path):
    """Metabolite names that start with one of skip_prefixes must not be looked up."""
    cache_path = tmp_path / "smilesDB.tsv"

    requested_urls: list[str] = []

    class _CaptureSession:
        def get(self, url, timeout=None):
            requested_urls.append(url)
            return _FakeResponse(404, "")

    from raven_toolbox.annotation import smiles as mod
    monkeypatch.setattr(mod, "_make_session", lambda: _CaptureSession())
    monkeypatch.setattr(mod, "_INTER_REQUEST_SLEEP", 0.0)

    model = _build_model([
        ("prot_P1_c", "prot_P1"),
        ("prot_pool_c", "prot_pool"),
        ("pseudo_c", "pseudo_x"),
    ])
    find_met_smiles(model, cache_path=cache_path, skip_prefixes=("prot_", "pseudo_"))

    assert requested_urls == []


def test_no_prefix_is_skipped_by_default(monkeypatch, tmp_path):
    cache_path = tmp_path / "smilesDB.tsv"
    _install_fake_session(monkeypatch, {"prot_P1": (200, "CCO\n")})
    model = _build_model([("prot_P1_c", "prot_P1")])
    find_met_smiles(model, cache_path=cache_path)
    assert _smiles(model, "prot_P1_c") == "CCO"


def test_skips_metabolites_with_existing_smiles(monkeypatch, tmp_path):
    """If a metabolite already has annotation['smiles'], do not overwrite."""
    cache_path = tmp_path / "smilesDB.tsv"
    _install_fake_session(monkeypatch, {
        "glucose": (200, "should_not_be_used\n"),
    })

    model = _build_model([("glc_c", "glucose")])
    model.metabolites.get_by_id("glc_c").annotation["smiles"] = "PRESERVE_ME"

    find_met_smiles(model, cache_path=cache_path)

    assert _smiles(model, "glc_c") == "PRESERVE_ME"


def test_same_name_in_multiple_compartments_gets_same_smiles(
    monkeypatch, tmp_path,
):
    """Glucose in cytoplasm and extracellular compartments share a
    name and therefore one PubChem lookup; both metabolites should
    receive the SMILES."""
    cache_path = tmp_path / "smilesDB.tsv"
    _install_fake_session(monkeypatch, {
        "glucose": (200, "OC1OC(CO)C(O)C(O)C1O\n"),
    })

    model = cobra.Model("test")
    m_c = cobra.Metabolite("glc_c", name="glucose", compartment="c")
    m_e = cobra.Metabolite("glc_e", name="glucose", compartment="e")
    model.add_metabolites([m_c, m_e])

    find_met_smiles(model, cache_path=cache_path)

    assert _smiles(model, "glc_c") == "OC1OC(CO)C(O)C(O)C1O"
    assert _smiles(model, "glc_e") == "OC1OC(CO)C(O)C(O)C1O"


def test_empty_model_does_not_crash(monkeypatch, tmp_path):
    cache_path = tmp_path / "smilesDB.tsv"
    model = cobra.Model("empty")
    find_met_smiles(model, cache_path=cache_path)


# --------------------------------------------------------------------------- #
# Cache file creation
# --------------------------------------------------------------------------- #

def test_creates_cache_file_if_missing(monkeypatch, tmp_path):
    cache_path = tmp_path / "subdir" / "smilesDB.tsv"
    _install_fake_session(monkeypatch, {
        "glucose": (200, "C\n"),
    })
    model = _build_model([("glc_c", "glucose")])
    find_met_smiles(model, cache_path=cache_path)
    assert cache_path.is_file()


# --------------------------------------------------------------------------- #
# URL escaping
# --------------------------------------------------------------------------- #

def test_url_escapes_special_characters(monkeypatch, tmp_path):
    """A name with spaces or slashes should be URL-encoded before being
    inserted into the URL."""
    cache_path = tmp_path / "smilesDB.tsv"
    captured: list[str] = []

    class _CaptureSession:
        def get(self, url, timeout=None):
            captured.append(url)
            return _FakeResponse(404, "")

    from raven_toolbox.annotation import smiles as mod
    monkeypatch.setattr(mod, "_make_session", lambda: _CaptureSession())
    monkeypatch.setattr(mod, "_INTER_REQUEST_SLEEP", 0.0)

    model = _build_model([("x", "L-alanyl-L-glycine")])
    find_met_smiles(model, cache_path=cache_path)

    assert len(captured) == 1
    # Hyphens are URL-safe; we just check the name is in the URL exactly
    # (no spaces, no broken escaping).
    assert "L-alanyl-L-glycine" in captured[0]


def test_url_escapes_spaces_and_slashes(monkeypatch, tmp_path):
    cache_path = tmp_path / "smilesDB.tsv"
    captured: list[str] = []

    class _CaptureSession:
        def get(self, url, timeout=None):
            captured.append(url)
            return _FakeResponse(404, "")

    from raven_toolbox.annotation import smiles as mod
    monkeypatch.setattr(mod, "_make_session", lambda: _CaptureSession())
    monkeypatch.setattr(mod, "_INTER_REQUEST_SLEEP", 0.0)

    model = _build_model([("x", "weird name/here")])
    find_met_smiles(model, cache_path=cache_path)

    assert len(captured) == 1
    assert " " not in captured[0]
    # Unescaped slash would still be valid in a URL but PubChem would
    # interpret it as a path separator. Our quote(safe='') should
    # escape it.
    assert "weird%20name%2Fhere" in captured[0]


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

def test_logs_summary_after_run(monkeypatch, tmp_path, caplog):
    cache_path = tmp_path / "smilesDB.tsv"
    _install_fake_session(monkeypatch, {
        "glucose": (200, "C\n"),
    })
    model = _build_model([
        ("glc_c", "glucose"),
        ("u_c", "unknown_compound"),
    ])

    with caplog.at_level(logging.INFO):
        find_met_smiles(model, cache_path=cache_path)

    assert "SMILES found for" in caplog.text


# --------------------------------------------------------------------------- #
# Reading the annotation back
# --------------------------------------------------------------------------- #

def test_met_smiles_accepts_the_shape_a_loaded_model_carries():
    """cobra returns a list for a model read from a file, a string for one
    annotated in memory. Both mean the same molecule."""
    import cobra

    from raven_toolbox.annotation import met_smiles

    met = cobra.Metabolite("etoh")
    met.annotation["smiles"] = "CCO"
    assert met_smiles(met) == "CCO"

    met.annotation["smiles"] = ["CCO"]
    assert met_smiles(met) == "CCO"

    met.annotation["smiles"] = []
    assert met_smiles(met) == ""

    del met.annotation["smiles"]
    assert met_smiles(met) == ""


def test_duplicate_cache_name_uses_the_last_row(monkeypatch, tmp_path):
    cache_path = tmp_path / "smilesDB.tsv"
    cache_path.write_text("glucose\tFIRST\nglucose\tLAST\n")
    model = _build_model([("glc_c", "glucose")])
    find_met_smiles(model, cache_path=cache_path)
    assert _smiles(model, "glc_c") == "LAST"


def test_metabolite_without_name_is_skipped_with_a_warning(monkeypatch, tmp_path, caplog):
    cache_path = tmp_path / "smilesDB.tsv"
    cache_path.write_text("m1_c\tFROM_ID\nglucose\tCCO\n")
    model = _build_model([("m1_c", ""), ("glc_c", "glucose")])
    with caplog.at_level(logging.WARNING):
        find_met_smiles(model, cache_path=cache_path)
    assert _smiles(model, "m1_c") == ""
    assert _smiles(model, "glc_c") == "CCO"
    assert any("m1_c" in r.getMessage() and "no name" in r.getMessage() for r in caplog.records)
