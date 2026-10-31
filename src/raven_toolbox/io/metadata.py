"""Model version, date and descriptive metadata.

RAVEN keeps ``version`` and ``date`` on the model itself (``model.version``,
``model.date``). A cobra model has no such attributes, so raven-toolbox keeps them in
``model.notes``: ``notes["version"]`` and ``notes["metaData"]`` (which holds ``date`` and
the descriptive fields). :func:`write_yaml_model` reads the version from the first and the
date from the second, so both entries change together. :func:`get_model_metadata` and
:func:`set_model_metadata` read and write them.
"""
from __future__ import annotations

import datetime

import cobra

from raven_toolbox.io.yaml import _META_ANNOTATION_FIELDS

__all__ = ["get_model_metadata", "set_model_metadata"]


def get_model_metadata(model: cobra.Model) -> dict:
    """All metadata of the model as a plain dict (a copy).

    The dict holds ``version``, ``date`` and the descriptive fields, and any other entry that
    :func:`read_yaml_model` put in ``metaData`` (for example ``id``, ``name`` and the
    default bounds). ``version`` is taken from ``notes["version"]``, falling back to
    ``notes["metaData"]``, the same order :func:`write_yaml_model` uses. Fields the model
    does not carry are absent.
    """
    notes = model.notes or {}
    meta = dict(notes.get("metaData") or {})
    version = notes.get("version", meta.get("version"))
    if version is not None:
        meta["version"] = version
    return meta


def set_model_metadata(
    model: cobra.Model,
    *,
    version: str | None = None,
    date: str | datetime.date | None = None,
    **fields: str,
) -> None:
    """Set the version, date and descriptive fields that the YAML export writes, in place.

    Parameters
    ----------
    version
        Model version, for example ``"1.2.0"``.
    date
        Model date, as ``YYYY-MM-DD`` or a ``datetime.date`` (a ``datetime`` is reduced to its
        date). Any other value raises ``ValueError``. The YAML writer stamps today's date only
        when the model has none.
    **fields
        Descriptive ``metaData`` fields: ``givenName``, ``familyName``, ``authors``,
        ``email``, ``organization``, ``taxonomy``, ``note`` and ``sourceUrl``.

    Arguments left out are not touched.
    """
    unknown = sorted(set(fields) - set(_META_ANNOTATION_FIELDS))
    if unknown:
        raise ValueError(f"unknown metadata field(s) {unknown}; allowed: {list(_META_ANNOTATION_FIELDS)}")
    notes = dict(model.notes or {})
    meta = dict(notes.get("metaData") or {})
    if version is not None:
        notes["version"] = meta["version"] = str(version)
    if date is not None:
        meta["date"] = _iso_date(date)
    meta.update(fields)
    notes["metaData"] = meta
    model.notes = notes


def _iso_date(value: str | datetime.date) -> str:
    """``YYYY-MM-DD`` for a date, a datetime or an ISO date string; ValueError otherwise."""
    if isinstance(value, datetime.datetime):
        return value.date().isoformat()
    if isinstance(value, datetime.date):
        return value.isoformat()
    try:
        if not isinstance(value, str):
            raise ValueError
        return datetime.date.fromisoformat(value).isoformat()
    except ValueError:
        raise ValueError(f"date must be YYYY-MM-DD or a datetime.date, got {value!r}") from None
