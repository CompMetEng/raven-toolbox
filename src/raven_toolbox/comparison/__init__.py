"""Model comparison utilities.

Two functions:

* :func:`compare_models`: presence-matrix overview of N models, including
  identifier, EC-code, metabolite-name and equation overlap (the counterpart of
  RAVEN's ``compareMultipleModels``). It shows how the models relate.
* :func:`diff_models`: semantic-equality diff of two models for CI
  gates. It shows whether the two models are the same.
"""
from raven_toolbox.comparison.compare import ModelComparison, compare_models
from raven_toolbox.comparison.diff import (
    DEFAULT_ANNOTATION_KEYS,
    DiffReport,
    diff_models,
)

__all__ = [
    "DEFAULT_ANNOTATION_KEYS",
    "DiffReport",
    "ModelComparison",
    "compare_models",
    "diff_models",
]
