"""Modular thermodynamic-model framework.

Public surface:

* activity models — :class:`IdealModel`, :class:`WilsonModel`, :class:`NRTLModel`,
  :class:`UNIQUACModel` (all subclasses of :class:`ActivityModel`);
* :class:`ThermodynamicModel` — the bubble/dew/flash solver built on any of them;
* :func:`build_model` — construct a ready-to-use model for a binary pair by name;
* :data:`MODEL_NAMES` and :func:`has_parameters` for UI model selection.

The architecture is open to future equation-of-state models: add a class with a
``gamma``-equivalent fugacity routine and register it; the VLE solver and UI
consume the abstract interface, not the concrete model.
"""

from __future__ import annotations

from ..data_models import ChemicalSpecies
from .base import ActivityModel, R_CAL
from .ideal import IdealModel
from .nrtl import NRTLModel
from .parameters import (
    AZEOTROPES,
    STRUCTURAL,
    NoParametersError,
    build_activity_model,
    has_parameters,
)
from .uniquac import UNIQUACModel
from .vle import ThermodynamicModel
from .wilson import WilsonModel

#: Selectable model names, ideal first.
MODEL_NAMES = [IdealModel.name, WilsonModel.name, NRTLModel.name, UNIQUACModel.name]


def build_model(
    model_name: str, sp1: ChemicalSpecies, sp2: ChemicalSpecies
) -> ThermodynamicModel:
    """Build a :class:`ThermodynamicModel` for ``sp1``/``sp2`` using ``model_name``.

    Raises :class:`NoParametersError` if a non-ideal model has no parameters for
    the pair.
    """
    activity, provenance = build_activity_model(model_name, sp1, sp2)
    return ThermodynamicModel([sp1, sp2], activity, provenance)


__all__ = [
    "ActivityModel",
    "IdealModel",
    "WilsonModel",
    "NRTLModel",
    "UNIQUACModel",
    "ThermodynamicModel",
    "build_model",
    "build_activity_model",
    "has_parameters",
    "NoParametersError",
    "MODEL_NAMES",
    "STRUCTURAL",
    "AZEOTROPES",
    "R_CAL",
]
