"""Curated structural data and binary parameters for the activity models.

Binary interaction parameters are **derived by fitting each model to the
experimental azeotrope** of the system. At an azeotrope y_i = x_i, so modified
Raoult's law gives the exact identity γ_i = P / P_i^sat(T_az). Fitting the two
binary parameters so the model reproduces (γ_1, γ_2) at the literature azeotrope
composition/temperature therefore reproduces the real azeotrope by construction
and yields physically correct activity coefficients. Independent full-dataset
validation (MAE/RMSE vs. T–x–y data) is a later increment.

Sources
-------
* Structural r, q and molar volumes: Poling, Prausnitz & O'Connell, *The
  Properties of Gases and Liquids*, 5th ed., App. A / DECHEMA.
* Azeotrope loci (1 atm): Gmehling et al., *Azeotropic Data*; NIST WebBook.
* Models: Renon & Prausnitz (NRTL, 1968); Wilson (1964); Abrams & Prausnitz
  (UNIQUAC, 1975).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.optimize import fsolve

from .. import thermo_engine as engine
from ..data_models import ChemicalSpecies
from .base import ActivityModel
from .ideal import IdealModel
from .nrtl import NRTLModel
from .uniquac import UNIQUACModel
from .wilson import WilsonModel

# Pure-component structural parameters, keyed by species key.
#   r, q : UNIQUAC volume / surface-area parameters
#   v    : liquid molar volume (cm³/mol) for Wilson
STRUCTURAL: dict[str, dict[str, float]] = {
    "water":    {"r": 0.9200, "q": 1.4000, "v": 18.07},
    "methanol": {"r": 1.4311, "q": 1.4320, "v": 40.73},
    "ethanol":  {"r": 2.1055, "q": 1.9720, "v": 58.68},
    "acetone":  {"r": 2.5735, "q": 2.3360, "v": 74.05},
    "benzene":  {"r": 3.1878, "q": 2.4000, "v": 89.41},
    "toluene":  {"r": 3.9228, "q": 2.9680, "v": 106.85},
}

# Experimental azeotropes at 1 atm (760 mmHg). ``x_first`` is the mole fraction
# of ``first`` at the azeotrope.
AZEOTROPES: dict[frozenset, dict] = {
    frozenset(("ethanol", "water")): {
        "first": "ethanol", "x_first": 0.894, "T_C": 78.15, "P_mmHg": 760.0,
        "source": "Ethanol–water azeotrope, 95.6 wt% EtOH, 78.15 °C (Gmehling; NIST).",
    },
    frozenset(("acetone", "methanol")): {
        "first": "acetone", "x_first": 0.800, "T_C": 55.5, "P_mmHg": 760.0,
        "source": "Acetone–methanol azeotrope, ~0.80 mol acetone, 55.5 °C (Gmehling; NIST).",
    },
}

_NRTL_ALPHA = 0.30  # Renon-Prausnitz recommended non-randomness for these systems.


class NoParametersError(LookupError):
    """Raised when a non-ideal model lacks parameters for a given pair."""


def _a_matrix(a12: float, a21: float) -> np.ndarray:
    return np.array([[0.0, a12], [a21, 0.0]], dtype=float)


def _make_builder(model_name: str, sp1: ChemicalSpecies, sp2: ChemicalSpecies):
    """Return a function (a12, a21) -> ActivityModel for the chosen model."""
    if model_name == NRTLModel.name:
        alpha = np.array([[0.0, _NRTL_ALPHA], [_NRTL_ALPHA, 0.0]])
        return lambda a12, a21: NRTLModel(_a_matrix(a12, a21), alpha)
    if model_name == WilsonModel.name:
        v = np.array([STRUCTURAL[sp1.key]["v"], STRUCTURAL[sp2.key]["v"]])
        return lambda a12, a21: WilsonModel(_a_matrix(a12, a21), v)
    if model_name == UNIQUACModel.name:
        r = np.array([STRUCTURAL[sp1.key]["r"], STRUCTURAL[sp2.key]["r"]])
        q = np.array([STRUCTURAL[sp1.key]["q"], STRUCTURAL[sp2.key]["q"]])
        return lambda a12, a21: UNIQUACModel(_a_matrix(a12, a21), r, q)
    raise NoParametersError(f"Unknown non-ideal model {model_name!r}.")


def _fit_to_azeotrope(builder, sp1, sp2, x1_az, t_c, p_mmhg) -> tuple[float, float]:
    """Solve for (a12, a21) reproducing γ_i = P/P_i^sat at the azeotrope."""
    psat1 = engine.vapor_pressure(sp1.antoine, t_c, temp_unit="Celsius",
                                  pressure_unit="mmHg", check_range=False)
    psat2 = engine.vapor_pressure(sp2.antoine, t_c, temp_unit="Celsius",
                                  pressure_unit="mmHg", check_range=False)
    targets = np.array([p_mmhg / psat1, p_mmhg / psat2])
    x = np.array([x1_az, 1.0 - x1_az])
    t_k = t_c + 273.15

    def residual(a):
        g = builder(a[0], a[1]).gamma(x, t_k)
        return g - targets

    solution, _info, ier, _msg = fsolve(residual, x0=[600.0, 600.0], full_output=True)
    if ier != 1 or not np.allclose(residual(solution), 0.0, atol=1e-6):
        raise NoParametersError(
            f"Could not fit {sp1.key}/{sp2.key} parameters to the azeotrope."
        )
    return float(solution[0]), float(solution[1])


def has_parameters(model_name: str, sp1: ChemicalSpecies, sp2: ChemicalSpecies) -> bool:
    if model_name == IdealModel.name:
        return True
    pair = frozenset((sp1.key, sp2.key))
    if pair not in AZEOTROPES:
        return False
    if model_name in (WilsonModel.name, UNIQUACModel.name):
        return sp1.key in STRUCTURAL and sp2.key in STRUCTURAL
    return model_name == NRTLModel.name


@lru_cache(maxsize=64)
def _cached_fit(model_name: str, k1: str, k2: str, sp1, sp2):
    pair = AZEOTROPES[frozenset((k1, k2))]
    x1 = pair["x_first"] if pair["first"] == k1 else 1.0 - pair["x_first"]
    builder = _make_builder(model_name, sp1, sp2)
    a12, a21 = _fit_to_azeotrope(builder, sp1, sp2, x1, pair["T_C"], pair["P_mmHg"])
    return builder(a12, a21), pair["source"]


def build_activity_model(
    model_name: str, sp1: ChemicalSpecies, sp2: ChemicalSpecies
) -> tuple[ActivityModel, str]:
    """Return ``(model, provenance)`` for the chosen model and binary pair.

    Raises :class:`NoParametersError` for non-ideal models without parameters.
    """
    if model_name == IdealModel.name:
        return IdealModel(2), "Raoult's law (ideal liquid; γ = 1)."
    if not has_parameters(model_name, sp1, sp2):
        raise NoParametersError(
            f"No {model_name} parameters for {sp1.name}/{sp2.name}."
        )
    return _cached_fit(model_name, sp1.key, sp2.key, sp1, sp2)
