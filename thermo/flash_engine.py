"""Isothermal VLE flash via the Rachford-Rice equation.

A self-contained module (per the modularity constraint) that performs an
isothermal two-phase flash for an *ideal* vapor-liquid system described by
Raoult's law:

    K_i = P_i^sat(T) / P                 (equilibrium ratio)

The saturation pressures ``P_i^sat(T)`` are **not** recomputed here — they are
obtained from the existing :mod:`thermo.thermo_engine` (which evaluates the
Antoine equation and handles unit conversion), so the flash never duplicates the
property logic and works for any species in ``chemical_data.json``, even when
components carry different native units (e.g. mmHg/°C vs Pa/K).

Given a feed composition ``z_i`` (mole fractions), temperature ``T`` and
pressure ``P``, the vapor fraction ``β = V/F`` solves the Rachford-Rice
objective::

    f(β) = Σ_i  z_i (K_i − 1) / (1 + β (K_i − 1)) = 0

``f`` is strictly decreasing on ``β ∈ [0, 1]`` and free of poles there, so the
phase regime follows from its endpoints and the two-phase root is bracketed
exactly on ``[0, 1]`` for :func:`scipy.optimize.brentq`:

* ``f(0) ≤ 0``  → **subcooled liquid**  (β = 0, all liquid; P ≥ bubble pressure)
* ``f(1) ≥ 0``  → **superheated vapor**  (β = 1, all vapor; P ≤ dew pressure)
* otherwise     → **two-phase**  (0 < β < 1)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from . import thermo_engine as engine
from .data_models import ChemicalSpecies

# Regime identifiers.
REGIME_SUBCOOLED = "subcooled_liquid"
REGIME_TWO_PHASE = "two_phase"
REGIME_SUPERHEATED = "superheated_vapor"

_REGIME_LABELS = {
    REGIME_SUBCOOLED: "Subcooled liquid (single phase)",
    REGIME_TWO_PHASE: "Two-phase (vapor + liquid)",
    REGIME_SUPERHEATED: "Superheated vapor (single phase)",
}


class FlashError(ValueError):
    """Raised for invalid flash inputs (bad mole fractions, pressure, etc.)."""


@dataclass(frozen=True)
class FlashResult:
    """Outcome of an isothermal flash.

    Compositions for an absent phase are filled with ``NaN`` (e.g. the vapor
    composition ``y`` in the subcooled-liquid regime).
    """

    regime: str
    vapor_fraction: float  # β = V/F  (0 = all liquid, 1 = all vapor)
    z: np.ndarray          # normalised feed composition
    K: np.ndarray          # equilibrium ratios P_sat/P
    psat: np.ndarray       # saturation pressures, in `pressure_unit`
    x: np.ndarray          # liquid composition (NaN if no liquid)
    y: np.ndarray          # vapor composition (NaN if no vapor)
    bubble_pressure: float  # Raoult bubble pressure at T, in `pressure_unit`
    dew_pressure: float     # Raoult dew pressure at T, in `pressure_unit`
    temperature: float
    temp_unit: str
    pressure: float
    pressure_unit: str

    @property
    def regime_label(self) -> str:
        return _REGIME_LABELS[self.regime]

    @property
    def liquid_fraction(self) -> float:
        return 1.0 - self.vapor_fraction


# --------------------------------------------------------------------------- #
# Property retrieval (delegated to thermo_engine — no duplicated logic)
# --------------------------------------------------------------------------- #
def saturation_pressures(
    species: list[ChemicalSpecies],
    temperature: float,
    temp_unit: str,
    pressure_unit: str,
) -> np.ndarray:
    """Saturation pressure of each species at ``temperature``, in ``pressure_unit``.

    Delegates to :func:`thermo.thermo_engine.vapor_pressure`, which evaluates the
    Antoine equation in each species' native units and converts the result, so
    mixed-unit feeds are handled transparently.
    """
    return np.array(
        [
            engine.vapor_pressure(
                sp.antoine,
                temperature,
                temp_unit=temp_unit,
                pressure_unit=pressure_unit,
                check_range=False,
            )
            for sp in species
        ],
        dtype=float,
    )


def k_values(
    species: list[ChemicalSpecies],
    temperature: float,
    pressure: float,
    temp_unit: str,
    pressure_unit: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(K, psat)``: equilibrium ratios and saturation pressures."""
    if pressure <= 0:
        raise FlashError("System pressure must be strictly positive.")
    psat = saturation_pressures(species, temperature, temp_unit, pressure_unit)
    return psat / pressure, psat


# --------------------------------------------------------------------------- #
# Rachford-Rice
# --------------------------------------------------------------------------- #
def rachford_rice_objective(beta: float, z: np.ndarray, K: np.ndarray) -> float:
    """The Rachford-Rice function f(β)."""
    return float(np.sum(z * (K - 1.0) / (1.0 + beta * (K - 1.0))))


def solve_vapor_fraction(z: np.ndarray, K: np.ndarray) -> float:
    """Solve Rachford-Rice for β ∈ (0, 1) on a confirmed two-phase mixture.

    Uses :func:`scipy.optimize.brentq`; the bracket ``[0, 1]`` is guaranteed to
    contain the sign change because the caller has checked ``f(0) > 0 > f(1)``.
    """
    return float(brentq(lambda b: rachford_rice_objective(b, z, K), 0.0, 1.0))


# --------------------------------------------------------------------------- #
# Input handling
# --------------------------------------------------------------------------- #
def normalize_composition(z) -> np.ndarray:
    """Validate and normalise a feed composition to sum to 1."""
    z = np.asarray(z, dtype=float)
    if z.ndim != 1 or z.size < 2:
        raise FlashError("Provide mole fractions for at least two components.")
    if np.any(z < 0) or not np.all(np.isfinite(z)):
        raise FlashError("Mole fractions must be finite and non-negative.")
    total = z.sum()
    if total <= 0:
        raise FlashError("Mole fractions must sum to a positive value.")
    return z / total


# --------------------------------------------------------------------------- #
# Flash
# --------------------------------------------------------------------------- #
def flash(
    species: list[ChemicalSpecies],
    z,
    temperature: float,
    pressure: float,
    temp_unit: str = "Celsius",
    pressure_unit: str = "mmHg",
) -> FlashResult:
    """Perform an isothermal VLE flash.

    Parameters
    ----------
    species:
        Components as :class:`~thermo.data_models.ChemicalSpecies` (as loaded
        from ``chemical_data.json``). At least two are required.
    z:
        Feed mole fractions (need not sum exactly to 1 — they are normalised).
    temperature, pressure:
        Flash conditions, expressed in ``temp_unit`` / ``pressure_unit``.
    """
    if len(species) < 2:
        raise FlashError("A flash requires at least two components.")
    z = normalize_composition(z)
    if len(species) != z.size:
        raise FlashError("The number of components and mole fractions must match.")

    K, psat = k_values(species, temperature, pressure, temp_unit, pressure_unit)

    # Raoult bubble/dew pressures at T — useful context and the regime boundary.
    bubble_pressure = float(np.sum(z * psat))
    dew_pressure = float(1.0 / np.sum(z / psat))

    f0 = float(np.sum(z * (K - 1.0)))            # f(β = 0); = Σ z_i K_i − 1
    f1 = float(np.sum(z * (K - 1.0) / K))        # f(β = 1); = 1 − Σ z_i / K_i
    nan = np.full(z.size, np.nan)

    if f0 <= 0.0:
        regime, beta = REGIME_SUBCOOLED, 0.0
        x, y = z.copy(), nan
    elif f1 >= 0.0:
        regime, beta = REGIME_SUPERHEATED, 1.0
        x, y = nan, z.copy()
    else:
        regime = REGIME_TWO_PHASE
        beta = solve_vapor_fraction(z, K)
        x = z / (1.0 + beta * (K - 1.0))
        y = K * x

    return FlashResult(
        regime=regime,
        vapor_fraction=beta,
        z=z,
        K=K,
        psat=psat,
        x=x,
        y=y,
        bubble_pressure=bubble_pressure,
        dew_pressure=dew_pressure,
        temperature=temperature,
        temp_unit=temp_unit,
        pressure=pressure,
        pressure_unit=pressure_unit,
    )
