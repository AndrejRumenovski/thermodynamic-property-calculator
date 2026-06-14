"""Layer 2 — the thermodynamic calculation engine.

Pure functions implementing the Antoine equation and the unit handling around
it. Nothing here performs I/O or touches the UI, so the whole layer is trivially
unit-testable.

Antoine equation (constants are unit-specific, typically mmHg / Celsius)::

    log10(P) = A - B / (T + C)          forward   (T -> P)
    T        = B / (A - log10(P)) - C   inverse   (P -> T, boiling temperature)

The engine always evaluates the equation in the species' *native* units (those
the constants were fitted in) and converts user-facing inputs/outputs to and
from the requested display units.
"""

from __future__ import annotations

import math
import warnings
from typing import Optional, Union

import numpy as np
from scipy.optimize import brentq

from .data_models import (
    SUPPORTED_PRESSURE_UNITS,
    SUPPORTED_TEMPERATURE_UNITS,
    AntoineConstants,
)

Number = Union[float, int]
ArrayLike = Union[Number, np.ndarray]

# One atmosphere expressed in each supported pressure unit is derived from these
# Pa-per-unit factors. mmHg is defined as 101325/760 so that exactly 760 mmHg
# equals 101325 Pa equals 1 atm.
_PA_PER_UNIT = {
    "Pa": 1.0,
    "kPa": 1_000.0,
    "bar": 100_000.0,
    "atm": 101_325.0,
    "mmHg": 101_325.0 / 760.0,
}


# --------------------------------------------------------------------------- #
# Unit conversion helpers
# --------------------------------------------------------------------------- #
def convert_pressure(value: ArrayLike, from_unit: str, to_unit: str) -> ArrayLike:
    """Convert a pressure (scalar or array) between supported units."""
    for unit in (from_unit, to_unit):
        if unit not in _PA_PER_UNIT:
            raise ValueError(
                f"Unsupported pressure unit {unit!r}; expected one of "
                f"{SUPPORTED_PRESSURE_UNITS}."
            )
    if from_unit == to_unit:
        return value
    return value * _PA_PER_UNIT[from_unit] / _PA_PER_UNIT[to_unit]


def convert_temperature(value: ArrayLike, from_unit: str, to_unit: str) -> ArrayLike:
    """Convert a temperature (scalar or array) between supported units."""
    for unit in (from_unit, to_unit):
        if unit not in SUPPORTED_TEMPERATURE_UNITS:
            raise ValueError(
                f"Unsupported temperature unit {unit!r}; expected one of "
                f"{SUPPORTED_TEMPERATURE_UNITS}."
            )
    if from_unit == to_unit:
        return value

    if from_unit == "Celsius":
        celsius = value
    elif from_unit == "Kelvin":
        celsius = value - 273.15
    else:  # Fahrenheit
        celsius = (value - 32.0) * 5.0 / 9.0

    if to_unit == "Celsius":
        return celsius
    if to_unit == "Kelvin":
        return celsius + 273.15
    return celsius * 9.0 / 5.0 + 32.0  # Fahrenheit


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #
def temperature_in_range(antoine: AntoineConstants, t_native: ArrayLike) -> bool:
    """Return ``True`` if every temperature lies within the validated range.

    ``t_native`` must already be expressed in the constants' temperature unit.
    Always ``True`` when the species declares no ``t_min``/``t_max``.
    """
    if antoine.t_min is None or antoine.t_max is None:
        return True
    arr = np.asarray(t_native, dtype=float)
    return bool(np.all((arr >= antoine.t_min) & (arr <= antoine.t_max)))


def _warn_if_out_of_range(antoine: AntoineConstants, t_native: ArrayLike) -> None:
    if not temperature_in_range(antoine, t_native):
        warnings.warn(
            f"Temperature outside the correlation's validated range "
            f"[{antoine.t_min}, {antoine.t_max}] {antoine.temperature_unit}; "
            f"results are extrapolated.",
            stacklevel=3,
        )


def _scalarize(value: np.ndarray) -> ArrayLike:
    """Return a Python float for 0-d results, otherwise the array unchanged."""
    arr = np.asarray(value)
    return float(arr) if arr.ndim == 0 else arr


# --------------------------------------------------------------------------- #
# Core Antoine evaluation (native units)
# --------------------------------------------------------------------------- #
def _native_pressure(antoine: AntoineConstants, t_native: ArrayLike) -> np.ndarray:
    """Evaluate the Antoine equation in native units, guarding the singularity."""
    denom = np.asarray(t_native, dtype=float) + antoine.C
    if np.any(denom <= 0):
        raise ValueError(
            f"Temperature at or below the Antoine singularity (T + C <= 0) for "
            f"{antoine.temperature_unit}; cannot evaluate vapor pressure."
        )
    return np.power(10.0, antoine.A - antoine.B / denom)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def vapor_pressure(
    antoine: AntoineConstants,
    temperature: ArrayLike,
    temp_unit: Optional[str] = None,
    pressure_unit: Optional[str] = None,
    check_range: bool = True,
) -> ArrayLike:
    """Vapor pressure at ``temperature`` (forward Antoine, ``T -> P``).

    ``temp_unit`` is the unit of the supplied temperature and ``pressure_unit``
    the unit of the returned pressure; both default to the species' native
    units. Accepts scalars or NumPy arrays.
    """
    temp_unit = temp_unit or antoine.temperature_unit
    pressure_unit = pressure_unit or antoine.pressure_unit

    t_native = convert_temperature(temperature, temp_unit, antoine.temperature_unit)
    if check_range:
        _warn_if_out_of_range(antoine, t_native)

    p_native = _native_pressure(antoine, t_native)
    p_out = convert_pressure(p_native, antoine.pressure_unit, pressure_unit)
    return _scalarize(p_out)


def vapor_pressure_curve(
    antoine: AntoineConstants,
    temperatures: np.ndarray,
    temp_unit: Optional[str] = None,
    pressure_unit: Optional[str] = None,
) -> np.ndarray:
    """Vectorised vapor pressure over an array of temperatures (for plotting)."""
    temp_unit = temp_unit or antoine.temperature_unit
    pressure_unit = pressure_unit or antoine.pressure_unit

    t = np.asarray(temperatures, dtype=float)
    t_native = convert_temperature(t, temp_unit, antoine.temperature_unit)
    p_native = _native_pressure(antoine, t_native)
    return convert_pressure(p_native, antoine.pressure_unit, pressure_unit)


def _solve_temperature(
    antoine: AntoineConstants, p_target_native: float, t_guess: float
) -> float:
    """Numerically invert Antoine for ``T`` with SciPy's Brent solver.

    The forward relation is strictly increasing in ``T`` above the singularity
    at ``T = -C``, so a bracket of ``(-C, large)`` always contains the root.
    ``t_guess`` (the analytic inverse) seeds the upper bound.
    """

    def residual(t: float) -> float:
        return float(10.0 ** (antoine.A - antoine.B / (t + antoine.C))) - p_target_native

    lo = -antoine.C + 1e-6  # just above the singularity: residual(lo) ~ -p_target < 0
    hi = max(t_guess, lo) + 10.0
    for _ in range(200):
        if residual(hi) > 0:
            break
        hi += 50.0
    else:  # pragma: no cover - unreachable for physically sane inputs
        raise ValueError("Could not bracket a boiling temperature for the given pressure.")

    return float(brentq(residual, lo, hi))


def boiling_temperature(
    antoine: AntoineConstants,
    pressure: Number,
    pressure_unit: Optional[str] = None,
    temp_unit: Optional[str] = None,
) -> float:
    """Temperature at which vapor pressure equals ``pressure`` (inverse Antoine).

    ``pressure_unit`` is the unit of the supplied pressure and ``temp_unit`` the
    unit of the returned temperature; both default to the species' native units.
    """
    pressure_unit = pressure_unit or antoine.pressure_unit
    temp_unit = temp_unit or antoine.temperature_unit

    p_native = float(convert_pressure(float(pressure), pressure_unit, antoine.pressure_unit))
    if p_native <= 0:
        raise ValueError("Pressure must be strictly positive.")

    denom = antoine.A - math.log10(p_native)
    if denom <= 0:
        raise ValueError(
            f"No real boiling temperature: log10(P) >= A for "
            f"P = {pressure} {pressure_unit}."
        )

    t_analytic = antoine.B / denom - antoine.C
    t_native = _solve_temperature(antoine, p_native, t_analytic)
    return float(convert_temperature(t_native, antoine.temperature_unit, temp_unit))


def normal_boiling_point(antoine: AntoineConstants, temp_unit: Optional[str] = None) -> float:
    """Boiling temperature at 1 atm, in ``temp_unit`` (native unit by default)."""
    return boiling_temperature(antoine, 1.0, pressure_unit="atm", temp_unit=temp_unit)
