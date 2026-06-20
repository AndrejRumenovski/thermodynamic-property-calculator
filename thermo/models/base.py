"""Activity-coefficient model interface.

An :class:`ActivityModel` computes liquid-phase activity coefficients γ_i(x, T)
for a fixed set of components. These are pure thermodynamic models — no I/O, no
property data — consumed by :mod:`thermo.models.vle`, which combines them with
Antoine saturation pressures to solve bubble/dew/flash problems.

All temperatures are in **Kelvin** and interaction energies in **cal/mol**;
the gas constant below is in the matching units.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

# Gas constant in cal/(mol·K) — matches interaction parameters expressed in cal/mol.
R_CAL = 1.98720425864


class ActivityModel(ABC):
    """Base class for liquid-phase activity-coefficient models."""

    #: Human-readable model name (e.g. "NRTL").
    name: str = "Activity model"

    @abstractmethod
    def gamma(self, x: np.ndarray, temperature_k: float) -> np.ndarray:
        """Return activity coefficients γ_i at liquid composition ``x`` and ``T``.

        ``x`` is a mole-fraction vector (need not be normalised) and
        ``temperature_k`` is in Kelvin. The result has the same length as ``x``.
        """

    def describe(self) -> dict:
        """Return a small dict of parameters for display/provenance."""
        return {"model": self.name}

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        total = x.sum()
        if total <= 0:
            raise ValueError("Composition must sum to a positive value.")
        # Guard against exact zeros, which make ln(x) singular in some models.
        return np.clip(x / total, 1e-12, 1.0)
