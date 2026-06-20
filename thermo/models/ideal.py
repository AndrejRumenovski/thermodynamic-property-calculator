"""Ideal-solution model (Raoult's law): γ_i = 1 for all components."""

from __future__ import annotations

import numpy as np

from .base import ActivityModel


class IdealModel(ActivityModel):
    """Raoult's law — unit activity coefficients (ideal liquid mixture)."""

    name = "Ideal (Raoult)"

    def __init__(self, n_components: int) -> None:
        self.n = int(n_components)

    def gamma(self, x: np.ndarray, temperature_k: float) -> np.ndarray:
        return np.ones(self.n, dtype=float)
