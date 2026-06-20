"""Wilson activity-coefficient model (Wilson, J. Am. Chem. Soc. 1964, 86, 127).

    Λ_ij = (V_j / V_i) exp(-a_ij / (R T))     (a_ij in cal/mol, a_ii = 0, Λ_ii = 1)

    ln γ_i = 1 − ln(Σ_j x_j Λ_ij) − Σ_k x_k Λ_ki / (Σ_j x_j Λ_kj)

V_i are pure-liquid molar volumes (cm³/mol). Wilson cannot predict liquid-liquid
splitting, but is excellent for the miscible VLE systems used here.
"""

from __future__ import annotations

import numpy as np

from .base import ActivityModel, R_CAL


class WilsonModel(ActivityModel):
    name = "Wilson"

    def __init__(self, a: np.ndarray, volumes: np.ndarray) -> None:
        """``a`` is the N×N interaction matrix (cal/mol, zero diagonal);
        ``volumes`` the pure-liquid molar volumes (cm³/mol)."""
        self.a = np.asarray(a, dtype=float)
        self.v = np.asarray(volumes, dtype=float)

    def _lambda(self, temperature_k: float) -> np.ndarray:
        vol_ratio = self.v[None, :] / self.v[:, None]   # (V_j / V_i)
        lam = vol_ratio * np.exp(-self.a / (R_CAL * temperature_k))
        np.fill_diagonal(lam, 1.0)
        return lam

    def gamma(self, x: np.ndarray, temperature_k: float) -> np.ndarray:
        x = self._normalize(x)
        lam = self._lambda(temperature_k)
        sj = lam @ x                       # Σ_j x_j Λ_ij
        term = lam.T @ (x / sj)            # Σ_k x_k Λ_ki / (Σ_j x_j Λ_kj)
        return np.exp(1.0 - np.log(sj) - term)

    def describe(self) -> dict:
        return {
            "model": self.name,
            "a12 (cal/mol)": round(float(self.a[0, 1]), 2),
            "a21 (cal/mol)": round(float(self.a[1, 0]), 2),
        }
