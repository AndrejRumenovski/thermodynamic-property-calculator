"""NRTL activity-coefficient model (Renon & Prausnitz, AIChE J. 1968, 14, 135).

Multicomponent form::

    τ_ij = a_ij / (R T)              (a_ij in cal/mol, a_ii = 0)
    G_ij = exp(-α_ij τ_ij)           (α symmetric, G_ii = 1)

    ln γ_i = Σ_j τ_ji G_ji x_j / Σ_k G_ki x_k
             + Σ_j [ x_j G_ij / Σ_k G_kj x_k ]
                   ( τ_ij − Σ_m x_m τ_mj G_mj / Σ_k G_kj x_k )
"""

from __future__ import annotations

import numpy as np

from .base import ActivityModel, R_CAL


class NRTLModel(ActivityModel):
    name = "NRTL"

    def __init__(self, a: np.ndarray, alpha: np.ndarray) -> None:
        """``a`` is the N×N interaction matrix (cal/mol, zero diagonal); ``alpha``
        the N×N symmetric non-randomness matrix (zero diagonal)."""
        self.a = np.asarray(a, dtype=float)
        self.alpha = np.asarray(alpha, dtype=float)

    def gamma(self, x: np.ndarray, temperature_k: float) -> np.ndarray:
        x = self._normalize(x)
        tau = self.a / (R_CAL * temperature_k)
        np.fill_diagonal(tau, 0.0)
        G = np.exp(-self.alpha * tau)
        np.fill_diagonal(G, 1.0)

        S = G.T @ x                       # S_j = Σ_k G_kj x_k
        ratio = (tau * G).T @ x / S       # = Σ_m x_m τ_mj G_mj / S_j  (indexed by j)
        term1 = ratio                     # term1_i = Σ_j τ_ji G_ji x_j / S_i
        A = G * (x / S)[None, :] * (tau - ratio[None, :])
        term2 = A.sum(axis=1)
        return np.exp(term1 + term2)

    def describe(self) -> dict:
        return {
            "model": self.name,
            "a12 (cal/mol)": round(float(self.a[0, 1]), 2),
            "a21 (cal/mol)": round(float(self.a[1, 0]), 2),
            "alpha": round(float(self.alpha[0, 1]), 3),
        }
