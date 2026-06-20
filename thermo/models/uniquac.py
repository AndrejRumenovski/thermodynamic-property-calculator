"""UNIQUAC activity-coefficient model (Abrams & Prausnitz, AIChE J. 1975, 21, 116).

Combinatorial (size/shape) + residual (energetic) contributions::

    φ_i = r_i x_i / Σ_j r_j x_j      θ_i = q_i x_i / Σ_j q_j x_j
    l_i = (z/2)(r_i − q_i) − (r_i − 1)                z = 10

    ln γ_i^C = ln(φ_i/x_i) + (z/2) q_i ln(θ_i/φ_i) + l_i − (φ_i/x_i) Σ_j x_j l_j
    τ_ij     = exp(-a_ij / (R T))                     (a_ij in cal/mol, τ_ii = 1)
    ln γ_i^R = q_i [ 1 − ln(Σ_j θ_j τ_ji) − Σ_j θ_j τ_ij / (Σ_k θ_k τ_kj) ]

r_i, q_i are van-der-Waals volume/area structural parameters.
"""

from __future__ import annotations

import numpy as np

from .base import ActivityModel, R_CAL

_Z = 10.0  # lattice coordination number


class UNIQUACModel(ActivityModel):
    name = "UNIQUAC"

    def __init__(self, a: np.ndarray, r: np.ndarray, q: np.ndarray) -> None:
        """``a`` is the N×N interaction matrix (cal/mol, zero diagonal);
        ``r``/``q`` the structural volume/area parameters."""
        self.a = np.asarray(a, dtype=float)
        self.r = np.asarray(r, dtype=float)
        self.q = np.asarray(q, dtype=float)

    def gamma(self, x: np.ndarray, temperature_k: float) -> np.ndarray:
        x = self._normalize(x)
        r, q = self.r, self.q

        phi = r * x / np.sum(r * x)
        theta = q * x / np.sum(q * x)
        l = (_Z / 2.0) * (r - q) - (r - 1.0)
        ln_c = (
            np.log(phi / x)
            + (_Z / 2.0) * q * np.log(theta / phi)
            + l
            - (phi / x) * np.sum(x * l)
        )

        tau = np.exp(-self.a / (R_CAL * temperature_k))
        np.fill_diagonal(tau, 1.0)
        S = tau.T @ theta                  # S_m = Σ_k θ_k τ_km
        ln_r = q * (1.0 - np.log(S) - tau @ (theta / S))
        return np.exp(ln_c + ln_r)

    def describe(self) -> dict:
        return {
            "model": self.name,
            "a12 (cal/mol)": round(float(self.a[0, 1]), 2),
            "a21 (cal/mol)": round(float(self.a[1, 0]), 2),
        }
