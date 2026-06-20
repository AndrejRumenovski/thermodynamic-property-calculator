"""γ–φ vapor-liquid equilibrium solver built on an :class:`ActivityModel`.

Uses the **modified Raoult's law** closure (ideal vapor, real liquid):

    y_i P = x_i γ_i(x, T) P_i^sat(T)        ⇒    K_i = γ_i P_i^sat / P

Saturation pressures come from :mod:`thermo.thermo_engine` (Antoine), and the
two-phase vapor fraction from :func:`thermo.flash_engine.solve_vapor_fraction`
(Rachford-Rice) — so no property or flash logic is duplicated here. For
composition-dependent γ (Wilson/NRTL/UNIQUAC) bubble-T, dew and flash are solved
by successive substitution; for the ideal model they reduce exactly to the
existing Raoult flash.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .. import flash_engine as fe
from .. import thermo_engine as engine
from ..data_models import ChemicalSpecies
from .base import ActivityModel

_MAX_ITER = 200
_TOL = 1e-10


class ThermodynamicModel:
    """A binary/multicomponent VLE model: species + an activity model."""

    def __init__(self, species: list[ChemicalSpecies], activity: ActivityModel,
                 provenance: str = "") -> None:
        if len(species) < 2:
            raise ValueError("A thermodynamic model needs at least two components.")
        self.species = species
        self.activity = activity
        self.provenance = provenance

    # -- properties -------------------------------------------------------- #
    def psat(self, t_k: float, pressure_unit: str = "mmHg") -> np.ndarray:
        return np.array(
            [
                engine.vapor_pressure(sp.antoine, t_k, temp_unit="Kelvin",
                                      pressure_unit=pressure_unit, check_range=False)
                for sp in self.species
            ],
            dtype=float,
        )

    def gamma(self, x: np.ndarray, t_k: float) -> np.ndarray:
        return self.activity.gamma(np.asarray(x, dtype=float), t_k)

    # -- bubble point ------------------------------------------------------ #
    def bubble_pressure_k(self, x: np.ndarray, t_k: float, pressure_unit="mmHg"):
        psat = self.psat(t_k, pressure_unit)
        g = self.gamma(x, t_k)
        partial = x * g * psat
        P = float(partial.sum())
        return P, partial / P

    def bubble_temperature_k(self, x: np.ndarray, pressure: float, pressure_unit="mmHg"):
        x = np.asarray(x, dtype=float)
        tb = [engine.boiling_temperature(sp.antoine, pressure,
                                         pressure_unit=pressure_unit, temp_unit="Kelvin")
              for sp in self.species]
        lo, hi = min(tb) - 60.0, max(tb) + 60.0

        def f(t_k):
            return self.bubble_pressure_k(x, t_k, pressure_unit)[0] - pressure

        lo = max(lo, 1.0)
        for _ in range(60):                      # widen until the root is bracketed
            if f(lo) < 0.0 < f(hi):
                break
            lo = max(lo - 40.0, 1.0)
            hi += 40.0
        t_k = float(brentq(f, lo, hi, xtol=1e-8))
        _, y = self.bubble_pressure_k(x, t_k, pressure_unit)
        return t_k, y

    # -- dew point --------------------------------------------------------- #
    def dew_pressure_k(self, y: np.ndarray, t_k: float, pressure_unit="mmHg"):
        y = np.asarray(y, dtype=float)
        psat = self.psat(t_k, pressure_unit)
        x = y.copy()                              # initial guess: x = y
        for _ in range(_MAX_ITER):
            g = self.gamma(x, t_k)
            weights = y / (g * psat)
            x_new = weights / weights.sum()
            if np.max(np.abs(x_new - x)) < _TOL:
                x = x_new
                break
            x = x_new
        g = self.gamma(x, t_k)
        P = float(1.0 / np.sum(y / (g * psat)))
        return P, x

    # -- isothermal flash -------------------------------------------------- #
    def flash(self, z, temperature: float, pressure: float,
              temp_unit: str = "Celsius", pressure_unit: str = "mmHg") -> fe.FlashResult:
        z = fe.normalize_composition(z)
        if len(self.species) != z.size:
            raise fe.FlashError("Components and mole fractions must match.")
        if pressure <= 0:
            raise fe.FlashError("System pressure must be strictly positive.")

        t_k = engine.convert_temperature(temperature, temp_unit, "Kelvin")
        psat = self.psat(t_k, pressure_unit)

        p_bub, _ = self.bubble_pressure_k(z, t_k, pressure_unit)
        p_dew, _ = self.dew_pressure_k(z, t_k, pressure_unit)
        nan = np.full(z.size, np.nan)

        if pressure >= p_bub:                     # subcooled liquid
            regime, beta, x, y = fe.REGIME_SUBCOOLED, 0.0, z.copy(), nan
            gamma = self.gamma(z, t_k)
        elif pressure <= p_dew:                   # superheated vapor
            regime, beta, x, y = fe.REGIME_SUPERHEATED, 1.0, nan, z.copy()
            gamma = self.gamma(z, t_k)
        else:                                     # two-phase: successive substitution
            regime = fe.REGIME_TWO_PHASE
            x = z.copy()
            beta = 0.5
            for _ in range(_MAX_ITER):
                gamma = self.gamma(x, t_k)
                K = gamma * psat / pressure
                beta = fe.solve_vapor_fraction(z, K)
                x_new = z / (1.0 + beta * (K - 1.0))
                x_new = x_new / x_new.sum()
                if np.max(np.abs(x_new - x)) < _TOL:
                    x = x_new
                    break
                x = x_new
            gamma = self.gamma(x, t_k)
            K = gamma * psat / pressure
            y = K * x
            y = y / y.sum()

        K_report = self.gamma(z if regime != fe.REGIME_TWO_PHASE else x, t_k) * psat / pressure
        return fe.FlashResult(
            regime=regime, vapor_fraction=float(beta), z=z, K=K_report, psat=psat,
            x=x, y=y, bubble_pressure=p_bub, dew_pressure=p_dew,
            temperature=temperature, temp_unit=temp_unit,
            pressure=pressure, pressure_unit=pressure_unit,
            gamma=gamma, model_name=self.activity.name,
        )

    # -- unit-friendly wrappers used by the diagram generator -------------- #
    def bubble_temperature(self, x, pressure, pressure_unit="mmHg", temp_unit="Celsius"):
        t_k, y = self.bubble_temperature_k(x, pressure, pressure_unit)
        return engine.convert_temperature(t_k, "Kelvin", temp_unit), y

    def bubble_pressure(self, x, temperature, temp_unit="Celsius", pressure_unit="mmHg"):
        t_k = engine.convert_temperature(temperature, temp_unit, "Kelvin")
        return self.bubble_pressure_k(x, t_k, pressure_unit)
