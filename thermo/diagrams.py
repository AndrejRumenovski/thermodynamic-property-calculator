"""Binary phase-diagram generation (T–x–y and P–x–y).

Sweeps the liquid mole fraction of component 1 across [0, 1] and evaluates the
bubble point at each step with a :class:`~thermo.models.vle.ThermodynamicModel`.
The bubble calculation yields both curves at once: plotting the temperature (or
pressure) against ``x₁`` traces the **bubble (liquid) line**, and against the
equilibrium ``y₁`` traces the **dew (vapor) line**. The region between them is
the two-phase envelope.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .models.vle import ThermodynamicModel


def txy(
    model: ThermodynamicModel,
    pressure: float,
    pressure_unit: str = "mmHg",
    temp_unit: str = "Celsius",
    n: int = 101,
) -> pd.DataFrame:
    """Isobaric T–x–y data: bubble & dew temperature vs. composition of comp 1."""
    x1 = np.linspace(0.0, 1.0, n)
    temps, y1 = np.empty(n), np.empty(n)
    for i, xi in enumerate(x1):
        t, y = model.bubble_temperature([xi, 1.0 - xi], pressure, pressure_unit, temp_unit)
        temps[i], y1[i] = t, y[0]
    df = pd.DataFrame({"x1": x1, "y1": y1, "T": temps})
    df.attrs.update(kind="Txy", pressure=pressure, pressure_unit=pressure_unit,
                    temp_unit=temp_unit)
    return df


def pxy(
    model: ThermodynamicModel,
    temperature: float,
    temp_unit: str = "Celsius",
    pressure_unit: str = "mmHg",
    n: int = 101,
) -> pd.DataFrame:
    """Isothermal P–x–y data: bubble & dew pressure vs. composition of comp 1."""
    x1 = np.linspace(0.0, 1.0, n)
    press, y1 = np.empty(n), np.empty(n)
    for i, xi in enumerate(x1):
        p, y = model.bubble_pressure([xi, 1.0 - xi], temperature, temp_unit, pressure_unit)
        press[i], y1[i] = p, y[0]
    df = pd.DataFrame({"x1": x1, "y1": y1, "P": press})
    df.attrs.update(kind="Pxy", temperature=temperature, temp_unit=temp_unit,
                    pressure_unit=pressure_unit)
    return df


def equilibrium_xy(df: pd.DataFrame) -> pd.DataFrame:
    """Return the x–y equilibrium curve (component 1) from a T–x–y/P–x–y frame."""
    return df[["x1", "y1"]].copy()
