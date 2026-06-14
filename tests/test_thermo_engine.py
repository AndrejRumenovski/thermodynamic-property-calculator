"""Tests for the calculation engine.

Assertions are anchored to physically meaningful values: at 1 atm (760 mmHg)
each species should boil at its known normal boiling point, and water's vapor
pressure at 100 °C should be ~760 mmHg.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from thermo import thermo_engine as engine
from thermo.data_models import load_species

DATA_PATH = Path(__file__).resolve().parent.parent / "chemical_data.json"

# Literature normal boiling points (°C) at 1 atm.
KNOWN_BOILING_POINTS_C = {
    "water": 100.0,
    "benzene": 80.1,
    "toluene": 110.6,
    "ethanol": 78.37,
    "methanol": 64.7,
    "acetone": 56.05,
}


@pytest.fixture(scope="module")
def registry():
    return load_species(DATA_PATH)


def test_water_vapor_pressure_at_100C(registry):
    p = engine.vapor_pressure(registry["water"].antoine, 100.0)
    assert p == pytest.approx(760.0, abs=1.0)
    assert isinstance(p, float)


def test_benzene_boiling_temperature_at_1atm(registry):
    t = engine.boiling_temperature(registry["benzene"].antoine, 760.0, pressure_unit="mmHg")
    assert t == pytest.approx(80.1, abs=0.5)


@pytest.mark.parametrize("key,expected_c", KNOWN_BOILING_POINTS_C.items())
def test_normal_boiling_points(registry, key, expected_c):
    t = engine.normal_boiling_point(registry[key].antoine)
    assert t == pytest.approx(expected_c, abs=1.0)


def test_round_trip_T_to_P_to_T(registry):
    antoine = registry["water"].antoine
    for t in (25.0, 50.0, 75.0, 99.0):
        p = engine.vapor_pressure(antoine, t)
        t_back = engine.boiling_temperature(antoine, p)
        assert t_back == pytest.approx(t, abs=1e-4)


def test_vapor_pressure_curve_is_array_and_monotonic(registry):
    antoine = registry["water"].antoine
    temps = np.linspace(10.0, 100.0, 50)
    pressures = engine.vapor_pressure_curve(antoine, temps)
    assert isinstance(pressures, np.ndarray)
    assert pressures.shape == temps.shape
    assert np.all(np.diff(pressures) > 0)  # vapor pressure rises with temperature


def test_pressure_unit_conversions():
    assert engine.convert_pressure(760.0, "mmHg", "Pa") == pytest.approx(101325.0)
    assert engine.convert_pressure(760.0, "mmHg", "atm") == pytest.approx(1.0)
    assert engine.convert_pressure(1.0, "atm", "kPa") == pytest.approx(101.325)
    with pytest.raises(ValueError, match="pressure unit"):
        engine.convert_pressure(1.0, "psi", "Pa")


def test_temperature_unit_conversions():
    assert engine.convert_temperature(0.0, "Celsius", "Kelvin") == pytest.approx(273.15)
    assert engine.convert_temperature(0.0, "Celsius", "Fahrenheit") == pytest.approx(32.0)
    assert engine.convert_temperature(100.0, "Celsius", "Fahrenheit") == pytest.approx(212.0)
    assert engine.convert_temperature(273.15, "Kelvin", "Celsius") == pytest.approx(0.0)


def test_vapor_pressure_with_explicit_units(registry):
    # 100 °C == 373.15 K; water vapor pressure there is ~1 atm.
    antoine = registry["water"].antoine
    p_atm = engine.vapor_pressure(
        antoine, 373.15, temp_unit="Kelvin", pressure_unit="atm"
    )
    assert p_atm == pytest.approx(1.0, abs=0.01)


def test_singularity_guard_raises(registry):
    antoine = registry["water"].antoine  # C = 233.426, singularity at -233.426 °C
    with pytest.raises(ValueError, match="singularity"):
        engine.vapor_pressure(antoine, -300.0, check_range=False)


def test_no_real_boiling_temperature_raises(registry):
    antoine = registry["water"].antoine  # 10**A ~ 1.2e8 mmHg
    with pytest.raises(ValueError, match="No real boiling temperature"):
        engine.boiling_temperature(antoine, 1.0e9, pressure_unit="mmHg")


def test_out_of_range_emits_warning(registry):
    antoine = registry["water"].antoine  # t_max = 100 °C
    with pytest.warns(UserWarning, match="validated range"):
        engine.vapor_pressure(antoine, 150.0)


def test_temperature_in_range(registry):
    antoine = registry["water"].antoine
    assert engine.temperature_in_range(antoine, 50.0) is True
    assert engine.temperature_in_range(antoine, 150.0) is False
