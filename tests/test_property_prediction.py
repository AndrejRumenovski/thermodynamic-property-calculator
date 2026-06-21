"""Tests for Joback group-contribution property prediction."""

from __future__ import annotations

import math

import numpy as np
import pytest

from thermo import property_prediction as pp


# --------------------------------------------------------------------------- #
# Formula parsing / molar mass
# --------------------------------------------------------------------------- #
def test_parse_formula_and_mass():
    assert pp.parse_formula("C2H6O") == {"C": 2, "H": 6, "O": 1}
    assert pp.molar_mass("C2H6O") == pytest.approx(46.069, abs=0.01)
    assert pp.atom_count("C6H6") == 12


def test_unknown_element_raises():
    with pytest.raises(ValueError):
        pp.molar_mass("C2X3")


# --------------------------------------------------------------------------- #
# Joback equations vs. known values
# --------------------------------------------------------------------------- #
def test_benzene_estimate_within_joback_accuracy():
    est = pp.joback_estimate("Benzene", "C6H6", {"ring=CH-": 6})
    assert est.Tb == pytest.approx(353.2, abs=8)     # Joback ~ 2% on Tb
    assert est.Tc == pytest.approx(562.2, abs=12)
    assert est.Pc == pytest.approx(48.95, abs=3)
    assert 0.18 < est.omega < 0.25                   # benzene ω ≈ 0.21


def test_acetic_acid_estimate():
    est = pp.joback_estimate("Acetic acid", "C2H4O2",
                             {"-CH3": 1, "-COOH (acid)": 1})
    assert est.Tb == pytest.approx(391.1, abs=5)
    assert est.Tc == pytest.approx(592.0, abs=10)


def test_unknown_group_raises():
    with pytest.raises(pp.PropertyPredictionError):
        pp.joback_estimate("X", "C2H6", {"-CF3": 1})


# --------------------------------------------------------------------------- #
# Vapor pressure (Lee–Kesler corresponding states)
# --------------------------------------------------------------------------- #
def test_vapor_pressure_is_one_atm_at_predicted_tb():
    est = pp.joback_estimate("Benzene", "C6H6", {"ring=CH-": 6})
    p_atm = est.vapor_pressure(est.Tb, "atm")
    assert p_atm == pytest.approx(1.0, abs=1e-6)     # by construction


def test_vapor_pressure_increases_with_temperature():
    est = pp.estimate_molecule(pp.LIBRARY_BY_NAME["n-Hexane"])
    p_low = est.vapor_pressure(300.0, "mmHg")
    p_high = est.vapor_pressure(330.0, "mmHg")
    assert p_high > p_low > 0


# --------------------------------------------------------------------------- #
# Benchmarking across the library
# --------------------------------------------------------------------------- #
def test_benchmark_accuracy_meets_joback_expectations():
    rows, metrics = pp.benchmark()
    assert len(rows) == len(pp.LIBRARY)
    # Documented Joback accuracy: ~ few-% on Tb/Tc, ~5% on Pc; strong correlation.
    assert metrics["Tb"]["MPE"] < 6.0 and metrics["Tb"]["R2"] > 0.75
    assert metrics["Tc"]["MPE"] < 5.0 and metrics["Tc"]["R2"] > 0.75
    assert metrics["Pc"]["MPE"] < 8.0 and metrics["Pc"]["R2"] > 0.80


def test_regression_metrics_values():
    m = pp.regression_metrics([100.0, 200.0, 300.0], [100.0, 200.0, 300.0])
    assert m["MAE"] == 0.0 and m["RMSE"] == 0.0 and m["R2"] == pytest.approx(1.0)


def test_every_library_molecule_mass_matches_formula():
    for mol in pp.LIBRARY:
        est = pp.estimate_molecule(mol)
        assert est.molar_mass == pytest.approx(pp.molar_mass(mol.formula), abs=1e-6)
        assert math.isfinite(est.Tc) and est.Tc > est.Tb   # Tc above boiling point
