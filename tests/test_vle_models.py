"""VLE-level tests: azeotrope reproduction, ideal regression, bubble points."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from thermo import flash_engine as fe
from thermo import models
from thermo import thermo_engine as engine
from thermo.data_models import load_species

DATA_PATH = Path(__file__).resolve().parent.parent / "chemical_data.json"


@pytest.fixture(scope="module")
def registry():
    return load_species(DATA_PATH)


def _scan_azeotrope(model, pressure=760.0, n=1001):
    """Return (x1*, T*) where the bubble curve gives y₁ = x₁ (interior)."""
    x1 = np.linspace(0.001, 0.999, n)
    best = None
    for xi in x1:
        t, y = model.bubble_temperature([xi, 1 - xi], pressure, "mmHg", "Celsius")
        d = abs(y[0] - xi)
        if best is None or d < best[0]:
            best = (d, xi, t)
    return best  # (residual, x1, T)


# --------------------------------------------------------------------------- #
# Azeotrope reproduction (the headline validation)
# --------------------------------------------------------------------------- #
AZ_CASES = [
    ("ethanol", "water", 0.894, 78.15),
    ("acetone", "methanol", 0.800, 55.5),
]


@pytest.mark.parametrize("model_name", ["Wilson", "NRTL", "UNIQUAC"])
@pytest.mark.parametrize("k1,k2,x_lit,t_lit", AZ_CASES)
def test_models_reproduce_literature_azeotrope(registry, model_name, k1, k2, x_lit, t_lit):
    model = models.build_model(model_name, registry[k1], registry[k2])
    resid, x_az, t_az = _scan_azeotrope(model)
    assert resid < 5e-3, f"no azeotrope found ({model_name} {k1}/{k2})"
    assert x_az == pytest.approx(x_lit, abs=0.02)
    assert t_az == pytest.approx(t_lit, abs=1.5)


def test_ideal_has_no_interior_azeotrope(registry):
    model = models.build_model("Ideal (Raoult)", registry["ethanol"], registry["water"])
    # bubble/dew only touch at the pure ends; no interior y=x crossing
    x1 = np.linspace(0.05, 0.95, 50)
    for xi in x1:
        _, y = model.bubble_temperature([xi, 1 - xi], 760.0, "mmHg", "Celsius")
        assert abs(y[0] - xi) > 1e-3


# --------------------------------------------------------------------------- #
# Ideal model reproduces the existing Raoult flash exactly (regression)
# --------------------------------------------------------------------------- #
def test_ideal_model_matches_flash_engine(registry):
    species = [registry["benzene"], registry["toluene"]]
    z = [0.5, 0.5]
    ref = fe.flash(species, z, 95.0, 750.0)
    model = models.build_model("Ideal (Raoult)", *species)
    got = model.flash(z, 95.0, 750.0)
    assert got.regime == ref.regime
    assert got.vapor_fraction == pytest.approx(ref.vapor_fraction, abs=1e-9)
    assert np.allclose(got.x, ref.x, atol=1e-9)
    assert np.allclose(got.y, ref.y, atol=1e-9)
    assert np.allclose(got.K, ref.K, atol=1e-9)


# --------------------------------------------------------------------------- #
# Bubble point sanity
# --------------------------------------------------------------------------- #
def test_bubble_temperature_pure_component_is_boiling_point(registry):
    model = models.build_model("NRTL", registry["ethanol"], registry["water"])
    t, y = model.bubble_temperature([1.0, 0.0], 760.0, "mmHg", "Celsius")
    tb_ethanol = engine.boiling_temperature(registry["ethanol"].antoine, 760.0,
                                            pressure_unit="mmHg", temp_unit="Celsius")
    assert t == pytest.approx(tb_ethanol, abs=0.2)
    assert y[0] == pytest.approx(1.0, abs=1e-6)


def test_bubble_pressure_matches_modified_raoult(registry):
    model = models.build_model("NRTL", registry["ethanol"], registry["water"])
    x = np.array([0.3, 0.7])
    T_K = 351.15
    P, y = model.bubble_pressure_k(x, T_K)
    psat = model.psat(T_K)
    g = model.gamma(x, T_K)
    assert P == pytest.approx(float(np.sum(x * g * psat)))
    assert np.allclose(y, x * g * psat / P)


def test_nonideal_flash_two_phase_material_balance(registry):
    model = models.build_model("NRTL", registry["ethanol"], registry["water"])
    res = model.flash([0.5, 0.5], 82.0, 760.0)  # between bubble & dew → two-phase
    if res.regime == fe.REGIME_TWO_PHASE:
        beta = res.vapor_fraction
        assert np.allclose(res.z, (1 - beta) * res.x + beta * res.y, atol=1e-6)
        assert res.gamma is not None and np.all(res.gamma > 0)
