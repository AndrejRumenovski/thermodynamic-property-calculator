"""Tests for the activity-coefficient models (pure math, no VLE)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from thermo import models
from thermo.data_models import load_species

DATA_PATH = Path(__file__).resolve().parent.parent / "chemical_data.json"


@pytest.fixture(scope="module")
def registry():
    return load_species(DATA_PATH)


@pytest.fixture(scope="module")
def nonideal_models(registry):
    out = {}
    for name in ("Wilson", "NRTL", "UNIQUAC"):
        out[name] = models.build_model(name, registry["ethanol"], registry["water"]).activity
    return out


def test_ideal_gamma_is_unity():
    g = models.IdealModel(3).gamma([0.2, 0.3, 0.5], 350.0)
    assert np.allclose(g, 1.0)


def test_pure_component_limit_gamma_to_one(nonideal_models):
    """As x_i → 1, γ_i → 1 for every model."""
    for model in nonideal_models.values():
        g = model.gamma([1.0 - 1e-9, 1e-9], 351.0)
        assert g[0] == pytest.approx(1.0, abs=1e-4)


def test_activity_coefficients_positive(nonideal_models):
    for model in nonideal_models.values():
        for x1 in (0.1, 0.5, 0.9):
            g = model.gamma([x1, 1 - x1], 351.0)
            assert np.all(g > 0.0)


def test_positive_deviation_ethanol_water(nonideal_models):
    """Ethanol–water shows positive deviation: γ > 1 across the range."""
    for model in nonideal_models.values():
        g = model.gamma([0.5, 0.5], 351.0)
        assert np.all(g >= 1.0)


def test_infinite_dilution_gamma_physical(nonideal_models):
    """γ_ethanol^∞ in water is large and positive (~4–10).

    An independent check — γ∞ was not fitted. The three models agree on order of
    magnitude; the spread (NRTL/UNIQUAC ≈ 5, Wilson higher) is the expected
    consequence of fitting to a single azeotrope point.
    """
    for name, model in nonideal_models.items():
        g_inf = model.gamma([1e-8, 1 - 1e-8], 351.15)[0]
        assert 3.0 < g_inf < 12.0, f"{name}: γ∞={g_inf}"


def test_gibbs_duhem_area_consistency(nonideal_models):
    """∫₀¹ ln(γ1/γ2) dx1 = 0 for a thermodynamically consistent model at fixed T."""
    x1 = np.linspace(1e-4, 1 - 1e-4, 4001)
    for name, model in nonideal_models.items():
        ratio = np.array([np.log(model.gamma([xi, 1 - xi], 350.0))[0]
                          - np.log(model.gamma([xi, 1 - xi], 350.0))[1] for xi in x1])
        area = np.trapezoid(ratio, x1)
        assert abs(area) < 5e-3, f"{name}: area={area}"
