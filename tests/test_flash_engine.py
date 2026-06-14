"""Tests for the Rachford-Rice VLE flash engine.

Anchored to a benzene/toluene binary (a textbook nearly-ideal system) and to the
conservation laws every flash must satisfy: overall material balance, normalised
phase compositions, y = K x, and a vanishing Rachford-Rice residual.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from thermo import flash_engine as fe
from thermo.data_models import load_species

DATA_PATH = Path(__file__).resolve().parent.parent / "chemical_data.json"


@pytest.fixture(scope="module")
def registry():
    return load_species(DATA_PATH)


@pytest.fixture(scope="module")
def binary(registry):
    return [registry["benzene"], registry["toluene"]]


# --------------------------------------------------------------------------- #
# Two-phase regime
# --------------------------------------------------------------------------- #
def test_two_phase_regime_and_invariants(binary):
    res = fe.flash(binary, [0.5, 0.5], temperature=95.0, pressure=750.0)
    assert res.regime == fe.REGIME_TWO_PHASE
    assert 0.0 < res.vapor_fraction < 1.0

    # phase compositions are normalised
    assert res.x.sum() == pytest.approx(1.0)
    assert res.y.sum() == pytest.approx(1.0)
    # equilibrium relation
    assert np.allclose(res.y, res.K * res.x)
    # overall material balance: z = (1-β) x + β y
    beta = res.vapor_fraction
    assert np.allclose(res.z, (1.0 - beta) * res.x + beta * res.y)
    # the solved β is a genuine root of Rachford-Rice
    assert fe.rachford_rice_objective(beta, res.z, res.K) == pytest.approx(0.0, abs=1e-9)


def test_two_phase_lies_between_dew_and_bubble(binary):
    res = fe.flash(binary, [0.4, 0.6], temperature=95.0, pressure=750.0)
    assert res.dew_pressure < res.pressure < res.bubble_pressure


# --------------------------------------------------------------------------- #
# Single-phase regimes
# --------------------------------------------------------------------------- #
def test_subcooled_liquid_high_pressure(binary):
    res = fe.flash(binary, [0.5, 0.5], temperature=95.0, pressure=5000.0)
    assert res.regime == fe.REGIME_SUBCOOLED
    assert res.vapor_fraction == 0.0
    assert np.allclose(res.x, res.z)          # the liquid is the whole feed
    assert np.all(np.isnan(res.y))            # no vapor present
    assert res.pressure >= res.bubble_pressure


def test_superheated_vapor_low_pressure(binary):
    res = fe.flash(binary, [0.5, 0.5], temperature=95.0, pressure=100.0)
    assert res.regime == fe.REGIME_SUPERHEATED
    assert res.vapor_fraction == 1.0
    assert np.allclose(res.y, res.z)          # the vapor is the whole feed
    assert np.all(np.isnan(res.x))            # no liquid present
    assert res.pressure <= res.dew_pressure


def test_bubble_point_boundary_is_liquid(binary):
    """At P == bubble pressure the mixture is at its bubble point (β → 0)."""
    res = fe.flash(binary, [0.5, 0.5], temperature=95.0, pressure=750.0)
    at_bubble = fe.flash(binary, [0.5, 0.5], temperature=95.0, pressure=res.bubble_pressure)
    assert at_bubble.vapor_fraction == pytest.approx(0.0, abs=1e-6)


def test_dew_point_boundary_is_vapor(binary):
    res = fe.flash(binary, [0.5, 0.5], temperature=95.0, pressure=750.0)
    at_dew = fe.flash(binary, [0.5, 0.5], temperature=95.0, pressure=res.dew_pressure)
    assert at_dew.vapor_fraction == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# K-values and Psat come from the shared engine
# --------------------------------------------------------------------------- #
def test_k_values_are_psat_over_p(binary):
    K, psat = fe.k_values(binary, temperature=95.0, pressure=760.0,
                          temp_unit="Celsius", pressure_unit="mmHg")
    assert np.allclose(K, psat / 760.0)
    # benzene is more volatile than toluene -> larger Psat and K
    assert K[0] > K[1]


def test_mixed_unit_components_supported(registry):
    """Components with different native units (mmHg/C and Pa/K) flash cleanly."""
    species = [registry["benzene"], registry["cyclopropane"]]
    res = fe.flash(species, [0.5, 0.5], temperature=25.0, pressure=760.0)
    assert res.regime in (
        fe.REGIME_SUBCOOLED, fe.REGIME_TWO_PHASE, fe.REGIME_SUPERHEATED
    )
    assert np.all(np.isfinite(res.K))
    # cyclopropane (a near-gas at 25 C) is far more volatile than benzene
    assert res.K[1] > res.K[0]


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #
def test_requires_two_components(registry):
    with pytest.raises(fe.FlashError, match="at least two"):
        fe.flash([registry["benzene"]], [1.0], temperature=95.0, pressure=760.0)


def test_composition_length_must_match(binary):
    with pytest.raises(fe.FlashError, match="match"):
        fe.flash(binary, [0.5, 0.3, 0.2], temperature=95.0, pressure=760.0)


def test_negative_pressure_rejected(binary):
    with pytest.raises(fe.FlashError, match="positive"):
        fe.flash(binary, [0.5, 0.5], temperature=95.0, pressure=-10.0)


def test_unnormalised_composition_is_normalised(binary):
    res = fe.flash(binary, [2.0, 2.0], temperature=95.0, pressure=750.0)
    assert res.z.sum() == pytest.approx(1.0)
    assert np.allclose(res.z, [0.5, 0.5])
