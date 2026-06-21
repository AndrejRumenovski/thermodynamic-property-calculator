"""Tests for the McCabe–Thiele binary distillation engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from thermo import distillation as dist
from thermo import models
from thermo.data_models import load_species

DATA_PATH = Path(__file__).resolve().parent.parent / "chemical_data.json"


@pytest.fixture(scope="module")
def registry():
    return load_species(DATA_PATH)


@pytest.fixture(scope="module")
def benzene_toluene(registry):
    return models.build_model("Ideal (Raoult)", registry["benzene"], registry["toluene"])


def test_classic_case_is_feasible_and_reasonable(benzene_toluene):
    res = dist.mccabe_thiele(benzene_toluene, z_F=0.5, q=1.0, R=2.0, x_D=0.95, x_B=0.05)
    assert res.feasible
    assert 0.9 < res.R_min < 1.4              # textbook benzene–toluene value
    assert 8 <= res.n_stages <= 16
    assert 1 < res.feed_stage < res.n_stages


def test_operating_line_geometry(benzene_toluene):
    res = dist.mccabe_thiele(benzene_toluene, 0.5, 1.0, 2.0, 0.95, 0.05)
    # ROL passes through (x_D, x_D) with slope R/(R+1)
    assert res.rol(res.x_D) == pytest.approx(res.x_D, abs=1e-9)
    assert (res.rol(1.0) - res.rol(0.0)) == pytest.approx(res.R / (res.R + 1.0), abs=1e-9)
    # SOL passes through (x_B, x_B)
    assert res.sol(res.x_B) == pytest.approx(res.x_B, abs=1e-9)
    # q=1 → intersection at x = z_F
    assert res.intersection[0] == pytest.approx(res.z_F, abs=1e-9)


def test_more_reflux_needs_fewer_stages(benzene_toluene):
    low = dist.mccabe_thiele(benzene_toluene, 0.5, 1.0, 1.5, 0.95, 0.05)
    high = dist.mccabe_thiele(benzene_toluene, 0.5, 1.0, 5.0, 0.95, 0.05)
    assert high.n_stages <= low.n_stages


def test_below_minimum_reflux_is_infeasible(benzene_toluene):
    res = dist.mccabe_thiele(benzene_toluene, 0.5, 1.0, 0.3, 0.95, 0.05)
    assert not res.feasible
    assert "minimum" in res.message.lower()
    # R below R_min by construction
    assert res.R < res.R_min


def test_stage_corners_lie_on_equilibrium_curve(benzene_toluene):
    res = dist.mccabe_thiele(benzene_toluene, 0.5, 1.0, 2.0, 0.95, 0.05)
    for x, y in res.stage_corners:
        y_eq = float(np.interp(x, res.x_eq, res.y_eq))
        assert y == pytest.approx(y_eq, abs=2e-3)


def test_tighter_separation_needs_more_stages(benzene_toluene):
    loose = dist.mccabe_thiele(benzene_toluene, 0.5, 1.0, 2.0, 0.90, 0.10)
    tight = dist.mccabe_thiele(benzene_toluene, 0.5, 1.0, 2.0, 0.99, 0.01)
    assert tight.n_stages > loose.n_stages


def test_invalid_specifications_raise(benzene_toluene):
    with pytest.raises(dist.DistillationError):
        dist.mccabe_thiele(benzene_toluene, z_F=0.5, q=1.0, R=2.0, x_D=0.4, x_B=0.05)  # x_D<z_F
    with pytest.raises(dist.DistillationError):
        dist.mccabe_thiele(benzene_toluene, z_F=0.5, q=1.0, R=2.0, x_D=0.95, x_B=0.6)  # x_B>z_F


def test_saturated_vapor_feed_q_zero(benzene_toluene):
    # A saturated-vapor feed raises R_min, so use a larger R to stay feasible.
    res = dist.mccabe_thiele(benzene_toluene, 0.5, 0.0, 4.0, 0.95, 0.05)
    assert res.feasible
    # q=0 → q-line is horizontal at y = z_F → intersection y-coordinate ≈ z_F
    assert res.intersection[1] == pytest.approx(res.z_F, abs=1e-6)
    # vapor feed needs more reflux than the saturated-liquid case
    liquid_feed = dist.mccabe_thiele(benzene_toluene, 0.5, 1.0, 4.0, 0.95, 0.05)
    assert res.R_min > liquid_feed.R_min
