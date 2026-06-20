"""Tests for the binary phase-diagram generator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from thermo import diagrams, models
from thermo.data_models import load_species

DATA_PATH = Path(__file__).resolve().parent.parent / "chemical_data.json"


@pytest.fixture(scope="module")
def registry():
    return load_species(DATA_PATH)


def test_txy_shape_and_endpoints(registry):
    model = models.build_model("NRTL", registry["ethanol"], registry["water"])
    df = diagrams.txy(model, 760.0, n=51)
    assert list(df.columns) == ["x1", "y1", "T"]
    assert len(df) == 51
    # endpoints are the pure-component boiling points; ethanol (78.4) < water (100)
    assert df["T"].iloc[0] == pytest.approx(100.0, abs=0.6)   # x1=0 → pure water
    assert df["T"].iloc[-1] == pytest.approx(78.4, abs=0.6)   # x1=1 → pure ethanol
    assert df["y1"].iloc[0] == pytest.approx(0.0, abs=1e-6)
    assert df["y1"].iloc[-1] == pytest.approx(1.0, abs=1e-6)


def test_txy_dew_above_bubble_is_consistent(registry):
    """For each interior composition the vapor is richer in the lighter species."""
    model = models.build_model("NRTL", registry["ethanol"], registry["water"])
    df = diagrams.txy(model, 760.0, n=81)
    # below the azeotrope (x1 < ~0.894) ethanol enriches the vapor: y1 > x1
    sub = df[(df["x1"] > 0.05) & (df["x1"] < 0.8)]
    assert np.all(sub["y1"] >= sub["x1"] - 1e-9)


def test_pxy_shape_and_monotone_pure_pressures(registry):
    model = models.build_model("NRTL", registry["acetone"], registry["methanol"])
    df = diagrams.pxy(model, 55.0, n=51)
    assert list(df.columns) == ["x1", "y1", "P"]
    # acetone is more volatile → higher pure vapor pressure at fixed T
    assert df["P"].iloc[-1] > df["P"].iloc[0]
    # positive-deviation azeotrope → interior pressure maximum exceeds both pure ends
    assert df["P"].max() > df["P"].iloc[0] + 1.0
    assert df["P"].max() > df["P"].iloc[-1] + 1.0


def test_ideal_txy_is_monotonic_no_azeotrope(registry):
    model = models.build_model("Ideal (Raoult)", registry["benzene"], registry["toluene"])
    df = diagrams.txy(model, 760.0, n=51)
    # benzene–toluene is near-ideal: bubble T decreases monotonically with x1(benzene)
    assert np.all(np.diff(df["T"].to_numpy()) < 1e-6)
