"""Tests for the validation framework: metrics, dataset integrity, model ranking."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from thermo import models
from thermo import thermo_engine as engine
from thermo import validation as val
from thermo.data_models import load_species

DATA_PATH = Path(__file__).resolve().parent.parent / "chemical_data.json"


@pytest.fixture(scope="module")
def registry():
    return load_species(DATA_PATH)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def test_metrics_values():
    m = val.metrics([1.0, 2.0, 3.0], [1.1, 1.9, 3.2])
    assert m["MAE"] == pytest.approx(0.13333, abs=1e-4)
    assert m["RMSE"] == pytest.approx(0.14142, abs=1e-4)
    assert m["MPE"] == pytest.approx(7.2222, abs=1e-3)


def test_metrics_perfect_prediction_is_zero():
    m = val.metrics([0.0, 0.5, 1.0], [0.0, 0.5, 1.0])
    assert m["MAE"] == 0.0 and m["RMSE"] == 0.0


# --------------------------------------------------------------------------- #
# Dataset integrity (physical-consistency guardrails)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ds", val.DATASETS, ids=lambda d: d.system)
def test_dataset_endpoints_and_monotonic_x(ds):
    x1, y1, T = ds.as_arrays()
    assert len(x1) == len(y1) == len(T)
    assert x1[0] == 0.0 and x1[-1] == 1.0
    assert y1[0] == 0.0 and y1[-1] == 1.0
    assert np.all(np.diff(x1) > 0)               # x increases monotonically
    assert np.all((y1 >= -1e-9) & (y1 <= 1 + 1e-9))


@pytest.mark.parametrize("ds", val.DATASETS, ids=lambda d: d.system)
def test_dataset_temperature_endpoints_match_pure_boiling_points(registry, ds):
    """T at x₁=0/1 must equal the pure-component boiling points (catches mislabels)."""
    sp1, sp2 = registry[ds.comp1_key], registry[ds.comp2_key]
    tb1 = engine.boiling_temperature(sp1.antoine, ds.pressure_mmHg,
                                     pressure_unit="mmHg", temp_unit="Celsius")
    tb2 = engine.boiling_temperature(sp2.antoine, ds.pressure_mmHg,
                                     pressure_unit="mmHg", temp_unit="Celsius")
    assert ds.T_C[0] == pytest.approx(tb2, abs=1.0)    # x₁=0 → pure component 2
    assert ds.T_C[-1] == pytest.approx(tb1, abs=1.0)   # x₁=1 → pure component 1


def test_azeotropic_datasets_contain_crossing():
    """Ethanol–water & acetone–methanol include an interior y₁ = x₁ point."""
    for ds in (val.ETHANOL_WATER, val.ACETONE_METHANOL):
        x1, y1, _ = ds.as_arrays()
        interior = (x1 > 0.02) & (x1 < 0.98)
        assert np.any(np.abs(y1 - x1)[interior] < 1e-3)


# --------------------------------------------------------------------------- #
# Model ranking — the headline scientific result
# --------------------------------------------------------------------------- #
def test_ideal_excellent_for_benzene_toluene(registry):
    model = models.build_model("Ideal (Raoult)", registry["benzene"], registry["toluene"])
    res = val.validate(model, val.BENZENE_TOLUENE)
    assert res.y1_metrics["RMSE"] < 0.02          # near-ideal → Raoult suffices
    assert res.y1_badge in ("Excellent", "Good")


def test_nonideal_beats_ideal_for_ethanol_water(registry):
    sp1, sp2 = registry["ethanol"], registry["water"]
    ideal = val.validate(models.build_model("Ideal (Raoult)", sp1, sp2), val.ETHANOL_WATER)
    nrtl = val.validate(models.build_model("NRTL", sp1, sp2), val.ETHANOL_WATER)
    # NRTL should be far better than ideal Raoult on this azeotropic system
    assert nrtl.y1_metrics["RMSE"] < 0.25 * ideal.y1_metrics["RMSE"]
    assert nrtl.y1_badge == "Excellent"
    assert ideal.y1_badge == "Poor"


@pytest.mark.parametrize("model_name", ["Wilson", "NRTL", "UNIQUAC"])
def test_nonideal_models_excellent_on_azeotropic_systems(registry, model_name):
    for ds in (val.ETHANOL_WATER, val.ACETONE_METHANOL):
        model = models.build_model(model_name, registry[ds.comp1_key], registry[ds.comp2_key])
        res = val.validate(model, ds)
        assert res.y1_metrics["MAE"] < 0.02, f"{model_name} {ds.system}"


def test_badges_map_to_thresholds():
    from thermo.validation import _badge, Y1_THRESHOLDS
    assert _badge(0.005, Y1_THRESHOLDS) == "Excellent"
    assert _badge(0.015, Y1_THRESHOLDS) == "Good"
    assert _badge(0.03, Y1_THRESHOLDS) == "Fair"
    assert _badge(0.10, Y1_THRESHOLDS) == "Poor"
