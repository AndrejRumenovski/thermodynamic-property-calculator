"""Tests for the data-model / loader layer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from thermo.data_models import (
    AntoineConstants,
    SpeciesDataError,
    load_species,
)

DATA_PATH = Path(__file__).resolve().parent.parent / "chemical_data.json"


def test_loads_bundled_data():
    registry = load_species(DATA_PATH)
    assert {"water", "benzene", "toluene", "ethanol", "methanol", "acetone"} <= set(registry)

    water = registry["water"]
    assert water.name == "Water"
    assert water.formula == "H2O"
    assert water.molar_mass == pytest.approx(18.015)
    assert water.antoine.A == pytest.approx(8.07131)
    assert water.antoine.pressure_unit == "mmHg"
    assert water.antoine.temperature_unit == "Celsius"


def test_valid_range_property():
    registry = load_species(DATA_PATH)
    assert registry["water"].valid_range == (1.0, 100.0)


def _write(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "data.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_missing_field_raises(tmp_path):
    payload = {"species": {"x": {"name": "X", "formula": "X"}}}  # no molar_mass / antoine
    with pytest.raises(SpeciesDataError, match="molar_mass"):
        load_species(_write(tmp_path, payload))


def test_bad_unit_raises(tmp_path):
    payload = {
        "species": {
            "x": {
                "name": "X",
                "formula": "X",
                "molar_mass": 1.0,
                "antoine_constants": {
                    "A": 1, "B": 2, "C": 3,
                    "units": {"P": "psi", "T": "Celsius"},
                },
            }
        }
    }
    with pytest.raises(SpeciesDataError, match="pressure unit"):
        load_species(_write(tmp_path, payload))


def test_invalid_json_raises(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(SpeciesDataError, match="not valid JSON"):
        load_species(path)


def test_missing_file_raises(tmp_path):
    with pytest.raises(SpeciesDataError, match="Could not read"):
        load_species(tmp_path / "does_not_exist.json")


def test_empty_species_raises(tmp_path):
    with pytest.raises(SpeciesDataError, match="non-empty"):
        load_species(_write(tmp_path, {"species": {}}))


def test_antoine_rejects_inverted_range():
    with pytest.raises(SpeciesDataError, match="t_min"):
        AntoineConstants(1, 2, 3, "mmHg", "Celsius", t_min=100.0, t_max=0.0)
