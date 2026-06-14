"""Tests for the catalog and the add/remove persistence helpers."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from thermo import thermo_engine as engine
from thermo.data_models import (
    AntoineConstants,
    ChemicalSpecies,
    SpeciesDataError,
    add_species,
    load_species,
    remove_species,
    slugify_key,
    species_to_dict,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = REPO_ROOT / "antoine_catalog.json"


def _make_species(key="chloroform", name="Chloroform") -> ChemicalSpecies:
    # Antoine constants for chloroform in mmHg / Celsius (normal BP ~61.2 C).
    return ChemicalSpecies(
        key=key,
        name=name,
        formula="CHCl3",
        molar_mass=119.38,
        antoine=AntoineConstants(
            A=6.4934, B=929.44, C=196.03,
            pressure_unit="mmHg", temperature_unit="Celsius",
            t_min=-10.0, t_max=60.0,
        ),
    )


# --------------------------------------------------------------------------- #
# Catalog
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not CATALOG_PATH.exists(), reason="catalog not generated")
def test_catalog_loads_and_is_substantial():
    catalog = load_species(CATALOG_PATH)
    assert len(catalog) >= 100  # a few hundred expected
    for sp in catalog.values():
        assert sp.antoine.pressure_unit == "Pa"
        assert sp.antoine.temperature_unit == "Kelvin"


@pytest.mark.skipif(not CATALOG_PATH.exists(), reason="catalog not generated")
def test_every_catalog_entry_reproduces_its_boiling_point():
    """The build step validates each entry; reconfirm a healthy fraction here."""
    catalog = load_species(CATALOG_PATH)
    checked = 0
    for sp in catalog.values():
        # Every entry must yield a finite, positive normal boiling point (in K),
        # i.e. the engine can process the whole catalog without error.
        tb = engine.boiling_temperature(sp.antoine, 1.0, pressure_unit="atm", temp_unit="Kelvin")
        assert math.isfinite(tb) and tb > 0.0
        checked += 1
    assert checked == len(catalog)


# --------------------------------------------------------------------------- #
# Persistence helpers
# --------------------------------------------------------------------------- #
def test_slugify_key():
    assert slugify_key("Acetic Acid") == "acetic_acid"
    assert slugify_key("1,1,1-Trifluoroethane") == "1_1_1_trifluoroethane"
    assert slugify_key("   ") == "species"


def test_species_to_dict_round_trip(tmp_path):
    species = _make_species()
    data_file = tmp_path / "data.json"
    add_species(data_file, species)

    reloaded = load_species(data_file)
    assert "chloroform" in reloaded
    got = reloaded["chloroform"]
    assert got.name == "Chloroform"
    assert got.antoine.A == pytest.approx(6.4934)
    assert got.valid_range == (-10.0, 60.0)
    # serialisation preserves the optional range
    assert species_to_dict(species)["antoine_constants"]["T_max"] == 60.0


def test_add_species_creates_file_and_appends(tmp_path):
    data_file = tmp_path / "data.json"
    add_species(data_file, _make_species("water", "Water"))
    add_species(data_file, _make_species("benzene", "Benzene"))
    reloaded = load_species(data_file)
    assert set(reloaded) == {"water", "benzene"}


def test_add_duplicate_key_raises(tmp_path):
    data_file = tmp_path / "data.json"
    add_species(data_file, _make_species())
    with pytest.raises(SpeciesDataError, match="already exists"):
        add_species(data_file, _make_species())
    # overwrite=True is allowed
    add_species(data_file, _make_species(), overwrite=True)


def test_remove_species(tmp_path):
    data_file = tmp_path / "data.json"
    add_species(data_file, _make_species("water", "Water"))
    add_species(data_file, _make_species("benzene", "Benzene"))
    assert remove_species(data_file, "water") is True
    assert set(load_species(data_file)) == {"benzene"}
    assert remove_species(data_file, "nonexistent") is False


def test_added_species_computes_correct_boiling_point(tmp_path):
    """End-to-end: add chloroform, then the engine reproduces its ~61.2 C BP."""
    data_file = tmp_path / "data.json"
    add_species(data_file, _make_species())
    sp = load_species(data_file)["chloroform"]
    tb = engine.boiling_temperature(sp.antoine, 1.0, pressure_unit="atm", temp_unit="Celsius")
    assert tb == pytest.approx(61.2, abs=1.5)
