"""Thermodynamic property calculator package.

Three layers, kept deliberately separate:

* :mod:`thermo.data_models`  – data schema + JSON loader (no math, no UI)
* :mod:`thermo.thermo_engine` – pure Antoine math + unit conversion (no I/O, no UI)
* :mod:`thermo.interface`     – Streamlit rendering (imported by ``app.py``)
"""

from .data_models import (
    AntoineConstants,
    ChemicalSpecies,
    SpeciesDataError,
    add_species,
    load_species,
    remove_species,
    slugify_key,
    species_to_dict,
)

__all__ = [
    "AntoineConstants",
    "ChemicalSpecies",
    "SpeciesDataError",
    "add_species",
    "load_species",
    "remove_species",
    "slugify_key",
    "species_to_dict",
]
