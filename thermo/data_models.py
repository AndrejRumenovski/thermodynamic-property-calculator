"""Layer 1 — data models and JSON loading.

Defines the schema for a chemical species and its Antoine constants, and loads
them from a structured JSON file. This layer is intentionally free of any
numerical or UI dependencies (no NumPy, SciPy, or Streamlit) so it can be reused
and unit-tested in isolation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

# Unit labels recognised by the engine. Kept here so the loader can validate the
# data file up-front rather than failing deep inside a calculation.
SUPPORTED_PRESSURE_UNITS = ("mmHg", "Pa", "kPa", "bar", "atm")
SUPPORTED_TEMPERATURE_UNITS = ("Celsius", "Kelvin", "Fahrenheit")


class SpeciesDataError(ValueError):
    """Raised when the chemical data file is missing fields or malformed."""


@dataclass(frozen=True)
class AntoineConstants:
    """Antoine-equation coefficients for a single species.

    The constants are unit-specific: ``A``, ``B`` and ``C`` only make sense
    together with ``pressure_unit`` and ``temperature_unit``. The optional
    ``t_min``/``t_max`` describe the temperature range (in ``temperature_unit``)
    over which the correlation is considered valid.
    """

    A: float
    B: float
    C: float
    pressure_unit: str
    temperature_unit: str
    t_min: Optional[float] = None
    t_max: Optional[float] = None

    def __post_init__(self) -> None:
        if self.pressure_unit not in SUPPORTED_PRESSURE_UNITS:
            raise SpeciesDataError(
                f"Unsupported pressure unit {self.pressure_unit!r}; "
                f"expected one of {SUPPORTED_PRESSURE_UNITS}."
            )
        if self.temperature_unit not in SUPPORTED_TEMPERATURE_UNITS:
            raise SpeciesDataError(
                f"Unsupported temperature unit {self.temperature_unit!r}; "
                f"expected one of {SUPPORTED_TEMPERATURE_UNITS}."
            )
        if self.t_min is not None and self.t_max is not None and self.t_min > self.t_max:
            raise SpeciesDataError(
                f"t_min ({self.t_min}) must not exceed t_max ({self.t_max})."
            )


@dataclass(frozen=True)
class ChemicalSpecies:
    """A chemical species and the data needed to compute its properties."""

    key: str
    name: str
    formula: str
    molar_mass: float  # g/mol
    antoine: AntoineConstants

    @property
    def valid_range(self) -> Optional[tuple[float, float]]:
        """Return ``(t_min, t_max)`` if both are defined, else ``None``."""
        if self.antoine.t_min is None or self.antoine.t_max is None:
            return None
        return (self.antoine.t_min, self.antoine.t_max)


def _require(mapping: dict, key: str, context: str) -> object:
    if not isinstance(mapping, dict):
        raise SpeciesDataError(f"{context} must be a JSON object.")
    if key not in mapping:
        raise SpeciesDataError(f"{context} is missing required field {key!r}.")
    return mapping[key]


def _require_number(mapping: dict, key: str, context: str) -> float:
    value = _require(mapping, key, context)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SpeciesDataError(f"{context} field {key!r} must be a number, got {value!r}.")
    return float(value)


def _parse_species(key: str, raw: dict) -> ChemicalSpecies:
    context = f"Species {key!r}"
    name = _require(raw, "name", context)
    formula = _require(raw, "formula", context)
    molar_mass = _require_number(raw, "molar_mass", context)

    antoine_raw = _require(raw, "antoine_constants", context)
    a_context = f"{context} antoine_constants"
    if not isinstance(antoine_raw, dict):
        raise SpeciesDataError(f"{a_context} must be a JSON object.")

    units = _require(antoine_raw, "units", a_context)
    if not isinstance(units, dict):
        raise SpeciesDataError(f"{a_context} field 'units' must be a JSON object.")

    antoine = AntoineConstants(
        A=_require_number(antoine_raw, "A", a_context),
        B=_require_number(antoine_raw, "B", a_context),
        C=_require_number(antoine_raw, "C", a_context),
        pressure_unit=str(_require(units, "P", f"{a_context} units")),
        temperature_unit=str(_require(units, "T", f"{a_context} units")),
        t_min=(
            float(antoine_raw["T_min"]) if antoine_raw.get("T_min") is not None else None
        ),
        t_max=(
            float(antoine_raw["T_max"]) if antoine_raw.get("T_max") is not None else None
        ),
    )

    return ChemicalSpecies(
        key=key,
        name=str(name),
        formula=str(formula),
        molar_mass=molar_mass,
        antoine=antoine,
    )


def load_species(path: Union[str, Path]) -> dict[str, ChemicalSpecies]:
    """Load and validate species from a JSON data file.

    Parameters
    ----------
    path:
        Path to a JSON file shaped like ``{"species": {key: {...}}}``.

    Returns
    -------
    dict[str, ChemicalSpecies]
        Mapping of species key to the parsed, validated species.

    Raises
    ------
    SpeciesDataError
        If the file is missing, not valid JSON, or does not match the schema.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SpeciesDataError(f"Could not read data file {path}: {exc}") from exc

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SpeciesDataError(f"{path} is not valid JSON: {exc}") from exc

    species_block = _require(payload, "species", "Top-level object")
    if not isinstance(species_block, dict) or not species_block:
        raise SpeciesDataError("'species' must be a non-empty JSON object.")

    return {
        key: _parse_species(key, raw) for key, raw in species_block.items()
    }


# --------------------------------------------------------------------------- #
# Writing / persistence
# --------------------------------------------------------------------------- #
def slugify_key(name: str) -> str:
    """Turn a display name into a JSON-friendly species key."""
    slug = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    return slug or "species"


def species_to_dict(species: ChemicalSpecies) -> dict:
    """Serialise a :class:`ChemicalSpecies` back to the on-disk JSON shape."""
    antoine: dict = {
        "A": species.antoine.A,
        "B": species.antoine.B,
        "C": species.antoine.C,
        "units": {
            "P": species.antoine.pressure_unit,
            "T": species.antoine.temperature_unit,
        },
    }
    if species.antoine.t_min is not None:
        antoine["T_min"] = species.antoine.t_min
    if species.antoine.t_max is not None:
        antoine["T_max"] = species.antoine.t_max
    return {
        "name": species.name,
        "formula": species.formula,
        "molar_mass": species.molar_mass,
        "antoine_constants": antoine,
    }


def _read_payload(path: Path) -> dict:
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SpeciesDataError(f"{path} is not valid JSON: {exc}") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("species"), dict):
            raise SpeciesDataError(f"{path} is not a valid species data file.")
        return payload
    return {"species": {}}


def _write_payload(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def add_species(
    path: Union[str, Path], species: ChemicalSpecies, overwrite: bool = False
) -> None:
    """Add ``species`` to the JSON file at ``path``, persisting it to disk.

    Creates the file if it does not exist. Raises :class:`SpeciesDataError` if
    the key is already present and ``overwrite`` is ``False``.
    """
    path = Path(path)
    payload = _read_payload(path)
    block = payload["species"]
    if species.key in block and not overwrite:
        raise SpeciesDataError(
            f"A species with key {species.key!r} already exists; "
            f"choose a different name or enable overwrite."
        )
    block[species.key] = species_to_dict(species)
    _write_payload(path, payload)


def remove_species(path: Union[str, Path], key: str) -> bool:
    """Remove the species ``key`` from the file. Returns ``True`` if removed."""
    path = Path(path)
    payload = _read_payload(path)
    block = payload["species"]
    if key in block:
        del block[key]
        _write_payload(path, payload)
        return True
    return False
