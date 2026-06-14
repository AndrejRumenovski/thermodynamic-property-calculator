# PLAN — Thermodynamic Property Calculator

> Required by the project brief: the file structure and data schema, approved
> before implementation. This is the as-built plan.

## Goal

A modular, data-driven Python app that calculates thermophysical properties
(vapor pressure ⇄ boiling temperature) for chemical species via the Antoine
equation, with a Streamlit UI and NumPy/SciPy numerics.

```
log10(P) = A − B / (T + C)          forward   (T → P)
T        = B / (A − log10(P)) − C   inverse   (P → T, boiling temperature)
```

Antoine constants are unit-specific (bundled data is **mmHg / °C**); the engine
evaluates in each species' native units and converts user input/output to/from
the selected display units.

## File structure

```
thermodynamic-property-calculator/
├── PLAN.md                  # this plan
├── README.md                # setup, run, test, data-schema docs
├── requirements.txt         # streamlit, numpy, scipy, pandas, pytest
├── chemical_data.json       # data-driven species constants
├── app.py                   # Streamlit entry point: `streamlit run app.py`
├── conftest.py              # puts repo root on sys.path for tests
├── thermo/                  # importable package (no install needed)
│   ├── __init__.py
│   ├── data_models.py       # LAYER 1: dataclasses + JSON loader/validator
│   ├── thermo_engine.py     # LAYER 2: pure Antoine math + unit conversion
│   └── interface.py         # LAYER 3: Streamlit rendering (called by app.py)
└── tests/
    ├── __init__.py
    ├── test_data_models.py
    └── test_thermo_engine.py
```

The three layers required by the brief map to `data_models`, `thermo_engine`,
and `interface`. Keeping `thermo/` at the repo root means `streamlit run app.py`
and `pytest` both work from the root with no editable install.

## Data schema (`chemical_data.json`)

Keeps the brief's shape (units nested under `antoine_constants`, since Antoine
constants are tied to a unit system) plus an optional validated temperature
range:

```jsonc
{
  "species": {
    "water": {
      "name": "Water",
      "formula": "H2O",
      "molar_mass": 18.015,
      "antoine_constants": {
        "A": 8.07131, "B": 1730.63, "C": 233.426,
        "units": { "P": "mmHg", "T": "Celsius" },
        "T_min": 1.0, "T_max": 100.0
      }
    }
  }
}
```

Dataset: the brief's **water** and **benzene** verbatim, plus common solvents
(**toluene, ethanol, methanol, acetone**) with standard mmHg/°C Antoine
constants. Each is validated in tests against its known normal boiling point.

## Layers

- **`data_models.py`** — `AntoineConstants` and `ChemicalSpecies` dataclasses;
  `load_species(path)` parses/validates JSON and raises `SpeciesDataError` on
  malformed data. No NumPy/Streamlit imports.
- **`thermo_engine.py`** — pure functions: `vapor_pressure` (forward),
  `boiling_temperature` (inverse via `scipy.optimize.brentq`, analytic bracket),
  `vapor_pressure_curve` (NumPy-vectorised for plotting), pressure/temperature
  unit conversions, domain guards and a soft out-of-range warning.
- **`interface.py` + `app.py`** — Streamlit UI: pick species + direction +
  units; results table, species metadata, vapor-pressure-vs-temperature chart,
  graceful error display.

## Tests (`pytest`)

Loader/schema validation; physical anchors (water vapor pressure at 100 °C ≈
760 mmHg, benzene boils at ≈ 80.1 °C at 1 atm, every solvent's normal boiling
point within ±1 °C); `T → P → T` round-trip; unit conversions.

## Verification

1. `py -m venv .venv` → install `requirements.txt` into it.
2. `pytest -q` → all green.
3. `streamlit run app.py` (headless) → boots and renders water ≈ 760 mmHg at
   100 °C.
