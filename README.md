# Thermodynamic Property Calculator

A modular, data-driven Python application that calculates thermophysical
properties of chemical species from the **Antoine equation** — converting
between **vapor pressure** and **boiling temperature**, with unit handling — and
presents them through a **Streamlit** UI. Numerics use **NumPy** and **SciPy**.

```
log10(P) = A − B / (T + C)          forward   (T → P)
T        = B / (A − log10(P)) − C   inverse   (P → T, boiling temperature)
```

## Architecture

Three deliberately separated layers (see [`PLAN.md`](PLAN.md)):

| Layer | Module | Responsibility |
| --- | --- | --- |
| Data models | [`thermo/data_models.py`](thermo/data_models.py) | Dataclasses + JSON loader/validator. No math, no UI. |
| Engine | [`thermo/thermo_engine.py`](thermo/thermo_engine.py) | Pure Antoine math + unit conversion (NumPy/SciPy). No I/O, no UI. |
| Interface | [`thermo/interface.py`](thermo/interface.py) | Streamlit rendering, called by [`app.py`](app.py). |

Chemical constants live in [`chemical_data.json`](chemical_data.json) — the app
is entirely data-driven, so adding a species is a JSON edit, not a code change.

## Setup

Python 3.13 is used here via the `py` launcher (on this machine the bare
`python` is the Microsoft Store stub). Adjust to `python` if that is your real
interpreter.

```powershell
py -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

## Run the app

```powershell
.\.venv\Scripts\streamlit run app.py
```

Then pick species, choose a calculation direction (T → P or P → T), select
units, and read the results table and vapor-pressure curve.

## Run the tests

```powershell
.\.venv\Scripts\python -m pytest -q
```

Tests anchor to physical reality: water boils at ~100 °C and benzene at
~80.1 °C at 1 atm, water's vapor pressure at 100 °C is ~760 mmHg, and
`T → P → T` round-trips.

## Data schema

```jsonc
{
  "species": {
    "<key>": {
      "name": "Water",
      "formula": "H2O",
      "molar_mass": 18.015,            // g/mol
      "antoine_constants": {
        "A": 8.07131, "B": 1730.63, "C": 233.426,
        "units": { "P": "mmHg", "T": "Celsius" },
        "T_min": 1.0, "T_max": 100.0   // optional validated range
      }
    }
  }
}
```

Supported units — pressure: `mmHg`, `Pa`, `kPa`, `bar`, `atm`; temperature:
`Celsius`, `Kelvin`, `Fahrenheit`. Antoine constants are unit-specific; the
engine evaluates in each species' native units and converts your inputs/outputs
to the units you select. Values outside `[T_min, T_max]` are flagged as
extrapolations.
