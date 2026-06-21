# Thermodynamic Property Calculator

A modular, data-driven Python application for chemical-engineering VLE work. It
opens on a **research Dashboard** (project KPIs, status indicators, and module
cards), with seven tools selectable from the sidebar:

1. **Property Lookup** — vapor pressure ⇄ boiling temperature from the
   **Antoine equation**, with unit handling.
2. **VLE Flash Calculation** — isothermal multicomponent flash via the
   **Rachford-Rice** equation, with a selectable activity-coefficient model
   (Raoult / Wilson / NRTL / UNIQUAC), classifying the feed as subcooled liquid,
   two-phase, or superheated vapor.
3. **Phase Diagram** — publication-quality **T–x–y / P–x–y** diagrams for binary
   systems (zoom, PNG/SVG/CSV export), with automatic azeotrope detection.
4. **Model Validation** — scores each model against literature VLE datasets
   (MAE / RMSE / mean % error), with sortable tables, quality badges, and
   predicted-vs-experimental parity plots.
5. **Distillation** — binary **McCabe–Thiele** simulator: operating lines,
   q-line, stage stepping against the real equilibrium curve, R_min, feed-stage
   location, theoretical stages, an interactive stage slider, and full
   calculation breakdown.
6. **Property Prediction** — **Joback** group-contribution estimates of Tb, Tc,
   Pc, Vc, ω, and molar mass (vapor pressure via Lee–Kesler corresponding
   states), with a benchmarking dashboard (MAE / RMSE / R²) against a reference
   library and parity plots.
7. **Molecular Viewer** — **RDKit** 2D depiction + interactive **3Dmol.js** 3D
   structure (rotate / zoom / atom labels) for any SMILES, with formula, molar
   mass, atom counts, and predicted properties.

Built on a **Streamlit** UI with **NumPy**, **SciPy**, and **Plotly** numerics. The
interface uses a terminal-style **command bar** (active module + live stats),
a footer **status bar**, collapsible input panels, a **Comfortable/Compact**
density control, and a single phase-colour system — **liquid = blue, vapor =
orange, two-phase = green** — applied consistently across charts, tables, badges,
and indicators.

```
log10(P) = A − B / (T + C)                          Antoine          (T ⇄ P)
K_i = γ_i(x,T) · P_i^sat(T) / P                     modified Raoult  (γ from the model)
Σ_i z_i (K_i − 1) / (1 + β(K_i − 1)) = 0            Rachford-Rice    (solve for β = V/F)
```

Non-ideal models (Wilson/NRTL/UNIQUAC) reproduce real **azeotropes** — e.g.
ethanol–water (x₁≈0.894, 78.15 °C) and acetone–methanol (x₁≈0.80, 55.5 °C). See
[`docs/thermodynamics.md`](docs/thermodynamics.md) for equations, parameters, and
provenance.

## Architecture

Deliberately separated layers (see [`PLAN.md`](PLAN.md)):

| Layer | Module | Responsibility |
| --- | --- | --- |
| Data models | [`thermo/data_models.py`](thermo/data_models.py) | Dataclasses + JSON loader/validator. No math, no UI. |
| Property engine | [`thermo/thermo_engine.py`](thermo/thermo_engine.py) | Pure Antoine math + unit conversion (NumPy/SciPy). No I/O, no UI. |
| Flash engine | [`thermo/flash_engine.py`](thermo/flash_engine.py) | Rachford-Rice VLE flash (SciPy `brentq`). Gets `P^sat` from the property engine. No I/O, no UI. |
| Thermo models | [`thermo/models/`](thermo/models) | `ActivityModel` framework — Ideal/Wilson/NRTL/UNIQUAC + the `ThermodynamicModel` bubble/dew/flash solver. Pure math. |
| Diagrams | [`thermo/diagrams.py`](thermo/diagrams.py) | Binary T–x–y / P–x–y curve generation. |
| Validation | [`thermo/validation.py`](thermo/validation.py) | Literature VLE datasets + MAE/RMSE/% error scoring of any model. |
| Distillation | [`thermo/distillation.py`](thermo/distillation.py) | McCabe–Thiele operating lines, q-line, stage stepping, R_min, feed stage. |
| Property prediction | [`thermo/property_prediction.py`](thermo/property_prediction.py) | Joback group contribution + Lee–Kesler; predictor interface + benchmarking. |
| Molecular viz | [`thermo/molviz.py`](thermo/molviz.py) | RDKit 2D SVG + 3D MolBlock; 3Dmol.js viewer HTML. |
| Interface | [`thermo/interface.py`](thermo/interface.py) | Streamlit rendering (all modes), called by [`app.py`](app.py). |

The flash engine consumes the **same `ChemicalSpecies` objects** loaded from
`chemical_data.json` and calls `thermo_engine.vapor_pressure(...)` for every
`P^sat`, so the two engines stay consistent and components may even mix native
units (e.g. mmHg/°C with Pa/K).

Chemical constants live in [`chemical_data.json`](chemical_data.json) — the app
is entirely data-driven, so adding a species is a JSON edit, not a code change.

## Adding species

You don't have to hand-edit JSON. In the app's sidebar, open **➕ Add / manage
species**:

- **Catalog** — search a bundled reference catalog of a few hundred validated
  compounds ([`antoine_catalog.json`](antoine_catalog.json)) and add one with a
  click.
- **Manual** — enter Antoine constants for *any* species; the app validates them
  and flags values that don't reproduce a boiling point you provide.
- **Remove** — drop species you no longer want.

Additions are **saved to `chemical_data.json`**, so they persist across restarts
and can be committed.

### Where the catalog comes from (and why it's trustworthy)

The catalog is generated from the BSD-licensed
[`chemicals`](https://github.com/CalebBell/chemicals) library (Antoine data from
Poling et al., *The Properties of Gases and Liquids*) by
[`tools/build_catalog.py`](tools/build_catalog.py). Every entry is **validated**:
its constants must reproduce the compound's known normal boiling point within
5 K, or it's dropped. The library is a *build-time* dependency only — the app
itself just reads the bundled JSON and stays offline. To regenerate:

```powershell
.\.venv\Scripts\python -m pip install -r requirements-dev.txt
.\.venv\Scripts\python tools\build_catalog.py
```

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
