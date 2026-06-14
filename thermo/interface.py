"""Layer 3 — the Streamlit user interface.

This is the only layer that imports Streamlit. It wires user input to the
calculation engine and renders results; all numerics live in
:mod:`thermo.thermo_engine` and all data shaping in :mod:`thermo.data_models`.
Run it with ``streamlit run app.py`` from the repository root.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from . import thermo_engine as engine
from .data_models import (
    SUPPORTED_PRESSURE_UNITS,
    SUPPORTED_TEMPERATURE_UNITS,
    ChemicalSpecies,
    SpeciesDataError,
    load_species,
)

DATA_PATH = Path(__file__).resolve().parent.parent / "chemical_data.json"

MODE_P_FROM_T = "Vapor pressure from temperature  (T → P)"
MODE_T_FROM_P = "Boiling temperature from pressure  (P → T)"


@st.cache_data(show_spinner=False)
def _load(path_str: str) -> dict[str, ChemicalSpecies]:
    return load_species(path_str)


def _format_range(species: ChemicalSpecies) -> str:
    rng = species.valid_range
    if rng is None:
        return "—"
    unit = species.antoine.temperature_unit
    return f"{rng[0]:g} to {rng[1]:g} {unit}"


def _selected_species(
    registry: dict[str, ChemicalSpecies], keys: list[str]
) -> list[ChemicalSpecies]:
    return [registry[k] for k in keys]


def _vapor_pressure_table(
    species_list: list[ChemicalSpecies],
    temperature: float,
    temp_unit: str,
    pressure_unit: str,
) -> pd.DataFrame:
    rows = []
    for sp in species_list:
        row = {
            "Species": sp.name,
            "Formula": sp.formula,
            "Molar mass (g/mol)": sp.molar_mass,
            "Valid range": _format_range(sp),
        }
        try:
            t_native = engine.convert_temperature(
                temperature, temp_unit, sp.antoine.temperature_unit
            )
            in_range = engine.temperature_in_range(sp.antoine, t_native)
            pressure = engine.vapor_pressure(
                sp.antoine,
                temperature,
                temp_unit=temp_unit,
                pressure_unit=pressure_unit,
                check_range=False,
            )
            row[f"Vapor pressure ({pressure_unit})"] = round(pressure, 4)
            row["In range?"] = "yes" if in_range else "extrapolated"
        except (ValueError, SpeciesDataError) as exc:
            row[f"Vapor pressure ({pressure_unit})"] = "error"
            row["In range?"] = str(exc)
        rows.append(row)
    return pd.DataFrame(rows)


def _boiling_temperature_table(
    species_list: list[ChemicalSpecies],
    pressure: float,
    pressure_unit: str,
    temp_unit: str,
) -> pd.DataFrame:
    rows = []
    for sp in species_list:
        row = {
            "Species": sp.name,
            "Formula": sp.formula,
            "Molar mass (g/mol)": sp.molar_mass,
            "Valid range": _format_range(sp),
        }
        try:
            temperature = engine.boiling_temperature(
                sp.antoine,
                pressure,
                pressure_unit=pressure_unit,
                temp_unit=temp_unit,
            )
            row[f"Boiling temperature ({temp_unit})"] = round(temperature, 4)
        except (ValueError, SpeciesDataError) as exc:
            row[f"Boiling temperature ({temp_unit})"] = f"error: {exc}"
        rows.append(row)
    return pd.DataFrame(rows)


def _curve_chart(
    species_list: list[ChemicalSpecies],
    pressure_unit: str,
    log_scale: bool,
) -> None:
    """Render a vapor-pressure-vs-temperature line chart (x-axis in °C)."""
    lows, highs = [], []
    for sp in species_list:
        rng = sp.valid_range
        if rng is not None:
            lo_c = engine.convert_temperature(rng[0], sp.antoine.temperature_unit, "Celsius")
            hi_c = engine.convert_temperature(rng[1], sp.antoine.temperature_unit, "Celsius")
            lows.append(lo_c)
            highs.append(hi_c)
    t_lo = min(lows) if lows else 0.0
    t_hi = max(highs) if highs else 150.0
    if t_hi <= t_lo:
        t_hi = t_lo + 100.0

    t_axis = np.linspace(t_lo, t_hi, 200)
    data = {}
    for sp in species_list:
        pressures = engine.vapor_pressure_curve(
            sp.antoine, t_axis, temp_unit="Celsius", pressure_unit=pressure_unit
        )
        data[sp.name] = pressures

    y_label = f"Vapor pressure ({pressure_unit})"
    frame = pd.DataFrame(data, index=np.round(t_axis, 3))
    if log_scale:
        frame = np.log10(frame)
        y_label = f"log₁₀ vapor pressure ({pressure_unit})"
    frame.index.name = "Temperature (°C)"
    st.caption(y_label + "  vs.  Temperature (°C)")
    st.line_chart(frame)


def render() -> None:
    """Render the full Streamlit page."""
    st.set_page_config(page_title="Thermodynamic Property Calculator", page_icon="🧪")
    st.title("🧪 Thermodynamic Property Calculator")
    st.write(
        "Vapor pressure ⇄ boiling temperature via the **Antoine equation**, "
        "with unit conversion. Pure-Python engine (NumPy + SciPy); data is "
        "driven entirely by `chemical_data.json`."
    )

    try:
        registry = _load(str(DATA_PATH))
    except SpeciesDataError as exc:
        st.error(f"Failed to load chemical data: {exc}")
        st.stop()

    keys = list(registry.keys())
    defaults = [k for k in ("water", "benzene") if k in registry] or keys[:1]

    with st.sidebar:
        st.header("Inputs")
        chosen = st.multiselect(
            "Species",
            options=keys,
            default=defaults,
            format_func=lambda k: registry[k].name,
        )
        mode = st.radio("Calculation", [MODE_P_FROM_T, MODE_T_FROM_P])

        if mode == MODE_P_FROM_T:
            temperature = st.number_input("Temperature", value=100.0, step=1.0)
            temp_unit = st.selectbox("Temperature unit", SUPPORTED_TEMPERATURE_UNITS, index=0)
            pressure_unit = st.selectbox("Pressure unit (output)", SUPPORTED_PRESSURE_UNITS, index=0)
        else:
            pressure = st.number_input("Pressure", value=760.0, step=1.0, min_value=0.0)
            pressure_unit = st.selectbox("Pressure unit", SUPPORTED_PRESSURE_UNITS, index=0)
            temp_unit = st.selectbox("Temperature unit (output)", SUPPORTED_TEMPERATURE_UNITS, index=0)

        log_scale = st.checkbox("Log scale on chart", value=True)

    if not chosen:
        st.info("Pick one or more species in the sidebar to begin.")
        return

    species_list = _selected_species(registry, chosen)

    if mode == MODE_P_FROM_T:
        st.subheader(f"Vapor pressure at {temperature:g} {temp_unit}")
        table = _vapor_pressure_table(species_list, temperature, temp_unit, pressure_unit)
    else:
        st.subheader(f"Boiling temperature at {pressure:g} {pressure_unit}")
        table = _boiling_temperature_table(species_list, pressure, pressure_unit, temp_unit)

    st.dataframe(table, hide_index=True, width="stretch")

    st.subheader("Vapor pressure curve")
    _curve_chart(species_list, pressure_unit, log_scale)

    with st.expander("About the Antoine equation"):
        st.markdown(
            "log₁₀(P) = A − B / (T + C)\n\n"
            "Constants are unit-specific (the bundled data is mmHg / °C). The "
            "engine evaluates in each species' native units and converts your "
            "inputs and outputs to the units selected above. Results outside a "
            "species' validated temperature range are extrapolations."
        )
