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
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from . import diagrams
from . import distillation as dist
from . import flash_engine as flash
from . import models as tmodels
from . import molviz
from . import property_prediction as predict
from . import thermo_engine as engine
from . import validation
from .data_models import (
    SUPPORTED_PRESSURE_UNITS,
    SUPPORTED_TEMPERATURE_UNITS,
    AntoineConstants,
    ChemicalSpecies,
    SpeciesDataError,
    add_species,
    load_species,
    remove_species,
    slugify_key,
)

DATA_PATH = Path(__file__).resolve().parent.parent / "chemical_data.json"
CATALOG_PATH = Path(__file__).resolve().parent.parent / "antoine_catalog.json"

MODE_P_FROM_T = "Vapor pressure from temperature  (T → P)"
MODE_T_FROM_P = "Boiling temperature from pressure  (P → T)"

APP_MODE_DASHBOARD = "Dashboard"
APP_MODE_LOOKUP = "Property Lookup"
APP_MODE_FLASH = "VLE Flash Calculation"
APP_MODE_DIAGRAM = "Phase Diagram"
APP_MODE_VALIDATION = "Model Validation"
APP_MODE_DISTILL = "Distillation"
APP_MODE_PREDICT = "Property Prediction"
APP_MODE_MOLVIZ = "Molecular Viewer"

# Grouped navigation: (section, [(mode, glyph, short label)]).
NAV_GROUPS = [
    ("Workspace", [(APP_MODE_DASHBOARD, "▦", "Dashboard")]),
    ("Analysis", [
        (APP_MODE_LOOKUP, "≈", "Property Lookup"),
        (APP_MODE_DIAGRAM, "△", "Phase Diagram"),
        (APP_MODE_VALIDATION, "✓", "Model Validation"),
        (APP_MODE_DISTILL, "≣", "Distillation"),
    ]),
    ("Modeling", [
        (APP_MODE_FLASH, "◑", "VLE Flash"),
        (APP_MODE_PREDICT, "ƒ", "Property Prediction"),
        (APP_MODE_MOLVIZ, "⬡", "Molecular Viewer"),
    ]),
]

_BADGE_COLOR = {
    "Excellent": "#34D399", "Good": "#38BDF8", "Fair": "#F6A93B", "Poor": "#F87171",
}

# The phase triad — colours encode physical state, reused everywhere.
_LIQUID = "#38BDF8"
_TWO_PHASE = "#34D399"
_VAPOR = "#F6A93B"
_PHASE_COLOR = {
    flash.REGIME_SUBCOOLED: _LIQUID,
    flash.REGIME_TWO_PHASE: _TWO_PHASE,
    flash.REGIME_SUPERHEATED: _VAPOR,
}

_THEME_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
:root{
  --bg:#171b21; --panel:#242a32; --panel-head:#2b313b; --raised:#2e353f;
  --border:#363d47; --text:#d7dce3; --muted:#838c99; --grid:#363d47;
  --primary:#4a9eff; --liquid:#38BDF8; --two:#34D399; --vapor:#F6A93B;
}
/* Type */
html, body, .stApp, [data-testid="stAppViewContainer"]{
  font-family:'IBM Plex Sans', system-ui, sans-serif; color:var(--text); }
h1,h2,h3,h4,[data-testid="stHeading"]{
  font-family:'Space Grotesk','IBM Plex Sans',sans-serif !important; letter-spacing:-0.01em; }
[data-testid="stCaptionContainer"], .stCaption, small{
  color:var(--muted) !important; font-family:'IBM Plex Mono', monospace; letter-spacing:0.02em; }
/* Flat neutral workspace ground */
.stApp{ background:var(--bg); }
[data-testid="stHeader"]{ background:transparent; }
.block-container{ padding-top:1.1rem !important; max-width:1320px; }
/* Panels (st.container(border=True)) -> steamplot-style cards */
[data-testid="stVerticalBlockBorderWrapper"]{
  background:var(--panel); border:1px solid var(--border) !important; border-radius:10px; }
/* Section headings -> panel header strips */
[data-testid="stMain"] :is(h2,h3){
  background:var(--panel-head); border:1px solid var(--border);
  border-left:3px solid var(--primary); border-radius:8px;
  padding:0.42rem 0.8rem !important; margin:1.1rem 0 0.7rem !important; font-size:1.15rem !important; }
/* Hero (dashboard only) */
.tpc-hero{ margin:0.2rem 0 1.4rem; animation:tpcRise .5s ease-out both; }
.tpc-eyebrow{ font-family:'IBM Plex Mono', monospace; text-transform:uppercase;
  letter-spacing:0.34em; font-size:0.72rem; color:var(--muted); margin-bottom:0.5rem; }
.tpc-title{ font-family:'Space Grotesk',sans-serif; font-weight:600;
  font-size:2.2rem; line-height:1.05; margin:0; color:#f1f4f9; }
.tpc-sub{ color:var(--muted); font-size:1.0rem; margin:0.55rem 0 0.9rem; max-width:62ch; }
.tpc-sub b{ color:var(--text); font-weight:600; }
.tpc-spine{ height:3px; width:100%; border-radius:3px;
  background:linear-gradient(90deg,var(--liquid),var(--two) 52%,var(--vapor)); }
.tpc-legend{ display:flex; gap:1.4rem; margin-top:0.7rem;
  font-family:'IBM Plex Mono',monospace; font-size:0.76rem; color:var(--muted); }
.tpc-legend span{ display:inline-flex; align-items:center; gap:0.42rem; }
.tpc-legend i{ width:9px; height:9px; border-radius:50%; display:inline-block; }
.tpc-legend .ph-liq{ background:var(--liquid); }
.tpc-legend .ph-two{ background:var(--two); }
.tpc-legend .ph-vap{ background:var(--vapor); }
/* Regime card */
.tpc-regime{ display:flex; gap:0.95rem; align-items:center; margin:0.3rem 0 1.1rem;
  padding:0.9rem 1.1rem; border-radius:10px;
  border:1px solid color-mix(in srgb, var(--accent) 38%, var(--border));
  background:linear-gradient(90deg, color-mix(in srgb, var(--accent) 14%, var(--panel)), var(--panel) 60%);
  position:relative; overflow:hidden; }
.tpc-regime::before{ content:""; position:absolute; left:0; top:0; bottom:0; width:4px; background:var(--accent); }
.tpc-regime-dot{ width:13px; height:13px; border-radius:50%; background:var(--accent);
  box-shadow:0 0 0 5px color-mix(in srgb, var(--accent) 22%, transparent); flex:0 0 auto; }
.tpc-regime-label{ font-family:'Space Grotesk',sans-serif; font-weight:600; font-size:1.15rem; color:#f1f4f9; }
.tpc-regime-meta{ font-family:'IBM Plex Mono',monospace; font-size:0.82rem; color:var(--muted); margin-top:0.12rem; }
/* Metric cards -> readout tiles */
[data-testid="stMetric"]{ background:var(--raised); border:1px solid var(--border);
  border-radius:8px; padding:0.65rem 0.85rem 0.7rem; position:relative; overflow:hidden; }
[data-testid="stMetric"]::before{ content:""; position:absolute; left:0; top:0; bottom:0; width:2px;
  background:var(--primary); opacity:0.7; }
[data-testid="stMetricValue"]{ font-family:'IBM Plex Mono',monospace !important; font-weight:600;
  font-size:1.45rem !important; line-height:1.15; color:#eef2f7; }
[data-testid="stMetricLabel"]{ color:var(--muted) !important; }
[data-testid="stAppDeployButton"]{ display:none; }
/* Sidebar = left tool panel */
[data-testid="stSidebar"]{ background:#1d2229; border-right:1px solid var(--border); }
[data-testid="stSidebar"] [data-testid="stHeading"]{ background:transparent; border:none;
  padding:0 !important; font-size:0.74rem !important; text-transform:uppercase;
  letter-spacing:0.18em; color:var(--muted); }
/* Buttons */
.stButton>button{ border-radius:7px; border:1px solid var(--border); font-weight:600;
  background:var(--raised); transition:all .15s ease; }
.stButton>button:hover{ border-color:var(--primary); color:var(--primary); }
/* Tabs -> steamplot tab strip */
[data-baseweb="tab-list"]{ gap:0.2rem; border-bottom:1px solid var(--border); }
[data-baseweb="tab"]{ background:var(--bg); border:1px solid var(--border); border-bottom:none;
  border-radius:8px 8px 0 0; padding:0.35rem 0.85rem !important; color:var(--muted) !important; }
[data-baseweb="tab"][aria-selected="true"]{ background:var(--panel); color:var(--text) !important; }
[data-baseweb="tab-highlight"]{ background:var(--primary); }
/* Tables, inputs, dividers */
[data-testid="stDataFrame"]{ border:1px solid var(--border); border-radius:8px; }
hr{ border-color:var(--border); }
:focus-visible{ outline:2px solid var(--primary); outline-offset:2px; }
@keyframes tpcRise{ from{ opacity:0; transform:translateY(9px);} to{ opacity:1; transform:none;} }
@media (prefers-reduced-motion: reduce){ *{ animation:none !important; transition:none !important; } }
/* Status chips */
.tpc-chips{ display:flex; flex-wrap:wrap; gap:0.5rem; margin:0.2rem 0 0.4rem; }
.tpc-chip{ display:inline-flex; align-items:center; gap:0.4rem; padding:0.2rem 0.7rem;
  border-radius:999px; font-family:'IBM Plex Mono',monospace; font-size:0.76rem;
  color:var(--muted); background:var(--raised); border:1px solid var(--border); }
.tpc-chip b{ color:var(--text); font-weight:600; }
.tpc-chip i{ width:8px; height:8px; border-radius:50%; display:inline-block; background:var(--two); }
.tpc-section{ font-family:'IBM Plex Mono',monospace; text-transform:uppercase;
  letter-spacing:0.18em; font-size:0.74rem; color:var(--muted); margin:1.1rem 0 0.4rem; }
/* Label/value rows */
.tpc-kv{ display:flex; justify-content:space-between; gap:1rem; padding:0.5rem 0.1rem;
  border-bottom:1px solid var(--border); font-size:0.92rem; }
.tpc-kv .k{ color:var(--muted); } .tpc-kv .v{ font-family:'IBM Plex Mono',monospace; color:var(--text); }
/* Minimal top command bar */
.tpc-cmdbar{ position:sticky; top:0; z-index:50; display:flex; flex-wrap:wrap;
  justify-content:space-between; align-items:center; gap:0.6rem 1rem;
  padding:0.45rem 0.2rem 0.5rem; margin:0 0 1rem; background:var(--bg);
  border-bottom:1px solid var(--border); }
.tpc-cmd-left{ display:flex; align-items:center; gap:0.6rem; }
.tpc-logo{ font-family:'Space Grotesk',sans-serif; font-weight:700; color:#f1f4f9;
  letter-spacing:0.02em; background:var(--raised); border:1px solid var(--border);
  padding:0.15rem 0.55rem; border-radius:7px; }
.tpc-sep{ color:var(--border); }
.tpc-module{ font-family:'IBM Plex Mono',monospace; color:var(--text); font-size:0.82rem;
  text-transform:uppercase; letter-spacing:0.12em; }
.tpc-cmd-right{ display:flex; flex-wrap:wrap; align-items:center; gap:0.9rem;
  font-family:'IBM Plex Mono',monospace; font-size:0.74rem; color:var(--muted); }
.tpc-cmd-right b{ color:var(--text); font-weight:600; }
.tpc-online{ display:inline-flex; align-items:center; gap:0.4rem; color:var(--two); }
.tpc-online i{ width:8px; height:8px; border-radius:50%; background:var(--two);
  box-shadow:0 0 0 3px color-mix(in srgb, var(--two) 25%, transparent); }
/* Footer status bar */
.tpc-statusbar{ display:flex; flex-wrap:wrap; justify-content:space-between; gap:0.8rem;
  margin-top:1.8rem; padding:0.55rem 0.2rem; border-top:1px solid var(--border);
  font-family:'IBM Plex Mono',monospace; font-size:0.72rem; color:var(--muted); }
.tpc-statusbar i{ width:8px; height:8px; border-radius:50%; display:inline-block; margin:0 0.3rem; }
.tpc-statusbar .ph-liq{ background:var(--liquid); }
.tpc-statusbar .ph-two{ background:var(--two); }
.tpc-statusbar .ph-vap{ background:var(--vapor); }
.tpc-ok i{ background:var(--two); margin-left:0; }
/* --- Grouped scientific navigation --- */
.nav-brand{ font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:1.02rem;
  color:#f1f4f9; letter-spacing:0.06em; padding:0.1rem 0.2rem 0.6rem; }
.nav-brand small{ display:block; font-family:'IBM Plex Mono',monospace; font-weight:400;
  font-size:0.62rem; letter-spacing:0.22em; color:var(--muted); margin-top:0.15rem; }
.nav-group{ font-family:'IBM Plex Mono',monospace; text-transform:uppercase;
  letter-spacing:0.2em; font-size:0.66rem; color:var(--muted); margin:0.9rem 0.2rem 0.25rem; }
[data-testid="stSidebar"] .stButton>button{ justify-content:flex-start; text-align:left;
  border:1px solid transparent; background:transparent; color:var(--muted);
  font-weight:500; padding:0.34rem 0.6rem; border-radius:7px; box-shadow:none; }
[data-testid="stSidebar"] .stButton>button:hover{ background:var(--raised); color:var(--text);
  border-color:transparent; transform:none; }
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"]{
  background:color-mix(in srgb, var(--primary) 16%, transparent) !important;
  color:#eef2f7 !important; border:1px solid color-mix(in srgb, var(--primary) 45%, transparent) !important;
  border-left:3px solid var(--primary) !important; font-weight:600; }
/* --- Workstation panels & research metadata --- */
.tpc-ptitle{ display:flex; align-items:center; justify-content:space-between; gap:0.6rem;
  font-family:'Space Grotesk',sans-serif; font-weight:600; font-size:0.96rem; color:#eef2f7;
  padding:0.45rem 0.7rem; background:var(--panel-head); border:1px solid var(--border);
  border-left:3px solid var(--primary); border-radius:8px; margin:0.2rem 0 0.6rem; }
.tpc-ptitle .tag{ font-family:'IBM Plex Mono',monospace; font-size:0.66rem; font-weight:500;
  text-transform:uppercase; letter-spacing:0.12em; color:var(--muted); }
.tpc-note{ font-size:0.82rem; color:var(--muted); line-height:1.5; }
.tpc-note b{ color:var(--text); }
.tpc-meta{ display:grid; grid-template-columns:auto 1fr; gap:0.3rem 0.9rem; font-size:0.84rem; }
.tpc-meta .k{ color:var(--muted); font-family:'IBM Plex Mono',monospace; }
.tpc-meta .v{ color:var(--text); }
/* KPI strip in the command bar */
.tpc-kpis{ display:flex; flex-wrap:wrap; gap:1.1rem; font-family:'IBM Plex Mono',monospace;
  font-size:0.74rem; color:var(--muted); }
.tpc-kpis .kpi b{ color:#eef2f7; font-size:0.92rem; font-weight:600; }
.tpc-kpis .kpi span{ display:block; font-size:0.6rem; text-transform:uppercase; letter-spacing:0.12em; }
"""

# Extra overrides injected only when the user picks the Compact density.
_COMPACT_CSS = """
.block-container{ padding-top:0.5rem !important; }
html, body, .stApp, [data-testid="stAppViewContainer"]{ font-size:0.92rem; }
[data-testid="stMetric"]{ padding:0.5rem 0.7rem 0.6rem; }
[data-testid="stMetricValue"]{ font-size:1.3rem !important; }
[data-testid="stVerticalBlock"]{ gap:0.55rem; }
.tpc-cmdbar{ padding:0.35rem 0.8rem; margin-bottom:0.7rem; }
"""


@st.cache_data(show_spinner=False)
def _load(path_str: str) -> dict[str, ChemicalSpecies]:
    return load_species(path_str)


@st.cache_data(show_spinner=False)
def _load_catalog(path_str: str) -> dict[str, ChemicalSpecies]:
    """Load the read-only reference catalog; empty dict if it's absent."""
    if not Path(path_str).exists():
        return {}
    return load_species(path_str)


def _unique_key(name: str, taken: set[str]) -> str:
    """A slug key for ``name`` that does not collide with ``taken``."""
    base = slugify_key(name)
    key = base
    n = 2
    while key in taken:
        key = f"{base}_{n}"
        n += 1
    return key


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


def _add_from_catalog_ui(registry: dict[str, ChemicalSpecies]) -> None:
    """Search the reference catalog and add a species to the active data file."""
    catalog = _load_catalog(str(CATALOG_PATH))
    if not catalog:
        st.caption("Reference catalog not found — use the Manual tab to add a species.")
        return

    existing_names = {sp.name.lower() for sp in registry.values()}
    options = sorted(
        (k for k, sp in catalog.items() if sp.name.lower() not in existing_names),
        key=lambda k: catalog[k].name.lower(),
    )
    st.caption(f"{len(catalog)} validated compounds in the reference catalog.")
    if not options:
        st.caption("Every catalog species is already in your list.")
        return

    sel = st.selectbox(
        "Search the catalog",
        options=options,
        format_func=lambda k: f"{catalog[k].name}  ({catalog[k].formula})",
        index=None,
        placeholder="Type a name or formula…",
        key="catalog_select",
    )
    if sel and st.button("Add to my species", key="add_catalog", width="stretch"):
        src = catalog[sel]
        key = _unique_key(src.name, set(_load(str(DATA_PATH)).keys()))
        species = ChemicalSpecies(
            key=key,
            name=src.name,
            formula=src.formula,
            molar_mass=src.molar_mass,
            antoine=src.antoine,
        )
        try:
            add_species(DATA_PATH, species)
            _load.clear()
            st.success(f"Added {src.name}.")
            st.rerun()
        except SpeciesDataError as exc:
            st.error(str(exc))


def _manual_add_ui(registry: dict[str, ChemicalSpecies]) -> None:
    """Add an arbitrary species by entering its Antoine constants."""
    st.caption("Add any species by entering its Antoine constants (P = 10^(A − B/(T+C))).")
    with st.form("manual_add", clear_on_submit=False):
        name = st.text_input("Name", placeholder="e.g. Chloroform")
        formula = st.text_input("Formula", placeholder="e.g. CHCl3")
        molar_mass = st.number_input("Molar mass (g/mol)", min_value=0.0, value=None)
        c1, c2, c3 = st.columns(3)
        A = c1.number_input("A", value=None, format="%.5f")
        B = c2.number_input("B", value=None, format="%.4f")
        C = c3.number_input("C", value=None, format="%.4f")
        u1, u2 = st.columns(2)
        p_unit = u1.selectbox("Pressure unit", SUPPORTED_PRESSURE_UNITS, index=0)
        t_unit = u2.selectbox("Temperature unit", SUPPORTED_TEMPERATURE_UNITS, index=0)
        r1, r2 = st.columns(2)
        t_min = r1.number_input("T min (optional)", value=None)
        t_max = r2.number_input("T max (optional)", value=None)
        known_bp = st.number_input(
            "Known boiling point at 1 atm (optional)",
            value=None,
            help="In the selected temperature unit. If given, the constants are "
            "checked against it.",
        )
        submitted = st.form_submit_button("Validate & add", width="stretch")

    if not submitted:
        return
    if not name or not formula or molar_mass is None or None in (A, B, C):
        st.error("Provide at least a name, formula, molar mass, and the A, B, C constants.")
        return

    try:
        antoine = AntoineConstants(
            A=float(A),
            B=float(B),
            C=float(C),
            pressure_unit=p_unit,
            temperature_unit=t_unit,
            t_min=t_min,
            t_max=t_max,
        )
        pred_bp = engine.boiling_temperature(antoine, 1.0, pressure_unit="atm", temp_unit=t_unit)
        bp_msg = f"Predicted normal boiling point: {pred_bp:.2f} {t_unit}."
        if known_bp is not None and abs(pred_bp - known_bp) > 5.0:
            st.warning(
                f"{bp_msg} That differs from your stated {known_bp:g} {t_unit} by "
                f"{abs(pred_bp - known_bp):.1f}° — double-check the constants and units."
            )
        else:
            st.info(bp_msg)

        key = _unique_key(name, set(_load(str(DATA_PATH)).keys()))
        species = ChemicalSpecies(
            key=key,
            name=name.strip(),
            formula=formula.strip(),
            molar_mass=float(molar_mass),
            antoine=antoine,
        )
        add_species(DATA_PATH, species)
        _load.clear()
        st.success(f"Added {name}.")
        st.rerun()
    except (ValueError, SpeciesDataError) as exc:
        st.error(f"Could not add species: {exc}")


def _manage_ui(registry: dict[str, ChemicalSpecies]) -> None:
    """Remove species from the active data file."""
    keys = list(registry.keys())
    to_remove = st.multiselect(
        "Remove species",
        options=keys,
        format_func=lambda k: registry[k].name,
        key="remove_sel",
    )
    if to_remove and st.button("Remove selected", key="remove_btn", width="stretch"):
        for k in to_remove:
            remove_species(DATA_PATH, k)
        _load.clear()
        st.success(f"Removed {len(to_remove)} species.")
        st.rerun()


def _inject_theme() -> None:
    """Inject the console theme (fonts + phase-coded styling) once per render."""
    st.markdown(f"<style>{_THEME_CSS}</style>", unsafe_allow_html=True)


def _hero(registry: dict[str, ChemicalSpecies]) -> None:
    st.markdown(
        f"""
<div class="tpc-hero">
  <div class="tpc-eyebrow">Vapor–Liquid Equilibrium Toolkit</div>
  <h1 class="tpc-title">Thermodynamic Property Calculator</h1>
  <p class="tpc-sub">Look up Antoine vapor pressures and boiling points, or run an
  isothermal Rachford–Rice flash — across <b>{len(registry)}</b> validated species.</p>
  <div class="tpc-spine"></div>
  <div class="tpc-legend">
    <span><i class="ph-liq"></i>liquid</span>
    <span><i class="ph-two"></i>two-phase</span>
    <span><i class="ph-vap"></i>vapor</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def _molecular_viewer_mode(registry: dict[str, ChemicalSpecies]) -> None:
    """Mode 7 — 2D (RDKit) + interactive 3D (3Dmol.js) molecular visualization."""
    with st.sidebar:
        st.subheader("Molecule")
        mv_names = list(molviz.MOLECULE_SMILES)
        active_names = [registry[k].name for k in _active_keys(registry)]
        mv_default = next((n for n in active_names if n in molviz.MOLECULE_SMILES), "Toluene")
        pick = st.selectbox("Pick a molecule", mv_names,
                            index=mv_names.index(mv_default) if mv_default in mv_names else 0,
                            key="mv_pick")
        smiles = st.text_input("…or enter SMILES", value=molviz.MOLECULE_SMILES[pick],
                               key="mv_smiles")
        style = st.selectbox("3D style", ["Ball & stick", "Stick", "Space-filling"],
                             key="mv_style")
        show_labels = st.checkbox("Show atom labels", value=False, key="mv_labels")

    try:
        info = molviz.analyze(smiles)
    except molviz.MolVizError as exc:
        st.error(str(exc))
        return
    _record_calc("molviz")

    is_library = molviz.MOLECULE_SMILES.get(pick) == smiles
    name = pick if is_library else "Custom molecule"
    _panel_title(f"Molecular workspace — {name}", tag=info.formula)

    viz, inspect = st.columns([1.7, 1], gap="large")
    with viz:
        _panel_title("Interactive 3D structure",
                     tag="3Dmol.js · drag to rotate · scroll to zoom")
        components.html(molviz.viewer_html(info.molblock_3d, show_labels, style, height=540),
                        height=556)
        _panel_title("2D structure", tag="RDKit depiction")
        components.html(f'<div style="background:transparent">{info.svg_2d}</div>', height=290)

    with inspect:
        with st.container(border=True):
            _panel_title("Molecular identity")
            _meta_table([
                ("Name", name), ("Formula", info.formula), ("SMILES", info.smiles),
                ("Molar mass", f"{info.molar_mass:.3f} g/mol"),
                ("Heavy atoms", str(info.heavy_atoms)),
                ("Total atoms (+H)", str(info.total_atoms)),
                ("Ring count", str(info.rings)),
            ])
        with st.container(border=True):
            _panel_title("Atomic composition")
            _meta_table([(el, str(n)) for el, n in info.atom_counts.items()])

        mol = predict.LIBRARY_BY_NAME.get(pick) if is_library else None
        if mol is not None:
            est = predict.estimate_molecule(mol)
            with st.container(border=True):
                _panel_title("Predicted properties", tag="Joback + Lee–Kesler")
                _meta_table([
                    ("Boiling point Tb", f"{est.Tb - 273.15:.1f} °C  ({est.Tb:.1f} K)"),
                    ("Critical temp Tc", f"{est.Tc:.1f} K"),
                    ("Critical pressure Pc", f"{est.Pc:.2f} bar"),
                    ("Critical volume Vc", f"{est.Vc:.1f} cm³/mol"),
                    ("Acentric factor ω", f"{est.omega:.3f}"),
                ])
                if st.button("Open in Property Prediction", key="mv_to_predict",
                             width="stretch", on_click=_goto_mode, args=(APP_MODE_PREDICT,)):
                    pass
        else:
            with st.container(border=True):
                _panel_title("Predicted properties")
                st.caption("Property estimates are available for reference-library "
                           "molecules (Joback group decomposition required). Pick a "
                           "library molecule to see Tb / Tc / Pc / ω.")


_PUB_FONT = "IBM Plex Sans, system-ui, sans-serif"
_PUB_MONO = "IBM Plex Mono, monospace"


def _sidebar_nav() -> None:
    """Grouped scientific navigation with a strong active state."""
    current = st.session_state.get("app_mode", APP_MODE_DASHBOARD)
    for group, items in NAV_GROUPS:
        st.markdown(f'<div class="nav-group">{group}</div>', unsafe_allow_html=True)
        for mode, glyph, label in items:
            st.button(f"{glyph}  {label}", key=f"nav_{mode}", width="stretch",
                      on_click=_goto_mode, args=(mode,),
                      type="primary" if mode == current else "tertiary")


def _panel_title(title: str, tag: str = "") -> None:
    """Render a workstation-style panel header bar."""
    tag_html = f'<span class="tag">{tag}</span>' if tag else ""
    st.markdown(f'<div class="tpc-ptitle"><span>{title}</span>{tag_html}</div>',
                unsafe_allow_html=True)


def _pub_layout(fig, *, title: str = "", xtitle: str = "", ytitle: str = "",
                height: int = 560, legend: str = "bottom", x_range=None, y_range=None,
                square: bool = False, log_y: bool = False):
    """Apply a consistent publication-quality layout to a Plotly figure."""
    fig.update_layout(
        template=None, height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#222831",
        font=dict(family=_PUB_FONT, color="#c7cdd6", size=13),
        title=dict(text=title, font=dict(family="Space Grotesk, sans-serif", size=15,
                   color="#eef2f7"), x=0.01, xanchor="left", y=0.98),
        margin=dict(l=74, r=28, t=44 if title else 16,
                    b=86 if legend == "bottom" else 56),
        hoverlabel=dict(font_family=_PUB_MONO, bgcolor="#1d2229", bordercolor="#363d47"),
    )
    if legend == "bottom":
        fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.16, x=0.5,
            xanchor="center", font=dict(family=_PUB_MONO, size=11), bgcolor="rgba(0,0,0,0)"))
    elif legend == "topright":
        fig.update_layout(legend=dict(orientation="v", yanchor="top", y=0.99, x=0.985,
            xanchor="right", font=dict(family=_PUB_MONO, size=11),
            bgcolor="rgba(29,34,41,0.88)", bordercolor="#363d47", borderwidth=1))
    elif legend == "none":
        fig.update_layout(showlegend=False)
    axis = dict(gridcolor="#333b45", griddash="dot", zeroline=False, showline=True,
                linecolor="#4a525e", linewidth=1, ticks="outside", tickcolor="#4a525e",
                ticklen=5, tickfont=dict(family=_PUB_MONO, size=11), mirror=True,
                minor=dict(showgrid=True, gridcolor="#272d36", griddash="dot"))
    fig.update_xaxes(title=dict(text=xtitle, font=dict(size=13)), range=x_range, **axis)
    y_axis = dict(axis)
    if log_y:
        y_axis.update(type="log", exponentformat="power")
    fig.update_yaxes(title=dict(text=ytitle, font=dict(size=13)), range=y_range, **y_axis)
    if square:
        fig.update_xaxes(scaleanchor="y", scaleratio=1)
    return fig


def _command_bar(app_mode: str, registry: dict[str, ChemicalSpecies]) -> None:
    """Top status bar: platform, active module, and live research KPIs."""
    n_calc = st.session_state.get("sim_count", 0)
    st.markdown(
        f"""
<div class="tpc-cmdbar">
  <div class="tpc-cmd-left">
    <span class="tpc-logo">⬡ VLE&nbsp;PLATFORM</span>
    <span class="tpc-sep">/</span>
    <span class="tpc-module">{app_mode}</span>
  </div>
  <div class="tpc-kpis">
    <span class="kpi"><b>{len(registry)}</b><span>species</span></span>
    <span class="kpi"><b>{len(tmodels.MODEL_NAMES)}</b><span>models</span></span>
    <span class="kpi"><b>{len(validation.DATASETS)}</b><span>datasets</span></span>
    <span class="kpi"><b>{len(predict.LIBRARY)}</b><span>ref mols</span></span>
    <span class="kpi"><b>{n_calc}</b><span>runs</span></span>
    <span class="tpc-online"><i></i>engine ready</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def _status_bar() -> None:
    """Footer status bar reinforcing the global phase-colour key."""
    st.markdown(
        """
<div class="tpc-statusbar">
  <span class="tpc-ok"><i></i>Engine operational</span>
  <span>NumPy · SciPy · Plotly</span>
  <span>Phase key:<i class="ph-liq"></i>liquid<i class="ph-two"></i>two-phase<i class="ph-vap"></i>vapor</span>
  <span>VLE Console · 7 tools</span>
</div>
""",
        unsafe_allow_html=True,
    )


def _regime_card(result: "flash.FlashResult") -> None:
    color = _PHASE_COLOR[result.regime]
    st.markdown(
        f"""
<div class="tpc-regime" style="--accent:{color}">
  <span class="tpc-regime-dot"></span>
  <div>
    <div class="tpc-regime-label">{result.regime_label}</div>
    <div class="tpc-regime-meta">β = {result.vapor_fraction:.4f} · vapor / feed</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def _init_active_species(registry: dict[str, ChemicalSpecies]) -> None:
    """Seed/sanitise the global active-species selection (shared across pages)."""
    keys = list(registry.keys())
    if "active_species" not in st.session_state:
        default = [k for k in ("benzene", "toluene") if k in registry] or keys[: min(2, len(keys))]
        st.session_state["active_species"] = default
    else:
        st.session_state["active_species"] = [
            k for k in st.session_state["active_species"] if k in registry
        ]


def _active_keys(registry: dict[str, ChemicalSpecies]) -> list[str]:
    return [k for k in st.session_state.get("active_species", []) if k in registry]


def _active_species_list(registry: dict[str, ChemicalSpecies]) -> list[ChemicalSpecies]:
    return [registry[k] for k in _active_keys(registry)]


def _property_lookup_mode(registry: dict[str, ChemicalSpecies]) -> None:
    """Mode 1 — single-species property lookup (vapor pressure ⇄ boiling T)."""
    chosen = _active_keys(registry)

    with st.sidebar:
        st.subheader("Property inputs")
        mode = st.radio("Calculation", [MODE_P_FROM_T, MODE_T_FROM_P], key="lookup_calc")

        if mode == MODE_P_FROM_T:
            temperature = st.number_input("Temperature", value=100.0, step=1.0, key="lookup_T")
            temp_unit = st.selectbox(
                "Temperature unit", SUPPORTED_TEMPERATURE_UNITS, index=0, key="lookup_Tunit"
            )
            pressure_unit = st.selectbox(
                "Pressure unit (output)", SUPPORTED_PRESSURE_UNITS, index=0, key="lookup_Pout"
            )
        else:
            pressure = st.number_input(
                "Pressure", value=760.0, step=1.0, min_value=0.0, key="lookup_P"
            )
            pressure_unit = st.selectbox(
                "Pressure unit", SUPPORTED_PRESSURE_UNITS, index=0, key="lookup_Punit"
            )
            temp_unit = st.selectbox(
                "Temperature unit (output)", SUPPORTED_TEMPERATURE_UNITS, index=0, key="lookup_Tout"
            )

        log_scale = st.checkbox("Log scale on chart", value=True, key="lookup_log")

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


def _flash_mode(registry: dict[str, ChemicalSpecies]) -> None:
    """Mode 2 — isothermal VLE flash (Rachford-Rice, ideal/Raoult)."""
    chosen = _active_keys(registry)

    with st.sidebar:
        st.subheader("Flash inputs")
        st.caption("Components = the active species (set in the sidebar above).")
        model_name = st.selectbox(
            "Thermodynamic model", tmodels.MODEL_NAMES, index=0, key="flash_model"
        )
        c1, c2 = st.columns(2)
        temperature = c1.number_input("Temperature", value=95.0, step=1.0, key="flash_T")
        temp_unit = c2.selectbox(
            "T unit", SUPPORTED_TEMPERATURE_UNITS, index=0, key="flash_Tunit"
        )
        c3, c4 = st.columns(2)
        pressure = c3.number_input(
            "Pressure", value=760.0, step=1.0, min_value=0.0, key="flash_P"
        )
        pressure_unit = c4.selectbox(
            "P unit", SUPPORTED_PRESSURE_UNITS, index=0, key="flash_Punit"
        )

        st.caption("Feed mole fractions zᵢ (normalised automatically):")
        default_z = round(1.0 / len(chosen), 4) if chosen else 0.0
        z_inputs = [
            st.number_input(
                f"z — {registry[k].name}",
                min_value=0.0,
                max_value=1.0,
                value=default_z,
                step=0.05,
                key=f"flash_z_{k}",
            )
            for k in chosen
        ]

    if len(chosen) < 2:
        st.info("Select at least two components in the sidebar to run a flash.")
        return
    if sum(z_inputs) <= 0:
        st.error("Mole fractions must sum to a positive value.")
        return

    species_list = _selected_species(registry, chosen)

    # Route through the activity-coefficient framework. Non-ideal models need a
    # binary pair with fitted parameters; otherwise fall back to ideal Raoult.
    notice = None
    use_model = model_name
    if model_name != tmodels.IdealModel.name:
        if len(species_list) != 2:
            notice = f"{model_name} supports binary mixtures only — using Ideal (Raoult)."
            use_model = tmodels.IdealModel.name
        elif not tmodels.has_parameters(model_name, species_list[0], species_list[1]):
            notice = (
                f"No {model_name} parameters for this pair — using Ideal (Raoult)."
            )
            use_model = tmodels.IdealModel.name

    try:
        if use_model == tmodels.IdealModel.name:
            result = flash.flash(species_list, z_inputs, temperature, pressure,
                                 temp_unit=temp_unit, pressure_unit=pressure_unit)
            provenance = "Raoult's law (ideal liquid; γ = 1)."
        else:
            model = tmodels.build_model(use_model, species_list[0], species_list[1])
            provenance = model.provenance
            result = model.flash(z_inputs, temperature, pressure,
                                 temp_unit=temp_unit, pressure_unit=pressure_unit)
    except (flash.FlashError, tmodels.NoParametersError, ValueError) as exc:
        st.error(f"Flash failed: {exc}")
        return

    if notice:
        st.warning(notice)
    _record_calc("flash")
    _render_flash_result(registry, chosen, z_inputs, result, provenance)


def _render_flash_result(
    registry: dict[str, ChemicalSpecies],
    chosen: list[str],
    z_inputs: list[float],
    result: "flash.FlashResult",
    provenance: str = "",
) -> None:
    names = [registry[k].name for k in chosen]
    st.subheader(
        f"Flash at {result.temperature:g} {result.temp_unit} "
        f"and {result.pressure:g} {result.pressure_unit}"
    )
    _regime_card(result)
    st.caption(f"Model: {result.model_name} — {provenance}")

    m1, m2, m3 = st.columns(3)
    m1.metric("Vapor fraction  β = V/F", f"{result.vapor_fraction:.4f}")
    m2.metric("Liquid fraction  L/F", f"{result.liquid_fraction:.4f}")
    m3.metric(
        f"Bubble / Dew P ({result.pressure_unit})",
        f"{result.bubble_pressure:,.1f} / {result.dew_pressure:,.1f}",
    )

    entered_sum = sum(z_inputs)
    if abs(entered_sum - 1.0) > 1e-6:
        st.caption(f"Entered zᵢ summed to {entered_sum:.3f}; normalised to 1.")

    out_of_range = [
        registry[k].name
        for k in chosen
        if not engine.temperature_in_range(
            registry[k].antoine,
            engine.convert_temperature(
                result.temperature, result.temp_unit, registry[k].antoine.temperature_unit
            ),
        )
    ]
    if out_of_range:
        st.warning(
            "Temperature is outside the validated Antoine range for: "
            + ", ".join(out_of_range)
            + " — saturation pressures are extrapolated."
        )

    columns = {
        "Component": names,
        "zᵢ (feed)": np.round(result.z, 4),
        f"Pˢᵃᵗ ({result.pressure_unit})": np.round(result.psat, 3),
    }
    if result.gamma is not None:
        columns["γᵢ (activity)"] = np.round(result.gamma, 4)
    columns["Kᵢ = γᵢPˢᵃᵗ/P"] = np.round(result.K, 4)
    columns["xᵢ (liquid)"] = np.round(result.x, 4)
    columns["yᵢ (vapor)"] = np.round(result.y, 4)
    st.dataframe(pd.DataFrame(columns), hide_index=True, width="stretch")

    phase_colors = {"zᵢ feed": "#93A4C2", "xᵢ liquid": _LIQUID, "yᵢ vapor": _VAPOR}
    chart_df = pd.DataFrame(
        {"zᵢ feed": result.z, "xᵢ liquid": result.x, "yᵢ vapor": result.y}, index=names
    ).dropna(axis=1, how="all")
    st.caption("Phase compositions — feed vs. equilibrium liquid and vapor")
    st.bar_chart(
        chart_df,
        color=[phase_colors[c] for c in chart_df.columns],
        stack=False,
    )

    with st.expander("About the VLE flash"):
        st.markdown(
            "Isothermal flash using **Raoult's law** (ideal VLE): "
            "Kᵢ = Pˢᵃᵗᵢ(T) / P, with Pˢᵃᵗ from the Antoine engine. The vapor "
            "fraction β solves the **Rachford-Rice** equation "
            "Σᵢ zᵢ(Kᵢ − 1) / (1 + β(Kᵢ − 1)) = 0 via SciPy's `brentq`.\n\n"
            "- **Subcooled liquid** when P ≥ bubble pressure (β = 0)\n"
            "- **Two-phase** when dew < P < bubble (0 < β < 1), with "
            "xᵢ = zᵢ / (1 + β(Kᵢ − 1)) and yᵢ = Kᵢxᵢ\n"
            "- **Superheated vapor** when P ≤ dew pressure (β = 1)\n\n"
            "Saturation pressures are retrieved from the shared `thermo_engine`, "
            "so components may even use different native units."
        )


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _plotly_config(image_format: str, filename: str) -> dict:
    """Modebar config: zoom/pan on, image button exports PNG or SVG client-side."""
    return {
        "displaylogo": False,
        "scrollZoom": True,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        "toImageButtonOptions": {
            "format": image_format,
            "filename": filename,
            "scale": 3 if image_format == "png" else 1,
        },
    }


def _find_azeotrope(x1, y1, yv):
    """Return (x_az, y_value) for an interior azeotrope (y₁≈x₁), else None."""
    if len(x1) < 5:
        return None
    d = np.abs(y1 - x1)[1:-1]
    i = int(np.argmin(d)) + 1
    if d[i - 1] < 3e-3 and 0.02 < x1[i] < 0.98:
        return float(x1[i]), float(yv[i])
    return None


def _phase_diagram_mode(registry: dict[str, ChemicalSpecies]) -> None:
    """Mode 3 — binary T–x–y / P–x–y phase diagrams from any activity model."""
    active = _active_keys(registry)

    with st.sidebar:
        st.subheader("Phase diagram")
        if len(active) >= 2:
            st.caption(f"Binary: **{registry[active[0]].name}** (1) / "
                       f"**{registry[active[1]].name}** (2) — first two active species.")
        model_name = st.selectbox("Thermodynamic model", tmodels.MODEL_NAMES,
                                  index=tmodels.MODEL_NAMES.index(tmodels.NRTLModel.name),
                                  key="dia_model")
        is_txy = st.radio("Diagram", ["T–x–y (isobaric)", "P–x–y (isothermal)"],
                          key="dia_type").startswith("T")
        if is_txy:
            cval = st.number_input("Pressure", value=760.0, min_value=1.0, step=10.0, key="dia_P")
            cunit = st.selectbox("Pressure unit", SUPPORTED_PRESSURE_UNITS, 0, key="dia_Pu")
            tunit = st.selectbox("Temperature unit", SUPPORTED_TEMPERATURE_UNITS, 0, key="dia_Tu")
        else:
            cval = st.number_input("Temperature", value=78.0, step=1.0, key="dia_T")
            tunit = st.selectbox("Temperature unit", SUPPORTED_TEMPERATURE_UNITS, 0, key="dia_Tu2")
            cunit = st.selectbox("Pressure unit", SUPPORTED_PRESSURE_UNITS, 0, key="dia_Pu2")
        npts = st.slider("Resolution (points)", 41, 201, 121, step=20, key="dia_n")
        img_fmt = st.selectbox("Image export format", ["png", "svg"], 0, key="dia_img")

    if len(active) < 2:
        st.info("Select at least two active species in the sidebar to build a diagram.")
        return

    k1, k2 = active[0], active[1]
    sp1, sp2 = registry[k1], registry[k2]
    notice, use_model = None, model_name
    if model_name != tmodels.IdealModel.name and not tmodels.has_parameters(model_name, sp1, sp2):
        notice = (f"No {model_name} parameters for {sp1.name}/{sp2.name} — "
                  f"showing Ideal (Raoult). Add parameters in thermo/models/parameters.py.")
        use_model = tmodels.IdealModel.name

    try:
        model = tmodels.build_model(use_model, sp1, sp2)
        if is_txy:
            df = diagrams.txy(model, cval, pressure_unit=cunit, temp_unit=tunit, n=npts)
        else:
            df = diagrams.pxy(model, cval, temp_unit=tunit, pressure_unit=cunit, n=npts)
    except (tmodels.NoParametersError, ValueError) as exc:
        st.error(f"Could not build diagram: {exc}")
        return

    if notice:
        st.warning(notice)
    _record_calc("diagram")
    _render_phase_diagram(df, sp1, sp2, model, use_model, img_fmt,
                          is_txy, cval, cunit, tunit)


def _meta_table(rows: list[tuple[str, str]]) -> None:
    body = "".join(f'<span class="k">{k}</span><span class="v">{v}</span>' for k, v in rows)
    st.markdown(f'<div class="tpc-meta">{body}</div>', unsafe_allow_html=True)


def _render_phase_diagram(df, sp1, sp2, model, model_name, img_fmt,
                          is_txy, cval, cunit, tunit) -> None:
    LIQ, VAP, TWO = _LIQUID, _VAPOR, _TWO_PHASE
    ykey = "T" if is_txy else "P"
    yunit = tunit if is_txy else cunit
    ylabel = f"Temperature ({yunit})" if is_txy else f"Pressure ({yunit})"
    cond = f"{cval:g} {cunit}" if is_txy else f"{cval:g} {tunit}"
    kind = "T–x–y" if is_txy else "P–x–y"
    x1, y1, yv = df["x1"].to_numpy(), df["y1"].to_numpy(), df[ykey].to_numpy()
    az = _find_azeotrope(x1, y1, yv)

    _panel_title(f"{kind} phase diagram — {sp1.name} (1) / {sp2.name} (2)",
                 tag=f"{model_name} · {cond}")

    main, side = st.columns([3, 1.2], gap="large")
    with main:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=np.concatenate([x1, y1[::-1]]), y=np.concatenate([yv, yv[::-1]]),
            fill="toself", fillcolor=_rgba(TWO, 0.10), line=dict(width=0),
            hoverinfo="skip", name="Two-phase region"))
        fig.add_trace(go.Scatter(x=x1, y=yv, mode="lines", name="Bubble (liquid)",
                                 line=dict(color=LIQ, width=3.2)))
        fig.add_trace(go.Scatter(x=y1, y=yv, mode="lines", name="Dew (vapor)",
                                 line=dict(color=VAP, width=3.2)))
        fig.add_trace(go.Scatter(
            x=[0.0, 1.0], y=[yv[0], yv[-1]], mode="markers", name="Pure components",
            marker=dict(color="#cfd6df", size=8, symbol="circle-open", line=dict(width=2)),
            hoverinfo="skip"))
        if az:
            fig.add_trace(go.Scatter(
                x=[az[0]], y=[az[1]], mode="markers", name="Azeotrope",
                marker=dict(color=TWO, size=13, symbol="diamond",
                            line=dict(color="#eef2f7", width=1.4))))
            fig.add_annotation(x=az[0], y=az[1], text=f"azeotrope  x₁={az[0]:.3f}",
                               showarrow=True, arrowhead=2, arrowcolor=TWO, ax=36, ay=-36,
                               font=dict(family=_PUB_MONO, size=10, color="#eef2f7"),
                               bgcolor="rgba(29,34,41,0.92)", bordercolor=TWO, borderwidth=1)
        _pub_layout(fig, xtitle=f"x₁, y₁  —  mole fraction {sp1.name}", ytitle=ylabel,
                    height=640, legend="bottom", x_range=[0, 1])
        st.plotly_chart(fig, width="stretch", theme=None,
                        config=_plotly_config(img_fmt, f"{kind}_{sp1.key}_{sp2.key}"))

    with side:
        with st.container(border=True):
            _panel_title("Equilibrium analysis")
            rows = [("Component 1", sp1.name), ("Component 2", sp2.name),
                    ("Activity model", model_name), ("Condition", cond),
                    (f"Pure-1 {ykey}", f"{yv[-1]:.2f} {yunit}"),
                    (f"Pure-2 {ykey}", f"{yv[0]:.2f} {yunit}")]
            rows += ([("Azeotrope x₁", f"{az[0]:.3f}"),
                      (f"Azeotrope {ykey}", f"{az[1]:.2f} {yunit}")]
                     if az else [("Azeotrope", "none (zeotropic)")])
            _meta_table(rows)
        with st.container(border=True):
            _panel_title("Model provenance")
            st.caption(model.provenance)
            st.json(model.activity.describe())
        with st.container(border=True):
            _panel_title("Export")
            st.download_button("Tabulated data (CSV)", df.to_csv(index=False).encode(),
                               file_name=f"{kind}_{sp1.key}_{sp2.key}.csv", mime="text/csv",
                               width="stretch")
            st.caption(f"Chart camera → {img_fmt.upper()}; modebar zoom/pan enabled.")

    _panel_title(f"x–y equilibrium — {sp1.name} (1)", tag="McCabe–Thiele basis")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="y = x (diagonal)",
                              line=dict(color="#5b6470", dash="dash", width=1.6)))
    fig2.add_trace(go.Scatter(x=x1, y=y1, mode="lines", name="Equilibrium curve",
                              line=dict(color=TWO, width=3.2)))
    if az:
        fig2.add_trace(go.Scatter(x=[az[0]], y=[az[0]], mode="markers", name="Azeotrope",
                       marker=dict(color=TWO, size=12, symbol="diamond",
                                   line=dict(color="#eef2f7", width=1.3))))
    _pub_layout(fig2, xtitle="x₁ (liquid mole fraction)", ytitle="y₁ (vapor mole fraction)",
                height=470, legend="topright", x_range=[0, 1], y_range=[0, 1], square=True)
    st.plotly_chart(fig2, width="stretch", theme=None,
                    config=_plotly_config(img_fmt, f"xy_{sp1.key}_{sp2.key}"))

    with st.expander("Assumptions & governing equations"):
        st.markdown(
            "- **Closure:** modified Raoult's law, yᵢP = xᵢγᵢ(x,T)Pᵢˢᵃᵗ(T) — ideal vapor, "
            "real liquid.\n"
            "- **Saturation pressure** from the Antoine equation; **activity coefficients** "
            f"from the {model_name} model.\n"
            "- **Bubble line:** P or T where the first vapor bubble forms; **dew line:** "
            "where the first liquid drop forms.\n"
            "- Valid at low-to-moderate pressure, away from the critical region. See "
            "`docs/thermodynamics.md` for full derivations and parameter sources.")


def _badge(title: str, label: str) -> str:
    color = _BADGE_COLOR.get(label, "#93A4C2")
    return (
        f'<div style="font-family:IBM Plex Mono,monospace;font-size:.72rem;'
        f'color:#93A4C2;margin-bottom:2px">{title}</div>'
        f'<span style="display:inline-block;padding:.18rem .7rem;border-radius:999px;'
        f'font-weight:600;font-size:.85rem;color:{color};'
        f'background:{_rgba(color, 0.14)};border:1px solid {_rgba(color, 0.5)}">{label}</span>'
    )


def _validation_mode(registry: dict[str, ChemicalSpecies]) -> None:
    """Mode 4 — score activity models against literature VLE datasets."""
    with st.sidebar:
        st.subheader("Validation")
        model_name = st.selectbox(
            "Model to inspect", tmodels.MODEL_NAMES,
            index=tmodels.MODEL_NAMES.index(tmodels.NRTLModel.name), key="val_model")

    st.subheader("Model validation against literature VLE data")
    st.caption(
        "Experimental points are digitized from cited standard references "
        "(Gmehling/DECHEMA; Perry's CEH). Predictions use the modified-Raoult γ–φ "
        "model; lower MAE/RMSE is better. Activity-model parameters were fit only "
        "to each azeotrope — matching the full curves here is genuine validation."
    )

    # Score every model × dataset once.
    results: dict[tuple, validation.ValidationResult] = {}
    for ds in validation.DATASETS:
        sp1, sp2 = registry.get(ds.comp1_key), registry.get(ds.comp2_key)
        if sp1 is None or sp2 is None:
            continue
        for m in tmodels.MODEL_NAMES:
            try:
                mdl = tmodels.build_model(m, sp1, sp2)
            except tmodels.NoParametersError:
                continue
            results[(ds.system, m)] = validation.validate(mdl, ds, m)

    _record_calc("validation")

    # Aggregate KPIs for the inspected model.
    chosen = [results[(ds.system, model_name)] for ds in validation.DATASETS
              if (ds.system, model_name) in results]
    if chosen:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Systems scored", f"{len(chosen)}")
        k2.metric("Mean y₁ MAE", f"{np.mean([r.y1_metrics['MAE'] for r in chosen]):.4f}")
        k3.metric("Mean y₁ RMSE", f"{np.mean([r.y1_metrics['RMSE'] for r in chosen]):.4f}")
        k4.metric("Mean T MAE (°C)", f"{np.mean([r.T_metrics['MAE'] for r in chosen]):.2f}")

    # Model-comparison overview: y₁ RMSE per system × model.
    st.markdown("**Vapor-composition error (y₁ RMSE) by model**")
    overview = []
    for ds in validation.DATASETS:
        row = {"System": ds.system}
        for m in tmodels.MODEL_NAMES:
            res = results.get((ds.system, m))
            row[m] = round(res.y1_metrics["RMSE"], 4) if res else None
        overview.append(row)
    st.dataframe(pd.DataFrame(overview), hide_index=True, width="stretch")

    for ds in validation.DATASETS:
        res = results.get((ds.system, model_name))
        st.markdown("---")
        if res is None:
            st.info(f"**{ds.system}** — no {model_name} parameters for this pair "
                    f"(it has no azeotrope record). Try Ideal (Raoult) or another system.")
            continue
        _render_validation_system(res)


def _render_validation_system(res: "validation.ValidationResult") -> None:
    ds = res.dataset
    st.subheader(ds.system)
    st.caption(f"{ds.source}  ·  {ds.note}")

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(_badge("Vapor y₁", res.y1_badge), unsafe_allow_html=True)
    c2.metric("y₁ MAE / RMSE",
              f"{res.y1_metrics['MAE']:.4f} / {res.y1_metrics['RMSE']:.4f}")
    c3.markdown(_badge("Temperature", res.T_badge), unsafe_allow_html=True)
    c4.metric("T MAE / mean %err",
              f"{res.T_metrics['MAE']:.2f} °C / {res.T_metrics['MPE']:.2f}%")

    with np.errstate(divide="ignore", invalid="ignore"):
        dy = np.abs(res.y1_pred - res.y1_exp)
        t_pct = np.where(res.T_exp != 0,
                         np.abs(res.T_pred - res.T_exp) / np.abs(res.T_exp) * 100, np.nan)
    table = pd.DataFrame({
        "x₁": np.round(res.x1, 4),
        "y₁ exp": np.round(res.y1_exp, 4),
        "y₁ pred": np.round(res.y1_pred, 4),
        "|Δy₁|": np.round(dy, 4),
        "T exp (°C)": np.round(res.T_exp, 2),
        "T pred (°C)": np.round(res.T_pred, 2),
        "T %err": np.round(t_pct, 3),
    })
    st.dataframe(table, hide_index=True, width="stretch")

    fit_col, res_col = st.columns([1.25, 1], gap="large")
    with fit_col:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="y = x",
                                 line=dict(color="#5b6470", dash="dash", width=1.6)))
        fig.add_trace(go.Scatter(x=res.x1, y=res.y1_pred, mode="lines",
                                 name=f"{res.model_name} predicted",
                                 line=dict(color=_TWO_PHASE, width=3.0)))
        fig.add_trace(go.Scatter(x=res.x1, y=res.y1_exp, mode="markers", name="experimental",
                                 marker=dict(color=_VAPOR, size=9, symbol="circle-open",
                                             line=dict(width=2))))
        _pub_layout(fig, title="x–y fit", xtitle="x₁ (liquid)", ytitle="y₁ (vapor)",
                    height=440, legend="topright", x_range=[0, 1], y_range=[0, 1], square=True)
        st.plotly_chart(fig, width="stretch", theme=None,
                        config=_plotly_config("png", f"validation_{ds.comp1_key}_{ds.comp2_key}"))
    with res_col:
        residual = res.y1_pred - res.y1_exp
        rfig = go.Figure()
        rfig.add_trace(go.Scatter(x=[0, 1], y=[0, 0], mode="lines", showlegend=False,
                                  line=dict(color="#5b6470", width=1.4)))
        rfig.add_trace(go.Scatter(x=res.x1, y=residual, mode="markers", showlegend=False,
                                  marker=dict(color=_LIQUID, size=8,
                                              line=dict(color="#0d1117", width=1))))
        lim = float(max(0.03, np.nanmax(np.abs(residual)) * 1.25))
        _pub_layout(rfig, title="Residuals  (y₁ pred − exp)", xtitle="x₁ (liquid)",
                    ytitle="Δy₁", height=440, legend="none", x_range=[0, 1], y_range=[-lim, lim])
        st.plotly_chart(rfig, width="stretch", theme=None,
                        config=_plotly_config("png", f"residual_{ds.comp1_key}_{ds.comp2_key}"))


_Q_PRESETS = {
    "Saturated liquid (q = 1)": 1.0,
    "Saturated vapor (q = 0)": 0.0,
    "Custom q": None,
}


def _distillation_mode(registry: dict[str, ChemicalSpecies]) -> None:
    """Mode 5 — binary distillation by the McCabe–Thiele method."""
    active = _active_keys(registry)

    with st.sidebar:
        st.subheader("Distillation column")
        if len(active) >= 2:
            st.caption(f"Light: **{registry[active[0]].name}** · "
                       f"Heavy: **{registry[active[1]].name}** — first two active species.")
        model_name = st.selectbox("Thermodynamic model", tmodels.MODEL_NAMES, 0, key="dist_model")
        c1, c2 = st.columns(2)
        pressure = c1.number_input("Pressure", value=760.0, min_value=1.0, step=10.0, key="dist_P")
        punit = c2.selectbox("P unit", SUPPORTED_PRESSURE_UNITS, 0, key="dist_Pu")
        z_F = st.slider("Feed composition z_F (light)", 0.05, 0.95, 0.50, 0.01, key="dist_zF")
        q_choice = st.selectbox("Feed condition", list(_Q_PRESETS), 0, key="dist_qsel")
        q = _Q_PRESETS[q_choice]
        if q is None:
            q = st.number_input("q (custom)", value=0.5, step=0.1, key="dist_q")
        x_D = st.slider("Distillate x_D (light)", 0.50, 0.999, 0.95, 0.005, key="dist_xD")
        x_B = st.slider("Bottoms x_B (light)", 0.001, 0.50, 0.05, 0.005, key="dist_xB")
        R = st.number_input("Reflux ratio R = L/D", value=2.0, min_value=0.01, step=0.1, key="dist_R")

    if len(active) < 2:
        st.info("Select at least two active species in the sidebar to run the column.")
        return

    sp1, sp2 = registry[active[0]], registry[active[1]]
    use_model = model_name
    if model_name != tmodels.IdealModel.name and not tmodels.has_parameters(model_name, sp1, sp2):
        st.warning(f"No {model_name} parameters for {sp1.name}/{sp2.name} — using Ideal (Raoult).")
        use_model = tmodels.IdealModel.name

    # Convert column pressure to mmHg (the engine's internal basis).
    p_mmhg = engine.convert_pressure(pressure, punit, "mmHg")
    try:
        model = tmodels.build_model(use_model, sp1, sp2)
        res = dist.mccabe_thiele(model, z_F=z_F, q=q, R=R, x_D=x_D, x_B=x_B,
                                 pressure_mmHg=p_mmhg)
    except (dist.DistillationError, tmodels.NoParametersError, ValueError) as exc:
        st.error(f"Cannot run column: {exc}")
        return

    _record_calc("distillation")
    _render_distillation(res, sp1, sp2, use_model, pressure, punit)


def _render_distillation(res, sp1, sp2, model_name, pressure, punit) -> None:
    _panel_title(f"Binary distillation column — {sp1.name} / {sp2.name}",
                 tag=f"McCabe–Thiele · {model_name} · {pressure:g} {punit}")
    if not res.feasible:
        st.error(res.message)

    main, side = st.columns([2.2, 1], gap="large")
    with side:
        with st.container(border=True):
            _panel_title("Column performance")
            _meta_table([
                ("Theoretical stages", f"{res.n_stages}" if res.feasible else "∞ (pinch)"),
                ("Column trays", f"{res.n_trays}" if res.feasible else "—"),
                ("Optimal feed stage", f"{res.feed_stage}" if res.feasible else "—"),
                ("Minimum reflux R_min", f"{res.R_min:.3f}"),
                ("Operating reflux R", f"{res.R:g}"),
                ("R / R_min", f"{res.reflux_ratio_to_min:.2f}"),
            ])
        with st.container(border=True):
            _panel_title("Product specification")
            _meta_table([
                ("Feed z_F (light)", f"{res.z_F:.3f}"),
                ("Feed quality q", f"{res.q:g}"),
                ("Distillate x_D", f"{res.x_D:.3f}"),
                ("Bottoms x_B", f"{res.x_B:.3f}"),
                ("Distillate purity", f"{res.x_D * 100:.1f}% {sp1.name}"),
                ("Bottoms purity", f"{(1 - res.x_B) * 100:.1f}% {sp2.name}"),
            ])
        n_show = res.n_stages if res.feasible else min(res.n_stages, 30)
        k = (st.slider("Stages shown", 1, max(n_show, 1), max(n_show, 1), key="dist_show")
             if n_show > 1 else 1)

    with main:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="y = x (diagonal)",
                                 line=dict(color="#5b6470", dash="dash", width=1.6)))
        fig.add_trace(go.Scatter(x=res.x_eq, y=res.y_eq, mode="lines", name="Equilibrium",
                                 line=dict(color=_TWO_PHASE, width=3.2)))
        xi, yi = res.intersection
        fig.add_trace(go.Scatter(x=[res.x_D, xi], y=[res.x_D, yi], mode="lines",
                                 name="Rectifying line", line=dict(color=_LIQUID, width=2.8)))
        fig.add_trace(go.Scatter(x=[xi, res.x_B], y=[yi, res.x_B], mode="lines",
                                 name="Stripping line", line=dict(color=_VAPOR, width=2.8)))
        fig.add_trace(go.Scatter(x=[res.z_F, xi], y=[res.z_F, yi], mode="lines",
                                 name="q-line", line=dict(color="#C792EA", width=2.0, dash="dot")))
        stair = res.steps[: 2 * k + 1]
        fig.add_trace(go.Scatter(x=[p[0] for p in stair], y=[p[1] for p in stair],
                                 mode="lines", name="Equilibrium stages",
                                 line=dict(color="#e3e8ef", width=1.6)))
        fig.add_trace(go.Scatter(
            x=[res.x_B, res.z_F, res.x_D], y=[res.x_B, res.z_F, res.x_D],
            mode="markers+text", text=["x_B", "z_F", "x_D"], textposition="top left",
            textfont=dict(family=_PUB_MONO, size=11, color="#cfd6df"),
            marker=dict(color=_LIQUID, size=9, line=dict(color="#0d1117", width=1)),
            name="Specs", showlegend=False))
        if res.feasible and 1 <= res.feed_stage <= len(res.stage_corners):
            fx, fy = res.stage_corners[res.feed_stage - 1]
            fig.add_trace(go.Scatter(x=[fx], y=[fy], mode="markers", name="Feed stage",
                                     marker=dict(color=_VAPOR, size=14, symbol="diamond",
                                                 line=dict(color="#eef2f7", width=1.4))))
            fig.add_annotation(x=fx, y=fy, text=f"feed · stage {res.feed_stage}",
                               showarrow=True, arrowhead=2, arrowcolor=_VAPOR, ax=40, ay=20,
                               font=dict(family=_PUB_MONO, size=10, color="#eef2f7"),
                               bgcolor="rgba(29,34,41,0.92)", bordercolor=_VAPOR, borderwidth=1)
        _pub_layout(fig, xtitle=f"x — liquid mole fraction {sp1.name}",
                    ytitle=f"y — vapor mole fraction {sp1.name}", height=660,
                    legend="bottom", x_range=[0, 1], y_range=[0, 1], square=True)
        st.plotly_chart(fig, width="stretch", theme=None,
                        config=_plotly_config("png", f"mccabe_thiele_{sp1.key}_{sp2.key}"))

    a = res.R / (res.R + 1.0)
    b = res.x_D / (res.R + 1.0)
    with st.expander("Complete calculations", expanded=False):
        lines = [
            f"**Rectifying operating line (ROL):**  y = {a:.4f}·x + {b:.4f}  "
            f"(slope R/(R+1), intercept x_D/(R+1))",
        ]
        if abs(res.q - 1.0) < 1e-9:
            lines.append(f"**q-line:**  vertical at x = z_F = {res.z_F:g}  (saturated liquid, q = 1)")
        else:
            sq, bq = res.q / (res.q - 1.0), -res.z_F / (res.q - 1.0)
            lines.append(f"**q-line:**  y = {sq:.4f}·x + {bq:.4f}")
        msol = (yi - res.x_B) / (xi - res.x_B)
        lines += [
            f"**ROL ∩ q-line:**  ({xi:.4f}, {yi:.4f})",
            f"**Stripping operating line (SOL):**  through (x_B, x_B) and the intersection, "
            f"slope = {msol:.4f}",
            f"**Pinch (q-line ∩ equilibrium):**  ({res.pinch[0]:.4f}, {res.pinch[1]:.4f})  "
            f"→  **R_min = {res.R_min:.4f}**,  operating R = {res.R:g}  (R/R_min = "
            f"{res.reflux_ratio_to_min:.2f})",
            f"**Result:**  {res.n_stages} theoretical stages (incl. reboiler) = "
            f"{res.n_trays} trays + reboiler; optimal feed on stage {res.feed_stage}.",
        ]
        st.markdown("\n\n".join(lines))

    if res.feasible:
        corners = pd.DataFrame(
            {"Stage": range(1, len(res.stage_corners) + 1),
             "x (liquid)": [round(p[0], 4) for p in res.stage_corners],
             "y (vapor)": [round(p[1], 4) for p in res.stage_corners]})
        st.download_button("⬇ Stage data (CSV)", corners.to_csv(index=False).encode(),
                           file_name=f"stages_{sp1.key}_{sp2.key}.csv", mime="text/csv")


def _record_calc(kind: str) -> None:
    """Increment the session activity counter (shown on the dashboard)."""
    st.session_state["sim_count"] = st.session_state.get("sim_count", 0) + 1
    by_kind = st.session_state.setdefault("sim_by_kind", {})
    by_kind[kind] = by_kind.get(kind, 0) + 1


@st.cache_data(show_spinner=False)
def _dashboard_metrics(data_path: str) -> dict:
    """Compute headline accuracy KPIs once (cached; static reference data)."""
    registry = load_species(data_path)
    out = {"vle_mae": None, "joback_tb_mape": None, "val_rows": []}
    try:
        maes = []
        for ds in validation.DATASETS:
            sp1, sp2 = registry.get(ds.comp1_key), registry.get(ds.comp2_key)
            if sp1 is None or sp2 is None:
                continue
            name = tmodels.NRTLModel.name if tmodels.has_parameters(
                tmodels.NRTLModel.name, sp1, sp2) else tmodels.IdealModel.name
            res = validation.validate(tmodels.build_model(name, sp1, sp2), ds)
            maes.append(res.y1_metrics["MAE"])
            out["val_rows"].append({
                "System": ds.system, "Best model": name,
                "y₁ MAE": round(res.y1_metrics["MAE"], 4),
                "T MAE (°C)": round(res.T_metrics["MAE"], 2), "Rating": res.y1_badge})
        if maes:
            out["vle_mae"] = float(np.mean(maes))
    except Exception:  # noqa: BLE001 - the dashboard must never crash
        pass
    try:
        _, metrics = predict.benchmark()
        out["joback_tb_mape"] = metrics["Tb"]["MPE"]
    except Exception:  # noqa: BLE001
        pass
    return out


def _goto_mode(mode: str) -> None:
    """Switch the active page. Run as a button callback (before the Mode radio is
    re-instantiated), so writing its widget-bound session-state key is allowed."""
    st.session_state["app_mode"] = mode


def _reset_session() -> None:
    """Clear all session state — active species, navigation, inputs, and the
    run counter — back to defaults. Run as a button callback so widget-bound
    keys can be cleared before the widgets are re-instantiated."""
    st.session_state.clear()


def _summary_card(title: str, description: str, stat: str, target_mode: str) -> None:
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.caption(description)
        st.markdown(stat)
        st.button("Open", key=f"goto_{target_mode}", width="stretch",
                  on_click=_goto_mode, args=(target_mode,))


def _dashboard_mode(registry: dict[str, ChemicalSpecies]) -> None:
    """Top-level research dashboard — project KPIs, status, and module cards."""
    metrics = _dashboard_metrics(str(DATA_PATH))
    vle = f"{metrics['vle_mae']:.4f}" if metrics["vle_mae"] is not None else "—"
    tb = f"{metrics['joback_tb_mape']:.1f}%" if metrics["joback_tb_mape"] is not None else "—"
    n_calc = st.session_state.get("sim_count", 0)

    st.markdown('<div class="tpc-section">Project metrics</div>', unsafe_allow_html=True)
    a = st.columns(3)
    a[0].metric("Species in database", f"{len(registry)}")
    a[1].metric("Thermodynamic models", f"{len(tmodels.MODEL_NAMES)}",
                help="Ideal (Raoult), Wilson, NRTL, UNIQUAC")
    a[2].metric("Validation datasets", f"{len(validation.DATASETS)}")
    b = st.columns(3)
    b[0].metric("Reference molecules", f"{len(predict.LIBRARY)}", help="Joback benchmark set")
    b[1].metric("Mean VLE error (y₁ MAE)", vle, help="Best model per validation system")
    b[2].metric("Joback Tb error (MAPE)", tb)

    st.markdown(
        f'<div class="tpc-chips">'
        f'<span class="tpc-chip"><i></i>Engine <b>operational</b></span>'
        f'<span class="tpc-chip"><i></i><b>{len(tmodels.MODEL_NAMES)}</b> models loaded</span>'
        f'<span class="tpc-chip"><i></i><b>{len(validation.DATASETS)}</b> datasets validated</span>'
        f'<span class="tpc-chip"><i style="background:var(--liquid)"></i>'
        f'<b>{n_calc}</b> calculations this session</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    active = _active_species_list(registry)
    if active:
        names = ", ".join(s.name for s in active)
        st.caption(f"**Active species:** {names} — chosen in the sidebar, these drive "
                   "Property Lookup, Flash, Phase Diagram, and Distillation "
                   "(the first two are used for the binary diagrams and the column).")
    else:
        st.caption("**Active species:** none selected — pick species in the sidebar to "
                   "drive the VLE tools.")

    st.markdown('<div class="tpc-section">Research overview</div>', unsafe_allow_html=True)
    ov1, ov2 = st.columns(2, gap="large")
    with ov1:
        with st.container(border=True):
            _panel_title("Thermodynamic model inventory")
            inventory = pd.DataFrame([
                {"Model": "Ideal (Raoult)", "Type": "Reference (γ = 1)",
                 "Parameters": "none", "Status": "ready"},
                {"Model": "Wilson", "Type": "Activity coefficient",
                 "Parameters": "a₁₂, a₂₁ + Vᵢ", "Status": "ready"},
                {"Model": "NRTL", "Type": "Activity coefficient",
                 "Parameters": "a₁₂, a₂₁, α", "Status": "ready"},
                {"Model": "UNIQUAC", "Type": "Activity coefficient",
                 "Parameters": "a₁₂, a₂₁ + r, q", "Status": "ready"},
            ])
            st.dataframe(inventory, hide_index=True, width="stretch")
            st.caption("γ–φ closure (modified Raoult). EOS models can be added to the "
                       "same `ActivityModel` interface.")
    with ov2:
        with st.container(border=True):
            _panel_title("Validation accuracy", tag="vs. literature VLE")
            if metrics["val_rows"]:
                st.dataframe(pd.DataFrame(metrics["val_rows"]), hide_index=True, width="stretch")
                st.caption("Best activity model per system; parameters fitted only to the "
                           "azeotrope, so matching the full curve is genuine validation.")
            else:
                st.caption("Validation unavailable (missing reference species).")

    st.markdown('<div class="tpc-section">Simulation capabilities</div>',
                unsafe_allow_html=True)
    row1 = st.columns(3)
    with row1[0]:
        _summary_card("Property Lookup",
                      "Antoine vapor pressure ⇄ boiling point with unit conversion.",
                      f"**{len(registry)}** species · 5 P-units · 3 T-units", APP_MODE_LOOKUP)
    with row1[1]:
        _summary_card("VLE Flash",
                      "Isothermal Rachford–Rice flash with selectable activity model.",
                      "Raoult · Wilson · NRTL · UNIQUAC", APP_MODE_FLASH)
    with row1[2]:
        _summary_card("Phase Diagram",
                      "T–x–y / P–x–y diagrams with azeotrope detection and export.",
                      "PNG · SVG · CSV · zoom", APP_MODE_DIAGRAM)
    row2 = st.columns(3)
    with row2[0]:
        _summary_card("Model Validation",
                      "Predictions vs. literature VLE data with quality badges.",
                      f"mean y₁ MAE **{vle}**", APP_MODE_VALIDATION)
    with row2[1]:
        _summary_card("Distillation",
                      "Binary McCabe–Thiele: stages, feed location, R_min.",
                      "operating lines · q-line · stepping", APP_MODE_DISTILL)
    with row2[2]:
        _summary_card("Property Prediction",
                      "Joback group contribution + Lee–Kesler, benchmarked.",
                      f"Tb MAPE **{tb}** over **{len(predict.LIBRARY)}** compounds",
                      APP_MODE_PREDICT)
    row3 = st.columns(3)
    with row3[0]:
        _summary_card("Molecular Viewer",
                      "2D depiction (RDKit) + interactive 3D (3Dmol.js).",
                      f"**{len(molviz.MOLECULE_SMILES)}** molecules · any SMILES",
                      APP_MODE_MOLVIZ)


def _property_prediction_mode(registry: dict[str, ChemicalSpecies]) -> None:
    """Mode 6 — Joback group-contribution property prediction + benchmarking."""
    with st.sidebar:
        st.subheader("Property prediction")
        source = st.radio("Molecule input", ["From library", "Custom (Joback groups)"],
                          key="pred_source")
        if source == "From library":
            lib = list(predict.LIBRARY_BY_NAME)
            active_names = [registry[k].name for k in _active_keys(registry)]
            default = next((n for n in active_names if n in predict.LIBRARY_BY_NAME), "Benzene")
            mol_name = st.selectbox("Molecule", lib,
                                    index=lib.index(default) if default in lib else 0,
                                    key="pred_lib")
            mol = predict.LIBRARY_BY_NAME[mol_name]
            name, formula, groups = mol.name, mol.formula, dict(mol.groups)
            reference = mol
        else:
            reference = None
            name = st.text_input("Name", value="My molecule", key="pred_name")
            formula = st.text_input("Molecular formula", value="C4H10O",
                                    help="e.g. C4H10O — used for molar mass and atom count.",
                                    key="pred_formula")
            chosen_groups = st.multiselect("Joback groups present",
                                           list(predict.JOBACK_GROUPS), key="pred_groups")
            groups = {}
            for g in chosen_groups:
                groups[g] = st.number_input(f"count · {g}", min_value=1, value=1, step=1,
                                            key=f"pred_n_{g}")
        st.text_input("SMILES (optional)", value="", key="pred_smiles",
                      help="Automatic SMILES → groups needs RDKit (planned, Phase 6). "
                           "For now use the library or enter groups directly.")
        t_unit = st.selectbox("Temperature unit (Pˢᵃᵗ curve)",
                              SUPPORTED_TEMPERATURE_UNITS, 0, key="pred_tunit")

    st.subheader("Pure-component property prediction — Joback")
    st.caption("Group-contribution estimates (Joback & Reid 1987); vapor pressure via "
               "Lee–Kesler corresponding states. No training data — fully citable.")

    if not groups:
        st.info("Select at least one Joback group (or pick a library molecule).")
    else:
        try:
            est = predict.joback_estimate(name, formula, groups)
        except (predict.PropertyPredictionError, ValueError) as exc:
            st.error(f"Cannot estimate: {exc}")
            return
        _record_calc("prediction")
        _render_property_estimate(est, reference, registry, t_unit)

    st.markdown("---")
    _render_benchmark_dashboard()


def _render_property_estimate(est, reference, registry, t_unit) -> None:
    _panel_title(f"Property estimate — {est.name}",
                 tag=f"{est.formula} · {est.method} + Lee–Kesler")

    ident, crit = st.columns([1, 1.25], gap="large")
    with ident:
        with st.container(border=True):
            _panel_title("Molecular identity")
            _meta_table([
                ("Name", est.name), ("Formula", est.formula),
                ("Molar mass", f"{est.molar_mass:.3f} g/mol"),
                ("Method", f"{est.method} group contribution"),
            ])
    with crit:
        with st.container(border=True):
            _panel_title("Critical & physical properties")
            _meta_table([
                ("Normal boiling point Tb", f"{est.Tb:.1f} K  ({est.Tb - 273.15:.1f} °C)"),
                ("Critical temperature Tc", f"{est.Tc:.1f} K"),
                ("Critical pressure Pc", f"{est.Pc:.2f} bar"),
                ("Critical volume Vc", f"{est.Vc:.1f} cm³/mol"),
                ("Acentric factor ω", f"{est.omega:.3f}"),
            ])

    if reference is not None:
        rows = [("Tb (K)", est.Tb, reference.Tb_exp), ("Tc (K)", est.Tc, reference.Tc_exp),
                ("Pc (bar)", est.Pc, reference.Pc_exp)]
        errs = [abs(p - e) / e * 100 for _, p, e in rows]
        mpe = float(np.mean(errs))
        conf = "High" if mpe < 2.0 else "Good" if mpe < 5.0 else "Moderate"
        with st.container(border=True):
            _panel_title("Validation vs. literature", tag="Poling 5th ed. / NIST")
            df = pd.DataFrame({
                "Property": [r[0] for r in rows],
                "Predicted": [round(r[1], 2) for r in rows],
                "Experimental": [round(r[2], 2) for r in rows],
                "Abs. error": [round(abs(r[1] - r[2]), 2) for r in rows],
                "% error": [round(err, 2) for err in errs],
            })
            st.dataframe(df, hide_index=True, width="stretch")
            cc1, cc2 = st.columns([1, 2])
            cc1.markdown(_badge("Confidence",
                                {"High": "Excellent", "Good": "Good",
                                 "Moderate": "Fair"}[conf]), unsafe_allow_html=True)
            cc2.caption(f"Mean absolute error {mpe:.2f}% across Tb, Tc, Pc — within Joback's "
                        "documented accuracy. Alcohols' Tb are systematically under-predicted.")

    # Vapor-pressure curve (Lee–Kesler); overlay Antoine if the species is in the DB.
    t_lo, t_hi = 0.55 * est.Tc, min(est.Tc, est.Tb + 60.0)
    t_grid = np.linspace(t_lo, t_hi, 120)
    psat = np.array([est.vapor_pressure(t, "mmHg") for t in t_grid])
    t_disp = engine.convert_temperature(t_grid, "Kelvin", t_unit)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_disp, y=psat, mode="lines",
                             name="Joback + Lee–Kesler",
                             line=dict(color=_LIQUID, width=3.0)))
    match = next((s for s in registry.values() if s.name.lower() == est.name.lower()), None)
    if match is not None:
        psat_antoine = engine.vapor_pressure_curve(match.antoine, t_grid,
                                                   temp_unit="Kelvin", pressure_unit="mmHg")
        fig.add_trace(go.Scatter(x=t_disp, y=psat_antoine, mode="lines",
                                 name="Antoine (measured fit)",
                                 line=dict(color=_VAPOR, width=2.4, dash="dot")))
    _pub_layout(fig, title="Vapor-pressure curve", xtitle=f"Temperature ({t_unit})",
                ytitle="Vapor pressure (mmHg)", height=420, legend="topright")
    st.caption("Estimated by corresponding states"
               + (" — orange is the independent Antoine fit (cross-check)." if match
                  else " (no measured Antoine fit in the database for this molecule)."))
    st.plotly_chart(fig, width="stretch", theme=None,
                    config=_plotly_config("png", f"psat_{est.formula}"))

    with st.expander("Prediction methodology & references"):
        st.markdown(
            "**Group contribution (Joback & Reid, 1987):**\n"
            "- Tb = 198.2 + Σ nᵢ ΔTb,ᵢ\n"
            "- Tc = Tb / (0.584 + 0.965 ΣΔTc − (ΣΔTc)²)\n"
            "- Pc = (0.113 + 0.0032 N_atoms − ΣΔPc)⁻²;  Vc = 17.5 + ΣΔVc\n\n"
            "**Corresponding states (Lee–Kesler):** the acentric factor ω is obtained "
            "from Tb, Tc, Pc, then vapor pressure follows ln(P/Pc) = f⁽⁰⁾(Tr) + ω f⁽¹⁾(Tr), "
            "which returns 1 atm at Tb by construction.\n\n"
            "**Assumptions:** additive group contributions; no isomer/steric corrections; "
            "ideal-gas reference. Accuracy is typically ~2–3% (Tb, Tc) and ~5% (Pc).\n\n"
            "*Refs: Joback & Reid, Chem. Eng. Commun. 57 (1987) 233; Poling, Prausnitz & "
            "O'Connell, The Properties of Gases and Liquids, 5th ed. See "
            "`docs/thermodynamics.md` §8.*")


def _render_benchmark_dashboard() -> None:
    st.subheader("Benchmarking dashboard")
    st.caption("Joback predictions vs. experimental across the reference library. "
               "Architecture supports adding data-driven models (Random Forest / "
               "Gradient Boosting / XGBoost / GNN) to this table later.")
    rows, metrics = predict.benchmark()

    k = st.columns(3)
    for col, prop, unit in zip(k, ("Tb", "Tc", "Pc"), ("K", "K", "bar")):
        m = metrics[prop]
        col.metric(f"{prop} — MAE / R²", f"{m['MAE']:.2f} {unit}  /  {m['R2']:.3f}",
                   help=f"RMSE {m['RMSE']:.2f} {unit}, mean abs % error {m['MPE']:.2f}%")

    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    # Parity plots for Tb, Tc, Pc.
    cols = st.columns(3)
    for col, prop, unit in zip(cols, ("Tb", "Tc", "Pc"), ("K", "K", "bar")):
        exp = [r[f"{prop} exp ({unit})"] for r in rows]
        pred_v = [r[f"{prop} pred ({unit})"] for r in rows]
        lo, hi = min(exp + pred_v), max(exp + pred_v)
        pad = (hi - lo) * 0.06 or 1.0
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[lo - pad, hi + pad], y=[lo - pad, hi + pad], mode="lines",
                                 line=dict(color="#5b6470", dash="dash", width=1.6),
                                 showlegend=False))
        fig.add_trace(go.Scatter(x=exp, y=pred_v, mode="markers",
                                 marker=dict(color=_TWO_PHASE, size=9,
                                             line=dict(color="#0d1117", width=1)),
                                 showlegend=False, text=[r["Molecule"] for r in rows],
                                 hovertemplate="%{text}<br>exp %{x:.1f}<br>pred %{y:.1f}<extra></extra>"))
        _pub_layout(fig, title=f"{prop} ({unit})", xtitle="experimental", ytitle="predicted",
                    height=340, legend="none",
                    x_range=[lo - pad, hi + pad], y_range=[lo - pad, hi + pad], square=True)
        col.plotly_chart(fig, width="stretch", theme=None,
                         config=_plotly_config("png", f"parity_{prop}"))


def render() -> None:
    """Render the full Streamlit page."""
    st.set_page_config(
        page_title="Thermodynamic Property Calculator",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_theme()
    if st.session_state.get("density") == "Compact":
        st.markdown(f"<style>{_COMPACT_CSS}</style>", unsafe_allow_html=True)

    try:
        registry = _load(str(DATA_PATH))
    except SpeciesDataError as exc:
        st.error(f"Failed to load chemical data: {exc}")
        st.stop()

    _init_active_species(registry)
    st.session_state.setdefault("app_mode", APP_MODE_DASHBOARD)

    with st.sidebar:
        st.markdown('<div class="nav-brand">⬡ VLE PLATFORM<small>THERMODYNAMIC '
                    'MODELING SUITE</small></div>', unsafe_allow_html=True)
        _sidebar_nav()
        st.divider()
        st.markdown('<div class="nav-group">Active species</div>', unsafe_allow_html=True)
        st.multiselect(
            "Active species",
            options=list(registry.keys()),
            key="active_species",
            format_func=lambda k: registry[k].name,
            label_visibility="collapsed",
            help="One selection drives Property Lookup, Flash, Phase Diagram, and "
                 "Distillation. Phase Diagram and Distillation use the first two.",
        )
        with st.expander("Add / manage species"):
            tab_cat, tab_manual, tab_manage = st.tabs(["Catalog", "Manual", "Remove"])
            with tab_cat:
                _add_from_catalog_ui(registry)
            with tab_manual:
                _manual_add_ui(registry)
            with tab_manage:
                _manage_ui(registry)
        st.selectbox("Layout density", ["Comfortable", "Compact"], key="density")
        st.divider()
        st.button("↺  Reset session", key="reset_session", width="stretch",
                  on_click=_reset_session,
                  help="Clear active species, navigation, inputs, and the run counter "
                       "back to defaults. Saved species in chemical_data.json are kept.")

    st.session_state.setdefault("app_mode", APP_MODE_DASHBOARD)
    app_mode = st.session_state["app_mode"]
    _command_bar(app_mode, registry)

    if app_mode == APP_MODE_DASHBOARD:
        _hero(registry)
        _dashboard_mode(registry)
    elif app_mode == APP_MODE_LOOKUP:
        _property_lookup_mode(registry)
    elif app_mode == APP_MODE_FLASH:
        _flash_mode(registry)
    elif app_mode == APP_MODE_DIAGRAM:
        _phase_diagram_mode(registry)
    elif app_mode == APP_MODE_VALIDATION:
        _validation_mode(registry)
    elif app_mode == APP_MODE_DISTILL:
        _distillation_mode(registry)
    elif app_mode == APP_MODE_PREDICT:
        _property_prediction_mode(registry)
    else:
        _molecular_viewer_mode(registry)

    _status_bar()
