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

from . import flash_engine as flash
from . import thermo_engine as engine
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

APP_MODE_LOOKUP = "Property Lookup"
APP_MODE_FLASH = "VLE Flash Calculation"

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
  --bg:#0E1726; --panel:#15203A; --border:#273656;
  --text:#E7EDF7; --muted:#93A4C2;
  --liquid:#38BDF8; --two:#34D399; --vapor:#F6A93B;
}
/* Type */
html, body, .stApp, [data-testid="stAppViewContainer"]{
  font-family:'IBM Plex Sans', system-ui, sans-serif;
  color:var(--text);
}
h1,h2,h3,h4,[data-testid="stHeading"]{
  font-family:'Space Grotesk','IBM Plex Sans',sans-serif !important;
  letter-spacing:-0.015em;
}
[data-testid="stCaptionContainer"], .stCaption, small{
  color:var(--muted) !important;
  font-family:'IBM Plex Mono', monospace;
  letter-spacing:0.02em;
}
/* Phase-diagram ground: cool (liquid) glow top-left, warm (vapor) top-right, fine grid */
.stApp{
  background-color:var(--bg);
  background-image:
    radial-gradient(1100px 560px at 8% -12%, rgba(56,189,248,0.10), transparent 60%),
    radial-gradient(1000px 620px at 104% -4%, rgba(246,169,59,0.07), transparent 56%),
    linear-gradient(rgba(39,54,86,0.30) 1px, transparent 1px),
    linear-gradient(90deg, rgba(39,54,86,0.30) 1px, transparent 1px);
  background-size:100% 100%, 100% 100%, 46px 46px, 46px 46px;
  background-attachment:fixed;
}
[data-testid="stHeader"]{ background:transparent; }
.block-container{ padding-top:2.2rem; max-width:1180px; }
/* Hero */
.tpc-hero{ margin:0.2rem 0 1.6rem; animation:tpcRise .5s ease-out both; }
.tpc-eyebrow{
  font-family:'IBM Plex Mono', monospace; text-transform:uppercase;
  letter-spacing:0.34em; font-size:0.72rem; color:var(--liquid); margin-bottom:0.5rem;
}
.tpc-title{
  font-family:'Space Grotesk',sans-serif; font-weight:600;
  font-size:2.5rem; line-height:1.04; margin:0; color:#F3F7FE;
}
.tpc-sub{ color:var(--muted); font-size:1.02rem; margin:0.55rem 0 0.9rem; max-width:60ch; }
.tpc-sub b{ color:var(--text); font-weight:600; }
.tpc-spine{
  height:3px; width:100%; border-radius:3px;
  background:linear-gradient(90deg,var(--liquid),var(--two) 52%,var(--vapor));
  box-shadow:0 0 22px rgba(56,189,248,0.25);
}
.tpc-legend{ display:flex; gap:1.4rem; margin-top:0.7rem;
  font-family:'IBM Plex Mono',monospace; font-size:0.76rem; color:var(--muted); }
.tpc-legend span{ display:inline-flex; align-items:center; gap:0.42rem; }
.tpc-legend i{ width:9px; height:9px; border-radius:50%; display:inline-block; }
.tpc-legend .ph-liq{ background:var(--liquid); }
.tpc-legend .ph-two{ background:var(--two); }
.tpc-legend .ph-vap{ background:var(--vapor); }
/* Regime card */
.tpc-regime{
  display:flex; gap:0.95rem; align-items:center; margin:0.3rem 0 1.1rem;
  padding:1rem 1.15rem; border-radius:0.85rem;
  border:1px solid color-mix(in srgb, var(--accent) 40%, var(--border));
  background:linear-gradient(90deg, color-mix(in srgb, var(--accent) 15%, var(--panel)), var(--panel) 62%);
  position:relative; overflow:hidden;
}
.tpc-regime::before{ content:""; position:absolute; left:0; top:0; bottom:0; width:4px; background:var(--accent); }
.tpc-regime-dot{ width:13px; height:13px; border-radius:50%; background:var(--accent);
  box-shadow:0 0 0 5px color-mix(in srgb, var(--accent) 24%, transparent); flex:0 0 auto; }
.tpc-regime-label{ font-family:'Space Grotesk',sans-serif; font-weight:600; font-size:1.18rem; color:#F3F7FE; }
.tpc-regime-meta{ font-family:'IBM Plex Mono',monospace; font-size:0.82rem; color:var(--muted); margin-top:0.12rem; }
/* Metric cards -> instrument readouts */
[data-testid="stMetric"]{
  background:linear-gradient(180deg, var(--panel), #111B30);
  border:1px solid var(--border); border-radius:0.75rem;
  padding:0.85rem 1rem 0.95rem; position:relative; overflow:hidden;
}
[data-testid="stMetric"]::before{ content:""; position:absolute; left:0; top:0; bottom:0; width:3px;
  background:var(--liquid); opacity:0.75; }
[data-testid="stMetricValue"]{ font-family:'IBM Plex Mono',monospace !important; font-weight:600;
  font-size:1.65rem !important; line-height:1.15; }
[data-testid="stMetricLabel"]{ color:var(--muted) !important; }
[data-testid="stAppDeployButton"]{ display:none; }
/* Sidebar */
[data-testid="stSidebar"]{ background:#0F182B; border-right:1px solid var(--border); }
[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{ font-size:0.78rem; text-transform:uppercase;
  letter-spacing:0.18em; color:var(--muted); }
/* Buttons */
.stButton>button{ border-radius:0.6rem; border:1px solid var(--border); font-weight:600; transition:all .15s ease; }
.stButton>button:hover{ border-color:var(--liquid); color:var(--liquid); transform:translateY(-1px); }
/* Tables, tabs, dividers */
[data-testid="stDataFrame"]{ border:1px solid var(--border); border-radius:0.6rem; }
[data-baseweb="tab-list"]{ gap:0.25rem; }
hr{ border-color:var(--border); }
:focus-visible{ outline:2px solid var(--liquid); outline-offset:2px; }
@keyframes tpcRise{ from{ opacity:0; transform:translateY(9px);} to{ opacity:1; transform:none;} }
@media (prefers-reduced-motion: reduce){ *{ animation:none !important; transition:none !important; } }
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


def _property_lookup_mode(registry: dict[str, ChemicalSpecies]) -> None:
    """Mode 1 — single-species property lookup (vapor pressure ⇄ boiling T)."""
    keys = list(registry.keys())
    defaults = [k for k in ("water", "benzene") if k in registry] or keys[:1]

    with st.sidebar:
        st.subheader("Property inputs")
        chosen = st.multiselect(
            "Species",
            options=keys,
            default=defaults,
            format_func=lambda k: registry[k].name,
            key="lookup_species",
        )
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
    keys = list(registry.keys())
    default_pair = [k for k in ("benzene", "toluene") if k in registry]
    if len(default_pair) < 2:
        default_pair = keys[:2]

    with st.sidebar:
        st.subheader("Flash inputs")
        chosen = st.multiselect(
            "Components (≥ 2)",
            options=keys,
            default=default_pair,
            format_func=lambda k: registry[k].name,
            key="flash_components",
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
    try:
        result = flash.flash(
            species_list,
            z_inputs,
            temperature,
            pressure,
            temp_unit=temp_unit,
            pressure_unit=pressure_unit,
        )
    except (flash.FlashError, ValueError) as exc:
        st.error(f"Flash failed: {exc}")
        return

    _render_flash_result(registry, chosen, z_inputs, result)


def _render_flash_result(
    registry: dict[str, ChemicalSpecies],
    chosen: list[str],
    z_inputs: list[float],
    result: "flash.FlashResult",
) -> None:
    names = [registry[k].name for k in chosen]
    st.subheader(
        f"Flash at {result.temperature:g} {result.temp_unit} "
        f"and {result.pressure:g} {result.pressure_unit}"
    )
    _regime_card(result)

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

    table = pd.DataFrame(
        {
            "Component": names,
            "zᵢ (feed)": np.round(result.z, 4),
            f"Pˢᵃᵗ ({result.pressure_unit})": np.round(result.psat, 3),
            "Kᵢ = Pˢᵃᵗ/P": np.round(result.K, 4),
            "xᵢ (liquid)": np.round(result.x, 4),
            "yᵢ (vapor)": np.round(result.y, 4),
        }
    )
    st.dataframe(table, hide_index=True, width="stretch")

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


def render() -> None:
    """Render the full Streamlit page."""
    st.set_page_config(
        page_title="Thermodynamic Property Calculator",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_theme()

    try:
        registry = _load(str(DATA_PATH))
    except SpeciesDataError as exc:
        st.error(f"Failed to load chemical data: {exc}")
        st.stop()

    _hero(registry)

    with st.sidebar:
        st.header("Mode")
        app_mode = st.radio(
            "Mode",
            [APP_MODE_LOOKUP, APP_MODE_FLASH],
            key="app_mode",
            label_visibility="collapsed",
        )
        st.divider()
        with st.expander("➕ Add / manage species"):
            tab_cat, tab_manual, tab_manage = st.tabs(["Catalog", "Manual", "Remove"])
            with tab_cat:
                _add_from_catalog_ui(registry)
            with tab_manual:
                _manual_add_ui(registry)
            with tab_manage:
                _manage_ui(registry)
        st.divider()

    if app_mode == APP_MODE_LOOKUP:
        _property_lookup_mode(registry)
    else:
        _flash_mode(registry)
