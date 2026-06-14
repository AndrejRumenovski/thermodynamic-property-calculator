"""Build a validated Antoine catalog from the `chemicals` library.

This is a *build-time* tool, not part of the app's runtime. It reads the Antoine
(Poling) coefficient table bundled with the BSD-licensed ``chemicals`` package
(Caleb Bell; data from Poling et al., *The Properties of Gases and Liquids*),
and writes ``antoine_catalog.json`` in the same schema as ``chemical_data.json``.

Crucially, every entry is *validated*: the Antoine constants must reproduce the
compound's known normal boiling point (within ``BP_TOLERANCE_K``) before it is
included. This guarantees the catalog ships "correct data" rather than whatever
happens to be in the source table. Entries whose boiling point cannot be
verified are dropped and reported.

Run with the dev dependency installed::

    pip install -r requirements-dev.txt
    python tools/build_catalog.py
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import chemicals
from chemicals.vapor_pressure import Psat_data_AntoinePoling as ANTOINE_TABLE

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = REPO_ROOT / "antoine_catalog.json"

ATM_PA = 101_325.0
BP_TOLERANCE_K = 5.0  # max |predicted normal BP - known Tb|; also catches unit errors


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    return slug or "species"


def _predicted_normal_bp_k(A: float, B: float, C: float) -> float | None:
    """Temperature (K) at which the Antoine pressure equals 1 atm, or None."""
    denom = A - math.log10(ATM_PA)
    if denom <= 0:
        return None
    return B / denom - C


def _try(fn, *args):
    try:
        value = fn(*args)
    except Exception:  # noqa: BLE001 - the library raises many error types
        return None
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return value


def build() -> dict:
    species: dict[str, dict] = {}
    used_keys: set[str] = set()
    stats = {"total": 0, "included": 0, "no_tb": 0, "bp_mismatch": 0, "no_mw": 0}

    for cas, row in ANTOINE_TABLE.iterrows():
        stats["total"] += 1
        A, B, C = float(row["A"]), float(row["B"]), float(row["C"])
        name = str(row["Chemical"]).strip()

        tb = _try(chemicals.Tb, cas)
        if tb is None:
            stats["no_tb"] += 1
            continue

        predicted = _predicted_normal_bp_k(A, B, C)
        if predicted is None or abs(predicted - tb) > BP_TOLERANCE_K:
            stats["bp_mismatch"] += 1
            continue

        mw = _try(chemicals.MW, cas)
        if mw is None:
            stats["no_mw"] += 1
            continue

        # Optional, nicer metadata; fall back gracefully if unavailable.
        meta = _try(lambda c: chemicals.identifiers.pubchem_db.search_CAS(c), cas)
        display_name = name
        formula = ""
        if meta is not None:
            display_name = (getattr(meta, "common_name", None) or name).strip()
            formula = (getattr(meta, "formula", "") or "").strip()

        key = _slugify(display_name)
        base_key = key
        suffix = 2
        while key in used_keys:
            key = f"{base_key}_{suffix}"
            suffix += 1
        used_keys.add(key)

        species[key] = {
            "name": display_name,
            "formula": formula,
            "molar_mass": round(float(mw), 4),
            "cas": cas,
            "antoine_constants": {
                "A": A,
                "B": B,
                "C": C,
                "units": {"P": "Pa", "T": "Kelvin"},
                "T_min": round(float(row["Tmin"]), 2),
                "T_max": round(float(row["Tmax"]), 2),
            },
        }
        stats["included"] += 1

    payload = {
        "_meta": {
            "source": "chemicals library (Poling Antoine table); validated vs known Tb",
            "validation": f"|predicted normal BP - Tb| <= {BP_TOLERANCE_K} K",
            "count": stats["included"],
        },
        "species": dict(sorted(species.items(), key=lambda kv: kv[1]["name"].lower())),
    }
    print(
        f"Catalog: {stats['included']} included of {stats['total']} "
        f"(dropped: {stats['bp_mismatch']} BP mismatch, "
        f"{stats['no_tb']} no Tb, {stats['no_mw']} no MW)"
    )
    return payload


def main() -> None:
    payload = build()
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH} ({len(payload['species'])} species)")


if __name__ == "__main__":
    main()
