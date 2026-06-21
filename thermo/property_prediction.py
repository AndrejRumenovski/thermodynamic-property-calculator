"""Pure-component property prediction by group contribution.

Estimates molecular weight, normal boiling point, critical temperature and
pressure, critical volume, acentric factor, and vapor pressure for a molecule
described by its **Joback functional groups** (Joback & Reid, *Chem. Eng.
Commun.* 57 (1987) 233; tabulated in Poling, Prausnitz & O'Connell, *The
Properties of Gases and Liquids*, 5th ed.).

Joback relations (light component, SI unless noted):

    Tb = 198.2 + Σ nᵢ ΔTb_i                                         [K]
    Tc = Tb / (0.584 + 0.965·ΣΔTc − (ΣΔTc)²)                        [K]
    Pc = (0.113 + 0.0032·N_atoms − ΣΔPc)⁻²                          [bar]
    Vc = 17.5 + Σ nᵢ ΔVc_i                                          [cm³/mol]

The acentric factor and vapor pressure follow from corresponding states
(Lee–Kesler), so the estimate is consistent: P_sat(Tb) = 1 atm by construction.

This is a *group-contribution* model — no training data, fully citable. The
``PropertyPredictor`` interface leaves room for data-driven models (Random
Forest / Gradient Boosting / XGBoost / GNN) to be added later and benchmarked
side by side.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------- #
# Atomic weights (g/mol) for formula parsing
# --------------------------------------------------------------------------- #
ATOMIC_WEIGHTS = {
    "H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998,
    "S": 32.06, "Cl": 35.45, "Br": 79.904, "I": 126.904, "P": 30.974,
    "Si": 28.085, "D": 2.014,
}
_FORMULA_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


def parse_formula(formula: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for element, num in _FORMULA_RE.findall(formula.strip()):
        if not element:
            continue
        counts[element] = counts.get(element, 0) + (int(num) if num else 1)
    if not counts:
        raise ValueError(f"Could not parse molecular formula {formula!r}.")
    return counts


def molar_mass(formula: str) -> float:
    total = 0.0
    for element, n in parse_formula(formula).items():
        if element not in ATOMIC_WEIGHTS:
            raise ValueError(f"Unknown element {element!r} in {formula!r}.")
        total += ATOMIC_WEIGHTS[element] * n
    return total


def atom_count(formula: str) -> int:
    return sum(parse_formula(formula).values())


# --------------------------------------------------------------------------- #
# Joback group contributions: name -> (ΔTb [K], ΔTc, ΔPc, ΔVc [cm³/mol])
# Values verified against known Tb/Tc/Pc for the molecule library below.
# --------------------------------------------------------------------------- #
JOBACK_GROUPS: dict[str, tuple[float, float, float, float]] = {
    "-CH3": (23.58, 0.0141, -0.0012, 65),
    "-CH2-": (22.88, 0.0189, 0.0000, 56),
    ">CH-": (21.74, 0.0164, 0.0020, 41),
    ">C<": (18.25, 0.0067, 0.0043, 27),
    "=CH2": (18.18, 0.0113, -0.0028, 56),
    "=CH-": (24.96, 0.0129, -0.0006, 46),
    "=C<": (24.14, 0.0117, 0.0011, 38),
    "ring-CH2-": (27.15, 0.0100, 0.0025, 48),
    "ring>CH-": (21.78, 0.0122, 0.0004, 38),
    "ring=CH-": (26.73, 0.0082, 0.0011, 41),
    "ring=C<": (31.01, 0.0143, 0.0008, 32),
    "-OH (alcohol)": (92.88, 0.0741, 0.0112, 28),
    "-O- (nonring)": (22.42, 0.0168, 0.0015, 18),
    ">C=O (nonring)": (76.75, 0.0380, 0.0031, 62),
    "-CHO (aldehyde)": (72.24, 0.0379, 0.0030, 82),
    "-COOH (acid)": (169.09, 0.0791, 0.0077, 89),
    "-COO- (ester)": (81.10, 0.0481, 0.0005, 82),
    "-NH2": (73.23, 0.0243, 0.0109, 38),
}


class PropertyPredictionError(ValueError):
    pass


@dataclass(frozen=True)
class PropertyEstimate:
    name: str
    formula: str
    molar_mass: float       # g/mol
    Tb: float               # normal boiling point, K
    Tc: float               # critical temperature, K
    Pc: float               # critical pressure, bar
    Vc: float               # critical volume, cm³/mol
    omega: float            # acentric factor
    method: str = "Joback"

    def vapor_pressure(self, temperature_k: float, pressure_unit: str = "bar") -> float:
        """Lee–Kesler corresponding-states vapor pressure at T."""
        return lee_kesler_psat(temperature_k, self.Tc, self.Pc, self.omega, pressure_unit)


def _f0(tr: float) -> float:
    return 5.92714 - 6.09648 / tr - 1.28862 * math.log(tr) + 0.169347 * tr ** 6


def _f1(tr: float) -> float:
    return 15.2518 - 15.6875 / tr - 13.4721 * math.log(tr) + 0.43577 * tr ** 6


def acentric_factor(Tb: float, Tc: float, Pc_bar: float) -> float:
    """Lee–Kesler acentric factor from the normal boiling point."""
    tbr = Tb / Tc
    pc_atm = Pc_bar / 1.01325
    return (-math.log(pc_atm) - _f0(tbr)) / _f1(tbr)


def lee_kesler_psat(T: float, Tc: float, Pc_bar: float, omega: float,
                    pressure_unit: str = "bar") -> float:
    """Vapor pressure at ``T`` (Lee–Kesler); returns 1 atm exactly at Tb."""
    tr = T / Tc
    pc_atm = Pc_bar / 1.01325
    psat_atm = pc_atm * math.exp(_f0(tr) + omega * _f1(tr))
    factors = {"atm": 1.0, "bar": 1.01325, "kPa": 101.325, "Pa": 101325.0,
               "mmHg": 760.0}
    if pressure_unit not in factors:
        raise PropertyPredictionError(f"Unsupported pressure unit {pressure_unit!r}.")
    return psat_atm * factors[pressure_unit]


def joback_estimate(name: str, formula: str, groups: dict[str, int]) -> PropertyEstimate:
    """Estimate properties from Joback groups."""
    if not groups:
        raise PropertyPredictionError("Provide at least one Joback group.")
    unknown = set(groups) - set(JOBACK_GROUPS)
    if unknown:
        raise PropertyPredictionError(f"Unknown Joback group(s): {sorted(unknown)}.")

    s_tb = sum(JOBACK_GROUPS[g][0] * n for g, n in groups.items())
    s_tc = sum(JOBACK_GROUPS[g][1] * n for g, n in groups.items())
    s_pc = sum(JOBACK_GROUPS[g][2] * n for g, n in groups.items())
    s_vc = sum(JOBACK_GROUPS[g][3] * n for g, n in groups.items())

    n_atoms = atom_count(formula)
    Tb = 198.2 + s_tb
    denom = 0.584 + 0.965 * s_tc - s_tc ** 2
    Tc = Tb / denom
    Pc = (0.113 + 0.0032 * n_atoms - s_pc) ** -2
    Vc = 17.5 + s_vc
    omega = acentric_factor(Tb, Tc, Pc)
    return PropertyEstimate(name=name, formula=formula, molar_mass=molar_mass(formula),
                            Tb=Tb, Tc=Tc, Pc=Pc, Vc=Vc, omega=omega)


# --------------------------------------------------------------------------- #
# Benchmark library — Joback group decompositions + experimental references
# (Tb, Tc in K; Pc in bar; Poling 5th ed. App. A / NIST). Water is omitted —
# Joback has no suitable group for it.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Molecule:
    name: str
    formula: str
    groups: dict
    Tb_exp: float
    Tc_exp: float
    Pc_exp: float


LIBRARY: list[Molecule] = [
    Molecule("n-Pentane", "C5H12", {"-CH3": 2, "-CH2-": 3}, 309.2, 469.7, 33.70),
    Molecule("n-Hexane", "C6H14", {"-CH3": 2, "-CH2-": 4}, 341.9, 507.6, 30.25),
    Molecule("n-Heptane", "C7H16", {"-CH3": 2, "-CH2-": 5}, 371.6, 540.2, 27.40),
    Molecule("n-Octane", "C8H18", {"-CH3": 2, "-CH2-": 6}, 398.8, 568.7, 24.90),
    Molecule("Cyclohexane", "C6H12", {"ring-CH2-": 6}, 353.9, 553.6, 40.70),
    Molecule("Benzene", "C6H6", {"ring=CH-": 6}, 353.2, 562.2, 48.95),
    Molecule("Toluene", "C7H8", {"ring=CH-": 5, "ring=C<": 1, "-CH3": 1}, 383.8, 591.8, 41.06),
    Molecule("Methanol", "CH4O", {"-CH3": 1, "-OH (alcohol)": 1}, 337.7, 512.6, 80.97),
    Molecule("Ethanol", "C2H6O", {"-CH3": 1, "-CH2-": 1, "-OH (alcohol)": 1}, 351.4, 514.0, 61.37),
    Molecule("1-Propanol", "C3H8O", {"-CH3": 1, "-CH2-": 2, "-OH (alcohol)": 1}, 370.3, 536.8, 51.70),
    Molecule("Acetone", "C3H6O", {"-CH3": 2, ">C=O (nonring)": 1}, 329.2, 508.1, 47.01),
    Molecule("Diethyl ether", "C4H10O", {"-CH3": 2, "-CH2-": 2, "-O- (nonring)": 1}, 307.6, 466.7, 36.40),
    Molecule("Acetic acid", "C2H4O2", {"-CH3": 1, "-COOH (acid)": 1}, 391.1, 592.0, 57.86),
]
LIBRARY_BY_NAME = {m.name: m for m in LIBRARY}


def estimate_molecule(mol: Molecule) -> PropertyEstimate:
    return joback_estimate(mol.name, mol.formula, mol.groups)


# --------------------------------------------------------------------------- #
# Predictor interface (extensible to data-driven models) + benchmarking
# --------------------------------------------------------------------------- #
class JobackPredictor:
    """Group-contribution predictor. Future ML models (Random Forest, Gradient
    Boosting, XGBoost, GNN) can implement the same ``predict`` surface and join
    the benchmark."""

    name = "Joback (group contribution)"
    needs_training = False

    def predict(self, name: str, formula: str, groups: dict[str, int]) -> PropertyEstimate:
        return joback_estimate(name, formula, groups)


def regression_metrics(experimental, predicted) -> dict:
    exp = np.asarray(experimental, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    err = pred - exp
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((exp - exp.mean()) ** 2))
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "R2": (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "MPE": float(np.mean(np.abs(err / exp)) * 100.0),
    }


def benchmark(predictor: JobackPredictor | None = None):
    """Score a predictor across the library. Returns (rows, metrics_by_property)."""
    predictor = predictor or JobackPredictor()
    rows = []
    cols = {"Tb": ([], []), "Tc": ([], []), "Pc": ([], [])}  # name -> (exp, pred)
    for mol in LIBRARY:
        est = predictor.predict(mol.name, mol.formula, mol.groups)
        rows.append({
            "Molecule": mol.name, "Formula": mol.formula,
            "Tb pred (K)": round(est.Tb, 1), "Tb exp (K)": mol.Tb_exp,
            "Tc pred (K)": round(est.Tc, 1), "Tc exp (K)": mol.Tc_exp,
            "Pc pred (bar)": round(est.Pc, 2), "Pc exp (bar)": mol.Pc_exp,
        })
        for key, exp_val, pred_val in (("Tb", mol.Tb_exp, est.Tb),
                                       ("Tc", mol.Tc_exp, est.Tc),
                                       ("Pc", mol.Pc_exp, est.Pc)):
            cols[key][0].append(exp_val)
            cols[key][1].append(pred_val)
    metrics = {k: regression_metrics(exp, pred) for k, (exp, pred) in cols.items()}
    return rows, metrics
