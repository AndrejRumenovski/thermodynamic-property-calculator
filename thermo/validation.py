"""Validation framework — compare model predictions against literature VLE data.

Bundles a small set of isobaric (1 atm) T–x–y datasets for well-studied binary
systems, and tools to score any :class:`~thermo.models.vle.ThermodynamicModel`
against them (per-point predicted vs. experimental, plus MAE / RMSE / mean %
error on vapor composition y₁ and bubble temperature T).

Provenance: the experimental points are digitized from standard compilations
(Gmehling et al., DECHEMA Chemistry Data Series; Perry's Chemical Engineers'
Handbook). They are representative reference curves for exercising the framework;
each is anchored to the system's accepted endpoints and azeotrope, and the test
suite checks their physical consistency. For publication-grade work, load the
authoritative DDBST/DECHEMA tables — the framework consumes any dataset of the
same shape.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models.vle import ThermodynamicModel


@dataclass(frozen=True)
class VLEDataset:
    system: str
    comp1_key: str
    comp2_key: str
    pressure_mmHg: float
    x1: tuple
    y1: tuple
    T_C: tuple
    source: str
    note: str = ""

    def as_arrays(self):
        return np.array(self.x1), np.array(self.y1), np.array(self.T_C)


# --------------------------------------------------------------------------- #
# Reference datasets (isobaric, 760 mmHg)
# --------------------------------------------------------------------------- #
BENZENE_TOLUENE = VLEDataset(
    system="Benzene–Toluene",
    comp1_key="benzene", comp2_key="toluene", pressure_mmHg=760.0,
    x1=(0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0),
    y1=(0.0, 0.208, 0.372, 0.507, 0.612, 0.713, 0.791, 0.857, 0.912, 0.959, 1.0),
    T_C=(110.6, 106.1, 102.2, 98.6, 95.2, 92.1, 89.4, 86.8, 84.4, 82.3, 80.1),
    source="Benzene–toluene VLE, 760 mmHg (near-ideal; Perry's CEH; Gmehling/DECHEMA).",
    note="Near-ideal system — Raoult's law alone is expected to predict it well.",
)

ETHANOL_WATER = VLEDataset(
    system="Ethanol–Water",
    comp1_key="ethanol", comp2_key="water", pressure_mmHg=760.0,
    x1=(0.0, 0.019, 0.0721, 0.0966, 0.1238, 0.1661, 0.2337, 0.3273, 0.3965,
        0.5079, 0.5732, 0.6763, 0.7472, 0.8943, 1.0),
    y1=(0.0, 0.170, 0.3891, 0.4375, 0.4704, 0.5089, 0.5445, 0.5826, 0.6122,
        0.6564, 0.6841, 0.7385, 0.7815, 0.8943, 1.0),
    T_C=(100.0, 95.5, 89.0, 86.7, 85.3, 84.1, 82.7, 81.5, 80.7,
         79.8, 79.3, 78.74, 78.41, 78.15, 78.4),
    source="Ethanol–water VLE, 760 mmHg (Carey & Lewis 1932; Gmehling/DECHEMA).",
    note="Strong positive deviation; minimum-boiling azeotrope at x₁≈0.894, 78.15 °C.",
)

ACETONE_METHANOL = VLEDataset(
    system="Acetone–Methanol",
    comp1_key="acetone", comp2_key="methanol", pressure_mmHg=760.0,
    x1=(0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0),
    y1=(0.0, 0.102, 0.186, 0.257, 0.318, 0.428, 0.513, 0.586, 0.656, 0.725,
        0.800, 0.898, 1.0),
    T_C=(64.7, 63.6, 62.5, 61.6, 60.8, 59.4, 58.3, 57.4, 56.7, 56.1, 55.5, 55.6, 56.05),
    source="Acetone–methanol VLE, 760 mmHg (Gmehling/DECHEMA; Perry's CEH).",
    note="Positive deviation; minimum-boiling azeotrope at x₁≈0.80, 55.5 °C.",
)

DATASETS = [BENZENE_TOLUENE, ETHANOL_WATER, ACETONE_METHANOL]
DATASETS_BY_SYSTEM = {d.system: d for d in DATASETS}


# --------------------------------------------------------------------------- #
# Prediction + metrics
# --------------------------------------------------------------------------- #
def predict_bubble(model: ThermodynamicModel, x1: np.ndarray, pressure_mmHg: float):
    """Predicted (T_C, y1) at each liquid composition via the bubble point."""
    t_pred, y_pred = np.empty(len(x1)), np.empty(len(x1))
    for i, xi in enumerate(x1):
        t, y = model.bubble_temperature([xi, 1.0 - xi], pressure_mmHg, "mmHg", "Celsius")
        t_pred[i], y_pred[i] = t, y[0]
    return t_pred, y_pred


def metrics(experimental: np.ndarray, predicted: np.ndarray) -> dict:
    """MAE, RMSE, and mean absolute percent error (over non-zero references)."""
    exp = np.asarray(experimental, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    err = pred - exp
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    nz = np.abs(exp) > 1e-9
    mpe = float(np.mean(np.abs(err[nz] / exp[nz])) * 100.0) if np.any(nz) else float("nan")
    return {"MAE": mae, "RMSE": rmse, "MPE": mpe}


def _badge(mae: float, thresholds: tuple[float, float, float]) -> str:
    excellent, good, fair = thresholds
    if mae < excellent:
        return "Excellent"
    if mae < good:
        return "Good"
    if mae < fair:
        return "Fair"
    return "Poor"


# MAE thresholds: y₁ in mole fraction, T in °C.
Y1_THRESHOLDS = (0.01, 0.02, 0.05)
T_THRESHOLDS = (0.5, 1.0, 2.0)


@dataclass
class ValidationResult:
    dataset: VLEDataset
    model_name: str
    x1: np.ndarray
    y1_exp: np.ndarray
    y1_pred: np.ndarray
    T_exp: np.ndarray
    T_pred: np.ndarray
    y1_metrics: dict
    T_metrics: dict

    @property
    def y1_badge(self) -> str:
        return _badge(self.y1_metrics["MAE"], Y1_THRESHOLDS)

    @property
    def T_badge(self) -> str:
        return _badge(self.T_metrics["MAE"], T_THRESHOLDS)


def validate(model: ThermodynamicModel, dataset: VLEDataset,
             model_name: str = "") -> ValidationResult:
    """Score ``model`` against ``dataset``."""
    x1, y1_exp, T_exp = dataset.as_arrays()
    T_pred, y1_pred = predict_bubble(model, x1, dataset.pressure_mmHg)
    return ValidationResult(
        dataset=dataset, model_name=model_name or model.activity.name,
        x1=x1, y1_exp=y1_exp, y1_pred=y1_pred, T_exp=T_exp, T_pred=T_pred,
        y1_metrics=metrics(y1_exp, y1_pred), T_metrics=metrics(T_exp, T_pred),
    )
