# Thermodynamic models — equations, parameters, assumptions

This document covers the activity-coefficient framework added in the non-ideal
VLE increment: the models, how their parameters are obtained, the phase-equilibrium
algorithms, and the assumptions and limits of validity.

## 1. Phase-equilibrium closure

All VLE here uses the **modified Raoult's law** (real liquid, ideal vapor):

$$ y_i\,P = x_i\,\gamma_i(\mathbf{x},T)\,P_i^{\text{sat}}(T), \qquad K_i=\frac{\gamma_i P_i^{\text{sat}}}{P} $$

- $P_i^{\text{sat}}$ — Antoine equation, evaluated by `thermo.thermo_engine` in each
  species' native units and converted as needed.
- $\gamma_i$ — liquid activity coefficient from the selected model (below).
- Setting all $\gamma_i = 1$ recovers ideal Raoult's law.

**Assumptions:** ideal vapor phase (no fugacity/Poynting corrections), no
association in the vapor, condensed-phase non-ideality entirely in $\gamma$. Valid
at low-to-moderate pressure, away from the critical region.

## 2. Activity-coefficient models

`R = 1.98720 cal/(mol·K)`; temperatures in K; interaction energies in cal/mol;
`a_ii = 0`.

### Ideal (Raoult) — `thermo/models/ideal.py`
$\gamma_i = 1$.

### Wilson (1964) — `thermo/models/wilson.py`
$$ \Lambda_{ij}=\frac{V_j}{V_i}\exp\!\left(-\frac{a_{ij}}{RT}\right),\quad
\ln\gamma_i = 1-\ln\!\Big(\textstyle\sum_j x_j\Lambda_{ij}\Big)-\sum_k\frac{x_k\Lambda_{ki}}{\sum_j x_j\Lambda_{kj}} $$
$V_i$ = pure-liquid molar volume. Cannot represent liquid–liquid splitting.

### NRTL (Renon & Prausnitz, 1968) — `thermo/models/nrtl.py`
$$ \tau_{ij}=\frac{a_{ij}}{RT},\quad G_{ij}=\exp(-\alpha_{ij}\tau_{ij}),\quad
\ln\gamma_i=\frac{\sum_j\tau_{ji}G_{ji}x_j}{\sum_k G_{ki}x_k}
+\sum_j\frac{x_jG_{ij}}{\sum_k G_{kj}x_k}\!\left(\tau_{ij}-\frac{\sum_m x_m\tau_{mj}G_{mj}}{\sum_k G_{kj}x_k}\right) $$
$\alpha_{ij}=\alpha_{ji}=0.3$ (recommended for the systems used here). Handles a
wide range of non-ideality including partial miscibility.

### UNIQUAC (Abrams & Prausnitz, 1975) — `thermo/models/uniquac.py`
Combinatorial (size/shape) + residual (energetic), with $z=10$, structural
$r_i,q_i$, and $\tau_{ij}=\exp(-a_{ij}/RT)$. See the module docstring for the full
expression. Only two adjustable binary parameters regardless of molecule size.

All three are thermodynamically consistent Gibbs-excess models; the test suite
verifies the Gibbs–Duhem area test $\int_0^1\ln(\gamma_1/\gamma_2)\,dx_1=0$.

## 3. Binary parameters and their provenance

At an azeotrope $y_i=x_i$, so modified Raoult's law gives the exact identity
$\gamma_i = P/P_i^{\text{sat}}(T_\text{az})$. The two binary parameters of each
model are obtained (`thermo/models/parameters.py`) by solving for the values that
make the model reproduce $(\gamma_1,\gamma_2)$ at the **literature azeotrope**:

| System | x₁ (azeotrope) | T (1 atm) | Source |
|---|---|---|---|
| ethanol (1) – water (2) | 0.894 | 78.15 °C | 95.6 wt% EtOH; Gmehling, *Azeotropic Data*; NIST |
| acetone (1) – methanol (2) | 0.800 | 55.5 °C | ~0.80 mol acetone; Gmehling; NIST |

This reproduces the real azeotrope by construction. As an **independent** check
(not fitted), the resulting infinite-dilution coefficient
$\gamma^\infty_\text{ethanol}\approx5$ (NRTL/UNIQUAC) matches literature (~4–6).

Structural parameters (Poling, Prausnitz & O'Connell, 5th ed.):

| Species | r | q | V (cm³/mol) |
|---|---|---|---|
| water | 0.9200 | 1.4000 | 18.07 |
| methanol | 1.4311 | 1.4320 | 40.73 |
| ethanol | 2.1055 | 1.9720 | 58.68 |
| acetone | 2.5735 | 2.3360 | 74.05 |
| benzene | 3.1878 | 2.4000 | 89.41 |
| toluene | 3.9228 | 2.9680 | 106.85 |

> Limitation: a single-azeotrope fit pins the azeotrope and mid-range behavior;
> the dilute extrapolation ($\gamma^\infty$) is model-dependent. Regressing full
> T–x–y datasets (independent literature validation, MAE/RMSE) is the Phase 2
> increment.

## 4. Algorithms — `thermo/models/vle.py`

- **Bubble P** (x, T): direct, $P=\sum_i x_i\gamma_i P_i^{\text{sat}}$.
- **Bubble T** (x, P): 1-D Brent root on $\sum_i x_i\gamma_i(x,T)P_i^{\text{sat}}(T)-P$.
- **Dew P** (y, T): fixed-point iteration on the liquid composition (γ depends on
  the unknown x).
- **Flash** (z, T, P): regime from bubble/dew pressure, then successive
  substitution — Rachford-Rice (`thermo.flash_engine.solve_vapor_fraction`) for β,
  update γ(x, T) until x converges.

## 5. Phase diagrams — `thermo/diagrams.py`

`txy` / `pxy` sweep x₁∈[0,1] and evaluate the bubble point. Plotting T (or P)
against x₁ traces the bubble line and against y₁ the dew line. Exports: PNG/SVG
(chart modebar) and CSV.

## References
1. G. M. Wilson, *J. Am. Chem. Soc.* **86** (1964) 127.
2. H. Renon, J. M. Prausnitz, *AIChE J.* **14** (1968) 135.
3. D. S. Abrams, J. M. Prausnitz, *AIChE J.* **21** (1975) 116.
4. Poling, Prausnitz, O'Connell, *The Properties of Gases and Liquids*, 5th ed.
5. Gmehling et al., *Azeotropic Data*; NIST Chemistry WebBook.
