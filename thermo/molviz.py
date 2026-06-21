"""Molecular visualization — 2D depiction (RDKit) and interactive 3D (3Dmol.js).

RDKit parses a SMILES string, computes the formula / molar mass / atom counts,
renders a 2D structure as SVG, and embeds + force-field-optimises a 3D
conformer (ETKDG + MMFF) exported as a MolBlock. The MolBlock is handed to
3Dmol.js (loaded in the browser) for an interactive viewer with rotate, zoom,
and atom labels.
"""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

# Common molecules → SMILES (names align with the Joback library where possible
# so predicted properties can be cross-linked).
MOLECULE_SMILES: dict[str, str] = {
    "Water": "O",
    "Methanol": "CO",
    "Ethanol": "CCO",
    "1-Propanol": "CCCO",
    "n-Pentane": "CCCCC",
    "n-Hexane": "CCCCCC",
    "n-Heptane": "CCCCCCC",
    "n-Octane": "CCCCCCCC",
    "Cyclohexane": "C1CCCCC1",
    "Benzene": "c1ccccc1",
    "Toluene": "Cc1ccccc1",
    "Acetone": "CC(=O)C",
    "Diethyl ether": "CCOCC",
    "Dimethyl ether": "COC",
    "Acetic acid": "CC(=O)O",
}

_STYLE_JS = {
    "Ball & stick": "{stick:{radius:0.13}, sphere:{scale:0.25}}",
    "Stick": "{stick:{radius:0.15}}",
    "Space-filling": "{sphere:{scale:0.9}}",
}


class MolVizError(ValueError):
    """Raised when a SMILES string cannot be parsed or embedded."""


@dataclass(frozen=True)
class MoleculeInfo:
    smiles: str
    formula: str
    molar_mass: float       # g/mol
    heavy_atoms: int
    total_atoms: int        # including explicit hydrogens
    rings: int
    atom_counts: dict[str, int]
    svg_2d: str
    molblock_3d: str


def parse_smiles(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles((smiles or "").strip())
    if mol is None:
        raise MolVizError(f"Could not parse SMILES {smiles!r}.")
    return mol


def _atom_counts(mol_with_h: Chem.Mol) -> dict[str, int]:
    counts: dict[str, int] = {}
    for atom in mol_with_h.GetAtoms():
        sym = atom.GetSymbol()
        counts[sym] = counts.get(sym, 0) + 1
    return dict(sorted(counts.items()))


def to_2d_svg(mol: Chem.Mol, size: tuple[int, int] = (380, 260)) -> str:
    drawer = rdMolDraw2D.MolDraw2DSVG(*size)
    try:
        rdMolDraw2D.SetDarkMode(drawer)  # light bonds on a dark background
    except Exception:  # noqa: BLE001 - older RDKit lacks SetDarkMode
        pass
    drawer.drawOptions().clearBackground = False
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def _embed_3d(mol: Chem.Mol) -> Chem.Mol:
    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xF00D
    if AllChem.EmbedMolecule(mol_h, params) != 0:
        AllChem.EmbedMolecule(mol_h, useRandomCoords=True, randomSeed=1)
    try:
        AllChem.MMFFOptimizeMolecule(mol_h)
    except Exception:  # noqa: BLE001 - MMFF may not cover every atom type
        pass
    return mol_h


def analyze(smiles: str) -> MoleculeInfo:
    mol = parse_smiles(smiles)
    mol_h = _embed_3d(mol)
    return MoleculeInfo(
        smiles=smiles.strip(),
        formula=rdMolDescriptors.CalcMolFormula(mol),
        molar_mass=round(Descriptors.MolWt(mol), 3),
        heavy_atoms=mol.GetNumAtoms(),
        total_atoms=mol_h.GetNumAtoms(),
        rings=rdMolDescriptors.CalcNumRings(mol),
        atom_counts=_atom_counts(mol_h),
        svg_2d=to_2d_svg(mol),
        molblock_3d=Chem.MolToMolBlock(mol_h),
    )


def viewer_html(molblock: str, show_labels: bool = False,
                style: str = "Ball & stick", height: int = 430) -> str:
    """Return self-contained HTML embedding a 3Dmol.js viewer for ``molblock``."""
    safe = molblock.replace("\\", "\\\\").replace("`", "\\`")
    style_js = _STYLE_JS.get(style, _STYLE_JS["Ball & stick"])
    labels_js = (
        "viewer.addPropertyLabels('elem', {}, {fontColor:'white', fontSize:11,"
        " backgroundColor:'0x15203A', backgroundOpacity:0.65});" if show_labels else ""
    )
    return f"""<div id="mv" style="width:100%;height:{height}px;position:relative;
border-radius:10px;overflow:hidden;border:1px solid #273656;"></div>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<script>
(function(){{
  var run=function(){{
    var el=document.getElementById('mv');
    var viewer=$3Dmol.createViewer(el,{{backgroundColor:'0x0E1726'}});
    viewer.addModel(`{safe}`,'sdf');
    viewer.setStyle({{}}, {style_js});
    {labels_js}
    viewer.zoomTo(); viewer.render(); viewer.zoom(1.15,800);
  }};
  if(window.$3Dmol){{run();}}else{{
    var t=setInterval(function(){{if(window.$3Dmol){{clearInterval(t);run();}}}},60);
  }}
}})();
</script>"""
