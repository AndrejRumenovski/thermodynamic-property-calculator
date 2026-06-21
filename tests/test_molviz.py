"""Tests for the molecular-visualization helpers (RDKit-backed)."""

from __future__ import annotations

import pytest

pytest.importorskip("rdkit")  # skip cleanly if RDKit is unavailable

from thermo import molviz  # noqa: E402


def test_analyze_ethanol():
    info = molviz.analyze("CCO")
    assert info.formula == "C2H6O"
    assert info.molar_mass == pytest.approx(46.069, abs=0.01)
    assert info.heavy_atoms == 3
    assert info.total_atoms == 9          # 3 heavy + 6 H
    assert info.atom_counts == {"C": 2, "H": 6, "O": 1}
    assert info.rings == 0


def test_benzene_is_aromatic_ring():
    info = molviz.analyze("c1ccccc1")
    assert info.formula == "C6H6"
    assert info.rings == 1
    assert info.atom_counts["C"] == 6 and info.atom_counts["H"] == 6


def test_2d_svg_and_3d_molblock_generated():
    info = molviz.analyze("CC(=O)C")            # acetone
    assert "<svg" in info.svg_2d.lower()
    assert "V2000" in info.molblock_3d           # MolBlock with 3D coords
    # MolBlock atom block should contain non-trivial 3D coordinates
    assert info.molblock_3d.count("\n") > 10


def test_invalid_smiles_raises():
    with pytest.raises(molviz.MolVizError):
        molviz.analyze("this-is-not-smiles((")


def test_viewer_html_embeds_molblock_and_3dmol():
    info = molviz.analyze("CO")
    html = molviz.viewer_html(info.molblock_3d, show_labels=True, style="Stick")
    assert "3Dmol-min.js" in html
    assert "addModel" in html
    assert "addPropertyLabels" in html          # labels requested


def test_library_smiles_all_parse():
    for name, smiles in molviz.MOLECULE_SMILES.items():
        info = molviz.analyze(smiles)
        assert info.heavy_atoms >= 1, name
