"""Streamlit entry point.

Run from the repository root with::

    streamlit run app.py

Kept deliberately thin — all UI logic lives in :mod:`thermo.interface`.
"""

from thermo.interface import render

render()
