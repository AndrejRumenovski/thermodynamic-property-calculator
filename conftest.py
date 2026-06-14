"""Ensure the repository root is importable so ``import thermo`` works in tests
regardless of how pytest is invoked."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
