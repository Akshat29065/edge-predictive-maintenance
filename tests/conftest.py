"""Lets the tests import the service modules, which use flat imports (`from tools import ...`)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for sub in ("agent", "analytics", "vision"):
    sys.path.insert(0, str(ROOT / sub))
