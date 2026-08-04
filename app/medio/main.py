"""Streamlit app — Ensino Médio school abandonment early-warning."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ui_common import run_app

run_app("medio", "Brazil School Risk — Ensino Médio")
