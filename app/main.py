"""Unified entrypoint: `streamlit run app/main.py`.

Runs all three views (Model Results Lab, Fundamental triage, Medio triage)
as pages of one app with a single sidebar, instead of three separate
`streamlit run` processes. The individual entrypoints
(app/fundamental/main.py, app/medio/main.py, app/model_results/main.py)
still work standalone for anyone with muscle memory or a script depending
on them; this file does not replace them, it adds one place to reach all
three.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

st.set_page_config(page_title="Brazil School Risk", layout="wide")

model_results_page = st.Page(
    "model_results/main.py",
    title="Model Results Lab",
    icon="📊",
    url_path="model-results",
    default=True,
)
fundamental_page = st.Page(
    "fundamental/main.py", title="Fundamental triage", icon="🏫", url_path="fundamental"
)
medio_page = st.Page("medio/main.py", title="Medio triage", icon="🏫", url_path="medio")

nav = st.navigation([model_results_page, fundamental_page, medio_page])
nav.run()
