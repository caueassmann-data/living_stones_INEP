"""Inference helpers for the Brazil school-level dropout-rate models."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import joblib
import pandas as pd

from src.utils import LIMITATION_STATEMENT, MODEL_VERSION, model_dir


@lru_cache(maxsize=None)
def load_artifacts(level: str) -> dict[str, Any]:
    """Load a trained pipeline and its metadata for one education level.

    Cached with lru_cache: the pipeline file can be tens of MB, and without
    caching the Streamlit apps were re-reading it from disk on every
    interaction (see docs/validity_and_english_revision.md).
    """
    d = model_dir(level)
    pipe_path = d / "pipeline.joblib"
    if not pipe_path.exists():
        raise FileNotFoundError(
            f"Missing model for {level}: {pipe_path}. Run python -m src.train --level {level}"
        )
    pipe = joblib.load(pipe_path)
    feature_payload = json.loads((d / "feature_list.json").read_text(encoding="utf-8"))
    numeric_features = feature_payload.get("numeric_features", [])
    categorical_features = feature_payload.get("categorical_features", [])
    features = numeric_features + categorical_features
    metrics = {}
    metrics_path = d / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        "pipeline": pipe,
        "features": features,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "metrics": metrics,
        "model_version": MODEL_VERSION,
        "level": level,
        "limitation": LIMITATION_STATEMENT,
    }


def predict_frame(level: str, df: pd.DataFrame) -> pd.DataFrame:
    art = load_artifacts(level)
    X = df.reindex(columns=art["features"])
    for c in art["numeric_features"]:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    out = df.copy()
    out["pred_dropout_rate"] = art["pipeline"].predict(X)
    return out


def score_single(level: str, row: dict[str, Any]) -> dict[str, Any]:
    df = pd.DataFrame([row])
    pred = float(predict_frame(level, df)["pred_dropout_rate"].iloc[0])
    art = load_artifacts(level)
    return {
        "education_level": level,
        "pred_dropout_rate": pred,
        "model_version": art["model_version"],
        "selected_model": art["metrics"].get("selected_model"),
        "limitation": art["limitation"],
    }
