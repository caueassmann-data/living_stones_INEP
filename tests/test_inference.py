"""Unit tests for src/inference.py using a tiny in-memory-fitted pipeline.

These do not depend on the real (multi-GB-sourced) trained models: a small
Ridge pipeline is fit on synthetic data and saved to a tmp_path, standing in
for models/{level}/. This exercises the load/predict/score contract that the
Streamlit apps depend on without requiring `python -m src.train` to have run.
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

import src.inference as inference
from src.preprocessing import build_model_pipeline


@pytest.fixture
def fake_model(tmp_path, monkeypatch):
    level = "faketest"
    model_root = tmp_path / level
    model_root.mkdir(parents=True)

    numeric_features = ["enrollment_level", "dropout_rate_lag1"]
    categorical_features = ["state_code"]
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "enrollment_level": rng.normal(200, 50, 30),
            "dropout_rate_lag1": rng.normal(1.0, 0.5, 30),
            "state_code": rng.choice(["SP", "RJ"], 30),
        }
    )
    y = pd.Series(rng.normal(2.0, 1.0, 30))
    pipe = build_model_pipeline(Ridge(alpha=1.0), numeric_features, categorical_features)
    pipe.fit(X, y)
    joblib.dump(pipe, model_root / "pipeline.joblib")
    (model_root / "feature_list.json").write_text(
        json.dumps({"numeric_features": numeric_features, "categorical_features": categorical_features}),
        encoding="utf-8",
    )
    (model_root / "metrics.json").write_text(
        json.dumps({"selected_model": "ridge", "test_metrics": {"mae": 1.23}}), encoding="utf-8"
    )

    monkeypatch.setattr(inference, "model_dir", lambda lvl: model_root)
    inference.load_artifacts.cache_clear()
    yield level
    inference.load_artifacts.cache_clear()


def test_load_artifacts_reads_split_feature_lists(fake_model):
    art = inference.load_artifacts(fake_model)
    assert art["numeric_features"] == ["enrollment_level", "dropout_rate_lag1"]
    assert art["categorical_features"] == ["state_code"]
    assert set(art["features"]) == {"enrollment_level", "dropout_rate_lag1", "state_code"}


def test_predict_frame_adds_pred_dropout_rate_column(fake_model):
    df = pd.DataFrame(
        {"enrollment_level": [100, 200], "dropout_rate_lag1": [0.5, 1.5], "state_code": ["SP", "RJ"]}
    )
    scored = inference.predict_frame(fake_model, df)
    assert "pred_dropout_rate" in scored.columns
    assert len(scored) == 2
    assert np.isfinite(scored["pred_dropout_rate"]).all()


def test_predict_frame_handles_missing_columns_gracefully(fake_model):
    # A caller CSV that is missing a feature column entirely should not crash
    # (the imputer fills it); this mirrors a real "user uploaded a partial CSV" case.
    df = pd.DataFrame({"enrollment_level": [100]})
    scored = inference.predict_frame(fake_model, df)
    assert len(scored) == 1


def test_score_single_returns_expected_contract(fake_model):
    result = inference.score_single(
        fake_model, {"enrollment_level": 150, "dropout_rate_lag1": 1.0, "state_code": "SP"}
    )
    assert "pred_dropout_rate" in result
    assert result["selected_model"] == "ridge"
    assert "limitation" in result


def test_missing_model_raises_actionable_error(tmp_path, monkeypatch):
    monkeypatch.setattr(inference, "model_dir", lambda lvl: tmp_path / "nonexistent")
    inference.load_artifacts.cache_clear()
    with pytest.raises(FileNotFoundError, match="src.train"):
        inference.load_artifacts("nonexistent")
    inference.load_artifacts.cache_clear()
