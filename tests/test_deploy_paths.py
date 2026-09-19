"""The mart-path split that lets a hosted deployment run from a plain clone.

The full marts are gigabytes of gitignored ETL output, so the deployed app
reads a slim committed copy instead. These tests pin the two rules that keep
that safe: the app falls back, and training never does.
"""

from __future__ import annotations

import pandas as pd

from src import utils
from src.utils import app_mart_path, mart_path, resolve_app_mart_path


def test_app_prefers_the_full_mart_when_it_exists(tmp_path, monkeypatch):
    full = tmp_path / "marts" / "school_risk_br_fundamental"
    full.mkdir(parents=True)
    (full / "school_risk_br_fundamental.parquet").write_bytes(b"")
    monkeypatch.setattr(utils, "MARTS_ROOT", tmp_path / "marts")
    monkeypatch.setattr(utils, "APP_MARTS_ROOT", tmp_path / "marts_app")

    assert resolve_app_mart_path("fundamental") == utils.mart_path("fundamental")


def test_app_falls_back_to_the_slim_mart_when_the_full_one_is_absent(tmp_path, monkeypatch):
    """This is the deployed case: a fresh clone has no full mart at all."""
    monkeypatch.setattr(utils, "MARTS_ROOT", tmp_path / "marts")
    monkeypatch.setattr(utils, "APP_MARTS_ROOT", tmp_path / "marts_app")

    assert resolve_app_mart_path("medio") == utils.app_mart_path("medio")


def test_training_path_never_resolves_to_the_slim_mart():
    """src/data_loader.py::load_school_mart uses mart_path, not the resolver.
    If that ever changed, training would silently fit on a two-year window."""
    for level in ("fundamental", "medio"):
        assert mart_path(level) != app_mart_path(level)
        assert "marts_app" not in str(mart_path(level))


def test_slim_mart_carries_the_columns_the_app_needs():
    """Guards scripts/build_deploy_artifacts.py against dropping a column the
    prioritization UI filters on — which would only surface after deploying."""
    path = app_mart_path("fundamental")
    if not path.exists():
        import pytest

        pytest.skip("Deploy artifacts not built; run scripts/build_deploy_artifacts.py")

    slim = pd.read_parquet(path)
    required = {
        "year",
        "school_id",
        "school_name",
        "state_code",
        "municipality_id",
        "municipality_name",
        "admin_dependency_type",
        "enrollment_level",
        "target_dropout_rate",
        "dropout_rate_lag1",
        "has_history",
    }
    assert required <= set(slim.columns)
    assert len(slim) > 0
