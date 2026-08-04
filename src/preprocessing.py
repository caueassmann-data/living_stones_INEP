"""Preprocessing pipelines for Brazil school-level abandonment models."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_model_pipeline(estimator, feature_columns: list[str]) -> Pipeline:
    """Impute + scale numeric school features, then fit estimator."""
    prep = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                feature_columns,
            )
        ],
        remainder="drop",
    )
    return Pipeline(steps=[("prep", prep), ("model", estimator)])
