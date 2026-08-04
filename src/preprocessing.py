"""Preprocessing pipeline for Brazil school-level dropout-rate models.

Two kinds of input columns need different treatment:
  - Numeric/binary features (enrollment counts, infrastructure flags, lagged
    dropout-rate history, ...): missing values are imputed with the column
    median, then standardized.
  - Categorical/nominal codes (admin_dependency_type, location_type,
    state_code): these are category *codes*, not ordered magnitudes — e.g.
    admin_dependency_type 3 ("Municipal") is not "more" of anything than 1
    ("Federal") — so they are one-hot encoded rather than scaled like a
    number. Encoding these as raw integers (as earlier versions of this
    pipeline did) implicitly tells linear models "3 is three times 1," which
    is meaningless for a category code.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_model_pipeline(
    estimator, numeric_columns: list[str], categorical_columns: list[str] | None = None
) -> Pipeline:
    """Impute + encode school features, then fit estimator."""
    categorical_columns = categorical_columns or []
    transformers = [
        (
            "num",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numeric_columns,
        )
    ]
    if categorical_columns:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            )
        )
    prep = ColumnTransformer(transformers=transformers, remainder="drop")
    return Pipeline(steps=[("prep", prep), ("model", estimator)])
