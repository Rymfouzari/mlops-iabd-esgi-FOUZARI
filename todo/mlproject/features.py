"""Construction du pre-processing."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mlproject.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Construit le pre-processing a partir des colonnes configurees."""
    transformers = []

    if NUMERIC_FEATURES:
        transformers.append(("num", StandardScaler(), NUMERIC_FEATURES))

    if CATEGORICAL_FEATURES:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)
        )

    if not transformers:
        raise ValueError(
            "Aucune colonne configuree. "
            "Renseigner NUMERIC_FEATURES et/ou CATEGORICAL_FEATURES dans config.py."
        )

    return ColumnTransformer(transformers=transformers)