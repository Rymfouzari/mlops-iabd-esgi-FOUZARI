"""Chargement et decoupage des donnees."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from mlproject.config import DATA_PATH, RANDOM_STATE, TARGET


def load_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Charge le fichier CSV configure dans config.py."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Fichier de donnees introuvable : {path}")

    df = pd.read_csv(path)

    if TARGET not in df.columns:
        raise ValueError(f"Colonne cible introuvable dans le dataset : {TARGET}")

    target_values = set(df[TARGET].dropna().unique())
    if not target_values.issubset({0, 1}):
        raise ValueError(
            f"La colonne cible doit contenir uniquement 0/1. "
            f"Valeurs trouvees : {sorted(target_values)}"
        )

    return df


def split(df: pd.DataFrame, test_size: float = 0.2):
    """Separe les donnees en train/test avec stratification sur la cible."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )