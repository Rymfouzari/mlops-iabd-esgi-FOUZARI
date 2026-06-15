import pandas as pd

from mlproject.config import (
    CATEGORICAL_FEATURES,
    DATA_PATH,
    NUMERIC_FEATURES,
    TARGET,
)
from mlproject.data import load_data
from mlproject.features import build_preprocessor


def test_dataset_file_exists():
    assert DATA_PATH.exists()


def test_load_data_returns_dataframe():
    df = load_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_target_is_binary():
    df = load_data()
    assert TARGET in df.columns
    assert set(df[TARGET].dropna().unique()).issubset({0, 1})


def test_configured_features_exist_in_dataset():
    df = load_data()
    configured_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    assert configured_features
    assert TARGET not in configured_features

    for column in configured_features:
        assert column in df.columns


def test_preprocessor_can_be_built():
    preprocessor = build_preprocessor()
    assert preprocessor is not None