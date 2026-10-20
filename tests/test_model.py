import pytest
import pandas as pd
from src.data_preprocessing import preprocess_data, download_data

def test_data_loading():
    df = download_data()
    assert not df.empty, "Dataset should not be empty"
    assert len(df.columns) >= 14, "Dataset should have at least 14 columns"

def test_preprocessing_shapes():
    df = download_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(X_train.columns) == 18, f"Expected 18 columns after one-hot encoding, got {len(X_train.columns)}"