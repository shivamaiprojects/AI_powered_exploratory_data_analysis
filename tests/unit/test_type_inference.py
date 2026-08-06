import pandas as pd

from eda_agent.ingestion.type_inference import infer_column_type


def test_high_cardinality_numeric_is_numerical():
    series = pd.Series(range(1000))
    assert infer_column_type(series) == "numerical"


def test_low_cardinality_numeric_is_categorical():
    series = pd.Series([0, 1] * 500)
    assert infer_column_type(series) == "categorical"


def test_zip_codes_are_categorical():
    zips = pd.Series(["12345", "67890", "54321"] * 100)
    assert infer_column_type(zips) == "categorical"


def test_text_labels_are_categorical():
    series = pd.Series(["Male", "Female"] * 500)
    assert infer_column_type(series) == "categorical"


def test_date_strings_are_datetime():
    dates = pd.Series(["2024-01-01", "2024-02-15", "2024-03-20"] * 100)
    assert infer_column_type(dates) == "datetime"


def test_all_missing_is_categorical():
    series = pd.Series([None] * 100)
    assert infer_column_type(series) == "categorical"