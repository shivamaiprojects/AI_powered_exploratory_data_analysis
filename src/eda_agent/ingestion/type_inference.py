import pandas as pd
from pandas.api import types as pdt
from eda_agent.schemas.dataset import ColumnProfile, DatasetProfile
import warnings

CATEGORICAL_CARDINALITY_RATIO = 0.05
CATEGORICAL_ABSOLUTE_MAX = 20



def _looks_like_dates(series: pd.Series) -> bool:
    non_null = series.dropna()
    if non_null.empty:
        return False

    sample = non_null.astype(str).head(50)
    has_date_chars = sample.str.contains(r"[-/:]", regex=True).mean()
    if has_date_chars < 0.5:
        return False

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        converted = pd.to_datetime(non_null, errors="coerce")
    return converted.notna().mean() > 0.9

def infer_column_type(series: pd.Series) -> str:
    # Note: low-cardinality integer codes (e.g. education-num) are treated as
    # categorical. They are technically ordinal, but ordinality can't be
    # reliably auto-detected and isn't needed for exploratory analysis.
    if pdt.is_datetime64_any_dtype(series):
        return "datetime"

    if pdt.is_numeric_dtype(series):
        n_unique = series.nunique(dropna=True)
        n_rows = len(series)
        ratio = n_unique / n_rows if n_rows else 0
        if n_unique <= CATEGORICAL_ABSOLUTE_MAX and ratio < CATEGORICAL_CARDINALITY_RATIO:
            return "categorical"
        return "numerical"

    if _looks_like_dates(series):
        return "datetime"

    return "categorical"


def infer_schema(frame: pd.DataFrame) -> dict[str, str]:
    return {column: infer_column_type(frame[column]) for column in frame.columns}

def build_profile(frame: pd.DataFrame) -> DatasetProfile:
    schema = infer_schema(frame)
    columns = [
        ColumnProfile(
            name=name,
            inferred_type=schema[name],
            dtype=str(frame[name].dtype),
            n_missing=int(frame[name].isna().sum()),
            n_unique=int(frame[name].nunique(dropna=True)),
        )
        for name in frame.columns
    ]
    return DatasetProfile(n_rows=len(frame), n_columns=frame.shape[1], columns=columns)