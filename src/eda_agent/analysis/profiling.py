import numpy as np
import pandas as pd

from eda_agent.ingestion.type_inference import infer_column_type
from eda_agent.schemas.dataset import ColumnProfile, DatasetProfile
from eda_agent.schemas.statistics import CategoricalStats, NumericalStats


def _finite_or_none(value: float) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def profile_numerical(series: pd.Series) -> NumericalStats:
    clean = series.dropna()
    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)
    std = clean.std()
    has_spread = bool(np.isfinite(std) and std > 0)
    return NumericalStats(
        mean=_finite_or_none(clean.mean()),
        median=_finite_or_none(clean.median()),
        std=_finite_or_none(std),
        minimum=_finite_or_none(clean.min()),
        maximum=_finite_or_none(clean.max()),
        q1=_finite_or_none(q1),
        q3=_finite_or_none(q3),
        iqr=_finite_or_none(q3 - q1),
        skewness=_finite_or_none(clean.skew()) if has_spread else None,
        kurtosis=_finite_or_none(clean.kurt()) if has_spread else None,
    )


def profile_categorical(series: pd.Series, top_n: int = 10) -> CategoricalStats:
    counts = series.dropna().value_counts()
    if counts.empty:
        return CategoricalStats(
            n_categories=0,
            most_frequent=None,
            most_frequent_count=0,
            top_values={},
        )
    top = counts.head(top_n)
    return CategoricalStats(
        n_categories=int(counts.size),
        most_frequent=str(counts.index[0]),
        most_frequent_count=int(counts.iloc[0]),
        top_values={str(key): int(value) for key, value in top.items()},
    )


def profile_column(series: pd.Series) -> ColumnProfile:
    inferred = infer_column_type(series)
    profile = ColumnProfile(
        name=str(series.name),
        inferred_type=inferred,
        dtype=str(series.dtype),
        n_missing=int(series.isna().sum()),
        missing_pct=round(float(series.isna().mean() * 100), 2),
        n_unique=int(series.nunique(dropna=True)),
    )
    if inferred == "numerical":
        profile.numerical = profile_numerical(series)
    else:
        profile.categorical = profile_categorical(series)
    return profile


def profile_dataset(frame: pd.DataFrame) -> DatasetProfile:
    columns = [profile_column(frame[column]) for column in frame.columns]
    return DatasetProfile(n_rows=len(frame), n_columns=frame.shape[1], columns=columns)