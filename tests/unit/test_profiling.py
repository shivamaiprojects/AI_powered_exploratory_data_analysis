import pandas as pd

from eda_agent.analysis.profiling import profile_categorical, profile_numerical


def test_median_is_robust_to_outlier():
    normal = pd.Series([10, 11, 12, 13, 14])
    with_outlier = pd.Series([10, 11, 12, 13, 10000])
    base = profile_numerical(normal)
    shifted = profile_numerical(with_outlier)
    assert abs(shifted.median - base.median) <= 1
    assert shifted.mean - base.mean > 1000


def test_iqr_is_robust_to_outlier():
    normal = pd.Series([10, 11, 12, 13, 14])
    with_outlier = pd.Series([10, 11, 12, 13, 10000])
    base = profile_numerical(normal)
    shifted = profile_numerical(with_outlier)
    assert abs(shifted.iqr - base.iqr) <= 1
    assert shifted.std - base.std > 1000


def test_right_skew_is_positive():
    series = pd.Series([1, 1, 1, 1, 2, 2, 3, 100])
    stats = profile_numerical(series)
    assert stats.skewness > 1


def test_constant_column_has_undefined_skew():
    series = pd.Series([5, 5, 5, 5])
    stats = profile_numerical(series)
    assert stats.skewness is None
    assert stats.std == 0.0


def test_categorical_finds_most_frequent():
    series = pd.Series(["a", "b", "a", "a", "c"])
    stats = profile_categorical(series)
    assert stats.most_frequent == "a"
    assert stats.most_frequent_count == 3
    assert stats.n_categories == 3