from pydantic import BaseModel


class NumericalStats(BaseModel):
    mean: float | None
    median: float | None
    std: float | None
    minimum: float | None
    maximum: float | None
    q1: float | None
    q3: float | None
    iqr: float | None
    skewness: float | None
    kurtosis: float | None


class CategoricalStats(BaseModel):
    n_categories: int
    most_frequent: str | None
    most_frequent_count: int
    top_values: dict[str, int]