from pydantic import BaseModel, Field

from eda_agent.schemas.statistics import CategoricalStats, NumericalStats


class ColumnProfile(BaseModel):
    name: str
    inferred_type: str
    dtype: str
    n_missing: int = Field(ge=0)
    missing_pct: float = Field(ge=0)
    n_unique: int = Field(ge=0)
    numerical: NumericalStats | None = None
    categorical: CategoricalStats | None = None


class DatasetProfile(BaseModel):
    n_rows: int = Field(ge=0)
    n_columns: int = Field(ge=0)
    columns: list[ColumnProfile]