from pydantic import BaseModel, Field


class ColumnProfile(BaseModel):
    name: str
    inferred_type: str
    dtype: str
    n_missing: int = Field(ge=0)
    n_unique: int = Field(ge=0)


class DatasetProfile(BaseModel):
    n_rows: int = Field(ge=0)
    n_columns: int = Field(ge=0)
    columns: list[ColumnProfile]