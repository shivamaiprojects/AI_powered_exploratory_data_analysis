from pathlib import Path

import pandas as pd


class UnsupportedFormatError(ValueError):
    pass


def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No file found at: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path, na_values=["?", "", "NA", "N/A"], skipinitialspace=True)
    elif suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(path, na_values=["?", "", "NA", "N/A"])
    else:
        raise UnsupportedFormatError(f"Unsupported file type: {suffix}")

    if frame.empty:
        raise ValueError("Loaded dataset is empty.")

    return frame