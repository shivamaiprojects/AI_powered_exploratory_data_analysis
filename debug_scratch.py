from eda_agent.ingestion.loader import load_dataset
from eda_agent.ingestion.type_inference import build_profile
from eda_agent.config.settings import get_settings

settings = get_settings()
frame = load_dataset(settings.data_raw_dir / "adult.csv")
profile = build_profile(frame)

for column in profile.columns:
    print(f"{column.name:20s} {column.inferred_type:12s} missing={column.n_missing}")