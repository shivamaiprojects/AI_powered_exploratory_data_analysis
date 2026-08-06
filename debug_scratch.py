from eda_agent.analysis.profiling import profile_dataset
from eda_agent.config.settings import get_settings
from eda_agent.ingestion.loader import load_dataset

settings = get_settings()
frame = load_dataset(settings.data_raw_dir / "adult.csv")

for column in frame.columns:
    if frame[column].dtype == "object" or frame[column].dtype == "string":
        frame[column] = frame[column].str.strip().str.rstrip(".")

profile = profile_dataset(frame)

for column in profile.columns:
    if column.numerical:
        stats = column.numerical
        print(f"{column.name:16s} mean={stats.mean:>12.2f}  median={stats.median:>10.2f}  skew={stats.skewness:>6.2f}")
    elif column.categorical:
        stats = column.categorical
        print(f"{column.name:16s} categories={stats.n_categories:>3d}  top='{stats.most_frequent}' ({stats.most_frequent_count})")