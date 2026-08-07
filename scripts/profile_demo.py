from eda_agent.analysis.profiling import profile_dataset
from eda_agent.config.settings import get_settings
from eda_agent.ingestion.loader import load_dataset


def clean_text_columns(frame):
    for column in frame.columns:
        if frame[column].dtype == "object" or frame[column].dtype == "string":
            frame[column] = frame[column].str.strip().str.rstrip(".")
    return frame


def main() -> None:
    settings = get_settings()
    frame = load_dataset(settings.data_raw_dir / "adult.csv")
    frame = clean_text_columns(frame)
    profile = profile_dataset(frame)

    print(f"Dataset: {profile.n_rows} rows x {profile.n_columns} columns\n")
    for column in profile.columns:
        if column.numerical:
            stats = column.numerical
            print(
                f"{column.name:16s} [num] mean={stats.mean:>12.2f}  "
                f"median={stats.median:>10.2f}  skew={stats.skewness:>6.2f}  "
                f"missing={column.missing_pct}%"
            )
        elif column.categorical:
            stats = column.categorical
            print(
                f"{column.name:16s} [cat] categories={stats.n_categories:>3d}  "
                f"top='{stats.most_frequent}' ({stats.most_frequent_count})  "
                f"missing={column.missing_pct}%"
            )


if __name__ == "__main__":
    main()