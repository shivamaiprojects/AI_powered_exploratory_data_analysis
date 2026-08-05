from pathlib import Path

from ucimlrepo import fetch_ucirepo

from eda_agent.config.settings import get_settings


def download_adult(destination: Path) -> Path:
    dataset = fetch_ucirepo(id=2)
    frame = dataset.data.original
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return destination


def main() -> None:
    settings = get_settings()
    target = settings.data_raw_dir / "adult.csv"
    path = download_adult(target)
    print(f"Saved {path} with shape {__import__('pandas').read_csv(path).shape}")


if __name__ == "__main__":
    main()