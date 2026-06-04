from pathlib import Path

from chap_core.database.dataset_manager import DataSetManager


def seed_example_datasets(session_wrapper):
    base_path = Path(__file__).parent.parent.parent / "example_data"

    datasets: list[tuple[str, Path, Path | None]] = [
        ("example_data1", base_path / "v0/training_data.csv", None),
    ]

    manager = DataSetManager(session_wrapper.session)
    for name, data_path, geojson_path in datasets:
        if not manager.find_by_name(name):
            print(f"Seeding dataset: {name}")
            manager.save_dataset_from_csv(name, data_path, geojson_path)
