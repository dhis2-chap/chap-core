from pathlib import Path


def seed_example_datasets(session_wrapper):
    base_path = Path(__file__).parent.parent.parent / "example_data"

    datasets = {"example_data1": {"data": base_path / "v0/training_data.csv", "geojson": None}}

    for name, paths in datasets.items():
        if not session_wrapper.get_dataset_by_name(name):
            print(f"Seeding dataset: {name}")
            session_wrapper.add_dataset_from_csv(
                name,
                paths["data"],
                paths["geojson"],
            )
