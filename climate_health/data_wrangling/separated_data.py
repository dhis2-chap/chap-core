from omnipy import StrDataset


def load_separated_data(datafile_paths: tuple[str]) -> StrDataset:
    dataset = StrDataset()
    for datafile_path in datafile_paths:
        dataset.load(datafile_path)
    return dataset
