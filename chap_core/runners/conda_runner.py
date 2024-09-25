import yaml
from chap_core.runners.runner import Runner


class CondaRunner(Runner):
    def __init__(self, conda_yaml_file: str):
        pass


def _get_conda_environment_name():
    """Returns a name that is a hash of the content of the conda env file, so that identical file
    gives same name and changes in the file leads to new name"""
    with open(str(self._working_dir / self._conda_env_file), "r") as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
        # convert to json to avoid minor changes affecting the hash
        checksum = md5(json.dumps(data).encode("utf-8")).hexdigest()
        return f"{self._name}_{checksum}"
