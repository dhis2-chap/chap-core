from chap_core.model_spec import ModelSpec
import chap_core.predictor.feature_spec as fs


class ExternalModelSpec(ModelSpec):
    github_link: str


base_features = [fs.rainfall, fs.mean_temperature, fs.population]

# NB: This is maybe outdated and not being used?
# Rest api uses db directly and is populated in  seed_with_session_wrapper
