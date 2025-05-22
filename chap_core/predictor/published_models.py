from chap_core.model_spec import ModelSpec, PeriodType
import chap_core.predictor.feature_spec as fs


class ExternalModelSpec(ModelSpec):
    github_link: str


base_features = [fs.rainfall, fs.mean_temperature, fs.population]

# NB: This is maybe outdated and not being used?
# Rest api uses db directly and is populated in  seed_with_session_wrapper

models = (
    ExternalModelSpec(
        name="chap_ewars_monthly",
        parameters={},
        features=base_features,
        period=PeriodType.month,
        description="Monthly EWARS model",
        author="CHAP",
        github_link="https://github.com/sandvelab/chap_auto_ewars@58d56f86641f4c7b09bbb635afd61740deff0640",
    ),
    ExternalModelSpec(
        name="chap_ewars_weekly",
        parameters={},
        features=base_features,
        period=PeriodType.week,
        description="Weekly EWARS model",
        author="CHAP",
        github_link="https://github.com/sandvelab/chap_auto_ewars_weekly@737446a7accf61725d4fe0ffee009a682e7457f6",
    ),
    ExternalModelSpec(
        name="auto_regressive_weekly",
        parameters={},
        features=base_features,
        period=PeriodType.week,
        description="Weekly Deep Auto Regressive model",
        author="knutdrand",
        github_link="https://github.com/knutdrand/weekly_ar_model@1730b26996201d9ee0faf65695f44a2410890ea5",
    ),
    ExternalModelSpec(
        name="auto_regressive_monthly",
        parameters={},
        features=base_features,
        period=PeriodType.month,
        description="Monthly Deep Auto Regressive model",
        author="knutdrand",
        github_link="https://github.com/sandvelab/monthly_ar_model@89f070dbe6e480d1e594e99b3407f812f9620d6d",
    ),
)

model_dict = {model.name: model for model in models}
