from chap_core.model_spec import ModelSpec, PeriodType
import chap_core.predictor.feature_spec as fs


class ExternalModelSpec(ModelSpec):
    github_link: str


base_features = [fs.rainfall, fs.mean_temperature, fs.population]

models = (
    ExternalModelSpec(
        name='chap_ewars_monthly',
        parameters={},
        features=base_features,
        period=PeriodType.month,
        description='Monthly EWARS model',
        author='CHAP',
        github_link="https://github.com/sandvelab/chap_auto_ewars")
    ,
    ExternalModelSpec(
        name='chap_ewars_weekly',
        parameters={},
        features=base_features,
        period=PeriodType.week,
        description='Weekly EWARS model',
        author='CHAP',
        github_link="https://github.com/sandvelab/chap_auto_ewars_weekly"),
    ExternalModelSpec(
        name='auto_regressive_weekly',
        parameters={},
        features=base_features,
        period=PeriodType.week,
        description='Weekly Deep Auto Regressive model',
        author='knutdrand',
        github_link='https://github.com/knutdrand/weekly_ar_model'),
    ExternalModelSpec(
        name='auto_regressive_monthly',
        parameters={},
        features=base_features,
        period=PeriodType.month,
        description='Monthly Deep Auto Regressive model',
        author='knutdrand',
        github_link='https://github.com/sandvelab/monthly_ar_model')
)

model_dict = {model.name: model for model in models}
