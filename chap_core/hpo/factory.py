from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import yaml

from chap_core.api_types import SearcherType
from chap_core.hpo.hyperparameter_optimizer import HyperparameterOptimizer
from chap_core.hpo.objective import Objective
from chap_core.hpo.search_space import DEFAULT_HPO_TRIALS, search_space_from_config
from chap_core.hpo.searcher import GridSearcher, RandomSearcher, TPESearcher

if TYPE_CHECKING:
    from chap_core.api_types import BacktestParams, EstimatorOptions
    from chap_core.database.model_templates_and_config_tables import ModelConfiguration
    from chap_core.hpo.searcher import Searcher
    from chap_core.models.model_template import ModelTemplate

logger = logging.getLogger(__name__)


def create_hpo_estimator(
    *,
    template: ModelTemplate,
    configuration: ModelConfiguration | None,
    backtest_params: BacktestParams,
    options: EstimatorOptions,
) -> HyperparameterOptimizer:
    """
    Build an HPO-backend estimator from either:
    - an explicit YAML search space, or
    - the template's built-in hpo_search_space
    """

    if options.search_space is not None:
        logger.info(f"Loading hpo search space from {options.search_space}")
        with open(options.search_space, encoding="utf-8") as f:
            search_space_raw = yaml.safe_load(f)
    else:
        search_space_raw = template.model_template_config.hpo_search_space

    if not search_space_raw or not isinstance(search_space_raw, dict):
        raise ValueError(
            "HPO search space YAML must define a non-empty mapping of parameters, either "
            "through --estimator-options.search-space or the model's "
            "MLProject hpo_search_space."
        )

    search_space = search_space_from_config(search_space_raw)

    objective = Objective(model_template=template, backtest_params=backtest_params, metric=options.metric)

    searcher_type = options.searcher or SearcherType.TPE
    searcher: Searcher
    if searcher_type == SearcherType.GRID:
        searcher = GridSearcher()
        default_max_trials = None
    elif searcher_type == SearcherType.RANDOM:
        searcher = RandomSearcher()
        default_max_trials = DEFAULT_HPO_TRIALS
    elif searcher_type == SearcherType.TPE:
        searcher = TPESearcher(objective.direction.value)
        default_max_trials = DEFAULT_HPO_TRIALS
    else:
        raise ValueError(f"Unknown searcher: {searcher_type!r}")

    return HyperparameterOptimizer(
        objective=objective,
        searcher=searcher,
        model_configuration=configuration,
        search_space=search_space,
        max_trials=options.max_trials if options.max_trials is not None else default_max_trials,
        seed=options.seed,
    )
