"""Preference learning commands for CHAP CLI."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class PreferenceLearningParams(BaseModel):
    """Parameters for preference learning."""

    max_iterations: int = 10
    decision_mode: Literal["visual", "metric"] = "visual"
    decision_metrics: list[str] = ["mae"]
    lower_is_better: bool = True


def _compute_metrics(evaluation) -> dict:
    """
    Compute metrics from an evaluation.

    Args:
        evaluation: Evaluation object with forecasts and observations

    Returns:
        Dictionary of metric name to value
    """
    from chap_core.assessment.metrics import available_metrics

    flat_data = evaluation.to_flat()

    # Compute all aggregated metrics (those that return a single value)
    results = {}
    for metric_id, metric_cls in available_metrics.items():
        metric = metric_cls()
        if not metric.is_full_aggregate():
            continue
        try:
            metric_df = metric.get_metric(flat_data.observations, flat_data.forecasts)
            if len(metric_df) == 1:
                results[metric_id] = float(metric_df["metric"].iloc[0])
        except Exception as e:
            logger.warning(f"Failed to compute metric {metric_id}: {e}")

    return results


def _create_evaluation(model_candidate, dataset, backtest_params, run_config):
    """
    Create an Evaluation for a model candidate.

    Args:
        model_candidate: The model to evaluate
        dataset: Dataset to evaluate on
        backtest_params: Backtest parameters
        run_config: Run configuration

    Returns:
        Evaluation object with backtest results
    """
    from chap_core.assessment.evaluation import Evaluation
    from chap_core.database.model_templates_and_config_tables import (
        ConfiguredModelDB,
        ModelConfiguration,
        ModelTemplateDB,
    )
    from chap_core.models.model_template import ModelTemplate

    logger.info(f"Loading model template from {model_candidate.model_name}")
    template = ModelTemplate.from_directory_or_github_url(
        model_candidate.model_name,
        base_working_dir=Path("./runs/"),
        ignore_env=run_config.ignore_environment,
        run_dir_type=run_config.run_directory_type,
        is_chapkit_model=run_config.is_chapkit_model,
    )

    configuration = None
    if model_candidate.configuration:
        configuration = ModelConfiguration.model_validate(model_candidate.configuration)

    model = template.get_model(configuration)
    estimator = model()

    model_template_db = ModelTemplateDB(
        id=template.model_template_config.name,
        name=template.model_template_config.name,
        version=template.model_template_config.version or "unknown",
    )

    configured_model_db = ConfiguredModelDB(
        id=f"pref_learn_{model_candidate.model_name}",
        model_template_id=model_template_db.id,
        model_template=model_template_db,
        configuration=configuration.model_dump() if configuration else {},
    )

    logger.info(f"Running backtest for {model_candidate.model_name}")
    evaluation = Evaluation.create(
        configured_model=configured_model_db,
        estimator=estimator,
        dataset=dataset,
        backtest_params=backtest_params,
        backtest_name=f"{model_candidate.model_name}_preference_eval",
    )

    return evaluation


def preference_learn(
    model_name: str,
    dataset_csv: Path,
    search_space_yaml: Optional[Path] = None,
    state_file: Path = Path("preference_state.json"),
    n_periods: int = 3,
    n_splits: int = 7,
    stride: int = 1,
    ignore_environment: bool = False,
    debug: bool = False,
    log_file: Optional[str] = None,
    run_directory_type: Optional[Literal["latest", "timestamp", "use_existing"]] = "timestamp",
    is_chapkit_model: bool = False,
    max_iterations: int = 10,
    decision_mode: Literal["visual", "metric"] = "visual",
    decision_metrics: Optional[list[str]] = None,
    lower_is_better: bool = True,
):
    """
    Learn user preferences for model configurations through iterative A/B testing.

    This command implements a preference learning loop:
    1. Get candidate model configurations from the PreferenceLearner
    2. Run backtest/evaluation on all candidates
    3. Compute metrics for each evaluation
    4. Present evaluations to a DecisionMaker (visual or metric-based)
    5. Report preference back to PreferenceLearner
    6. Repeat until convergence or max iterations

    Args:
        model_name: Model identifier (path or GitHub URL)
        dataset_csv: Path to CSV file with disease data
        search_space_yaml: Path to YAML file defining hyperparameter search space
        state_file: Path to file for persisting learner state
        n_periods: Number of forecast periods
        n_splits: Number of train/test splits
        stride: Stride between splits
        ignore_environment: Ignore model environment requirements
        debug: Enable debug logging
        log_file: Path to log file
        run_directory_type: Type of run directory
        is_chapkit_model: Whether model is a chapkit model
        max_iterations: Maximum iterations for preference learning
        decision_mode: How to decide between candidates (visual or metric)
        decision_metrics: Metrics to use for metric-based decisions
        lower_is_better: Whether lower metric values are better
    """
    import yaml

    from chap_core.api_types import BackTestParams, RunConfig
    from chap_core.hpo.base import load_search_space_from_config
    from chap_core.log_config import initialize_logging
    from chap_core.models.model_template import ModelTemplate
    from chap_core.preference_learning.decision_maker import (
        DecisionMaker,
        MetricDecisionMaker,
        VisualDecisionMaker,
    )
    from chap_core.preference_learning.preference_learner import TournamentPreferenceLearner

    from chap_core.cli_endpoints._common import (
        discover_geojson,
        load_dataset_from_csv,
    )

    if decision_metrics is None:
        decision_metrics = ["mae"]

    backtest_params = BackTestParams(n_periods=n_periods, n_splits=n_splits, stride=stride)
    run_config = RunConfig(
        ignore_environment=ignore_environment,
        debug=debug,
        log_file=log_file,
        run_directory_type=run_directory_type,
        is_chapkit_model=is_chapkit_model,
    )
    learning_params = PreferenceLearningParams(
        max_iterations=max_iterations,
        decision_mode=decision_mode,
        decision_metrics=decision_metrics,
        lower_is_better=lower_is_better,
    )

    initialize_logging(run_config.debug, run_config.log_file)

    logger.info(f"Starting preference learning for model: {model_name}")

    # Load dataset
    geojson_path = discover_geojson(dataset_csv)
    dataset = load_dataset_from_csv(dataset_csv, geojson_path)

    # Load or create preference learner
    if state_file.exists():
        logger.info(f"Loading existing state from {state_file}")
        learner = TournamentPreferenceLearner.load(state_file)
    else:
        # Load search space
        if search_space_yaml is None:
            # Try to get search space from model template
            template = ModelTemplate.from_directory_or_github_url(
                model_name,
                base_working_dir=Path("./runs/"),
                ignore_env=run_config.ignore_environment,
                run_dir_type=run_config.run_directory_type,
                is_chapkit_model=run_config.is_chapkit_model,
            )
            raw_search_space = template.model_template_config.hpo_search_space
            if not raw_search_space:
                raise ValueError(
                    "No search space provided and model template has no hpo_search_space. "
                    "Please provide --search-space-yaml"
                )
        else:
            logger.info(f"Loading search space from {search_space_yaml}")
            with open(search_space_yaml, "r") as f:
                raw_search_space = yaml.safe_load(f)

        search_space = load_search_space_from_config(raw_search_space)

        logger.info(f"Initializing learner with search space: {search_space}")
        learner = TournamentPreferenceLearner.init(
            model_name=model_name,
            search_space=search_space,
            max_iterations=learning_params.max_iterations,
        )

    logger.info(f"Starting from iteration {learner.current_iteration}")

    # Main learning loop
    while not learner.is_complete():
        next_candidates = learner.get_next_candidates()
        if next_candidates is None:
            logger.info("No more candidates to compare")
            break

        # Run evaluations for all candidates
        evaluations = []
        for candidate in next_candidates:
            logger.info(f"Evaluating: {candidate.model_name} with config {candidate.configuration}")
            evaluation = _create_evaluation(candidate, dataset, backtest_params, run_config)
            evaluations.append(evaluation)

        # Compute metrics for all evaluations
        metrics = [_compute_metrics(e) for e in evaluations]

        # Create decision maker
        decision_maker: DecisionMaker
        if learning_params.decision_mode == "visual":
            decision_maker = VisualDecisionMaker()
        else:
            decision_maker = MetricDecisionMaker(
                metrics=metrics,
                metric_names=learning_params.decision_metrics,
                lower_is_better=learning_params.lower_is_better,
            )

        # Get decision
        preferred_index = decision_maker.decide(evaluations)

        # Report to learner
        learner.report_preference(next_candidates, preferred_index, metrics)

        # Save state after each iteration
        learner.save(state_file)

        logger.info(f"Iteration {learner.current_iteration} complete")

    # Report final results
    best = learner.get_best_candidate()
    if best:
        logger.info(f"Best model: {best.model_name}")
        print("\nPreference learning complete!")
        print(f"Best model: {best.model_name}")
        if best.configuration:
            print(f"Configuration: {best.configuration}")
    else:
        logger.warning("No best candidate found")


def register_commands(app):
    """Register preference learning commands with the CLI app."""
    app.command()(preference_learn)
