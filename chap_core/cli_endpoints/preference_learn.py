"""Preference learning commands for CHAP CLI."""

import logging
from pathlib import Path
from typing import Literal

from chap_core.api_types import BackTestParams, RunConfig
from chap_core.assessment.evaluation import Evaluation
from chap_core.database.model_templates_and_config_tables import (
    ConfiguredModelDB,
    ModelConfiguration,
    ModelTemplateDB,
)
from chap_core.log_config import initialize_logging
from chap_core.models.model_template import ModelTemplate
from chap_core.preference_learning.decision_maker import (
    DecisionMaker,
    InteractiveDecisionMaker,
    MetricBasedDecisionMaker,
)
from chap_core.preference_learning.preference_learner import (
    ModelCandidate,
    PreferenceLearner,
)

from chap_core.cli_endpoints._common import (
    discover_geojson,
    load_dataset_from_csv,
)

logger = logging.getLogger(__name__)


def _create_evaluation(
    model_candidate: ModelCandidate,
    dataset,
    backtest_params: BackTestParams,
    run_config: RunConfig,
) -> Evaluation:
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
    model_names: str,
    dataset_csv: Path,
    state_file: Path = Path("preference_state.json"),
    backtest_params: BackTestParams = BackTestParams(n_periods=3, n_splits=7, stride=1),
    run_config: RunConfig = RunConfig(),
    max_iterations: int = 10,
    decision_mode: Literal["interactive", "metric"] = "metric",
    decision_metric: str = "mae",
    lower_is_better: bool = True,
):
    """
    Learn user preferences for models through iterative A/B testing.

    This command implements a preference learning loop:
    1. Get two candidate models from the PreferenceLearner
    2. Run backtest/evaluation on both models
    3. Present both evaluations to a DecisionMaker
    4. Calculate metrics and determine preferred model
    5. Report preference back to PreferenceLearner
    6. Repeat until convergence or max iterations

    Args:
        model_names: Comma-separated list of model identifiers to compare
        dataset_csv: Path to CSV file with disease data
        state_file: Path to file for persisting learner state
        backtest_params: Backtest configuration (n_periods, n_splits, stride)
        run_config: Model run environment configuration
        max_iterations: Maximum number of comparison iterations
        decision_mode: How to decide preference ('interactive' or 'metric')
        decision_metric: Metric to use for automatic decisions
        lower_is_better: Whether lower metric values are preferred
    """
    initialize_logging(run_config.debug, run_config.log_file)

    logger.info(f"Starting preference learning with models: {model_names}")

    # Parse model names into candidates
    model_list = [name.strip() for name in model_names.split(",")]
    candidates = [ModelCandidate(model_name=name) for name in model_list]

    logger.info(f"Created {len(candidates)} model candidates")

    # Load dataset
    geojson_path = discover_geojson(dataset_csv)
    dataset = load_dataset_from_csv(dataset_csv, geojson_path)

    # Create decision maker
    decision_maker: DecisionMaker
    if decision_mode == "interactive":
        decision_maker = InteractiveDecisionMaker()
    else:
        decision_maker = MetricBasedDecisionMaker(
            metric_name=decision_metric,
            lower_is_better=lower_is_better,
        )

    # Create preference learner (will load state if file exists)
    learner = PreferenceLearner(
        candidates=candidates,
        state_file=state_file,
        max_iterations=max_iterations,
    )

    logger.info(f"Starting from iteration {learner.current_iteration}")

    # Main learning loop
    while not learner.is_complete():
        pair = learner.get_next_pair()
        if pair is None:
            logger.info("No more pairs to compare")
            break

        model_a, model_b = pair
        logger.info(f"Comparing: {model_a.model_name} vs {model_b.model_name}")

        # Run evaluations for both models
        logger.info(f"Evaluating model A: {model_a.model_name}")
        evaluation_a = _create_evaluation(model_a, dataset, backtest_params, run_config)

        logger.info(f"Evaluating model B: {model_b.model_name}")
        evaluation_b = _create_evaluation(model_b, dataset, backtest_params, run_config)

        # Get decision
        preferred, metrics_a, metrics_b = decision_maker.decide(model_a, evaluation_a, model_b, evaluation_b)

        # Report to learner
        learner.report_preference(model_a, model_b, preferred, metrics_a, metrics_b)

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

    # Print comparison history
    history = learner.get_comparison_history()
    if history:
        print(f"\nComparison history ({len(history)} iterations):")
        for i, result in enumerate(history):
            print(f"  {i + 1}. {result.model_a.model_name} vs {result.model_b.model_name}")
            print(f"     Winner: {result.preferred.model_name}")


def register_commands(app):
    """Register preference learning commands with the CLI app."""
    app.command()(preference_learn)
