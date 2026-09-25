import json
from dataclasses import asdict, replace
from pathlib import Path

from chap_core.api_types import BacktestParams
from chap_core.database.model_templates_and_config_tables import ModelConfiguration
from chap_core.hpo.search_space import Int
from chap_core.hpo.types import HyperparameterOptimization, Trial


def make_hpo() -> HyperparameterOptimization:
    return HyperparameterOptimization(
        searcher="GridSearcher",
        model_template_name="demo-template",
        model_template_version="2.0",
        backtest_params=BacktestParams(n_periods=2, n_splits=3, stride=1),
        metric="rmse",
        search_space={"depth": Int(1, 5), "kind": ["a", "b"]},
        max_trials=None,
        seed=123,
        model_configuration=ModelConfiguration(
            user_option_values={"depth": 3},
            additional_continuous_covariates=["rainfall"],
        ),
        best_params={"depth": 3, "kind": "a"},
        best_score=1.25,
        leaderboard=[
            Trial(
                trial_nr=0,
                params={"depth": 3, "kind": "a"},
                score=1.25,
                seconds=0.4,
                failure=None,
            ),
            Trial(
                trial_nr=1,
                params={"depth": 5, "kind": "b"},
                score=None,
                seconds=0.2,
                failure="RuntimeError: boom",
            ),
        ],
        seconds=1.2,
        stop_reason="search_exhausted",
    )


def test_hpo_to_flat_contains_reproducibility_and_summary_metadata() -> None:
    """to_flat preserves reproducibility metadata and summarizes trial outcomes for Evaluation/NetCDF."""
    hpo = make_hpo()

    flat = hpo.to_flat()

    assert flat.searcher == "GridSearcher"
    assert flat.model_template_name == "demo-template"
    assert flat.model_template_version == "2.0"
    assert flat.direction == "minimize"
    assert flat.metric == "rmse"
    assert flat.backtest_params == hpo.backtest_params.model_dump()
    assert flat.search_space == {
        "depth": {"type": "int", "low": 1, "high": 5, "step": 1, "log": False},
        "kind": ["a", "b"],
    }
    assert flat.max_trials is None
    assert flat.seed == 123
    assert flat.model_configuration == hpo.model_configuration.model_dump(mode="json")
    assert flat.best_params == {"depth": 3, "kind": "a"}
    assert flat.best_score == 1.25
    assert flat.n_trials == 2
    assert flat.n_successful_trials == 1
    assert flat.n_failed_trials == 1
    assert flat.seconds == 1.2
    assert flat.stop_reason == "search_exhausted"

    json.dumps(asdict(flat))


def test_write_trials_outputs_execution_order(tmp_path: Path) -> None:
    """write_trials writes JSONL in original trial-number order even when the leaderboard is ranked."""
    hpo = make_hpo()
    hpo = replace(hpo, leaderboard=[hpo.leaderboard[1], hpo.leaderboard[0]])
    output = tmp_path / "trials.jsonl"

    hpo.write_trials(output)

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["trial_nr"] for row in rows] == [0, 1]
    assert rows[0]["params"] == {"depth": 3, "kind": "a"}
    assert rows[1]["failure"] == "RuntimeError: boom"
