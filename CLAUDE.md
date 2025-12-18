## Workflow

1. Always run `make lint` and `make test` after adding a feature or fixing a bug
2. No commit attribution (no co-author or generated-by lines)
3. Use conventional commits format for commit messages
4. Branch naming should follow conventional commit prefixes: `feat/`, `fix/`, `docs/`, `refactor/`, etc.
5. No emojis in commit messages, PR descriptions, or code comments
6. Be concise when adding code/features. Don't add stuff not specifically related to the problem/task, and avoid nice-to-have extras
7. If you feel that the prompt is bad/unclear, always ask follow-up questions until you have high confidence you will be able to solve the problem given
8. When adding new features or fixes, always add a test. Never access private fields or methods (starting with underscore) of a class through testing. Also, when changing code, if there are no relevant tests, consider adding a simple test for the change.
9. When we ask you to go through some change (or finalize changes), always follow the rules in this document strictly.
10. When making pr that has design document only, don't use docs in title.
11. When writing tests, avoid creating new test data inline. Use existing fixtures from conftest.py files whenever possible. Only create new fixtures in conftest.py if testing edge cases not covered by existing fixtures. This improves test maintainability and reduces duplication.
12. Never access private variables
13. When creating Jira issues, always set at least one component

## Domain Knowledge
- To learn about domain-specific terms used in the codebase, refer to @docs/contributor/vocabulary.md.


## Code architecture

### Entry Points
- **CLI** (`chap_core/cli.py`): Commands like `evaluate`, `forecast`, `serve`
- **REST API** (`chap_core/rest_api/v1/rest_api.py`): FastAPI backend with Celery for async tasks
  - Routers in `rest_api/v1/routers/`: analytics, crud, visualization, debug, jobs

### Core Concepts
- **Models** (`chap_core/models/`):
  - `ModelTemplate`: Factory for creating configured models
  - `ExternalModel`: Wrapper for Docker/command-line/web-based models
- **Runners** (`chap_core/runners/`): Execute models in different environments (Docker, command-line, MLflow)
- **Assessment** (`chap_core/assessment/`): Model evaluation, train/test splitting, backtesting
- **Database** (`chap_core/database/`): SQLModel tables (find with `table=True`), Alembic migrations
- **Data Types** (`chap_core/datatypes.py`, `spatio_temporal_data/`): Core data structures for time series

### Key Patterns
- Load model: `ModelTemplate.from_directory_or_github_url()` → `get_model()` → configured model instance
- Execute model: Model uses appropriate `TrainPredictRunner` (Docker/CommandLine/MLflow) to run train/predict
- Evaluate: `evaluate_model()` in `prediction_evaluator.py` handles train/test splits and evaluation

See `docs/contributor/code_overview.md` for detailed architecture documentation.
