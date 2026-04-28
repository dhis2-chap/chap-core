# Adding a new CLI command

The `chap` command is built with [cyclopts](https://cyclopts.readthedocs.io/). Commands are organized as plain functions in `chap_core/cli_endpoints/` and registered with the top-level app in `chap_core/cli.py`.

The console-script entry point is declared in `pyproject.toml`:

```toml
[project.scripts]
chap = "chap_core.cli:main"
```

## File layout

```
chap_core/
  cli.py                     # Builds the cyclopts App and calls register_commands
  cli_endpoints/
    _common.py               # Shared helpers (load_dataset, get_estimator, ...)
    evaluate.py              # eval, evaluate-hpo
    forecast.py              # forecast, multi-forecast
    convert.py               # convert-request
    explain.py               # explain-lime
    preference_learn.py      # preference-learn
    utils.py                 # plot-backtest, export-metrics, ...
    validate.py              # validate
```

Each module exposes a `register_commands(app)` function that the top-level `cli.py` calls during startup.

## Adding a command

1. **Pick or create a module under `chap_core/cli_endpoints/`.** Group commands by topic — add to an existing module if it fits, otherwise create a new one.

2. **Write the command as a plain function.** Use `typing.Annotated` with `cyclopts.Parameter` to attach help text. The docstring becomes the command's help output. Cyclopts derives flag names from parameter names (`output_file` → `--output-file`).

   ```python
   from pathlib import Path
   from typing import Annotated

   from cyclopts import Parameter


   def my_command(
       input_file: Annotated[Path, Parameter(help="Path to the input NetCDF file")],
       output_file: Annotated[Path, Parameter(help="Path to the output CSV file")],
       verbose: bool = False,
   ) -> None:
       """One-line summary shown in `chap --help`.

       Longer description shown in `chap my-command --help`.
       """
       ...
   ```

3. **Register the function in the module's `register_commands`.** The function name becomes the command name (snake_case → kebab-case). Pass `name=` to override.

   ```python
   def register_commands(app):
       app.command()(my_command)              # -> chap my-command
       app.command(name="eval")(eval_cmd)     # -> chap eval
   ```

4. **Wire the module into `chap_core/cli.py`** if it is new — import it and call `register_commands(app)` alongside the existing modules.

5. **Reuse helpers from `_common.py`.** Loading datasets, resolving CSV paths/GeoJSON, and building estimators should go through the existing helpers (`load_dataset_from_csv`, `resolve_csv_path`, `get_estimator`, etc.) so behavior stays consistent across commands.

## Conventions

- Use `pathlib.Path` for file/directory parameters, not `str`.
- Use Pydantic models grouped under one `Annotated[..., Parameter(...)]` argument when a command takes a cluster of related options (see `BackTestParams` and `RunConfig` in `eval_cmd`). Cyclopts exposes nested fields as `--group.field` flags automatically.
- Call `chap_core.log_config.initialize_logging(debug, log_file)` at the top of any command that produces user-visible output.
- Keep heavy imports (plotting, model code) inside the function body, not at module top, so `chap --help` stays fast.

## Testing

Tests call the command function directly with keyword arguments — there is no need to spawn a subprocess. See `tests/test_cli.py` for the pattern:

```python
def test_my_command(tmp_path):
    output_file = tmp_path / "out.csv"
    my_command(input_file=fixture_path, output_file=output_file)
    assert output_file.exists()
```

Run the suite with `uv run pytest tests/test_cli.py`.

## Verifying

After registering, the command should appear in:

```
uv run chap --help
uv run chap my-command --help
```
