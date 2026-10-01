"""Validate a dataset against a model outside the UI process and print the issues as JSON.

Usage: python -m chap_core.ui.validation_runner <dataset_csv> <model_name>

Loading a model can start other programs (git for GitHub models, for instance). Doing that from
the UI process would fork it, which can crash once native libraries such as PROJ are loaded, so
the UI runs this helper instead (see services.validate_against_model).
"""

import json
import sys

from chap_core.ui.job_runner import close_inherited_descriptors


def main(dataset_csv: str, model_name: str) -> None:
    close_inherited_descriptors()
    from chap_core.cli_endpoints.validate import collect_validation_issues

    issues = collect_validation_issues(dataset_csv, model_name)
    print(json.dumps([issue.model_dump() for issue in issues]))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
