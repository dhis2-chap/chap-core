"""Test fixtures for documentation code block validation."""

import re

# Required columns for chapkit training data
CHAPKIT_REQUIRED_COLUMNS = [
    "time_period",
    "location",
    "disease_cases",
    "rainfall",
    "mean_temperature",
    "population",
]

# Time period format patterns
TIME_PERIOD_PATTERNS = {
    "weekly": re.compile(r"^\d{4}-W\d{2}$"),  # 2020-W01
    "monthly": re.compile(r"^\d{4}-\d{2}$"),  # 2020-01
}

# Example valid chapkit training data (weekly)
CHAPKIT_TRAINING_DATA_WEEKLY = {
    "columns": ["time_period", "location", "disease_cases", "rainfall", "mean_temperature", "population"],
    "data": [
        ["2020-W01", "district_a", 150, 45.2, 28.5, 50000],
        ["2020-W02", "district_a", 142, 52.1, 27.8, 50000],
        ["2020-W01", "district_b", 89, 38.7, 29.1, 35000],
        ["2020-W02", "district_b", 95, 41.3, 28.9, 35000],
    ],
}

# Example valid chapkit training data (monthly)
CHAPKIT_TRAINING_DATA_MONTHLY = {
    "columns": ["time_period", "location", "disease_cases", "rainfall", "mean_temperature", "population"],
    "data": [
        ["2020-01", "district_a", 580, 180.5, 28.2, 50000],
        ["2020-02", "district_a", 620, 165.3, 27.9, 50000],
        ["2020-01", "district_b", 340, 155.8, 29.0, 35000],
        ["2020-02", "district_b", 365, 148.2, 28.7, 35000],
    ],
}

# Example run_info structure
CHAPKIT_RUN_INFO = {
    "prediction_length": 3,
    "additional_continuous_covariates": ["humidity"],
}

# Valid model configuration structure
VALID_MODEL_CONFIG = {
    "user_option_values": {
        "max_epochs": 2,
    }
}

# Classes that should be importable from documentation examples
IMPORTABLE_CLASSES = [
    ("chap_core.assessment.prediction_evaluator", "evaluate_model"),
    ("chap_core.models.utils", "get_model_from_directory_or_github_url"),
    ("chap_core.file_io.example_data_set", "datasets"),
    ("chap_core.adaptors.gluonts", "GluonTSEstimator"),
]

# CLI commands that should have working --help
CLI_HELP_COMMANDS = [
    ["chap", "--help"],
    ["chap", "evaluate", "--help"],
    ["chap", "evaluate2", "--help"],
    ["chap", "plot-backtest", "--help"],
    ["chap", "export-metrics", "--help"],
    ["chap", "forecast", "--help"],
]

# SQLModel code examples that should compile
SQLMODEL_EXAMPLES = [
    """
from typing import Optional
from sqlmodel import SQLModel

class DataSetBase(SQLModel):
    name: str

class DataSet(DataSetBase, table=True):
    id: Optional[int] = None
    new_field: Optional[str] = None
""",
]

# Alembic migration function structure
MIGRATION_FUNCTION_EXAMPLE = """
def upgrade():
    op.add_column('dataset', sa.Column('new_field', sa.String(), nullable=True))

def downgrade():
    op.drop_column('dataset', 'new_field')
"""
