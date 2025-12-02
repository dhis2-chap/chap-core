from typing import Optional
from pydantic import BaseModel, Field


class ChapUserOptions(BaseModel):
    """CHAP-specific user options available for all models."""

    chap__covid_mask: Optional[bool] = Field(
        default=False, description="Exclude COVID-19 period (2020-2021) from training data"
    )

    @classmethod
    def extract_from_config(cls, user_option_values: dict) -> "ChapUserOptions":
        """Extract only chap__ prefixed options from user_option_values."""
        chap_values = {key: value for key, value in (user_option_values or {}).items() if key.startswith("chap__")}
        return cls(**chap_values)

    def to_json_schema_properties(self) -> dict:
        """Generate JSON schema properties to merge into user_options."""
        return {
            "chap__covid_mask": {
                "type": "boolean",
                "title": "Mask COVID-19 Period",
                "description": "Exclude 2020-2021 COVID period from training data",
                "default": False,
            }
        }
