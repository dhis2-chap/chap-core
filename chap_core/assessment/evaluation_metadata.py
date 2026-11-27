"""Metadata describing an evaluation result, independent of where the result is stored."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from chap_core.api_types import BacktestParams
from chap_core.database.base_tables import UtcDatetime  # noqa: TC001  (pydantic resolves the annotation at runtime)

if TYPE_CHECKING:
    from chap_core.database.model_templates_and_config_tables import ConfiguredModelDB, ModelConfiguration
    from chap_core.models.model_template import ModelTemplate


class EvaluatedModel(BaseModel):
    """The model an evaluation ran: a stored configured model, a CLI template plus configuration, or file attrs."""

    model_config = ConfigDict(frozen=True)

    name: str
    version: str | None = None
    configuration: dict[str, Any] = Field(default_factory=dict)
    info: dict[str, Any] | None = Field(
        default=None, description="Model template information, written to NetCDF files for model cards."
    )

    @classmethod
    def from_configured_model(cls, configured_model: ConfiguredModelDB) -> EvaluatedModel:
        return cls(
            name=configured_model.name,
            version=configured_model.model_template.version,
            configuration={
                "user_option_values": configured_model.user_option_values or {},
                "additional_continuous_covariates": configured_model.additional_continuous_covariates or [],
            },
        )

    @classmethod
    def from_template(
        cls,
        template: ModelTemplate,
        configuration: ModelConfiguration | None = None,
        info: BaseModel | None = None,
        name: str | None = None,
    ) -> EvaluatedModel:
        """The template's model, under ``name`` if given (the CLI records the name or URL it was run with)."""
        return cls(
            name=name or template.model_template_config.name,
            version=template.model_template_config.version or "unknown",
            configuration=configuration.model_dump() if configuration is not None else {},
            info=info.model_dump(mode="json", exclude_none=True) if info is not None else None,
        )


class EvaluationMetadata(BaseModel):
    """What an evaluation result is: how it was run, by which model, and when.

    Storage identity (database ids, the backtest name) is not part of it; it is given
    when the evaluation is persisted.
    """

    params: BacktestParams = Field(default_factory=BacktestParams)
    model: EvaluatedModel | None = None
    last_train_period: str | None = None
    historical_context_periods: int = 0
    chap_version: str | None = None
    created: UtcDatetime | None = None
