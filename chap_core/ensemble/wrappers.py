"""Small wrapper utilities for ensemble base models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from chap_core.models.model_template import ModelTemplate


@dataclass
class BaseModelSpec:
    template: ModelTemplate
    config: Any | None = None


class TemplateWithConfig:
    """Binds a model template to its configuration, and optionally to a longer horizon.

    ``extend_to_prediction_length`` mirrors what ``chap evaluate`` does for a single
    model: a base model whose ``max_prediction_length`` is shorter than the backtest
    horizon is wrapped in an :class:`ExtendedPredictor` so it is asked for the number
    of steps it supports and iterated, rather than silently asked for too many.
    """

    def __init__(
        self,
        template: ModelTemplate,
        config: Any | None,
        extend_to_prediction_length: int | None = None,
    ) -> None:
        self._template = template
        self._config = config
        self._extend_to = extend_to_prediction_length

    def get_model(self, _: Any = None) -> Any:
        model = self._template.get_model(self._config)
        if self._extend_to is None:
            return model
        from chap_core.external.ExtendedPredictor import ExtendedPredictor

        # Callers instantiate what get_model returns; ExternalModel is self-returning
        # when called, while ExtendedPredictor is not, so hand back a factory.
        return lambda: ExtendedPredictor(model, self._extend_to)

    @property
    def name(self) -> str | None:
        return getattr(self._template, "name", None)

    @property
    def repo(self) -> str | None:
        return getattr(self._template, "repo", None)

    def __str__(self) -> str:
        return str(self._template)


__all__ = ["BaseModelSpec", "TemplateWithConfig"]
