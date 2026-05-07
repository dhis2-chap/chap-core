# chap_core/ensemble/_legacy_wrappers.py

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from chap_core.models.model_template import ModelTemplate


@dataclass
class BaseModelSpec:
    template: ModelTemplate
    config: Any | None = None


class _TemplateWithConfig:
    def __init__(self, template: ModelTemplate, config: Any | None) -> None:
        self._template = template
        self._config = config

    def get_model(self, _: Any = None) -> Any:
        # Always forward the config; None means "default config".
        return self._template.get_model(self._config)

    @property
    def name(self) -> str | None:
        return getattr(self._template, "name", None)

    @property
    def repo(self) -> str | None:
        return getattr(self._template, "repo", None)

    def __str__(self) -> str:
        return str(self._template)
