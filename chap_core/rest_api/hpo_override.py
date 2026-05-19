from __future__ import annotations

from typing import TYPE_CHECKING

from chap_core.hpo.base import load_search_space_from_config
from chap_core.hpo.hpoModel import HpoModel
from chap_core.hpo.objective import Objective

if TYPE_CHECKING:
    from chap_core.database.database import SessionWrapper


class HpoOverride:
    HPO_SUFFIX = ":hpo"
    _HPO_DISPLAY_SUFFIX = "[Hpo]"

    def __init__(self) -> None:
        self._seen_model_template_ids: set[int] = set()

    @classmethod
    def is_hpo_model_name(cls, model_name: str | None) -> bool:
        return model_name is not None and model_name.endswith(cls.HPO_SUFFIX)

    def seed_hpo_model_hack(
        self,
        configured_model,
        merged_data: dict,
        template_data: dict,
    ) -> dict | None:
        model_template_db = configured_model.model_template

        if model_template_db.hpo_search_space is None:
            return None
        if model_template_db.id in self._seen_model_template_ids:
            return None

        self._seen_model_template_ids.add(model_template_db.id)
        hpo_data = merged_data.copy()
        # use template name, not configured model/merged_data name, hpo model name should not depend on whichever configured model with the same template happend to be selected first
        hpo_data["name"] = f"{template_data['name']}{self.HPO_SUFFIX}"

        if (
            model_template_db.display_name != "No Display Name Yet"
        ):  # change to check whether it matches default rather than hardcoded str
            hpo_data["display_name"] = f"{model_template_db.display_name} {self._HPO_DISPLAY_SUFFIX}"

        return hpo_data

    @classmethod
    def get_hpo_configured_model_and_estimator(cls, model_id: str, session: SessionWrapper):
        if not model_id.endswith(cls.HPO_SUFFIX):
            raise ValueError(f"Expected HPO model id ending with {cls.HPO_SUFFIX}, got {model_id}")
        configured_model = session.get_configured_model_by_id_or_name(model_id.removesuffix(cls.HPO_SUFFIX))
        assert configured_model.id is not None, "configured_model.id is required"
        template = session.get_model_template_with_code(configured_model.model_template_id)
        configuration = template.model_template_config.hpo_search_space
        if configuration is None:
            raise ValueError(f"Expects model template {template.name} to have hpo search space defined")
        search_space = load_search_space_from_config(configuration)
        objective = Objective(model_template=template)
        estimator = HpoModel(objective=objective, model_configuration=search_space)
        return configured_model, estimator
