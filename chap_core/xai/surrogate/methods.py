from chap_core.xai.method_registry import LIME_AUTO, SHAP_AUTO

METHOD_TO_MODEL_TYPE: dict[str, str] = {
    SHAP_AUTO: "auto",
    "shap_xgboost": "xgboost",
    "shap_lightgbm": "lightgbm",
    "shap_hist_gradient_boosting": "hist_gradient_boosting",
    "shap_gradient_boosting": "gradient_boosting",
    "shap_random_forest": "random_forest",
    "shap_extra_trees": "extra_trees",
    LIME_AUTO: "auto",
}
