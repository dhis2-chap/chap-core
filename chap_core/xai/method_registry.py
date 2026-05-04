NATIVE_SHAP = "native_shap"
SHAP_AUTO = "shap_auto"
LIME_AUTO = "lime_auto"


METHOD_TYPE_LABELS: dict[str, str] = {
    "surrogate_shap_auto": "Auto-tuned SHAP",
    "surrogate_shap": "Surrogate SHAP",
    "surrogate_lime_auto": "Auto-tuned LIME",
    "native_shap": "Native SHAP",
}

VISUALIZATION_LABELS: dict[str, str] = {
    "importance": "Feature importance",
    "waterfall": "Waterfall",
    "beeswarm": "Beeswarm",
}


XAI_METHODS = [
    {
        "id": 1,
        "name": "shap_auto",
        "display_name": "SHAP — Auto (best surrogate)",
        "description": (
            "Automatically benchmarks all available surrogate models using leave-one-out R² "
            "(XGBoost, LightGBM, Histogram Gradient Boosting, Random Forest, and others), "
            "tunes the top candidates with Optuna, and applies TreeSHAP for exact, "
            "additive feature attributions. Recommended for most use cases."
        ),
        "method_type": "surrogate_shap_auto",
        "author": "CHAP",
        "archived": False,
        "is_auto": True,
        "supported_visualizations": ["importance", "waterfall", "beeswarm"],
        "default_visualization": "waterfall",
    },
    {
        "id": 2,
        "name": "shap_xgboost",
        "display_name": "SHAP — XGBoost",
        "description": (
            "Fits an XGBoost surrogate on stored predictions, then applies TreeSHAP "
            "for exact, additive feature attributions. Often the most accurate surrogate "
            "for structured tabular data."
        ),
        "method_type": "surrogate_shap",
        "author": "CHAP",
        "archived": False,
        "is_auto": False,
        "supported_visualizations": ["importance", "waterfall", "beeswarm"],
        "default_visualization": "waterfall",
    },
    {
        "id": 3,
        "name": "shap_lightgbm",
        "display_name": "SHAP — LightGBM",
        "description": (
            "Fits a LightGBM surrogate on stored predictions, then applies TreeSHAP "
            "for exact, additive feature attributions. Fast training with strong accuracy."
        ),
        "method_type": "surrogate_shap",
        "author": "CHAP",
        "archived": False,
        "is_auto": False,
        "supported_visualizations": ["importance", "waterfall", "beeswarm"],
        "default_visualization": "waterfall",
    },
    {
        "id": 4,
        "name": "shap_hist_gradient_boosting",
        "display_name": "SHAP — Histogram Gradient Boosting",
        "description": (
            "Fits a scikit-learn HistGradientBoostingRegressor surrogate on stored predictions, "
            "then applies TreeSHAP for exact feature attributions. Native missing-value support."
        ),
        "method_type": "surrogate_shap",
        "author": "CHAP",
        "archived": False,
        "is_auto": False,
        "supported_visualizations": ["importance", "waterfall", "beeswarm"],
        "default_visualization": "waterfall",
    },
    {
        "id": 5,
        "name": "shap_random_forest",
        "display_name": "SHAP — Random Forest",
        "description": (
            "Fits a Random Forest surrogate on stored predictions, "
            "then applies TreeSHAP for exact, additive feature attributions."
        ),
        "method_type": "surrogate_shap",
        "author": "CHAP",
        "archived": False,
        "is_auto": False,
        "supported_visualizations": ["importance", "waterfall", "beeswarm"],
        "default_visualization": "waterfall",
    },
    {
        "id": 6,
        "name": "shap_gradient_boosting",
        "display_name": "SHAP — Gradient Boosted Trees (sklearn)",
        "description": (
            "Fits a scikit-learn GradientBoostingRegressor surrogate on stored predictions, "
            "then applies TreeSHAP for exact feature attributions."
        ),
        "method_type": "surrogate_shap",
        "author": "CHAP",
        "archived": False,
        "is_auto": False,
        "supported_visualizations": ["importance", "waterfall", "beeswarm"],
        "default_visualization": "waterfall",
    },
    {
        "id": 7,
        "name": "shap_extra_trees",
        "display_name": "SHAP — Extra Trees",
        "description": (
            "Fits an Extra Trees surrogate on stored predictions, "
            "then applies TreeSHAP for exact, additive feature attributions. "
            "Faster training than Random Forest with comparable accuracy."
        ),
        "method_type": "surrogate_shap",
        "author": "CHAP",
        "archived": False,
        "is_auto": False,
        "supported_visualizations": ["importance", "waterfall", "beeswarm"],
        "default_visualization": "waterfall",
    },
    {
        "id": 8,
        "name": "lime_auto",
        "display_name": "LIME — Auto (best surrogate)",
        "description": (
            "Automatically selects the surrogate model with the best leave-one-out R², "
            "then applies LIME for local, per-instance feature attribution."
        ),
        "method_type": "surrogate_lime_auto",
        "author": "CHAP",
        "archived": False,
        "is_auto": True,
        "supported_visualizations": ["importance"],
        "default_visualization": "importance",
    },
    {
        "id": 9,
        "name": "native_shap",
        "display_name": "SHAP — Native (from model)",
        "description": (
            "Uses SHAP values computed directly by the prediction model. "
            "No surrogate approximation is needed — these are exact attributions "
            "from the model itself. Only available when the model provides native SHAP output."
        ),
        "method_type": "native_shap",
        "author": "Model",
        "archived": False,
        "is_auto": False,
        "supported_visualizations": ["importance", "waterfall", "beeswarm"],
        "default_visualization": "waterfall",
    },
]
