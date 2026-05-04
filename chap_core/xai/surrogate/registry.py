from typing import Any

SUPPORTED_MODELS: dict[str, dict[str, Any]] = {
    "hist_gradient_boosting": {
        "display_name": "Histogram Gradient Boosting",
        "class_dotted": "sklearn.ensemble.HistGradientBoostingRegressor",
        "default_params": {
            "max_iter": 600,
            "max_depth": 6,
            "learning_rate": 0.05,
            "early_stopping": True,
            "n_iter_no_change": 20,
            "validation_fraction": 0.15,
            "max_leaf_nodes": 31,
            "l2_regularization": 0.1,
            "min_samples_leaf": 10,
        },
        "loo_params": {"max_iter": 250, "max_depth": 6, "early_stopping": False},
        "tunable_params": {
            "max_iter": {"type": "int", "low": 300, "high": 2000},
            "max_depth": {"type": "int", "low": 2, "high": 12},
            "learning_rate": {"type": "float", "low": 0.005, "high": 0.2, "log": True},
            "min_samples_leaf": {"type": "int", "low": 1, "high": None, "high_n_fraction": 4},
            "max_leaf_nodes": {"type": "int", "low": 15, "high": 255},
            "l2_regularization": {"type": "float", "low": 0.0, "high": 10.0},
            "max_features": {"type": "float", "low": 0.3, "high": 1.0},
        },
        "shap_type": "tree",
        "optional": False,
    },
    "xgboost": {
        "display_name": "XGBoost",
        "class_dotted": "xgboost.XGBRegressor",
        "default_params": {
            "n_estimators": 600,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "min_child_weight": 3,
            "verbosity": 0,
            "eval_metric": "rmse",
        },
        "loo_params": {
            "n_estimators": 250,
            "max_depth": 4,
            "reg_lambda": 1.0,
            "min_child_weight": 3,
            "subsample": 0.8,
            "verbosity": 0,
        },
        "tunable_params": {
            "n_estimators": {"type": "int", "low": 200, "high": 2000},
            "max_depth": {"type": "int", "low": 2, "high": 12},
            "learning_rate": {"type": "float", "low": 0.005, "high": 0.3, "log": True},
            "subsample": {"type": "float", "low": 0.5, "high": 1.0},
            "colsample_bytree": {"type": "float", "low": 0.4, "high": 1.0},
            "reg_alpha": {"type": "float", "low": 1e-4, "high": 10.0, "log": True},
            "reg_lambda": {"type": "float", "low": 1e-4, "high": 10.0, "log": True},
            "min_child_weight": {"type": "int", "low": 1, "high": None, "high_n_fraction": 5},
            "gamma": {"type": "float", "low": 0.0, "high": 5.0},
        },
        "shap_type": "tree",
        "optional": True,
    },
    "lightgbm": {
        "display_name": "LightGBM",
        "class_dotted": "lightgbm.LGBMRegressor",
        "default_params": {
            "n_estimators": 600,
            "max_depth": -1,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "min_child_samples": 10,
            "verbose": -1,
        },
        "loo_params": {
            "n_estimators": 250,
            "num_leaves": 20,
            "verbose": -1,
        },
        "tunable_params": {
            "n_estimators": {"type": "int", "low": 200, "high": 2000},
            "num_leaves": {"type": "int", "low": 10, "high": None, "high_n_fraction": 2},
            "learning_rate": {"type": "float", "low": 0.005, "high": 0.3, "log": True},
            "subsample": {"type": "float", "low": 0.5, "high": 1.0},
            "colsample_bytree": {"type": "float", "low": 0.4, "high": 1.0},
            "reg_alpha": {"type": "float", "low": 1e-4, "high": 10.0, "log": True},
            "reg_lambda": {"type": "float", "low": 1e-4, "high": 10.0, "log": True},
            "min_child_samples": {"type": "int", "low": 1, "high": None, "high_n_fraction": 4},
            "min_split_gain": {"type": "float", "low": 0.0, "high": 1.0},
        },
        "shap_type": "tree",
        "optional": True,
    },
    "gradient_boosting": {
        "display_name": "Gradient Boosted Trees (sklearn)",
        "class_dotted": "sklearn.ensemble.GradientBoostingRegressor",
        "default_params": {
            "n_estimators": 600,
            "max_depth": 4,
            "learning_rate": 0.05,
            "n_iter_no_change": 20,
            "validation_fraction": 0.15,
            "subsample": 0.8,
            "max_features": 0.8,
            "min_samples_leaf": 5,
        },
        "loo_params": {
            "n_estimators": 180,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "max_features": 0.8,
            "min_samples_leaf": 5,
        },
        "tunable_params": {
            "n_estimators": {"type": "int", "low": 180, "high": 1300},
            "max_depth": {"type": "int", "low": 2, "high": 8},
            "learning_rate": {"type": "float", "low": 0.005, "high": 0.2, "log": True},
            "min_samples_leaf": {"type": "int", "low": 1, "high": None, "high_n_fraction": 5},
            "subsample": {"type": "float", "low": 0.5, "high": 1.0},
            "min_samples_split": {"type": "int", "low": 2, "high": None, "high_n_fraction": 10},
            "max_features": {"type": "float", "low": 0.3, "high": 1.0},
        },
        "shap_type": "tree",
        "optional": False,
    },
    "random_forest": {
        "display_name": "Random Forest",
        "class_dotted": "sklearn.ensemble.RandomForestRegressor",
        "default_params": {"n_estimators": 800, "max_depth": None, "min_samples_leaf": 2, "max_features": 0.5},
        "loo_params": {"n_estimators": 80, "max_depth": 8},
        "tunable_params": {
            "n_estimators": {"type": "int", "low": 200, "high": 1200},
            "max_depth": {"type": "int", "low": 3, "high": 30},
            "min_samples_leaf": {"type": "int", "low": 1, "high": None, "high_n_fraction": 5},
            "min_samples_split": {"type": "int", "low": 2, "high": None, "high_n_fraction": 10},
            "max_features": {"type": "float", "low": 0.2, "high": 1.0},
        },
        "shap_type": "tree",
        "optional": False,
    },
    "extra_trees": {
        "display_name": "Extra Trees",
        "class_dotted": "sklearn.ensemble.ExtraTreesRegressor",
        "default_params": {"n_estimators": 800, "max_depth": None, "min_samples_leaf": 2, "max_features": 0.5},
        "loo_params": {"n_estimators": 80, "max_depth": 8},
        "tunable_params": {
            "n_estimators": {"type": "int", "low": 200, "high": 1200},
            "max_depth": {"type": "int", "low": 3, "high": 30},
            "min_samples_leaf": {"type": "int", "low": 1, "high": None, "high_n_fraction": 5},
            "min_samples_split": {"type": "int", "low": 2, "high": None, "high_n_fraction": 10},
            "max_features": {"type": "float", "low": 0.2, "high": 1.0},
        },
        "shap_type": "tree",
        "optional": False,
    },
    "decision_tree": {
        "display_name": "Decision Tree",
        "class_dotted": "sklearn.tree.DecisionTreeRegressor",
        "default_params": {"max_depth": 6, "min_samples_leaf": 2},
        "loo_params": {"max_depth": 4, "min_samples_leaf": 2},
        "tunable_params": {
            "max_depth": {"type": "int", "low": 2, "high": 15},
            "min_samples_leaf": {"type": "int", "low": 1, "high": None, "high_n_fraction": 5},
            "min_samples_split": {"type": "int", "low": 2, "high": None, "high_n_fraction": 10},
            "max_features": {"type": "float", "low": 0.3, "high": 1.0},
        },
        "shap_type": "tree",
        "optional": False,
        "auto_eligible": False,
    },
    "ridge": {
        "display_name": "Ridge Regression",
        "class_dotted": "sklearn.linear_model.Ridge",
        "default_params": {"alpha": 1.0},
        "loo_params": {"alpha": 1.0},
        "tunable_params": {
            "alpha": {"type": "float", "low": 0.001, "high": 1000.0, "log": True},
        },
        "shap_type": "linear",
        "optional": False,
        "auto_eligible": False,
    },
    "lasso": {
        "display_name": "Lasso Regression",
        "class_dotted": "sklearn.linear_model.Lasso",
        "default_params": {"alpha": 0.1, "max_iter": 10000},
        "loo_params": {"alpha": 0.1, "max_iter": 5000},
        "tunable_params": {
            "alpha": {"type": "float", "low": 0.0001, "high": 100.0, "log": True},
        },
        "shap_type": "linear",
        "optional": False,
        "auto_eligible": False,
    },
    "catboost": {
        "display_name": "CatBoost",
        "class_dotted": "catboost.CatBoostRegressor",
        "default_params": {
            "iterations": 600,
            "depth": 6,
            "learning_rate": 0.05,
            "l2_leaf_reg": 3.0,
            "min_data_in_leaf": 5,
            "verbose": 0,
        },
        "loo_params": {
            "iterations": 250,
            "depth": 5,
            "verbose": 0,
        },
        "tunable_params": {
            "iterations": {"type": "int", "low": 200, "high": 2000},
            "depth": {"type": "int", "low": 2, "high": 10},
            "learning_rate": {"type": "float", "low": 0.005, "high": 0.3, "log": True},
            "l2_leaf_reg": {"type": "float", "low": 1e-3, "high": 10.0, "log": True},
            "min_data_in_leaf": {"type": "int", "low": 1, "high": None, "high_n_fraction": 4},
            "colsample_bylevel": {"type": "float", "low": 0.4, "high": 1.0},
        },
        "shap_type": "tree",
        "optional": True,
        "auto_eligible": False,
    },
}

DEFAULT_MODEL_TYPE = "hist_gradient_boosting"


def get_model_info(model_type: str) -> dict[str, Any]:
    if model_type not in SUPPORTED_MODELS:
        raise ValueError(f"Unknown surrogate model type '{model_type}'. Supported: {sorted(SUPPORTED_MODELS)}")
    return SUPPORTED_MODELS[model_type]


def get_display_name(model_type: str) -> str:
    info = SUPPORTED_MODELS.get(model_type)
    if info is None:
        return model_type
    value = info.get("display_name", model_type)
    return str(value)


def is_model_available(model_type: str) -> bool:
    info = SUPPORTED_MODELS.get(model_type, {})
    if not info.get("optional", False):
        return True
    module_path = info["class_dotted"].rsplit(".", 1)[0]
    try:
        import importlib

        importlib.import_module(module_path)
        return True
    except (ImportError, OSError):
        return False
