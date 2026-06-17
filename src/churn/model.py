"""Model factory: full sklearn Pipeline (preprocessing + classifier)."""
from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from churn import config
from churn.features import build_preprocessor

MODELS = ("logreg", "xgboost")


def build_pipeline(model_name: str, numeric: list[str], categorical: list[str]) -> Pipeline:
    pre = build_preprocessor(numeric, categorical)

    if model_name == "logreg":
        clf = LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=config.RANDOM_STATE
        )
    elif model_name == "xgboost":
        from xgboost import XGBClassifier

        clf = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {MODELS}.")

    return Pipeline([("preprocessor", pre), ("classifier", clf)])
