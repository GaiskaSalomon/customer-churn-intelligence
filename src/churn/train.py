"""End-to-end training pipeline.

Loads data -> stratified split -> trains baseline (logreg) and XGBoost ->
evaluates both -> saves the best model, a metrics report, and figures (ROC, PR,
confusion, SHAP).

Usage:
    python -m churn.train
    python -m churn.train --test-size 0.2 --threshold 0.5
"""
from __future__ import annotations

import argparse
import json

import joblib
from sklearn.model_selection import train_test_split

from churn import config, data, evaluate
from churn.model import build_pipeline


def main() -> dict:
    ap = argparse.ArgumentParser(description="Train churn models end to end.")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--no-shap", action="store_true", help="Skip SHAP plot.")
    args = ap.parse_args()

    df = data.load_raw()
    X, y = data.clean(df)
    numeric, categorical = data.split_columns(X)
    print(f"[data] {len(X)} rows | churn rate = {y.mean():.1%} | "
          f"{len(numeric)} numeric, {len(categorical)} categorical features")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=config.RANDOM_STATE
    )

    results, fitted = {}, {}
    for name in ("logreg", "xgboost"):
        pipe = build_pipeline(name, numeric, categorical)
        pipe.fit(X_train, y_train)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        metrics = evaluate.compute_metrics(y_test, y_prob, threshold=args.threshold)
        results[name] = metrics
        fitted[name] = pipe
        print(f"[{name}] ROC-AUC={metrics['roc_auc']:.3f} "
              f"PR-AUC={metrics['pr_auc']:.3f} F1={metrics['f1']:.3f} "
              f"recall={metrics['recall']:.3f}")

    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best = fitted[best_name]
    print(f"[best] {best_name} (ROC-AUC={results[best_name]['roc_auc']:.3f})")

    # Persist artifacts
    config.MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, config.MODELS / "best_model.joblib")
    report = {"best_model": best_name, "threshold": args.threshold, "metrics": results}
    (config.REPORTS).mkdir(parents=True, exist_ok=True)
    (config.REPORTS / "metrics.json").write_text(json.dumps(report, indent=2))

    # Figures for the best model
    y_prob_best = best.predict_proba(X_test)[:, 1]
    evaluate.plot_roc(y_test, y_prob_best, config.FIGURES / "roc_curve.png")
    evaluate.plot_pr(y_test, y_prob_best, config.FIGURES / "pr_curve.png")
    evaluate.plot_confusion(y_test, y_prob_best, config.FIGURES / "confusion_matrix.png",
                            threshold=args.threshold)

    if not args.no_shap and best_name == "xgboost":
        try:
            from churn.explain import shap_summary

            shap_summary(best, X_train, config.FIGURES / "shap_summary.png")
            print("[shap] saved shap_summary.png")
        except Exception as exc:  # SHAP is optional; don't fail the pipeline
            print(f"[shap] skipped ({exc})")

    print(f"[done] model -> {config.MODELS / 'best_model.joblib'} | "
          f"report -> {config.REPORTS / 'metrics.json'}")
    return report


if __name__ == "__main__":
    main()
