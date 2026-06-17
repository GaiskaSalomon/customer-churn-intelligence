"""Model explainability with SHAP (tree models)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.sparse as sp

from churn import config


def shap_summary(pipeline, X, out_path: Path, max_samples: int = 500) -> None:
    """Save a SHAP summary plot for a fitted tree-based pipeline."""
    import shap

    pre = pipeline.named_steps["preprocessor"]
    clf = pipeline.named_steps["classifier"]

    sample = X.sample(min(len(X), max_samples), random_state=config.RANDOM_STATE)
    transformed = pre.transform(sample)
    if sp.issparse(transformed):
        transformed = transformed.toarray()
    feature_names = pre.get_feature_names_out()

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(transformed)

    shap.summary_plot(
        shap_values, transformed, feature_names=feature_names, show=False, max_display=15
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
