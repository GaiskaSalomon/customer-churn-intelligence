import numpy as np

from churn import data, evaluate
from churn.model import build_pipeline


def _small_split():
    df = data.load_raw().head(1500)
    X, y = data.clean(df)
    numeric, categorical = data.split_columns(X)
    return X, y, numeric, categorical


def test_logreg_pipeline_fits_and_predicts_probabilities():
    X, y, numeric, categorical = _small_split()
    pipe = build_pipeline("logreg", numeric, categorical)
    pipe.fit(X, y)
    prob = pipe.predict_proba(X)[:, 1]

    assert prob.shape == (len(X),)
    assert np.all((prob >= 0) & (prob <= 1))


def test_compute_metrics_keys_and_ranges():
    X, y, numeric, categorical = _small_split()
    pipe = build_pipeline("logreg", numeric, categorical)
    pipe.fit(X, y)
    prob = pipe.predict_proba(X)[:, 1]

    m = evaluate.compute_metrics(y, prob)
    for k in ("roc_auc", "pr_auc", "f1", "precision", "recall", "accuracy"):
        assert k in m and 0.0 <= m[k] <= 1.0
    # a fitted model should beat random on training data
    assert m["roc_auc"] > 0.6
