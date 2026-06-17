import numpy as np

from churn import data
from churn.features import build_preprocessor


def test_preprocessor_fit_transform_shape_and_no_nan():
    df = data.load_raw().head(800)
    X, _ = data.clean(df)
    numeric, categorical = data.split_columns(X)

    pre = build_preprocessor(numeric, categorical)
    Xt = pre.fit_transform(X)

    # rows preserved; output is 2D; NaNs imputed away
    assert Xt.shape[0] == len(X)
    dense = Xt.toarray() if hasattr(Xt, "toarray") else Xt
    assert dense.ndim == 2
    assert not np.isnan(dense).any()


def test_preprocessor_handles_unseen_categories():
    df = data.load_raw()
    X, _ = data.clean(df)
    numeric, categorical = data.split_columns(X)

    pre = build_preprocessor(numeric, categorical)
    pre.fit(X.iloc[:500])
    # transforming later rows (possibly unseen categories) must not raise
    Xt = pre.transform(X.iloc[500:1000])
    assert Xt.shape[0] == 500
