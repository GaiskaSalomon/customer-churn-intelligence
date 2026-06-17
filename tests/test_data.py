from churn import data


def test_clean_produces_binary_target_and_numeric_charges():
    df = data.load_raw()
    X, y = data.clean(df)

    # target is binary 0/1, no missing
    assert set(y.unique()) <= {0, 1}
    assert y.isna().sum() == 0
    # id and target are not in features
    assert "customerID" not in X.columns
    assert "Churn" not in X.columns
    # TotalCharges coerced to numeric
    assert X["TotalCharges"].dtype.kind in "fi"
    # rows preserved
    assert len(X) == len(y) == len(df)


def test_split_columns_partitions_all_features():
    df = data.load_raw()
    X, _ = data.clean(df)
    numeric, categorical = data.split_columns(X)

    assert set(numeric).isdisjoint(categorical)
    assert set(numeric) | set(categorical) == set(X.columns)
    # known numeric fields
    assert "tenure" in numeric and "MonthlyCharges" in numeric
