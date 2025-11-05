from __future__ import annotations

import numpy as np

from stages.s08_feature_engineering.starter.encoding import (
    label_encode,
    one_hot_encode,
)
from stages.s08_feature_engineering.starter.imputation import (
    simple_imputer_strategy,
)
from stages.s08_feature_engineering.starter.scaling import (
    correlation_filter,
    detect_outliers_iqr,
    min_max_scale,
    variance_threshold_select,
)


class TestPreprocessingPipeline:
    def test_complete_pipeline(self):
        np.random.seed(42)

        X_train = np.random.randn(100, 10)
        X_train[5, 2] = np.nan
        X_train[10, 5] = np.nan

        X_train_clean = simple_imputer_strategy(X_train, strategy="mean")
        assert not np.isnan(X_train_clean).any()

        X_train_scaled, train_min, train_max = min_max_scale(X_train_clean)
        assert X_train_scaled.min() >= 0.0
        assert X_train_scaled.max() <= 1.0

        variance_mask = variance_threshold_select(X_train_scaled, threshold=0.01)
        X_train_var = X_train_scaled[:, variance_mask]
        assert X_train_var.shape[1] <= X_train_scaled.shape[1]

        corr_mask = correlation_filter(X_train_var, threshold=0.95)
        X_train_final = X_train_var[:, corr_mask]
        assert X_train_final.shape[1] <= X_train_var.shape[1]

    def test_train_test_consistency(self):
        np.random.seed(42)

        X_train = np.random.randn(50, 5)
        X_test = np.random.randn(20, 5)

        X_train[0, 0] = np.nan
        X_test[0, 0] = np.nan

        X_train_clean = simple_imputer_strategy(X_train, strategy="median")
        X_test_clean = simple_imputer_strategy(X_test, strategy="median")

        X_train_scaled, train_min, train_max = min_max_scale(X_train_clean)
        X_test_scaled = (X_test_clean - train_min) / (train_max - train_min)

        variance_mask = variance_threshold_select(X_train_scaled, threshold=0.0)
        X_train_var = X_train_scaled[:, variance_mask]
        X_test_var = X_test_scaled[:, variance_mask]

        assert X_train_var.shape[1] == X_test_var.shape[1]

    def test_outlier_handling_pipeline(self):
        np.random.seed(42)

        X = np.random.randn(100, 3)
        X[0, 0] = 100.0
        X[1, 1] = -100.0

        outliers = detect_outliers_iqr(X, multiplier=1.5)

        assert outliers[0, 0]
        assert outliers[1, 1]

        X_clean = X[~outliers.any(axis=1)]
        assert X_clean.shape[0] < X.shape[0]

    def test_encoding_in_pipeline(self):
        y_train = np.array(["cat", "dog", "cat", "bird", "dog"])

        y_encoded, classes = label_encode(y_train)
        y_onehot = one_hot_encode(y_encoded, n_classes=len(classes))

        assert y_onehot.shape == (5, 3)
        assert y_onehot.sum() == 5

    def test_no_data_leakage(self):
        np.random.seed(42)

        X_all = np.random.randn(100, 5)
        X_all[10, 2] = np.nan

        X_train = X_all[:80]
        X_test = X_all[80:]

        X_train_clean = simple_imputer_strategy(X_train, strategy="mean")
        X_train_scaled, train_min, train_max = min_max_scale(X_train_clean)

        X_test_clean = simple_imputer_strategy(X_test, strategy="mean")
        (X_test_clean - train_min) / (train_max - train_min)

        assert train_min.shape == (5,)
        assert train_max.shape == (5,)

    def test_feature_selection_pipeline(self):
        X = np.random.randn(50, 10)

        X[:, 0] = 5.0
        X[:, 5] = X[:, 6] + np.random.randn(50) * 0.01

        variance_mask = variance_threshold_select(X, threshold=0.01)
        assert not variance_mask[0]

        X_var = X[:, variance_mask]

        corr_mask = correlation_filter(X_var, threshold=0.95)
        X_final = X_var[:, corr_mask]

        assert X_final.shape[1] < X.shape[1]
