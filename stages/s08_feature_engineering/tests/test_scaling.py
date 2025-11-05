from __future__ import annotations

import numpy as np

from stages.s08_feature_engineering.starter.scaling import (
    correlation_filter,
    detect_outliers_iqr,
    min_max_scale,
    robust_scale,
    variance_threshold_select,
)


class TestMinMaxScale:
    def test_scale_to_01(self) -> None:
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        X_scaled, X_min, X_max = min_max_scale(X, feature_range=(0, 1))

        np.testing.assert_allclose(X_scaled, [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        np.testing.assert_allclose(X_min, [1.0, 2.0])
        np.testing.assert_allclose(X_max, [5.0, 6.0])

    def test_scale_to_custom_range(self) -> None:
        X = np.array([[0.0], [5.0], [10.0]])

        X_scaled, X_min, X_max = min_max_scale(X, feature_range=(-1, 1))

        np.testing.assert_allclose(X_scaled, [[-1.0], [0.0], [1.0]])

    def test_constant_feature(self) -> None:
        X = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])

        X_scaled, X_min, X_max = min_max_scale(X)

        assert not np.isnan(X_scaled).any()
        assert not np.isinf(X_scaled).any()

    def test_single_sample(self) -> None:
        X = np.array([[1.0, 2.0]])

        X_scaled, X_min, X_max = min_max_scale(X)

        assert X_scaled.shape == (1, 2)

    def test_already_scaled(self) -> None:
        X = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])

        X_scaled, X_min, X_max = min_max_scale(X)

        np.testing.assert_allclose(X_scaled, X)


class TestRobustScale:
    def test_basic_scaling(self) -> None:
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])

        X_scaled, median, iqr = robust_scale(X)

        np.testing.assert_allclose(median, [3.0])
        np.testing.assert_allclose(iqr, [2.0])

    def test_with_outliers(self) -> None:
        X = np.array([[1.0], [2.0], [3.0], [4.0], [100.0]])

        X_scaled, median, iqr = robust_scale(X)

        assert np.abs(X_scaled[-1, 0]) > 10

    def test_zero_iqr(self) -> None:
        X = np.array([[5.0], [5.0], [5.0], [5.0]])

        X_scaled, median, iqr = robust_scale(X)

        assert not np.isnan(X_scaled).any()
        assert not np.isinf(X_scaled).any()

    def test_multiple_features(self) -> None:
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0], [5.0, 50.0]])

        X_scaled, median, iqr = robust_scale(X)

        assert X_scaled.shape == X.shape
        assert median.shape == (2,)
        assert iqr.shape == (2,)


class TestVarianceThresholdSelect:
    def test_remove_zero_variance(self) -> None:
        X = np.array([[0, 1, 2], [0, 3, 4], [0, 5, 6]])

        selected = variance_threshold_select(X, threshold=0.0)

        expected = np.array([False, True, True])
        np.testing.assert_array_equal(selected, expected)

    def test_all_features_selected(self) -> None:
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        selected = variance_threshold_select(X, threshold=0.0)

        assert selected.all()

    def test_high_threshold(self) -> None:
        X = np.array([[1, 100], [2, 200], [3, 300]])

        selected = variance_threshold_select(X, threshold=100.0)

        assert selected[1]
        assert selected.sum() >= 1

    def test_shape(self) -> None:
        X = np.array([[1, 2, 3, 4]])

        selected = variance_threshold_select(X)

        assert selected.shape == (4,)
        assert selected.dtype == bool


class TestCorrelationFilter:
    def test_remove_correlated_features(self) -> None:
        X = np.array([[1, 2, 2.1], [2, 4, 4.2], [3, 6, 6.3], [4, 8, 8.4]])

        selected = correlation_filter(X, threshold=0.95)

        assert selected.sum() < 3

    def test_no_correlation(self) -> None:
        np.random.seed(42)
        X = np.random.randn(50, 5)

        selected = correlation_filter(X, threshold=0.95)

        assert selected.sum() == 5

    def test_perfect_correlation(self) -> None:
        X = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])

        selected = correlation_filter(X, threshold=0.99)

        assert selected.sum() == 1

    def test_negative_correlation(self) -> None:
        X = np.array([[1, -1], [2, -2], [3, -3], [4, -4]])

        selected = correlation_filter(X, threshold=0.95)

        assert selected.sum() == 1

    def test_single_feature(self) -> None:
        X = np.array([[1], [2], [3]])

        selected = correlation_filter(X, threshold=0.9)

        assert selected[0]


class TestDetectOutliersIQR:
    def test_basic_outlier_detection(self) -> None:
        X = np.array([[1], [2], [3], [4], [100]])

        outlier_mask = detect_outliers_iqr(X, multiplier=1.5)

        assert outlier_mask[-1, 0]
        assert not outlier_mask[:-1, 0].any()

    def test_no_outliers(self) -> None:
        X = np.array([[1], [2], [3], [4], [5]])

        outlier_mask = detect_outliers_iqr(X, multiplier=1.5)

        assert not outlier_mask.any()

    def test_multiple_outliers(self) -> None:
        X = np.array([[-100], [1], [2], [3], [100]])

        outlier_mask = detect_outliers_iqr(X, multiplier=1.5)

        assert outlier_mask[0, 0]
        assert outlier_mask[-1, 0]

    def test_multiple_features(self) -> None:
        X = np.array([[1, 1], [2, 2], [3, 3], [4, 100]])

        outlier_mask = detect_outliers_iqr(X, multiplier=1.5)

        assert outlier_mask.shape == (4, 2)
        assert outlier_mask[3, 1]

    def test_custom_multiplier(self) -> None:
        X = np.array([[1], [2], [3], [4], [10]])

        outliers_strict = detect_outliers_iqr(X, multiplier=1.0)
        outliers_lenient = detect_outliers_iqr(X, multiplier=3.0)

        assert outliers_strict.sum() >= outliers_lenient.sum()

    def test_all_same_values(self) -> None:
        X = np.array([[5], [5], [5], [5]])

        outlier_mask = detect_outliers_iqr(X, multiplier=1.5)

        assert not outlier_mask.any()
