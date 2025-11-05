from __future__ import annotations

import numpy as np
import pytest

from stages.s08_feature_engineering.starter.imputation import (
    find_missing_mask,
    impute_with_constant,
    simple_imputer_strategy,
)


class TestSimpleImputerStrategy:
    def test_mean_imputation(self):
        X = np.array([[1.0, 2.0], [np.nan, 4.0], [3.0, np.nan]])

        X_imputed = simple_imputer_strategy(X, strategy="mean")

        assert not np.isnan(X_imputed).any()
        np.testing.assert_allclose(X_imputed[1, 0], 2.0)
        np.testing.assert_allclose(X_imputed[2, 1], 3.0)

    def test_median_imputation(self):
        X = np.array([[1.0, 100.0], [np.nan, 2.0], [3.0, np.nan], [5.0, 4.0]])

        X_imputed = simple_imputer_strategy(X, strategy="median")

        assert not np.isnan(X_imputed).any()
        np.testing.assert_allclose(X_imputed[1, 0], 3.0)

    def test_most_frequent_imputation(self):
        X = np.array([[1.0, 1.0], [np.nan, 2.0], [1.0, np.nan], [2.0, 1.0]])

        X_imputed = simple_imputer_strategy(X, strategy="most_frequent")

        assert not np.isnan(X_imputed).any()
        assert X_imputed[1, 0] in [1.0, 2.0]

    def test_no_missing_values(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])

        X_imputed = simple_imputer_strategy(X, strategy="mean")

        np.testing.assert_array_equal(X, X_imputed)

    def test_all_nan_column(self):
        X = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]])

        X_imputed = simple_imputer_strategy(X, strategy="mean")

        assert np.isnan(X_imputed[:, 1]).all()

    def test_single_value_column(self):
        X = np.array([[5.0], [np.nan], [np.nan]])

        X_imputed = simple_imputer_strategy(X, strategy="mean")

        np.testing.assert_allclose(X_imputed, [[5.0], [5.0], [5.0]])


class TestFindMissingMask:
    def test_basic_mask(self):
        X = np.array([[1.0, np.nan], [2.0, 3.0], [np.nan, 4.0]])

        mask = find_missing_mask(X)

        expected = np.array([[False, True], [False, False], [True, False]])
        np.testing.assert_array_equal(mask, expected)

    def test_no_missing(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])

        mask = find_missing_mask(X)

        assert not mask.any()

    def test_all_missing(self):
        X = np.full((3, 2), np.nan)

        mask = find_missing_mask(X)

        assert mask.all()

    def test_shape_matches_input(self):
        X = np.array([[1.0, np.nan, 3.0]])

        mask = find_missing_mask(X)

        assert mask.shape == X.shape


class TestImputeWithConstant:
    def test_zero_fill(self):
        X = np.array([[1.0, np.nan], [np.nan, 3.0]])

        X_imputed = impute_with_constant(X, fill_value=0.0)

        expected = np.array([[1.0, 0.0], [0.0, 3.0]])
        np.testing.assert_array_equal(X_imputed, expected)

    def test_custom_fill_value(self):
        X = np.array([[np.nan, 2.0], [3.0, np.nan]])

        X_imputed = impute_with_constant(X, fill_value=-999.0)

        expected = np.array([[-999.0, 2.0], [3.0, -999.0]])
        np.testing.assert_array_equal(X_imputed, expected)

    def test_no_missing(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])

        X_imputed = impute_with_constant(X, fill_value=0.0)

        np.testing.assert_array_equal(X, X_imputed)

    def test_all_missing(self):
        X = np.full((2, 2), np.nan)

        X_imputed = impute_with_constant(X, fill_value=42.0)

        assert np.all(X_imputed == 42.0)
