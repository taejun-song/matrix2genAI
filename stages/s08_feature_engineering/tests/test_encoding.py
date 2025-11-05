from __future__ import annotations

import numpy as np

from stages.s08_feature_engineering.starter.encoding import (
    label_decode,
    label_encode,
    one_hot_decode,
    one_hot_encode,
)


class TestLabelEncode:
    def test_string_labels(self):
        y = np.array(["cat", "dog", "cat", "bird", "dog"])

        y_encoded, classes = label_encode(y)

        assert y_encoded.dtype in [np.int32, np.int64, int]
        assert len(classes) == 3
        assert set(classes) == {"bird", "cat", "dog"}
        assert y_encoded[0] == y_encoded[2]
        assert y_encoded[1] == y_encoded[4]

    def test_numeric_labels(self):
        y = np.array([10, 20, 10, 30])

        y_encoded, classes = label_encode(y)

        assert len(classes) == 3
        assert y_encoded[0] == y_encoded[2]
        assert all(0 <= val < 3 for val in y_encoded)

    def test_single_class(self):
        y = np.array(["a", "a", "a"])

        y_encoded, classes = label_encode(y)

        assert len(classes) == 1
        assert all(y_encoded == 0)

    def test_classes_sorted(self):
        y = np.array(["zebra", "apple", "mango"])

        y_encoded, classes = label_encode(y)

        assert classes[0] == "apple"
        assert classes[-1] == "zebra"


class TestLabelDecode:
    def test_decode_integers(self):
        y_encoded = np.array([0, 1, 0, 2])
        classes = np.array(["bird", "cat", "dog"])

        y = label_decode(y_encoded, classes)

        expected = np.array(["bird", "cat", "bird", "dog"])
        np.testing.assert_array_equal(y, expected)

    def test_roundtrip(self):
        y_original = np.array(["cat", "dog", "cat", "bird"])

        y_encoded, classes = label_encode(y_original)
        y_decoded = label_decode(y_encoded, classes)

        np.testing.assert_array_equal(y_original, y_decoded)

    def test_numeric_classes(self):
        y_encoded = np.array([0, 1, 2])
        classes = np.array([100, 200, 300])

        y = label_decode(y_encoded, classes)

        np.testing.assert_array_equal(y, [100, 200, 300])


class TestOneHotEncode:
    def test_basic_encoding(self):
        y = np.array([0, 1, 0, 2])

        y_onehot = one_hot_encode(y, n_classes=3)

        expected = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]])
        np.testing.assert_array_equal(y_onehot, expected)

    def test_auto_detect_classes(self):
        y = np.array([0, 1, 2])

        y_onehot = one_hot_encode(y)

        assert y_onehot.shape == (3, 3)
        assert y_onehot.sum() == 3

    def test_binary_classification(self):
        y = np.array([0, 1, 1, 0])

        y_onehot = one_hot_encode(y, n_classes=2)

        assert y_onehot.shape == (4, 2)
        expected = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])
        np.testing.assert_array_equal(y_onehot, expected)

    def test_single_class(self):
        y = np.array([0, 0, 0])

        y_onehot = one_hot_encode(y, n_classes=1)

        expected = np.array([[1], [1], [1]])
        np.testing.assert_array_equal(y_onehot, expected)

    def test_row_sum_equals_one(self):
        y = np.array([0, 1, 2, 1, 0])

        y_onehot = one_hot_encode(y, n_classes=3)

        assert np.all(y_onehot.sum(axis=1) == 1)


class TestOneHotDecode:
    def test_basic_decoding(self):
        y_onehot = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        y = one_hot_decode(y_onehot)

        expected = np.array([0, 1, 2])
        np.testing.assert_array_equal(y, expected)

    def test_roundtrip(self):
        y_original = np.array([0, 1, 0, 2, 1])

        y_onehot = one_hot_encode(y_original, n_classes=3)
        y_decoded = one_hot_decode(y_onehot)

        np.testing.assert_array_equal(y_original, y_decoded)

    def test_binary_classification(self):
        y_onehot = np.array([[1, 0], [0, 1], [1, 0]])

        y = one_hot_decode(y_onehot)

        expected = np.array([0, 1, 0])
        np.testing.assert_array_equal(y, expected)

    def test_many_classes(self):
        n_classes = 10
        y_original = np.arange(n_classes)

        y_onehot = one_hot_encode(y_original, n_classes)
        y_decoded = one_hot_decode(y_onehot)

        np.testing.assert_array_equal(y_original, y_decoded)
