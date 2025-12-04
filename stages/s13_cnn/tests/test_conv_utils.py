from __future__ import annotations

import numpy as np

from stages.s13_cnn.starter.conv_utils import col2im, get_output_shape, im2col, pad2d


class TestPad2d:
    def test_no_padding(self) -> None:
        x = np.ones((1, 3, 3, 1))
        result = pad2d(x, 0)
        assert result.shape == (1, 3, 3, 1)

    def test_uniform_padding(self) -> None:
        x = np.ones((1, 3, 3, 1))
        result = pad2d(x, 1)
        assert result.shape == (1, 5, 5, 1)
        assert result[0, 0, 0, 0] == 0
        assert result[0, 1, 1, 0] == 1

    def test_asymmetric_padding(self) -> None:
        x = np.ones((2, 4, 5, 3))
        result = pad2d(x, (1, 2))
        assert result.shape == (2, 6, 9, 3)

    def test_custom_value(self) -> None:
        x = np.zeros((1, 2, 2, 1))
        result = pad2d(x, 1, value=-1)
        assert result[0, 0, 0, 0] == -1
        assert result[0, 1, 1, 0] == 0


class TestGetOutputShape:
    def test_no_padding_stride1(self) -> None:
        assert get_output_shape((5, 5), 3) == (3, 3)

    def test_with_padding(self) -> None:
        assert get_output_shape((5, 5), 3, padding=1) == (5, 5)

    def test_stride2(self) -> None:
        assert get_output_shape((8, 8), 2, stride=2) == (4, 4)

    def test_asymmetric(self) -> None:
        assert get_output_shape((10, 8), (3, 3), stride=(2, 1), padding=(1, 0)) == (5, 6)


class TestIm2col:
    def test_output_shape(self) -> None:
        x = np.random.randn(2, 8, 8, 3)
        col = im2col(x, kernel_size=3, stride=1, padding=0)
        assert col.shape == (2 * 6 * 6, 3 * 3 * 3)

    def test_with_padding(self) -> None:
        x = np.random.randn(1, 4, 4, 1)
        col = im2col(x, kernel_size=3, stride=1, padding=1)
        assert col.shape == (1 * 4 * 4, 3 * 3 * 1)

    def test_stride2(self) -> None:
        x = np.random.randn(1, 8, 8, 1)
        col = im2col(x, kernel_size=2, stride=2)
        assert col.shape == (1 * 4 * 4, 2 * 2 * 1)

    def test_values_match_naive(self) -> None:
        x = np.arange(16).reshape(1, 4, 4, 1).astype(float)
        col = im2col(x, kernel_size=2, stride=1)
        expected_first_patch = [0, 1, 4, 5]
        np.testing.assert_array_equal(col[0], expected_first_patch)


class TestCol2im:
    def test_reconstruction_no_overlap(self) -> None:
        x = np.random.randn(1, 4, 4, 1)
        col = im2col(x, kernel_size=2, stride=2)
        reconstructed = col2im(col, x.shape, kernel_size=2, stride=2)
        np.testing.assert_allclose(x, reconstructed)

    def test_shape_preservation(self) -> None:
        x = np.random.randn(2, 8, 8, 3)
        col = im2col(x, kernel_size=3, stride=1, padding=1)
        reconstructed = col2im(col, x.shape, kernel_size=3, stride=1, padding=1)
        assert reconstructed.shape == x.shape


class TestConvEquivalence:
    def test_im2col_matches_naive_conv(self) -> None:
        np.random.seed(42)
        x = np.random.randn(1, 5, 5, 1)
        W = np.random.randn(3, 3, 1, 1)
        col = im2col(x, kernel_size=3, stride=1, padding=0)
        W_col = W.reshape(-1, 1)
        out_im2col = (col @ W_col).reshape(1, 3, 3, 1)
        out_naive = np.zeros((1, 3, 3, 1))
        for i in range(3):
            for j in range(3):
                out_naive[0, i, j, 0] = np.sum(x[0, i:i+3, j:j+3, 0] * W[:, :, 0, 0])
        np.testing.assert_allclose(out_im2col, out_naive, rtol=1e-5)
