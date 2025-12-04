from __future__ import annotations

import numpy as np

from stages.s15_attention.starter.positional import (
    add_positional_encoding,
    create_causal_mask,
    create_padding_mask,
    learned_positional_encoding,
    sinusoidal_encoding,
)


class TestSinusoidalEncoding:
    def test_output_shape(self) -> None:
        pe = sinusoidal_encoding(100, 512)
        assert pe.shape == (100, 512)

    def test_bounded_values(self) -> None:
        pe = sinusoidal_encoding(1000, 256)
        assert np.all(pe >= -1) and np.all(pe <= 1)

    def test_alternating_sin_cos(self) -> None:
        pe = sinusoidal_encoding(10, 4)
        pos_1_even = pe[1, 0]
        pos_1_odd = pe[1, 1]
        expected_sin = np.sin(1.0 / (10000 ** 0))
        expected_cos = np.cos(1.0 / (10000 ** 0))
        np.testing.assert_allclose(pos_1_even, expected_sin, rtol=1e-5)
        np.testing.assert_allclose(pos_1_odd, expected_cos, rtol=1e-5)

    def test_position_zero(self) -> None:
        pe = sinusoidal_encoding(10, 4)
        np.testing.assert_allclose(pe[0, 0], 0.0, atol=1e-10)
        np.testing.assert_allclose(pe[0, 1], 1.0, atol=1e-10)

    def test_unique_positions(self) -> None:
        pe = sinusoidal_encoding(100, 64)
        for i in range(99):
            assert not np.allclose(pe[i], pe[i + 1])


class TestAddPositionalEncoding:
    def test_output_shape(self) -> None:
        x = np.random.randn(2, 10, 64)
        pe = sinusoidal_encoding(100, 64)
        output = add_positional_encoding(x, pe)
        assert output.shape == x.shape

    def test_adds_correctly(self) -> None:
        x = np.ones((1, 5, 4))
        pe = np.arange(20).reshape(20, 1).repeat(4, axis=1)[:20]
        pe = pe.astype(float)
        output = add_positional_encoding(x, pe)
        expected = x + pe[:5]
        np.testing.assert_array_equal(output, expected)


class TestLearnedPositionalEncoding:
    def test_output_shape(self) -> None:
        pe = learned_positional_encoding(100, 512)
        assert pe.shape == (100, 512)

    def test_random_initialization(self) -> None:
        pe1 = learned_positional_encoding(10, 32)
        pe2 = learned_positional_encoding(10, 32)
        assert not np.allclose(pe1, pe2)


class TestCreateCausalMask:
    def test_shape(self) -> None:
        mask = create_causal_mask(5)
        assert mask.shape == (5, 5)

    def test_lower_triangular(self) -> None:
        mask = create_causal_mask(4)
        expected = np.array([
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True]
        ])
        np.testing.assert_array_equal(mask, expected)

    def test_first_row(self) -> None:
        mask = create_causal_mask(10)
        assert mask[0, 0] == True
        assert np.all(mask[0, 1:] == False)

    def test_last_row(self) -> None:
        mask = create_causal_mask(10)
        assert np.all(mask[-1] == True)


class TestCreatePaddingMask:
    def test_shape(self) -> None:
        lengths = np.array([3, 5])
        mask = create_padding_mask(lengths, max_length=10)
        assert mask.shape == (2, 10)

    def test_values(self) -> None:
        lengths = np.array([3, 2])
        mask = create_padding_mask(lengths, max_length=4)
        expected = np.array([
            [True, True, True, False],
            [True, True, False, False]
        ])
        np.testing.assert_array_equal(mask, expected)

    def test_full_length(self) -> None:
        lengths = np.array([5])
        mask = create_padding_mask(lengths, max_length=5)
        assert np.all(mask[0] == True)

    def test_zero_length(self) -> None:
        lengths = np.array([0])
        mask = create_padding_mask(lengths, max_length=5)
        assert np.all(mask[0] == False)
