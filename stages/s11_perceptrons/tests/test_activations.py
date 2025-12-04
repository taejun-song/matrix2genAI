from __future__ import annotations

import numpy as np

from stages.s11_perceptrons.starter.activations import (
    elu,
    gelu,
    leaky_relu,
    mish,
    relu,
    softplus,
    swish,
    tanh_activation,
)


class TestReLU:
    def test_positive_values(self) -> None:
        z = np.array([1.0, 2.0, 3.0])
        result = relu(z)
        np.testing.assert_array_equal(result, z)

    def test_negative_values(self) -> None:
        z = np.array([-1.0, -2.0, -3.0])
        result = relu(z)
        np.testing.assert_array_equal(result, np.zeros(3))

    def test_zero(self) -> None:
        z = np.array([0.0])
        result = relu(z)
        np.testing.assert_array_equal(result, [0.0])

    def test_mixed(self) -> None:
        z = np.array([-2, -1, 0, 1, 2])
        result = relu(z)
        np.testing.assert_array_equal(result, [0, 0, 0, 1, 2])


class TestLeakyReLU:
    def test_positive_values(self) -> None:
        z = np.array([1.0, 2.0])
        result = leaky_relu(z, alpha=0.1)
        np.testing.assert_array_equal(result, z)

    def test_negative_values(self) -> None:
        z = np.array([-1.0, -2.0])
        result = leaky_relu(z, alpha=0.1)
        np.testing.assert_allclose(result, [-0.1, -0.2])

    def test_default_alpha(self) -> None:
        z = np.array([-100.0])
        result = leaky_relu(z)
        np.testing.assert_allclose(result, [-1.0])

    def test_zero(self) -> None:
        z = np.array([0.0])
        result = leaky_relu(z, alpha=0.1)
        np.testing.assert_allclose(result, [0.0])


class TestELU:
    def test_positive_values(self) -> None:
        z = np.array([1.0, 2.0])
        result = elu(z, alpha=1.0)
        np.testing.assert_array_equal(result, z)

    def test_negative_values(self) -> None:
        z = np.array([-1.0])
        result = elu(z, alpha=1.0)
        expected = np.exp(-1.0) - 1
        np.testing.assert_allclose(result, [expected], rtol=1e-5)

    def test_zero(self) -> None:
        z = np.array([0.0])
        result = elu(z)
        np.testing.assert_allclose(result, [0.0])

    def test_different_alpha(self) -> None:
        z = np.array([-1.0])
        result = elu(z, alpha=2.0)
        expected = 2.0 * (np.exp(-1.0) - 1)
        np.testing.assert_allclose(result, [expected], rtol=1e-5)


class TestGELU:
    def test_zero(self) -> None:
        z = np.array([0.0])
        result = gelu(z)
        np.testing.assert_allclose(result, [0.0], atol=1e-5)

    def test_positive(self) -> None:
        z = np.array([1.0])
        result = gelu(z)
        assert result[0] > 0.8
        assert result[0] < 0.9

    def test_negative(self) -> None:
        z = np.array([-1.0])
        result = gelu(z)
        assert result[0] < 0
        assert result[0] > -0.2

    def test_large_positive(self) -> None:
        z = np.array([5.0])
        result = gelu(z)
        np.testing.assert_allclose(result, z, rtol=0.01)


class TestSwish:
    def test_zero(self) -> None:
        z = np.array([0.0])
        result = swish(z)
        np.testing.assert_allclose(result, [0.0])

    def test_positive(self) -> None:
        z = np.array([1.0])
        result = swish(z)
        expected = 1.0 / (1 + np.exp(-1.0))
        np.testing.assert_allclose(result, [expected], rtol=1e-5)

    def test_symmetry(self) -> None:
        z = np.array([1.0])
        neg_z = np.array([-1.0])
        assert swish(z)[0] > 0
        assert swish(neg_z)[0] < 0

    def test_large_values(self) -> None:
        z = np.array([10.0])
        result = swish(z)
        np.testing.assert_allclose(result, z, rtol=0.01)


class TestTanh:
    def test_zero(self) -> None:
        z = np.array([0.0])
        result = tanh_activation(z)
        np.testing.assert_allclose(result, [0.0])

    def test_positive(self) -> None:
        z = np.array([1.0])
        result = tanh_activation(z)
        np.testing.assert_allclose(result, [np.tanh(1.0)])

    def test_range(self) -> None:
        z = np.array([-10, -1, 0, 1, 10])
        result = tanh_activation(z)
        assert np.all(result >= -1)
        assert np.all(result <= 1)

    def test_symmetry(self) -> None:
        z = np.array([1.0, 2.0, 3.0])
        pos = tanh_activation(z)
        neg = tanh_activation(-z)
        np.testing.assert_allclose(pos, -neg)


class TestSoftplus:
    def test_zero(self) -> None:
        z = np.array([0.0])
        result = softplus(z)
        np.testing.assert_allclose(result, [np.log(2)], rtol=1e-5)

    def test_large_positive(self) -> None:
        z = np.array([100.0])
        result = softplus(z)
        np.testing.assert_allclose(result, z, rtol=0.01)

    def test_always_positive(self) -> None:
        z = np.array([-10, -5, 0, 5, 10])
        result = softplus(z)
        assert np.all(result > 0)

    def test_numerical_stability(self) -> None:
        z = np.array([1000.0])
        result = softplus(z)
        assert not np.isinf(result).any()
        assert not np.isnan(result).any()


class TestMish:
    def test_zero(self) -> None:
        z = np.array([0.0])
        result = mish(z)
        np.testing.assert_allclose(result, [0.0], atol=1e-5)

    def test_positive(self) -> None:
        z = np.array([1.0])
        result = mish(z)
        assert result[0] > 0.8

    def test_negative(self) -> None:
        z = np.array([-1.0])
        result = mish(z)
        assert result[0] < 0

    def test_smooth(self) -> None:
        z = np.linspace(-2, 2, 100)
        result = mish(z)
        diff = np.diff(result)
        assert not np.any(np.isnan(result))
