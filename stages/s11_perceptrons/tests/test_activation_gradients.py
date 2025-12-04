from __future__ import annotations

import numpy as np

from stages.s11_perceptrons.starter.activation_gradients import (
    elu_derivative,
    gelu_derivative,
    leaky_relu_derivative,
    mish_derivative,
    relu_derivative,
    softplus_derivative,
    swish_derivative,
    tanh_derivative,
)
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


def numerical_gradient(f, z, eps=1e-5):
    return (f(z + eps) - f(z - eps)) / (2 * eps)


class TestReLUDerivative:
    def test_positive(self) -> None:
        z = np.array([1.0, 2.0, 3.0])
        result = relu_derivative(z)
        np.testing.assert_array_equal(result, [1.0, 1.0, 1.0])

    def test_negative(self) -> None:
        z = np.array([-1.0, -2.0, -3.0])
        result = relu_derivative(z)
        np.testing.assert_array_equal(result, [0.0, 0.0, 0.0])

    def test_zero(self) -> None:
        z = np.array([0.0])
        result = relu_derivative(z)
        assert result[0] == 0.0

    def test_numerical(self) -> None:
        z = np.array([0.5, 1.0, 2.0])
        analytical = relu_derivative(z)
        numerical = numerical_gradient(relu, z)
        np.testing.assert_allclose(analytical, numerical, rtol=1e-3)


class TestLeakyReLUDerivative:
    def test_positive(self) -> None:
        z = np.array([1.0, 2.0])
        result = leaky_relu_derivative(z, alpha=0.1)
        np.testing.assert_array_equal(result, [1.0, 1.0])

    def test_negative(self) -> None:
        z = np.array([-1.0, -2.0])
        result = leaky_relu_derivative(z, alpha=0.1)
        np.testing.assert_array_equal(result, [0.1, 0.1])

    def test_numerical(self) -> None:
        z = np.array([-1.0, 0.5, 1.0])
        analytical = leaky_relu_derivative(z, alpha=0.01)
        numerical = numerical_gradient(lambda x: leaky_relu(x, 0.01), z)
        np.testing.assert_allclose(analytical, numerical, rtol=0.1)


class TestELUDerivative:
    def test_positive(self) -> None:
        z = np.array([1.0, 2.0])
        result = elu_derivative(z)
        np.testing.assert_array_equal(result, [1.0, 1.0])

    def test_negative(self) -> None:
        z = np.array([-1.0])
        result = elu_derivative(z, alpha=1.0)
        expected = elu(z, alpha=1.0) + 1.0
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_numerical(self) -> None:
        z = np.array([-0.5, 0.5, 1.0])
        analytical = elu_derivative(z)
        numerical = numerical_gradient(elu, z)
        np.testing.assert_allclose(analytical, numerical, rtol=1e-3)


class TestGELUDerivative:
    def test_zero(self) -> None:
        z = np.array([0.0])
        result = gelu_derivative(z)
        np.testing.assert_allclose(result, [0.5], rtol=0.01)

    def test_numerical(self) -> None:
        z = np.array([-1.0, 0.0, 1.0])
        analytical = gelu_derivative(z)
        numerical = numerical_gradient(gelu, z)
        np.testing.assert_allclose(analytical, numerical, rtol=0.01)


class TestSwishDerivative:
    def test_zero(self) -> None:
        z = np.array([0.0])
        result = swish_derivative(z)
        np.testing.assert_allclose(result, [0.5], rtol=1e-5)

    def test_numerical(self) -> None:
        z = np.array([-1.0, 0.0, 1.0])
        analytical = swish_derivative(z)
        numerical = numerical_gradient(swish, z)
        np.testing.assert_allclose(analytical, numerical, rtol=1e-3)


class TestTanhDerivative:
    def test_zero(self) -> None:
        z = np.array([0.0])
        result = tanh_derivative(z)
        np.testing.assert_allclose(result, [1.0])

    def test_numerical(self) -> None:
        z = np.array([-1.0, 0.0, 1.0])
        analytical = tanh_derivative(z)
        numerical = numerical_gradient(tanh_activation, z)
        np.testing.assert_allclose(analytical, numerical, rtol=1e-3)

    def test_range(self) -> None:
        z = np.array([-5, -1, 0, 1, 5])
        result = tanh_derivative(z)
        assert np.all(result >= 0)
        assert np.all(result <= 1)


class TestSoftplusDerivative:
    def test_zero(self) -> None:
        z = np.array([0.0])
        result = softplus_derivative(z)
        np.testing.assert_allclose(result, [0.5])

    def test_is_sigmoid(self) -> None:
        z = np.array([-2, -1, 0, 1, 2])
        result = softplus_derivative(z)
        sigmoid = 1 / (1 + np.exp(-z))
        np.testing.assert_allclose(result, sigmoid, rtol=1e-5)

    def test_numerical(self) -> None:
        z = np.array([-1.0, 0.0, 1.0])
        analytical = softplus_derivative(z)
        numerical = numerical_gradient(softplus, z)
        np.testing.assert_allclose(analytical, numerical, rtol=1e-3)


class TestMishDerivative:
    def test_numerical(self) -> None:
        z = np.array([-1.0, 0.0, 1.0])
        analytical = mish_derivative(z)
        numerical = numerical_gradient(mish, z)
        np.testing.assert_allclose(analytical, numerical, rtol=0.01)

    def test_zero(self) -> None:
        z = np.array([0.0])
        result = mish_derivative(z)
        assert 0.5 < result[0] < 0.7
