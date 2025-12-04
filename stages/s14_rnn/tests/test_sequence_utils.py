from __future__ import annotations

import numpy as np

from stages.s14_rnn.starter.sequence_utils import (
    clip_gradients,
    create_sequences,
    sequence_loss,
)


class TestClipGradients:
    def test_no_clipping_needed(self) -> None:
        grads = [np.array([1.0, 2.0])]
        clipped = clip_gradients(grads, max_norm=10.0)
        np.testing.assert_array_equal(clipped[0], grads[0])

    def test_clipping_applied(self) -> None:
        grads = [np.array([3.0, 4.0])]
        clipped = clip_gradients(grads, max_norm=2.5)
        total_norm = np.linalg.norm(clipped[0])
        np.testing.assert_allclose(total_norm, 2.5, rtol=1e-5)

    def test_multiple_gradients(self) -> None:
        grads = [np.array([3.0, 0.0]), np.array([0.0, 4.0])]
        clipped = clip_gradients(grads, max_norm=2.5)
        total_norm = np.sqrt(sum(np.sum(g ** 2) for g in clipped))
        np.testing.assert_allclose(total_norm, 2.5, rtol=1e-5)

    def test_preserves_direction(self) -> None:
        grads = [np.array([3.0, 4.0])]
        clipped = clip_gradients(grads, max_norm=2.5)
        original_direction = grads[0] / np.linalg.norm(grads[0])
        clipped_direction = clipped[0] / np.linalg.norm(clipped[0])
        np.testing.assert_allclose(original_direction, clipped_direction)

    def test_zero_gradients(self) -> None:
        grads = [np.zeros(5)]
        clipped = clip_gradients(grads, max_norm=1.0)
        np.testing.assert_array_equal(clipped[0], np.zeros(5))


class TestSequenceLoss:
    def test_perfect_predictions(self) -> None:
        predictions = np.array([[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]]])
        targets = np.array([[0, 1]])
        loss = sequence_loss(predictions, targets)
        assert loss < 0.01

    def test_uniform_predictions(self) -> None:
        predictions = np.zeros((1, 2, 3))
        targets = np.array([[0, 1]])
        loss = sequence_loss(predictions, targets)
        expected = -np.log(1/3)
        np.testing.assert_allclose(loss, expected, rtol=0.1)

    def test_with_mask(self) -> None:
        predictions = np.zeros((1, 3, 2))
        predictions[0, :, 0] = 10.0
        targets = np.array([[0, 0, 1]])
        mask = np.array([[1, 1, 0]])
        loss = sequence_loss(predictions, targets, mask)
        assert loss < 0.1

    def test_batch_processing(self) -> None:
        predictions = np.random.randn(4, 10, 100)
        targets = np.random.randint(0, 100, (4, 10))
        loss = sequence_loss(predictions, targets)
        assert loss > 0


class TestCreateSequences:
    def test_output_shapes(self) -> None:
        data = np.random.randn(100, 5)
        X, y = create_sequences(data, seq_length=10)
        assert X.shape == (90, 10, 5)
        assert y.shape == (90, 5)

    def test_with_stride(self) -> None:
        data = np.random.randn(100, 5)
        X, y = create_sequences(data, seq_length=10, stride=5)
        assert X.shape[0] == 18

    def test_values_correct(self) -> None:
        data = np.arange(20).reshape(10, 2)
        X, y = create_sequences(data, seq_length=3, stride=1)
        np.testing.assert_array_equal(X[0], data[:3])
        np.testing.assert_array_equal(y[0], data[3])
        np.testing.assert_array_equal(X[1], data[1:4])
        np.testing.assert_array_equal(y[1], data[4])

    def test_single_feature(self) -> None:
        data = np.arange(100).reshape(100, 1)
        X, y = create_sequences(data, seq_length=5)
        assert X.shape == (95, 5, 1)
        assert y.shape == (95, 1)
